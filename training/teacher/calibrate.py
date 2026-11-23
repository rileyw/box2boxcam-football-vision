"""Gate a teacher against a student and choose its pseudo-label thresholds, on VAL only.

The teacher proposes training labels for new footage. It is worth using only if it finds
more balls than the student on val, especially small ones. Its auto-accept threshold
``high`` is the lowest confidence at which its boxes on val are at least
``--precision`` precise. Boxes between ``low`` and ``high`` go to human review. Never run
this on the test split: test stays human-labelled and is never used to tune labelling.

    python teacher/calibrate.py --teacher runs/teacher_x960_s42/weights/best.pt \
        --student eval/models/base_s42.pt --out teacher/thresholds.json
    # ensemble teacher: several models, horizontal-flip TTA, weighted box fusion
    python teacher/calibrate.py --teacher eval/models/base_s42.pt eval/models/base_s7.pt --flip \
        --student eval/models/base_s42.pt --out teacher/thresholds.json
"""

import argparse
import json
from pathlib import Path

import sys

import numpy as np

from box2boxcam.evaluation.detect import run_detector
from box2boxcam.evaluation.ground_truth import load_ground_truth
from box2boxcam.evaluation.matching import (
    count_small_tp,
    count_tp_fp,
    match_all,
    max_tp_under_fp_budget,
    small_balls,
)

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "teacher"))
from ensemble import predict as ensemble_predict  # noqa: E402
BUDGETS = (25, 60, 100, 200)


def precision_curve(matches, thresholds):
    rows = []
    for t in thresholds:
        tp, fp, _ = count_tp_fp(matches, t)
        rows.append({"conf": float(t), "tp": tp, "fp": fp,
                     "precision": tp / (tp + fp) if tp + fp else 1.0})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, nargs="+", required=True,
                   help="One model, or several for an ensemble (weighted box fusion).")
    p.add_argument("--flip", action="store_true", help="Add horizontal-flip TTA to the teacher.")
    p.add_argument("--fusion-iou", type=float, default=0.55)
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--images", type=Path, default=HERE / "data/public/images/val")
    p.add_argument("--labels", type=Path, default=HERE / "data/public/labels/val")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--precision", type=float, default=0.95, help="Target for auto-accept.")
    p.add_argument("--low", type=float, default=0.15, help="Review floor.")
    p.add_argument("--out", type=Path, default=HERE / "teacher/thresholds.json")
    args = p.parse_args()
    if "test" in args.images.parts:
        raise SystemExit("Calibrate on val, never on the test split.")

    frames = load_ground_truth(args.images, args.labels)
    gts = [f.boxes for f in frames]
    small = small_balls(gts)
    paths = [f.path for f in frames]
    report = {"val_frames": len(frames), "val_balls": int(sum(len(g) for g in gts)),
              "val_small_balls": len(small)}
    matches = {}
    detections = {
        "teacher": ensemble_predict(args.teacher, paths, args.imgsz, 0.05, flip=args.flip,
                                    iou=args.fusion_iou),
        "student": run_detector(args.student, paths, args.imgsz, 0.05, 0),
    }
    for name, weights in (("teacher", args.teacher), ("student", [args.student])):
        matches[name] = match_all(gts, detections[name])
        report[name] = {
            "weights": [str(w) for w in weights],
            "max_tp_under_fp_budget": {b: max_tp_under_fp_budget(matches[name], b) for b in BUDGETS},
            "small_tp": {t: count_small_tp(matches[name], small, t) for t in (0.15, 0.25)},
            "tp_fp": {t: count_tp_fp(matches[name], t)[:2] for t in (0.15, 0.25, 0.5)},
        }
    curve = precision_curve(matches["teacher"], np.round(np.arange(0.10, 0.96, 0.05), 2))
    high = next((r["conf"] for r in curve if r["precision"] >= args.precision and r["tp"] > 0), None)
    teacher_better = all(
        report["teacher"]["max_tp_under_fp_budget"][b] >= report["student"]["max_tp_under_fp_budget"][b]
        for b in BUDGETS
    ) and report["teacher"]["small_tp"][0.25] > report["student"]["small_tp"][0.25]
    report.update({
        "teacher_config": {"weights": [str(w) for w in args.teacher], "flip": args.flip,
                           "fusion_iou": args.fusion_iou, "imgsz": args.imgsz},
        "teacher_precision_curve": curve,
        "thresholds": {"low": args.low, "high": high, "target_precision": args.precision},
        "gate_teacher_beats_student_on_val": teacher_better,
    })
    args.out.write_text(json.dumps(report, indent=1, default=str))

    for name in ("teacher", "student"):
        r = report[name]
        print(f"{name:8s} max TP at FP<=25/60/100/200: "
              + "/".join(str(r["max_tp_under_fp_budget"][b]) for b in BUDGETS)
              + f" | small TP @0.15/0.25: {r['small_tp'][0.15]}/{r['small_tp'][0.25]}"
              + f" of {len(small)}")
    print(f"auto-accept threshold (val precision >= {args.precision}): {high}; review floor: {args.low}")
    print("gate, teacher beats student on val:", "PASS" if teacher_better else "FAIL")
    print("written", args.out)


if __name__ == "__main__":
    main()
