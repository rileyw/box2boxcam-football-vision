"""Evaluate a ball detector on a held-out set of frames, optionally against a second model.

Reports, at IoU >= 0.5: true/false positives at several confidence thresholds, true
positives on small balls (longest side <= 16 px), the maximum number of true positives
reachable under fixed false-positive budgets, and, when --compare is given, a paired
bootstrap over frames of the difference model A (--weights) minus model B (--compare).

Ground truth (pixel boxes) is read from --labels, either a directory of YOLO .txt files or a
JSON file; see box2boxcam.evaluation.ground_truth for the exact layouts.

Example::

    python scripts/evaluate_detector.py --weights new.pt --compare reference.pt \\
        --frames-dir heldout/images --labels heldout/labels --out-json report.json

Core ML parity (macOS only): pass the export as one of the two models and add
--agreement for box-level agreement between them, independent of ground truth::

    python scripts/evaluate_detector.py --weights model.mlpackage --compare model.pt \\
        --frames-dir heldout/images --labels heldout/labels --agreement

Ultralytics runs a Core ML model exported with NMS on its own confidence threshold
(0.1 by default), so its boxes below that value are missing whatever --conf says;
the script warns when this cuts the comparison.

Frames of a held-out match must never be used for training or model selection.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from box2boxcam.evaluation.agreement import agreement, agreement_table
from box2boxcam.evaluation.detect import (
    coreml_confidence_floor,
    is_coreml,
    load_detections,
    run_detector,
    save_detections,
)
from box2boxcam.evaluation.ground_truth import load_ground_truth
from box2boxcam.evaluation.matching import IOU_THRESHOLD, SMALL_MAX_SIDE, match_all
from box2boxcam.evaluation.report import (
    DEFAULT_BUDGETS,
    DEFAULT_THRESHOLDS,
    bootstrap_table,
    evaluate,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--weights", type=Path, required=True, help="Model A (.pt or .mlpackage)."
    )
    p.add_argument(
        "--compare",
        type=Path,
        help="Model B (.pt or .mlpackage): adds the A - B bootstrap.",
    )
    p.add_argument(
        "--frames-dir", type=Path, required=True, help="Directory of evaluation images."
    )
    p.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="YOLO label directory or JSON ground-truth file.",
    )
    p.add_argument("--imgsz", type=int, default=960, help="Inference size.")
    p.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="Lowest confidence kept at inference; thresholds are applied afterwards.",
    )
    p.add_argument(
        "--class-filter",
        action="append",
        default=[],
        metavar="STEM=ID[,ID]",
        help="Keep only these class ids for the model whose file stem is STEM, e.g. "
        "yolo26s=32 for the COCO sports ball of a stock model. Repeatable.",
    )
    p.add_argument("--device", help="Torch device (default: chosen by Ultralytics).")
    p.add_argument(
        "--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS)
    )
    p.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    p.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="Matching IoU.")
    p.add_argument(
        "--small-px",
        type=float,
        default=SMALL_MAX_SIDE,
        help="A ball is small when its longest side is at most this many pixels.",
    )
    p.add_argument("--resamples", type=int, default=2000, help="Bootstrap resamples.")
    p.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")
    p.add_argument(
        "--cache-dir",
        type=Path,
        help="Cache raw detections here (one JSON per weights file) to skip inference on re-runs.",
    )
    p.add_argument(
        "--agreement",
        action="store_true",
        help="With --compare, also report box-level agreement between A and B at "
        "each threshold (e.g. a model and its Core ML export).",
    )
    p.add_argument("--out-json", type=Path, help="Write the full report as JSON.")
    return p


def parse_class_filters(specs):
    """``["stem=32,33"]`` -> ``{"stem": [32, 33]}``."""
    out = {}
    for spec in specs:
        stem, _, ids = spec.partition("=")
        if not stem or not ids:
            raise SystemExit(f"--class-filter expects STEM=ID[,ID], got {spec!r}")
        out[stem] = [int(i) for i in ids.split(",")]
    return out


def model_labels(paths):
    """Report names: the file stems, or the full file names when two stems collide
    (``model.pt`` and its export ``model.mlpackage``)."""
    stems = [p.stem for p in paths]
    if len(set(stems)) == len(stems):
        return stems
    names = [p.name for p in paths]
    if len(set(names)) != len(names):
        raise SystemExit("--weights and --compare must have different file names")
    return names


def check_coreml(weights, conf):
    if not is_coreml(weights):
        return
    if platform.system() != "Darwin":
        raise SystemExit(f"{weights.name}: Core ML inference needs macOS")
    floor = coreml_confidence_floor(weights)
    if floor is not None and floor > conf:
        print(
            f"warning: {weights.name} filters its own boxes below confidence "
            f"{floor:g} (NMS inside the model), above --conf {conf:g}: thresholds "
            f"and false-positive budgets that rely on lower confidences are not "
            "comparable with the other model.",
            file=sys.stderr,
        )


def detections_for(weights, label, frames, args):
    filters = parse_class_filters(args.class_filter)
    classes = filters.get(label, filters.get(weights.stem))
    ids = [f.id for f in frames]
    cache = None
    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        tag = "" if classes is None else "_c" + "-".join(map(str, classes))
        key = f"{label}_{args.imgsz}_{args.conf}{tag}.json"
        cache = args.cache_dir / key
        if cache.exists():
            return load_detections(cache, ids)
    dets = run_detector(
        weights,
        [f.path for f in frames],
        args.imgsz,
        args.conf,
        args.device,
        classes,
    )
    if cache:
        save_detections(cache, ids, dets)
    return dets


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = [args.weights] + ([args.compare] if args.compare else [])
    labels = model_labels(paths)
    if args.agreement and not args.compare:
        raise SystemExit("--agreement needs --compare")
    for weights in paths:
        check_coreml(weights, args.conf)
    frames = load_ground_truth(args.frames_dir, args.labels)
    gts = [f.boxes for f in frames]
    print(
        f"{len(frames)} frames, {sum(len(g) for g in gts)} balls",
        file=sys.stderr,
    )
    detections = {
        label: detections_for(weights, label, frames, args)
        for weights, label in zip(paths, labels)
    }
    matches = {
        label: match_all(gts, dets, args.iou) for label, dets in detections.items()
    }
    report = evaluate(
        gts,
        matches,
        thresholds=tuple(args.thresholds),
        budgets=tuple(args.budgets),
        small_max_side=args.small_px,
        resamples=args.resamples,
        seed=args.seed,
    )
    print(report["table"])
    if "bootstrap" in report:
        b = report["bootstrap"]
        print(f"\nBootstrap over frames, A = {b['A']}, B = {b['B']}:")
        print(bootstrap_table(b["stats"]))
        print(
            "\nRecovery at the first threshold:", report["recovery_at_first_threshold"]
        )
    if args.agreement:
        a, b = labels
        report["agreement"] = {
            "A": a,
            "B": b,
            "iou": args.iou,
            "rows": [
                agreement(detections[a], detections[b], t, args.iou)
                for t in args.thresholds
            ],
        }
        print(f"\nBox agreement, A = {a}, B = {b}, IoU >= {args.iou:g}:")
        print(agreement_table(report["agreement"]["rows"], "A", "B"))
    if args.out_json:
        args.out_json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
