"""Pseudo-label one match video with the teacher, ready for human review in Label Studio.

Samples frames, runs the teacher, and sorts every frame:

- auto    (role train only) every teacher box >= low is also >= high: the boxes become
          YOLO labels without review.
- review  a box between low and high, or role reserved (every frame): goes to Label
          Studio with the teacher's boxes as pre-annotations.
- skipped no box >= low. These are not used: a missed small ball would otherwise
          become a false "no ball" label. --review-empty sends a sample to review.

A reserved match is a future held-out test. Nothing is auto-accepted and
build_dataset.py never puts it in training.

    python teacher/pseudo_label.py --video match.mp4 --match club_20261130 --role train \
        --teacher runs/teacher_x960_s42/weights/best.pt --thresholds teacher/thresholds.json

Output: pseudo/<match>/ with frames/, labels/ (auto), manifest.csv, review/tasks.json
(import into Label Studio) and summary.json. Frames stay local: they show people.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "teacher"))
from ensemble import predict as ensemble_predict  # noqa: E402


def yolo_line(box, width, height):
    x0, y0, x1, y1 = box
    return (f"0 {(x0 + x1) / 2 / width:.8f} {(y0 + y1) / 2 / height:.8f} "
            f"{(x1 - x0) / width:.8f} {(y1 - y0) / height:.8f}\n")


def rectangle(index, box, conf, width, height):
    x0, y0, x1, y1 = box
    return {
        "id": f"teacher-{index}", "from_name": "bbox", "to_name": "image", "type": "rectangle",
        "original_width": width, "original_height": height, "image_rotation": 0, "score": conf,
        "value": {"x": 100 * x0 / width, "y": 100 * y0 / height,
                  "width": 100 * (x1 - x0) / width, "height": 100 * (y1 - y0) / height,
                  "rotation": 0},
    }


def triage(confidences, low, high, role):
    kept = [c for c in confidences if c >= low]
    if role == "reserved":
        return "review"
    if not kept:
        return "skipped"
    return "auto" if all(c >= high for c in kept) else "review"


def sample_frames(video, stride, out_dir, prefix):
    """Every stride-th frame as JPEG. Returns [(item_id, frame_number, path)], fps, size."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out_dir.mkdir(parents=True, exist_ok=True)
    items, index = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if index % stride == 0:
            ok, image = cap.retrieve()
            if ok:
                item = f"{prefix}_{index:08d}"
                path = out_dir / f"{item}.jpg"
                if not path.exists():
                    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                items.append((item, index, path))
        index += 1
    cap.release()
    return items, fps, size


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--match", required=True, help="Match id, e.g. club_20261130 (one per match).")
    p.add_argument("--role", choices=("train", "reserved"), required=True)
    p.add_argument("--teacher", type=Path, nargs="+", required=True,
                   help="Same model(s) as calibrated (thresholds.json).")
    p.add_argument("--flip", action="store_true", help="Same as calibrated.")
    p.add_argument("--fusion-iou", type=float, default=0.55)
    p.add_argument("--thresholds", type=Path, required=True)
    p.add_argument("--stride", type=int, default=15, help="Every Nth frame (15 = 2 per second at 30 fps).")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--review-empty", type=float, default=0.0,
                   help="Share of skipped frames sent to review as negative candidates.")
    p.add_argument("--audit-auto", type=float, default=0.0,
                   help="Share of auto-accepted frames sent to review instead, to measure "
                        "how precise auto-accept really is on this footage.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=HERE / "pseudo")
    args = p.parse_args()

    cal = json.loads(args.thresholds.read_text())
    low, high = cal["thresholds"]["low"], cal["thresholds"]["high"]
    if high is None:
        raise SystemExit("thresholds.json has no auto-accept threshold; recalibrate.")
    config = cal.get("teacher_config", {})
    if config and (sorted(map(str, args.teacher)) != sorted(config["weights"])
                   or args.flip != config["flip"] or args.imgsz != config["imgsz"]):
        raise SystemExit(f"Teacher setup differs from the calibrated one: {config}")
    if not cal.get("gate_teacher_beats_student_on_val", False):
        print("warning: this teacher did not beat the student on val (see thresholds.json)")

    root = args.out / args.match
    frames, labels, review = root / "frames", root / "labels", root / "review"
    labels.mkdir(parents=True, exist_ok=True)
    review.mkdir(parents=True, exist_ok=True)
    items, fps, (width, height) = sample_frames(args.video, args.stride, frames, args.match)
    print(f"{len(items)} frames sampled from {args.video.name} ({width}x{height}, {fps:.1f} fps)")

    rng = random.Random(args.seed)
    rows, tasks, counts = [], [], {"auto": 0, "review": 0, "skipped": 0}
    for start in range(0, len(items), args.batch):
        chunk = items[start:start + args.batch]
        fused = ensemble_predict(args.teacher, [path for _, _, path in chunk], args.imgsz, 0.05,
                                 flip=args.flip, iou=args.fusion_iou, floor=low, batch=args.batch)
        for (item, frame, path), (fused_boxes, fused_scores) in zip(chunk, fused):
            boxes = fused_boxes.tolist()
            confs = [float(c) for c in fused_scores]
            status = triage(confs, low, high, args.role)
            audit = False
            if status == "skipped" and args.review_empty and rng.random() < args.review_empty:
                status = "review"
            elif status == "auto" and args.audit_auto and rng.random() < args.audit_auto:
                status, audit = "review", True
            counts[status] += 1
            if status == "auto":
                (labels / f"{item}.txt").write_text(
                    "".join(yolo_line(b, width, height) for b in boxes))
            if status == "review":
                context = " · ".join(f"{c:.0%}" for c in sorted(confs, reverse=True)) or "no box"
                tasks.append({
                    "data": {
                        "image": f"/data/local-files/?d={args.match}/frames/{item}.jpg",
                        "item_id": item, "match": args.match, "role": args.role, "audit": audit,
                        "source_frame": frame, "timestamp_seconds": round(frame / fps, 3),
                        "frame_info": f"{item} · {frame / fps:.1f} s · {args.role}"
                                      + (" · audit of an auto-accepted frame" if audit else ""),
                        "luna_context": "",
                        "teacher_context": f"Teacher boxes (>= {low:.2f}): {context}",
                    },
                    "predictions": [{
                        "model_version": "+".join(w.stem for w in args.teacher) + ("+flip" if args.flip else ""),
                        "score": max(confs, default=0.0),
                        "result": [rectangle(i, b, c, width, height)
                                   for i, (b, c) in enumerate(zip(boxes, confs))],
                    }],
                })
            rows.append({"item_id": item, "match": args.match, "role": args.role,
                         "source_frame": frame, "status": status, "audit": audit, "boxes": len(confs),
                         "max_conf": round(max(confs, default=0.0), 4)})

    with (root / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (review / "tasks.json").write_text(json.dumps(tasks, indent=1))
    summary = {"match": args.match, "role": args.role, "video": str(args.video),
               "teacher": [str(w) for w in args.teacher], "flip": args.flip, "low": low, "high": high, "stride": args.stride,
               "frames": len(items), **counts}
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
