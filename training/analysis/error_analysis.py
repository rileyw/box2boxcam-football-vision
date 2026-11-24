"""Break a detector's errors on VAL into kinds, to decide which experiment is worth it.

At one confidence threshold, every false positive is classed by its first matching rule:
- near_miss: overlaps a ground-truth ball with 0.1 <= IoU < 0.5 (a box on the ball,
  misplaced or mis-sized; common for 8-16 px balls)
- near_ball: centre within 3 ball widths of a ground-truth ball, IoU < 0.1
- duplicate: overlaps a higher-scoring detection of the same frame with IoU > 0.3
- static: a box within 1 % of the frame from it, in at least 3 other frames of the clip
  (spare ball, cone, light, marking)
- isolated: anything else
Every missed ball is classed as no_candidate (no detection >= 0.05 within 3 ball widths),
weak (a detection near it, all below the threshold) or misplaced (near it, above the
threshold, but IoU < 0.5).

Val only: never run this on the test split.

    python analysis/error_analysis.py --weights eval/models/base_s42.pt --conf 0.25
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from box2boxcam.evaluation.detect import run_detector
from box2boxcam.evaluation.ground_truth import load_ground_truth
from box2boxcam.evaluation.matching import iou, match_frame

HERE = Path(__file__).resolve().parents[1]


def centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def near(b, g, widths=3.0):
    (bx, by), (gx, gy) = centre(b), centre(g)
    size = max(g[2] - g[0], g[3] - g[1], 1.0)
    return abs(bx - gx) <= widths * size and abs(by - gy) <= widths * size


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--images", type=Path, default=HERE / "data/public/images/val")
    p.add_argument("--labels", type=Path, default=HERE / "data/public/labels/val")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if "test" in args.images.parts:
        raise SystemExit("Error analysis runs on val, never on the test split.")

    frames = load_ground_truth(args.images, args.labels)
    dets = run_detector(args.weights, [f.path for f in frames], args.imgsz, 0.05, 0)
    width = 1920.0
    fp_kinds, fn_kinds = Counter(), Counter()
    fp_sizes, fn_sizes = defaultdict(list), defaultdict(list)
    fp_boxes = []  # (clip, frame_no, box, index) for the static rule
    per_fp = []
    for f, (boxes, confs) in zip(frames, dets):
        gts = np.asarray(f.boxes, dtype=float).reshape(-1, 4)
        order = np.argsort(-confs)
        matches = dict(zip(order, (j for _, j in match_frame(gts, boxes, confs))))
        m = re.match(r"(.+)_(\d+)$", f.path.stem)
        clip, frame_no = (m.group(1), int(m.group(2))) if m else (f.path.stem, 0)
        kept = [k for k in order if confs[k] >= args.conf]
        for rank, k in enumerate(kept):
            if matches[k] >= 0:
                continue
            b = boxes[k]
            overlaps = [iou(b, g) for g in gts]
            if overlaps and max(overlaps) >= 0.1:
                kind = "near_miss"
            elif any(near(b, g) for g in gts):
                kind = "near_ball"
            elif any(iou(b, boxes[h]) > 0.3 for h in kept[:rank]):
                kind = "duplicate"
            else:
                kind = "isolated"  # may become static below
            per_fp.append([kind, clip, frame_no, b])
        found = {matches[k] for k in kept if matches[k] >= 0}
        for j, g in enumerate(gts):
            if j in found:
                continue
            around = [k for k in range(len(boxes)) if near(boxes[k], g) or iou(boxes[k], g) > 0]
            if not around:
                kind = "no_candidate"
            elif all(confs[k] < args.conf for k in around):
                kind = "weak"
            else:
                kind = "misplaced"
            fn_kinds[kind] += 1
            fn_sizes[kind].append(max(g[2] - g[0], g[3] - g[1]))
    # Static: an isolated false positive whose position repeats in >= 3 other frames.
    by_clip = defaultdict(list)
    for entry in per_fp:
        by_clip[entry[1]].append(entry)
    for entry in per_fp:
        if entry[0] != "isolated":
            continue
        cx, cy = centre(entry[3])
        repeats = sum(1 for other in by_clip[entry[1]]
                      if other is not entry and other[2] != entry[2]
                      and abs(centre(other[3])[0] - cx) <= 0.01 * width
                      and abs(centre(other[3])[1] - cy) <= 0.01 * width)
        if repeats >= 3:
            entry[0] = "static"
    for kind, _, _, b in per_fp:
        fp_kinds[kind] += 1
        fp_sizes[kind].append(max(b[2] - b[0], b[3] - b[1]))

    def table(counts, sizes):
        total = sum(counts.values()) or 1
        return {k: {"n": v, "share": round(v / total, 3), "median_px": round(float(np.median(sizes[k])), 1)}
                for k, v in counts.most_common()}

    report = {"weights": str(args.weights), "split": str(args.images), "conf": args.conf,
              "balls": int(sum(len(f.boxes) for f in frames)),
              "false_positives": table(fp_kinds, fp_sizes), "misses": table(fn_kinds, fn_sizes)}
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
