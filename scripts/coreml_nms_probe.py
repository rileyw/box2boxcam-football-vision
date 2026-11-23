"""Fork-only: probe the NMS stage of a Core ML detection pipeline (macOS only).

Calls the pipeline directly on 960x960 letterboxed test frames in several ways and
scores each against the ground truth:

  V0 default inputs (as Ultralytics and the app's Vision request call it)
  V1 iouThreshold = 0.7 and confidenceThreshold = 0.1 passed explicitly
  V2 iouThreshold = 0.45 passed explicitly
  V3 V0 followed by a greedy NMS at IoU 0.7 (an app-side deduplication)

For each variant it also counts pairs of returned boxes in the same frame that still
overlap above IoU 0.7, which a working NMS at 0.7 never returns::

    python scripts/coreml_nms_probe.py MODEL.mlpackage IMAGES LABELS
"""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from box2boxcam.evaluation.ground_truth import load_ground_truth
from box2boxcam.evaluation.matching import (
    count_small_tp,
    count_tp_fp,
    iou,
    match_all,
    small_balls,
)

SIZE = 960
THRESHOLDS = (0.25, 0.35, 0.45, 0.55)


def letterbox(img):
    h, w = img.shape[:2]
    r = min(SIZE / h, SIZE / w)
    nh, nw = round(h * r), round(w * r)
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
    canvas[top : top + nh, left : left + nw] = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_LINEAR
    )
    return canvas[:, :, ::-1].copy(), r, left, top


def to_frame(coordinates, confidence, r, left, top):
    """Normalized centre xywh on the 960 square -> original-image xyxy, ball score."""
    c = np.asarray(coordinates, dtype=float).reshape(-1, 4) * SIZE
    if len(c) == 0:
        return np.empty((0, 4)), np.empty(0)
    xyxy = np.concatenate([c[:, :2] - c[:, 2:] / 2, c[:, :2] + c[:, 2:] / 2], 1)
    xyxy[:, [0, 2]] -= left
    xyxy[:, [1, 3]] -= top
    scores = np.asarray(confidence, dtype=float).reshape(len(c), -1)[:, 0]
    return xyxy / r, scores


def greedy_nms(boxes, scores, threshold):
    keep = []
    for i in np.argsort(-scores):
        if all(iou(boxes[i], boxes[j]) <= threshold for j in keep):
            keep.append(i)
    keep = np.array(keep, dtype=int)
    return boxes[keep].reshape(-1, 4), scores[keep]


def overlapping_pairs(boxes, threshold=0.7):
    n = len(boxes)
    return sum(
        iou(boxes[i], boxes[j]) > threshold for i in range(n) for j in range(i + 1, n)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mlpackage", type=Path)
    p.add_argument("images", type=Path)
    p.add_argument("labels", type=Path)
    p.add_argument("--units", default="CPU_ONLY")
    args = p.parse_args()

    model = ct.models.MLModel(
        str(args.mlpackage), compute_units=getattr(ct.ComputeUnit, args.units)
    )
    inputs = {i.name for i in model.get_spec().description.input}
    frames = load_ground_truth(args.images, args.labels)
    gts = [f.boxes for f in frames]
    small = small_balls(gts)

    variants = {
        "V0 default inputs": {},
        "V1 iou 0.7, conf 0.1 explicit": {
            "iouThreshold": 0.7,
            "confidenceThreshold": 0.1,
        },
        "V2 iou 0.45 explicit": {"iouThreshold": 0.45, "confidenceThreshold": 0.1},
    }
    results = {name: [] for name in [*variants, "V3 V0 + greedy NMS 0.7"]}
    for f in frames:
        rgb, r, left, top = letterbox(cv2.imread(str(f.path)))
        image = Image.fromarray(rgb)
        for name, extra in variants.items():
            feed = {"image": image, **{k: v for k, v in extra.items() if k in inputs}}
            out = model.predict(feed)
            dets = to_frame(out["coordinates"], out["confidence"], r, left, top)
            results[name].append(dets)
            if name.startswith("V0"):
                results["V3 V0 + greedy NMS 0.7"].append(greedy_nms(*dets, 0.7))

    print(
        f"Core ML NMS probe, {len(frames)} frames, units {args.units}, inputs {sorted(inputs)}"
    )
    print(
        "| variant | "
        + " | ".join(f"{t} TP/FP" for t in THRESHOLDS)
        + " | small TP@0.25 | boxes | kept pairs with IoU > 0.7 |"
    )
    print("|---|" + "---|" * (len(THRESHOLDS) + 3))
    for name, dets in results.items():
        m = match_all(gts, dets)
        cells = " | ".join("{}/{}".format(*count_tp_fp(m, t)[:2]) for t in THRESHOLDS)
        boxes = sum(len(b) for b, _ in dets)
        pairs = sum(overlapping_pairs(b) for b, _ in dets)
        print(
            f"| {name} | {cells} | {count_small_tp(m, small, 0.25)} | {boxes} | {pairs} |"
        )


if __name__ == "__main__":
    main()
