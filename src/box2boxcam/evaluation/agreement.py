"""Box-level agreement between two detectors on the same frames.

Meant for checking that an export (e.g. Core ML, fp16, NMS inside the model) behaves
like the model it came from, independently of ground truth: both detection lists are
cut at the same confidence, then the boxes of model A are matched to those of model B
with the greedy matching used for ground truth (by decreasing confidence of A,
IoU >= ``iou_threshold``).
"""

import numpy as np

from box2boxcam.evaluation.matching import IOU_THRESHOLD, iou, match_frame


def agreement(detections_a, detections_b, conf, iou_threshold=IOU_THRESHOLD):
    """Agreement of two detection lists ``[(boxes_xyxy, confidences), ...]``.

    Returns the number of matched boxes, boxes found only by A or only by B, the
    frames where both models give the same number of boxes and all are matched, and,
    over matched pairs, the mean IoU and the mean / max absolute confidence difference.
    """
    if len(detections_a) != len(detections_b):
        raise ValueError("Both models must be run on the same frames")
    matched = only_a = only_b = identical = 0
    ious, deltas = [], []
    for (boxes_a, conf_a), (boxes_b, conf_b) in zip(detections_a, detections_b):
        keep_a = np.asarray(conf_a, dtype=float) >= conf
        keep_b = np.asarray(conf_b, dtype=float) >= conf
        boxes_a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)[keep_a]
        boxes_b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)[keep_b]
        conf_a = np.asarray(conf_a, dtype=float)[keep_a]
        conf_b = np.asarray(conf_b, dtype=float)[keep_b]
        pairs = match_frame(boxes_b, boxes_a, conf_a, iou_threshold)
        order = np.argsort(-conf_a)
        frame_matched = 0
        for k, (_, j) in zip(order, pairs):
            if j < 0:
                continue
            frame_matched += 1
            ious.append(iou(boxes_a[k], boxes_b[j]))
            deltas.append(abs(conf_a[k] - conf_b[j]))
        matched += frame_matched
        only_a += len(boxes_a) - frame_matched
        only_b += len(boxes_b) - frame_matched
        identical += frame_matched == len(boxes_a) == len(boxes_b)
    return {
        "conf": conf,
        "frames": len(detections_a),
        "identical_frames": identical,
        "matched": matched,
        "only_a": only_a,
        "only_b": only_b,
        "mean_iou": float(np.mean(ious)) if ious else None,
        "mean_abs_conf_diff": float(np.mean(deltas)) if deltas else None,
        "max_abs_conf_diff": float(np.max(deltas)) if deltas else None,
    }


def agreement_table(rows, name_a, name_b):
    """Markdown table of :func:`agreement` results, one row per confidence."""

    def fmt(value):
        return "-" if value is None else f"{value:.3f}"

    lines = [
        (
            f"| conf | matched | only {name_a} | only {name_b} | identical frames "
            "| mean IoU | mean abs conf diff | max abs conf diff |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['conf']:.2f} | {r['matched']} | {r['only_a']} | {r['only_b']} "
            f"| {r['identical_frames']} / {r['frames']} | {fmt(r['mean_iou'])} "
            f"| {fmt(r['mean_abs_conf_diff'])} | {fmt(r['max_abs_conf_diff'])} |"
        )
    return "\n".join(lines)
