"""Run an Ultralytics detector on evaluation frames and cache the raw detections."""

import json
from pathlib import Path

import numpy as np


def run_detector(weights, image_paths, imgsz=960, conf=0.05, device=None, classes=None):
    """Detections ``[(boxes_xyxy (m, 4), confidences (m,)), ...]`` in image pixels.

    Images are read at full resolution and resized by the model to ``imgsz``;
    ``conf`` is the lowest confidence kept (all thresholds are applied afterwards).
    ``classes`` restricts the output to those class ids (e.g. ``[32]`` keeps only the
    COCO "sports ball" of a stock model).
    """
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise OSError(f"cannot read {path}")
        result = model.predict(
            image,
            imgsz=imgsz,
            conf=conf,
            device=device,
            classes=classes,
            verbose=False,
        )[0].boxes
        out.append((result.xyxy.cpu().numpy(), result.conf.cpu().numpy()))
    return out


def is_coreml(weights):
    return Path(weights).suffix == ".mlpackage"


def coreml_confidence_floor(weights):
    """Confidence threshold of the NMS stage inside a Core ML pipeline, or ``None``.

    Ultralytics passes only the image to a Core ML model exported with ``nms=True``,
    so the model's own threshold, not ``conf``, decides which boxes come out.
    Reading the spec needs ``coremltools``; returns ``None`` when it is missing or
    the model has no NMS stage.
    """
    try:
        import coremltools as ct
    except ImportError:
        return None
    spec = ct.utils.load_spec(str(weights))
    if spec.WhichOneof("Type") != "pipeline":
        return None
    for model in spec.pipeline.models:
        if model.WhichOneof("Type") == "nonMaximumSuppression":
            return float(model.nonMaximumSuppression.confidenceThreshold)
    return None


def save_detections(path, ids, detections):
    """Write detections as JSON (``{frame id: {"boxes": ..., "conf": ...}}``)."""
    data = {
        i: {"boxes": np.asarray(b).tolist(), "conf": np.asarray(c).tolist()}
        for i, (b, c) in zip(ids, detections)
    }
    Path(path).write_text(json.dumps(data))


def load_detections(path, ids):
    data = json.loads(Path(path).read_text())
    return [
        (
            np.asarray(data[i]["boxes"], dtype=float).reshape(-1, 4),
            np.asarray(data[i]["conf"], dtype=float),
        )
        for i in ids
    ]
