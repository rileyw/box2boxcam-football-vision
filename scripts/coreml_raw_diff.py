"""Fork-only: compare raw Core ML outputs with the PyTorch graph they were converted from.

Feeds the same 960x960 letterboxed frames to a raw (no NMS) Core ML export and to the
PyTorch model prepared exactly as the exporter traces it, then compares the per-anchor
ball scores and boxes. macOS only (Core ML inference)::

    python scripts/coreml_raw_diff.py MODEL.pt RAW.mlpackage FRAMES_DIR [--frames 60]
"""

import argparse
import copy
from pathlib import Path

import coremltools as ct
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect


def letterbox(img, size=960):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_LINEAR
    )
    return canvas[:, :, ::-1].copy()  # RGB


def export_mode(pt):
    model = copy.deepcopy(YOLO(str(pt)).model).float().eval().fuse()
    for m in model.modules():
        if isinstance(m, Detect):
            m.dynamic, m.export, m.format, m.xyxy = False, True, "coreml", False
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pt", type=Path)
    p.add_argument("mlpackage", type=Path)
    p.add_argument("frames_dir", type=Path)
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--units", default="CPU_ONLY")
    args = p.parse_args()

    torch_model = export_mode(args.pt)
    coreml = ct.models.MLModel(
        str(args.mlpackage), compute_units=getattr(ct.ComputeUnit, args.units)
    )
    output = coreml.get_spec().description.output[0].name
    paths = sorted(args.frames_dir.glob("*.jpg"))[: args.frames]

    score_diff, box_diff, above_t, above_c, flips = [], [], 0, 0, 0
    for path in paths:
        rgb = letterbox(cv2.imread(str(path)))
        with torch.no_grad():
            x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255
            t = torch_model(x)
            t = (t[0] if isinstance(t, (list, tuple)) else t)[0].numpy()  # (5, A)
        c = np.asarray(coreml.predict({"image": Image.fromarray(rgb)})[output])[0]
        score_diff.append(np.abs(t[4] - c[4]))
        box_diff.append(np.abs(t[:4] - c[:4]).max(axis=0))
        above_t += int((t[4] >= 0.1).sum())
        above_c += int((c[4] >= 0.1).sum())
        flips += int(((t[4] >= 0.25) != (c[4] >= 0.25)).sum())

    s = np.concatenate(score_diff)
    b = np.concatenate(box_diff)
    print(
        f"raw graph diff, {len(paths)} frames, {s.size} anchors, Core ML on {args.units}:"
    )
    print(
        f"  score |torch - coreml|: mean {s.mean():.5f}, p99 {np.quantile(s, 0.99):.5f}, max {s.max():.4f}"
    )
    print(
        f"  box   |torch - coreml| px: mean {b.mean():.4f}, p99 {np.quantile(b, 0.99):.4f}, max {b.max():.3f}"
    )
    print(f"  anchors with score >= 0.1: torch {above_t}, coreml {above_c}")
    print(f"  anchors crossing 0.25 between the two: {flips}")


if __name__ == "__main__":
    main()
