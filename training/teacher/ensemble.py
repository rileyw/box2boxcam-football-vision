"""Ensemble teacher: several detectors, optional horizontal-flip TTA, weighted box fusion.

Each member is one model on the original frame, or, with flip, also on the mirrored
frame (its boxes mirrored back). Ultralytics' own TTA is avoided on purpose: it also
runs downscaled copies, which hurt balls a few pixels wide.

Weighted box fusion (WBF), per frame:
1. Take all member boxes by decreasing score.
2. Add each box to the first cluster whose fused box overlaps it with IoU > iou and
   that has no box from the same member yet; otherwise it starts a new cluster.
3. A cluster's box is the score-weighted mean of its boxes, and its score is the sum of
   its scores divided by the number of members. A ball that only one member sees is
   therefore down-weighted.
"""

import numpy as np


def iou_one(box, boxes):
    """IoU of one xyxy box against an (n, 4) array."""
    if len(boxes) == 0:
        return np.zeros(0)
    x0 = np.maximum(box[0], boxes[:, 0])
    y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2])
    y1 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def weighted_box_fusion(member_boxes, member_scores, members, iou=0.55, floor=0.0):
    """Fuse [(n_i, 4)] boxes and [(n_i,)] scores from ``members`` predictions."""
    entries = [(float(s), m, np.asarray(b, dtype=float))
               for m, (bs, ss) in enumerate(zip(member_boxes, member_scores))
               for b, s in zip(bs, ss)]
    entries.sort(key=lambda e: -e[0])
    clusters = []  # each: {"boxes": [...], "scores": [...], "members": set(), "fused": box}
    for score, member, box in entries:
        fused = np.array([c["fused"] for c in clusters]).reshape(-1, 4)
        overlaps = iou_one(box, fused)
        target = None
        for index in np.argsort(-overlaps):
            if overlaps[index] <= iou:
                break
            if member not in clusters[index]["members"]:
                target = clusters[index]
                break
        if target is None:
            clusters.append({"boxes": [box], "scores": [score], "members": {member}, "fused": box})
            continue
        target["boxes"].append(box)
        target["scores"].append(score)
        target["members"].add(member)
        w = np.asarray(target["scores"])
        target["fused"] = (np.asarray(target["boxes"]) * w[:, None]).sum(0) / w.sum()
    boxes = np.array([c["fused"] for c in clusters]).reshape(-1, 4)
    scores = np.array([sum(c["scores"]) / members for c in clusters])
    keep = scores >= floor
    order = np.argsort(-scores[keep])
    return boxes[keep][order], scores[keep][order]


def predict(weights, image_paths, imgsz=960, conf=0.05, flip=False, iou=0.55, floor=None,
            device=0, batch=8):
    """[(boxes xyxy (n, 4), scores (n,))] per image from the fused ensemble."""
    import cv2
    from ultralytics import YOLO

    models = [YOLO(str(w)) for w in weights]
    views = (False, True) if flip else (False,)
    members = len(models) * len(views)
    floor = conf if floor is None else floor
    out = []
    for start in range(0, len(image_paths), batch):
        paths = image_paths[start:start + batch]
        images = [cv2.imread(str(p)) for p in paths]
        if any(i is None for i in images):
            raise OSError(f"cannot read one of {paths}")
        per_image = [([], []) for _ in images]
        for model in models:
            for mirrored in views:
                batch_images = [cv2.flip(i, 1) if mirrored else i for i in images]
                results = model.predict(batch_images, imgsz=imgsz, conf=conf, device=device, verbose=False)
                for k, (image, result) in enumerate(zip(images, results)):
                    b = result.boxes.xyxy.cpu().numpy().astype(float)
                    if mirrored and len(b):
                        width = image.shape[1]
                        b = np.stack([width - b[:, 2], b[:, 1], width - b[:, 0], b[:, 3]], 1)
                    per_image[k][0].append(b)
                    per_image[k][1].append(result.boxes.conf.cpu().numpy().astype(float))
        for boxes, scores in per_image:
            out.append(weighted_box_fusion(boxes, scores, members, iou=iou, floor=floor))
    return out
