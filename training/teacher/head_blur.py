"""De-identify a frame before it is ever saved: blur every person's head, and vehicles.

Modelled on what street-imagery services do (automatic blurring tuned for recall, an
irreversible blur, and a removal process), applied to match frames:

- People: a pose model (stock YOLO26s-pose) and a detector (stock YOLO26s, COCO) run on
  the whole frame and on 4 overlapping tiles, so far and small people are found too.
  Every person gets a head region: from the nose, eye and ear keypoints when at least
  two are confident, otherwise the top 22 % of the person's box. Back-turned heads are
  blurred as well as faces.
- Vehicles (car, motorcycle, bus, truck): the lower half of each vehicle is blurred,
  where number plates are (no plate detector is available offline).
- Tiles: a box that touches an inner tile edge (a person or vehicle cut by the tile) is
  dropped; the whole-frame pass or the neighbouring tile sees it whole.
- Blur: each region is enlarged, pixelated in blocks, then Gaussian-blurred, which can't
  be reversed.

This reduces identifiability; it does not anonymise. Kit, numbers, build and context
remain, and a person no detector finds is not blurred. Check samples with
teacher/privacy_qa.py.
"""

import cv2
import numpy as np

HEAD_KEYPOINTS = (0, 1, 2, 3, 4)  # nose, left eye, right eye, left ear, right ear
VEHICLE_CLASSES = (2, 3, 5, 7)    # COCO car, motorcycle, bus, truck


def head_boxes(person_boxes, keypoints, keypoint_conf, kp_threshold=0.3):
    """Head regions (x0, y0, x1, y1) for each person, before enlarging."""
    out = []
    for i, (x0, y0, x1, y1) in enumerate(person_boxes):
        h = max(y1 - y0, 1.0)
        size = 0.16 * h  # a head is roughly 1/7 to 1/8 of a standing body
        if keypoints is not None and i < len(keypoints):
            pts = keypoints[i][list(HEAD_KEYPOINTS)]
            conf = keypoint_conf[i][list(HEAD_KEYPOINTS)]
            good = pts[conf >= kp_threshold]
            if len(good) >= 2:
                cx, cy = good.mean(axis=0)
                spread = float(np.ptp(good, axis=0).max())
                size = max(size, 2.2 * spread)
                out.append((cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2))
                continue
        # No usable keypoints: a square at the top centre of the box, about one head plus margin.
        side = min(x1 - x0, 0.22 * h)
        cx = (x0 + x1) / 2
        out.append((cx - side / 2, y0, cx + side / 2, y0 + side))
    return out


def blur_regions(image, boxes, margin=0.25, min_size=6):
    """Pixelate then blur each box (enlarged by margin) in place. Returns the count."""
    height, width = image.shape[:2]
    count = 0
    for x0, y0, x1, y1 in boxes:
        w, h = x1 - x0, y1 - y0
        x0, x1 = int(max(0, x0 - margin * w)), int(min(width, x1 + margin * w))
        y0, y1 = int(max(0, y0 - margin * h)), int(min(height, y1 + margin * h))
        if x1 - x0 < min_size or y1 - y0 < min_size:
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            x0, x1 = max(0, cx - min_size), min(width, cx + min_size)
            y0, y1 = max(0, cy - min_size), min(height, cy + min_size)
        region = image[y0:y1, x0:x1]
        if region.size == 0:
            continue
        rh, rw = region.shape[:2]
        block = max(2, min(rw, rh) // 6)  # about 6 blocks across: no facial detail survives
        small = cv2.resize(region, (max(1, rw // block), max(1, rh // block)), interpolation=cv2.INTER_AREA)
        region[:] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        k = max(3, (min(rw, rh) // 2) | 1)
        region[:] = cv2.GaussianBlur(region, (k, k), 0)
        count += 1
    return count


def tiles(width, height, overlap=0.15):
    """The whole frame plus a 2x2 grid of overlapping tiles, as (x0, y0, x1, y1)."""
    tw, th = int(width / 2 * (1 + overlap)), int(height / 2 * (1 + overlap))
    out = [(0, 0, width, height)]
    for x0 in (0, width - tw):
        for y0 in (0, height - th):
            out.append((x0, y0, x0 + tw, y0 + th))
    return out


def touches_inner_edge(box, tile, width, height, tolerance=2):
    """True when the box reaches a tile edge that is not also a frame edge."""
    x0, y0, x1, y1 = tile
    bx0, by0, bx1, by1 = box
    return ((x0 > 0 and bx0 <= x0 + tolerance) or (y0 > 0 and by0 <= y0 + tolerance)
            or (x1 < width and bx1 >= x1 - tolerance) or (y1 < height and by1 >= y1 - tolerance))


def nms(boxes, scores, threshold=0.5):
    """Greedy NMS over (n, 4) boxes; returns kept indices."""
    import torch
    import torchvision

    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    keep = torchvision.ops.nms(torch.as_tensor(boxes, dtype=torch.float32),
                               torch.as_tensor(scores, dtype=torch.float32), threshold)
    return keep.numpy()


class HeadBlurrer:
    """Callable: blur(image) -> number of regions blurred (image modified in place)."""

    def __init__(self, weights="yolo26s-pose.pt", detector="yolo26s.pt", imgsz=1280,
                 conf=0.15, device=0, tiled=True, vehicles=True):
        from ultralytics import YOLO

        self.pose = YOLO(weights)
        self.detector = YOLO(detector) if detector else None
        self.imgsz, self.conf, self.device = imgsz, conf, device
        self.tiled, self.vehicles = tiled, vehicles

    def _people_and_vehicles(self, image):
        height, width = image.shape[:2]
        regions = tiles(width, height) if self.tiled else [(0, 0, width, height)]
        people, kps, kconf, scores, vehicles = [], [], [], [], []
        for x0, y0, x1, y1 in regions:
            crop = np.ascontiguousarray(image[y0:y1, x0:x1])
            pose = self.pose.predict(crop, imgsz=self.imgsz, conf=self.conf,
                                     device=self.device, verbose=False)[0]
            tile = (x0, y0, x1, y1)
            if pose.boxes is not None and len(pose.boxes):
                b = pose.boxes.xyxy.cpu().numpy() + [x0, y0, x0, y0]
                c = pose.boxes.conf.cpu().numpy()
                has_kp = pose.keypoints is not None and pose.keypoints.conf is not None
                k_xy = pose.keypoints.xy.cpu().numpy() + [x0, y0] if has_kp else np.zeros((len(b), 17, 2))
                k_c = pose.keypoints.conf.cpu().numpy() if has_kp else np.zeros((len(b), 17))
                for i in range(len(b)):
                    if touches_inner_edge(b[i], tile, width, height):
                        continue
                    people.append(b[i])
                    scores.append(c[i])
                    kps.append(k_xy[i])
                    kconf.append(k_c[i])
            if self.detector is None:
                continue
            det = self.detector.predict(crop, imgsz=self.imgsz, conf=self.conf,
                                        device=self.device, verbose=False)[0]
            if det.boxes is None or len(det.boxes) == 0:
                continue
            cls = det.boxes.cls.cpu().numpy().astype(int)
            b = det.boxes.xyxy.cpu().numpy() + [x0, y0, x0, y0]
            c = det.boxes.conf.cpu().numpy()
            for box, k, s in zip(b, cls, c):
                if touches_inner_edge(box, tile, width, height):
                    continue
                if k == 0:  # a person the pose model may have missed: no keypoints
                    people.append(box)
                    scores.append(s * 0.999)  # prefer the pose box when both exist
                    kps.append(np.zeros((17, 2)))
                    kconf.append(np.zeros(17))
                elif self.vehicles and k in VEHICLE_CLASSES:
                    vx0, vy0, vx1, vy1 = box
                    vehicles.append((vx0, (vy0 + vy1) / 2, vx1, vy1))  # lower half: plates
        if not people:
            return [], vehicles
        people, scores = np.asarray(people), np.asarray(scores)
        keep = nms(people, scores, 0.5)
        heads = head_boxes(people[keep], np.asarray(kps)[keep], np.asarray(kconf)[keep])
        return heads, vehicles

    def __call__(self, image):
        heads, vehicles = self._people_and_vehicles(image)
        return blur_regions(image, heads) + blur_regions(image, vehicles, margin=0.05)
