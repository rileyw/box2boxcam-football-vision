"""Short local smoke test of the box2boxcam v1.0 recipe on the public dataset.

Same settings as configs/mobile_small_auto_field_v006_960.json and
scripts/train_mobile_hf.py (COCO YOLO26s, full sports-ball transfer, 960 px, AdamW
lr 0.001, cosine, mosaic 0, scale 0.3, AMP, seed 42), but a fraction of an epoch.

    python smoke_train.py --batch 8 --fraction 0.25 --epochs 1
"""

import argparse
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from box2boxcam.training.sports_ball_transfer import SportsBallTransferTrainer

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--fraction", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--name", default="smoke")
    args = p.parse_args()

    data = HERE / "data/public"
    yaml = HERE / "data/public_sports_ball.yaml"
    yaml.write_text(
        f"path: {json.dumps(str(data))}\ntrain: images/train\nval: images/val\n"
        "test: null\nnames:\n  0: \"sports ball\"\n"
    )
    torch.cuda.reset_peak_memory_stats()
    model = YOLO("yolo26s.pt")
    started = time.perf_counter()
    model.train(
        data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        fraction=args.fraction, seed=42, device=0, workers=4, deterministic=True,
        optimizer="AdamW", lr0=0.001, cos_lr=True, amp=True, cache=False,
        mosaic=0.0, mixup=0.0, close_mosaic=0, scale=0.3, plots=False, val=True,
        project=str(HERE / "runs"), name=args.name, exist_ok=True, verbose=False,
        trainer=SportsBallTransferTrainer,
    )
    seconds = time.perf_counter() - started
    images = int(1673 * args.fraction) * args.epochs
    print(json.dumps({
        "batch": args.batch, "imgsz": args.imgsz, "train_images_seen": images,
        "wall_seconds_incl_val": round(seconds, 1),
        "peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }))


if __name__ == "__main__":
    main()
