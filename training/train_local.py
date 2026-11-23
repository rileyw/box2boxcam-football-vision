"""Local training with the box2boxcam v1.0 recipe on the public dataset (5070 Ti).

Same settings as configs/mobile_small_auto_field_v006_960.json and
scripts/train_mobile_hf.py (COCO YOLO26s, full sports-ball transfer, 960 px, AdamW
lr 0.001, cosine, mosaic 0, scale 0.3, AMP, seed 42), by default a full 30-epoch run; --fraction/--epochs make a smoke test.

    python train_local.py --seed 42 --name base_s42            # full baseline
    python train_local.py --fraction 0.25 --epochs 1 --name smoke
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
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--model", default="yolo26s.pt")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--name", default="smoke")
    p.add_argument("--data", type=Path, help="Dataset yaml (default: the public splits).")
    args = p.parse_args()

    if args.data:
        yaml = args.data.resolve()
    else:
        data = HERE / "data/public"
        yaml = HERE / "data/public_sports_ball.yaml"
        yaml.write_text(
            f"path: {json.dumps(str(data))}\ntrain: images/train\nval: images/val\n"
            "test: null\nnames:\n  0: \"sports ball\"\n"
        )
    torch.cuda.reset_peak_memory_stats()
    model = YOLO(args.model)
    started = time.perf_counter()
    model.train(
        data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        fraction=args.fraction, seed=args.seed, patience=args.patience, device=0, workers=4, deterministic=True,
        optimizer="AdamW", lr0=0.001, cos_lr=True, amp=True, cache=False,
        mosaic=0.0, mixup=0.0, close_mosaic=0, scale=0.3, plots=False, val=True,
        project=str(HERE / "runs"), name=args.name, exist_ok=True, verbose=False,
        trainer=SportsBallTransferTrainer,
    )
    seconds = time.perf_counter() - started
    images = int(1673 * args.fraction) * args.epochs
    print(json.dumps({
        "seed": args.seed, "batch": args.batch, "imgsz": args.imgsz, "train_images_seen": images,
        "wall_seconds_incl_val": round(seconds, 1),
        "peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }))


if __name__ == "__main__":
    main()
