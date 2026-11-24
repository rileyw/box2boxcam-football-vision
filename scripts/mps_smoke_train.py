"""Fork-only: smoke-test training on Apple's MPS backend with the box2boxcam v1.0 recipe.

Runs a short training (COCO YOLO26s, sports-ball transfer, 960 px, AdamW 0.001, mosaic
0, scale 0.3) on part of the public train split for one or more batch sizes, and reports:

- whether training runs at all on the device, and whether losses stay finite
- training images per second and validation time
- peak memory
- PyTorch ops that fell back to the CPU (PYTORCH_ENABLE_MPS_FALLBACK)
- the time a full 30-epoch run would take, extrapolated

Writes a JSON report and prints a markdown table. Also runs on CUDA (--device 0), which
is how it was tested.

    python scripts/mps_smoke_train.py --data parity-run/dataset --out parity-run/mps
"""

import argparse
import json
import math
import platform
import time
import warnings
from pathlib import Path

import torch

FULL_TRAIN_IMAGES = 1673
REFERENCE = "RTX 5070 Ti (CUDA, AMP): 30 epochs at batch 8 in 12.3 min including validation"


def peak_memory_gib(device):
    if device == "mps" and torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 2**30
    if torch.cuda.is_available() and device not in ("cpu", "mps"):
        return torch.cuda.max_memory_allocated() / 2**30
    return None


def loss_values(tloss):
    """Mean training losses of the epoch as floats, whatever container Ultralytics uses."""
    if tloss is None:
        return None
    if isinstance(tloss, dict):
        return [float(v) for v in tloss.values()]
    if isinstance(tloss, torch.Tensor):
        return [float(v) for v in tloss.detach().flatten().cpu()]
    return [float(v) for v in tloss]


def run(data_yaml, device, batch, fraction, epochs, imgsz, out):
    from ultralytics import YOLO

    from box2boxcam.training.sports_ball_transfer import SportsBallTransferTrainer

    if torch.cuda.is_available() and device not in ("cpu", "mps"):
        torch.cuda.reset_peak_memory_stats()
    timing = {"train": [], "val": [], "peak_mem": 0.0, "images": None}
    marks = {}

    def epoch_start(trainer):
        marks["train"] = time.perf_counter()
        timing["images"] = len(trainer.train_loader.dataset)

    def batch_end(trainer):
        mem = peak_memory_gib(device)
        if mem is not None:
            timing["peak_mem"] = max(timing["peak_mem"], mem)

    def epoch_end(trainer):
        now = time.perf_counter()
        timing["train"].append(now - marks["train"])
        marks["val"] = now
        timing["loss"] = loss_values(trainer.tloss)

    def fit_epoch_end(trainer):
        timing["val"].append(time.perf_counter() - marks["val"])

    model = YOLO("yolo26s.pt")
    for event, fn in (("on_train_epoch_start", epoch_start), ("on_train_batch_end", batch_end),
                      ("on_train_epoch_end", epoch_end), ("on_fit_epoch_end", fit_epoch_end)):
        model.add_callback(event, fn)
    started = time.perf_counter()
    error = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model.train(
                data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, fraction=fraction,
                seed=42, patience=epochs, device=device, workers=4, deterministic=True,
                optimizer="AdamW", lr0=0.001, cos_lr=True, amp=True, cache=False,
                mosaic=0.0, mixup=0.0, close_mosaic=0, scale=0.3, plots=False, val=True,
                project=str(out), name=f"b{batch}", exist_ok=True, verbose=False,
                trainer=SportsBallTransferTrainer,
            )
        except Exception as exc:  # noqa: BLE001 - a failure on MPS is a result to report
            error = f"{type(exc).__name__}: {exc}"[:400]
    wall = time.perf_counter() - started
    fallbacks = sorted({str(w.message).split("'")[1] for w in caught
                        if "fall back" in str(w.message) and "'" in str(w.message)})
    loss = timing.get("loss")
    finite = loss is not None and all(math.isfinite(x) for x in loss)
    # The first epoch includes warm-up and compilation; use the last one when there are several.
    train_s = timing["train"][-1] if timing["train"] else None
    val_s = timing["val"][-1] if timing["val"] else None
    ips = timing["images"] / train_s if train_s and timing["images"] else None
    full_epoch_s = (FULL_TRAIN_IMAGES / ips + (val_s or 0)) if ips else None
    return {
        "device": device, "batch": batch, "imgsz": imgsz, "fraction": fraction, "epochs": epochs,
        "train_images_per_epoch": timing["images"], "error": error,
        "losses_last_epoch": loss, "losses_finite": finite,
        "train_seconds_last_epoch": train_s, "val_seconds_last_epoch": val_s,
        "train_images_per_second": ips, "peak_memory_gib": timing["peak_mem"] or None,
        "cpu_fallback_ops": fallbacks, "wall_seconds": wall,
        "extrapolated_30_epochs_minutes": full_epoch_s * 30 / 60 if full_epoch_s else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="Folder with images/{train,val} and labels/{train,val}.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--batches", type=int, nargs="+", default=[8, 16])
    p.add_argument("--fraction", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=960)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    yaml = args.out / "data.yaml"
    yaml.write_text(f"path: {json.dumps(str(args.data.resolve()))}\ntrain: images/train\n"
                    "val: images/val\ntest: null\nnames:\n  0: \"sports ball\"\n")
    env = {
        "python": platform.python_version(), "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(), "machine": platform.platform(),
    }
    print(json.dumps(env))
    results = []
    for batch in args.batches:
        print(f"\n==> device {args.device}, batch {batch}, {args.epochs} epoch(s) on {args.fraction:.0%} of train")
        result = run(yaml, args.device, batch, args.fraction, args.epochs, args.imgsz, args.out)
        results.append(result)
        print(json.dumps(result))
        if args.device == "mps":
            torch.mps.empty_cache()
    report = {"environment": env, "reference": REFERENCE, "results": results}
    (args.out / "report.json").write_text(json.dumps(report, indent=1))

    def fmt(value, spec):
        return "-" if value is None else format(value, spec)

    lines = [
        "| batch | runs | losses finite | train img/s | val s | peak GiB | CPU-fallback ops | ≈ 30 epochs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['batch']} | {'no: ' + r['error'] if r['error'] else 'yes'} | {r['losses_finite']} "
            f"| {fmt(r['train_images_per_second'], '.1f')} | {fmt(r['val_seconds_last_epoch'], '.0f')} "
            f"| {fmt(r['peak_memory_gib'], '.1f')} | {', '.join(r['cpu_fallback_ops']) or 'none'} "
            f"| {fmt(r['extrapolated_30_epochs_minutes'], '.0f')} min |"
        )
    table = "\n".join(lines)
    (args.out / "table.md").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
