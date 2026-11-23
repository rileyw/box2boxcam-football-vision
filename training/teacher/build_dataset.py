"""Assemble a training dataset: public train + pseudo-labelled matches; and reserved tests.

Training images are the public train split plus, for every match with role "train",
the auto-accepted frames (labels/) and the human-reviewed ones (labels_reviewed/, which
win over auto labels). Val stays the public val split, so checkpoint selection is
comparable to the baseline. Matches with role "reserved" never enter training: their
reviewed frames become data/reserved/<match>/ as a new held-out test.

Files are symlinked, not copied.

    python teacher/build_dataset.py --name public_plus_club --matches club_20261130 club_20261207
    python train_local.py --data data/public_plus_club.yaml --seed 42 --name plus_club_s42
"""

import argparse
import csv
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def match_frames(match_root):
    """{item_id: (image, label)} with reviewed labels over auto, plus the match role."""
    rows = list(csv.DictReader((match_root / "manifest.csv").open()))
    roles = {r["role"] for r in rows}
    if len(roles) != 1:
        raise SystemExit(f"{match_root.name}: mixed roles {roles}")
    out = {}
    for sub in ("labels", "labels_reviewed"):  # reviewed second, so it wins
        for label in sorted((match_root / sub).glob("*.txt")):
            out[label.stem] = (match_root / "frames" / f"{label.stem}.jpg", label)
    return out, roles.pop()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--matches", nargs="*", default=[])
    p.add_argument("--pseudo", type=Path, default=HERE / "pseudo")
    p.add_argument("--public", type=Path, default=HERE / "data/public")
    args = p.parse_args()

    out = HERE / "data" / args.name
    summary = {"public_train": 0, "matches": {}}
    for image in sorted((args.public / "images/train").glob("*.jpg")):
        link(image, out / "images/train" / image.name)
        label = args.public / "labels/train" / f"{image.stem}.txt"
        if label.exists():
            link(label, out / "labels/train" / label.name)
        summary["public_train"] += 1
    for split in ("val",):
        for image in sorted((args.public / f"images/{split}").glob("*.jpg")):
            link(image, out / f"images/{split}" / image.name)
            label = args.public / f"labels/{split}" / f"{image.stem}.txt"
            if label.exists():
                link(label, out / f"labels/{split}" / label.name)

    for match in args.matches:
        frames, role = match_frames(args.pseudo / match)
        if role == "reserved":
            reserved = HERE / "data/reserved" / match
            for item, (image, label) in frames.items():
                if label.parent.name != "labels_reviewed":
                    raise SystemExit(f"{match}: reserved frames must be human-reviewed ({item})")
                link(image, reserved / "images" / image.name)
                link(label, reserved / "labels" / label.name)
            summary["matches"][match] = {"role": role, "reserved_frames": len(frames)}
            continue
        auto = sum(1 for _, label in frames.values() if label.parent.name == "labels")
        for item, (image, label) in frames.items():
            link(image, out / "images/train" / image.name)
            link(label, out / "labels/train" / f"{item}.txt")
        summary["matches"][match] = {"role": role, "train_frames": len(frames),
                                     "auto": auto, "reviewed": len(frames) - auto}

    yaml = HERE / "data" / f"{args.name}.yaml"
    yaml.write_text(f"path: {json.dumps(str(out))}\ntrain: images/train\nval: images/val\n"
                    "test: null\nnames:\n  0: \"sports ball\"\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    print("dataset yaml:", yaml)


if __name__ == "__main__":
    main()
