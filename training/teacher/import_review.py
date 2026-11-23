"""Turn a Label Studio JSON export of one match's review into YOLO labels.

Only tasks with a submitted human annotation count; teacher pre-annotations alone are
never accepted. A frame marked "Exclure image" is dropped. A submitted annotation with no
box is an explicit negative (empty label file). If a frame has several annotations, the
most recently updated one is used.

    python teacher/import_review.py --match club_20261130 --export export.json
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def boxes_from(annotation):
    """[(x0, y0, x1, y1) normalized], excluded flag."""
    boxes, excluded = [], False
    for result in annotation.get("result", []):
        if result.get("type") == "choices":
            excluded |= "Exclure image" in result.get("value", {}).get("choices", [])
        if result.get("type") != "rectangle":
            continue
        v = result["value"]
        if float(v.get("rotation", 0)) != 0:
            raise ValueError("Rotated boxes are not supported")
        x0, y0 = float(v["x"]) / 100, float(v["y"]) / 100
        x1, y1 = x0 + float(v["width"]) / 100, y0 + float(v["height"]) / 100
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1))
    return boxes, excluded


def import_review(tasks, match):
    """{item_id: [boxes] | None (excluded)} for reviewed tasks of this match."""
    reviewed = {}
    for task in tasks:
        data = task.get("data", {})
        if data.get("match") != match:
            continue
        annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not annotations:
            continue
        latest = max(annotations, key=lambda a: a.get("updated_at") or a.get("created_at") or "")
        boxes, excluded = boxes_from(latest)
        reviewed[data["item_id"]] = None if excluded else boxes
    return reviewed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--match", required=True)
    p.add_argument("--export", type=Path, required=True, help="Label Studio JSON export.")
    p.add_argument("--out", type=Path, default=HERE / "pseudo")
    args = p.parse_args()

    reviewed = import_review(json.loads(args.export.read_text()), args.match)
    target = args.out / args.match / "labels_reviewed"
    target.mkdir(parents=True, exist_ok=True)
    for old in target.glob("*.txt"):
        old.unlink()
    kept = excluded = negatives = 0
    for item, boxes in sorted(reviewed.items()):
        if boxes is None:
            excluded += 1
            continue
        negatives += not boxes
        kept += 1
        (target / f"{item}.txt").write_text("".join(
            f"0 {(a + c) / 2:.8f} {(b + d) / 2:.8f} {c - a:.8f} {d - b:.8f}\n" for a, b, c, d in boxes))
    summary = {"match": args.match, "reviewed": len(reviewed), "kept": kept,
               "empty_kept": negatives, "excluded": excluded}
    (args.out / args.match / "review" / "imported.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
