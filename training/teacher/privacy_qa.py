"""Check and fix the de-identification of a match's frames with a human sample.

head_blur.py is tuned for recall, but it misses people no detector finds. As with the
"report a problem" process of street-imagery services, a person checks a random sample
and marks anything still identifiable. Those regions are then blurred in place, and the
residual rate is reported.

    # 1. sample 100 frames into Label Studio tasks (import pseudo/<match>/privacy/tasks.json
    #    into a project using pseudo/<match>/privacy/label_config.xml)
    python teacher/privacy_qa.py sample --match club_20261130 --n 100
    # 2. after review, blur every marked region in the frames and report the residual rate
    python teacher/privacy_qa.py apply --match club_20261130 --export ~/Downloads/privacy-export.json

Mark every readable face, a recognisable back-turned head, and readable number plates.
Submit with no box when the frame is clean.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "teacher"))
from head_blur import blur_regions  # noqa: E402

LABEL_CONFIG = """<View>
  <Header value="Privacy check: mark anything still identifiable" />
  <Header value="Draw a box on every readable face, a recognisable head, or a readable number plate. Submit with no box when the frame is clean." size="4" />
  <Text name="frame_info" value="$frame_info" />
  <Image name="image" value="$image" maxWidth="1400px" zoom="true" zoomControl="true" />
  <RectangleLabels name="leak" toName="image">
    <Label value="Face or head" background="#e11d48" />
    <Label value="Number plate" background="#f59e0b" />
  </RectangleLabels>
</View>
"""


def sample(args):
    frames = sorted((args.pseudo / args.match / "frames").glob("*.jpg"))
    if not frames:
        raise SystemExit(f"No frames in {args.pseudo / args.match / 'frames'}")
    marker = args.pseudo / args.match / "frames" / "blur.json"
    if not marker.exists() or not json.loads(marker.read_text()).get("blurred"):
        print("warning: these frames were extracted without head blurring")
    chosen = sorted(random.Random(args.seed).sample(frames, min(args.n, len(frames))))
    out = args.pseudo / args.match / "privacy"
    out.mkdir(parents=True, exist_ok=True)
    tasks = [{"data": {"image": f"/data/local-files/?d={args.match}/frames/{f.name}",
                       "item_id": f.stem, "match": args.match, "privacy_qa": True,
                       "frame_info": f"{f.stem} · privacy check"}} for f in chosen]
    (out / "tasks.json").write_text(json.dumps(tasks, indent=1))
    (out / "label_config.xml").write_text(LABEL_CONFIG)
    print(f"{len(tasks)} frames sampled of {len(frames)}: import {out / 'tasks.json'} "
          f"into a project using {out / 'label_config.xml'}")


def apply(args):
    tasks = json.loads(args.export.read_text())
    frames_dir = args.pseudo / args.match / "frames"
    checked = leaking = marks = 0
    for task in tasks:
        data = task.get("data", {})
        if data.get("match") != args.match or not data.get("privacy_qa"):
            continue
        annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not annotations:
            continue
        checked += 1
        latest = max(annotations, key=lambda a: a.get("updated_at") or a.get("created_at") or "")
        path = frames_dir / f"{data['item_id']}.jpg"
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"cannot read {path}")
        h, w = image.shape[:2]
        boxes = []
        for r in latest.get("result", []):
            if r.get("type") not in ("rectanglelabels", "rectangle"):
                continue
            v = r["value"]
            x0, y0 = v["x"] / 100 * w, v["y"] / 100 * h
            boxes.append((x0, y0, x0 + v["width"] / 100 * w, y0 + v["height"] / 100 * h))
        if boxes:
            leaking += 1
            marks += len(boxes)
            blur_regions(image, boxes, margin=0.2)
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    summary = {"match": args.match, "frames_checked": checked, "frames_with_leaks": leaking,
               "regions_blurred_now": marks,
               "leak_rate": round(leaking / checked, 4) if checked else None}
    (args.pseudo / args.match / "privacy" / "qa_result.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    if checked and leaking / checked > 0.02:
        print("More than 2 % of checked frames leaked: sample more frames, and consider "
              "re-extracting with --blur-imgsz 1600 or --blur-conf 0.1.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--match", required=True)
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("apply")
    a.add_argument("--match", required=True)
    a.add_argument("--export", type=Path, required=True)
    for parser in (s, a):
        parser.add_argument("--pseudo", type=Path, default=HERE / "pseudo")
    args = p.parse_args()
    sample(args) if args.command == "sample" else apply(args)


if __name__ == "__main__":
    main()
