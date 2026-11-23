#!/bin/bash
# Fork-only: is it Apple's NMS stage or the converted network? One command:
#
#   scripts/mac_nms_experiment.sh
#
# Needs finished scripts/mac_parity_run.sh and scripts/mac_fp32_experiment.sh runs.
# Exports v1.0 to Core ML without the NMS pipeline (raw (1, 5, 18900) output, same
# one-to-many head; nms left unset since nms=False would switch to the end2end head), so
# Ultralytics applies its own NMS in Python:
#
#   R5 raw fp32  CPU_ONLY   vs R4 (fp32 with Apple NMS): same network, different NMS
#   R6 raw fp16  CPU_ONLY
#   D1 raw graph diff: per-anchor scores of raw fp32 Core ML vs the PyTorch graph
#
# If R5 comes back near the .pt, Apple's NMS stage is the cause; if not, the conversion
# is, and D1 shows by how much. Opens a prefilled GitHub issue. About 6 minutes.
set -euo pipefail

cd "$(dirname "$0")/.."
work=parity-run
py="$work/.venv/bin/python"
exp="$work/fp32"
issue_repo="rileyw/box2boxcam-football-vision"
pt="$work/model/box2boxcam-ball-v1.0.pt"
frames="$work/dataset/images/test"
labels="$work/dataset/labels/test"

[ "$(uname -s)" = Darwin ] || { echo "Core ML inference needs macOS." >&2; exit 1; }
[ -x "$py" ] && [ -f "$pt" ] && [ -d "$frames" ] && [ -s "$exp/R4.json" ] || {
    echo "Run scripts/mac_parity_run.sh and scripts/mac_fp32_experiment.sh first." >&2
    exit 1
}

printf '\n==> 1/4 Exporting v1.0 to Core ML without NMS (fp32 and fp16)\n'
for q in 32 16; do
    dir="$exp/export-raw-fp$q"
    model="$dir/box2boxcam-ball-v1.0-raw-fp$q.mlpackage"
    if [ ! -d "$model" ]; then
        mkdir -p "$dir"
        cp "$pt" "$dir/box2boxcam-ball-v1.0-raw-fp$q.pt"
        "$py" -c '
import sys
from ultralytics import YOLO
YOLO(sys.argv[1]).export(format="coreml", imgsz=960, quantize=int(sys.argv[2]), device="cpu")
' "$dir/box2boxcam-ball-v1.0-raw-fp$q.pt" "$q"
    fi
    [ -d "$model" ] || { echo "Export raw fp$q failed." >&2; exit 1; }
done

printf '\n==> 2/4 Comparing the raw exports with the .pt (Ultralytics NMS in Python)\n'
run() {  # run NAME UNITS MODEL
    echo
    echo "--- $1: $(basename "$3") on $2"
    rm -f "$exp/$1.json"
    COREML_UNITS="$2" "$py" scripts/coreml_units_eval.py \
        --weights "$3" --compare "$pt" --frames-dir "$frames" --labels "$labels" \
        --conf 0.1 --device cpu --agreement --out-json "$exp/$1.json" \
        2>&1 | tee "$exp/$1.txt" | grep -E "forced|frames, |\| box2boxcam|ANEC|Error" || true
    [ -s "$exp/$1.json" ] || { echo "$1 failed, see $exp/$1.txt" >&2; exit 1; }
}
run R5 CPU_ONLY "$exp/export-raw-fp32/box2boxcam-ball-v1.0-raw-fp32.mlpackage"
run R6 CPU_ONLY "$exp/export-raw-fp16/box2boxcam-ball-v1.0-raw-fp16.mlpackage"

printf '\n==> 3/4 Raw graph diff: Core ML fp32 vs PyTorch, anchor by anchor\n'
"$py" scripts/coreml_raw_diff.py "$pt" \
    "$exp/export-raw-fp32/box2boxcam-ball-v1.0-raw-fp32.mlpackage" "$frames" \
    2>&1 | tee "$exp/D1.txt" | grep -vi warning || true

printf '\n==> 4/4 Summary and GitHub issue\n'
"$py" - "$exp" <<'PY' > "$exp/nms_summary.md"
import json, sys
from pathlib import Path

exp = Path(sys.argv[1])
runs = [
    ("R4", "fp32 + Apple NMS pipeline", "CPU_ONLY"),
    ("R5", "raw fp32, Ultralytics NMS", "CPU_ONLY"),
    ("R6", "raw fp16, Ultralytics NMS", "CPU_ONLY"),
]


def cell(row):
    return f"{row['tp']}/{row['fp']} ({row['small_tp']})"


print("## Core ML NMS vs conversion experiment: box2boxcam v1.0")
print()
print("TP/FP (small TP) on the public test split; the bootstrap columns are Core ML minus .pt.")
print()
print("| run | model | compute units | 0.25 | 0.35 | 0.45 | 0.55 "
      "| FP@0.25 diff [95 % CI] | TP@0.25 diff [95 % CI] | mean IoU / mean abs conf diff @0.25 |")
print("|---|---|---|---|---|---|---|---|---|---|")
pt_row = None
for name, model, units in runs:
    path = exp / f"{name}.json"
    if not path.exists():
        continue
    report = json.loads(path.read_text())
    a, b = report["bootstrap"]["A"], report["bootstrap"]["B"]
    rows = report["models"][a]["thresholds"]
    pt_row = report["models"][b]["thresholds"]
    stats = report["bootstrap"]["stats"]

    def diff(key):
        s = stats[key]
        return f"{s['observed']:+g} [{s['low']:+g}, {s['high']:+g}]"

    agree = report["agreement"]["rows"][0]
    iou = "-" if agree["mean_iou"] is None else f"{agree['mean_iou']:.3f}"
    dconf = "-" if agree["mean_abs_conf_diff"] is None else f"{agree['mean_abs_conf_diff']:.3f}"
    print(f"| {name} | {model} | {units} | " + " | ".join(cell(r) for r in rows)
          + f" | {diff('FP@0.25')} | {diff('TP@0.25')} | {iou} / {dconf} |")
print("| .pt | PyTorch | CPU | " + " | ".join(cell(r) for r in pt_row) + " | | | |")
print()
print("```")
print((exp / "D1.txt").read_text().strip())
print("```")
PY
{
    cat "$exp/nms_summary.md"
    echo
    echo "- Machine: $(sysctl -n machdep.cpu.brand_string), $(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
    echo "- Commit: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
} > "$exp/nms_report.md"
cat "$exp/nms_report.md"

url="$("$py" -c '
import sys, urllib.parse
repo, path = sys.argv[1], sys.argv[2]
body = open(path).read()[:7000]
print(f"https://github.com/{repo}/issues/new?" + urllib.parse.urlencode({"title": "Core ML NMS vs conversion experiment (Mac)", "body": body}))
' "$issue_repo" "$exp/nms_report.md")"
open "$url"
echo
echo "The browser now shows a prefilled issue on $issue_repo: press \"Submit new issue\"."
