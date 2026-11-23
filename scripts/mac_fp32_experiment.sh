#!/bin/bash
# Fork-only: is fp16 the cause of the Core ML export's extra false positives? One command:
#
#   scripts/mac_fp32_experiment.sh
#
# Needs a finished scripts/mac_parity_run.sh (its parity-run/.venv, model and test split).
# Re-exports v1.0 to Core ML in fp32 and, as a control, in fp16 (same settings as the
# published export), then compares each against the .pt on the public test split:
#
#   R1 published fp16  CPU_AND_NE  (Ultralytics' default, as in the first parity run)
#   R2 published fp16  CPU_ONLY
#   R3 re-export fp16  CPU_ONLY    (control: should match R2)
#   R4 re-export fp32  CPU_ONLY    (if this matches the .pt, fp16 is the cause)
#
# Results land in parity-run/fp32/, and a prefilled GitHub issue opens in the browser:
# press "Submit new issue". About 10 minutes.
set -euo pipefail

cd "$(dirname "$0")/.."
work=parity-run
py="$work/.venv/bin/python"
exp="$work/fp32"
issue_repo="rileyw/box2boxcam-football-vision"
pt="$work/model/box2boxcam-ball-v1.0.pt"
published="$work/model/box2boxcam-ball-v1.0.mlpackage"
frames="$work/dataset/images/test"
labels="$work/dataset/labels/test"

[ "$(uname -s)" = Darwin ] || { echo "Core ML inference needs macOS." >&2; exit 1; }
[ -x "$py" ] && [ -f "$pt" ] && [ -d "$published" ] && [ -d "$frames" ] || {
    echo "Run scripts/mac_parity_run.sh first (it sets up $work)." >&2
    exit 1
}
mkdir -p "$exp"

printf '\n==> 1/3 Exporting v1.0 to Core ML in fp32 and fp16\n'
for q in 32 16; do
    dir="$exp/export-fp$q"
    model="$dir/box2boxcam-ball-v1.0-fp$q.mlpackage"
    if [ ! -d "$model" ]; then
        mkdir -p "$dir"
        cp "$pt" "$dir/box2boxcam-ball-v1.0-fp$q.pt"
        "$py" -c '
import sys
from ultralytics import YOLO
YOLO(sys.argv[1]).export(format="coreml", imgsz=960, nms=True, conf=0.1, iou=0.7,
                         quantize=int(sys.argv[2]), device="cpu")
' "$dir/box2boxcam-ball-v1.0-fp$q.pt" "$q"
    fi
    [ -d "$model" ] || { echo "Export fp$q failed." >&2; exit 1; }
done

printf '\n==> 2/3 Comparing four Core ML variants with the .pt (about 2 minutes each)\n'
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
run R1 CPU_AND_NE "$published"
run R2 CPU_ONLY "$published"
run R3 CPU_ONLY "$exp/export-fp16/box2boxcam-ball-v1.0-fp16.mlpackage"
run R4 CPU_ONLY "$exp/export-fp32/box2boxcam-ball-v1.0-fp32.mlpackage"

printf '\n==> 3/3 Summary and GitHub issue\n'
"$py" - "$exp" <<'PY' > "$exp/summary.md"
import json, sys
from pathlib import Path

exp = Path(sys.argv[1])
runs = [
    ("R1", "published fp16", "CPU_AND_NE"),
    ("R2", "published fp16", "CPU_ONLY"),
    ("R3", "re-export fp16", "CPU_ONLY"),
    ("R4", "re-export fp32", "CPU_ONLY"),
]


def cell(row):
    return f"{row['tp']}/{row['fp']} ({row['small_tp']})"


print("## Core ML fp32 vs fp16 experiment: box2boxcam v1.0")
print()
print("TP/FP (small TP) on the public test split; the bootstrap columns are Core ML minus .pt.")
print()
print("| run | model | compute units | 0.25 | 0.35 | 0.45 | 0.55 "
      "| FP@0.25 diff [95 % CI] | TP@0.25 diff [95 % CI] | mean IoU / mean abs conf diff @0.25 |")
print("|---|---|---|---|---|---|---|---|---|---|")
pt_row = None
for name, model, units in runs:
    report = json.loads((exp / f"{name}.json").read_text())
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
          + f" | {diff('FP@0.25')} | {diff('TP@0.25')} "
          + f"| {iou} / {dconf} |")
print("| .pt | PyTorch | CPU | " + " | ".join(cell(r) for r in pt_row) + " | | | |")
PY
{
    cat "$exp/summary.md"
    echo
    echo "- Machine: $(sysctl -n machdep.cpu.brand_string), $(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
    echo "- Commit: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
    echo "- Packages: $("$py" -c 'import torch, coremltools, numpy, ultralytics; print(f"torch {torch.__version__}, coremltools {coremltools.__version__}, numpy {numpy.__version__}, ultralytics {ultralytics.__version__}")')"
    if grep -q "ANECCompile" "$exp"/R*.txt; then
        echo "- Neural Engine compile failures in: $(grep -l "ANECCompile" "$exp"/R*.txt | xargs -n1 basename | tr '\n' ' ')"
    fi
} > "$exp/report.md"
cat "$exp/report.md"

# Prefilled "new issue" URL; python -c, since /bin/bash 3.2 mishandles heredocs inside $( ).
url="$("$py" -c '
import sys, urllib.parse
repo, path = sys.argv[1], sys.argv[2]
body = open(path).read()[:7000]
print(f"https://github.com/{repo}/issues/new?" + urllib.parse.urlencode({"title": "Core ML fp32 vs fp16 experiment (Mac)", "body": body}))
' "$issue_repo" "$exp/report.md")"
open "$url"
echo
echo "The browser now shows a prefilled issue on $issue_repo: press \"Submit new issue\"."
