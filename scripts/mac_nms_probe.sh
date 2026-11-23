#!/bin/bash
# Fork-only: does Apple's NMS stage honour iouThreshold, and does a fix work? One command:
#
#   scripts/mac_nms_probe.sh
#
# Needs a finished scripts/mac_parity_run.sh. Calls the published v1.0 Core ML pipeline
# directly (scripts/coreml_nms_probe.py) with default inputs, with iouThreshold passed
# explicitly (0.7, 0.45), and with a greedy NMS 0.7 applied after it, on CPU_ONLY and on
# CPU_AND_NE (the app runs on the Neural Engine). Opens a prefilled GitHub issue.
# About 5 minutes.
set -euo pipefail

cd "$(dirname "$0")/.."
work=parity-run
py="$work/.venv/bin/python"
out="$work/nms-probe"
issue_repo="rileyw/box2boxcam-football-vision"
model="$work/model/box2boxcam-ball-v1.0.mlpackage"

[ "$(uname -s)" = Darwin ] || { echo "Core ML inference needs macOS." >&2; exit 1; }
[ -x "$py" ] && [ -d "$model" ] || { echo "Run scripts/mac_parity_run.sh first." >&2; exit 1; }
mkdir -p "$out"

for units in CPU_ONLY CPU_AND_NE; do
    printf '\n==> Probing the published model on %s\n' "$units"
    "$py" scripts/coreml_nms_probe.py "$model" "$work/dataset/images/test" \
        "$work/dataset/labels/test" --units "$units" 2>&1 \
        | grep -v -i "warning\|Torch version\|has been tested" | tee "$out/$units.txt"
done

{
    echo "## Core ML NMS probe: box2boxcam v1.0 published export"
    echo
    echo "V0 = default inputs (as Ultralytics and the app call it); V1/V2 = iouThreshold passed explicitly;"
    echo "V3 = V0 then greedy NMS 0.7. Last column: returned box pairs still overlapping above IoU 0.7."
    echo "Reference, PyTorch export graph + NMS 0.7 on the same square frames: 97/79 at 0.25."
    for units in CPU_ONLY CPU_AND_NE; do echo; cat "$out/$units.txt"; done
    echo
    echo "- Machine: $(sysctl -n machdep.cpu.brand_string), $(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
    echo "- Commit: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
} > "$out/report.md"
cat "$out/report.md"

url="$("$py" -c '
import sys, urllib.parse
repo, path = sys.argv[1], sys.argv[2]
body = open(path).read()[:7000]
print(f"https://github.com/{repo}/issues/new?" + urllib.parse.urlencode({"title": "Core ML NMS probe (Mac)", "body": body}))
' "$issue_repo" "$out/report.md")"
open "$url"
echo
echo "The browser now shows a prefilled issue on $issue_repo: press \"Submit new issue\"."
