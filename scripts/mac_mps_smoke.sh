#!/bin/bash
# Fork-only: can the Mac train the ball model on its GPU (MPS), and how fast? One command:
#
#   scripts/mac_mps_smoke.sh
#
# Needs a finished scripts/mac_parity_run.sh (its parity-run/.venv: torch 2.7.1,
# ultralytics 8.4.153). Downloads the public train and val splits (about 0.7 GB; the test
# split is never used here), then trains the v1.0 recipe for 2 short epochs on 25 % of
# train at batch 8 and 16 on MPS (scripts/mps_smoke_train.py). It reports whether training
# runs and stays finite, images per second, memory, ops that fell back to the CPU, and an
# estimate for a full 30-epoch run. Opens a prefilled GitHub issue. About 10-20 minutes.
# Close other heavy apps first: the GPU and memory are shared.
set -euo pipefail

cd "$(dirname "$0")/.."
work=parity-run
py="$work/.venv/bin/python"
out="$work/mps-smoke"
issue_repo="rileyw/box2boxcam-football-vision"

[ "$(uname -s)" = Darwin ] || { echo "This is the Mac MPS test." >&2; exit 1; }
[ -x "$py" ] || { echo "Run scripts/mac_parity_run.sh first (it sets up $work/.venv)." >&2; exit 1; }
mkdir -p "$out"

printf '\n==> 1/3 Downloading the public train and val splits\n'
"$py" - "$work" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("Totorotibo84/box2boxcam-football-ball-detection-dataset", repo_type="dataset",
                  local_dir=f"{sys.argv[1]}/dataset",
                  allow_patterns=["images/train/*", "labels/train/*", "images/val/*", "labels/val/*"])
PY

printf '\n==> 2/3 Training on MPS (batch 8, then 16)\n'
PYTORCH_ENABLE_MPS_FALLBACK=1 "$py" scripts/mps_smoke_train.py --data "$work/dataset" --out "$out" \
    --device mps --batches 8 16 --fraction 0.25 --epochs 2 2>&1 | tee "$out/log.txt" \
    | grep -E "^==>|^\{|^\||Error|Traceback" || true
[ -s "$out/table.md" ] || { echo "No report written, see $out/log.txt" >&2; exit 1; }

printf '\n==> 3/3 Report and GitHub issue\n'
{
    echo "## Mac MPS training smoke test: box2boxcam v1.0 recipe"
    echo
    echo "2 epochs on 25 % of the public train split, 960 px, COCO YOLO26s with sports-ball transfer."
    echo "Speed from the last epoch; the 30-epoch estimate covers the full train split plus validation."
    echo "Reference: $("$py" -c 'import json,sys;print(json.load(open(sys.argv[1]))["reference"])' "$out/report.json")"
    echo
    cat "$out/table.md"
    echo
    echo "- Machine: $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, $(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
    echo "- Packages: $("$py" -c 'import torch, ultralytics; print(f"torch {torch.__version__}, ultralytics {ultralytics.__version__}")')"
    echo "- Commit: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
} > "$out/report.md"
cat "$out/report.md"

url="$("$py" -c '
import sys, urllib.parse
repo, path = sys.argv[1], sys.argv[2]
body = open(path).read()[:7000]
print(f"https://github.com/{repo}/issues/new?" + urllib.parse.urlencode({"title": "Mac MPS training smoke test", "body": body}))
' "$issue_repo" "$out/report.md")"
open "$url"
echo
echo "The browser now shows a prefilled issue on $issue_repo: press \"Submit new issue\"."
