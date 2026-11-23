#!/bin/bash
# Fork-only helper: Core ML parity of box2boxcam v1.0 on a Mac, in one command.
#
#   scripts/mac_parity_run.sh
#
# Creates parity-run/.venv with uv's Python 3.12 (the export's versions: torch 2.7.1,
# coremltools 9.0, numpy <= 2.3.5), downloads v1.0 (.pt and .mlpackage) and the public
# test split, compares the two with evaluate_detector.py --agreement, saves everything in
# parity-run/out/, then opens a prefilled GitHub issue with the report in the browser:
# press "Submit new issue" to send it. Safe to rerun: finished steps are reused.
set -euo pipefail

cd "$(dirname "$0")/.."
work=parity-run
venv="$work/.venv"
out="$work/out"
issue_repo="rileyw/box2boxcam-football-vision"

step=0
say() { step=$((step + 1)); printf '\n==> %d/5 %s\n' "$step" "$1"; }

[ "$(uname -s)" = Darwin ] || { echo "Core ML inference needs macOS." >&2; exit 1; }
command -v uv >/dev/null || {
    echo "uv is missing. Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
}
mkdir -p "$out"

say "Python 3.12 environment"
uv python install 3.12
[ -x "$venv/bin/python" ] || uv venv --python 3.12 "$venv"
py="$venv/bin/python"

say "Installing torch 2.7.1, coremltools 9.0 and the project"
uv pip install --python "$py" "torch==2.7.1" "torchvision==0.22.1" "coremltools==9.0" \
    "numpy>=1.26,<=2.3.5" -e '.[benchmark]'

say "Downloading box2boxcam v1.0 and the public test split"
"$py" - "$work" <<'PY'
import sys
from huggingface_hub import snapshot_download

work = sys.argv[1]
snapshot_download("Totorotibo84/box2boxcam-football-ball-detection", local_dir=f"{work}/model",
                  allow_patterns=["box2boxcam-ball-v1.0.pt", "box2boxcam-ball-v1.0.mlpackage/*"])
snapshot_download("Totorotibo84/box2boxcam-football-ball-detection-dataset", repo_type="dataset",
                  local_dir=f"{work}/dataset", allow_patterns=["images/test/*", "labels/test/*"])
PY

say "Comparing the Core ML export with the PyTorch model (a few minutes)"
"$py" scripts/evaluate_detector.py \
    --weights "$work/model/box2boxcam-ball-v1.0.mlpackage" \
    --compare "$work/model/box2boxcam-ball-v1.0.pt" \
    --frames-dir "$work/dataset/images/test" --labels "$work/dataset/labels/test" \
    --conf 0.1 --device cpu --agreement --out-json "$out/parity_conf010.json" \
    2>&1 | tee "$out/parity_conf010.txt"

say "Writing the report and opening the GitHub issue"
{
    echo "## Core ML parity: box2boxcam v1.0, .mlpackage vs .pt"
    echo
    echo "- Machine: $(sysctl -n machdep.cpu.brand_string), $(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
    echo "- Commit: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
    echo "- Packages: $("$py" -c 'import torch, coremltools, numpy, ultralytics; print(f"torch {torch.__version__}, coremltools {coremltools.__version__}, numpy {numpy.__version__}, ultralytics {ultralytics.__version__}")')"
    echo "- Command: evaluate_detector.py --weights .mlpackage --compare .pt --conf 0.1 --device cpu --agreement"
    echo
    echo '```'
    # The tables and bootstrap, without download/progress noise.
    sed -n '/frames, .* balls/,$p' "$out/parity_conf010.txt"
    echo '```'
} > "$out/report.md"

# Prefilled "new issue" URL; python -c, since /bin/bash 3.2 mishandles heredocs inside $( ).
url="$("$py" -c '
import sys, urllib.parse
repo, path = sys.argv[1], sys.argv[2]
body = open(path).read()
if len(body) > 7000:  # keep the URL within browser limits
    body = body[:7000] + "\n```\n(truncated; full output in parity-run/out/parity_conf010.txt)"
print(f"https://github.com/{repo}/issues/new?" + urllib.parse.urlencode({"title": "Core ML parity run (Mac)", "body": body}))
' "$issue_repo" "$out/report.md")"
open "$url"
echo
echo "Report saved in $out/report.md."
echo "The browser now shows a prefilled issue on $issue_repo: press \"Submit new issue\"."
