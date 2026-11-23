#!/bin/bash
# Start Label Studio for reviewing pseudo-labels: frames served from training/pseudo,
# data kept in training/.ls-data, no update check or error reporting. Open
# http://localhost:8080 in the Windows browser; the first visit asks you to sign up
# (a local account only).
#
#   teacher/label_studio.sh            # install on first run, then start
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv-ls/bin/label-studio ]; then
    uv venv -q --python 3.12 .venv-ls
    # label-studio 1.23.2 crashes at start with sentry-sdk 3 ("Unknown option 'send_default_pii'").
    uv pip install -q --python .venv-ls/bin/python label-studio "sentry-sdk<3"
fi
mkdir -p pseudo .ls-data
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$PWD/pseudo"
export LABEL_STUDIO_BASE_DATA_DIR="$PWD/.ls-data"
export LATEST_VERSION_CHECK=false COLLECT_ANALYTICS=false SENTRY_DSN= FRONTEND_SENTRY_DSN=
exec .venv-ls/bin/label-studio start --no-browser --host 127.0.0.1 --port 8080
