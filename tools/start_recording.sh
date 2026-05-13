#!/usr/bin/env bash
# Launcher for the Jetson DMS recorder.
# Tries the dms-infer conda env first, then falls back to the legacy mp env.
# On any failure, keeps the terminal window open so the error is readable
# when launched by double-clicking.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

pause_on_exit() {
    local code=$?
    if [ $code -ne 0 ]; then
        echo ""
        echo "----------------------------------------"
        echo "Launcher exited with code $code"
        read -n 1 -s -r -p "Press any key to close..."
        echo ""
    fi
    exit $code
}
trap pause_on_exit EXIT

# shellcheck disable=SC1091
if ! source "$HOME/miniforge3/bin/activate"; then
    echo "ERROR: could not source miniforge3 at $HOME/miniforge3/bin/activate"
    exit 1
fi

if conda activate dms-infer 2>/dev/null; then
    echo "[launcher] activated env: dms-infer"
elif conda activate mp 2>/dev/null; then
    echo "[launcher] activated env: mp (legacy fallback)"
else
    echo "ERROR: neither 'dms-infer' nor 'mp' conda env exists."
    echo "Create one with:"
    echo "    conda env create -f $REPO_DIR/inference/environment.yml"
    exit 1
fi

cd "$REPO_DIR"
python tools/record.py "$@"
