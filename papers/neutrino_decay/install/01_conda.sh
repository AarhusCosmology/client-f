#!/bin/bash
# Step 1: Miniforge into $CONDA_ROOT, unless it is already there.
#
# A private Miniforge rather than a site module because the env pins python 3.10 and
# TensorFlow 2.20, and a module that is upgraded under us would silently change the
# environment the surrogate was trained in. We never run `conda init`: env.sh
# activates explicitly, so your .bashrc and the other rounds are untouched.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/env.sh"

if [ -x "$CONDA_ROOT/bin/conda" ]; then
  echo "conda already present: $("$CONDA_ROOT/bin/conda" --version) at $CONDA_ROOT"
  exit 0
fi

installer=$(mktemp "${TMPDIR:-/tmp}/miniforge_XXXXXX.sh")
trap 'rm -f "$installer"' EXIT
curl -fsSL -o "$installer" \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname -s | sed s/Darwin/MacOSX/)-$(uname -m).sh"
bash "$installer" -b -p "$CONDA_ROOT"
"$CONDA_ROOT/bin/conda" --version
echo "Miniforge installed at $CONDA_ROOT (no shell init was changed)."
