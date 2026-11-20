#!/bin/bash
# Step 4: CAMB, as a pip wheel into the env.
#
#   source env.sh && bash install/04_camb.sh          # login node: it downloads
#
# CAMB $CAMB_VERSION (env.sh; the laptop's) has a manylinux_2_28 wheel on PyPI with its
# Fortran library inside, so nothing is compiled. --only-binary makes pip fail instead of
# quietly building another CAMB from source with whatever gfortran is on PATH. numpy and scipy
# stay where the environment put them (--no-deps; sympy and packaging are named explicitly).
#
# Cobaya's camb theory uses the first CAMB it finds: packages_path/code/CAMB if that exists,
# else the importable module. Nothing here creates that directory, and install/check.py fails
# if it appears (a `cobaya-install camb` would make it).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/env.sh"
[ "${CONDA_PREFIX:-}" = "$CLIENT_ENV" ] || { echo "FATAL: env $CLIENT_ENV is not active" >&2; exit 1; }
# The versions are asked of the modules themselves: `pip freeze` lists conda-installed
# packages as "numpy @ file://...", so a grep for "numpy==" finds nothing (and, under
# `set -o pipefail`, stopped the first version of this script without a word).
versions() { python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"; }
before=$(versions)
python -m pip install --only-binary=camb --no-deps "camb==$CAMB_VERSION"
python -m pip install "sympy>=1.0" "packaging>=24.2"
after=$(versions)
[ "$before" = "$after" ] || { echo "FATAL: numpy/scipy changed: $before -> $after" >&2; exit 1; }
echo "camb $CAMB_VERSION installed; $after unchanged"
mkdir -p "$HERE/logs"
python -m pip freeze > "$HERE/logs/pip_freeze.txt"
cd "$HERE"
[ -z "${CLIENT_RUN_CHECK:-}" ] || python install/check.py --stage camb
