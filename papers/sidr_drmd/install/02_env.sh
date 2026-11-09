#!/bin/bash
# Step 2: the Python environment, from CLiENT's own environment.yaml.
#
# CLiENT pins python 3.10, numpy 2.0.2 and TensorFlow 2.20. Those pins are what the
# trained surrogate is tied to (a .keras file is loaded by the Keras that wrote it),
# so we start from that file rather than a hand-written list, and record the result
# in logs/ so the machine at home can be matched to it.
#
# On top of CLiENT's list: Keras pinned to the laptop's line, the build tools classy needs
# in step 3 (scikit-build-core and a cmake of its own), and mpi4py built from source against
# GRENDEL'S MPI MODULE -- the cluster's runtime, which knows its interconnect and Slurm.
# No MPI is ever installed from conda.
#
# (This script is the harness's, with three changes: Keras is pinned, cmake comes with the
# env so that step 3 does not depend on the system's, and the mpi4py version is read from
# env.sh.)
#
# ── CONDA-FORGE ONLY, AND WHY NOT `conda env create -f` ─────────────────────
# CLiENT's file lists the `defaults` channel: Anaconda's, under their terms of service
# and a commercial licence for organisations this size. `conda env create` cannot be
# told to ignore channels from ~/.condarc, and a user-level `channels: [defaults]` there
# silently put Anaconda packages into the laptop rehearsal (scipy 1.15.3 exists only
# there). So this script reads environment.yaml itself and runs
# `conda create --override-channels -c conda-forge`, which ignores every .condarc, then
# fails the install if any conda package came from anywhere else.
#
# conda-forge substitutions (checked 2026-09-15 on Grendel, and again 2026-10-07 by a
# dry-run solve for linux-64 with glibc 2.28: the list below solves, scipy 1.15.3 does not):
#   scipy 1.15.3 -> 1.15.2   conda-forge's last python-3.10 build; a patch release
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/env.sh"
[ -x "$CONDA_ROOT/bin/conda" ] || { echo "FATAL: no conda at $CONDA_ROOT -- run step 1" >&2; exit 1; }
[ -f "$CLIENT_REPO/environment.yaml" ] || { echo "FATAL: no CLiENT checkout at $CLIENT_REPO -- run sync_to_hpc.sh on the laptop" >&2; exit 1; }
. "$CONDA_ROOT/etc/profile.d/conda.sh"

if [ -e "$CLIENT_ENV/conda-meta/history" ]; then
  echo "FATAL: $CLIENT_ENV already exists. Remove it deliberately (conda env remove -p $CLIENT_ENV) and re-run." >&2
  exit 1
elif [ -d "$CLIENT_ENV" ]; then
  echo "removing $CLIENT_ENV: a prefix without conda-meta/history, left by an attempt that never finished"
  rm -rf "$CLIENT_ENV"
fi

# CLiENT's pins -> conda specs and pip specs. TF_EXTRA (env.sh) replaces the extra on
# tensorflow; anything unexpected in the file's layout stops here, not halfway.
parsed=$("$CONDA_ROOT/bin/python" - "$CLIENT_REPO/environment.yaml" "$TF_EXTRA" <<'PY'
import sys
from ruamel.yaml import YAML

path, tf_extra = sys.argv[1], sys.argv[2]
SUBSTITUTE = {"scipy=1.15.3": "scipy=1.15.2"}

deps = YAML(typ="safe").load(open(path))["dependencies"]
conda = [SUBSTITUTE.get(d, d) for d in deps if isinstance(d, str)]
pip = [p for d in deps if isinstance(d, dict) for p in d.get("pip", [])]
tf = [p for p in pip if p.split("==")[0].split("[")[0] == "tensorflow"]
if len(tf) != 1 or "==" not in tf[0] or not any(c.startswith("python=") for c in conda):
    sys.exit(f"unexpected environment.yaml layout: conda={conda} pip={pip}")
version = tf[0].split("==")[1]
pip = [(f"tensorflow[{tf_extra}]=={version}" if tf_extra else f"tensorflow=={version}")
       if p == tf[0] else p for p in pip]
print("CONDA", *conda)
print("PIP", *pip)
PY
) || { echo "FATAL: could not read $CLIENT_REPO/environment.yaml" >&2; exit 1; }
# read -a splits on spaces without glob expansion ("tensorflow[and-cuda]" stays literal).
read -ra CONDA_SPECS <<< "$(sed -n 's/^CONDA //p' <<< "$parsed")"
read -ra PIP_SPECS <<< "$(sed -n 's/^PIP //p' <<< "$parsed")"
echo "conda-forge: ${CONDA_SPECS[*]}"
echo "pip:         ${PIP_SPECS[*]}"

conda create -y -p "$CLIENT_ENV" --override-channels -c conda-forge "${CONDA_SPECS[@]}"
conda activate "$CLIENT_ENV"
case "$(command -v python)" in
  "$CLIENT_ENV"/*) ;;
  *) echo "FATAL: python is $(command -v python), not the env's (a site anaconda/python module still loaded?)" >&2; exit 1 ;;
esac

NUMPY_PIN=$(python -c "import numpy; print(numpy.__version__)")
# CLiENT's pip pins with Keras held at the laptop's version, then classy's build tools; keep
# numpy where CLiENT put it.
pip install "${PIP_SPECS[@]}" "keras==$KERAS_VERSION" "numpy==$NUMPY_PIN"
pip install "scikit-build-core>=0.11" "cmake>=3.21,<4" "numpy==$NUMPY_PIN"

# ── mpi4py against the module MPI ───────────────────────────────────────────
command -v module >/dev/null 2>&1 || { echo "FATAL: no module system: cannot load the cluster MPI" >&2; exit 1; }
module load "$GCC_MODULE" && module load "$MPI_MODULE" \
  || { echo "FATAL: module load $GCC_MODULE $MPI_MODULE failed (see logs/probe_*.txt for the names)" >&2; exit 1; }
MPICC=$(command -v mpicc) || { echo "FATAL: no mpicc after loading $MPI_MODULE" >&2; exit 1; }
case "$MPICC" in "$CLIENT_ENV"/*|"$CONDA_ROOT"/*) echo "FATAL: mpicc is conda's ($MPICC), not the module's" >&2; exit 1 ;; esac
if conda list -p "$CLIENT_ENV" | grep -qE '^(openmpi|mpich|impi_rt|mpi4py)[[:space:]]'; then
  echo "FATAL: a conda MPI package is in the env; it would shadow the module runtime" >&2; exit 1
fi
echo "building mpi4py against $MPICC ($("$MPICC" -showme:version 2>/dev/null || "$MPICC" -v 2>&1 | head -1))"
env MPICC="$MPICC" pip install --no-binary=mpi4py --no-cache-dir "mpi4py==$MPI4PY_VERSION"

# ── the channel guard: every conda package from conda-forge ──────────────────
mkdir -p "$HERE/logs"
conda list -p "$CLIENT_ENV" --explicit > "$HERE/logs/conda_explicit.txt"
foreign=$(grep -E '^https?://' "$HERE/logs/conda_explicit.txt" \
          | grep -vE '^https?://(conda\.anaconda\.org|prefix\.dev)/conda-forge/' || true)
if [ -n "$foreign" ]; then
  echo "FATAL: conda packages from outside conda-forge:" >&2; echo "$foreign" >&2; exit 1
fi

pip freeze > "$HERE/logs/pip_freeze.txt"
python - <<'EOF'
import sys, numpy, scipy, tensorflow as tf, keras, cobaya, mpi4py
from mpi4py import MPI
print("python", sys.version.split()[0], "| numpy", numpy.__version__, "| scipy", scipy.__version__,
      "| tensorflow", tf.__version__, "| keras", keras.__version__, "| cobaya", cobaya.__version__,
      "| mpi4py", mpi4py.__version__)
print("MPI:", MPI.Get_library_version().splitlines()[0])
EOF
echo "environment ready at $CLIENT_ENV; package lists in logs/conda_explicit.txt and logs/pip_freeze.txt"
