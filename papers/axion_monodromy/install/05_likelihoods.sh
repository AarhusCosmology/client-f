#!/bin/bash
# Step 5: the likelihood data a run needs. RUN ON THE LOGIN NODE -- it downloads.
#
#   source env.sh && bash install/05_likelihoods.sh                 # every run under runs/
#   source env.sh && bash install/05_likelihoods.sh lcdm_planck_desi  # only what that run names
#
# The likelihoods are read from each run's cobaya.yaml and installed BY NAME into
# $COBAYA_PACKAGES_PATH, so a compute node never needs the internet. `cobaya-install
# <yaml>` is not used: it would also install its own copies of the Boltzmann codes
# (class_public and CAMB) under packages_path/code/, and Cobaya would then use those instead
# of the CLASS++ and CAMB built in steps 3 and 4. install/check.py fails if they appear.
#
# Planck low-l TT and EE, NPIPE CamSpec and DESI DR2 BAO together are about 1.1 GB.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/env.sh"
[ "${CONDA_PREFIX:-}" = "$CLIENT_ENV" ] || { echo "FATAL: env $CLIENT_ENV is not active" >&2; exit 1; }
cd "$HERE"
mkdir -p "$COBAYA_PACKAGES_PATH" logs

names=$(python - "$@" <<'PY'
import sys
from pathlib import Path
import yaml
runs = sys.argv[1:] or sorted(p.parent.name for p in Path("runs").glob("*/cobaya.yaml"))
found = []
for run in runs:
    path = Path("runs") / run / "cobaya.yaml"
    if not path.exists():
        sys.exit(f"no {path}")
    for name in yaml.safe_load(path.read_text())["likelihood"]:
        if name not in found:
            found.append(name)
print(" ".join(found))
PY
) || { echo "FATAL: could not read the runs' likelihoods" >&2; exit 1; }
[ -n "$names" ] || { echo "FATAL: no likelihoods found" >&2; exit 1; }
echo "likelihoods: $names"
read -ra LIKELIHOODS <<< "$names"

cobaya-install "${LIKELIHOODS[@]}" -p "$COBAYA_PACKAGES_PATH"
# Checked by Cobaya without installing anything.
cobaya-install --test -p "$COBAYA_PACKAGES_PATH" "${LIKELIHOODS[@]}"

for code in classy CAMB camb; do
  [ ! -e "$COBAYA_PACKAGES_PATH/code/$code" ] \
    || { echo "FATAL: $COBAYA_PACKAGES_PATH/code/$code exists: Cobaya would use it instead of the env's build. Remove it." >&2; exit 1; }
done
python -m pip freeze > logs/pip_freeze.txt
du -sh "$COBAYA_PACKAGES_PATH"
echo "likelihoods installed: $names"
