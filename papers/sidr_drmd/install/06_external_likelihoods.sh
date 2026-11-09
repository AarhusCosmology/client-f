#!/bin/bash
# Step 6 (only for runs that need them): likelihood codes that are separate Python packages,
# and then the data of one run. RUN ON THE LOGIN NODE -- it downloads.
#
#   source env.sh && bash install/06_external_likelihoods.sh RUN
#
# for RUN = a folder under runs/ whose cobaya.yaml names such likelihoods, for instance
# planck20lh_lensing_baodr2_union31_desy1_12_param_with_Al_Omk.
#
# What it adds to the environment of steps 2 to 5, with pip:
#   planck-2020-hillipop   planck_2020_hillipop.TTTEEE (Planck PR4 high-l)
#   planck-2020-lollipop   planck_2020_lollipop.lowlE  (Planck PR4 low-l E)
#   act_dr6_lenslike       act_dr6_lenslike.ACTDR6LensLike (ACT DR6 + Planck PR4 lensing)
#   planckpr4lensing       planckpr4lensing.PlanckPR4Lensing (Planck PR4 lensing alone; from
#                          GitHub at $PR4_LENSING_COMMIT, its data inside the package)
#   astropy                the two Planck packages read their FITS files with it
# at the versions in env.sh. The four likelihood packages are installed without their
# dependencies (everything they import is there already, astropy apart), and astropy under a
# constraint that pins every package now installed: nothing that the earlier runs were made
# with can change, and the script stops if a version did.
#
# Then install/05_likelihoods.sh RUN fetches the data of every likelihood RUN's cobaya.yaml
# names into $COBAYA_PACKAGES_PATH. Cobaya does that itself for these three packages once
# they can be imported (they are "installable likelihoods"); planckpr4lensing needs nothing more.
#
# Afterwards, still on the login node (one evaluation, about a minute):
#   python install/check_likelihoods.py RUN
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/env.sh"
[ "${CONDA_PREFIX:-}" = "$CLIENT_ENV" ] || { echo "FATAL: env $CLIENT_ENV is not active" >&2; exit 1; }
RUN=${1:?usage: bash install/06_external_likelihoods.sh RUN}
[ -f "$HERE/runs/$RUN/cobaya.yaml" ] || { echo "FATAL: no runs/$RUN/cobaya.yaml" >&2; exit 1; }
cd "$HERE"
mkdir -p logs

# Every installed distribution at its present version: the "before", and pip's constraint
# (the same list without the few versions pip cannot parse; those are still compared below).
before=$(mktemp); constraint=$(mktemp)
python - "$before" "$constraint" <<'PY'
import sys
from importlib.metadata import distributions
from packaging.version import InvalidVersion, Version

seen = {}
for d in distributions():
    name = (d.metadata["Name"] or "").lower().replace("_", "-")
    if name:
        seen[name] = d.version
with open(sys.argv[1], "w") as everything, open(sys.argv[2], "w") as pinned:
    for name in sorted(seen):
        everything.write(f"{name}=={seen[name]}\n")
        try:
            Version(seen[name])
        except InvalidVersion:
            continue
        pinned.write(f"{name}=={seen[name]}\n")
PY
echo "== $(wc -l < "$before") packages recorded, $(wc -l < "$constraint") pinned for pip"

echo "== likelihood packages (no dependencies)"
python -m pip install --no-deps \
  "planck-2020-hillipop==$HILLIPOP_VERSION" \
  "planck-2020-lollipop==$LOLLIPOP_VERSION" \
  "act_dr6_lenslike==$ACT_LENSLIKE_VERSION" \
  "planckpr4lensing @ git+$PR4_LENSING_URL@$PR4_LENSING_COMMIT"

echo "== astropy $ASTROPY_VERSION, nothing installed may change"
python -m pip install --constraint "$constraint" "astropy==$ASTROPY_VERSION"

echo "== what changed"
python - "$before" <<'PY'
import sys
from importlib.metadata import distributions

before = dict(line.strip().split("==") for line in open(sys.argv[1]) if "==" in line)
now = {}
for d in distributions():
    name = (d.metadata["Name"] or "").lower().replace("_", "-")
    if name:
        now[name] = d.version
changed = {n: (before[n], now.get(n)) for n in before if now.get(n) != before[n]}
for name in sorted(set(now) - set(before)):
    print(f"  new   {name} {now[name]}")
if changed:
    for name, (old, new) in sorted(changed.items()):
        print(f"  CHANGED {name}: {old} -> {new}")
    sys.exit("FATAL: packages of the existing environment changed (above). Put them back before any run.")
print("  nothing that was installed has changed")

import astropy
import act_dr6_lenslike
import planck_2020_hillipop
import planck_2020_lollipop
import planckpr4lensing
import numpy
print(f"  imports: astropy {astropy.__version__}, planck_2020_hillipop, planck_2020_lollipop, "
      f"act_dr6_lenslike, planckpr4lensing {planckpr4lensing.__version__}, with numpy {numpy.__version__}")
PY
rm -f "$before" "$constraint"

echo "== data for $RUN"
bash install/05_likelihoods.sh "$RUN"

echo
echo "done. Now:  python install/check_likelihoods.py $RUN"
