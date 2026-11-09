#!/bin/bash
# Step 11 (only for runs that need them): the likelihood codes of the H0 World Cup data that are separate Python
# packages, and then the data of one run. RUN ON THE LOGIN NODE -- it downloads.
#
#   source env.sh && bash install/11_lite_likelihoods.sh RUN        # RUN = runs/sidr_drmd_wc_low, for instance
#
# What it adds to the environment, with pip and without dependencies (versions and commits in env.sh):
#   act_dr6_cmbonly   ACT DR6 ACT-lite and the Planck PR3 cut (act_dr6_cmbonly, act_dr6_cmbonly.PlanckActCut), from
#                     GitHub at $ACT_LITE_COMMIT; it reads its data with sacc ($SACC_VERSION, also step 7's)
#   candl-like        candl, the SPT likelihood framework (candl.interface.CandlCobayaLikelihood), and clipy-like,
#                     which it imports. Without JAX in the env candl runs on numpy; JAX is NOT installed.
#   spt_candl_data    the SPT-3G data sets for candl (spt_candl_data.SPT3G_D1_TnE_lite), from GitHub at
#                     $SPT_CANDL_DATA_COMMIT; its jax/gpjax dependencies serve another data set and are left out
#   muse3glike        SPT-3G 2019-2020 MUSE lensing (muse3glike.cobaya.spt3g_2yr_delensed_ee_optimal_pp_muse), from
#                     the zip at $MUSE3G_URL, unpacked into $MUSE3G_DIR (kept: the package is installed from it)
#   act_dr6_lenslike  ACT DR6 + Planck PR4 lensing: step 6's, at $ACT_LENSLIKE_VERSION, installed here if missing
# and then sacc under a constraint that pins every package now installed: nothing that the earlier runs were made
# with can change, and the script stops if a version did.
#
# Then install/05_likelihoods.sh RUN fetches the data of every likelihood RUN's cobaya.yaml names into
# $COBAYA_PACKAGES_PATH (ACT-lite's from LAMBDA, SRoll2 EE, ACT lensing; candl, the SPT data and MUSE carry theirs).
#
# Afterwards, still on the login node (one evaluation; the high-l spectra take a minute or more on one thread):
#   python install/check_likelihoods.py RUN
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/env.sh"
[ "${CONDA_PREFIX:-}" = "$CLIENT_ENV" ] || { echo "FATAL: env $CLIENT_ENV is not active" >&2; exit 1; }
RUN=${1:?usage: bash install/11_lite_likelihoods.sh RUN}
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

echo "== the MUSE lensing likelihood: $MUSE3G_URL -> $MUSE3G_DIR"
if [ ! -d "$MUSE3G_DIR" ]; then
  zip=$(mktemp --suffix=.zip)
  curl -fsSL -o "$zip" "$MUSE3G_URL" || { echo "FATAL: could not download $MUSE3G_URL" >&2; exit 1; }
  got=$(sha256sum "$zip" | cut -d' ' -f1)
  echo "   sha256 $got  ($(stat -c %s "$zip") bytes)"
  if [ -n "$MUSE3G_SHA256" ] && [ "$got" != "$MUSE3G_SHA256" ]; then
    echo "FATAL: the download's sha256 is not MUSE3G_SHA256=$MUSE3G_SHA256" >&2; exit 1
  fi
  mkdir -p "$MUSE3G_DIR"
  python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$zip" "$MUSE3G_DIR"
  echo "$got  $(basename "$MUSE3G_URL")" > "$MUSE3G_DIR/SHA256"
  rm -f "$zip"
else
  echo "   unpacked before ($(cat "$MUSE3G_DIR/SHA256" 2>/dev/null || echo 'no SHA256 record')): reused"
fi
# The Python package is the folder whose pyproject.toml or setup.py names muse3glike (the zip also holds a Julia one).
MUSE3G_PKG=$(python - "$MUSE3G_DIR" <<'PY'
import sys
from pathlib import Path
found = []
for f in sorted(Path(sys.argv[1]).rglob("*")):
    if f.name in ("pyproject.toml", "setup.py", "setup.cfg") and "muse3glike" in f.read_text(errors="replace"):
        found.append(str(f.parent))
found = sorted(set(found), key=len)
if not found:
    sys.exit("FATAL: no Python package named muse3glike in " + sys.argv[1])
print(found[0])
PY
)
echo "   the Python package: $MUSE3G_PKG"

echo "== likelihood packages (no dependencies; ACT-lite's distribution is called \"ACT DR6 CMBonly\")"
python -m pip install --no-deps \
  "git+$ACT_LITE_URL@$ACT_LITE_COMMIT" \
  "candl-like==$CANDL_VERSION" \
  "clipy-like==$CLIPY_VERSION" \
  "git+$SPT_CANDL_DATA_URL@$SPT_CANDL_DATA_COMMIT" \
  "$MUSE3G_PKG"
python -c "import act_dr6_lenslike" 2>/dev/null \
  || python -m pip install --no-deps "act_dr6_lenslike==$ACT_LENSLIKE_VERSION"

echo "== sacc $SACC_VERSION, nothing installed may change"
python -m pip install --constraint "$constraint" "sacc==$SACC_VERSION"

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

import importlib
for module in ("act_dr6_cmbonly", "candl", "candl.interface", "clipy", "spt_candl_data", "muse3glike",
               "muse3glike.cobaya", "act_dr6_lenslike", "sacc"):
    try:
        importlib.import_module(module)
    except Exception as error:                     # noqa: BLE001 - every failure is reported, then fatal
        sys.exit(f"FATAL: import {module}: {type(error).__name__}: {error}")
import candl.lib
import spt_candl_data
from pathlib import Path
if candl.lib.JAX_IMPORT_SUCCEEDED:
    sys.exit("FATAL: candl found JAX: this env is meant to run it on numpy (JAX compilations stay mapped for good)")
if not Path(spt_candl_data.SPT3G_D1_TnE_lite).exists():
    sys.exit(f"FATAL: no {spt_candl_data.SPT3G_D1_TnE_lite}")
print("  imports: act_dr6_cmbonly, candl (numpy), clipy, spt_candl_data (SPT3G_D1_TnE_lite there), muse3glike, "
      "act_dr6_lenslike, sacc")
PY
rm -f "$before" "$constraint"

echo "== data for $RUN"
bash install/05_likelihoods.sh "$RUN"

echo
echo "done. Now:  python install/check_likelihoods.py $RUN"
