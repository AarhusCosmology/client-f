#!/bin/bash
# Step 8 (only for `client.py evidence`): the BEST nested sampler and what it imports.
# RUN ON THE LOGIN NODE -- it downloads.
#
#   source env.sh && bash install/08_best.sh
#
# CLiENT computes the Bayesian evidence of a run from its trained network with BEST
# (https://github.com/AndreasNygaard/best-inference; CLiENT's docs/evidence.md). No likelihood
# is evaluated. What this does:
#
#   1. clones BEST to $BEST_REPO and checks out $BEST_COMMIT (env.sh: the commit of the laptop's
#      checkout). It is NOT installed with pip: its metadata asks for numpy below 2, and this
#      environment has numpy 2.0.2, as the laptop does. CLiENT imports it from $BEST_PATH.
#   2. installs what BEST imports and the environment lacks: tensorflow-probability, tf-keras
#      and hypersphere-sampler, with tensorflow-probability's two missing dependencies
#      (dm-tree, decorator), at the laptop's versions, as wheels, under a constraint that
#      pins every package now installed. The script stops if a version changed.
#   3. imports BEST's NestedSampler the way CLiENT does.
#
# Afterwards:  sbatch --array=10,15,20,25 analysis/submit_evidence.sh results/<timestamp>_<RUN>
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/env.sh"
[ "${CONDA_PREFIX:-}" = "$CLIENT_ENV" ] || { echo "FATAL: env $CLIENT_ENV is not active" >&2; exit 1; }
cd "$HERE"
mkdir -p logs

echo "== 1. BEST at $BEST_COMMIT in $BEST_REPO"
if [ ! -d "$BEST_REPO/.git" ]; then
  git clone "$BEST_URL" "$BEST_REPO"
fi
[ -z "$(git -C "$BEST_REPO" status --porcelain --untracked-files=no)" ] \
  || { echo "FATAL: $BEST_REPO has local changes" >&2; exit 1; }
git -C "$BEST_REPO" cat-file -e "$BEST_COMMIT^{commit}" 2>/dev/null || git -C "$BEST_REPO" fetch --quiet origin
git -C "$BEST_REPO" -c advice.detachedHead=false checkout --quiet --detach "$BEST_COMMIT"
[ "$(git -C "$BEST_REPO" rev-parse HEAD)" = "$BEST_COMMIT" ] \
  || { echo "FATAL: $BEST_REPO is not at $BEST_COMMIT" >&2; exit 1; }
git -C "$BEST_REPO" log -1 --format='   %h %ci %s'

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
echo "== 2. $(wc -l < "$before" | tr -d ' ') packages recorded, $(wc -l < "$constraint" | tr -d ' ') pinned for pip"
python -m pip install --only-binary=:all: --constraint "$constraint" \
  "tensorflow-probability==$TFP_VERSION" "tf-keras==$TF_KERAS_VERSION" "hypersphere-sampler==$HYPERSPHERE_VERSION" \
  "dm-tree==$DM_TREE_VERSION" "decorator==$DECORATOR_VERSION"

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
PY
rm -f "$before" "$constraint"

echo "== 3. BEST as CLiENT imports it"
TF_CPP_MIN_LOG_LEVEL=2 python - <<'PY'
import os
import sys

sys.path.insert(0, os.environ["CLIENT_REPO"])
from evidence.sampler import import_best

sampler = import_best()
import numpy
import tensorflow
import tensorflow_probability
import tf_keras
import best

print(f"  ok  {sampler.__module__}.{sampler.__name__} from {os.path.dirname(os.path.dirname(best.__file__))}; "
      f"tensorflow {tensorflow.__version__}, tensorflow-probability {tensorflow_probability.__version__}, "
      f"tf-keras {tf_keras.__version__}, numpy {numpy.__version__}")
PY
{
  echo "best $(git -C "$BEST_REPO" rev-parse HEAD) ($(git -C "$BEST_REPO" log -1 --format=%ci))"
  echo "tensorflow-probability $TFP_VERSION, tf-keras $TF_KERAS_VERSION, hypersphere-sampler $HYPERSPHERE_VERSION"
  echo "installed $(date '+%Y-%m-%d %H:%M:%S') on $(hostname -s)"
} > logs/best_install.txt
python -m pip freeze > logs/pip_freeze.txt
cat logs/best_install.txt
echo
echo "done. Now:  sbatch --array=10,15,20,25 analysis/submit_evidence.sh results/<timestamp>_<RUN>"
