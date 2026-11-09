#!/bin/bash
# Step 10 (optional): CLASS++ at another commit than the kit's pin, UNPATCHED, in a folder of its own. The env's
# classy (step 3), which every other run uses, and the patched build of step 9 are NOT touched; a run uses this
# build only when its job.env sets CLASSPP_BUILD_DIR and CLASSPP_BUILD_COMMIT and puts the folder's lib/ first on
# PYTHONPATH (the SIDR + DRMD runs on the H0 World Cup data, runs/sidr_drmd_wc*/job.env).
#
#   source env.sh && bash install/10_classpp_build.sh bfe6729e        # login node, about 10 minutes
#
# The commit is fetched from $CLASSPP_URL into the existing clone $CLASSPP_REPO (git fetch: the clone's checkout,
# the kit's pin that the env's build was made from, stays where it is) and exported with git archive.
#
# ── WHY A FOLDER OF ITS OWN, WITH ITS SOURCE ─────────────────────────────────
# CLASS++ compiles the path of its source tree into the library (__CLASSDIR__, for the BBN and HyRec tables), so
# the tree must stay where it was built: $DIR/src. The build goes to $DIR/lib, with the env build's options
# (CLASS_ARCH_FLAGS, fast-math default ON). PROVENANCE.txt's first line names the commit; install/check.py
# compares it with the run's CLASSPP_BUILD_COMMIT (boltzmann_build.py, the rule launch.py and the profile backend
# use). An existing folder is never replaced (runs may be using it): the script then only checks it.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/env.sh"
WANT=${1:?usage: bash install/10_classpp_build.sh COMMIT   (8 or more hex digits)}

[ "${CONDA_PREFIX:-}" = "$CLIENT_ENV" ] || { echo "FATAL: env $CLIENT_ENV is not active (env.sh sourced?)" >&2; exit 1; }
CC=$(command -v gcc); CXX=$(command -v g++)
case "$CC" in /usr/bin/*) echo "FATAL: gcc is the SYSTEM compiler ($CC), not the module" >&2; exit 1 ;; esac
case "$(command -v cmake)" in "$CLIENT_ENV"/*) ;; *) echo "FATAL: cmake is not the env's" >&2; exit 1 ;; esac
[ -d "$CLASSPP_REPO/.git" ] || { echo "FATAL: no clone $CLASSPP_REPO (install/03_classy.sh makes it)" >&2; exit 1; }

head_before=$(git -C "$CLASSPP_REPO" rev-parse HEAD)
git -C "$CLASSPP_REPO" fetch --quiet "$CLASSPP_URL" '+refs/heads/*:refs/remotes/origin/*'
FULL=$(git -C "$CLASSPP_REPO" rev-parse --verify --quiet "$WANT^{commit}") \
  || { echo "FATAL: $WANT is not a commit of $CLASSPP_URL" >&2; exit 1; }
COMMIT=${FULL:0:8}
[ "$(git -C "$CLASSPP_REPO" rev-parse HEAD)" = "$head_before" ] || { echo "FATAL: the clone's checkout moved" >&2; exit 1; }
DIR="${CLASSPP_BUILD_DIR:-$HOME/codes/classpp_${COMMIT}}"
FIRST="CLASS++ $COMMIT ($CLASSPP_URL), unpatched"

if [ -e "$DIR" ]; then
  head -1 "$DIR/PROVENANCE.txt" 2>/dev/null | grep -qxF "$FIRST" \
    || { echo "FATAL: $DIR exists and is not a build of $COMMIT by this script (or is incomplete); not replaced. Move it away to rebuild." >&2; exit 1; }
  [ ! -e "$DIR/applied.patch" ] || { echo "FATAL: $DIR holds a patch" >&2; exit 1; }
  ls "$DIR"/lib/classy*.so >/dev/null 2>&1 || { echo "FATAL: $DIR has no lib/classy*.so" >&2; exit 1; }
  echo "== $DIR: built before from $COMMIT, checked only"
else
  echo "== build CLASS++ $COMMIT into $DIR  (CC=$CC $("$CC" -dumpfullversion), arch $CLASS_ARCH_FLAGS, fast-math default)"
  mkdir -p "$DIR/src"
  git -C "$CLASSPP_REPO" archive "$FULL" | tar -x -C "$DIR/src"
  jobs=$(nproc); [ "$jobs" -le 8 ] || jobs=8          # the shared login node
  ( cd "$DIR/src" && CC="$CC" CXX="$CXX" python -m pip install --no-build-isolation --no-deps --target "$DIR/lib" . \
      -Cbuild-dir=build/skbuild-kit \
      "-Ccmake.define.CMAKE_C_COMPILER=$CC" "-Ccmake.define.CMAKE_CXX_COMPILER=$CXX" \
      "-Ccmake.define.CLASS_ARCH_FLAGS=$CLASS_ARCH_FLAGS" \
      "-Cbuild.tool-args=-j$jobs" ) > "$DIR/build.log" 2>&1 \
    || { echo "FATAL: the build failed, see $DIR/build.log" >&2; exit 1; }
  { echo "$FIRST"
    echo "commit $FULL"
    echo "built $(date) on $(hostname -s) by $USER with $CC $("$CC" -dumpfullversion), CLASS_ARCH_FLAGS=$CLASS_ARCH_FLAGS, CLASS_FAST_MATH default (ON)"
  } > "$DIR/PROVENANCE.txt"
fi

echo "== check: the build is the one found, its version is its source's, and it computes a LambdaCDM point"
PYTHONPATH="$DIR/lib:$PYTHONPATH" OMP_NUM_THREADS=1 python - "$DIR" <<'PY'
import re
import sys
import time
from pathlib import Path

import classy

folder = Path(sys.argv[1])
if not classy.__file__.startswith(str(folder / "lib") + "/"):
    sys.exit(f"FATAL: classy comes from {classy.__file__}")
version = re.search(r'^version\s*=\s*"([^"]+)"', (folder / "src" / "pyproject.toml").read_text(), re.M).group(1)
if classy.__version__ != version:
    sys.exit(f"FATAL: classy.__version__ {classy.__version__}, its source says {version}")
c = classy.Class()
c.set(dict(omega_b=0.02238, omega_cdm=0.1201, H0=67.4, A_s=2.1e-9, n_s=0.9659, tau_reio=0.0543,
           output="tCl,pCl,lCl", lensing="yes", l_max_scalars=2500))
t = time.time()
c.compute()
tt = c.lensed_cl(2000)["tt"]
if not (tt[2:] > 0).all():
    sys.exit("FATAL: lensed TT is not positive")
print(f"   classy {classy.__version__} from {classy.__file__}: LambdaCDM lensed TT in {time.time() - t:.1f} s on one thread")
PY
echo "== done. A run uses this build with, in its job.env:"
echo "   export CLASSPP_BUILD_DIR=\"$DIR\" CLASSPP_BUILD_COMMIT=$COMMIT"
echo "   export PYTHONPATH=\"\$CLASSPP_BUILD_DIR/lib\${PYTHONPATH:+:\$PYTHONPATH}\""
