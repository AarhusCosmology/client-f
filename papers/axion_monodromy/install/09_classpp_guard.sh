#!/bin/bash
# Step 9 (optional): CLASS++ at the kit's pin WITH the extrapolation guard, in a folder of its own. The env's
# classy (step 3), which every other run uses, is NOT touched; a run uses this build only when its job.env
# puts it first on PYTHONPATH (the SIDR + DRMD second set, runs/sidr_drmd2_*/job.env).
#
#   source env.sh && bash install/09_classpp_guard.sh        # login node, about 10 minutes
#
# ── WHAT THE PATCH DOES ──────────────────────────────────────────────────────
# smoke/classpp_extrap_guard.patch changes one case of NonlinearModule::nonlinear_get_source, the
# extrapolation of the sources beyond the computed k that halofit and HMcode integrate over. The default
# (extrap_max_scaled) continues each source as source_max ln(a k) / ln(a k_max), a fitted to the last two
# points. Where the linear spectrum has dark acoustic oscillations (f_idm >~ 0.2 with late z_stop) the table
# oscillates between grid points and its last two values can be nearly equal (1-2 parts in 10^4 at the failing
# points): |ln(a k_max)| is then 800-1900 and a = exp(ln(a k_max)) / k_max over- or underflows, so P(k) is NaN
# beyond k_max at z = 0; where |source| falls more steeply the fit can also reach zero. The patch continues
# with the slope-free extrap_hmcode form where |source| does not grow at the end of the table (underflow,
# zero crossing), and writes the same function without the overflow where it grows but ln(a k_max) >= 600;
# everywhere else, every LCDM-like point, the expression is the original. Checked with smoke/submit_guard_test.sh: job 6927182
# (first version: spectra identical bit for bit at 40 random points and the best fit, 22 of the 41 NaN
# points fixed) and its rerun with this version. Steen's decision of 21 October 2026 to run the second set
# with it.
# A regularisation, not a model: halofit and HMcode are not calibrated for such spectra (arXiv:2502.19383).
#
# ── WHY A FOLDER OF ITS OWN, WITH ITS SOURCE ─────────────────────────────────
# CLASS++ compiles the path of its source tree into the library (__CLASSDIR__, for the BBN and HyRec
# tables), so the tree must stay where it was built: $CLASSPP_GUARD_DIR/src. The build goes to
# $CLASSPP_GUARD_DIR/lib. An existing folder is never replaced (runs may be using it): the script then only
# checks it, and stops when it was built from another patch.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/env.sh"
EXPECTED=$(tr -d '[:space:]' < "$HERE/EXPECTED_COMMIT")
PATCH="${CLIENT_REPO:?source env.sh first}/hpc/patches/classpp_extrap_guard.patch"
DIR="${CLASSPP_GUARD_DIR:-$HOME/codes/classpp_${EXPECTED}_guard}"

[ "${CONDA_PREFIX:-}" = "$CLIENT_ENV" ] || { echo "FATAL: env $CLIENT_ENV is not active (env.sh sourced?)" >&2; exit 1; }
CC=$(command -v gcc); CXX=$(command -v g++)
case "$CC" in /usr/bin/*) echo "FATAL: gcc is the SYSTEM compiler ($CC), not the module" >&2; exit 1 ;; esac
case "$(command -v cmake)" in "$CLIENT_ENV"/*) ;; *) echo "FATAL: cmake is not the env's" >&2; exit 1 ;; esac
got=$(git -C "$CLASSPP_REPO" rev-parse --short=8 HEAD)
[ "$got" = "$EXPECTED" ] || { echo "FATAL: $CLASSPP_REPO is at $got, the kit's pin is $EXPECTED" >&2; exit 1; }
[ -f "$PATCH" ] || { echo "FATAL: no $PATCH (sync the kit)" >&2; exit 1; }

if [ -e "$DIR" ]; then
  cmp -s "$PATCH" "$DIR/applied.patch" \
    || { echo "FATAL: $DIR exists and was built from another patch (or is incomplete); not replaced. Move it away to rebuild." >&2; exit 1; }
  ls "$DIR"/lib/classy*.so >/dev/null 2>&1 || { echo "FATAL: $DIR has no lib/classy*.so" >&2; exit 1; }
  echo "== $DIR: built before from this patch, checked only"
else
  echo "== build CLASS++ $got + $(basename "$PATCH") into $DIR  (CC=$CC $("$CC" -dumpfullversion), arch $CLASS_ARCH_FLAGS, fast-math default)"
  mkdir -p "$DIR/src"
  git -C "$CLASSPP_REPO" archive HEAD | tar -x -C "$DIR/src"
  ( cd "$DIR/src" && patch -p1 < "$PATCH" )
  jobs=$(nproc); [ "$jobs" -le 8 ] || jobs=8          # the shared login node
  ( cd "$DIR/src" && CC="$CC" CXX="$CXX" python -m pip install --no-build-isolation --no-deps --target "$DIR/lib" . \
      -Cbuild-dir=build/skbuild-guard \
      "-Ccmake.define.CMAKE_C_COMPILER=$CC" "-Ccmake.define.CMAKE_CXX_COMPILER=$CXX" \
      "-Ccmake.define.CLASS_ARCH_FLAGS=$CLASS_ARCH_FLAGS" \
      "-Cbuild.tool-args=-j$jobs" ) > "$DIR/build.log" 2>&1 \
    || { echo "FATAL: the build failed, see $DIR/build.log" >&2; exit 1; }
  cp "$PATCH" "$DIR/applied.patch"
  { echo "CLASS++ $got ($CLASSPP_REPO) + $(basename "$PATCH") (sha256 $(sha256sum "$PATCH" | cut -c1-16))"
    echo "built $(date) on $(hostname -s) by $USER with $CC $("$CC" -dumpfullversion), CLASS_ARCH_FLAGS=$CLASS_ARCH_FLAGS, CLASS_FAST_MATH default (ON)"
  } > "$DIR/PROVENANCE.txt"
fi

echo "== check: the build is the one found, and it gives the env build's spectra at an ordinary point"
found=$(PYTHONPATH="$DIR/lib:$PYTHONPATH" python -c "import classy; print(classy.__file__)")
case "$found" in "$DIR/lib/"*) echo "   classy from $found" ;; *) echo "FATAL: classy comes from $found" >&2; exit 1 ;; esac
RUN=$(ls -d "$HERE"/results/*_sidr_drmd_pp 2>/dev/null | head -1)
if [ -n "$RUN" ]; then
  cd "$HERE"
  export OMP_NUM_THREADS=1
  python smoke/kmax_ladder.py "$RUN" runs/sidr_drmd2_pp_low/cobaya.yaml install_env_best --module halofit --kmax 5 --points best | grep SUMMARY
  PYTHONPATH="$DIR/lib:$PYTHONPATH" python smoke/kmax_ladder.py "$RUN" runs/sidr_drmd2_pp_low/cobaya.yaml install_guard_best \
      --module halofit --kmax 5 --points best | grep SUMMARY
  python smoke/compare_ladders.py logs/kmax_ladder_install_env_best.npz logs/kmax_ladder_install_guard_best.npz | tail -2
else
  echo "   (no results/*_sidr_drmd_pp run here: the spectra check is skipped)"
fi
echo "== done. A run uses this build with, in its job.env:"
echo "   export PYTHONPATH=\"$DIR/lib\${PYTHONPATH:+:\$PYTHONPATH}\""
