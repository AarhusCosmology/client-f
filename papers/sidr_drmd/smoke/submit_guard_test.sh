#!/bin/bash -l
# The extrapolation guard (smoke/classpp_extrap_guard.patch) on Grendel. Beyond the computed k, CLASS++
# continues each source as source_max ln(a k) / ln(a k_max), with a fitted to the last two points
# (extrap_max_scaled, the default). At the dark acoustic oscillations of the SIDR + DRMD points (f_idm >~ 0.2,
# late z_stop) the table oscillates between grid points and its last two values can be nearly equal: at the
# failing points |ln(a k_max)| is 800-1900, a = exp(ln(a k_max)) / k_max over- or underflows and P(k) is NaN
# beyond k_max at z = 0 (where |source| falls more steeply the fit can also reach zero). The guard keeps the
# form wherever it is well posed; where |source| does not grow at the end of the table it continues with the
# slope-free extrap_hmcode form (the first version, job 6927182: 22 of 41 points fixed); where it grows but
# ln(a k_max) >= 600 the same function is written as 1 + ln(k / k_max) / ln(a k_max) (second version, which
# fixed the other 19). Below 600 (every LCDM-like point) the expression is the original.
#
#   sbatch smoke/submit_guard_test.sh results/20261020_122013_sidr_drmd_pp
#
# The pinned CLASS++ commit is exported (git archive) into $HOME/codes/classy_build_test/src_guard, patched
# there and built into $HOME/codes/classy_build_test/guard with the env build's options (-march=x86-64-v3,
# fast-math). The env's classy, which the runs use, and $CLASSPP_REPO are NOT touched. Then, with each build
# (the env's and the guard's), at the runs' k_max (5), halofit and HMcode:
#   the run's 41 refused points, 40 random points of the second set's box with delta_Neff < 0.13, the best
#   point, and the two builds compared point by point (smoke/compare_ladders.py: identical, or how far apart
#   against cosmic variance).
# Made 21 October 2026. About 15 minutes of building and 20 of evaluating on 16 cores.
# Output: logs/guard_test_<job>.out, logs/kmax_ladder_{env,guard}_*.{json,npz}.
#
#SBATCH --job-name=guard_test
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/guard_test_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ./env.sh
TARGET=${1:?usage: sbatch smoke/submit_guard_test.sh results/<timestamp>_<RUN>}
COBAYA=runs/sidr_drmd2_pp_low/cobaya.yaml
PATCH="$CLIENT_REPO/hpc/patches/classpp_extrap_guard.patch"
[ -d "$TARGET" ] || { echo "FATAL: no run directory $TARGET" >&2; exit 1; }
[ -f "$COBAYA" ] && [ -f "$PATCH" ] || { echo "FATAL: no $COBAYA or $PATCH (sync the kit)" >&2; exit 1; }
[ "${CONDA_PREFIX:-}" = "$CLIENT_ENV" ] || { echo "FATAL: env $CLIENT_ENV is not active" >&2; exit 1; }
CC=$(command -v gcc); CXX=$(command -v g++)
case "$CC" in /usr/bin/*) echo "FATAL: gcc is the SYSTEM compiler ($CC), not the module" >&2; exit 1 ;; esac
EXPECTED=$(tr -d '[:space:]' < EXPECTED_COMMIT)
got=$(git -C "$CLASSPP_REPO" rev-parse --short=8 HEAD)
[ "$got" = "$EXPECTED" ] || { echo "FATAL: $CLASSPP_REPO is at $got, the kit's pin is $EXPECTED" >&2; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
P=${SLURM_CPUS_PER_TASK:-16}
OUT=$HOME/codes/classy_build_test
SRC=$OUT/src_guard
mkdir -p "$OUT" logs
echo "== $(date)  job ${SLURM_JOB_ID:-local} on $(hostname -s)  CLASS++ $got  CC=$CC ($("$CC" -dumpfullversion))  run=$TARGET"

if ls "$OUT/guard"/classy*.so >/dev/null 2>&1 && cmp -s "$PATCH" "$OUT/guard/applied.patch"; then
  echo "== the guard build: built before with this patch, in $OUT/guard: reused"
else
  echo "== the guard build: export $got, patch, build"
  rm -rf "$SRC" "$OUT/guard"
  mkdir -p "$SRC"
  git -C "$CLASSPP_REPO" archive HEAD | tar -x -C "$SRC"
  ( cd "$SRC" && patch -p1 < "$PATCH" ) || { echo "FATAL: the patch does not apply to $got" >&2; exit 1; }
  ( cd "$SRC" && CC="$CC" CXX="$CXX" python -m pip install --no-build-isolation --no-deps --target "$OUT/guard" . \
      -Cbuild-dir="build/skbuild-guard" \
      "-Ccmake.define.CMAKE_C_COMPILER=$CC" "-Ccmake.define.CMAKE_CXX_COMPILER=$CXX" \
      "-Ccmake.define.CLASS_ARCH_FLAGS=-march=x86-64-v3" "-Ccmake.define.CLASS_FAST_MATH=ON" \
      "-Cbuild.tool-args=-j$P" ) > logs/classy_build_guard.build.log 2>&1 \
    || { echo "FATAL: the guard build failed, see logs/classy_build_guard.build.log" >&2; exit 1; }
  cp "$PATCH" "$OUT/guard/applied.patch"
fi

BASE_PYTHONPATH=${PYTHONPATH:-}                # env.sh puts CLiENT there
ladder() { python smoke/kmax_ladder.py "$TARGET" "$COBAYA" "$@" --kmax 5; }
for build in env guard; do
  if [ "$build" = guard ]; then export PYTHONPATH="$OUT/guard${BASE_PYTHONPATH:+:$BASE_PYTHONPATH}"; else export PYTHONPATH="$BASE_PYTHONPATH"; fi
  for module in halofit hmcode; do
    echo "== $build build, $module"
    ladder ${build}_${module}_refused --module $module --points refused --processes "$P"
    ladder ${build}_${module}_sample40 --module $module --points sample:40 --processes "$P"
    ladder ${build}_${module}_best --module $module --points best
  done
done
export PYTHONPATH="$BASE_PYTHONPATH"
echo "== the env's build (A) against the guard's (B)"
for module in halofit hmcode; do
  for points in refused sample40 best; do
    python smoke/compare_ladders.py logs/kmax_ladder_env_${module}_${points}.npz logs/kmax_ladder_guard_${module}_${points}.npz
  done
done
echo "== summaries"
grep -h "SUMMARY" "logs/guard_test_${SLURM_JOB_ID:-local}.out" 2>/dev/null || true
echo "== $(date)  done"
