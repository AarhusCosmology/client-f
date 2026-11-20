#!/bin/bash
# Build one CLASS binary per Slurm partition, with that partition's -march.
#
#   ./build.sh /path/to/class_repo              # every partition in the table
#   ./build.sh /path/to/class_repo q64 q48      # just these
#   ./build.sh /path/to/class_repo --generic    # one portable binary, ./class
#
# Produces ./class_<partition> next to this script (and ./class for --generic),
# which is exactly what the submit_*.sh scripts look for.
#
# THIS CAMPAIGN NEEDS, at minimum:
#     q64                      family A (cmb), 64 cores
#     q40 q48                  family C (gal), 32 cores
#     q40 q48 q64 q128 q144    family B (iso) -- it is submitted with a partition
#                              LIST so it lands wherever there is a gap, and the
#                              submit script picks class_$SLURM_JOB_PARTITION, so
#                              EVERY partition in that list must have a binary or
#                              the task silently falls back to the portable one.
#
# ── WHY PER-PARTITION ────────────────────────────────────────────────────────
# Each partition is a homogeneous pair of sockets, and the collision kernel --
# ~95% of a high-Gamma run -- has a stride-1 inner loop over the multipole index
# and therefore vectorises.  Vector width is worth real time and it differs
# across the fleet: q20 (Ivy Bridge) has AVX but no FMA and no AVX2, everything
# else has at least AVX2, and q36 upward have AVX-512.  A single binary would
# have to target q20 and leave that on the table everywhere else; --generic does
# exactly that if you want one file.
#
# ── WHY CC/CXX ARE SET EXPLICITLY ────────────────────────────────────────────
# `ml load gcc` puts gcc 14.2 on PATH, but CMake's default compiler search looks
# for `cc` and `c++`, which are symlinked to /usr/bin/gcc = 8.5.  You then get a
# binary built by 8.5 against 14.2's headers/runtime, which has shown up here as
# a std::filesystem ABI segfault.  Passing the module's compilers by absolute
# path is the fix; do not rely on PATH order.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${1:?usage: build.sh <path-to-class-repo> [partition ...] | --generic}
shift || true

# ── compiler, from the module and not from /usr/bin ──────────────────────────
if ! command -v gcc >/dev/null 2>&1; then
  echo "no gcc on PATH -- run 'ml load gcc' first" >&2; exit 1
fi
export CC=$(command -v gcc)
export CXX=$(command -v g++)
echo "CC  = $CC  ($($CC  -dumpfullversion 2>/dev/null || $CC  -dumpversion))"
echo "CXX = $CXX ($($CXX -dumpfullversion 2>/dev/null || $CXX -dumpversion))"
case "$CC" in
  /usr/bin/*) echo "WARNING: that is the SYSTEM compiler, not the module." >&2 ;;
esac

# partition -> -march.  Keep in step with PARTITION_MARCH in config.py.
#
# ⚠ q192 DOES NOT EXIST on this cluster.  Every earlier hpc_* config carried a
# q192 row; it was a typo for q144 and the entry was dead weight that would have
# built a binary for a partition nothing can be submitted to.  Fixed here.
declare -A MARCH=(
  [q20]=ivybridge      [q24]=haswell        [q28]=broadwell
  [q36]=skylake-avx512 [q40]=cascadelake    [q48]=cascadelake
  [q64]=icelake-server [q128]=znver4        [q144]=znver5
)

build_one() {   # build_one <output-name> <arch-flags-or-empty> <label>
  local out=$1 flags=$2 label=$3
  local dir="$HERE/build_$label"
  echo "── $label: ${flags:-<portable, no -march>}"
  cmake -S "$REPO" -B "$dir" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
        ${flags:+-DCLASS_ARCH_FLAGS="$flags"} > "$dir.cmake.log" 2>&1 \
    || { echo "  cmake FAILED, see $dir.cmake.log" >&2; return 1; }
  cmake --build "$dir" --target class -j "$(nproc)" > "$dir.build.log" 2>&1 \
    || { echo "  build FAILED, see $dir.build.log" >&2; return 1; }
  # CLASS puts the binary at the top of the source tree.
  cp "$REPO/class" "$HERE/$out"
  echo "  -> $HERE/$out"
}

if [ "${1:-}" = "--generic" ]; then
  # Lowest common denominator across the fleet is q20: AVX, no FMA, no AVX2.
  build_one class ivybridge generic
  exit 0
fi

PARTS=("$@")
if [ ${#PARTS[@]} -eq 0 ]; then PARTS=("${!MARCH[@]}"); fi

for p in "${PARTS[@]}"; do
  arch=${MARCH[$p]:-}
  if [ -z "$arch" ]; then echo "unknown partition '$p'" >&2; exit 1; fi
  # Verify this compiler actually knows the target before spending a build on it
  # (znver5 needs gcc >= 14.1; older toolchains reject it outright).
  if ! echo 'int main(){}' | $CXX -march="$arch" -x c++ -c -o /dev/null - 2>/dev/null; then
    echo "── $p: $CXX does not support -march=$arch, SKIPPING" >&2
    continue
  fi
  build_one "class_$p" "-march=$arch" "$p"
done

# ── provenance ───────────────────────────────────────────────────────────────
# The submit scripts refuse to run against a binary whose commit does not match
# config.EXPECTED_COMMIT, because the merged master's DNCDM knob set is SMALLER
# than the one the pre-merge campaigns used and CLASS silently ignores keys it
# does not recognise.  Record what was actually built.
if git -C "$REPO" rev-parse --short HEAD >/dev/null 2>&1; then
  commit=$(git -C "$REPO" rev-parse --short HEAD)
  dirty=$(git -C "$REPO" status --porcelain | head -1)
  printf '%s%s\n' "$commit" "${dirty:+ (DIRTY WORKING TREE)}" > "$HERE/BUILD_COMMIT"
  echo
  echo "built from commit $commit${dirty:+  ** DIRTY WORKING TREE **}"
  echo "recorded in $HERE/BUILD_COMMIT -- gen.py checks it against EXPECTED_COMMIT"
fi

echo
echo "Binaries built.  submit_*.sh pick ./class_\$SLURM_JOB_PARTITION automatically"
echo "and fall back to ./class, so you can mix partitions in one submission."
