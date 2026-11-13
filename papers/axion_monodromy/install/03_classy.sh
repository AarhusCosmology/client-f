#!/bin/bash
# Step 3: CLASS++'s classy, built from a PINNED commit of the public repository into the env.
#
#   source env.sh && bash install/03_classy.sh        # login node: it clones from GitHub
#
# ── WHICH COMMIT ─────────────────────────────────────────────────────────────
# EXPECTED_COMMIT holds 7cc41a51 (16 September 2026, "classy: the LSS products Cobaya asks
# for"). The laptop's classy was built from exactly that tree (all 520 files compared by
# hash on 7 October), so the cluster and the laptop evaluate the same code and a run can be
# continued on either. Newer commits change, among other things, how ultra-relativistic
# species are approximated; move the pin deliberately, by editing EXPECTED_COMMIT and
# re-running this script, never by accident.
#
# ── WHY CC/CXX EXPLICITLY ────────────────────────────────────────────────────
# `module load gcc` puts 14.2 on PATH, but CMake looks for cc/c++, which are /usr/bin
# (GCC 8.5). The result links against 14.2's runtime and segfaults in std::filesystem
# (found by the harness this script comes from). The compilers are passed by absolute path.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/env.sh"
EXPECTED=$(tr -d '[:space:]' < "$HERE/EXPECTED_COMMIT")

[ "${CONDA_PREFIX:-}" = "$CLIENT_ENV" ] || { echo "FATAL: env $CLIENT_ENV is not active (step 2 done? env.sh sourced?)" >&2; exit 1; }
command -v gcc >/dev/null || { echo "FATAL: no gcc on PATH -- module load $GCC_MODULE" >&2; exit 1; }
CC=$(command -v gcc); CXX=$(command -v g++)
case "$CC" in /usr/bin/*) echo "FATAL: gcc is the SYSTEM compiler ($CC), not the module" >&2; exit 1 ;; esac
major=$("$CC" -dumpfullversion | cut -d. -f1)
[ "$major" -ge 9 ] || { echo "FATAL: gcc $major < 9 (std::filesystem ABI)" >&2; exit 1; }
case "$(command -v cmake)" in "$CLIENT_ENV"/*) ;; *) echo "FATAL: cmake is $(command -v cmake || echo missing), not the env's (step 2 installs it)" >&2; exit 1 ;; esac
echo "CC=$CC ($("$CC" -dumpfullversion))  arch=$CLASS_ARCH_FLAGS  $(cmake --version | head -1)"

if [ -d "$CLASSPP_REPO/.git" ]; then
  [ -z "$(git -C "$CLASSPP_REPO" status --porcelain --untracked-files=no)" ] \
    || { echo "FATAL: $CLASSPP_REPO has local changes" >&2; exit 1; }
  git -C "$CLASSPP_REPO" fetch origin
elif [ -e "$CLASSPP_REPO" ]; then
  echo "FATAL: $CLASSPP_REPO exists and is not a git clone" >&2; exit 1
else
  git clone "$CLASSPP_URL" "$CLASSPP_REPO"
fi
git -C "$CLASSPP_REPO" checkout --detach "$EXPECTED"
got=$(git -C "$CLASSPP_REPO" rev-parse --short=8 HEAD)
[ "$got" = "$EXPECTED" ] || { echo "FATAL: clone at $got, expected $EXPECTED" >&2; exit 1; }
echo "CLASS++ at $(git -C "$CLASSPP_REPO" log -1 --format='%h %ci %s')"

# A leftover classy (class_public's, or a pure-python package) would shadow the extension.
pip uninstall -y classy classy-community >/dev/null 2>&1 || true
site=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
[ ! -e "$site/classy/__init__.py" ] || { echo "FATAL: stale $site/classy/__init__.py shadows the build" >&2; exit 1; }

# At most 8 build jobs: this runs on the shared login node.
jobs=$(nproc); [ "$jobs" -le 8 ] || jobs=8
cd "$CLASSPP_REPO"
rm -f "$HERE/BUILD_COMMIT"
CC="$CC" CXX="$CXX" pip install --no-build-isolation . \
  -Cbuild-dir=build/skbuild-client \
  "-Ccmake.define.CMAKE_C_COMPILER=$CC" \
  "-Ccmake.define.CMAKE_CXX_COMPILER=$CXX" \
  "-Ccmake.define.CLASS_ARCH_FLAGS=$CLASS_ARCH_FLAGS" \
  "-Cbuild.tool-args=-j$jobs"

echo "$EXPECTED" > "$HERE/BUILD_COMMIT"
cd "$HERE"
[ -z "${CLIENT_RUN_CHECK:-}" ] || python install/check.py --stage classy
