#!/bin/bash
# Run ONE cell of ONE family.  Called by every submit_*.sh, which are otherwise
# nothing but #SBATCH headers -- four copies of this logic is four chances for
# the families to drift apart, and the guards below are the whole reason this
# campaign can be trusted.
#
#   ./runcell.sh <family>        family in {cmb, iso, gal, bg}
#
# Reads runfile_<family> line $SLURM_ARRAY_TASK_ID.
set -uo pipefail

FAMILY=${1:?usage: runcell.sh <family>}
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" || exit 1

echo "========= $FAMILY cell started at $(date) =========="
echo "jobid ${SLURM_JOB_ID:-none}  array id ${SLURM_ARRAY_TASK_ID:-1}  host $(hostname)"

# --export=NONE means nothing is inherited from the submitting shell, so the
# compiler runtime (libgomp, libstdc++) has to be loaded here.  It must be the
# SAME gcc the binary was built with -- build.sh uses the module's gcc 14.2,
# whose libstdc++ the system 8.5 runtime cannot satisfy.
ml load gcc 2>/dev/null || module load gcc 2>/dev/null || true

# ── the binary ───────────────────────────────────────────────────────────────
# Per-partition build, with that partition's -march.  Family `iso` is submitted
# to a partition LIST so it lands wherever there is a gap, so this has to resolve
# at run time and not at submit time.
if [ -n "${CLASS_BIN:-}" ]; then
  CLASS=$CLASS_BIN
elif [ -x "./class_${SLURM_JOB_PARTITION:-none}" ]; then
  CLASS="./class_${SLURM_JOB_PARTITION}"
else
  CLASS=./class
  echo "note: no class_${SLURM_JOB_PARTITION:-none}, using the portable ./class"
fi
[ -x "$CLASS" ] || { echo "FATAL: $CLASS is not executable -- run ./build.sh"; exit 1; }
# Resolve to an ABSOLUTE path once.  The key probe below runs in a temp directory,
# and everything downstream is easier to reason about when the binary path does
# not depend on the current working directory.
case "$CLASS" in /*) ;; *) CLASS="$PWD/${CLASS#./}" ;; esac
echo "binary $CLASS"

# ── GUARD 1: the binary is the commit this campaign was generated against ────
# Authoritative and precise, so it runs FIRST -- a wrong commit should say so
# rather than be diagnosed by the softer probe below.
if [ -f BUILD_COMMIT ] && [ -f EXPECTED_COMMIT ]; then
  built=$(cut -d' ' -f1 < BUILD_COMMIT)
  want=$(cut -d' ' -f1 < EXPECTED_COMMIT)
  if [ "$built" != "$want" ]; then
    echo "FATAL: binary built from $built, config.py expects $want."
    echo "       Rebuild, or update EXPECTED_COMMIT in config.py if that is intended."
    exit 1
  fi
  grep -q DIRTY BUILD_COMMIT && echo "WARNING: binary built from a DIRTY working tree"
  echo "commit $built"
elif [ -f EXPECTED_COMMIT ]; then
  # ⚠ THIS BRANCH USED TO BE A `note:` AND CARRY ON. That is how hpc_gal ran its entire
  # round -- 99 cells, ~7000 charged core-hours -- against a binary that did not have
  # the code change the round existed to measure, reproducing hpc_prod2 to 1e-6 and
  # costing the campaign a decision cycle. Every one of its 125 logs says
  # "commit check skipped" and nothing stopped.
  #
  # A guard that can be skipped by the ABSENCE of a file is not a guard. build.sh
  # writes BUILD_COMMIT; if it is not here, build.sh did not run in this directory or
  # it could not read the repo, and either way the binary's provenance is unknown.
  echo "FATAL: EXPECTED_COMMIT is present but BUILD_COMMIT is not, so the binary's"
  echo "       provenance is unknown. Run ./build.sh <repo> in THIS directory."
  echo "       (hpc_gal skipped this check and voided a 7000 core-hour round.)"
  exit 1
fi

# ── GUARD 2: the binary understands this campaign's DNCDM key set ────────────
# A FUNCTIONAL PROBE, not a `strings` scan.
#
# An earlier version of this grepped the binary for `dr_reduced_moments` and
# refused to run if the literal was absent.  That was wrong, and it rejected good
# binaries: whether a key literal survives into .rodata is a COMPILER decision,
# not a property of the code.  In the very same binary, `dr_q_min` and `dr_q_max`
# do NOT appear as standalone strings at all -- they exist only inside longer
# diagnostic messages, their own literals having been merged away -- while
# `dr_reduced_moments` happens to survive.  Change compiler, -march or inlining
# and that flips.
#
# ⚠ `dr_table_max_dlna` (PR #413) IS THE KEY THIS ROUND TURNS ON, and it is in the probe
# for the reason hpc_ratchet put `balanced_gather` in its own: CLASS ignores unrecognised
# keys with NO warning, so a binary built from a commit before #413 would pass every other
# guard, silently ignore the spacing this round exists to set, and reproduce hpc_prod2
# exactly -- which is precisely what hpc_gal did. This probe is what makes that
# impossible rather than merely unlikely.
#
# So ask the parser instead.  Feed it one background-only cell carrying the keys
# this campaign depends on and see whether it reports any of them unread.  That
# is exactly the property that matters, it costs well under a second, and it
# cannot be fooled by string merging.  (Guard 4 makes the same check on the real
# ini at full strength; this one just fails fast, before a 15-day job starts.)
PROBE=$(mktemp -d) || PROBE=""
if [ -n "$PROBE" ]; then
  cat > "$PROBE/probe.ini" <<'PROBEINI'
output =
write warnings = yes
write background = no
dncdm1.type = ncdm_decay_dr
dncdm1.m = 0.3
dncdm1.T = 0.71611
dncdm1.deg = 1
dncdm1.inverse_decays = yes
dncdm1.quantum_statistics = yes
dncdm1.dr_f_ini_l = 1
dncdm1.dr_f_ini_phi = 0
dncdm1.quadrature_strategy = 3
dncdm1.momenta_bins = 16
dncdm1.dr_q_min = 1e-2
dncdm1.dr_q_max = 50
dncdm1.dr_N_q = 41
dncdm1.dr_bg_refine = 1
dncdm1.dr_reduced_moments = 6
dncdm1.dr_table_max_dlna = 3.5e-3
dncdm1.Gamma = 1e5
evolver_background = 3
evolver_perturbations = 3
l_max_ncdm = 17
PROBEINI
  probe_out=$( cd "$PROBE" && "$CLASS" probe.ini 2>&1 || true )
  if [ -z "$probe_out" ]; then
    echo "note: key probe produced no output -- inconclusive, continuing"
  elif echo "$probe_out" | grep -q "input line not recognized"; then
    echo "FATAL: $CLASS does not read keys this campaign depends on:"
    echo "$probe_out" | grep "input line not recognized" | sed 's/^/       /'
    echo "       That is a pre-merge / campaign-branch binary.  Rebuild from"
    echo "       master (config.EXPECTED_COMMIT) with ./build.sh."
    rm -rf "$PROBE"
    exit 1
  else
    echo "key probe: every campaign key was read"
  fi
  rm -rf "$PROBE"
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

RUNFILE="runfile_${FAMILY}"
INI=$(awk "NR == ${SLURM_ARRAY_TASK_ID:-1}" "$RUNFILE")
if [ -z "$INI" ]; then
  echo "no line ${SLURM_ARRAY_TASK_ID:-1} in $RUNFILE -- the array is larger than the runfile"
  exit 1
fi
TAG=$(basename "$INI" .ini)
ROOT="out/${TAG}_"
echo "ini  $INI"
echo "root $ROOT"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

# ── resume, but ONLY for a byte-identical ini ────────────────────────────────
# Array jobs get killed at the wall limit and requeued, so a finished cell must
# not be redone.  But the tag encodes (family, Gamma, mass, scheme, rung) and NOT
# the grid, so changing the bins-per-decade rule in config.py leaves every tag
# unchanged -- a bare "output exists" test would then skip the cell and silently
# keep the STALE result.  So the ini that produced the output is stored beside it
# and compared.
DONE_MARK="${ROOT}cl_lensed.dat"
[ "$FAMILY" = "iso" ] && DONE_MARK="${ROOT}tk.dat"
[ "$FAMILY" = "bg" ]  && DONE_MARK="${ROOT}bgslim.dat"
if [ -s "$DONE_MARK" ] || [ -s "${DONE_MARK}.gz" ]; then
  if [ -f "${ROOT}used.ini" ] && cmp -s "${ROOT}used.ini" "$INI"; then
    echo "already complete with an identical ini, skipping"
    echo "ELAPSED_S 0"
    echo "========= finished at $(date) =========="
    exit 0
  fi
  echo "output exists but the ini differs (or was not recorded) -- recomputing"
fi

mkdir -p out logs
LOG="${ROOT}class.out"
"$CLASS" "$INI" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "class exit $rc"

# ── GUARD 4: every key in the ini was actually READ ──────────────────────────
# THE most important guard in this campaign.  CLASS silently ignores keys it does
# not recognise -- no warning, no non-zero exit, not even at input_verbose = 2 --
# so a stale or misspelled key produces a perfectly healthy-looking run of a
# DIFFERENT model.  `write warnings = yes` (gen.py puts it in every ini) turns
# each one into a line on stdout, and this turns that line into a failure.
#
# Note the stdout form is the right one to grep, not the `unused_parameters`
# file: that file is written BEFORE `write warnings` is itself marked read, so it
# always lists `write warnings = yes` and would need a whitelist.
if grep -q "input line not recognized" "$LOG"; then
  echo "FATAL: CLASS did not read every key in $INI --"
  grep "input line not recognized" "$LOG" | sed 's/^/       /'
  echo "       the cell ran, but not the model this campaign meant.  Output discarded."
  mv -f "$LOG" "${ROOT}REJECTED.out" 2>/dev/null
  exit 2
fi

# ── GUARD 5: the daughter grid was wide enough to hold the emission band ─────
# dr_q_max below q* is an ENERGY LEAK, not a resolution trade-off: TwoBinSplit
# clamps the off-grid deposit into the top bin, conserving number but filing the
# energy at q_max instead of at q*.  It is silent because the identities the
# kernel asserts are NUMBER identities.  CLASS warns once per run; surface it.
if grep -q "is below the highest momentum this decay can emit" "$LOG"; then
  echo "QMAX_WARNING yes"
  grep "is below the highest momentum" "$LOG" | sed 's/^/       /'
else
  echo "QMAX_WARNING no"
fi

if [ $rc -eq 0 ]; then
  cp "$INI" "${ROOT}used.ini"
  # How many k modes were actually solved.  Free: CLASS prints it in the P(k)
  # header, and the transfer file has one row per mode.  Recorded because the
  # cost model is per-mode and because a preset that quietly changed its mode
  # count is exactly the kind of thing that shows up as an unexplained 2x.
  nk=$(sed -n 's/.*number of wavenumbers equal to *\([0-9]*\).*/\1/p' \
         "${ROOT}pk.dat" 2>/dev/null | head -1)
  [ -z "$nk" ] && nk=$(grep -vc '^#' "${ROOT}tk.dat" 2>/dev/null)
  echo "N_K ${nk:-unknown}"

  # The background pre-flight's table carries lnf_dncdm1[i], dlnfdlnq[i] and
  # f_dr_*[j] for EVERY momentum bin, so it is ~2*dr_N_q + 3*momenta_bins columns
  # wide -- of order a GB at dr_N_q = 6401.  Everything the convergence
  # diagnostics need (Ecomov, N_eff, a_half, a_ext) lives in about 20 of them, so
  # slim it first and keep the full table gzipped beside it.
  if [ "$FAMILY" = "bg" ] && [ -s "${ROOT}background.dat" ]; then
    awk -f slim_background.awk "${ROOT}background.dat" > "${ROOT}bgslim.dat" \
      || echo "WARNING: slim_background.awk failed; the full table is still there"
  fi

  # Compress the bulky text output in place.  The k_output tables run to hundreds
  # of MB each (one row per evolver step, ~40 columns) and gzip takes ~5x off
  # them.  numpy.loadtxt reads .gz directly, so nothing downstream changes.
  for f in "${ROOT}"perturbations_k*_s.dat "${ROOT}"background.dat; do
    [ -s "$f" ] && gzip -f "$f"
  done
fi

# Machine-readable for collect.py: $SECONDS is locale-free, unlike diffing two
# date(1) strings.
echo "ELAPSED_S $SECONDS"
echo "CPUS $OMP_NUM_THREADS"
echo "PARTITION ${SLURM_JOB_PARTITION:-none}"
echo "========= finished at $(date) =========="
exit $rc
