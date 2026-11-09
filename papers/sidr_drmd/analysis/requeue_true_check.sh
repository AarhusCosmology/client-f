#!/bin/bash
# A combined emulator's true check and summary submitted again, after a failed true check (22 October 2026:
# the first true checks of the SIDR + DRMD combined emulators stopped at install/check.py, whose run check
# found the runs/<RUN>/client.yaml pointing at the low part's cobaya.yaml; fixed in the kit):
#
#   bash analysis/requeue_true_check.sh results/<time>_<RUN> EVIDENCE_JOB
#
# The report that sampled network 0 is done, so the true check starts at once (500 points of network 0);
# the summary waits for it and for the evidence job EVIDENCE_JOB (a finished one is fine). Scancel the old
# summary first: it waits for the failed true check and never runs.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: bash analysis/requeue_true_check.sh results/<time>_<RUN> EVIDENCE_JOB}
E=${2:?usage: bash analysis/requeue_true_check.sh results/<time>_<RUN> EVIDENCE_JOB}
[ -d "$OUT" ] || { echo "FATAL: no run directory $OUT" >&2; exit 1; }
[ -f "$OUT/analysis/posterior_it_0.npz" ] || { echo "FATAL: $OUT has no analysis/posterior_it_0.npz: its report has not run" >&2; exit 1; }
[[ "$E" =~ ^[0-9]+$ ]] || { echo "FATAL: EVIDENCE_JOB must be a Slurm job id, not '$E'" >&2; exit 1; }
T=$(sbatch --parsable --time=12:00:00 analysis/submit_true_check.sh "$OUT" --networks 0 --points 500)
# afterany for the evidence while it is still queued or running (a summary without it says so); a job that
# has left the queue is not a dependency Slurm accepts
DEP="afterok:${T%%;*}"
if squeue -h -j "$E" 2>/dev/null | grep -q .; then DEP="afterany:$E,$DEP"; fi
M=$(sbatch --parsable --dependency=$DEP --job-name=ev_summary --partition=q48 --nodes=1 \
      --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:30:00 --output=logs/ev_summary_%j.out \
      --wrap "source ./env.sh && python \$CLIENT_REPO/hpc/analysis/evidence_summary.py $OUT")
echo "$OUT: true check ${T%%;*}, summary ${M%%;*} (after evidence $E and the true check)"
