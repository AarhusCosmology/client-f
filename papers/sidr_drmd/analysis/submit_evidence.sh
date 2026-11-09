#!/bin/bash -l
# The Bayesian evidence of chosen networks of a run, from the networks alone
# (`client.py evidence`, the BEST nested sampler; CLiENT's docs/evidence.md). One array task
# per network, after install/08_best.sh:
#
#   sbatch --array=10,15,20,25 analysis/submit_evidence.sh results/<timestamp>_<RUN>
#   sbatch --array=25 analysis/submit_evidence.sh results/<timestamp>_<RUN> --seed 43 --tag seed43
#   sbatch --array=10,15,20,25 analysis/submit_evidence.sh results/<timestamp>_<RUN> --n-live 1024
#   sbatch --array=21 analysis/submit_evidence.sh results/<timestamp>_<RUN> --region 1
#
# Everything after the run directory goes to `client.py evidence` (--n-live, default 2048;
# --seed; --tag for a second copy beside the first; --n-sigma, --prior, ...). No likelihood is
# evaluated: one process, TensorFlow on the task's cores.
#
# A network trained on boost points with acquisition.boost.region is integrated inside its
# region (--region), as its chains are sampled: outside, such a network is not to be used.
# A network from before the boosts has no region and is integrated over the whole training
# box. EVIDENCE_REGION=no in the environment integrates every network over the whole box;
# such an evidence gets no correction from a true check, which samples inside the region
# (another target, the summary says). `--region E` among the arguments (the fourth line above)
# integrates inside the same region with its edge at E times the reach of the boost points, as
# the profile setting `region: E` does; its output stands beside the default one (_edgeE).
#
# The second line above repeats one network with another seed: BEST's own error is
# sqrt(H / n_live), not a test, and two seeds that agree within it are (docs/evidence.md).
#
# Output: logs/evidence_<job>_<network>.out; RUN_DIR/evidence/evidence_it_N_nliveM[_TAG].json and
# the weighted chain beside it; and, rewritten by every task as it ends,
# RUN_DIR/analysis/evidence.md, .json and .png (analysis/evidence_summary.py): the networks side
# by side, with the evidence the boosts measured themselves and the correction to the true
# likelihood where a true check of the network exists.
#
#SBATCH --job-name=evidence
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=logs/evidence_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ./env.sh
TARGET=${1:?usage: sbatch --array=N,... analysis/submit_evidence.sh results/<timestamp>_<RUN> [client.py evidence arguments]}
shift
ITERATION=${SLURM_ARRAY_TASK_ID:?submit with --array=N,...: one task per network}
[ -d "$TARGET" ] || { echo "FATAL: no run directory $TARGET" >&2; exit 1; }
[ -f "$TARGET/trained_models/model_it_$ITERATION.keras" ] || { echo "FATAL: $TARGET has no network $ITERATION" >&2; exit 1; }
[ "$(git -C "$BEST_REPO" rev-parse HEAD 2>/dev/null)" = "$BEST_COMMIT" ] \
  || { echo "FATAL: $BEST_REPO is not at BEST commit ${BEST_COMMIT:0:10} (install/08_best.sh)" >&2; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMPI_MCA_mpi_yield_when_idle=1
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-16} TF_NUM_INTEROP_THREADS=2
export TF_CPP_MIN_LOG_LEVEL=2 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl_$USER
# BEST draws its progress display as wide as the terminal, and a batch job's "terminal" is 0
# columns wide, which stopped the first jobs with "Requires nbins > 0". Python reads these first.
export COLUMNS=100 LINES=40

# The region of the latest boost at or before this network, if the run has one. The command
# takes that region by itself; given here, it stops if the file cannot be used. Without a
# region it is the whole box, and EVIDENCE_REGION=no has to ask for that (--region none).
REGION=""
for word in "$@"; do
  case "$word" in --region*) REGION=given ;; esac
done
GIVEN=""
if [ "$REGION" = given ]; then
  REGION="" GIVEN="region as given"           # the user's own --region, among the arguments
elif [ "${EVIDENCE_REGION:-auto}" = no ]; then
  REGION="--region none"
else
  for file in "$TARGET"/boost/region_it_*.npz; do
    [ -e "$file" ] || continue
    number=${file##*_it_}; number=${number%.npz}
    if [ "$number" -le "$ITERATION" ]; then REGION="--region"; fi
  done
fi

echo "== $(date)  job ${SLURM_JOB_ID:-local} task $ITERATION on $(hostname -s)  cores=${SLURM_CPUS_PER_TASK:-?}  $TARGET  ${REGION:-${GIVEN:-whole box}}  $*"
echo "   CLiENT@$(git -C "$CLIENT_REPO" rev-parse --short HEAD)  BEST@$(git -C "$BEST_REPO" rev-parse --short HEAD)"
# shellcheck disable=SC2086
$MPI_LAUNCHER -n 1 python "$CLIENT_REPO/client.py" evidence "$TARGET" -i "$ITERATION" $REGION "$@"
python "${CLIENT_REPO:?source env.sh first}/hpc/analysis/evidence_summary.py" "$TARGET" || echo "(the summary could not be written; the evidence above stands)"
echo "== $(date)  done"
