#!/bin/bash -l
# The true likelihood at posterior samples of chosen networks of a run (analysis/true_check.py):
#
#   sbatch analysis/submit_true_check.sh results/<timestamp>_<RUN>                      # network 10 (before the boosts) and the last
#   sbatch analysis/submit_true_check.sh results/<timestamp>_<RUN> --networks 5 10 15 --points 500
#
# 500 points per network: about two minutes of evaluation each on 48 ranks. The T = 1 chains
# come from analysis/report.py if it has run (RUN_DIR/analysis/posterior_it_N.npz); a network
# without one is sampled first, on rank 0. Output: logs/true_check_<job>.out and
# RUN_DIR/analysis/true_check.json.
#
#SBATCH --job-name=true_check
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/true_check_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ./env.sh
TARGET=${1:?usage: sbatch analysis/submit_true_check.sh results/<timestamp>_<RUN> [--networks N ...] [--points M]}
shift
[ -d "$TARGET" ] || { echo "FATAL: no run directory $TARGET" >&2; exit 1; }
RUN=$(basename "$TARGET" | sed -E 's/^[0-9]{8}_[0-9]{6}_//')
# A run's own job settings (runs/<RUN>/job.env), e.g. CLIENT_LOAD_GROUP for likelihoods that
# cannot be loaded on every rank at once.
[ ! -f "runs/$RUN/job.env" ] || source "runs/$RUN/job.env"
[ -f "runs/$RUN/cobaya.yaml" ] && [ -f "runs/$RUN/domain_bounds.json" ] || { echo "FATAL: no runs/$RUN/cobaya.yaml or domain_bounds.json" >&2; exit 1; }
# A run whose training set was evaluated again with another likelihood (analysis/relabel_run.py, 30 October 2026: the
# H0WC branches with runs/axion_wc_fast/cobaya.yaml) is checked against that one; every other run against runs/<RUN>/.
COBAYA=runs/$RUN/cobaya.yaml
if [ -f "$TARGET/relabel/relabel.json" ]; then
  COBAYA=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["cobaya"])' "$TARGET/relabel/relabel.json")
  [ -f "$COBAYA" ] || { echo "FATAL: $TARGET was relabelled with $COBAYA, which is not here" >&2; exit 1; }
  echo "   relabelled run: the true likelihood is $COBAYA (relabel/relabel.json)"
fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMPI_MCA_mpi_yield_when_idle=1
export CLIENT_MPI_SCHEDULE=${CLIENT_MPI_SCHEDULE:-chunked}
export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-8} TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS:-4}
export TF_CPP_MIN_LOG_LEVEL=2 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1

echo "== $(date)  job ${SLURM_JOB_ID:-local} on $(hostname -s)  run=$RUN  $TARGET  $*"
# Under mpirun (2 November 2026): since Grendel's Slurm 26.05 sets SLURM_STEP_ID in a batch script, a Python that
# starts MPI by itself there aborts in MPI_Init_thread (Open MPI 4.0.3); one started by mpirun does not.
[ -z "${CLIENT_RUN_CHECK:-}" ] || $MPI_LAUNCHER -n 1 python install/check.py --stage all --run "$RUN"
$MPI_LAUNCHER -x CLIENT_MPI_SCHEDULE ${CLIENT_LOAD_GROUP:+-x CLIENT_LOAD_GROUP} ${CLIENT_REFUSED_VALUE:+-x CLIENT_REFUSED_VALUE} -n "${SLURM_NTASKS:-1}" \
  bash -c '[ "${OMPI_COMM_WORLD_RANK:-0}" = 0 ] || export CUDA_VISIBLE_DEVICES=""; exec python "$@"' _ \
  "$CLIENT_REPO/hpc/analysis/true_check.py" "$TARGET" --cobaya "$COBAYA" --domain "runs/$RUN/domain_bounds.json" "$@"
echo "== $(date)  done"
