#!/bin/bash -l
# Cost of one evaluation of a run's likelihood with the whole node busy, before the real job:
#
#   sbatch smoke/submit_timing.sh                      # lcdm_planck_desi, one q48 node
#   sbatch smoke/submit_timing.sh lcdm_planck_desi_camb
#
# 1. install/check.py for the run
# 2. smoke/timing.py: 4 points per rank, uniform in the training box, through CLiENT's own
#    dispatcher; prints s/evaluation, failures, peak memory, and hours of evaluation for the run
#
#SBATCH --job-name=client_timing
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=logs/timing_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ./env.sh
RUN=${1:-lcdm_planck_desi}
# A run's own job settings (runs/<RUN>/job.env), e.g. CLIENT_LOAD_GROUP for likelihoods that
# cannot be loaded on every rank at once.
[ ! -f "runs/$RUN/job.env" ] || source "runs/$RUN/job.env"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMPI_MCA_mpi_yield_when_idle=1
export CLIENT_MPI_SCHEDULE=${CLIENT_MPI_SCHEDULE:-chunked}   # as submit_client.sh
export TF_CPP_MIN_LOG_LEVEL=2 PYTHONDONTWRITEBYTECODE=1

echo "== $(date)  job ${SLURM_JOB_ID:-local} on $(hostname -s)  run=$RUN"
echo "== 1. guards"
# Under mpirun (2 November 2026): since Grendel's Slurm 26.05 sets SLURM_STEP_ID in a batch script, a Python that
# starts MPI by itself there aborts in MPI_Init_thread (Open MPI 4.0.3); one started by mpirun does not.
[ -z "${CLIENT_RUN_CHECK:-}" ] || $MPI_LAUNCHER -n 1 python install/check.py --stage all --run "$RUN"

echo "== 2. timing on ${SLURM_NTASKS:-1} ranks"
$MPI_LAUNCHER -x CLIENT_MPI_SCHEDULE ${CLIENT_LOAD_GROUP:+-x CLIENT_LOAD_GROUP} ${CLIENT_REFUSED_VALUE:+-x CLIENT_REFUSED_VALUE} -n "${SLURM_NTASKS:-1}" python smoke/timing.py "$RUN"

echo "== $(date)  done"
