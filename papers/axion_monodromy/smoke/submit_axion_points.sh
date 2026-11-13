#!/bin/bash -l
# True values of a run's likelihood at the points of a CSV, against the values beside them
# (smoke/axion_points.py). True evaluations only, no network:
#
#   sbatch smoke/submit_axion_points.sh e2e_axion_tt15 runs/e2e_axion_tt15/student_points.csv
#
# A hundred points take a few minutes on one q48 node. Output: logs/axion_points_<job>.out
# and logs/axion_points_<job>.json.
#
#SBATCH --job-name=axion_points
#SBATCH --partition=q48
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/axion_points_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ./env.sh
RUN=${1:?usage: sbatch smoke/submit_axion_points.sh RUN POINTS.csv}
POINTS=${2:?usage: sbatch smoke/submit_axion_points.sh RUN POINTS.csv}
[ ! -f "runs/$RUN/job.env" ] || source "runs/$RUN/job.env"
unset CLIENT_REFUSED_VALUE                 # the likelihood's own -inf
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMPI_MCA_mpi_yield_when_idle=1
export CLIENT_MPI_SCHEDULE=${CLIENT_MPI_SCHEDULE:-chunked}
export TF_CPP_MIN_LOG_LEVEL=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

echo "== $(date)  job ${SLURM_JOB_ID:-local} on $(hostname -s)  run=$RUN  points=$POINTS"
# Under mpirun (2 November 2026): since Grendel's Slurm 26.05 sets SLURM_STEP_ID in a batch script, a Python that
# starts MPI by itself there aborts in MPI_Init_thread (Open MPI 4.0.3); one started by mpirun does not.
$MPI_LAUNCHER -n 1 python install/check.py --stage all --run "$RUN"
$MPI_LAUNCHER -x CLIENT_MPI_SCHEDULE ${CLIENT_LOAD_GROUP:+-x CLIENT_LOAD_GROUP} -n "${SLURM_NTASKS:-1}" \
  python smoke/axion_points.py "$RUN" "$POINTS"
echo "== $(date)  done"
