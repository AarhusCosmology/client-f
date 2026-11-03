#!/bin/bash -l
# An example Slurm job for one CLiENT run on one node: adapt the #SBATCH lines and the environment.
# Run it from this folder (hpc/), with the run's settings in runs/<RUN>/:
#
#   sbatch submit_client.sh lcdm_planck_desi -i 16            # new run -> results/<timestamp>_lcdm_planck_desi
#   sbatch submit_client.sh results/<timestamp>_<RUN> -i 2    # continue that run by two iterations
#
# Everything after the first argument is passed to client.py.
#
#SBATCH --job-name=client
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --output=logs/client_%j.out
#
# Ranks and threads. CLiENT alternates likelihood evaluations on all ranks (each a full Cobaya +
# Boltzmann code, one thread) with training and surrogate sampling in TensorFlow on rank 0 only.
# Use mpirun with --bind-to none rather than srun, so that rank 0's TensorFlow threads can use
# the cores the idle workers leave; during a training the workers should be near 0% CPU. On one
# 48-core machine one thread per rank gave about twice the evaluations per core-hour of 16 x 4.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

# Activate the environment here, e.g.:  conda activate clienv

TARGET=${1:?usage: sbatch submit_client.sh <run name under runs/> | <results/... dir to continue> [client.py arguments]}
shift
if [ -d "$TARGET" ]; then
  RUN=$(basename "$TARGET" | sed -E 's/^[0-9]{8}_[0-9]{6}_//')
  CLIENT_ARGS=("$TARGET" "$@")
else
  RUN=$TARGET
  CLIENT_ARGS=("runs/$RUN/client.yaml" -n "$RUN" -o results "$@")
fi
DOMAIN=runs/$RUN/domain_bounds.json
# A run's own job settings, e.g. CLIENT_LOAD_GROUP or CLIENT_REFUSED_VALUE (README.md).
[ ! -f "runs/$RUN/job.env" ] || source "runs/$RUN/job.env"
[ -f "runs/$RUN/client.yaml" ] || { echo "FATAL: no runs/$RUN/client.yaml" >&2; exit 1; }
[ -f "$DOMAIN" ] || { echo "FATAL: no $DOMAIN (the run's training box)" >&2; exit 1; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMPI_MCA_mpi_yield_when_idle=1      # idle Open MPI ranks yield instead of spinning
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
# One scatter and one gather per evaluation phase; the values are identical to the default `static`.
export CLIENT_MPI_SCHEDULE=${CLIENT_MPI_SCHEDULE:-chunked}
# TensorFlow on rank 0. Measured on 48 cores: 8 threads train about twice as fast as 1, and 48
# (TensorFlow's own default) are slower than 8.
export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-8}
export TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS:-4}
export TF_CPP_MIN_LOG_LEVEL=1

# -x: every rank must see the same settings. Only rank 0 runs TensorFlow: hide any GPU from the others.
mpirun --bind-to none -x CLIENT_MPI_SCHEDULE ${CLIENT_LOAD_GROUP:+-x CLIENT_LOAD_GROUP} ${CLIENT_REFUSED_VALUE:+-x CLIENT_REFUSED_VALUE} \
  bash -c '[ "${OMPI_COMM_WORLD_RANK:-0}" = 0 ] || export CUDA_VISIBLE_DEVICES=""; exec python "$@"' _ \
  launch.py "$DOMAIN" "${CLIENT_ARGS[@]}"
