#!/bin/bash -l
# Before an axion run: where to put its phi window, whether CLASS++'s sampling resolves its
# oscillations, and what one evaluation costs (smoke/axion_probe.py). True evaluations only, no
# network:
#
#   sbatch smoke/submit_axion_probe.sh axion_pp_high
#   sbatch --nodes=2 --ntasks=96 --time=04:00:00 smoke/submit_axion_probe.sh axion_wc_high    # H0WC: ~65 s per evaluation
#
# About 3200 evaluations of the phase probe (197 omegas x 8 phases x 2 amplitudes) and a dozen
# per setting of the sampling check's ladder (the last, converged one about a minute per point),
# on one q48 node. Output: logs/axion_probe_<job>.out and
# logs/axion_probe_<job>.json; phi0 then goes into the run's cobaya.yaml and domain_bounds.json
# on the laptop, and the kit is synced again before the run.
#
#SBATCH --job-name=axion_probe
#SBATCH --partition=q48
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/axion_probe_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ./env.sh
RUN=${1:?usage: sbatch smoke/submit_axion_probe.sh RUN [axion_probe.py options]}
shift
[ ! -f "runs/$RUN/job.env" ] || source "runs/$RUN/job.env"
unset CLIENT_REFUSED_VALUE                 # the probe wants the likelihood's own -inf
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMPI_MCA_mpi_yield_when_idle=1
export CLIENT_MPI_SCHEDULE=${CLIENT_MPI_SCHEDULE:-chunked}
export TF_CPP_MIN_LOG_LEVEL=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

echo "== $(date)  job ${SLURM_JOB_ID:-local} on $(hostname -s)  run=$RUN  $*"
# Under mpirun (2 November 2026): since Grendel's Slurm 26.05 sets SLURM_STEP_ID in a batch script, a Python that
# starts MPI by itself there aborts in MPI_Init_thread (Open MPI 4.0.3); one started by mpirun does not.
$MPI_LAUNCHER -n 1 python install/check.py --stage all --run "$RUN"
# -x: on a job of several nodes every rank must see the same PYTHONPATH and CLASS++ build variables (a run's job.env),
# as in submit_client.sh: a rank without them would import the env's classy.
$MPI_LAUNCHER -x CLIENT_MPI_SCHEDULE ${CLIENT_LOAD_GROUP:+-x CLIENT_LOAD_GROUP} \
  -x PYTHONPATH ${CLASSPP_GUARD_DIR:+-x CLASSPP_GUARD_DIR} ${CLASSPP_BUILD_DIR:+-x CLASSPP_BUILD_DIR -x CLASSPP_BUILD_COMMIT} -n "${SLURM_NTASKS:-1}" \
  python smoke/axion_probe.py "$RUN" "$@"
echo "== $(date)  done"
