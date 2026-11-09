#!/bin/bash -l
# The noise of the true objective at the scale of the profile stage's probes, and the best
# saved training values recomputed (true_profiles/noise_probe.py), on ONE q48 node:
#
#   sbatch true_profiles/submit_probe.sh results/<timestamp>_<RUN>
#
# About 1100 true evaluations for 49 parameters: a few minutes after the likelihoods are built.
# Run it before a profile study: it says whether the study's default tolerance can be met.
#
#SBATCH --job-name=profile_probe
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/probe_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ./env.sh

TARGET=${1:?usage: sbatch true_profiles/submit_probe.sh results/<timestamp>_<RUN> [noise_probe.py arguments]}
shift
RUN=$(basename "$TARGET" | sed -E 's/^[0-9]{8}_[0-9]{6}_//')
[ ! -f "runs/$RUN/job.env" ] || source "runs/$RUN/job.env"
unset CLIENT_REFUSED_VALUE
# The backend processes are not MPI ranks: Cobaya is told not to initialise MPI in each of them.
export COBAYA_NOMPI=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=1 PYTHONDONTWRITEBYTECODE=1
echo "== $(date)  job ${SLURM_JOB_ID:-local} on $(hostname -s)  noise probe of $TARGET $*  load group ${CLIENT_LOAD_GROUP:-all}  CAMB AccuracyTarget ${CLIENT_CAMB_ACCURACY_TARGET:-as the run}"
python true_profiles/noise_probe.py "$TARGET" --workers 46 "$@"
echo "== $(date)  done"
