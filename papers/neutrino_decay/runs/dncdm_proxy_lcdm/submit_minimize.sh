#!/bin/bash -l
# True best fits of the LambdaCDM pair beside the DNCDM proxy (9 November 2026; make_inputs.py says what and why):
#
#   sbatch runs/dncdm_proxy_lcdm/submit_minimize.sh [--only pp_m005 pp_m0 ...]       # from this folder
#
# 4 inputs x 6 starts = 24 minimizations (Py-BOBYQA on Cobaya's log posterior, whitened by the CLiENT posterior's
# covariance: minimize_lcdm.py says why), two threads each, on one q48 node; 150-450 true evaluations a start.
# A resubmitted job reads the finished starts (logs/lcdm_minimize/<input>/start<k>.json) and runs the others again.
# Output: logs/lcdm_minimize_<job>.out and .json, and logs/lcdm_minimize/ (every start and its trace; logs/ is synced).
#
#SBATCH --job-name=lcdm_minimize
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/lcdm_minimize_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source ./env.sh
source runs/dncdm_proxy_pp/job.env          # CLASS++ cb2255f4, as both proxy runs (their job.env files are identical)
export COBAYA_NOMPI=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
echo "== $(date)  job ${SLURM_JOB_ID:-local} on $(hostname -s)  CLASS++ $CLASSPP_BUILD_COMMIT  $*"
python runs/dncdm_proxy_lcdm/minimize_lcdm.py --workers 24 "$@"
echo "== $(date)  done"
