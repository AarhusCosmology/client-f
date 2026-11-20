#!/bin/bash -l
# chi^2 of every likelihood along the true log10Gamma profiles of the DNCDM proxy (9 November 2026; node_breakdown.py
# says what and why):
#
#   sbatch runs/dncdm_proxy_lcdm/submit_breakdown.sh [--only pp wc]       # from this folder
#
# 91 nodes a data set (the full grid outside [11.5, 13.75] and the peak grid), one true evaluation each, two threads
# each, on one q48 node: the peak nodes take a few minutes, the others seconds; well under an hour for both.
# Output: logs/node_breakdown_<job>.out and .json (logs/ is synced).
#
#SBATCH --job-name=node_breakdown
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/node_breakdown_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source ./env.sh
source runs/dncdm_proxy_pp/job.env          # CLASS++ cb2255f4, as both proxy runs (their job.env files are identical)
export COBAYA_NOMPI=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
echo "== $(date)  job ${SLURM_JOB_ID:-local} on $(hostname -s)  CLASS++ $CLASSPP_BUILD_COMMIT  $*"
python runs/dncdm_proxy_lcdm/node_breakdown.py --workers 24 "$@"
echo "== $(date)  done"
