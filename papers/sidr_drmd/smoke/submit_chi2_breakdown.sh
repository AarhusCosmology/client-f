#!/bin/bash -l
# chi^2 of every likelihood at the S+5 network best fits of the SIDR + DRMD combined networks (22 October 2026):
# full model, DRMD, SInu strong mode, SInu weak plateau (log10 Geff ~ -4) and LCDM, for Pantheon+ and for
# Pantheon+SH0ES, with the patched CLASS++ of the runs (runs/sidr_drmd2_<data>_combined_boost5/job.env):
#
#   sbatch smoke/submit_chi2_breakdown.sh                                            # the S+5 network best fits
#   sbatch smoke/submit_chi2_breakdown.sh smoke/chi2_points_boost10.json boost10     # the S+10 ones
#
# The second argument is the label whose runs/sidr_drmd2_<data>_combined_<label>/job.env is sourced.
#
# Points: smoke/chi2_points_s5.json (made on the laptop from the networks, results/20261022_s5_aic_preview).
# 10 true evaluations, a few minutes. Output: logs/chi2_breakdown_<job>.out and
# logs/chi2_breakdown_{pp,ppshoes}_<job>.json.
#
#SBATCH --job-name=chi2_breakdown
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/chi2_breakdown_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ./env.sh
POINTS=${1:-smoke/chi2_points_s5.json}
LABEL=${2:-boost5}
[ -f "$POINTS" ] || { echo "FATAL: no $POINTS (sync the kit)" >&2; exit 1; }
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
echo "== $(date)  job ${SLURM_JOB_ID:-local} on $(hostname -s)"
for data in pp ppshoes; do
  # each data set in its own shell: the run's job.env puts the patched CLASS++ first on PYTHONPATH
  ( source runs/sidr_drmd2_${data}_combined_${LABEL}/job.env
    [[ "$PYTHONPATH" == "$CLASSPP_GUARD_DIR/lib"* ]] || { echo "FATAL: the patched CLASS++ is not first on PYTHONPATH" >&2; exit 1; }
    python smoke/chi2_breakdown.py "$POINTS" "$data" )
done
echo "== $(date)  done"
