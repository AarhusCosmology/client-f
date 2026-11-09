#!/bin/bash -l
# The World Cup runs' AIC points and the true chi^2 of every likelihood there (24 October 2026):
#
#   sbatch --dependency=afterok:<wc best_posterior job>:<wcmb best_posterior job> analysis/submit_aic_points.sh LABEL
#
# LABEL is a package of analysis/queue_sidr_drmd_wc_combined.sh (boost1, boost5, boost10); that script queues this job
# behind both data sets' pooled posteriors. What the second set's AIC table and per-likelihood table were made from,
# done here instead of on the laptop, of results/*_sidr_drmd_<wc|wcmb>_combined_<LABEL> network 0:
# 1. analysis/aic/aic_preview.py and analysis/aic/profile_logGeff.py, side by side: the network best fits on the four
#    faces (full, DRMD, SInu, LCDM; starts from the training points and the BEST samples) and the network profile in
#    log10 Geff, into logs/aic_wc_<LABEL>/ (aic_preview.json, profile_logGeff.json and .png, and their logs).
# 2. analysis/aic/make_points.py: smoke/chi2_points_wc_<LABEL>_bestfits.json, sets wc_bestfits and wcmb_bestfits
#    (full, DRMD, SInu_best, SInu_weak, LCDM, strong); analysis/aic/posterior_mean_points.py:
#    smoke/chi2_points_wc_<LABEL>_posterior_mean.json (needs analysis/best_posterior_it_0.npz). Both copied to
#    logs/aic_wc_<LABEL>/.
# 3. smoke/submit_chi2_points.sh on both files, in this job: logs/chi2_points_<data>_bestfits_<job>.json and
#    logs/chi2_points_<data>_posterior_mean_<job>.json, with truth minus network at each fit.
# In wcmb SH0ES is the M_B prior: aic_preview.py counts it in chi^2 (DATA_PRIORS); in the chi^2 job's output it is
# minus2_logprior_gaussian["Mb"], not a likelihood (add it to chi2_total for that data set's total chi^2).
#
#SBATCH --job-name=aic_points
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=logs/aic_points_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ./env.sh
LABEL=${1:?usage: sbatch analysis/submit_aic_points.sh LABEL (boost1, boost5 or boost10)}
for data in wc wcmb; do
  run=$(ls -d results/*_sidr_drmd_${data}_combined_${LABEL} 2>/dev/null | tail -1)
  [ -n "$run" ] || { echo "FATAL: no results/*_sidr_drmd_${data}_combined_${LABEL}" >&2; exit 1; }
  ls "$run"/evidence/evidence_it_0_nlive2048*.txt >/dev/null 2>&1 \
    || { echo "FATAL: $run has no BEST samples (evidence/evidence_it_0_nlive2048*.txt)" >&2; exit 1; }
  [ -f "$run/analysis/best_posterior_it_0.npz" ] || { echo "FATAL: no $run/analysis/best_posterior_it_0.npz" >&2; exit 1; }
done
OUT=logs/aic_wc_$LABEL
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_NUM_INTRAOP_THREADS=8
echo "== $(date)  job ${SLURM_JOB_ID:-local} on $(hostname -s)  AIC points of the $LABEL combined networks -> $OUT"
python analysis/aic/aic_preview.py --series wc --label "$LABEL" --out "$OUT" > "$OUT/aic_preview.log" 2>&1 &
A=$!
python analysis/aic/profile_logGeff.py --series wc --label "$LABEL" --out "$OUT" > "$OUT/profile_logGeff.log" 2>&1 &
P=$!
wait $A || { echo "FATAL: aic_preview.py failed (see $OUT/aic_preview.log)" >&2; exit 1; }
wait $P || { echo "FATAL: profile_logGeff.py failed (see $OUT/profile_logGeff.log)" >&2; exit 1; }
grep -E "^(wc|wcmb) " "$OUT/aic_preview.log"
python analysis/aic/make_points.py --series wc --label "$LABEL" --out "$OUT" 2>&1 | grep -E "^(wc|wcmb|written)"
python analysis/aic/posterior_mean_points.py --series wc --label "$LABEL" 2>&1 | grep -E "^(wc|wcmb|written)"
POINTS=(smoke/chi2_points_wc_${LABEL}_bestfits.json smoke/chi2_points_wc_${LABEL}_posterior_mean.json)
cp "${POINTS[@]}" "$OUT/"
echo "== $(date)  network fits done; true chi^2 per likelihood"
for points in "${POINTS[@]}"; do
  bash smoke/submit_chi2_points.sh "$points"
done
echo "== $(date)  done"
