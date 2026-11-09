#!/bin/bash -l
# Where does the noise of the SIDR + DRMD true likelihood come from? (23 October 2026)
#
#   sbatch smoke/submit_noise_variants.sh                    # all variants of smoke/noise_variants.json
#   sbatch smoke/submit_noise_variants.sh lcdm_massless      # or some of them
#
# smoke/noise_probe_cobaya.py measures, one variant after another, the noise of the true log-posterior along short lines
# through a point. All variants use the likelihoods of the Pantheon+ runs (Planck low-l TT and EE, CamSpec PR4 TTTEEE, PR4
# lensing, DESI DR2, Pantheon+), the patched CLASS++ of the runs (runs/sidr_drmd2_pp_combined_boost10/job.env), and that
# run's precision settings. Each step switches off one ingredient (results/20261023_noise_variants/make_variants.py):
#   lcdm_massless   LCDM, massless neutrinos                         (runs/noise_lcdm_massless/cobaya.yaml)
#   lcdm_mnu        LCDM, three standard massive neutrinos, 0.08 eV  (runs/noise_lcdm_mnu/cobaya.yaml)
#   sinu_off        the SIDR + DRMD code on its LCDM face: log10 Geff = -5.5, Delta Neff = f_idm = 0
#   sinu_strong     the SInu face in the strong mode (log10 Geff = -1.6), Delta Neff = f_idm = 0
#   full            the full model at its global best fit
# About 2100 true evaluations on one node; tens of minutes. Output: logs/noise_variants_<job>.out and
# logs/noise_probe_<label>_<job>.json.
#
#SBATCH --job-name=noise_variants
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/noise_variants_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ./env.sh
source runs/sidr_drmd2_pp_combined_boost10/job.env
[[ "$PYTHONPATH" == "$CLASSPP_GUARD_DIR/lib"* ]] || { echo "FATAL: the patched CLASS++ is not first on PYTHONPATH" >&2; exit 1; }
unset CLIENT_REFUSED_VALUE
# The worker processes are not MPI ranks: Cobaya is told not to initialise MPI in each of them.
export COBAYA_NOMPI=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
LABELS=("$@")
[ ${#LABELS[@]} -gt 0 ] || mapfile -t LABELS < <(python -c "import json; print('\n'.join(json.load(open('smoke/noise_variants.json'))['variants']))")
echo "== $(date)  job ${SLURM_JOB_ID:-local} on $(hostname -s)  variants: ${LABELS[*]}"
for label in "${LABELS[@]}"; do
  python smoke/noise_probe_cobaya.py smoke/noise_variants.json "$label" --workers 46
done
echo "== $(date)  done"
