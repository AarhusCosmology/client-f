#!/bin/bash
# The evidence of the featureless spectrum (LambdaCDM with running) from the axion combined emulators, as the limit
# A_osc -> 0 (2 November 2026). `client.py evidence --prior A_osc=0:1e-4` integrates the network with A_osc uniform on
# [0, 1e-4], normalised; the network barely changes over that range, and omega and phi drop out of the likelihood at
# A_osc = 0, so the result is ln Z of the model without the oscillation, with the same priors on every other parameter.
# The Bayes factor of the oscillation is then ln Z(oscillation) - ln Z(this), both in Cobaya's convention (the paper adds
# ln(0.5 / 0.1) to the first for its prior A_osc in [0, 0.1]).
#
#   bash analysis/queue_lcdm_evidence.sh            # both datasets
#   DRY=1 bash analysis/queue_lcdm_evidence.sh      # print the sbatch commands only
#
# Per dataset, six tasks of analysis/submit_evidence.sh on network 0 (each about 40 minutes on 16 cores):
#   lcdm, lcdm_seed43, lcdm_seed44, lcdm_seed45   omega and phi over their whole prior, four seeds as for the oscillation
#   lcdm_low, lcdm_high                           omega only where the low or the high part of the combined network is
#                                                 used: the two parts must give the same number (a test of the networks)
# Files: RUN/evidence/evidence_it_0_nlive2048_lcdm*.json and their chains. A tag whose file exists is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
PP=results/20261029_074412_axion_pp_combined          # low part used for omega < 2.6 (blend 2.6:3.0)
WC=results/20261031_092109_axion_wc_combined          # low part used for omega < 6.0 (blend 6.0:6.1)
THIN="A_osc=0:1e-4"
submit() {                                             # run tag [client.py evidence arguments]
  local run=$1 tag=$2; shift 2
  [ -d "$run" ] || { echo "FATAL: no $run" >&2; exit 1; }
  if [ -e "$run/evidence/evidence_it_0_nlive2048_$tag.json" ]; then echo "skip $run $tag (done)"; return; fi
  local cmd=(sbatch --parsable --job-name="ev_$tag" --array=0 analysis/submit_evidence.sh "$run" --tag "$tag" "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[@]}"; else echo "$("${cmd[@]}")  $run $tag"; fi
}
for run in "$PP" "$WC"; do
  submit "$run" lcdm --prior "$THIN"
  for seed in 43 44 45; do submit "$run" "lcdm_seed$seed" --seed "$seed" --prior "$THIN"; done
  submit "$run" lcdm_high --prior "$THIN,omega=10:59"
done
submit "$PP" lcdm_low --prior "$THIN,omega=1.2:2.4"
submit "$WC" lcdm_low --prior "$THIN,omega=1.2:5.0"
