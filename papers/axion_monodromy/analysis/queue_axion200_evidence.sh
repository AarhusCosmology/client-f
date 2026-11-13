#!/bin/bash
# The evidences of the omega <= 200 combined emulators that the paper's Bayes factors need (9 November 2026), as
# analysis/queue_lcdm_evidence.sh and the BEST seeds did for the omega <= 60 study. Copy it into the kit's analysis/ on
# Grendel and run it from the kit:
#
#   bash analysis/queue_axion200_evidence.sh            # both datasets
#   DRY=1 bash analysis/queue_axion200_evidence.sh      # print the sbatch commands only
#
# Per dataset, tasks of analysis/submit_evidence.sh on network 0 of the combined emulator (each roughly 40 minutes on
# 16 cores):
#   seed43, seed44, seed45                       the oscillation model, three more seeds beside the first run
#   lcdm, lcdm_seed43, lcdm_seed44, lcdm_seed45  the featureless spectrum, A_osc uniform on [0, 1e-4] (omega and phi drop
#                                                out of the likelihood there), four seeds
#   lcdm_part1 ... lcdm_partN                    the same with omega only inside one part of the combined network (between
#                                                its bands): all parts must give the same number (a test of the networks)
# Files: RUN/evidence/evidence_it_0_nlive2048_<tag>.json and their chains. A tag whose file exists is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
PP=results/20261107_084147_axion_pp_combined200       # seven parts
WC=results/20261107_084828_axion_wc_combined200       # eight parts
THIN="A_osc=0:1e-4"
submit() {                                             # run tag [client.py evidence arguments]
  local run=$1 tag=$2; shift 2
  [ -d "$run" ] || { echo "FATAL: no $run" >&2; exit 1; }
  if [ -e "$run/evidence/evidence_it_0_nlive2048_$tag.json" ]; then echo "skip $run $tag (done)"; return; fi
  local cmd=(sbatch --parsable --job-name="ev_$tag" --array=0 analysis/submit_evidence.sh "$run" --tag "$tag" "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[@]}"; else echo "$("${cmd[@]}")  $run $tag"; fi
}
for run in "$PP" "$WC"; do
  for seed in 43 44 45; do submit "$run" "seed$seed" --seed "$seed"; done
  submit "$run" lcdm --prior "$THIN"
  for seed in 43 44 45; do submit "$run" "lcdm_seed$seed" --seed "$seed" --prior "$THIN"; done
done
# omega inside each part, the bands left out
k=1; for r in 1:2.6 3.0:36.75 38.25:60.5 62.0:87.25 88.75:110.5 112.0:153.5 155.0:200; do
  submit "$PP" "lcdm_part$k" --prior "$THIN,omega=$r"; k=$((k + 1)); done
k=1; for r in 1:6.0 6.1:36.75 38.25:60.5 62.0:87.25 88.75:110.5 112.0:153.5 155.0:173.75 175.25:200; do
  submit "$WC" "lcdm_part$k" --prior "$THIN,omega=$r"; k=$((k + 1)); done
