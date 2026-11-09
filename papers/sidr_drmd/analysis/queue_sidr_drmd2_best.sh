#!/bin/bash
# BEST seeds 43, 44 and 45 and the posterior from BEST's samples of the SIDR + DRMD combined networks
# (22 October 2026), through analysis/queue_best_seeds.sh:
#
#   bash analysis/queue_sidr_drmd2_best.sh                         # all six
#   bash analysis/queue_sidr_drmd2_best.sh pp_boost1 pp_boost5     # some of them
#
# Why: the report's chain of a combined network mixes per-cell chains by bridge-sampling evidences that
# are marked as not reliable, and on pp S+1 BEST's samples put 37% of the posterior in the strong SIDR
# mode where the chain put 51%. One BEST run's share of a mode is uncertain by roughly sqrt(H / n_k) in
# log, so the seeds measure that scatter and the pool halves it. Network only: no true evaluation.
# The S+10 seeds wait for their combine jobs; every posterior job also waits for the first evidence
# (seed 42) of its network while that is in the queue.
set -euo pipefail
cd "$(dirname "$0")/.."
# name:run directory:first evidence job:combine job (only for a network not written yet). The boost10
# directories are named by queue_sidr_drmd2_combined.sh at queue time (one timestamp for both data sets:
# 20261022_081631); the first version of this file had another, and its S+10 seeds stopped at once.
ENTRIES="pp_boost1:results/20261022_081615_sidr_drmd2_pp_combined_boost1:6933458:
ppshoes_boost1:results/20261022_081615_sidr_drmd2_ppshoes_combined_boost1:6933464:
pp_boost5:results/20261022_081623_sidr_drmd2_pp_combined_boost5:6933470:
ppshoes_boost5:results/20261022_081623_sidr_drmd2_ppshoes_combined_boost5:6933476:
pp_boost10:results/20261022_081631_sidr_drmd2_pp_combined_boost10:6933484:6933483
ppshoes_boost10:results/20261022_081631_sidr_drmd2_ppshoes_combined_boost10:6933490:6933489"
WANTED=" ${*:-pp_boost1 ppshoes_boost1 pp_boost5 ppshoes_boost5 pp_boost10 ppshoes_boost10} "
for name in $WANTED; do
  echo "$ENTRIES" | grep -q "^$name:" || { echo "FATAL: no entry $name (one of: $(echo "$ENTRIES" | cut -d: -f1 | tr '\n' ' '))" >&2; exit 1; }
done
echo "$ENTRIES" | while IFS=: read -r name out evidence combine; do
  case "$WANTED" in *" $name "*) ;; *) continue ;; esac
  # wait for the combine only while it is in the queue (a finished job is no dependency Slurm accepts);
  # without it queue_best_seeds.sh requires the network to be there
  after=""
  if [ -n "$combine" ] && squeue -h -j "$combine" 2>/dev/null | grep -q .; then after=$combine; fi
  bash analysis/queue_best_seeds.sh "$out" --evidence "$evidence" ${after:+--after "$after"} \
    -- --split logGeff:-2.5 --split delta_Neff_drmd:0.1
done
