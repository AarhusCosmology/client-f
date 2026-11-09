#!/bin/bash
# More BEST seeds of one network, then its posterior from BEST's samples (22 October 2026):
#
#   bash analysis/queue_best_seeds.sh results/<time>_<RUN> [--network 0] [--seeds "43 44 45"]
#                                     [--after JOB] [--evidence JOB] [-- best_posterior.py arguments]
#
# Each seed is one task of analysis/submit_evidence.sh with `--seed S --tag seedS`: the same target,
# its own files beside the first run (evidence_it_N_nliveM_seedS.*). A seed whose files exist is
# skipped. Then one small job runs analysis/best_posterior.py (the posterior pooled over every run of
# the target, the shares of modes and region cells with their scatter over seeds, the corner plot)
# and analysis/evidence_summary.py (which lists the seeds beside the first run).
#   --after JOB     the seeds wait for that job: the combine that writes the network
#   --evidence JOB  the first evidence job (seed 42): the posterior job waits for it too, if it is
#                   still in the queue
# Dependencies are afterany: a seed that fails leaves the posterior to the runs that finished, and
# best_posterior.py lists which those are.
set -euo pipefail
cd "$(dirname "$0")/.."
USAGE='usage: bash analysis/queue_best_seeds.sh results/<time>_<RUN> [--network N] [--seeds "43 44 45"] [--after JOB] [--evidence JOB] [-- best_posterior.py arguments]'
OUT=${1:?$USAGE}
shift
NETWORK=0 SEEDS="43 44 45" AFTER="" EVIDENCE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --network) NETWORK=${2:?$USAGE}; shift 2 ;;
    --seeds) SEEDS=${2:?$USAGE}; shift 2 ;;
    --after) AFTER=${2:?$USAGE}; shift 2 ;;
    --evidence) EVIDENCE=${2:?$USAGE}; shift 2 ;;
    --) shift; break ;;
    *) echo "FATAL: unknown argument '$1'. $USAGE" >&2; exit 1 ;;
  esac
done
for number in "$NETWORK" $SEEDS ${AFTER:+"$AFTER"} ${EVIDENCE:+"$EVIDENCE"}; do
  [[ "$number" =~ ^[0-9]+$ ]] || { echo "FATAL: '$number' is not a number" >&2; exit 1; }
done
if [ -z "$AFTER" ]; then
  [ -f "$OUT/trained_models/model_it_$NETWORK.keras" ] \
    || { echo "FATAL: $OUT has no network $NETWORK (give --after JOB when a job is still writing it)" >&2; exit 1; }
fi

SEED_JOBS=""
for seed in $SEEDS; do
  if ls "$OUT"/evidence/evidence_it_${NETWORK}_nlive*_seed$seed.json >/dev/null 2>&1; then
    echo "$OUT: seed $seed of network $NETWORK is there already; not submitted again"
    continue
  fi
  J=$(sbatch --parsable ${AFTER:+--dependency=afterok:$AFTER} --array=$NETWORK analysis/submit_evidence.sh "$OUT" \
        --seed $seed --tag seed$seed)
  SEED_JOBS="$SEED_JOBS:${J%%;*}"
done

DEP=""
[ -z "$SEED_JOBS" ] || DEP="afterany$SEED_JOBS"
if [ -n "$EVIDENCE" ] && squeue -h -j "$EVIDENCE" 2>/dev/null | grep -q .; then
  DEP="${DEP:+$DEP,}afterany:$EVIDENCE"
fi
P=$(sbatch --parsable ${DEP:+--dependency=$DEP} --job-name=best_posterior --partition=q48 --nodes=1 --ntasks=1 \
      --cpus-per-task=2 --mem=8G --time=00:30:00 --output=logs/best_posterior_%j.out \
      --wrap "source ./env.sh && export MPLBACKEND=Agg MPLCONFIGDIR=\${TMPDIR:-/tmp}/mpl_\$USER && python analysis/best_posterior.py $OUT --network $NETWORK $* && python \$CLIENT_REPO/hpc/analysis/evidence_summary.py $OUT")
echo "$OUT: seeds${SEED_JOBS//:/ } (${SEEDS// /, }${AFTER:+, after job $AFTER}), posterior and summary ${P%%;*}${DEP:+ ($DEP)}"
