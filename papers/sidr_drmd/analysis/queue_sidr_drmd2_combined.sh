#!/bin/bash
# The SIDR + DRMD second set's combined emulators and their analysis (22 October 2026), one package per call:
#
#   bash analysis/queue_sidr_drmd2_combined.sh boost1 1           # networks S+1: the first trained on boost points
#   bash analysis/queue_sidr_drmd2_combined.sh boost5 5           # networks S+5: after five boosts
#   bash analysis/queue_sidr_drmd2_combined.sh boost10 10 wait    # the final networks S+10, after the runs end
#
# S is each part's hand-over (its boost/start.json): pp_low 9, pp_high 7, ppshoes_low 8, ppshoes_high 8. For
# each data set (pp, ppshoes):
#   combine     client.py combine of the low part's network S+OFFSET and the high part's (band 0.08 to 0.12 in
#               delta_Neff), into results/<time>_sidr_drmd2_<data>_combined_<LABEL>; at once, or with "wait"
#               after both parts' client jobs (the networks S+OFFSET must exist by then)
#   evidence    of the combined network 0 (after the combine)
#   report      its posterior samples, parameter table and a corner plot of the 12 cosmological parameters
#   true check  500 of its posterior points, with the patched CLASS++ (runs/sidr_drmd2_<data>_combined_<LABEL>/job.env)
#   summary     analysis/evidence_summary.py (after the evidence and the true check)
#   profiles    true 1D profiles of the 12 cosmological parameters from network 0 (after the report, whose chain
#               gives the ranges: +-3 sigma, 25 nodes each, hybrid_hops 30), into
#               results_profiles/sidr_drmd2_<data>_combined_<LABEL>_cosmo/; a full node, up to 72 h
# The four combinations of boost1 and boost5 were tried with client.py combine on the laptop (22 October:
# 8, 10, 5 and 5 region cells, every training point taken). A job whose dependency fails stays pending
# ("DependencyNeverSatisfied"): scancel it.
set -euo pipefail
cd "$(dirname "$0")/.."
LABEL=${1:?usage: bash analysis/queue_sidr_drmd2_combined.sh LABEL OFFSET [wait]}
OFFSET=${2:?usage: bash analysis/queue_sidr_drmd2_combined.sh LABEL OFFSET [wait]}
WAIT=${3:-}
[[ "$OFFSET" =~ ^[0-9]+$ ]] || { echo "FATAL: OFFSET must be a number of boosts, not '$OFFSET'" >&2; exit 1; }
[ -z "$WAIT" ] || [ "$WAIT" = wait ] || { echo "FATAL: the third argument is 'wait' or nothing, not '$WAIT'" >&2; exit 1; }

# part:client-job:S (S read from each run's boost/start.json, 22 October 2026)
PARTS="pp_low:6927875:9 pp_high:6927876:7 ppshoes_low:6927877:8 ppshoes_high:6927878:8"
TS=$(date +%Y%m%d_%H%M%S)
PROFILED="omega_b omega_cdm H0 logA n_s tau_reio m_int logGeff delta_Neff_drmd f_idm_drmd log10_zstop log10_G_over_aH"

field() { for p in $PARTS; do [ "${p%%:*}" = "$1" ] && { echo "$p" | cut -d: -f"$2"; return; }; done; echo "FATAL: no part $1" >&2; exit 1; }
run_dir() {
  local found
  found=$(ls -d results/*_sidr_drmd2_"$1" 2>/dev/null | tail -1)
  [ -n "$found" ] || { echo "FATAL: no results/*_sidr_drmd2_$1" >&2; exit 1; }
  echo "$found"
}

for data in pp ppshoes; do
  NAME=sidr_drmd2_${data}_combined_${LABEL}
  OUT=results/${TS}_${NAME}
  [ -f runs/$NAME/cobaya.yaml ] && [ -f runs/$NAME/domain_bounds.json ] && [ -f runs/$NAME/job.env ] \
    || { echo "FATAL: runs/$NAME is missing its cobaya.yaml, domain_bounds.json or job.env (sync the kit)" >&2; exit 1; }
  PART_ARGS=""; JOBS=""
  for side in low high; do
    part=${data}_${side}; dir=$(run_dir $part); s=$(field $part 3); job=$(field $part 2)
    recorded=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['start_iteration'])" "$dir/boost/start.json")
    [ "$recorded" = "$s" ] || { echo "FATAL: $dir hands over at $recorded, this script says $s" >&2; exit 1; }
    n=$((s + OFFSET))
    if [ -z "$WAIT" ] && [ ! -f "$dir/trained_models/model_it_$n.keras" ]; then
      echo "FATAL: $dir has no network $n yet: queue this package with 'wait'" >&2; exit 1
    fi
    PART_ARGS="$PART_ARGS --part $dir:$n"; JOBS="$JOBS:$job"
  done
  DEP=""; [ -z "$WAIT" ] || DEP="--dependency=afterok$JOBS"
  C=$(sbatch --parsable $DEP --job-name=combine_${data} --partition=q48 --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --mem=32G --time=02:00:00 --output=logs/combine_%j.out \
        --wrap "source ./env.sh && python \$CLIENT_REPO/client.py combine $OUT --parameter delta_Neff_drmd$PART_ARGS --band 0.08:0.12")
  E=$(sbatch --parsable --dependency=afterok:${C%%;*} --array=0 analysis/submit_evidence.sh "$OUT")
  R=$(sbatch --parsable --dependency=afterok:${C%%;*} analysis/submit_report.sh "$OUT" --networks 0 --corner 0 --corner-parameters 12)
  T=$(sbatch --parsable --time=12:00:00 --dependency=afterok:${R%%;*} analysis/submit_true_check.sh "$OUT" --networks 0 --points 500)
  M=$(sbatch --parsable --dependency=afterok:${E%%;*}:${T%%;*} --job-name=ev_summary --partition=q48 --nodes=1 \
        --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:30:00 --output=logs/ev_summary_%j.out \
        --wrap "source ./env.sh && python \$CLIENT_REPO/hpc/analysis/evidence_summary.py $OUT")
  P=$(sbatch --parsable --dependency=afterok:${R%%;*} true_profiles/submit_profiles_latest.sh "$OUT" ${NAME}_cosmo \
        --parameters $PROFILED --points 25 --hybrid-hops 30)
  echo "$OUT ($PART_ARGS ): combine ${C%%;*}${WAIT:+ (after the client jobs${JOBS//:/ })}, evidence ${E%%;*}, report ${R%%;*}, true check ${T%%;*}, summary ${M%%;*}, profiles ${P%%;*}"
done
