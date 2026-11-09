#!/bin/bash
# The SIDR + DRMD runs on the H0 World Cup data: their combined emulators and the paper's full analysis, one package
# per call (24 October 2026; analysis/queue_sidr_drmd2_combined.sh and analysis/queue_sidr_drmd2_best.sh of the second
# set, together, with what the second set needed afterwards built in):
#
#   bash analysis/queue_sidr_drmd_wc_combined.sh boost1 1           # networks S+1: the first trained on boost points
#   bash analysis/queue_sidr_drmd_wc_combined.sh boost5 5           # networks S+5
#   bash analysis/queue_sidr_drmd_wc_combined.sh boost10 10 wait    # the final networks S+10 (wait: after the runs end)
#
# S is each part's hand-over (its boost/start.json): wc_low 7, wc_high 7, wcmb_low 6, wcmb_high 7. For each data set
# (wc: the World Cup baseline; wcmb: with SH0ES as the M_B prior):
#   combine     client.py combine of the low part's network S+OFFSET and the high part's (band 0.08 to 0.12 in
#               delta_Neff), into results/<time>_sidr_drmd_<data>_combined_<LABEL>; at once, or with "wait" after both
#               parts' client jobs
#   evidence    BEST, seed 42, of the combined network 0
#   seeds       BEST seeds 43, 44 and 45, and the posterior from all four pooled (analysis/queue_best_seeds.sh,
#               analysis/best_posterior.py; the paper's posteriors), split by log10 Geff at -2.5 and delta_Neff at 0.1
#   report      posterior samples, parameter table, corner plot of the 12 cosmological parameters
#   true check  500 posterior points with the runs' CLASS++ (runs/sidr_drmd_<data>_combined_<LABEL>/job.env)
#   summary     analysis/evidence_summary.py
# and, once per package, after both data sets' pooled posteriors:
#   AIC points  analysis/submit_aic_points.sh: the network best fits of the AIC table (full, DRMD, SInu, LCDM, the
#               weak SInu node and the strong mode), the posterior mean, and the true chi^2 of every likelihood there
# and, for boost10 only (the paper's profiles are of the final networks):
#   profiles    three true-profile studies from network 0, each on a node of its own, up to 72 h:
#                 _cosmo         the 12 cosmological parameters, 25 nodes each over +-3 sigma of the report's chain
#                 _fidm_wide     f_idm_drmd over its whole prior [0, 0.4] (the second set's SH0ES +-3 sigma reached
#                                only 0.064)
#                 _logGeff_wide  log10 Geff over its whole prior [-5.5, -0.8] with the region's edge at 2 reaches:
#                                the boost regions have mostly dropped the strong mode (24 October), and the comparison
#                                with the Planck-only runs needs the profile there
#               all with --hybrid-hops 30 and --gtol 1e-3 (the second set's pp study needed both)
#   breakdown   after the three: the studies' files copied to logs/ and every node split by likelihood
#               (analysis/submit_profile_breakdown.sh; set <data>_profiles)
# A job whose dependency fails stays pending ("DependencyNeverSatisfied"): scancel it.
#
# Options in the environment (25 October 2026, after wc_low stopped in its eighth boost and took the wc packages with it):
#   DATA_SETS="wc"         queue only those data sets (default "wc wcmb")
#   JOB_wc_low=<job>       a part's client job, when it was continued in another job than PARTS names (for "wait")
#   AIC_ALSO_AFTER=<jobs>  more jobs the AIC job waits for (afterok, colon-separated): the other data set's pooled
#                          posterior when only one data set is queued here, since the AIC job does both
set -euo pipefail
cd "$(dirname "$0")/.."
LABEL=${1:?usage: bash analysis/queue_sidr_drmd_wc_combined.sh LABEL OFFSET [wait]}
OFFSET=${2:?usage: bash analysis/queue_sidr_drmd_wc_combined.sh LABEL OFFSET [wait]}
WAIT=${3:-}
[[ "$OFFSET" =~ ^[0-9]+$ ]] || { echo "FATAL: OFFSET must be a number of boosts, not '$OFFSET'" >&2; exit 1; }
[ -z "$WAIT" ] || [ "$WAIT" = wait ] || { echo "FATAL: the third argument is 'wait' or nothing, not '$WAIT'" >&2; exit 1; }

# part:client-job:S (S read from each run's boost/start.json, 24 October 2026)
PARTS="wc_low:6950858:7 wc_high:6950859:7 wcmb_low:6950860:6 wcmb_high:6950862:7"
TS=$(date +%Y%m%d_%H%M%S)
PROFILED="omega_b omega_cdm H0 logA n_s tau_reio m_int logGeff delta_Neff_drmd f_idm_drmd log10_zstop log10_G_over_aH"
SCAN="--points 25 --hybrid-hops 30 --gtol 1e-3"

field() { for p in $PARTS; do [ "${p%%:*}" = "$1" ] && { echo "$p" | cut -d: -f"$2"; return; }; done; echo "FATAL: no part $1" >&2; exit 1; }
run_dir() {
  local found
  found=$(ls -d results/*_sidr_drmd_"$1" 2>/dev/null | tail -1)
  [ -n "$found" ] || { echo "FATAL: no results/*_sidr_drmd_$1" >&2; exit 1; }
  echo "$found"
}

AIC_AFTER="" AIC_SEED42=""
DATA_SETS=${DATA_SETS:-wc wcmb}
for data in $DATA_SETS; do
  [ "$data" = wc ] || [ "$data" = wcmb ] || { echo "FATAL: DATA_SETS holds '$data' (wc, wcmb)" >&2; exit 1; }
done
[ -z "${AIC_ALSO_AFTER:-}" ] || [[ "$AIC_ALSO_AFTER" =~ ^[0-9]+(:[0-9]+)*$ ]] \
  || { echo "FATAL: AIC_ALSO_AFTER is job numbers separated by colons, not '$AIC_ALSO_AFTER'" >&2; exit 1; }
for data in $DATA_SETS; do
  NAME=sidr_drmd_${data}_combined_${LABEL}
  OUT=results/${TS}_${NAME}
  [ -f runs/$NAME/cobaya.yaml ] && [ -f runs/$NAME/domain_bounds.json ] && [ -f runs/$NAME/job.env ] \
    || { echo "FATAL: runs/$NAME is missing its cobaya.yaml, domain_bounds.json or job.env (sync the kit)" >&2; exit 1; }
  PART_ARGS=""; JOBS=""
  for side in low high; do
    part=${data}_${side}; dir=$(run_dir $part); s=$(field $part 3); job=$(field $part 2)
    override=JOB_$part; job=${!override:-$job}
    [[ "$job" =~ ^[0-9]+$ ]] || { echo "FATAL: $override is a job number, not '$job'" >&2; exit 1; }
    recorded=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['start_iteration'])" "$dir/boost/start.json")
    [ "$recorded" = "$s" ] || { echo "FATAL: $dir hands over at $recorded, this script says $s" >&2; exit 1; }
    n=$((s + OFFSET))
    if [ -z "$WAIT" ] && [ ! -f "$dir/trained_models/model_it_$n.keras" ]; then
      echo "FATAL: $dir has no network $n yet: queue this package with 'wait'" >&2; exit 1
    fi
    PART_ARGS="$PART_ARGS --part $dir:$n"; JOBS="$JOBS:$job"
  done
  DEP=""
  if [ -n "$WAIT" ]; then
    # only the client jobs still in the queue: Slurm refuses a dependency on a job that has left it
    live=""; for j in ${JOBS//:/ }; do squeue -h -j "$j" 2>/dev/null | grep -q . && live="$live:$j"; done
    [ -z "$live" ] || DEP="--dependency=afterok$live"
  fi
  C=$(sbatch --parsable $DEP --job-name=combine_${data} --partition=q48 --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --mem=32G --time=02:00:00 --output=logs/combine_%j.out \
        --wrap "source ./env.sh && python \$CLIENT_REPO/client.py combine $OUT --parameter delta_Neff_drmd$PART_ARGS --band 0.08:0.12")
  C=${C%%;*}
  E=$(sbatch --parsable --dependency=afterok:$C --array=0 analysis/submit_evidence.sh "$OUT"); E=${E%%;*}
  B=$(bash analysis/queue_best_seeds.sh "$OUT" --after "$C" --evidence "$E" -- --split logGeff:-2.5 --split delta_Neff_drmd:0.1)
  R=$(sbatch --parsable --dependency=afterok:$C analysis/submit_report.sh "$OUT" --networks 0 --corner 0 --corner-parameters 12); R=${R%%;*}
  T=$(sbatch --parsable --time=12:00:00 --dependency=afterok:$R analysis/submit_true_check.sh "$OUT" --networks 0 --points 500); T=${T%%;*}
  M=$(sbatch --parsable --dependency=afterok:$E:$T --job-name=ev_summary --partition=q48 --nodes=1 \
        --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:30:00 --output=logs/ev_summary_%j.out \
        --wrap "source ./env.sh && python \$CLIENT_REPO/hpc/analysis/evidence_summary.py $OUT"); M=${M%%;*}
  echo "$OUT ($PART_ARGS ): combine $C${DEP:+ ($DEP)}, evidence $E, report $R, true check $T, summary $M"
  echo "   $B"
  BP=$(echo "$B" | sed -nE 's/.*posterior and summary ([0-9]+).*/\1/p')
  [ -n "$BP" ] || { echo "FATAL: no pooled-posterior job in the line above" >&2; exit 1; }
  AIC_AFTER="$AIC_AFTER:$BP" AIC_SEED42="$AIC_SEED42:$E"   # the pooled posterior; seed 42 too (the AIC starts read every seed)
  if [ "$LABEL" = boost10 ]; then
    P1=$(sbatch --parsable --dependency=afterok:$R true_profiles/submit_profiles_latest.sh "$OUT" ${NAME}_cosmo \
          --parameters $PROFILED $SCAN); P1=${P1%%;*}
    P2=$(sbatch --parsable --dependency=afterok:$R true_profiles/submit_profiles_latest.sh "$OUT" ${NAME}_fidm_wide \
          --parameters f_idm_drmd --range f_idm_drmd 0 0.4 $SCAN); P2=${P2%%;*}
    P3=$(sbatch --parsable --dependency=afterok:$R true_profiles/submit_profiles_latest.sh "$OUT" ${NAME}_logGeff_wide \
          --parameters logGeff --range logGeff -5.5 -0.8 --region 2 $SCAN); P3=${P3%%;*}
    K=$(sbatch --parsable --dependency=afterany:$P1:$P2:$P3 analysis/submit_profile_breakdown.sh "$OUT" "$NAME" \
          ${data}_profiles ${NAME}_cosmo ${NAME}_fidm_wide ${NAME}_logGeff_wide); K=${K%%;*}
    echo "   profiles $P1 (cosmo), $P2 (f_idm whole prior), $P3 (log10 Geff whole prior, edge 2); breakdown $K"
  fi
done
AIC_AFTER="$AIC_AFTER${AIC_ALSO_AFTER:+:$AIC_ALSO_AFTER}"
A=$(sbatch --parsable --dependency=afterok$AIC_AFTER,afterany$AIC_SEED42 analysis/submit_aic_points.sh "$LABEL"); A=${A%%;*}
echo "AIC points and their chi^2 per likelihood (both data sets): $A (afterok$AIC_AFTER,afterany$AIC_SEED42)"
