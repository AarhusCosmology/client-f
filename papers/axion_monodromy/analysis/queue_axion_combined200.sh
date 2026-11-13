#!/bin/bash
# The combined emulators for omega <= 200 and their analysis (6 November 2026), once their parts have ended:
#
#   bash analysis/queue_axion_combined200.sh pp                          # PDP+: the low part and five zones, on q48
#   bash analysis/queue_axion_combined200.sh wc                          # H0WC: the low part and six zones
#   AFTER=7032852:7032853 bash analysis/queue_axion_combined200.sh wc    # the combine waits for these jobs to end well
#   PARTITION=q40 bash analysis/queue_axion_combined200.sh pp            # on q40 (the true check on 40 ranks)
#
# Each part ends with network S + 10, S its own hand-over (boost/start.json). It submits, as
# analysis/queue_axion_pp_combined.sh did for the omega <= 60 study:
#   combine     client.py combine of the parts' final networks, blended over the bands of
#               runs/axion_<DATA>_combined200/client.yaml (make_combined200.py), into results/<time>_axion_<DATA>_combined200
#   evidence    of the combined network 0
#   report      its posterior samples (each region cell sampled on its own and weighted by its evidence), the parameter
#               table and a corner plot of the 10 cosmological and axion parameters
#   true check  500 of its posterior points, with runs/axion_<DATA>_combined200/job.env (CLASS++ 0a57bbc6)
#   summary     analysis/evidence_summary.py (after the evidence and the true check)
#   profiles    one true_profiles/submit_profiles.sh job (one q48 node each) per runs/axion_<DATA>_combined200/profiles_*.yaml,
#               after the combine, into results_profiles/axion_<DATA>200_<name>; for PDP+ the nine of make_profiles200.py
#               (four omega windows of (A_osc, omega) and of (omega, phi), and (A_osc, phi)). PROFILES=0 leaves them out.
#               A study whose node optimum lies outside the region stops after its scan, before any true call. A study
#               whose results_profiles/ folder exists (an earlier combine's) is not submitted; PROFILES=0 skips all.
# The PDP+ combine was tried on the laptop on 6 November with the code of d3dfcbe: six region cells, and the combined
# network equal to each part's own at 3000 of its boost points in its range; the nine profile studies passed
# client.py profiles --dry-run on it. A job whose dependency fails stays pending ("DependencyNeverSatisfied"): scancel it.
set -euo pipefail
cd "$(dirname "$0")/.."
USAGE="usage: bash analysis/queue_axion_combined200.sh pp|wc"
DATA=${1:?$USAGE}
case "$DATA" in
  # zone 1 of both split at omega = 37.5 on 6 November (make_runs_zone1_split.py); the PDP+ combine of that morning
  # (results/20261106_090834_axion_pp_combined200) had zone 1 whole
  pp) PARTS="axion_pp_low200 axion_pp_zone1a axion_pp_zone1b axion_pp_zone2a axion_pp_zone2b axion_pp_zone3 axion_pp_zone4"
      BANDS="2.6:3.0 36.75:38.25 60.5:62.0 87.25:88.75 110.5:112.0 153.5:155.0" ;;
  wc) PARTS="axion_wc_low200 axion_wc_zone1a axion_wc_zone1b axion_wc_zone2a axion_wc_zone2b axion_wc_zone3 axion_wc_zone4a axion_wc_zone4b"
      BANDS="6.0:6.1 36.75:38.25 60.5:62.0 87.25:88.75 110.5:112.0 153.5:155.0 173.75:175.25" ;;
  *) echo "FATAL: $USAGE" >&2; exit 1 ;;
esac
PARTITION=${PARTITION:-q48}
case "$PARTITION" in
  q48) RANKS=48 ;;
  q40) RANKS=40 ;;
  *) echo "FATAL: PARTITION is q48 or q40, not '$PARTITION'" >&2; exit 1 ;;
esac
AFTER=${AFTER:-}
[[ -z "$AFTER" || "$AFTER" =~ ^[0-9]+(:[0-9]+)*$ ]] || { echo "FATAL: AFTER is Slurm job ids separated by ':', not '$AFTER'" >&2; exit 1; }
BOOSTS=10

NAME=axion_${DATA}_combined200
OUT=results/$(date +%Y%m%d_%H%M%S)_$NAME
[ -f runs/$NAME/cobaya.yaml ] && [ -f runs/$NAME/domain_bounds.json ] && [ -f runs/$NAME/job.env ] \
  || { echo "FATAL: runs/$NAME is missing its cobaya.yaml, domain_bounds.json or job.env (sync the kit)" >&2; exit 1; }
BAND_ARGS=$(for b in $BANDS; do printf -- "--band %s " "$b"; done)
grep -qF -- "${BAND_ARGS% }" runs/$NAME/client.yaml \
  || { echo "FATAL: the bands here are not those of runs/$NAME/client.yaml" >&2; exit 1; }

PART_ARGS=""
for part in $PARTS; do
  dirs=$(ls -d results/2026*_"$part" 2>/dev/null || true)
  [ "$(echo "$dirs" | grep -c .)" = 1 ] || { echo "FATAL: not exactly one results/*_$part: ${dirs:-none}" >&2; exit 1; }
  dir=$dirs
  [ -f "$dir/boost/start.json" ] || { echo "FATAL: $dir has not reached the boosts (no boost/start.json)" >&2; exit 1; }
  start=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['start_iteration'])" "$dir/boost/start.json")
  last=$((start + BOOSTS))
  cmp -s "$dir/likelihood_input/cobaya.yaml" runs/$NAME/cobaya.yaml \
    || { echo "FATAL: $dir/likelihood_input/cobaya.yaml is not runs/$NAME/cobaya.yaml" >&2; exit 1; }
  if [ ! -f "$dir/trained_models/model_it_$last.keras" ]; then
    [ -n "$AFTER" ] || { echo "FATAL: $dir has no network $last yet (give AFTER=<jobs> if it is still being made)" >&2; exit 1; }
    echo "note: $dir has no network $last yet: the combine waits for job(s) $AFTER"
  fi
  PART_ARGS="$PART_ARGS --part $dir:$last"
done

DEPEND=${AFTER:+--dependency=afterok:$AFTER}
C=$(sbatch --parsable $DEPEND --job-name=combine_axion --partition=$PARTITION --nodes=1 --ntasks=1 --cpus-per-task=4 \
      --mem=32G --time=02:00:00 --output=logs/combine_%j.out \
      --wrap "source ./env.sh && python \$CLIENT_REPO/client.py combine $OUT --parameter omega$PART_ARGS ${BAND_ARGS% }")
E=$(sbatch --parsable --partition=$PARTITION --dependency=afterok:${C%%;*} --array=0 analysis/submit_evidence.sh "$OUT")
R=$(sbatch --parsable --partition=$PARTITION --dependency=afterok:${C%%;*} analysis/submit_report.sh "$OUT" \
      --networks 0 --corner 0 --corner-parameters 10)
T=$(sbatch --parsable --partition=$PARTITION --ntasks=$RANKS --time=12:00:00 --dependency=afterok:${R%%;*} \
      analysis/submit_true_check.sh "$OUT" --networks 0 --points 500)
M=$(sbatch --parsable --dependency=afterok:${E%%;*}:${T%%;*} --job-name=ev_summary --partition=$PARTITION --nodes=1 \
      --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:30:00 --output=logs/ev_summary_%j.out \
      --wrap "source ./env.sh && python analysis/evidence_summary.py $OUT")
PROFILES=${PROFILES:-1}
P_IDS=""
if [ "$PROFILES" = 1 ]; then
  shopt -s nullglob
  yamls=(runs/$NAME/profiles_*.yaml)
  [ ${#yamls[@]} -gt 0 ] || echo "note: no runs/$NAME/profiles_*.yaml: no profiles queued"
  for y in "${yamls[@]}"; do
    s=$(basename "$y" .yaml); s=${s#profiles_}
    # a study that exists belongs to an earlier combine: never resume it with another network
    if [ -e "results_profiles/${NAME%_combined200}200_$s" ]; then
      echo "note: results_profiles/${NAME%_combined200}200_$s exists (an earlier combine's study): not submitted"
      continue
    fi
    P=$(sbatch --parsable --dependency=afterok:${C%%;*} true_profiles/submit_profiles.sh "$OUT" "$y" \
          results_profiles/${NAME%_combined200}200_$s --iteration 0)
    P_IDS="$P_IDS $s ${P%%;*},"
  done
fi
echo "$OUT ($PART_ARGS, ${BAND_ARGS% }${AFTER:+, after job(s) $AFTER}): combine ${C%%;*}, evidence ${E%%;*}, report ${R%%;*}, true check ${T%%;*}, summary ${M%%;*}${P_IDS:+; profiles:${P_IDS%,}}"
