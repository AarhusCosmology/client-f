#!/bin/bash -l
# The finished profile studies of one run, brought into logs/ and split likelihood by likelihood (24 October 2026):
#
#   sbatch --dependency=afterany:<profile jobs> analysis/submit_profile_breakdown.sh results/<time>_<RUN> RUN SET STUDY [STUDY ...]
#
# RUN names the folder runs/<RUN>/ (its cobaya.yaml and job.env: the set-up the profiles were computed with).
# 1. For every STUDY, the files the paper's build reads are copied from results_profiles/ (which is not synced) to
#    logs/: logs/profiles_<STUDY>.csv (profiles.csv), logs/profiles_<STUDY>_nodes/ (true/node_*.json, and
#    surrogate_profiles.json and ranges.json where the study wrote them). A study that stopped before its profiles.csv
#    (at its call budget in the last step) still has its node checkpoints.
# 2. true_profiles/node_points.py makes smoke/chi2_points_<SET>.json from the node checkpoints, and
#    smoke/chi2_points_mp.py evaluates every likelihood at every node with the run's own set-up: logs/chi2_points_<SET>_<job>.json.
#    Each node's total must reproduce its logl (the job prints the largest difference).
# The step done by hand for the second set (22-24 October). About 1 hour for 350 nodes at the World Cup set-up.
#
#SBATCH --job-name=profile_breakdown
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=logs/profile_breakdown_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ./env.sh
TARGET=${1:?usage: sbatch analysis/submit_profile_breakdown.sh results/<time>_<RUN> RUN SET STUDY [STUDY ...]}
RUN=${2:?the folder runs/<RUN>/}
SET=${3:?the name of the points set}
shift 3
[ $# -gt 0 ] || { echo "FATAL: no study named" >&2; exit 1; }
[ -d "$TARGET" ] && [ -f "runs/$RUN/cobaya.yaml" ] && [ -f "runs/$RUN/job.env" ] \
  || { echo "FATAL: no $TARGET, or runs/$RUN has no cobaya.yaml or job.env" >&2; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
echo "== $(date)  job ${SLURM_JOB_ID:-local} on $(hostname -s)  $TARGET ($RUN): $*"
STUDIES=()
for study in "$@"; do
  dir=results_profiles/$study
  if ! ls "$dir"/true/node_*.json >/dev/null 2>&1; then
    echo "== $study: no node checkpoints in $dir/true/; left out"
    continue
  fi
  mkdir -p "logs/profiles_${study}_nodes"
  cp "$dir"/true/node_*.json "logs/profiles_${study}_nodes/"
  for extra in surrogate_profiles.json ranges.json; do
    [ ! -f "$dir/$extra" ] || cp "$dir/$extra" "logs/profiles_${study}_nodes/"
  done
  if [ -f "$dir/profiles.csv" ]; then cp "$dir/profiles.csv" "logs/profiles_$study.csv"; else echo "== $study: no profiles.csv (stopped before it)"; fi
  echo "== $study: $(ls "logs/profiles_${study}_nodes"/node_*.json | wc -l) nodes copied to logs/profiles_${study}_nodes/"
  STUDIES+=("$study")
done
[ ${#STUDIES[@]} -gt 0 ] || { echo "FATAL: none of the studies has node checkpoints" >&2; exit 1; }
POINTS=smoke/chi2_points_$SET.json
rm -f "$POINTS"
python true_profiles/node_points.py "$POINTS" --set "$SET" --cobaya "runs/$RUN/cobaya.yaml" --job-env "runs/$RUN/job.env" \
  --run "$TARGET" "${STUDIES[@]}"
( source "runs/$RUN/job.env"
  python smoke/chi2_points_mp.py "$POINTS" "$SET" --workers "${SLURM_CPUS_PER_TASK:-48}" )
echo "== $(date)  done"
