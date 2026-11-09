#!/bin/bash -l
# chi^2 of every likelihood at points, one Cobaya set-up after another (smoke/chi2_points_mp.py, 24 October 2026):
#
#   sbatch smoke/submit_chi2_points.sh smoke/chi2_compare_points.json        # why the preferred Geff moves (8 set-ups)
#   sbatch smoke/submit_chi2_points.sh smoke/chi2_profile_points.json        # every node of the earlier true profiles
#   sbatch smoke/submit_chi2_points.sh POINTS SET [SET ...]                   # only the named sets
#
# Points and set-ups: results/20261024_chi2_compare/make_points.py. Each set runs in a shell of its own that sources
# the set's job.env (its CLASS++ build: the patched 7cc41a51 of the earlier runs, or bfe6729e of the new ones). The
# compare file is 32 true evaluations (four set-ups use the new runs' expensive settings), about 30 minutes; the
# profile file 625 evaluations of the earlier set-up on 48 workers, about 10 minutes.
# Output: logs/chi2_points_<job>.out and logs/chi2_points_<SET>_<job>.json.
#
#SBATCH --job-name=chi2_points
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=logs/chi2_points_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ./env.sh
POINTS=${1:?usage: sbatch smoke/submit_chi2_points.sh POINTS [SET ...]}
shift
[ -f "$POINTS" ] || { echo "FATAL: no $POINTS (sync the kit)" >&2; exit 1; }
if [ $# -gt 0 ]; then SETS=("$@"); else mapfile -t SETS < <(python -c "import json,sys; print('\n'.join(json.load(open(sys.argv[1]))['sets']))" "$POINTS"); fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
echo "== $(date)  job ${SLURM_JOB_ID:-local} on $(hostname -s)  $POINTS: ${SETS[*]}"
for s in "${SETS[@]}"; do
  env_file=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['sets'][sys.argv[2]]['job_env'])" "$POINTS" "$s")
  ( source "$env_file"
    python smoke/chi2_points_mp.py "$POINTS" "$s" --workers "${SLURM_CPUS_PER_TASK:-48}" ) \
    || echo "== $s FAILED (continuing with the next set)"
done
echo "== $(date)  done"
