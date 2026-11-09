#!/bin/bash -l
# Minimise chi^2 with the true likelihood alone (smoke/minimise_true.py; 26 October 2026):
#
#   sbatch smoke/submit_minimise.sh SPEC NAME [--starts LABEL ...]
#
# SPEC: the starts and settings (e.g. smoke/minimise_drmd_wcmb.json, made by the paper's drmd_minimise_points.py:
# DRMD, log10 Geff = -5.5, on the World Cup data with SH0ES). The job sources the spec's job.env (its CLASS++ build).
# Two starts of 11 slow parameters fill one node (2 x 23 evaluations per gradient); give each pair of starts a job.
# Output: logs/minimise_<NAME>_<job>.out and logs/minimise_<NAME>_<job>.json (every evaluation, rewritten as it goes).
#
#SBATCH --job-name=minimise
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=16:00:00
#SBATCH --output=logs/minimise_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ./env.sh
SPEC=${1:?usage: sbatch smoke/submit_minimise.sh SPEC NAME [--starts LABEL ...]}
NAME=${2:?the name of the minimisation}
shift 2
[ -f "$SPEC" ] || { echo "FATAL: no $SPEC (sync the kit)" >&2; exit 1; }
env_file=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['job_env'])" "$SPEC")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
echo "== $(date)  job ${SLURM_JOB_ID:-local} on $(hostname -s)  $SPEC ($NAME) $*"
source "$env_file"
python smoke/minimise_true.py "$SPEC" "$NAME" --workers "${SLURM_CPUS_PER_TASK:-48}" "$@"
echo "== $(date)  done"
