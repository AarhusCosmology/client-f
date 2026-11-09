#!/bin/bash -l
# Profile likelihoods with the true likelihood alone, node by node along chains (smoke/profile_chain.py; 27 October 2026):
#
#   sbatch smoke/submit_profile_chain.sh SPEC NAME [--chains LABEL ...] [--node-budget N] [--resume logs/profile_chain_NAME_JOB.json]
#
# SPEC: the chains, seeds, nodes and covariance (e.g. smoke/profile_chain_wc_omega_b.json, made by the paper's
# wc_chain_setup.py). The job sources the spec's job.env (its CLASS++ build). Two chains of 11 free slow parameters fill
# one node (2 x 23 evaluations per gradient); a node takes about 10-20 minutes, a chain of 13 nodes about 3-4 hours.
# Output: logs/profile_chain_<NAME>_<job>.out and logs/profile_chain_<NAME>_<job>.json (rewritten as it goes); a job
# stopped at its time limit is continued with --resume and its .json.
#
#SBATCH --job-name=profchain
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/profile_chain_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ./env.sh
SPEC=${1:?usage: sbatch smoke/submit_profile_chain.sh SPEC NAME [--chains LABEL ...] [--resume LOG]}
NAME=${2:?the name of the profile set}
shift 2
[ -f "$SPEC" ] || { echo "FATAL: no $SPEC (sync the kit)" >&2; exit 1; }
env_file=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['job_env'])" "$SPEC")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
echo "== $(date)  job ${SLURM_JOB_ID:-local} on $(hostname -s)  $SPEC ($NAME) $*"
source "$env_file"
python smoke/profile_chain.py "$SPEC" "$NAME" --workers "${SLURM_CPUS_PER_TASK:-48}" "$@"
echo "== $(date)  done"
