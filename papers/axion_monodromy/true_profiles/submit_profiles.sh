#!/bin/bash -l
# True 1D profiles of a run's parameters (CLiENT's `client.py profiles`) on ONE q48 node.
#
#   sbatch true_profiles/submit_profiles.sh results/<timestamp>_<RUN> runs/<RUN>/profiles_all.yaml results_profiles/<name>
#   sbatch true_profiles/submit_profiles.sh <the same three>            # again: continues where it stopped
#   sbatch true_profiles/submit_profiles.sh <the same three> --dry-run  # checks the files and prints the plan; no evaluation
#
# Not an MPI job. The profile code starts `workers` backend processes of its own (the YAML's
# `workers`, 46 here: two cores are left to the parent, which evaluates the network), each with
# the run's Cobaya model. They build it CLIENT_LOAD_GROUP at a time (runs/<RUN>/job.env; HiLLiPoP).
# A study that exists is resumed: finished nodes are kept and every true value already
# computed is served from its journal. Arguments after the third go to client.py profiles.
#
#SBATCH --job-name=client_profiles
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --output=logs/profiles_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ./env.sh

TARGET=${1:?usage: sbatch true_profiles/submit_profiles.sh results/<timestamp>_<RUN> <profile yaml> <output directory> [client.py profiles arguments]}
CONFIG=${2:?the profile YAML (true_profiles/make_config.py writes one)}
OUTPUT=${3:?the directory of the study, e.g. results_profiles/<name>}
shift 3
[ -d "$TARGET" ] || { echo "FATAL: no run directory $TARGET" >&2; exit 1; }
[ -f "$CONFIG" ] || { echo "FATAL: no $CONFIG" >&2; exit 1; }
RUN=$(basename "$TARGET" | sed -E 's/^[0-9]{8}_[0-9]{6}_//')
[ ! -f "runs/$RUN/job.env" ] || source "runs/$RUN/job.env"
# A profile is of the true objective: a refused point is -inf here, never a training value.
unset CLIENT_REFUSED_VALUE
# The backend processes are not MPI ranks: Cobaya is told not to initialise MPI in each of them.
export COBAYA_NOMPI=1

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1      # one thread per backend process
export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-8} TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS:-2}
export TF_CPP_MIN_LOG_LEVEL=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1

resume=()
[ ! -f "$OUTPUT/manifest.json" ] || resume=(--resume)
echo "== $(date)  job ${SLURM_JOB_ID:-local} on $(hostname -s)  profiles of $TARGET  config $CONFIG  output $OUTPUT ${resume[*]:-(new study)} $*"
echo "   CLiENT@$(git -C "$CLIENT_REPO" rev-parse --short HEAD 2>/dev/null || cat CLIENT_COMMIT 2>/dev/null)  load group ${CLIENT_LOAD_GROUP:-all}  Newton settings ${CLIENT_PROFILE_NEWTON:-default}  CAMB AccuracyTarget ${CLIENT_CAMB_ACCURACY_TARGET:-as the run}  region $(grep -E '^region:' "$CONFIG" | head -1 | sed -E 's/^region: *//; s/ *#.*//' | grep . || echo 'run (CLiENT default)')"
mkdir -p "$(dirname "$OUTPUT")"
python "${CLIENT_REPO:?source env.sh first}/hpc/true_profiles/launch_profiles.py" "$TARGET" --config "$CONFIG" -o "$OUTPUT" ${resume[@]:+"${resume[@]}"} "$@"
python "${CLIENT_REPO:?source env.sh first}/hpc/true_profiles/status.py" "$OUTPUT" || true
echo "== $(date)  done"
