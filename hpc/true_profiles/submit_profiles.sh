#!/bin/bash -l
# An example Slurm job for true profiles of a run (client.py profiles) on one node: adapt the
# #SBATCH lines and the environment. Run it from hpc/:
#
#   sbatch true_profiles/submit_profiles.sh results/<timestamp>_<RUN> runs/<RUN>/profiles.yaml results_profiles/<name>
#   ... the same three again continue where it stopped; --dry-run as a fourth prints the plan
#
# Not an MPI job. The profile code starts the YAML's `workers` backend processes, each with the
# run's Cobaya model; leave a core or two to the parent, which evaluates the network. They build
# it CLIENT_LOAD_GROUP at a time when runs/<RUN>/job.env sets that.
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
mkdir -p logs

# Activate the environment here, e.g.:  conda activate clienv

TARGET=${1:?usage: sbatch true_profiles/submit_profiles.sh results/<timestamp>_<RUN> <profile yaml> <output directory> [arguments]}
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
mkdir -p "$(dirname "$OUTPUT")"
python true_profiles/launch_profiles.py "$TARGET" --config "$CONFIG" -o "$OUTPUT" ${resume[@]:+"${resume[@]}"} "$@"
python true_profiles/status.py "$OUTPUT" || true
