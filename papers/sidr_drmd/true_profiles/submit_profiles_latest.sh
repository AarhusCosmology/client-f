#!/bin/bash -l
# True 1D profiles from a run's LATEST network, with the ranges read off the report's chain of that
# network; made to be queued behind the run's report job (analysis/submit_report.sh), so that the
# profiles start by themselves when a run's boosts and its report are done:
#
#   sbatch --dependency=afterok:<report job> true_profiles/submit_profiles_latest.sh results/<timestamp>_<RUN> <name> \
#       [true_profiles/make_config.py arguments, e.g. --parameters a b c --points 25]
#
# At its start, N is the run's latest network (trained_models/model_it_N.keras), and the profile YAML
# is written by true_profiles/make_config.py --iteration N --chain RUN/analysis/posterior_it_N.npz to
# results_profiles/<name>.yaml (the arguments after <name> go to make_config.py). Then the study runs
# exactly as true_profiles/submit_profiles.sh runs it, into results_profiles/<name>/. Submitted again
# (without a dependency), it resumes that study with the YAML it wrote the first time.
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

TARGET=${1:?usage: sbatch true_profiles/submit_profiles_latest.sh results/<timestamp>_<RUN> <name> [make_config.py arguments]}
NAME=${2:?the name of the study: results_profiles/<name>/ and results_profiles/<name>.yaml}
shift 2
[ -d "$TARGET" ] || { echo "FATAL: no run directory $TARGET" >&2; exit 1; }
CONFIG=results_profiles/$NAME.yaml
if [ -f "$CONFIG" ]; then
  echo "== the profile YAML exists: $CONFIG (written when this study began; resuming with it)"
else
  N=$(ls "$TARGET/trained_models" | sed -nE 's/^model_it_([0-9]+)\.keras$/\1/p' | sort -n | tail -1)
  [ -n "$N" ] || { echo "FATAL: no trained network in $TARGET/trained_models" >&2; exit 1; }
  CHAIN=$TARGET/analysis/posterior_it_$N.npz
  [ -f "$CHAIN" ] || { echo "FATAL: no $CHAIN: the report has not sampled network $N (sbatch analysis/submit_report.sh $TARGET --networks $N)" >&2; exit 1; }
  mkdir -p results_profiles
  echo "== the latest network of $TARGET is $N; the ranges are read off $CHAIN"
  python "${CLIENT_REPO:?source env.sh first}/hpc/true_profiles/make_config.py" "$TARGET" --iteration "$N" --chain "$CHAIN" --out "$CONFIG" "$@"
fi
exec bash true_profiles/submit_profiles.sh "$TARGET" "$CONFIG" "results_profiles/$NAME"
