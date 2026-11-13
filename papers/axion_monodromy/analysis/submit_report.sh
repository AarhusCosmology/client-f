#!/bin/bash -l
# The convergence and error report of a run (analysis/report.py), on a compute node:
#
#   sbatch analysis/submit_report.sh results/<timestamp>_<RUN>             # also while the run is still going
#   sbatch analysis/submit_report.sh results/<timestamp>_<RUN> --no-posteriors
#
# No true likelihood is evaluated. Sampling every network at T = 1 takes a few minutes per
# network on 8 threads. Output: RUN_DIR/analysis/ (report.md, report.json, figures,
# posterior_it_N.npz) and logs/report_<job>.out. The same command works on the laptop after
# sync_from_hpc.sh: python analysis/report.py RUN_DIR --client <CLiENT checkout>.
#
#SBATCH --job-name=client_report
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/report_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ./env.sh
TARGET=${1:?usage: sbatch analysis/submit_report.sh results/<timestamp>_<RUN> [report.py arguments]}
shift
[ -d "$TARGET" ] || { echo "FATAL: no run directory $TARGET" >&2; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-8} TF_NUM_INTEROP_THREADS=2
export TF_CPP_MIN_LOG_LEVEL=2 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl_$USER

# The evaluation journals of the jobs that wrote into this run (their logs name its directory).
journals=()
for log in logs/client_*.out; do
  [ -f "$log" ] || continue
  if grep -q "$(basename "$TARGET")" "$log"; then
    job=${log#logs/client_}; job=${job%.out}
    [ -f "logs/evaluations_$job.jsonl" ] && journals+=("logs/evaluations_$job.jsonl")
  fi
done
echo "== $(date)  job ${SLURM_JOB_ID:-local} on $(hostname -s)  $TARGET  journals: ${journals[*]:-none}"
python "${CLIENT_REPO:?source env.sh first}/hpc/analysis/report.py" "$TARGET" --client "$CLIENT_REPO" ${journals[@]:+--journal "${journals[@]}"} "$@"
echo "== $(date)  done"
