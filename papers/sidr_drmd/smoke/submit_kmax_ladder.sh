#!/bin/bash -l
# How high must k_max go for halofit and for HMcode to give finite spectra on Grendel? The env's CLASS++
# build (the one the runs use), the second set's CLASS inputs (runs/sidr_drmd2_pp_low/cobaya.yaml), and
# P_k_max_1/Mpc from 5 (what the runs' 1 is, with a non-linear module on) to 30 (smoke/kmax_ladder.py):
#
#   sbatch smoke/submit_kmax_ladder.sh results/20261020_122013_sidr_drmd_pp
#
#   1. halofit at the run's 41 refused points (NaN on Grendel at k_max 5)
#   2. HMcode at the same points
#   3. both at 40 random points of the second set's box with delta_Neff < 0.13, k_max 5, 6, 8, 10, 20, 30
#   4. the time of one evaluation at the run's best point, one process, both modules, every k_max
#
# Made 21 October 2026; the laptop has its own run (logs/kmax_ladder_laptop.out): there halofit is
# finite at every point and HMcode is not. About 30 minutes on 16 cores. Output:
# logs/kmax_ladder_<job>.out and logs/kmax_ladder_grendel_*.json (.npz: the spectra).
#
#SBATCH --job-name=kmax_ladder
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/kmax_ladder_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ./env.sh
TARGET=${1:?usage: sbatch smoke/submit_kmax_ladder.sh results/<timestamp>_<RUN>}
COBAYA=runs/sidr_drmd2_pp_low/cobaya.yaml
[ -d "$TARGET" ] || { echo "FATAL: no run directory $TARGET" >&2; exit 1; }
[ -f "$COBAYA" ] || { echo "FATAL: no $COBAYA (sync the kit)" >&2; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
P=${SLURM_CPUS_PER_TASK:-16}
echo "== $(date)  job ${SLURM_JOB_ID:-local} on $(hostname -s)  run=$TARGET  $P processes"
ladder() { python smoke/kmax_ladder.py "$TARGET" "$COBAYA" "$@"; }
ladder grendel_halofit_refused --module halofit --kmax 5 6 7 8 10 15 20 30 --points refused --processes "$P"
ladder grendel_hmcode_refused --module hmcode --kmax 5 6 7 8 10 15 20 30 --points refused --processes "$P"
ladder grendel_halofit_sample40 --module halofit --kmax 5 6 8 10 20 30 --points sample:40 --processes "$P"
ladder grendel_hmcode_sample40 --module hmcode --kmax 5 6 8 10 20 30 --points sample:40 --processes "$P"
ladder grendel_halofit_best --module halofit --kmax 5 6 7 8 10 15 20 30 --points best
ladder grendel_hmcode_best --module hmcode --kmax 5 6 7 8 10 15 20 30 --points best
echo "== summaries"
grep -h "SUMMARY\|still not finite" "logs/kmax_ladder_${SLURM_JOB_ID:-local}.out" 2>/dev/null || true
echo "== $(date)  done"
