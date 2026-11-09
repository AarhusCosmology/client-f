#!/bin/bash -l
# CLASS++ master bfe6729e (install/10_classpp_build.sh) at the points that broke the pinned build, with the World Cup
# runs' settings (runs/sidr_drmd_wc_low/cobaya.yaml: pre/act_dr6.pre's lines without nonlinear_min_k_max, halofit).
#
#   sbatch smoke/submit_build_check.sh results/20261020_122013_sidr_drmd_pp
#
# The build is the one the runs' job.env asks for (CLASSPP_BUILD_DIR; the job stops when it is missing). With
# smoke/kmax_ladder.py, at P_k_max_1/Mpc = 1 (so k_max = the default floor of 2/Mpc, the sources continued past it as
# the runs will have them) and 10 (modes solved to 10/Mpc), for:
#   the first Pantheon+ run's 41 refused points (NaN spectra on Grendel with 7cc41a51, dark acoustic oscillations;
#   CLASS++ #467 is its own fix of that), 40 random points of the second set's box with delta_Neff < 0.13, and the
#   run's best point.
# It answers: are the spectra finite everywhere (no guard patch any more), and how far do the lensed TT, EE and phiphi
# at the default 2/Mpc lie from modes solved to 10/Mpc at points with dark acoustic oscillations (the continuation
# assumes the high-k form of cold matter). l_max_scalars is the ladder's 2500. Made 23 October 2026. About 10 minutes
# on 16 cores. Output: logs/build_check_<job>.out, logs/kmax_ladder_build_*.{json,npz}.
#
#SBATCH --job-name=build_check
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/build_check_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ./env.sh
TARGET=${1:?usage: sbatch smoke/submit_build_check.sh results/<timestamp>_sidr_drmd_pp}
RUN=sidr_drmd_wc_low
COBAYA=runs/$RUN/cobaya.yaml
[ -d "$TARGET" ] || { echo "FATAL: no run directory $TARGET" >&2; exit 1; }
[ -f "$COBAYA" ] || { echo "FATAL: no $COBAYA (sync the kit)" >&2; exit 1; }
source "runs/$RUN/job.env"                    # CLASSPP_BUILD_DIR first on PYTHONPATH, or stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
P=${SLURM_CPUS_PER_TASK:-16}
mkdir -p logs
echo "== $(date)  job ${SLURM_JOB_ID:-local} on $(hostname -s)  run=$TARGET  settings=$COBAYA"
[ -z "${CLIENT_RUN_CHECK:-}" ] || python install/check.py --stage classy
ladder() { python smoke/kmax_ladder.py "$TARGET" "$COBAYA" "$@" --module halofit --kmax 1 10; }
ladder build_halofit_refused --points refused --processes "$P"
ladder build_halofit_sample40 --points sample:40 --processes "$P"
ladder build_halofit_best --points best
echo "== summaries"
grep -h "SUMMARY\|still not finite" "logs/build_check_${SLURM_JOB_ID:-local}.out" 2>/dev/null || true
echo "== $(date)  done"
