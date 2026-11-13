#!/bin/bash -l
# One CLiENT run on ONE q48 node (48 cores, 384 GB): train the surrogate likelihood for runs/<RUN>.
#
#   sbatch submit_client.sh lcdm_planck_desi -i 16            # new run -> results/<timestamp>_lcdm_planck_desi/
#   sbatch submit_client.sh results/<timestamp>_<RUN> -i 2    # continue that run by two iterations
#   sbatch --time=5-00:00:00 submit_client.sh <RUN> -i 30     # a longer limit (q48 allows 15 days)
#   sbatch --nodes=2 --ntasks=96 submit_client.sh <RUN>        # two nodes: the evaluations on 96 ranks (23 Oct 2026)
#
# Everything after the first argument is passed to client.py (-i N: the number of networks
# of a new run, or of additional iterations of a continued one).
#
#SBATCH --job-name=client
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --output=logs/client_%j.out
#
# ── RANKS AND THREADS (the layout the harness this script comes from measured on q48) ──
# CLiENT alternates two phases:
#   * likelihood evaluations on ALL 48 ranks (each a full Cobaya + Boltzmann code, one thread), and
#   * training + surrogate sampling in TensorFlow on RANK 0 ONLY, while the others wait.
# Hence the module's mpirun inside the allocation ($MPI_LAUNCHER, "--bind-to none") rather than
# srun: the ranks are not pinned, so rank 0's TensorFlow threads can use the cores the idle
# workers leave. OMPI_MCA_mpi_yield_when_idle asks idle Open MPI ranks to yield instead of
# spinning. During a training phase `top` on the node should show the workers near 0% CPU;
# if they spin, they are taking the cores TensorFlow trains on.
#
# One thread per rank: measured there, 48 x 1 gives about twice the evaluations per
# core-hour of 16 x 4. --mem=0 is the whole node's memory (1.8 to 3 GB per rank at peak, and
# rank 0 also trains).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source ./env.sh

TARGET=${1:?usage: sbatch submit_client.sh <run name under runs/> | <results/... dir to continue> [client.py arguments]}
shift
if [ -d "$TARGET" ]; then
  RUN=$(basename "$TARGET" | sed -E 's/^[0-9]{8}_[0-9]{6}_//')
  CLIENT_ARGS=("$TARGET" "$@")
else
  RUN=$TARGET
  CLIENT_ARGS=("runs/$RUN/client.yaml" -n "$RUN" -o results "$@")
fi
DOMAIN=runs/$RUN/domain_bounds.json
# A run's own job settings (runs/<RUN>/job.env), e.g. CLIENT_LOAD_GROUP for likelihoods that
# cannot be loaded on every rank at once.
[ ! -f "runs/$RUN/job.env" ] || source "runs/$RUN/job.env"
[ -f "runs/$RUN/client.yaml" ] || { echo "FATAL: no runs/$RUN/client.yaml" >&2; exit 1; }
[ -f "$DOMAIN" ] || { echo "FATAL: no $DOMAIN (the run's training box)" >&2; exit 1; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMPI_MCA_mpi_yield_when_idle=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1

# ── WHO EVALUATES WHAT ──────────────────────────────────────────────────────
# "chunked": the whole batch is scattered once, one slice per rank, and gathered once; all 48
# ranks evaluate. It is the schedule every laptop run with the audit and the boost has used,
# and its values are identical to the default ("static", lock-step batches that each cost
# their slowest point). "dynamic" (rank 0 hands out one point at a time and does not evaluate)
# is what the harness uses for likelihoods whose cost varies by a factor of five; for
# LambdaCDM the two are a wash. Set CLIENT_MPI_SCHEDULE to change it.
export CLIENT_MPI_SCHEDULE=${CLIENT_MPI_SCHEDULE:-chunked}

# ── TENSORFLOW ON RANK 0 (measured on q48 with CLiENT's own model and sampler) ──
#                     training s/epoch   AIES
#                      5k pts   20k pts  ms/step
#   1 thread            0.51     1.75    4.37
#   8 threads           0.25     0.73    3.57    <- the default below
#   16 threads          0.24     0.72    3.78
#   48 threads          0.32     0.97    4.28    (TensorFlow's own default: all cores)
export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-8}
export TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS:-4}
export TF_CPP_MIN_LOG_LEVEL=1

echo "== $(date)  job ${SLURM_JOB_ID:-local} on $(hostname -s)  run=$RUN  arguments: ${CLIENT_ARGS[*]}"
echo "   ranks=${SLURM_NTASKS:-?} threads/rank=$OMP_NUM_THREADS schedule=$CLIENT_MPI_SCHEDULE  TF threads=$TF_NUM_INTRAOP_THREADS"
echo "   CLASS++@${CLASSPP_BUILD_COMMIT:-$(cat BUILD_COMMIT 2>/dev/null)}  CLiENT@$(git -C "$CLIENT_REPO" rev-parse --short HEAD)"

# Guards first: a wrong build, a wrong CLiENT branch or an input that does not evaluate must
# stop the job here, not 2000 evaluations later.
# Under mpirun (2 November 2026): since Grendel's Slurm 26.05 sets SLURM_STEP_ID in a batch script, a Python that
# starts MPI by itself there aborts in MPI_Init_thread (Open MPI 4.0.3); one started by mpirun does not.
[ -z "${CLIENT_RUN_CHECK:-}" ] || $MPI_LAUNCHER -n 1 python install/check.py --stage all --run "$RUN"

# Only rank 0 runs TensorFlow; hide any GPU from the evaluation ranks.
# -x: every rank must see the same schedule, and, on a job of several nodes (sbatch --nodes=2 --ntasks=96), the same
# PYTHONPATH and CLASS++ build variables (a run's job.env): a rank without them would import the env's classy, and
# with neither variable launch.py's check of its build would let that pass.
$MPI_LAUNCHER -x CLIENT_MPI_SCHEDULE ${CLIENT_LOAD_GROUP:+-x CLIENT_LOAD_GROUP} ${CLIENT_REFUSED_VALUE:+-x CLIENT_REFUSED_VALUE} \
  -x PYTHONPATH -x CLIENT_HPC_DIR ${CLASSPP_COMMIT:+-x CLASSPP_COMMIT} ${CLASSPP_GUARD_DIR:+-x CLASSPP_GUARD_DIR} \
  ${CLASSPP_BUILD_DIR:+-x CLASSPP_BUILD_DIR -x CLASSPP_BUILD_COMMIT} -n "${SLURM_NTASKS:-1}" \
  bash -c '[ "${OMPI_COMM_WORLD_RANK:-0}" = 0 ] || export CUDA_VISIBLE_DEVICES=""; exec python "$@"' _ \
  "$CLIENT_REPO/hpc/launch.py" "$DOMAIN" "${CLIENT_ARGS[@]}"

echo "== $(date)  done"
