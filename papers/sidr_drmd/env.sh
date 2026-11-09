# Source me, never run me:   source papers/sidr_drmd/env.sh   (from a checkout of this branch)
#
# The ONE place that says where things live on Grendel and which versions are pinned. Every
# install step, check and job script sources this. A value already set in the environment
# wins, which is how the same scripts are rehearsed off the cluster.
#
# Adapted from the hpc_client harness that has trained CLiENT surrogates on Grendel since
# September 2026 (README.md says what was changed and why).

# ── locations (all under your own home) ─────────────────────────────────────
_here=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
export HPC_CLIENT_DIR="${HPC_CLIENT_DIR:-$_here}"                               # this folder (papers/sidr_drmd/)
export CLIENT_REPO="${CLIENT_REPO:-$(cd "$HPC_CLIENT_DIR/../.." && pwd)}"         # CLiENT: the checkout of this branch
unset _here
# The launcher, the report and the box programs are CLiENT's (hpc/); their runs/ and logs/ are this folder's.
export CLIENT_HPC_DIR="${CLIENT_HPC_DIR:-$HPC_CLIENT_DIR}"
export CLASSPP_REPO="${CLASSPP_REPO:-$HOME/codes/CLASSpp_public}"               # clone made by install/03_classy.sh
export CLASSPP_URL="${CLASSPP_URL:-https://github.com/AarhusCosmology/CLASSpp_public.git}"
export CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"                             # private Miniforge
export CLIENT_ENV="${CLIENT_ENV:-$HOME/codes/envs/client}"                      # conda env PREFIX (a path, not a name)
export COBAYA_PACKAGES_PATH="${COBAYA_PACKAGES_PATH:-$HOME/codes/cobaya_packages}"  # likelihood data; Cobaya reads this
                                                                                # variable when a yaml has no packages_path

# ── pinned versions ─────────────────────────────────────────────────────────
# CLASS++ is pinned in the file EXPECTED_COMMIT (7cc41a51: the commit the laptop's classy was
# built from, so cluster and laptop evaluate the same code). The Python packages come from
# CLiENT's own environment.yaml (install/02_env.sh); the three below are not in that file.
export CAMB_VERSION="${CAMB_VERSION:-2.0.4}"      # the laptop's; PyPI has a manylinux_2_28 wheel
export KERAS_VERSION="${KERAS_VERSION:-3.12.4}"   # the laptop's: a .keras file loads with the Keras line that wrote it
export MPI4PY_VERSION="${MPI4PY_VERSION:-4.1.2}"  # the laptop's; built here from source against the module MPI
# Likelihood codes that are separate Python packages (install/06_external_likelihoods.sh). The
# newest releases on 8 October 2026; set them to a collaborator's versions to match their chains.
export HILLIPOP_VERSION="${HILLIPOP_VERSION:-4.3.1}"          # planck-2020-hillipop
export LOLLIPOP_VERSION="${LOLLIPOP_VERSION:-4.1.2}"          # planck-2020-lollipop
export ACT_LENSLIKE_VERSION="${ACT_LENSLIKE_VERSION:-1.2.1}"  # act_dr6_lenslike (data v1.2)
export ASTROPY_VERSION="${ASTROPY_VERSION:-6.1.7}"            # the laptop's; the two Planck packages read FITS with it
# planckpr4lensing (Carron, Mirmelstein & Lewis, arXiv:2206.07773): Planck PR4 lensing, not on PyPI,
# so pinned by commit: v1.0.3 of 20 August 2026, which still imports with Cobaya 3.6.2. Its data
# are inside the package.
export PR4_LENSING_URL="${PR4_LENSING_URL:-https://github.com/carronj/planck_PR4_lensing}"
export PR4_LENSING_COMMIT="${PR4_LENSING_COMMIT:-e08f1dc256f0c442662a708eb22e2cb488408fee}"
# HiLLik (install/07_hillik.sh): the joint Planck PR4 + ACT DR6 + SPT-3G D1 likelihood of Tristram et
# al. (arXiv:2511.04733). It has no releases, so it is pinned by commit: master of 6 October 2026.
export HILLIK_URL="${HILLIK_URL:-https://github.com/mtristram/hillik.git}"
export HILLIK_REPO="${HILLIK_REPO:-$HOME/codes/hillik}"                       # clone made by install/07_hillik.sh
export HILLIK_COMMIT="${HILLIK_COMMIT:-5a155da7429c7c14a10d6ba973d4275d0282f19e}"
export SACC_VERSION="${SACC_VERSION:-2.4}"                                    # the ACT data are a SACC file; newest on 9 October 2026
# The lite CMB likelihoods and the SPT lensing of the H0 World Cup data (install/11_lite_likelihoods.sh, the SIDR +
# DRMD runs of 23 October 2026). ACT-lite (Louis et al., arXiv:2503.14452) has no releases: pinned by commit, the
# newest on 23 October 2026. candl (Balkenhol et al., arXiv:2401.13433) from PyPI, with clipy-like, which it
# imports. The SPT-3G D1 data for candl (Camphuis et al., arXiv:2506.20707) by commit (v3.0.1). The SPT-3G 2019-2020
# MUSE lensing (Ge et al., arXiv:2411.06000) is a zip on the SPT site (version March 2025), checked against
# MUSE3G_SHA256 (the script prints the hash it got).
export ACT_LITE_URL="${ACT_LITE_URL:-https://github.com/ACTCollaboration/DR6-ACT-lite.git}"
export ACT_LITE_COMMIT="${ACT_LITE_COMMIT:-627aeafb88ae5ad1aa66b406bea2d65cfa66a27d}"
export CANDL_VERSION="${CANDL_VERSION:-2.2.0}"                # candl-like
export CLIPY_VERSION="${CLIPY_VERSION:-0.15}"                 # clipy-like
export SPT_CANDL_DATA_URL="${SPT_CANDL_DATA_URL:-https://github.com/SouthPoleTelescope/spt_candl_data.git}"
export SPT_CANDL_DATA_COMMIT="${SPT_CANDL_DATA_COMMIT:-f22611a9671d50af6135f048a2ea3f43ef4c2f62}"
export MUSE3G_URL="${MUSE3G_URL:-https://pole.uchicago.edu/public/data/ge25/muse_3g_like_march_2025.zip}"
export MUSE3G_DIR="${MUSE3G_DIR:-$HOME/codes/muse3glike_march2025}"   # the unpacked zip; the package is installed from it
export MUSE3G_SHA256="${MUSE3G_SHA256:-4933ceb993ceb92e48b2191b8ccb1595d182bc4bc3862c1fbad61d5f82136e9b}"   # the zip of 14 March 2025, downloaded 23 October 2026
# BEST (install/08_best.sh): the nested sampler `client.py evidence` computes ln Z with, on the
# network alone. Pinned to the commit of the laptop's checkout, which CLiENT's evidence command
# was validated with (docs/evidence.md), and used from the clone through BEST_PATH: its own
# package metadata asks for numpy below 2, which this environment is not.
export BEST_URL="${BEST_URL:-https://github.com/AndreasNygaard/best-inference.git}"
export BEST_REPO="${BEST_REPO:-$HOME/codes/best-inference}"                   # clone made by install/08_best.sh
export BEST_COMMIT="${BEST_COMMIT:-b5cd08b113fb77d4d243d596a4e8a42ac5503cad}"
export BEST_PATH="${BEST_PATH:-$BEST_REPO}"                                   # where CLiENT looks for it
export TFP_VERSION="${TFP_VERSION:-0.25.0}"                # tensorflow-probability, the laptop's
export TF_KERAS_VERSION="${TF_KERAS_VERSION:-2.19.0}"      # tf-keras, the laptop's (matches tensorflow 2.19)
export HYPERSPHERE_VERSION="${HYPERSPHERE_VERSION:-0.1.1}" # hypersphere-sampler, the laptop's
export DM_TREE_VERSION="${DM_TREE_VERSION:-0.1.10}"        # needed by tensorflow-probability; the laptop's
export DECORATOR_VERSION="${DECORATOR_VERSION:-5.3.1}"     # needed by tensorflow-probability; the laptop's

# ── toolchain (settled by the harness's probe of 2026-09-15) ────────────────
# classy is ONE shared object loaded by every rank. x86-64-v3 (AVX2) runs on every CPU
# partition of Grendel, q48 included.
export CLASS_ARCH_FLAGS="${CLASS_ARCH_FLAGS:--march=x86-64-v3}"
export GCC_MODULE="${GCC_MODULE:-gcc/14.2.0}"
# MPI: Grendel's own runtime from the module system, NEVER a conda MPI. mpi4py is built
# against this module's mpicc, and install/check.py fails if it links any other MPI.
export MPI_MODULE="${MPI_MODULE:-openmpi/4.0.3}"
export MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun --bind-to none}"
# Since Grendel's Slurm 26.05 (2 November 2026) a batch script carries SLURM_STEP_ID=SLURM_STEPID=-5 (the batch step),
# and Open MPI 4.0.3 then takes a Python that starts MPI by itself for a step started by srun: it aborts in
# MPI_Init_thread ("on a NULL communicator"). Without the two, such a start works, and mpirun on one or several nodes
# works either way (tests mpitest_A-D, jobs 7010672-7010677). The job scripts also run install/check.py under mpirun.
case "${SLURM_STEP_ID:-}" in -5|4294967291) unset SLURM_STEP_ID SLURM_STEPID ;; esac

# TensorFlow extra: "" gives the plain wheel (CPU; what a q48 node needs). "and-cuda" adds
# CUDA as pip wheels (~3 GB) and opens the GPU partitions for rank 0's training; changing it
# later means rebuilding the env.
export TF_EXTRA="${TF_EXTRA-}"

# ── modules and activation (quiet no-ops before the install has happened) ───
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  . "$CONDA_ROOT/etc/profile.d/conda.sh"
  [ -d "$CLIENT_ENV" ] && conda activate "$CLIENT_ENV"
fi
if command -v module >/dev/null 2>&1; then
  # The site anaconda modules put an old Python and pip on PATH. Never alongside this env.
  module unload anaconda3 anaconda2 >/dev/null 2>&1 || true
  module load "$GCC_MODULE" >/dev/null 2>&1 || true
  module load "$MPI_MODULE" >/dev/null 2>&1 || true   # after gcc: MPI modules are built per compiler
fi

# The commit of the env's CLASS++ (install/03_classy.sh writes BUILD_COMMIT), for logs/backend_<job>.json. A run on
# another build names its own in job.env (CLASSPP_BUILD_COMMIT), which the launcher records instead.
[ ! -f "$HPC_CLIENT_DIR/BUILD_COMMIT" ] || export CLASSPP_COMMIT="${CLASSPP_COMMIT:-$(tr -d '[:space:]' < "$HPC_CLIENT_DIR/BUILD_COMMIT")}"

# CLiENT is not a package: its modules import each other from the repo root.
export PYTHONPATH="$CLIENT_REPO${PYTHONPATH:+:$PYTHONPATH}"
