# Real likelihoods on a cluster

The [launcher and tools in `hpc/`](_generated/library/hpc/README.md) add what real runs need to `client.py`, without changing a file of CLiENT. They are written for Cobaya, and the example job scripts for Slurm with Open MPI. `hpc/runs/lcdm_planck_desi/` is a worked configuration.

## A sequence

1. Install the environment, the theory code and the likelihood data, and evaluate one point.
2. Train with an explicit box, saved configuration and call journals (`hpc/launch.py`, `hpc/submit_client.sh`). `hpc/analysis/find_centre.py` and `make_box.py` help with the box of a likelihood with many nuisance parameters. Likelihood loading can be serialised across ranks to control startup memory. Start from `input/recommended_near_gaussian.yaml` or `input/recommended_non_gaussian.yaml`.
3. Generate network summaries (`hpc/analysis/report.py`), inspect parameter boundaries and compare with the appropriate published reference.
4. Evaluate the true objective at saved posterior samples (`hpc/analysis/true_check.py`); keep those calls separate from training.
5. For profiles, run a pilot, then the full requested study (`hpc/true_profiles/`). Profiles use managed backend processes, not an MPI-launched profile command.
6. For evidence, distinguish BEST on the network (`client.py evidence`), boost mode-bridge estimates, and correction using unfiltered posterior true checks (`hpc/analysis/evidence_summary.py`).

`hpc/analysis/branch_run.py` makes a branch of a run at an earlier network, with other settings, leaving the source as it is.

## Refused evaluations

By default, a refused point remains nonfinite. Optional `CLIENT_REFUSED_VALUE` can substitute a fixed finite training label; `design` derives that value once after the initial design. The policy is written to the run and applies to training only: a profile uses the true objective, and a refused point there stays nonfinite.

## GPUs

Only rank 0 and the ranks that train members (`training.members`) run TensorFlow. Each takes one GPU of its node, the node's processes in turn over its GPUs, with memory growth on, so that several share a GPU; every other rank hides the GPUs from itself before the likelihood is built (`utils/devices.py`). This holds for `hpc/submit_client.sh` and for a plain `mpirun` alike. The environment needs TensorFlow with CUDA (for example `pip install tensorflow[and-cuda]`, the version `environment.yaml` pins); `training.member_device` (`auto`, `cpu`, `gpu`) decides for the members. The log says where rank 0 trains and where every member did.

On one NVIDIA L40S a network of a real 19-parameter run trained in full float32 exactly as on the CPU, about five times faster than on 8 CPU threads, and with TensorFlow's default TF32 to an equally good network. On a new GPU node, check with `tests/gpu_members_smoke.py` (rank 0 and the members on the GPUs, none failing) and `benchmarking/gpu_training.py RUN --network N` (TF32, full float32 and the CPU on one of your networks). The [members guide](_generated/library/docs/members.md) has the details and what is still open.
