# Real likelihoods on a cluster

`client.py` runs on its own under `mpirun`. This folder adds what the real runs of the method needed,
without changing a file of CLiENT. It is written for Cobaya, and the job scripts for Slurm with Open MPI;
adapt them to your machine.

| what | file |
|---|---|
| A run with an explicit training box, refused points and a call journal | `launch.py`, `domain.py`, `refused.py`, `loading.py`, `boltzmann_build.py` |
| An example job | `submit_client.sh` |
| The training box of a likelihood with many nuisance parameters | `analysis/find_centre.py`, `analysis/make_box.py` |
| A network against the true likelihood at its own posterior samples | `analysis/true_check.py`, `analysis/report.py`, `analysis/corner.py`, `analysis/chain_sampling.py` |
| Evidences corrected by those true values | `analysis/evidence_summary.py` |
| A branch of a run at an earlier network, with other settings | `analysis/branch_run.py` |
| True profiles with managed backend processes | `true_profiles/` |
| A worked real-data example, LCDM on Planck + DESI | `runs/lcdm_planck_desi/` |

Each program describes its use and its arguments at its head.

## A run

A run is a folder `runs/<RUN>/` holding `client.yaml`, `cobaya.yaml` and `domain_bounds.json`, the training box as
`{name: [low, high]}`, with paths relative to this folder. `prior.n_sigma` must be null: the box is the file's.
A site that keeps its runs and job logs in a folder of its own beside CLiENT sets `CLIENT_HPC_DIR` to that folder:
the launcher then writes `logs/` there, and `report.py`, `corner.py` and `make_box.py` read `runs/` there.

```bash
mpirun -n N python launch.py runs/RUN/domain_bounds.json runs/RUN/client.yaml -n RUN -o results   # a new run
mpirun -n N python launch.py runs/RUN/domain_bounds.json results/<timestamp>_RUN -i 2             # continue
```

Compared with `client.py` itself, the launcher

- applies the training box of the file on all ranks, leaving Cobaya's prior densities as they are, and stops when
  a continued run is given another box;
- rounds batch points to single precision before they are evaluated, so that the coordinates in the training
  files are exactly those the likelihood saw;
- turns a logged error of Cobaya at a point, and a log-posterior of NaN, into a refused point, in place of
  stopping the run or withholding corrected weights;
- writes `logs/backend_<tag>.json` (the Boltzmann code, the commits of CLiENT and CLASS++, the box, the likelihoods)
  and one line per batch of true evaluations to `logs/evaluations_<tag>.jsonl`. CLiENT's commit is read from its
  checkout; CLASS++'s is recorded only when the job names it, as `CLASSPP_COMMIT` or `CLASSPP_BUILD_COMMIT`.

A run's `runs/<RUN>/job.env` is sourced by the job script and can set these:

| variable | unset | meaning |
|---|---|---|
| `CLIENT_REFUSED_VALUE` | refused points are -inf and are dropped | a number, or `design[:Q]`, the quantile Q (0.5 unless given) of the design's finite values: refused points get this value and are trained on. The value belongs to the run (`refused_value.json`), and a continued run must be given the same one. It is never a finite true value: the true check and the evidence correction count refused points by it |
| `CLIENT_LOAD_GROUP` | all at once | ranks build their likelihood this many at a time, for likelihoods that need far more memory while reading their data (HiLLiPoP) |
| `CLIENT_ENV` | not checked | the conda environment whose Boltzmann code the job must evaluate with |
| `CLASSPP_GUARD_DIR` | not used | classy must come from this folder's `lib/`: the patched build (see below) |
| `CLASSPP_BUILD_DIR`, `CLASSPP_BUILD_COMMIT` | not used | classy must come from this folder's `lib/`: an unpatched build of another CLASS++ commit, the one `CLASSPP_BUILD_COMMIT` names. Not with `CLASSPP_GUARD_DIR` |
| `CLASSPP_COMMIT` | not recorded | the commit of the environment's own CLASS++ build, for `logs/backend_<tag>.json` |

The settings for a real likelihood are `input/recommended_near_gaussian.yaml` and `input/recommended_non_gaussian.yaml`;
`docs/boost.md` says which to take.

## Checking a network

```bash
python analysis/report.py results/<timestamp>_RUN                                      # T = 1 chains and a summary
mpirun -n N python analysis/true_check.py results/<timestamp>_RUN --cobaya runs/RUN/cobaya.yaml \
    --domain runs/RUN/domain_bounds.json --networks 10 15 --points 500
python ../client.py evidence results/<timestamp>_RUN -i 15
python analysis/evidence_summary.py results/<timestamp>_RUN
```

The true check evaluates the true likelihood at posterior samples of a network: the direct measure of the network's
error where its posterior is. The summary corrects each evidence by those true values. It withholds the correction
when more than 5% of the evidence's own posterior lies where the true check has no sample, when the two do not
record the same target, and when a chain's weights are marked as not reliable.

## Branching a run

```bash
python analysis/branch_run.py results/<timestamp>_RUN --at 10 --set acquisition.boost.region=1.5
```

makes a new run directory that holds the run up to network 10 and starts its boosts there; the network must lie at or
before the source's own boost start. The source run stays as it is, and the branch keeps the source's call journals
apart from its own. Continue the branch as any run.

```bash
python analysis/branch_run.py results/<timestamp>_RUN --train-from 8 --set training.warm_start=false
```

branches inside the boosts instead: the branch holds networks 0 to 7 and the training set of network 8, keeps the
source's boost start, and trains network 8 and the later ones itself with the settings given.

## True profiles

```bash
python true_profiles/make_config.py results/<timestamp>_RUN --iteration 25 --out runs/RUN/profiles.yaml
sbatch true_profiles/submit_profiles.sh results/<timestamp>_RUN runs/RUN/profiles.yaml results_profiles/NAME
python true_profiles/status.py results_profiles/NAME
```

`make_config.py` takes `--hybrid-hops` and `--gtol` for the network scan, which a rugged network may need above
CLiENT's defaults (12 and 1e-4), and sets the study's call budget to what its plan can use unless `--budget` is given.
The profile code starts its own backend processes: do not use `mpirun`. A refused point is -inf in a profile, never
the run's refused value. `CLIENT_CAMB_ACCURACY_TARGET` and `CLIENT_PROFILE_NEWTON` (overrides of `NewtonConfig`,
outside what that stage was validated with) are written beside the study, and a resumed study must be given the same.

## CLASS++ for spectra with dark acoustic oscillations

`patches/classpp_extrap_guard.patch` changes one case of `NonlinearModule::nonlinear_get_source`, the extrapolation of
the sources beyond the computed k that halofit and HMcode integrate over. Where the linear spectrum has dark acoustic
oscillations the last two table values can be nearly equal, and the default two-point fit then over- or underflows, which
gives a NaN P(k). Where the fit is well posed the expression is the original, and the spectra were bit for bit those of the
unpatched build at 40 random points and the best fit. It is a regularisation of uncalibrated halofit and HMcode output, not a model.

Apply it at the CLASS++ commit the run names and build into a folder of its own, source under `src/` and the Python extension
under `lib/` (CLASS++ compiles the path of its source tree into the library). Then set `CLASSPP_GUARD_DIR` to that
folder in the run's `job.env` and put its `lib/` first on `PYTHONPATH`. Leave the environment's own classy untouched.
An unpatched build of another commit is made and named the same way, with `CLASSPP_BUILD_DIR` and `CLASSPP_BUILD_COMMIT`.
On several nodes the job must pass these variables and `PYTHONPATH` to every rank (`mpirun -x`, as `submit_client.sh` does).
