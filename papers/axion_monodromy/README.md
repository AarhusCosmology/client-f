# Axion monodromy paper: scripts and run files

This folder holds the scripts and parameter files behind the paper *Features investigated: A detailed study of axion
monodromy inflation using CLiENT*. The rest of the branch is client-f itself, whose `hpc/` holds the generic cluster
tools (the launcher, the training box, refused points, branching, the true check, the report, the evidence summary and
the true profiles). The programs here use those tools and keep no copy of them.

No results are included, since the scripts regenerate them: the emulators, chains, profiles and evaluations of the true
likelihood. The data that the paper's numbers, tables and figures are built from are kept apart, with a manifest of
every file the build reads (see "Numbers, tables and figures").

## Contents

| Path | What |
|---|---|
| `runs/` | The run files: CLiENT's and Cobaya's input files, the training box and the job environment of each run, and the profile settings of the two combined emulators (`runs/axion_{pp,wc}_combined200/profiles_*.yaml`, with `windows/` and `fallback/`) |
| `submit_client.sh`, `env.sh` | The job script and the pinned versions. Set your own partition, modules and paths. `env.sh` sets `CLIENT_REPO` (this checkout) and `CLIENT_HPC_DIR` (this folder), and CLiENT's `hpc/` tools then keep their `runs/` and `logs/` here |
| `install/` | Recipes for the environment, CLASS++, BEST and the likelihoods |
| `analysis/` | Job scripts for the evidence, the report, the true check and the snapshots of the profile studies. The `queue_*.sh` and `resubmit_*.sh` scripts are a record of the pipelines as they ran, and `act_lite_bins.py` prints the bins of ACT-lite |
| `true_profiles/` | The job script of the profile likelihoods of the true likelihood |
| `smoke/` | `axion_probe.py`, the scan of the likelihood along omega at a reference point and the precision of CLASS++ at its maxima, and `axion_points.py`, the true likelihood at given points |
| `studies/` | The programs of the two smaller studies that the build reads: the sampling of CLASS++ at high frequency (`20261102_high_omega_ladder`, which also wrote the run and profile files) and the cost of standard samplers (`20261107_sampler_cost`) |
| `paper/` | `build200.py`, which writes the numbers and tables of the paper, `plots/` for the figures, and `analysis/bands200.py` and `binning200.py`, which write two of the build's inputs |
| `client_paths.py` | Puts CLiENT's `hpc/` tools on the path of the programs here |

## How to reproduce the results

The steps follow the order in which the results were made. A rerun cannot repeat them bit for bit (see the last
section), so the agreement is statistical.

### 1. Installation

Use this folder as your cluster folder, since the jobs run from it:

```bash
cd papers/axion_monodromy && source env.sh
```

`env.sh` holds the version pins of the environment, BEST and the likelihoods, as for the SInu+DRMD paper
(`install/`). Both datasets of this paper use CLASS++ `0a57bbc6`, unpatched, which each run's `job.env` selects
(`CLASSPP_BUILD_DIR`):

```bash
bash install/10_classpp_build.sh 0a57bbc6
```

CLASS++ computes the primordial spectrum with `runs/axion_pp_high200/generate_Pk.py` at every evaluation (Cobaya's
`P_k_ini type: external_Pk`), with a table of 1000 points per decade in k (`custom7`). The command is relative to this
folder, so the jobs must run from it.

### 2. Training the emulators

The prior volume is split in omega into seven parts for PDP+ and eight for H0WC (Table 8 of the paper). Each part is an
ordinary run of CLiENT with the same likelihood and the same prior, on two nodes:

```bash
sbatch --nodes=2 --ntasks=96 submit_client.sh axion_pp_zone3
```

Do not pass `-i`: the convergence test sets the hand-over iteration S, and ten boosts follow. In the paper's runs:

| Dataset | Part | Box in omega | S | Final network |
|---|---|---|---|---|
| PDP+ | `axion_pp_low200` | [1, 3.1] | 5 | 15 |
| | `axion_pp_zone1a` | [2.5, 39.25] | 6 | 16 |
| | `axion_pp_zone1b` | [35.75, 63] | 4 | 14 |
| | `axion_pp_zone2a` | [59.5, 89.75] | 5 | 15 |
| | `axion_pp_zone2b` | [86.25, 113] | 4 | 14 |
| | `axion_pp_zone3` | [109.5, 156] | 4 | 14 |
| | `axion_pp_zone4` | [152.5, 200] | 4 | 14 |
| H0WC | `axion_wc_low200` | [1, 6.1] | 6 | 16 |
| | `axion_wc_zone1a` | [5.5, 39.25] | 6 | 16 |
| | `axion_wc_zone1b` | [35.75, 63] | 5 | 15 |
| | `axion_wc_zone2a` | [59.5, 89.75] | 5 | 15 |
| | `axion_wc_zone2b` | [86.25, 113] | 4 | 14 |
| | `axion_wc_zone3` | [109.5, 156] | 4 | 14 |
| | `axion_wc_zone4a`, `zone4b` | [152.5, 176.25], [172.75, 200] | 5, 4 | 15, 14 |

The run folders `axion_pp_high200`, `axion_pp_zone1`, `axion_pp_zone2`, `axion_wc_high200` and `axion_wc_zone1` are the
emulators that were split later (Appendix A.2 of the paper). They are not part of the combined emulators, but the
paper's statistic of the pilot points counts their networks.

The parts of a dataset are combined over the bands of Table 8 with CLiENT's `combine`, from their final networks:

```bash
python $CLIENT_REPO/client.py combine results/<time>_axion_pp_combined200 --parameter omega \
    --part results/<time>_axion_pp_low200:15 --part results/<time>_axion_pp_zone1a:16 ... --band 2.6:3.0 --band 36.75:38.25 ...
```

`analysis/queue_axion_combined200.sh` shows the whole pipeline after the training as it ran, with the job numbers of the
paper's runs: use it as a record and not to run. The paper's combined emulators are
`20261107_084147_axion_pp_combined200` (seven parts) and `20261107_084828_axion_wc_combined200` (eight parts). The
profile studies of PDP+ ran on `20261106_090834_axion_pp_combined200`, the six parts before zone 1 was split at
omega = 37.5. Their values are those of the true likelihood, and the emulator only gives them their starting points.

### 3. Analysis of the emulators

Set `OUT` to a combined run:

```bash
sbatch --array=0 analysis/submit_evidence.sh $OUT --seed 42
sbatch --array=0 analysis/submit_evidence.sh $OUT --seed 43 --tag seed43
sbatch --array=0 analysis/submit_evidence.sh $OUT --seed 42 --tag lcdm --prior A_osc=0:1e-4
sbatch --time=12:00:00 analysis/submit_true_check.sh $OUT --networks 0 --points 500
```

The evidences used the seeds 42 to 45 with 2048 live points, for the oscillation model and for the featureless
spectrum, which is the same emulator with A_osc uniform on [0, 1e-4]. `analysis/queue_axion200_evidence.sh` is the
record. The runs' prior on A_osc is [0, 0.5], the training box [0, 0.1], and the paper quotes all results for the prior
[0, 0.1]: `paper/build200.py` adds ln 5 to the evidences of the oscillation model.

**Profiles of the true likelihood.** The three planes of each dataset, (A_osc, omega), (A_osc, phi) and (omega, phi), are
studies on 50 x 50 grids, split into windows in omega and joined by the build:

```bash
sbatch true_profiles/submit_profiles.sh $OUT runs/axion_wc_combined200/profiles_omega_phi_w1.yaml \
    results_profiles/axion_wc200_omega_phi_w1 --iteration 0
bash analysis/snapshot_profiles.sh axion_wc200_omega_phi_w1     # copy a study into logs/, which the build reads
```

The study names of the paper are `axion_<pp|wc>200_` followed by the name of the YAML without `profiles_`. A study that
reaches the time limit of its job continues when it is submitted again with the same arguments, `--iteration 0`
included. `analysis/resubmit_wc_profiles200.sh` is the record of the resubmissions of H0WC.

The profiles of H0WC were closed by hand on 13 November 2026, from the snapshot of 15:07, with 34 697 of the 35 000
nodes refined with the true likelihood. The other 303, all at A_osc >= 0.084 and more than 5.5 below the best fit in
chi^2, keep the value of the emulator, as the caption of Fig. 6 says (`WC_CLOSED` in `paper/build200.py`).

### 4. Further evaluations of the true likelihood

| Work | Program |
|---|---|
| The scan along omega at a reference point (Sec. 2.2, Table 8) and the precision of CLASS++ at its maxima (Appendix B, Table 9) | `smoke/axion_probe.py`, `smoke/submit_axion_probe.sh` |
| The spectra of CLASS++ with several samplings (Appendix B, Table 10) | `studies/20261102_high_omega_ladder/` (`ladder3.py`, `compare3.py`, `compare_lensing.py`, `run_all.sh`) |
| The true likelihood at given points | `smoke/axion_points.py`, `smoke/submit_axion_points.sh` |
| The cost of standard samplers, on the combined emulators (Appendix C, Table 11) | `studies/20261107_sampler_cost/` (`run_mcmc.py`, `run_nested.py`, `prospect/`, `summarise.py`) |

The programs of `studies/` are as they ran and read their inputs from the cluster folder (`CLIENT_PAPER_KIT`).

### 5. Numbers, tables and figures

```bash
export CLIENT_PAPER_KIT=$PWD                # the cluster folder: its runs/, logs/ and results/
export CLIENT_PAPER_DATA=/path/to/data      # holds results/20261102_high_omega_ladder and results/20261107_sampler_cost
cd paper && python analysis/binning200.py && python analysis/bands200.py
python build200.py --profiles && cd plots && python make_all.py
```

`build200.py` writes `numbers200.tex` (the numbers quoted in the text), the tables in `tables/` and the data of the
figures in `plots/data/`, which `plots/make_all.py` draws into `Figures/`. It reads the outputs of the jobs by the names
the original pipeline gave them, for example `logs/axion_probe_<job>.json` or `results/<time>_axion_wc_combined200`, so
a rerun must give its outputs the same names. `--profiles` rereads the node files of the profile studies; without it the
build uses its cache (`CLIENT_PAPER_CACHE`). `analysis/binning200.py` needs `COBAYA_PACKAGES_PATH` (the ACT and SPT data)
and `analysis/bands200.py` the networks of the parts.

## What a rerun reproduces

- **Not bit for bit, on any code.** AIES proposals under XLA do not reproduce from a seed, the Boltzmann code has a
  noise floor and the points CLASS++ refuses depend on the machine. The hand-over iteration comes from a convergence
  test and may fall elsewhere. How close a rerun comes to the paper's emulators has not been measured.
- **CLiENT.** The runs used development versions of CLiENT from before client-f was made, whose commits are not in
  client-f's history. The run and profile files here leave out `acquisition.batch_size`,
  `training.warm_start_preserve_function` and the profile keys `true_optimizer`, `corrected_method` and
  `surface_methods`, which client-f refuses: their values in the runs are client-f's only behaviour (the function
  preserved at a warm start, and the residual Newton refinement of a surface). The profile code of client-f and of the
  version the studies ran with was compared function by function, and the only differences found were in the counts of
  calls in a node's diagnostics and in the handling of points that CLASS++ refuses.
- **Data.** The likelihood data installed through Cobaya has no version pins.
- **Tested on this branch:** the run and profile files load with CLiENT's schema and all shell scripts parse
  (`tests/test_papers.py`). The build, run from this folder on the data of the paper, reproduces the paper's numbers and
  tables byte for byte and its figures (see the data manifest). Nothing here has been run with CLASS++ or a real
  likelihood from this branch, and the `submit_*.sh` and `queue_*.sh` scripts have not been submitted from it.
