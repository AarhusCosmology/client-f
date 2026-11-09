# SIν+DRMD paper: scripts and run files

This folder holds the scripts and parameter files behind the paper *Self-Interacting Neutrinos in NEDE and DRMD
Cosmologies: Posteriors, profiles and evidence using CLiENT*. The rest of the branch is client-f itself, whose `hpc/`
holds the generic cluster tools (the launcher, the training box, refused points, branching, the true check, the report,
the evidence summary and the true profiles). The programs here use those tools and keep no copy of them.

No results are included, since the scripts regenerate them: the emulators, chains, profiles and evaluations of the true
likelihood. The same holds for the lists of points that the true-likelihood jobs evaluate (`smoke/*.json`), which the
setup scripts in `paper/` write.

## Contents

| Path | What |
|---|---|
| `runs/` | The run files: CLiENT's and Cobaya's input files, the training box and the job environment of each run |
| `submit_client.sh`, `env.sh` | The job script and the pinned versions. Set your own partition, modules and paths. `env.sh` sets `CLIENT_REPO` (this checkout) and `CLIENT_HPC_DIR` (this folder), and CLiENT's `hpc/` tools then keep their `runs/` and `logs/` here |
| `install/` | Recipes for the environment, CLASS++, BEST and the likelihoods |
| `analysis/` | Job scripts for branching, the true check, the report and the evidence, and `best_posterior.py`, `make_cut_run.py` and `aic/`. The `queue_*.sh` scripts are a record of the pipelines as they ran |
| `true_profiles/` | Job scripts for the profile likelihoods of the true likelihood |
| `smoke/` | Programs that evaluate or minimise the true likelihood at given points, and the checks of the noise and of k_max |
| `paper/` | `build.py` and `build_wc.py`, which write the numbers and tables of the paper, `plots/` for the figures, and the setup scripts of the true-likelihood jobs |
| `studies/` | Programs of the smaller studies that the builds read |
| `client_paths.py` | Puts CLiENT's `hpc/` tools on the path of the programs here |

## How to reproduce the results

The steps follow the order in which the results were made. A rerun cannot repeat them bit for bit (see the last
section), so the agreement is statistical.

### 1. Installation

Use this folder as your cluster folder, since the jobs run from it:

```bash
cd papers/sidr_drmd && source env.sh
```

`env.sh` holds all the version pins. Set your own locations, and `GCC_MODULE` and `MPI_MODULE`.

| Step | Script | Version |
|---|---|---|
| Python environment | `install/01_conda.sh`, `02_env.sh` | CLiENT's `environment.yaml`, except scipy 1.15.2, conda-forge only and mpi4py 4.1.2 built against your MPI |
| CLASS++ for PDP+ | `install/09_classpp_guard.sh` | `7cc41a51` with client-f's `hpc/patches/classpp_extrap_guard.patch` |
| CLASS++ for H0WC | `install/10_classpp_build.sh bfe6729e` | `bfe6729e`, unpatched |
| BEST | `install/08_best.sh` | `b5cd08b`, from a clone through `BEST_PATH`, not pip |
| Likelihoods | `install/05_likelihoods.sh`, `06_external_likelihoods.sh`, `11_lite_likelihoods.sh` | Planck low-ℓ, CamSpec, DESI DR2, Pantheon+ and PR4 lensing at `e08f1dc`. ACT-lite at `627aeafb`, candl 2.2.0, `spt_candl_data` at `f22611a9`, MUSE by sha256 and `act_dr6_lenslike` 1.2.1 |

CLASS++ comes from https://github.com/AarhusCosmology/CLASSpp_public. Some traps:

- Set the compilers by absolute path. On the cluster we used, CMake otherwise took an old system gcc, and the build
  then crashed with a segmentation fault.
- Install the likelihood data by name. `cobaya-install <yaml>` puts a CLASS or CAMB under `packages_path/code`, and
  Cobaya then uses it in place of CLASS++.
- BEST in a batch job needs `COLUMNS=100 LINES=40`. `analysis/submit_evidence.sh` sets them.
- `install/check.py` was written for the cluster the runs used. It runs only with `CLIENT_RUN_CHECK` set, so leave
  that unset elsewhere.

### 2. Training the emulators

Each dataset is trained in two parts in Δ*N*eff, with boxes [0, 0.13] and [0.07, 2], which are combined across 0.08
to 0.12.

| Dataset | Run folders | CLASS++ | Ranks |
|---|---|---|---|
| PDP+ | `sidr_drmd2_pp_{low,high}` | `7cc41a51`, patched | 48 |
| PDP+SH0ES | `sidr_drmd2_ppshoes_{low,high}` | `7cc41a51`, patched | 48 |
| H0WC | `sidr_drmd_wc_{low,high}` | `bfe6729e` | 96, two nodes |
| H0WC+SH0ES | `sidr_drmd_wcmb_{low,high}` | `bfe6729e` | 96, two nodes |

```bash
sbatch submit_client.sh sidr_drmd2_pp_low
```

Use `--nodes=2 --ntasks=96` for H0WC. Do not pass `-i`: the convergence test sets the hand-over iteration S, and ten
boosts follow. Then combine the parts at S+1, S+5 and S+10 (the first two serve only the accuracy table and the spread
of the fits):

```bash
python ../../client.py combine results/<time>_sidr_drmd2_pp_combined_boost10 --parameter delta_Neff_drmd \
    --part results/<low>:<S_low+10> --part results/<high>:<S_high+10> --band 0.08:0.12
```

In the paper's runs:

| Part | S | Final network | Hours |
|---|---|---|---|
| `pp_low`, `pp_high` | 9, 7 | 19, 17 | 22.8, 19.9 |
| `ppshoes_low`, `ppshoes_high` | 8, 8 | 18, 18 | 22.3, 20.8 |
| `wc_low`, `wc_high` | 7, 7 | 17, 17 | About two days |
| `wcmb_low`, `wcmb_high` | 6, 7 | 16, 17 | About two days |

`analysis/queue_sidr_drmd2_combined.sh` and `queue_sidr_drmd_wc_combined.sh` show the whole pipeline after the
training. They hold the job numbers and S values of the paper's runs, so use them as a record and not to run.

### 3. Analysis of the emulators

Set `OUT` to a combined run after ten boosts, and `NAME` to its folder under `runs/`:

```bash
sbatch --array=0 analysis/submit_evidence.sh $OUT --seed 42
sbatch --array=0 analysis/submit_evidence.sh $OUT --seed 43 --tag seed43
sbatch analysis/submit_report.sh $OUT --networks 0 --corner 0 --corner-parameters 12
sbatch --time=12:00:00 analysis/submit_true_check.sh $OUT --networks 0 --points 500
python analysis/best_posterior.py $OUT --split logGeff:-2.5 --split delta_Neff_drmd:0.1
```

The evidences used the seeds 42 to 45, and 42 to 53 for H0WC, with 2048 live points, 128 updates and a tolerance of
0.001. Each run took 56 to 111 minutes on 16 cores. The posterior draws are new in a rerun, so the shares of the modes
vary with the seeds, by the scatter that `best_posterior.py` reports.

**The cut Δ*N*eff ≥ 0.01,** used for PDP+ and H0WC, makes a run of its own from a finished one:

```bash
python analysis/make_cut_run.py $OUT --parameter delta_Neff_drmd --lower 0.01 --suffix dncut
```

**Profiles of the true likelihood:**

```bash
python $CLIENT_REPO/hpc/true_profiles/make_config.py $OUT --iteration 0 --out runs/$NAME/profiles.yaml --points 25 --hybrid-hops 30 --gtol 1e-3 --parameters ...
sbatch true_profiles/submit_profiles.sh $OUT runs/$NAME/profiles.yaml results_profiles/${NAME}_cosmo
```

| Dataset | Studies used |
|---|---|
| PDP+ | `_cosmo_gtol3`, `_ocdm_maxfun` (maxfun 276, budget 7000) and six studies inside the cut, `_dncut_{H0,fidm,m_int,omega_cdm,zstop3,omega_b3}` |
| PDP+SH0ES | `_cosmo` and `_fidm_wide` (`--range f_idm_drmd 0 0.4`) |
| H0WC | `_dncut_cosmo`, `_dncut_fidm_wide` and `_dncut_logGeff_wide` (`--range logGeff -5.5 -0.8 --region 2`) |
| H0WC+SH0ES | `_cosmo`, `_fidm_wide` and `_logGeff_wide` (`--region 2`) |

The exact settings of `_gtol3`, `_ocdm_maxfun`, `_fidm_wide` and the H0WC `_dncut_*` studies beyond these are not
recorded here.

### 4. Further evaluations of the true likelihood

These programs evaluate or minimise the true likelihood at chosen points. The setup scripts write the lists of points
into `smoke/` from the results of steps 2 and 3.

| Work | Program | Points made by |
|---|---|---|
| Node-by-node profile chains, H0WC, about 116 000 evaluations | `smoke/profile_chain.py` | `paper/wc_chain_setup.py` |
| Polishing of the profile nodes | `smoke/chi2_points_mp.py` | `paper/pp_polish_setup.py`, `wc_polish_setup.py`, `profile_polish.py` |
| Plateau points, H0WC+SH0ES | `smoke/chi2_points_mp.py` | `paper/plateau_points.py` |
| Minimisations: DRMD, ΛCDM, valley, mass floor | `smoke/minimise_true.py` | `paper/drmd_minimise_points.py`, `drmd_gap_setup.py`, `wc_region_minimise_setup.py`, `lcdm_minimise_setup.py`, `wc_drmd_dao_setup.py`, `pp_region_minimise_setup.py` |
| χ² by likelihood | `smoke/chi2_breakdown.py`, `true_profiles/node_points.py` | `analysis/aic/make_points.py`, `studies/20261024_chi2_compare/make_points.py` |
| Emulator best fits, AIC | `analysis/aic/aic_preview.py`, `profile_logGeff.py` | |
| Noise of the likelihood | `true_profiles/noise_probe.py`, `smoke/noise_probe_cobaya.py` | `studies/20261023_noise_variants/make_variants.py` |
| The extrapolation patch, k_max ladders | `smoke/submit_guard_test.sh`, `smoke/kmax_ladder.py`, `smoke/compare_ladders.py` | |

Some of these jobs were stopped by hand once the result was flat, so running them to the end will not repeat them
exactly.

### 5. Numbers, tables and figures

```bash
export CLIENT_PAPER_KIT=$PWD                # the cluster folder: its runs/, logs/, smoke/ and results/
export CLIENT_PAPER_DATA=/path/to/data      # holds results/<date>_*: the emulator best fits and the smaller studies
cd paper && mkdir -p tables plots/data Figures && python build.py --label boost10 && python build_wc.py
cd plots && python make_all.py && python make_all.py --series wc
```

`build.py` writes the material of PDP+ and PDP+SH0ES, and `build_wc.py` that of H0WC and H0WC+SH0ES. Together they
write `numbers.tex` and `numbers_wc.tex` (the numbers quoted in the text), the nine tables and the data of the ten
figures, which `plots/` then draws. The builds find CLiENT in the checkout they lie in.

The builds read the outputs of the jobs by the names the original pipeline gave them, for example
`logs/chi2_breakdown_<data>_<job>.json` or `results/20261022_s10_best_fits` (written by `analysis/aic/`), and in some
places they take the newest matching file. A rerun must give its outputs the same names, or the paths at the
top of the builds must be changed.

## What a rerun reproduces

- **Not bit for bit, on any code.** AIES proposals under XLA do not reproduce from a seed, the Boltzmann code has a
  noise floor and the points CLASS++ refuses depend on the machine. The hand-over iteration comes from a convergence
  test and may fall elsewhere. How close a rerun comes to the paper's emulators has not been measured.
- **CLiENT.** The runs used development versions of CLiENT from before client-f was made, whose commits are not in
  client-f's history. Reading the two codes we found no difference that could change the results. The run files here
  leave out `acquisition.batch_size` and `training.warm_start_preserve_function`, which client-f refuses, and this
  does not change the behaviour.
- **Data.** The likelihood data installed through Cobaya has no version pins.
- **Tested on this branch:** the run and profile files load with CLiENT's schema and all shell scripts parse
  (`tests/test_papers.py`). CLiENT's cluster tools and `best_posterior.py` were run on a small analytic problem, and
  `profile_chain.py`, `minimise_true.py` and `chi2_points_mp.py` on analytic Cobaya models. Nothing here has been run
  with CLASS++ or a real likelihood from this branch, and the `submit_*.sh` and `queue_*.sh` scripts have not been
  submitted from it.
