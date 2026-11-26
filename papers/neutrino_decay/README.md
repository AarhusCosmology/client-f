# Neutrino decay paper: scripts and run files

This folder holds the scripts and parameter files behind the paper *Neutrino decay in cosmology: full solution and
constraints* (S. Hannestad and T. Tram). The rest of the branch is client-f itself, whose `hpc/` holds the generic
cluster tools (the launcher, the training box, refused points, the true check, the report and the true profiles). The
programs here use those tools and keep no copy of them.

The paper has two parts, and so has this folder. Section 4 (the full solution and its approximations) consists of
runs of the Einstein-Boltzmann solver CLASS++ alone, organised as campaigns on a Slurm cluster (`campaigns/`).
Section 5 (the constraints) consists of CLiENT runs, profile likelihoods and evaluations of the true likelihood
(`runs/`, `analysis/`, `true_profiles/`, `smoke/`). `paper/` builds every figure, table and number of the paper from
the outputs of both.

No results are included, since the scripts regenerate them: the spectra, the emulators, the chains, the profiles and
the evaluations of the true likelihood.

## Versions

Every CLASS++ commit below is on the main branch of the public repository,
<https://github.com/AarhusCosmology/CLASSpp_public>.

| Results | Code | Commit |
|---|---|---|
| Full solution and six moments: figures 1, 2 and 5, table 1 (`campaigns/hpc_prod2`) | CLASS++ | `37f95e77` (27 August 2026) |
| Isotropisation rate measured with six moments and 64 parent bins: figure 4 and the fit of A3 and A5 (`campaigns/hpc_gal2`, `campaigns/hpc_gal`) | CLASS++ | `cbf98030` (30 August 2026) and a build of 29 August (see below) |
| The departure from detailed balance of the background, epsilon, for the same fit (`campaigns/proxy_backgrounds`) | CLASS++ | builds of 18 August and 14 September (see below) |
| Proxy in figure 2 (`campaigns/proxy_rta`) | CLASS++ | `85deb9fc` (9 September 2026) |
| Proxy cost in table 1 (`paper/proxy_timings_laptop.txt`) | CLASS++ | `cb2255f4` (5 November 2026) |
| CLiENT runs, true checks, posteriors with a prior flat in Gamma, profiles, LambdaCDM pair: section 5, figures 6-9, table 2 | CLASS++ and CLiENT | `cb2255f4`, unpatched; client-f `918dc19` |

The amplitudes of the isotropisation rate, A3 = 0.0743 and A5 = 0.5532 (`paper/rate/fit_structured.py`), are the
defaults of the proxy in CLASS++ from `eb758874` (14 September 2026) on, so the proxy of section 5 uses them without setting them, and
the proxy cells of figure 2 set them explicitly.

Two inputs of the rate fit were made with builds whose commit was not recorded. The logs of `campaigns/hpc_gal`
(29 August) name no commit; its six-moment rates agree with those of `hpc_gal2` (`cbf98030`) to a median ratio of
1.0000 over 1269 matched bins, and the fit uses `hpc_gal2` wherever both exist. The proxy backgrounds that give epsilon
were made on a laptop on 18 August (integer decades of Gamma) and 14 September (half decades). Rerun with `85deb9fc`, a
background of 14 September (Gamma = 10^9.5, 0.06 eV) is reproduced bit for bit, and one of 18 August (10^8, 0.3 eV)
to 0.35% in epsilon, which enters the rate through its square root.

The full solution and the moments use the linear gather of the daughter occupations and the positivity limiter that
appendix A of the paper describes. The proxy of section 5 (`cb2255f4`) uses the log-odds gather of the same appendix,
which CLASS++ turns on by default for the proxy from `f69b65a1` on (`balanced_gather`).

This branch is client-f's main of 20 November 2026 (`e17059f`), which contains `918dc19`. To run CLiENT exactly as
the paper did, check out `918dc19` and copy this folder into it. The run and profile files here load with the schema
of both.

## Contents

| Path | What |
|---|---|
| `runs/dncdm_proxy_pp`, `runs/dncdm_proxy_wc` | The two CLiENT runs (Planck+BAO+SN and SPA+BAO+SN of the paper): CLiENT's and Cobaya's input files, the training box (`domain_bounds.json`), the job environment (`job.env`, which selects the CLASS++ build) and the settings of the profile studies (`profiles_*.yaml`) |
| `runs/dncdm_proxy_lcdm` | LambdaCDM with nu_3 at 0.05 eV and massless (the dotted lines of figure 8), and the chi^2 of every likelihood at the nodes of the profiles (`node_breakdown.py`: the totals of figure 8, and the contributions of the likelihoods in `paper/fig/numbers_data.txt`) |
| `likelihoods/act_lite_trimmed.py` | ACT-lite as SPA+BAO+SN uses it (its `python_path` in `runs/dncdm_proxy_wc/cobaya.yaml`) |
| `submit_client.sh`, `env.sh`, `EXPECTED_COMMIT` | The job script and the pinned versions. Set your own partition, modules and paths. `env.sh` sets `CLIENT_REPO` (this checkout) and `CLIENT_HPC_DIR` (this folder), and CLiENT's `hpc/` tools then keep their `runs/` and `logs/` here |
| `install/` | Recipes for the environment, CLASS++ and the likelihoods |
| `analysis/` | Job scripts for the report and the true check, and `linear_prior_posterior.py`, which samples a network with a prior flat in Gamma and checks those samples against the true likelihood (client-f's `report.py` and `true_check.py` have since gained `--linear-prior` for the same purpose) |
| `true_profiles/` | The job script of the profile likelihoods of the true likelihood |
| `smoke/` | `timing.py` and `submit_timing.sh`, the cost of the true likelihood in the training box (table 2's numbers came from CLiENT's own logs, `paper/eval_costs.py`), and `eval_threads.py`, the cost of one evaluation against the number of threads (section 5.3) |
| `campaigns/` | The CLASS++ campaigns of section 4, see below |
| `paper/` | `make_all.sh` and the programs it runs, which write every figure (`fig/`), table and number of the paper |

`campaigns/` keeps the original names of the campaigns, since the programs of `paper/` find their outputs by them:

| Folder | What |
|---|---|
| `hpc_prod2/` | The ladder of the full solution and the six moments at 0.3 and 0.06 eV, Gamma = 1e5 to 1e12 km/s/Mpc: CMB spectra (`cmb_*`), backgrounds (`bg_*`) and the rate measurements at 32 parent bins (`iso_*`). `config.py` defines the ladder and its resolution rules, `gen.py` writes `ini/` (the inputs of the paper's runs, as they ran) and the run lists, and `README.md` is the campaign's own description |
| `hpc_gal2/`, `hpc_gal/` | The same for the moments with 64 parent bins, whose rate measurements (`iso_*_m6`, seven wavenumbers each) are the data of the rate fit |
| `proxy_rta/` | The proxy cells of figure 2: the proxy with the rate of the paper (`px_ours_*`) and with that of Chen et al. (`px_copw_*`), each on the input of the full-solution cell of the same Gamma and mass (above the reach of the full solution, of the six-moment cell), apart from the representation of the daughters and the rate |
| `proxy_backgrounds/` | The backgrounds of the proxy that give epsilon to the rate fit (`batch_bg.py`, `batch_bg_half.py`) |

## How to reproduce the results

The steps follow the order in which the results were made. A rerun cannot repeat the CLiENT runs bit for bit (see
the last section), so the agreement there is statistical. The CLASS++ runs of section 4 are deterministic for a given
build.

### 1. Installation

Use this folder as your cluster folder, since the jobs run from it:

```bash
cd papers/neutrino_decay && source env.sh
```

`env.sh` holds the version pins of the environment and the likelihoods (`install/`). The CLiENT runs use CLASS++
`cb2255f4`, unpatched, which each run's `job.env` selects (`CLASSPP_BUILD_DIR`):

```bash
bash install/10_classpp_build.sh cb2255f4
```

The campaigns of section 4 need a `class` binary of their own commit (the table above), built from a checkout of
CLASSpp_public. `campaigns/hpc_prod2/build.sh` builds one binary per Slurm partition, and its README says how.

### 2. The full solution and its approximations (section 4)

Each `hpc_*` campaign is generated, built and submitted from its own folder, as its README describes:

```bash
cd campaigns/hpc_prod2
python3 gen.py                              # writes ini/, the run lists and manifest.csv (ini/ is already here)
./build.sh /path/to/CLASSpp_public q64      # at 37f95e77; writes BUILD_COMMIT, which runcell.sh checks
sbatch submit_bg.sh && sbatch submit_cmb.sh && sbatch submit_iso.sh
python3 reduce_iso.py                       # the rate measurements: derived/*.json
```

`hpc_gal2` (at `cbf98030`) and `hpc_gal` are run the same way; the rate fit reads their `iso_*_m6` cells. The proxy
cells of figure 2 and the proxy backgrounds run on any machine:

```bash
cd campaigns/proxy_rta && CLASS=/path/to/class ./run.sh 4                          # class at 85deb9fc
cd campaigns/proxy_backgrounds && CLASS=/path/to/class python3 batch_bg.py && CLASS=/path/to/class python3 batch_bg_half.py
```

The rate fit is `paper/rate/fit_structured.py`, which prints A3 and A5 (and `paper/rate/fig_ratefit.py` draws figure 4).

### 3. Training the emulators

Each dataset is an ordinary run of CLiENT (in the paper, three and four nodes of 48 cores):

```bash
sbatch --nodes=3 --ntasks=144 submit_client.sh dncdm_proxy_pp
sbatch --nodes=4 --ntasks=192 submit_client.sh dncdm_proxy_wc
```

Do not pass `-i`: the convergence test sets the hand-over iteration, and ten boosts follow. In the paper's runs the
hand-over came at iteration 4 for both, and the final network is network 14. The paper's runs are
`results/20261106_121906_dncdm_proxy_pp` and `results/20261106_122443_dncdm_proxy_wc`.

### 4. Analysis of the emulators

Set `OUT` to a run folder:

```bash
sbatch analysis/submit_report.sh $OUT
sbatch --time=03:00:00 analysis/submit_true_check.sh $OUT --networks 14 --points 500
python analysis/linear_prior_posterior.py sample $OUT --network 14 --parameter log10Gamma
python analysis/linear_prior_posterior.py check $OUT --cobaya runs/dncdm_proxy_pp/cobaya.yaml --network 14
```

The report writes the T = 1 chain of each network (`analysis/posterior_it_14.npz`, a prior flat in log10Gamma), and
`linear_prior_posterior.py sample` the chain with a prior flat in Gamma (`analysis/linear_prior/`). The paper's checks
of the latter used 500 points for Planck+BAO+SN and 300 for SPA+BAO+SN.

**Profiles of the true likelihood.** The header of each `runs/dncdm_proxy_*/profiles_*.yaml` gives its command, for
example

```bash
sbatch true_profiles/submit_profiles.sh results/20261106_122443_dncdm_proxy_wc runs/dncdm_proxy_wc/profiles_peak_log10Gamma.yaml results_profiles/dncdm_proxy_wc_peak_gamma
```

The study names of the paper are `dncdm_proxy_<pp|wc>_` followed by `gamma` (`profiles_log10Gamma.yaml`), `peak_gamma`,
`peak_ns`, `gamma_ocdm` and, for Planck+BAO+SN, `peak_gamma_wcstart`, which `runs/dncdm_proxy_pp/start_from_wc_peak.py`
starts from the best fit of SPA+BAO+SN. A study that reaches the time limit of its job continues when it is submitted
again with the same arguments. The studies `gamma_ocdm` (log10Gamma and omega_cdm) were not used in the paper.

**LambdaCDM and the likelihoods at the nodes.** `runs/dncdm_proxy_lcdm/make_inputs.py` writes the inputs of the
LambdaCDM minimisations from the runs, `submit_minimize.sh` runs them, and `submit_breakdown.sh` evaluates every node
of the 1D profiles likelihood by likelihood. Their outputs are `logs/lcdm_minimize_<job>.json` and
`logs/node_breakdown_<job>.json`.

### 5. Numbers, tables and figures

```bash
export DNCDM_HARNESS=/path/to/campaigns     # the campaign folders of step 2 with their out/ and logs/
export CLIENT_PAPER_KIT=$PWD                # this folder, with the results/, results_profiles/ and logs/ of steps 3-4
bash paper/make_all.sh
```

`make_all.sh` writes the figures of the paper into `paper/fig/` and prints the fit of the rate, table 1
(`cost_table.py`) and table 2 (`eval_costs.py`). `fig/numbers_data.txt` holds the numbers of section 5. The programs
read the outputs by the names the jobs gave them, for example `logs/node_breakdown_7063596.json` (`fig_data.py`
says which), so a rerun must either give its outputs the same names or change them there. The proxy column of table 1
is in `paper/proxy_timings_laptop.txt`; `paper/time_proxy.sh` measures it again.

## What a rerun reproduces

- **CLiENT, not bit for bit.** AIES proposals under XLA do not reproduce from a seed, the Boltzmann code has a noise
  floor and the points CLASS++ refuses depend on the machine. The hand-over iteration comes from a convergence test
  and may fall elsewhere. How close a rerun comes to the paper's emulators has not been measured.
- **CLASS++.** A campaign rerun with the same commit and inputs gives the same spectra up to the round-off of the build
  (compiler and machine), which is far below the differences the paper discusses. One cell of `proxy_rta` rerun on a
  laptop build of `85deb9fc` reproduced the original output bit for bit.
- **Data.** The likelihood data installed through Cobaya has no version pins.
- **Tested on this branch:** the run and profile files load with CLiENT's schema and all shell scripts parse
  (`tests/test_papers.py`). `paper/make_all.sh`, run from this branch on the outputs of the paper's jobs, writes all
  figures and tables of the paper; figures 3-9 and `numbers_data.txt` are identical, pixel for pixel and byte for
  byte, to those made before this branch existed, and figures 1 and 2 of the paper are this branch's. The proxy cells
  of figure 2 ran with `campaigns/proxy_rta`'s inputs on a laptop build of `85deb9fc`, and a cell rerun on that build
  reproduced an earlier output bit for bit. `smoke/eval_threads.py` ran on the laptop on 20 November 2026 and
  reproduced the best fit's logpost (-2867.212); it took 243 s on one thread, 119 s on three and 53 s for the
  background, against 306, 164 and 71 s in the measurement of 9 November that section 5.3 converts (with the factor
  1.6 from the same day's laptop), so the laptop's speed varies by about 25% from day to day. Nothing else here has
  been run from this branch: the `submit_*.sh` scripts have not been submitted from it, and the campaigns have not
  been regenerated with its `gen.py`.
