# Derived parameters

CLiENT emulates one number, the log-likelihood, as a function of the sampled
parameters. A Cobaya or MontePython chain also carries *derived* parameters: functions
of the sampled ones such as H0 (when 100θ_s is sampled), 100θ_s (when H0 is), σ8, S8
and Ω_m. This page describes how CLiENT records, emulates and reports them, and
how to make 1D and 2D profiles in them.

## Recorded with every true evaluation

Nothing needs switching on. Both backends compute their derived parameters on
every likelihood call, and CLiENT keeps whatever the likelihood input declares as
derived. A value is NaN wherever the backend has none, which includes every point whose
log-likelihood is not finite.

## Declaring derived parameters: Cobaya and MontePython

The two backends declare, name and scale derived parameters differently. CLiENT follows
each backend's own convention on input and reports every value in physical units.

### MontePython

A derived parameter is a `.param` entry with role `derived`. Its name is CLASS's own
name (anything `classy`'s `get_current_derived_parameters` accepts: `H0`,
`100*theta_s`, `sigma8`, `Omega_m`, `Omega_Lambda`, `z_reio`, `YHe`, `A_s`,
`rs_d`, `age`, ...). Any CLASS output a derived parameter needs must be requested in the
file:

```python
data.parameters['H0']           = [0, None, None, 0, 1,    'derived']
data.parameters['100*theta_s']  = [0, None, None, 0, 1,    'derived']
data.parameters['sigma8']       = [0, None, None, 0, 1,    'derived']
data.parameters['A_s']          = [0, None, None, 0, 1e-9, 'derived']
data.cosmo_arguments['output'] = 'mPk'        # sigma8 needs P(k); MontePython will not add it
data.cosmo_arguments['P_k_max_1/Mpc'] = 1.
```

- **Scale.** MontePython stores a `derived` value divided by its `scale`, 2.1 for A_s
  above. CLiENT multiplies it back and records 2.1e-9, as it does for the sampled
  parameters.
- **Role `derived_lkl`.** A value written by a likelihood rather than CLASS, currently
  only the Lyman-α likelihoods. MontePython never rescales these, so CLiENT records them
  as written, whatever `scale` says.
- **Labels.** They come from MontePython's `get_tex_name`, **without** the scale prefix,
  because the values are physical. MontePython's own chains show `10^{-9}A_s` because
  they store the scaled number. The labels of the *sampled* parameters in `metadata.json`
  carry MontePython's scale prefix, which is a convention of the wrapper.
- **Inputs for the emulator** (`inputs: auto`). The `cosmo`-role parameters: CLASS never
  sees a nuisance parameter. With any `derived_lkl` present, all parameters are used.
- **Failed CLASS runs.** MontePython returns `boundary_loglike` and leaves the *previous*
  point's derived values in place. CLiENT blanks them before every call and records NaN.
- **GetDist names.** `*`, `?` and spaces are dropped (`100*theta_s` becomes
  `100theta_s`). The CSV files and `derived_parameters.json` keep the original names.

### Cobaya

A derived parameter is any entry of `params` without a prior. Its name is Cobaya's, which
can differ from the theory code's:

```yaml
theory:
  classy:
    renames: {theta_s_100: 100*theta_s}      # only where the Cobaya name is not CLASS's
params:
  logA: {prior: {min: 2.5, max: 3.7}, drop: true, latex: '\ln(10^{10}A_s)'}
  A_s: {value: 'lambda logA: 1e-10*np.exp(logA)', latex: 'A_s'}   # also reported as derived
  theta_s_100: {latex: '100\theta_s'}          # an output of the theory code
  sigma8: {latex: '\sigma_8'}
  Omega_m: {latex: '\Omega_\mathrm{m}'}
  S8: {derived: 'lambda sigma8, Omega_m: sigma8*np.sqrt(Omega_m/0.3)', latex: 'S_8'}
```

- **Kinds.** Four kinds are recorded:
  - outputs of a theory code (`sigma8`, `theta_s_100`);
  - `derived: 'lambda ...'` functions of other parameters (`S8`);
  - a likelihood's `output_params`;
  - input parameters given by a `value: 'lambda ...'`, which Cobaya reports as derived
    unless they have `derived: false` (`A_s` above).
- **Names.** A theory output must be a name the theory code knows. CLASS++ accepts
  `100*theta_s` but not `theta_s_100`, hence the `renames` entry. With CAMB, use CAMB's
  names; see the CAMB example below.
- **Scale.** There is none; values are recorded as Cobaya reports them.
- **Extra outputs.** Cobaya asks the theory code for whatever the derived parameters need
  (`mPk` for σ8 with CLASS, for example). With CLASS++ this does not change the
  likelihood: on the Planck NPIPE TT setup it was bit-identical with and without the
  derived parameters. **With CAMB it does** (see below).
- **Left out.** Cobaya's automatic `chi2__<type>` totals (e.g. `chi2__CMB`, present for
  every likelihood with a `type`, as the Planck ones have). They are the log-likelihood
  itself, not smooth functions of the parameters.
- **Inputs for the emulator** (`inputs: auto`). Resolved from Cobaya's bookkeeping:
  - lambda dependencies (`S8` → `sigma8`, `Omega_m`);
  - input parameters that are functions of sampled ones (`A_s` → `logA`);
  - for an output of a component, every sampled parameter that reaches that component,
    as Cobaya records it (`Model.sampled_dependence`). This follows requirements between
    components, not only their parameters.

  A name that resolves to none of these makes CLiENT use all inputs.
- **Outside Cobaya's own prior.** Cobaya returns an empty list rather than NaNs. CLiENT
  records NaN.

#### Cobaya with CAMB

Tested with CAMB 2.0.4, with Cobaya's own `planck_2018_camb` parameterisation (what
`cobaya-cosmo-generator` writes):

```yaml
theory:
  camb:
    extra_args: {lens_potential_accuracy: 1, theta_H0_range: [20, 100]}
params:
  theta_MC_100: {prior: {min: 0.5, max: 10}, drop: true, renames: theta, latex: '100\theta_\mathrm{MC}'}
  cosmomc_theta: {value: 'lambda theta_MC_100: 1.e-2*theta_MC_100', derived: false}   # not recorded
  logA: {prior: {min: 1.61, max: 3.91}, drop: true, latex: '\log(10^{10} A_\mathrm{s})'}
  As: {value: 'lambda logA: 1e-10*np.exp(logA)', latex: 'A_\mathrm{s}'}                 # recorded
  H0: {min: 20, max: 100, latex: 'H_0'}          # derived, and its bounds cut the prior
  omegam: {latex: '\Omega_\mathrm{m}'}
  omega_de: {latex: '\Omega_\Lambda'}
  sigma8: {latex: '\sigma_8'}
  thetastar: {latex: '100\theta_*'}             # CAMB's theta_*, not CLASS's theta_s
  zrei: {latex: 'z_\mathrm{re}'}
  rdrag: {latex: 'r_\mathrm{drag}'}
  S8: {derived: 'lambda sigma8, omegam: sigma8*np.sqrt(omegam/0.3)', latex: 'S_8'}
```

- **Names are CAMB's.** Examples: `omegam`, `omega_de`, `zrei`, `rdrag`, `age`, `YHe`,
  `thetastar`. Values are 100θ* for `thetastar`, Gyr for `age` and Mpc for `rdrag`, as CAMB
  reports them.
- **CAMB is two components in Cobaya.** `camb.transfers` takes `ombh2`, `omch2`,
  `cosmomc_theta`, `mnu` and `tau`. `camb` takes only `As` and `ns`, requires the
  transfers and outputs everything above. Matching outputs to their own component's
  inputs therefore misses four of the six cosmological parameters: it gave
  `[logA, ns, tau]` for H0 and σ8. Hence the resolution above follows requirements.
- **σ8 changes CAMB's likelihood.** Requesting σ8, or any lambda of it (`S8`,
  `s8omegamp5`, ...), makes Cobaya switch on CAMB's matter transfer functions in
  high-precision mode. That changes the non-linear lensing of the CMB spectra. At a
  fiducial point on the Planck TT likelihoods, the log-posterior moved from −3538.10 to
  −3536.61 (Δχ² ≈ 3); none of the other derived parameters changed it at all. The shift
  is of CAMB's accuracy (with `AccuracyBoost: 2` the two differ by 0.6) and of the size
  of CAMB–CLASS++ differences. **Decide on σ8 before a CAMB run starts.** Within a run
  the likelihood is consistent, but adding or dropping σ8 on continuation changes the
  training target. A continued run whose likelihood input has changed is stopped by the
  objective identity (`docs/run_recovery.md`); continued with
  `--accept-identity-change`, `client.py` warns when its derived parameters differ from
  those it was recorded with.
- **Agreement with CLASS++** at the same physical point (Planck best fit, massless
  neutrinos with N_eff = 3.045 in both):
  - σ8, Ω_m, S8, Ω_Λ and z_reio agree to 2×10⁻⁴ relative or better;
  - H0 from `theta_MC_100` round-trips to 3×10⁻⁴;
  - CAMB's 100θ* is 1.03962 where CLASS's 100θ_s is 1.04033: the definitions of the
    decoupling redshift differ;
  - Δχ² between the codes is 1.2.
- **Failures.** A CAMB error (no H0 in `theta_H0_range`) and a point outside the prior
  both give NaN for every derived parameter.

### Summary

| | MontePython | Cobaya |
|---|---|---|
| declared as | role `derived` (CLASS) or `derived_lkl` (likelihood) | any `params` entry without a prior |
| names | CLASS's (`100*theta_s`, `Omega_Lambda`) | Cobaya's; `renames` maps them to the theory code |
| units recorded | physical: `scale` undone for `derived`; `derived_lkl` as written | physical, as reported |
| labels | `get_tex_name`, unscaled | `latex` |
| CLASS outputs they need | requested in the `.param` | added by Cobaya |
| automatic inputs | `cosmo` role (all, if any `derived_lkl`) | resolved dependencies, through component requirements |
| failed or out-of-prior point | NaN (stale values blanked) | NaN (empty list padded) |
| not recorded | — | `chi2__*` totals |
| tested end to end | Planck 2018 TTTEEE+lensing+BAO; SH0ES+BAO example | Planck NPIPE TT, ΛCDM: CLASS++ with H0 sampled; CAMB with θ_MC sampled |

## Files in a run directory

```
derived_data/
  derived_parameters.json     names, labels, scales, and the inputs they depend on
  derived_it_<N>.csv          inputs + derived values, row-aligned with training_data/data_it_<N>.csv
  derived_model_it_<N>.keras  the emulator trained on iteration N's data
  derived_history_it_<N>.csv  its training history
  derived_report_it_<N>.csv   its held-out accuracy
posterior_chains/
  surrogate_it_<N>.txt        GetDist chain: weight, -log posterior, parameters, derived
  surrogate_it_<N>.paramnames derived parameters marked with a trailing *
  surrogate_it_<N>.ranges
  surrogate_it_<N>_exact*.csv the exact check (below)
```

`training_data/data_it_<N>.csv` and `metadata.json` hold no derived values. The profile workflow
and several benchmarks read the former by column position, and profile studies
fingerprint the latter. `derived_it_<N>.csv` repeats the inputs, so it stands on its own.
On loading, CLiENT checks that its rows still match the training data.

## The emulator

The derived-parameter emulator is a separate network from the log-likelihood one,
because the problem is different. It is smooth regression of a few quantities with no
peak and little dynamic range, so it uses a plain mean-squared error on standardised
outputs rather than msre. NaN targets are masked per element. Its inputs are only the
parameters the derived ones can depend on (`inputs: auto`):
- **MontePython:** the `cosmo`-role parameters, 6 of 27 on Planck.
- **Cobaya:** the dependencies it resolves through lambdas, theory components and
  input functions.
- **Anything else:** all inputs.

Restricting the inputs matters. On the 16D banana test, with the same data and median
over three seeds, the held-out error was 4× smaller on the 4 relevant inputs than on
all 16.

The model takes the full parameter vector and selects its inputs internally, with the
normalisation built in, exactly as the log-likelihood network does. It can therefore be
called on the same positions inside the sampler's TF graph.

Train it in either of two ways:

- **During the run:** add a `derived:` section to the run YAML. It is retrained every
  iteration, right after the log-likelihood network, from all points with derived values.
  An empty section, `derived: {}`, uses the defaults.
- **Afterwards,** from the recorded data, without re-running anything:
  ```bash
  python client.py derived results/<run>            # latest iteration with derived data
  python client.py derived results/<run> -i 12 --retrain
  ```

The settings, all optional, go in the `derived:` section:

```yaml
derived:
  inputs: auto          # or: all, or a list of parameter names
  n_layers: 3
  n_neurons: 256        # chosen by a sweep: see "Network size" below
  activation: alsing
  learning_rate: 1.0e-3
  n_epochs: 3000        # a ceiling; the plateau schedule ends it
  batch_size: 256
  validation_split: 0.1
  lr_schedule: plateau  # annealing is safe here: see config/config.py DerivedConfig
```

## Posterior chain with derived columns

```bash
python client.py derived results/<run> --chain [-n STEPS] [--thin K] [--region run|none]
```

This samples the surrogate posterior at T = 1, evaluates the emulator at every chain point
(one batched call: under a second for 500k points), and writes
`posterior_chains/surrogate_it_<N>`. Load it with
`getdist.loadMCSamples(".../surrogate_it_<N>", settings={"ignore_rows": 0})`. It prints
the posterior mean and standard deviation of each derived parameter.

A network trained on boosts with `acquisition.boost.region` is sampled inside the region of
its boosts, as the evidence and the profiles use it (the latest `boost/region_it_M.npz` at or
before the network). `--region none` samples the unrestricted network; `--region run` asks
for the region by name and stops when the run has none.

When the saved region has several ellipsoids, the command samples each of their disjoint
nearest-ellipsoid cells separately, at T = 1. Overlapping ellipsoids do not count mass twice.
For normalized radii r_k(x), cell k consists of points where k is the first minimizer of
r_k(x). A bridge estimate gives each cell's integral Z_k of the **current saved surrogate**
(including its wall); its mixture probability is Z_k / sum_j Z_j. These are not the starting
walker fractions or a previous boost's true-corrected weights. A cell may contain more than
one physical mode: this partition does not discover missing modes or certify mixing.

`-n` is the production length **per cell**, so sampling cost grows with the number of cells.
The output has the same number of equal-weight rows as a single chain; rows are drawn from
the cell chains in the estimated proportions, without replacement (a cell is asked for no
more rows than it has). They are consequently not time-ordered walker trajectories. Do not
estimate autocorrelation from the shuffled output. A cell that is a small part of its
ellipsoid (one ellipsoid inside another) is started near that ellipsoid's centre; the same
code (`sampling/mixture.py`) samples such networks in `hpc/analysis/report.py` and `hpc/analysis/true_check.py`. The accompanying `.sampling.json` records each cell's weight, acceptance,
autocorrelation, bridge repeat spread and proposal efficiency, and the captured target
identity, with the sampler's method and version. Low bridge efficiency warns, on the
weights' own line and in `unreliable_cells` of the sidecar: such weights are not to be
used. A longer chain clears the mark only where the chain was too short for the bridge's
fit; the efficiency is that of a Gaussian mixture as a proposal for the cell, and a cell
it does not follow stays marked at any length (the one saved real two-mode network, a
poorly trained one: 0.047 at 20 000 steps, 0.035 at 80 000, its weight unchanged). A
failed/nonfinite estimate stops output. With short chains the evidence of a cell comes out low (0.1 in ln Z at 64
walkers and 4000 steps in 16 dimensions), which cancels between like cells and not between
unlike ones (a bent mode of true weight 0.0098 came out at 0.0083): use chains as long as
the boost's own for weights that matter. Repeat
spread measures numerical bridge stability, not uncertainty from incomplete exploration.
A `.target.json` also records the exact network/region snapshot. Neither method makes new
true-objective calls; `--exact` below is the separate check of the surrogate against truth.

Single-region and unrestricted chains use a single ensemble.

## Exact check

```bash
mpirun -n 8 python client.py derived results/<run> --exact 1000
```

This re-evaluates the true likelihood at 1000 random points of the posterior chain and
reports:

- the emulator's rms, worst-case and mean error for each derived parameter, in units of
  that parameter's posterior σ;
- true minus surrogate log-likelihood at the same points, and the importance-weight
  ESS. Below 10% the command warns that the surrogate is itself far from the truth on its
  posterior.
- the shift in each derived parameter's posterior mean that importance weights
  exp(true − surrogate) would apply. Not from a chain of several cells in which a cell's
  weight is marked as not reliable: the shares of the modes are then uncertain, and the
  shifts and the ESS are left out with the reason (`importance_unavailable` in the summary
  file). The emulator's errors are pointwise and are reported all the same.

A point the likelihood refuses (-inf) is a posterior draw of the surrogate with weight zero. It
stays in the baseline of the shift, the mean over all draws, with its emulated value there,
since it has no exact one (`baseline_emulated`). A point without any value (NaN, +inf) has an
unknown weight, so the shifts and the ESS are left out with the reason, as in
`hpc/analysis/true_check.py`.

This is the test to trust on a real likelihood, where there is no closed form. It costs
N true evaluations: 1000 Planck points at 1.9 s each is about 4 minutes on 8 ranks.

## Existing runs

A run recorded before CLiENT kept derived parameters has no derived values. Continuing it records them for
every new point: the loaded points get NaN, and the emulator masks them. One further
iteration therefore gives n_append points with derived values, concentrated near the
posterior, and with 6 inputs that is enough to train on. Then run `--chain` and
`--exact`.

## Accuracy so far

**Planck NPIPE TT ΛCDM with Cobaya** (H0 sampled; 13D; CLASS++). This is the SINU
experiment's likelihood
at G_eff = 0: `logGeff` is dropped, and CLASS++ skips the self-interaction entirely when
G_eff_ur = 0. The domain, nuisance ranges, run settings, seed and CLASS++ binary are
SINU's. SINU's own likelihood tends to this one as logGeff → −∞: the difference is
7×10⁻⁴ in log L at logGeff = −8 and −12.

The derived parameters were 100θ_s, σ8, Ω_m, S8, Ω_Λ, z_reio and A_s, with the emulator
retrained every iteration (`derived:` section) on the 6 cosmological inputs. Adding them
left the likelihood bit-identical. The run was stopped after iteration 3 (8298 points),
when the method was judged validated; the log-likelihood surrogate itself was not yet
converged, as SINU's was not at the same stage.

Held-out error of the it3 emulator (3 × 128, smaller than the default), in units of
published widths: DESI DR2+CMB for Ω_m and Ω_Λ (±0.0036), and Planck 2018
TT,TE,EE+lowE+lensing+BAO for the rest. The core is the most likely fifth of the
held-out points.

| parameter | σ | rms, core | worst core point | rms, all held-out |
|---|---|---|---|---|
| 100θ_s | 0.00029 | 0.037 | 0.14 | 0.19 |
| σ8 | 0.0060 | 0.013 | 0.062 | 0.055 |
| S8 | 0.011 | 0.015 | 0.048 | 0.044 |
| Ω_m | 0.0036 | 0.019 | 0.13 | 0.14 |
| Ω_Λ | 0.0036 | 0.030 | 0.15 | 0.14 |
| z_reio | 0.71 | 0.008 | 0.043 | 0.026 |
| A_s | 3.0e-11 | 0.004 | 0.015 | 0.041 |

These widths are much narrower than this TT-only run's own posterior, so the comparison
is deliberately strict.

**Planck 2018 ΛCDM with MontePython** (TTTEEE + lowl + lensing + BAO, 27D, CLASS++).
A copy of a Planck 2018 ΛCDM run was continued from it6 through it7, which
recorded derived values for the 2000 new points. The emulator (3 × 128) was trained on
those, with the six cosmological inputs. `--exact 1000` on 8 ranks gave:

| parameter | posterior mean ± σ | rms error / σ | worst / σ | bias / σ |
|---|---|---|---|---|
| H0 | 67.65 ± 0.41 | 0.0028 | 0.012 | −0.0003 |
| σ8 | 0.8090 ± 0.0058 | 0.0038 | 0.013 | 0.0014 |
| Ω_Λ | 0.6888 ± 0.0055 | 0.0038 | 0.015 | −0.0015 |
| z_reio | 7.72 ± 0.71 | 0.0036 | 0.013 | 0.0014 |
| A_s | (2.099 ± 0.029)e-9 | 0.0029 | 0.011 | 0.0009 |
| YHe | 0.24542 ± 0.00005 | 0.019 | 0.088 | 0.0022 |

At the same points, the true minus surrogate log-likelihood is −0.007 ± 0.144, and the
importance-weight ESS is 98%. Reweighting moves no derived mean by more than 0.03σ. The
emulator trained in about 2 minutes, and all derived values were evaluated at 491k chain
points in 0.4 s. YHe is the least accurate relative to its width, because its posterior
width is tiny: it is a nearly deterministic, tabulated function of ω_b.

**Banana test.** On the 16D banana smoke configuration, three analytic derived
parameters (H0-, Ω_m- and S8-like) were checked after two iterations and 440 training
points:

- on the posterior chain, the emulated values are within 0.4–0.8% of each posterior σ
  (rms), 3% at worst, with mean shifts below 0.4%;
- `--exact 500` under `mpirun -n 3` reproduces these numbers through the true backend.

**MontePython example** (`input/montepython/example.param`, SH0ES + BAO, with the
derived entries shown above plus `z_reio` at `scale` 0.1). This was run end to end with
in-loop training, `--chain` and `--exact 200`, to exercise MontePython's conventions:
- H0 = 100h to 6×10⁻⁶;
- z_reio is recorded as 6.2–10.1, where MontePython itself stored 62–101;
- Ω_m + Ω_Λ = 1 − Ω_r;
- the labels are unscaled;
- GetDist flags all six parameters as derived.

With only 210 training points, the emulator errors there are 2–8% of that broad
posterior's σ.

**Planck NPIPE TT ΛCDM with Cobaya + CAMB** (13D).
This is the Cobaya/CLASS++ TT run above, with CAMB 2.0.4 in the `planck_2018_camb`
parameterisation: `theta_MC_100` is sampled and H0 is derived. It used the same
likelihoods, neutrinos, run settings, nuisance ranges and 6 MPI ranks. The domain is
SINU's in CAMB names, with `theta_MC_100` ∈ [0.98, 1.10], which spans H0 50–90 at the
fiducial ω's. All 19 derived parameters were recorded; 6 of the 2100 initial points fail
in CAMB (no H0 in `theta_H0_range`) and get NaN. The emulator's inputs resolved to the six
cosmological parameters. The run was stopped after it1, as sufficient. There the
held-out core rms error, in units of the published widths used above, is:

| | θ* | σ8 | Ω_m | S8 | Ω_Λ | z_re | A_s | H0 |
|---|---|---|---|---|---|---|---|---|
| CAMB, it1 | 1.2% | 0.7% | 4.8% | 1.1% | 3.3% | 0.9% | 0.6% | 1.9% |
| CLASS++ run, it1 | 20% (θ_s) | 5.0% | 6.5% | 1.2% | 7.2% | 4.2% | 1.9% | (sampled) |

θ is far easier here, because the sampled θ_MC is almost CAMB's derived θ*. `--chain`
and `--exact 300` on 6 ranks ran end to end through Cobaya + CAMB. On that chain the
emulator's rms error is 1–9% of the chain's own widths (H0 1.6%, σ8 3.2%, θ* 8.2%). At
it1 the chain is still broad, however: true minus surrogate log-likelihood is −669 ± 544
there, and the importance-weight ESS is 0.3%. It checks the pipeline, not precision.

## Network size

The default of 3 × 256 comes from a sweep over 2, 3 or 4 layers × 64, 128 or 256 neurons,
with three seeds each and the same split per seed. It was run on the it3 data of the
ΛCDM TT run (8298 points) and scored in the same published σ units. Medians over seeds:

| layers × neurons | weights | train time | core rms, median parameter | core rms, worst parameter | worst core point | all held-out, worst |
|---|---|---|---|---|---|---|
| 2 × 64 | 5k | 40 s | 0.016 | 0.068 | 0.33 | 0.30 |
| 2 × 128 | 19k | 46 s | 0.012 | 0.043 | 0.15 | 0.21 |
| 2 × 256 | 70k | 59 s | 0.009 | 0.037 | 0.17 | 0.16 |
| 3 × 64 | 10k | 38 s | 0.017 | 0.041 | 0.17 | 0.25 |
| 3 × 128 | 36k | 52 s | 0.011 | 0.036 | 0.11 | 0.18 |
| **3 × 256** | 137k | 81 s | **0.008** | **0.021** | **0.094** | **0.11** |
| 4 × 64 | 14k | 46 s | 0.014 | 0.043 | 0.14 | 0.26 |
| 4 × 128 | 52k | 65 s | 0.008 | 0.027 | 0.11 | 0.17 |
| 4 × 256 | 203k | 93 s | 0.007 | 0.026 | 0.087 | 0.11 |

Width matters more than depth. The worst parameter is always 100θ_s, the most precisely
measured one. 3 × 256 had the lowest error on it (2.1% of σ, with a seed-to-seed range
of 1.9–2.1%) and the lowest whole-box worst case. A fourth layer bought nothing on
θ_s. Its cost, about an eighth of the log-likelihood network's per evaluation, is small
enough for use inside the sampler.

## Checks on the real backends, point by point

- **Planck 2018 MontePython setup with CLASS++:** H0 = 67.33, σ8 = 0.812 and Ω_Λ = 0.684
  at the fiducial point, and NaN for all six derived parameters at a failed CLASS point.
- **Cobaya + CLASS++ with H0 sampled:** 100θ_s = 1.04217, σ8 = 0.814, S8 = 0.833, with
  the dependencies resolved to the six cosmological parameters.

## Profiles in derived parameters

A profile in a sampled parameter holds that coordinate fixed and maximises over the
others. A derived parameter d(x) is not a coordinate, so its profile is a maximisation
on the level set d(x) = d₀. The same machinery handles it after a change of variables:

1. Pick a sampled **pivot** p on which d depends monotonically, such as θ_s via H0,
   S8 via ω_cdm or logA, or Ω_m via H0.
2. Use the coordinates z, which are x with x_p replaced by d.
3. Map z back to x by solving d_emu(x) = d₀ for x_p with Newton's method on the
   derived-parameter emulator. The solve is written out in TensorFlow, so the map clones
   to float64 and is twice differentiable, as the profile optimisers need.

A profile in d is then an ordinary coordinate profile in z. `client.py profiles` runs on
it **unchanged**: 1D and 2D, hybrid scan, residual Newton, resume and call accounting.

```bash
# 1. A view: a directory laid out like a run, in the z coordinates.
python client.py derived-view results/<run> --params theta_s_100 -o views/theta_s
python client.py derived-view results/<run> --params S8,Omega_m -o views/S8_Om       # two pivots, for a (S8, Omega_m) surface
python client.py derived-view results/<run> --params S8 --keep H0 -o views/S8        # H0 stays a coordinate, for a (S8, H0) surface
python client.py derived-view results/<run> --params S8 --pivots S8=logA -o views/S8_logA

# 2. Ordinary profiles of the view, with every usual option.
python client.py profiles views/theta_s --params theta_s_100 --output profiles/theta_s

# 3. The exact derived values at the final nodes, with the true backend.
python client.py derived-view --nodes profiles/theta_s --workers 4
```

A view needs the run's recorded derived values and derived-parameter emulator (the
`derived:` section, or `client.py derived RUN`). It contains:
- `metadata.json` in z;
- `trained_models/model_it_N.keras`: the pivot solve, then the original network, then a
  feasibility penalty;
- `training_data/data_it_N.csv`: the rows in z;
- the run YAML, whose backend is `derived_coordinates`: it maps z → x with the same
  float64 solve and calls the original likelihood;
- `view.json`: pivots, round-trip statistics and pivot scores; and what the view was made
  from: `source_target`, the identity of the network and metadata it loaded (from a
  verified private copy, as the chains and evidences record it), and `source_files`, the
  hashes of the derived emulator and the training set it read. One of those two replaced
  while the view is made stops it. For a network trained on boosts, also `source_region`
  and `region`: the run's region file and its copy in the view;
- `region/region_it_M.npz`, for a network trained on boosts: a byte copy of the region the
  run saved for this network (see below).

Nothing in the source run is modified.

**Networks trained on boosts.** A network trained on boosts is only good inside the region
of its boosts, and has maxima the truth does not have outside it; the profiles of a run
search it inside that region (`region` in the profile settings, [create_profiles.md](create_profiles.md#networks-trained-on-boosts)).
A view carries the region along, and its profiles do the same. The region stays in the
run's coordinates: a point z of the view is inside when x(z), the point the pivot solve
takes it to, is inside. The searches see the network there and the region's wall outside,
with the feasibility penalty applied after the wall, so targets out of reach still pay it
everywhere. Nodes whose optimum is outside are cut from the ranges as for a run, the manifest
records the region with the pivot map, and `region: E` and `region: null` work as they do
for a run. On an analytic test whose bare network rises 3.25 above the peak at the edge of
the box, the profile of the bare view found that maximum and the view searched in the region
gave the exact profile (`tests/test_derived_profiles.py`, `ViewRegionTests`).

A view made by a version of CLiENT that did not copy the region, from a network with a region,
holds no copy of it, and its profiles searched the bare network. Profiling it with the default
`region: run` stops with a message: make the view again, or set `region: null` to search the
bare network (a study of such a view made with `region: null` resumes unchanged).

**Pivots.** By default, for each derived parameter the candidates are its emulator inputs
that are neither `--keep` parameters nor pivots already chosen. Each is scored by
|∂d/∂x_i| × the width of x_i among the best 5% of training points: how far d moves over
a posterior-sized excursion. d must be strictly monotone in the candidate over the
candidate's whole range, with the other parameters at the best point. Give `--pivots` to
choose them yourself. The pivot should be a parameter the profile does not need as a
coordinate; use `--keep` for a sampled partner in a 2D pair.

**Accuracy.** The constraint is imposed through the emulator, so a node's exact derived
value differs from its grid value by the emulator's error there. Step 3 measures that
difference for every node (`derived_nodes.json`, `derived_nodes.csv`, where
`<name>_offset` = exact − grid). It also re-evaluates the objective, as a consistency
check on the view's backend. Plot the true profile against the exact values if the
offsets are not negligible against the parameter's width.

**Where a target is out of reach.** Near the corners of the box, reaching d₀ may need a
pivot value outside its bounds. The pivot is then clamped, and both network and backend
subtract 10⁶ × (residual / training spread)². Optimisers therefore avoid such points,
and where a node truly lies there the penalty shows up as a large drop. Check a 2D
grid's corners in `surrogate_surface.json` before the true stage, and narrow the ranges
if a corner is penalised. Its true calls are wasted, and it is not a profile value.

**Training rows in a view.** The profile workflow reads training coordinates as float32,
and that rounding of a precisely measured derived parameter moves the recovered pivot.
For 100θ_s that changes log L by about 10⁻³ at the Planck peak, and by much more far
out. That would trip the workflow's label audit, which requires Δχ² ≤ 0.01. A view
therefore keeps only rows that are:
- within 500 log-units of the best point;
- estimated, from the network's gradient, to move by at most 2×10⁻³ in log L.

`view.json` records how many rows were kept. Those rows are all a profile uses training
data for (audit and starting points) when `retrain` is false, the default. Profile
retraining on a view is not supported.

**Box edges.** A view inherits the run's box: coordinates keep their bounds and the pivot
keeps its bounds through the clamp above. Where the constrained optimum lies on a bound,
the profile is the profile within that box, as for a sampled parameter.

### Validation: H0 on Planck 2018 with MontePython

The run is a copy of a ΛCDM run (MontePython 3.6.1, CLASS++, 27D:
Planck 2018 high-ℓ TTTEEE, low-ℓ TT and EE, lensing, BOSS DR12 and 6dF/MGS BAO), at it7.
H0 is derived there, because 100θ_s is sampled. The derived-parameter emulator is 3 × 128
on the 6 cosmological inputs, trained on the 2000 it7 points with derived values. The H0
profile uses 11 nodes on [66.25, 69.05], 4 workers and the default 1D optimisers
(hybrid scan, residual Newton on the true likelihood). The same profile was made from two
views, one per pivot:

| pivot | rows kept | peak | Δχ² = 1 interval | converged nodes | true calls | exact H0 − grid, max / rms |
|---|---|---|---|---|---|---|
| ω_cdm (automatic) | 14612 / 15999 | 67.76 | [67.344, 68.200] = −0.41 / +0.44 | 10 / 11 (1 stalled) | 863 + 29 anchor | 0.0031 / 0.0012 |
| 100θ_s (`--pivots H0=100*theta_s`) | 14608 / 15999 | 67.78 | [67.344, 68.199] = −0.44 / +0.42 | 9 / 11 (2 stalled) | 696 + 115 anchor (stalled) | 0.0030 / 0.0012 |

- **The two pivots agree.** A profile does not depend on its coordinates, and here the
  two differ by at most 0.048 in Δχ² at any node, within the residual-Newton tolerance.
  The optima agree to 2×10⁻⁵ in ω_cdm and 1.5×10⁻⁵ in 100θ_s. The stalled nodes lie on
  the same curve.
- **The constraint holds exactly enough.** CLASS++'s H0 at every node is within 0.003
  of the grid value, below 1% of σ(H0). That error is the derived emulator's, and is
  largest at the low-H0 end, farthest from the 2000 points it was trained on.
  Re-evaluating the objective through the view's backend reproduces it exactly.
- **Against the published value.** Planck 2018 TT,TE,EE+lowE+lensing+BAO gives
  67.66 ± 0.42, and the surrogate chain of this run gives 67.65 ± 0.41. The profile's
  half-widths of 0.41–0.44 match. Its peak is 0.1 (0.25σ) higher than those posterior
  means, a profile-versus-marginal difference of the expected size.
- **This run's box.** The run's box caps 100θ_s at 1.0420: the MontePython `.param`
  centre is 1.0411, and the box is ±3σ around it, while the posterior peaks near
  1.0419. For H0 ≥ 67.93 the optimum sits on that edge in both views. The θ_s view
  reaches it through the clamp, with a pivot residual ≤ 1.3×10⁻⁵ and a penalty
  ≤ 2×10⁻⁴ in log L. The upper half of the profile is therefore the profile within this
  box. A run whose box contains the θ_s posterior would be needed for the unconstrained
  upper side.
- **Cost.** 11 nodes took 700–860 true evaluations, plus 30–115 for the anchor, at
  about 2 s each on 4 workers.

**A 2D surface in two derived parameters, (H0, σ8), on the same run.** The pivots were
chosen automatically: ω_cdm for H0 and ln10¹⁰A_s for σ8. The grid is 7 × 7 on
H0 ∈ [66.6, 68.9] and σ8 ∈ [0.795, 0.826], and 14559 / 15999 rows were kept. The run took
18 minutes and 2740 true evaluations on 4 workers. 42 nodes converged, 6 stalled and 1
exhausted its budget.
- **The surface is consistent with the 1D profile.** Minimising each H0 row over σ8
  (a parabola through the three lowest nodes) reproduces the independent H0 profile
  above to ≤ 0.06 in Δχ² at every H0. The σ8 of each row's minimum, from 0.8128 at 66.6
  down to 0.8060 at 68.9, matches the exact σ8 along the 1D profile's nodes.
- **Against the published value.** Minimising over H0 instead gives σ8 = 0.8099 ± 0.0059.
  Planck 2018 TT,TE,EE+lowE+lensing+BAO gives 0.8102 ± 0.0060. A quadratic fit to the
  43 nodes with Δχ² < 12 gives H0 = 67.77 ± 0.42 and σ8 = 0.8098 ± 0.0060, with
  correlation −0.18. It fits with an rms residual of 0.2.
- **The constraint.** At the 47 reachable nodes, exact − grid is at most 0.006 in H0
  (rms 0.0018) and at most 9×10⁻⁵ in σ8 (rms 3×10⁻⁵): both about 1.5% of σ or less.
- **Two corner nodes are out of reach.** (H0, σ8) = (68.9, 0.826) and (68.52, 0.826)
  would need ln10¹⁰A_s above the box's 3.0897. The pivot is clamped there, with
  residuals 0.35 and 0.01, and their objectives, penalty included, are Δχ² = 2.5×10⁵
  and 268. Their exact offsets
  (−0.021 in H0, −0.0038 in σ8 at the corner) are those of the clamped point, not
  emulator errors. Near that corner the optimum also has τ at its upper bound 0.0783,
  so the upper-right part of the 2σ contour is box-limited, like the H0 profile above
  67.9.

Interpolating the raw 2.5×10⁵ of such a node, the 2D report's isocontour spline rang through
the whole grid, down to Δχ² = −1.2×10⁴. The spline is therefore
fitted to Δχ² clipped at 25, far above both contour levels (`profiles/surface_reporting.py`).

On the ΛCDM TT (NPIPE) run stopped at it3, a 100θ_s view (pivot H0) was built and
passed the label audit. The surrogate profile found no two-sided 1σ interval, however,
and the true stage used up its call budget at every node. The it3 network is not yet
accurate enough for residual Newton to converge, and a profile in a sampled parameter
would hit the same limit. Use a network good enough for ordinary profiles.

## Likelihoods on derived parameters (not implemented)

The pieces above are laid out so that a likelihood on derived parameters can be added
without restructuring. Examples are an S8 or H0 prior, or any external constraint
expressed in σ8, Ω_m or θ_s. There are two cases.

1. **The new likelihood is part of the backend** (added to the Cobaya YAML or the
   MontePython `.param`). It is then part of the emulated target, and nothing here is
   needed.
2. **It is applied on top of an emulator**, for example combining a finished Planck
   emulator with a new constraint at no Boltzmann cost. The sampled target becomes
   ```
   log p(x) = surrogate.logpost(x) + Σ_j log L_j(d(x)),   d = derived emulator
   ```
   Both networks are TF models taking x, so this is a one-line change to
   `tempered_logpost_fn` in `client.py`, plus a config section naming each L_j. The
   log-likelihood network keeps learning the backend's target alone; L_j is cheap and
   exact given d. Two consequences:
   - **Accuracy:** the emulator's error enters as δlog L ≈ (∂log L_j/∂d)·δd, so it must
     be small against L_j's width σ_j, not just against the posterior width. `--exact`
     measures δd directly.
   - **Coverage:** if L_j moves the posterior away from where the training set lies (a
     constraint in tension with the backend), the combined posterior must drive the
     acquisition. The emulator must then be retrained every iteration (the `derived:`
     section already does this), and acquisition must use the combined chain. If L_j
     only narrows the posterior, sampling the combination once, or importance weighting
     the existing chain by L_j(d), is enough.

## Not done / limitations

- Profiles in derived parameters impose the constraint through the emulator. Its error
  is measured at every node by `derived-view --nodes`, but not corrected. Check that
  the offsets are small against the parameter's width.
- A view is fixed to one iteration's network and derived emulator. Profile
  retraining (`retrain: true`, `--add-points`) on a view is not supported.
- A pivot must be monotone in its derived parameter over the whole box, with the
  other parameters at the best point. The pivot choice refuses non-monotone
  candidates, but it does not prove monotonicity elsewhere in the box.
- A view inherits the run's box. Where the constrained optimum lies on a bound, as
  100θ_s does on the Planck example above, the profile is the profile within that
  box.
- The chain is written as one GetDist file with unit weights. Multi-region output is a
  mixture resample, not chronological walkers; inspect the per-cell `.sampling.json`
  diagnostics. Neither output format implies a convergence statistic.
- Cobaya's automatic input selection uses one private attribute
  (`parameterization._derived_dependencies`). If a future Cobaya changes it, selection
  falls back to all inputs, which is correct but less accurate.
- Cobaya's automatic χ² totals (`chi2__CMB` and the like, added for every likelihood
  that has a `type`, as the Planck ones do) are not recorded. They are the
  log-likelihood itself, not smooth regression targets, and they depend on nuisance
  parameters.
