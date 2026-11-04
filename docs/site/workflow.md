# Training and boosting

## The ordinary loop

CLiENT evaluates an initial design, trains a network on the native log-probability, samples the network, and selects a new batch for true evaluation. The sampling temperature and acquisition target temperature have separate settings. Use their documented relationship intentionally; a tempered acquisition distribution is not the final posterior.

Warm starts and function-preserving scaler compensation are on by default. Learning-rate annealing is not: `training.lr_schedule` defaults to `none`, although the real runs have used `plateau`. The dynamic loss-weight ESS floor defaults to `0.1`. [What this version adds](_generated/library/CHANGES.md) gives the measurements behind these choices.

## Settings to start from

Most methods below are opt-in, so a run that takes only the defaults is not the method of the real runs. Two complete configurations are the settings to start from: `input/recommended_near_gaussian.yaml` and `input/recommended_non_gaussian.yaml`, the second for likelihoods known to have oscillations, several modes or curved ridges. The [boost guide](_generated/library/docs/boost.md#two-recommended-settings) explains the difference between them and which to take.

## Four different ways to add information

| Mechanism | When it acts | Purpose | Default |
|---|---|---|---|
| Ordinary acquisition | Between trained iterations | Extend coverage of the tempered target | Normal loop |
| Peak injection | Initial design or ordinary acquisition | Draw additional points from a fitted quadratic | Off: both injection counts are zero |
| Optimum audit | After training, before acquisition sampling | Check and correct overestimated maxima | Off: audit section absent |
| Posterior boost | Configured later iterations | Improve the posterior core and separated-mode weights | Off: boost section absent |

An injection fits its quadratic only to the points near the best (within 16 × T n / 2 log-units). The window does nothing while both injection counts are zero. See [the injection study](_generated/library/docs/injection_window.md).

A profile study's true evaluations can also join the training set, as a separate command: see [feeding profile evaluations back into training](inference.md#feeding-profile-evaluations-back-into-training).

## Audit before trusting a network

An explicit `acquisition.audit: {}` activates the audit with its defaults. A search finds network maxima, true evaluations check them, and fine-tuning uses those checks. Audits address optimism; they cannot discover every region the network underrates.

In boost iterations, `chain: true` also looks where the cooled walkers of the tempered chain climb to, and `posterior` evaluates cooled walkers at T = 1 and fine-tunes when the network is too high there. Both are opt-in.

```{warning}
An uncleared audit is a scientific warning, not a guarantee that the loop will halt. Inspect the saved audit report before continuing. The optional failure penalty is a training-only synthetic constraint; it is not a finite likelihood measurement.
```

Read [audit methods, budgets and failure policies](_generated/library/docs/audit_optima.md).

## Boost a sufficiently good network

A minimal explicit configuration is:

```yaml
acquisition:
  # Keep the ordinary acquisition keys in your full configuration.
  boost:
    n: 1000
    start_iteration: 15
```

From that iteration onward, boost acquisition replaces the ordinary tempered batch and its injection. The default unfiltered path samples at T=1 and splits the requested evaluations equally among detected modes. This provides information about a mode even when the network initially gives it too little probability.

Detection and sampling operate on the network. A missed or severely underestimated mode may remain missed. The true-value correction must also have adequate effective sample size.

| Optional setting inside `boost` | Effect and qualification |
|---|---|
| `density_filter: true` | Selects density-deficit points; batches can underfill and mode counts can differ. The unweighted evidence correction is invalid for these selected rows. |
| `mixing: 0.1` | Uses measured chain mixing to group candidate basins; the threshold is an explicit choice. |
| `mode_depth` | Leaves out modes whose highest maximum lies this far below the highest mode's; `auto` is `temperature * ndim / 2`. |
| `balance_floor` | Gives a mode lighter than the floor less than an equal share. |
| `background: quadratic` | Fits a fixed quadratic at the boost transition and cold-starts the residual network. |
| `region: 2.0` | Restricts exploration using genuinely evaluated boost clouds; this describes known modes, not exhaustive coverage. |
| `region_quantile` | Needs `region`. Sets the region's reach by a quantile of the distances, not by the farthest point. |
| `train_domain` | Needs `region`. Trains only on points near the posterior from the first boost on. An oscillating likelihood needed it, and it cost near-Gaussian runs accuracy. |

These are all off by default. A quadratic background alone was unsafe in the documented non-Gaussian example; read the paired method before enabling it.

- [Boost algorithm and mode weights](_generated/library/docs/boost.md)
- [Density filtering](_generated/library/docs/posterior_density_boost.md)
- [Mixing-based mode definition](_generated/library/docs/boost_modes_mixing.md)
- [Quadratic background, region and training domain](_generated/library/docs/boost_background_region.md)
- [Whole-loop real-likelihood validation](_generated/library/docs/boost_real_lcdm.md)

## Several trainings per network in the boosts

A training ends either resolving the fine structure of an oscillating or multimodal likelihood, or
on a plateau where that structure is smoothed out, and which one depends on the random start. A
warm start keeps whatever state it inherits, and can grow a false maximum that the next boost's
points do not reveal. `training.members` trains each network of the boost phase several times and
keeps one:

```yaml
training:
  members: 12             # rank 0 warm-started as before, 11 from random weights on otherwise idle ranks
  member_pilot: 100       # choose by true evaluations at each member's own posterior points
  member_device: auto     # members on a GPU where their process sees one, else the CPU
acquisition:
  boost: ...              # required: the members act from the first training on boost points
```

The tempered phase trains every network once. Each member samples its own posterior; the pool of
`member_pilot` points per member is evaluated once with the true objective, and the warm network is
kept unless another member's error is lower by more than `member_margin` (20%). The pilot's points
join the training set. Members train on ranks that would otherwise wait, so the training lasts about
as long as its slowest member while each has cores (or a share of a GPU) of its own; the log warns
when they do not. On two oscillating likelihoods the pilot rejected a warm network with a false
maximum at five networks running, where the boost points showed nothing wrong.

- [Several trainings per network](_generated/library/docs/members.md): the method, its measurement,
  its cost and GPUs.

## Preserve the experiment

Keep configurations, source revision, true evaluation records and selected checkpoints together. Distinguish requested points from finite returns, imports and cache hits. A boost's true points enter the next iteration's dataset; its report records the acquisition that produced them. Preserve nonfinite/failure counts rather than silently replacing them in reported totals.

## Errors on each acquired batch

Each ordinary batch is evaluated against the network that selected it, before that batch is used for training and before injection is added. The reported native log-objective error is `network − truth`: finite count, bias, population scatter, RMS and largest absolute error. Refused-value sentinels and nonfinite pairs are excluded. These diagnostics add no true calls; they reuse the acquisition's returns. The batch is selected by the algorithm, so it is not independent validation of the entire posterior.

`metrics.json` holds `held_out_n`, `held_out_rms`, `held_out_bias`, `held_out_sd` and `held_out_largest` for ordinary acquisitions. Boost reports contain corresponding pooled/per-mode errors and their scatter history. Inspect both mode-specific errors and true-posterior checks before interpreting pooled accuracy.

## True-evaluation convergence and early boost handover

```yaml
convergence:
  metric: tempered_batch_scatter
  threshold: 1.2
  consecutive: 2
  max_iterations: 30
acquisition:
  # Also include the ordinary acquisition settings for your problem.
  boost:
    n: 3000
    start_iteration: 30
    boosts: 5
```

This is a fragment, not a complete run configuration. At trained network N, the errors measured at the batches chosen by N−2 and N−1 must both be strictly below `1.2 × sampling.temperature`. N is the first network trained on both batches. A missing/nonfinite error cannot pass. Increasing `consecutive` requires that many preceding batches. The [methods page](methods.md#batch-error-and-the-handover-test) defines the statistic.

Without `-i`, a passing verdict stops an ordinary-only run, or advances a still-future configured boost start to N. `boost/start.json` records the effective start and is read on continuation. The optional `boosts` key limits the scheduled phase, subject to the run's iteration ceiling. **With explicit `-i`, the requested schedule wins**: the verdict is reported, but does not stop the run or bring boosts forward. The other metric, `gaussian_posterior_drift`, has its own required threshold.

This test acts only before boosts. It is a practical criterion for ending ordinary acquisition, not certification of the true posterior or automatic termination of boosts. The documented threshold was supported by an analytic suite and one controlled 49D real-likelihood comparison; the 58D tests did not establish its adequacy there. Read the [measurements and limits](_generated/library/docs/tempered_convergence.md).

## A boost iteration that cannot find a mode

If mode search raises the specific `NoModeFound` condition, that iteration acquires an ordinary tempered batch (and usual injection if enabled). It writes `boost/fallback_it_N.json`, not boost-point files. The decision is shared across MPI ranks; arbitrary exceptions are not treated as a harmless no-mode result. The schedule remains unchanged, so the fallback consumes a phase slot. There is no automatic top-up or guarantee that later mode searches recover.

## Audit fine-tuning and memory

Audit fine-tunes are compiled without XLA, which avoids the persistent memory mappings observed after repeated fine-tuning on Linux. Main-network training and sampling use XLA.

## Stopping and continuing

An interrupted run is continued with the same command: a phase without a record of completion is done again, and its partial files are moved to `RUN/interrupted/`. A continued run also checks that the objective is the one the run recorded. Read [continuing and recovering a run](_generated/library/docs/run_recovery.md).
