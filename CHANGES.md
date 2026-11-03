# What this version adds

CLiENT as first released builds a network on the true likelihood in a tempered loop and samples it. This version keeps
that loop and adds what real, high-dimensional and non-Gaussian likelihoods turned out to need, and tools for inference
beyond the posterior. Most of it is opt-in: `input/recommended_near_gaussian.yaml` and `input/recommended_non_gaussian.yaml`
are the settings the real runs have used. [UPGRADING.md](UPGRADING.md) says what changes for an existing run or
setting; [PERFORMANCE.md](PERFORMANCE.md) has the speed and memory work with its measurements.

## The tempered loop

- **The loss.** `training.msre_ess_floor`, on by default: `msre`'s constant is raised until the weights' effective sample
  size holds, so that the loss does not collapse onto a few points when the design misses the peak. On a 29-parameter
  target the credible metric went from 260 times its noise floor to 8 times.
- **Warm starts.** `training.warm_start`, on by default, with the raw predictions kept when the scalers change: 2.95 and 2.54 times fewer
  epochs at 58 and 31 parameters, at equal or better accuracy.
- **The learning rate.** `training.lr_schedule: plateau`, opt-in, with a stopper that starts counting when the rate has
  reached its floor. Better on a dense design and worse on a sparse 58-parameter one; all real runs have used it.
- **Injection.** `prior.n_inject` and `acquisition.n_inject`: points from a quadratic fit. Both
  fit only the points near the best; on real likelihoods in a wide box the fit to all points put them thousands of
  log-units below the peak ([guide](docs/injection_window.md)).
- **The audit.** `acquisition.audit`: true evaluations at the network's own maxima before the acquisition chain runs. It
  corrects optimism only ([guide](docs/audit_optima.md)).
- **Stopping.** `convergence.metric: tempered_batch_scatter`: the scatter of network minus truth at the tempered batch is
  below 1.2 T for two networks running. A hand-over rule, not a certificate of accuracy ([guide](docs/tempered_convergence.md)).
- **The batch's own test.** Every acquisition reports the error of the network that chose it, at no extra true call.
- **Speed and memory.** Batched selection, a preallocated chain buffer, `sampling.thin`, MPI workers that
  sleep while rank 0 trains, and the `chunked` schedule (`CLIENT_MPI_SCHEDULE`).
- **Reproducibility.** A top-level `seed`, for all but the AIES proposals, which run under XLA.

## The boosts

- **`acquisition.boost`**: from an iteration on, the new points are posterior samples of the network, the same number from each mode.
  Weights of separated modes come from per-mode evidences, and are also given corrected by the true values
  ([guide](docs/boost.md)).
- **What counts as a mode.** `mixing` decides by measured mixing in place of a fixed barrier; `mode_depth` leaves out modes far
  below the highest; `balance_floor` gives very light modes less than an equal share ([guide](docs/boost_modes_mixing.md)).
- **Where the network is believed.** `region` confines chains and searches to where verified boost points reach; `region_quantile`
  makes that reach robust against a far point ([guide](docs/boost_background_region.md)).
- **Training.** `background` puts a fixed quadratic under the network; `train_domain` trains only near the posterior;
  a warm-started training of the boost phase keeps the weights at the end of its schedule. The warm start stays the default in the
  boosts; `training.warm_start: false` (random weights) is for a likelihood on which warm-started networks stop improving
  ([guide](docs/boost.md)).
- **Which points.** `density_filter` selects posterior candidates by the density-deficit rule ([guide](docs/posterior_density_boost.md)).
- **The audit in boost iterations.** `audit.chain` looks where the tempered chain's cooled walkers climb to, and `audit.posterior`
  evaluates cooled walkers and fine-tunes when the network is too high there.

## The analysis of a finished network

- **True profiles.** `client.py profiles`, in one and two dimensions: a hybrid scan of the network and residual Newton on the true
  objective, with budgets, journals and resume ([1D](docs/create_profiles.md), [2D](docs/create_profiles_2d.md), [methods](docs/hybrid_optimisers.md)).
- **Back into the run.** `client.py augment` adds a profile's true evaluations to the training set, for a region the acquisition
  does not reach ([guide](docs/augment_training.md)).
- **Domain splitting.** `client.py combine` joins runs on adjacent boxes into one emulator ([guide](docs/combine.md)).
- **Export.** `client.py export` writes a network as a TensorFlow SavedModel that loads without CLiENT ([guide](docs/export.md)).
- **Derived parameters.** Recorded with all true evaluations; emulated, added to chains and profiled through `client.py derived`
  and `derived-view` ([guide](docs/derived_parameters.md)).
- **The evidence.** `client.py evidence`, with the BEST nested sampler, in the run's own prior convention ([guide](docs/evidence.md)).
- **Modes.** `client.py modes` for any saved network.

## Running

- **Safe continuation.** Phase records with hashes, a run lock, the objective identity, `--rewind`, call journals
  ([guide](docs/run_recovery.md)).
- **Real likelihoods on a cluster.** [`hpc/`](hpc/README.md).
- **Strict settings.** Unknown keys and out-of-range values are refused.

## Deliberately not here

Measured and left out: input whitening (18 to 45 times lower validation loss, and a worse credible metric: it reparameterises the
target into radial form, exact only where one covariance describes the posterior everywhere), model transfer (its benefit was non-monotonic in
iteration), and the Muon and K-FAC-preconditioned optimisers (10.6 times slower per epoch; the preconditioner recovered 11 to 55% of a
harmful effect). The acquisition's own whitening and the whitened quadratic fit are core machinery and share only the word.

Removed before this release because no real run asked for them: the audit's `ascent` and `jit_compile`, `sampling.adaptive`, the `dynamic` MPI
schedule, the settings that only brought back earlier behaviour (`inject_window`, `batch_size`, `warm_start_preserve_function`, `warm_start_weights`,
`--adopt-legacy`), and in true profiles `coarse_stride`, the Gaussian-process corrected stage, Py-BOBYQA and the keys that chose them.
