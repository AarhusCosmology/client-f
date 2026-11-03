# Create profiles from a saved CLiENT iteration

`client.py profiles` performs a complete profile study without changing the
original run. By default it uses the saved network without adding training points
or retraining, and outputs three curves for each selected parameter:

1. **Surrogate profile:** maximize the selected network over all other parameters.
2. **True at surrogate:** evaluate the real objective at those nuisance optima.
3. **True optimized:** use the surrogate-corrected residual Newton method to
   refine the native objective, measuring corrections with real backend evaluations.

The same rule applies to every direction. There is no Gaussian/flat classification.
The surrogate stage of a 1D or 2D study is the hybrid scan (`surrogate_optimizer: hybrid`, the
default) or the L-BFGS-B scan with continuation passes (`legacy`), and the true stage is
residual Newton. A study resumes with its saved settings; a manifest that lacks a current key
is refused, so an older study cannot be continued. Optional augmentation geometry uses
L-BFGS-B. See
[the optimiser guide](hybrid_optimisers.md).

## Quick start

From the repository root, in the normal CLiENT environment:

```sh
python client.py profiles results/MY_RUN \
  --iteration 5 \
  --params omega_b,omega_cdm,theta_s_100,logA,n_s,tau_reio \
  --points 15 --sigma-range 3.5 \
  --workers 4 --output results/MY_RUN_profiles
```

`python -m profiles` accepts exactly the same arguments. Omit `--iteration` to
select the latest numeric iteration having **both** `model_it_N.keras` and
`data_it_N.csv`. Augmentation checkpoints with suffixes are not mistaken for
ordinary iterations. Parameter names must be sampled names from `metadata.json`,
not derived quantities or display labels. To profile a derived parameter (H0 when
θ_s is sampled, σ8, ...), first make a derived-coordinate view of the run, in which it
is a coordinate, and profile the view; see
[derived_parameters.md](derived_parameters.md#profiles-in-derived-parameters).

First check paths, parameter names, saved bounds and planned sizes without
loading a network or making **any backend evaluations**:

```sh
python client.py profiles results/MY_RUN \
  --params omega_b,omega_cdm --output results/MY_RUN_profiles --dry-run
```

The original likelihood packages and data must be installed. Nothing in this command downloads Planck data or substitutes a
Boltzmann solver.

On macOS, keep the machine awake for the duration with:

```sh
caffeinate -i python client.py profiles results/MY_RUN \
  --config input/profiles_example.yaml --output results/MY_RUN_profiles
```

**Do not launch the profile command under `mpirun`.** It manages its own persistent
backend processes, `workers` of them, at most the host's CPU count. Acquisition and
validation batches use these workers concurrently. The true stage
(see [hybrid_optimisers.md](hybrid_optimisers.md)) runs chains along each profile, outward from
its seed nodes, and submits each node's finite-difference probes as one concurrent batch. It
keeps an exact global cache and accounting, and gives results identical to one
worker. Chains serialise
some calls, so four workers do not imply fourfold speedup of every stage; see
[Many workers](#many-workers) before using more than about 8.
On multi-process hosts the numerical thread defaults are one per worker; explicit
thread environment settings are respected. Tests cover Cobaya with two workers;
other backend installations must support ordinary spawned Python processes.

### Many workers

With `corrected`/`newton`, a node starts from the state its predecessor in the chain
left: gradient, curvature correction and error estimate. That saves calls, but a chain's
nodes run one after another, and each node is about ten dependent rounds of calls. With
one seed per profile, at the node nearest the best fit, and two chains outward from it
(`chain_length: null`), a one-profile study keeps only
a handful of workers busy, whatever `workers` is. `chain_length: k` adds a seed, solved
cold as the first one is, after every k chained nodes. All seeds start at once, and a
chain starts as soon as its own seed is done. **The default is `chain_length: 2`.**
Results depend on `chain_length`, never on `workers` or timing.

Backend-time efficiency (work ÷ (workers × wall time)) with CLASS++-like call times,
simulated from each study's recorded journal by `benchmarking/schedule_newton.py`:

| study | 6 workers: `null` / `2` (default) | 32 workers | 48 workers |
|---|---|---|---|
| 1D, 1 profile × 11 points | 78% / 93% | 33% / 50% | 22% / 34% |
| 1D, 1 profile × 51 points (re-run with each setting) | 74% / 97% | 23% / 77% | 15% / 65% |
| 1D, 2 profiles × ~25 points | 91% / 97% | 39% / 87% | 26% / 77% |
| 1D, 6 profiles × 17 points | 99% / 100% | 96% / 99% | 93% / 98% |
| 2D, 7 × 7 | 96% / 96% | 71% / 75% | 58% / 62% |
| 2D, 15 × 15 | 97% / 100% | 77% / 94% | 63% / 88% |

The second row was re-run with each setting. The others keep each node's recorded cost,
which the testbed below puts at 1-3% too low for `chain_length: 2`.

On the residual-Newton testbed (a 51-node profile with value noise 2.5e-3),
`chain_length` 2-4 cost 1-3% more true calls than `null` at unchanged accuracy; 0,
every node independent, cost 11-17% more. On a real CLASS++ profile (Planck 2018 H0,
11 nodes), `2` gave the same profile and Δχ² = 1 interval as `null`, with nodes within
0.06 in Δχ², 2.8% fewer calls, and 5-7% less time at 4 workers. A small study stays
limited by its slowest single node and the final best fit, whatever the setting. With
six or more profiles on 4-6 workers, `null` saves the 1-3% extra calls, since every
worker is busy either way.

`python benchmarking/schedule_newton.py STUDY --workers 4 16 32 48` replays a finished
study's call journal on any number of workers (a discrete-event simulation of its
dependency structure, within about 10% of measured wall times). Use it to choose
`workers` and `chain_length` from a study already run; pass `--chain-length` if the
study used one. On `--resume`, `workers` may differ from the original run; any other
setting, `chain_length` included, may not.

## Configuration and profile ranges

Copy [the complete example](../input/profiles_example.yaml) and edit it:

```yaml
iteration: 5
parameters: [omega_b, omega_cdm, theta_s_100, logA, n_s, tau_reio]
retrain: false
added_points: 0
broad_fraction: 0.75
profile_points: 15
sigma_range: 3.5
workers: 4
ranges:
  omega_b: [0.0220, 0.0227]
  tau_reio: prior
```

An explicit `[low, high]` is in the parameter's **physical input units** and must
lie within the saved model bounds. `prior` means the full saved model domain,
which may be narrower than the backend's mathematical prior. For unspecified
parameters, the code finds the two surrogate 1-sigma crossings around the fitted
peak, scales the left and right widths by `sigma_range`, and clips to the saved
bounds. If it cannot identify a two-sided interval, it falls back to the full
saved domain and emits a warning. Flat and one-sided profiles are therefore
supported without pretending that a sigma width exists.

The final grid contains **exactly `profile_points` freshly optimized nodes** per
parameter. It is not a nearest-neighbor subsample of an existing scan. Explicit
ranges control the displayed/true-evaluated grid; acquisition retains full-domain
coverage for the selected parameters. `ranges.json` records requested and actual
ranges, clipping and fallback reasons. Inspect it before interpreting tails.
With a network trained on boosts a range can also be cut to the region of the boosts
(`region` in `ranges.json`; [Networks trained on boosts](#networks-trained-on-boosts)).
Without training, the preliminary full-domain scan only chooses ranges, so it is skipped
when every parameter has an explicit range (`ranges.json` then records no surrogate
1-sigma crossings); a resumed study that already has `baseline_profiles.json` keeps it.

Configuration keys are checked strictly: spelling mistakes are errors. Explicit
CLI options override YAML values. A dry run prints the fully resolved settings.

| Setting | Default | Meaning |
|---|---:|---|
| `iteration` | latest complete model/data pair | Original checkpoint iteration |
| `parameters` | required | Unique sampled names to acquire/profile |
| `retrain` | false | Opt in to profile-focused training and its validation calls |
| `added_points` | 0 | Optional distinct training rows; positive explicit requests enable retraining unless explicitly disabled |
| `broad_fraction` | 0.75 | Membership fraction in the broad proposal pool, between 0 and 1 |
| `profile_points` | 15 | Number of final true and surrogate grid nodes per parameter |
| `sigma_range` | 3.5 | Left/right surrogate sigma span unless explicitly overridden |
| `ranges` | `{}` | Per-parameter `[low,high]` or `prior` |
| `workers` | 1 | Backend processes, at most the host's CPU count; may change on `--resume` |
| `region` | run | Where a network trained on boosts is searched: `run`, inside the region the run saved for it; a number, the same region with its edge at that many times the reach of the boost points; null, the bare network over the whole box (what a study made before the setting used). See [Networks trained on boosts](#networks-trained-on-boosts) |
| `chain_length` | 2 | At most this many nodes solved in sequence from one cold start; null keeps one seed per profile (what older studies resume with). See [Many workers](#many-workers) |
| `continuation` | null | 2D only: `{seed: v, start: best or scan}`, each row continued along the second parameter from its node nearest v, every later node started from its predecessor's true optimum instead of the network's. Needs `chain_length: null`. See [Continued rows](create_profiles_2d.md#continued-rows-continuation) |
| `first_epochs`, `polish_epochs` | 400, 1600 | Two cosine phases, each at least 2 epochs |
| `initial_lr`, `transition_lr`, `final_lr` | 1e-5, 1e-6, 1e-7 | Cosine endpoints |
| `batch_size`, `seed` | 128, 42 | Profile training/design settings |
| `validation_global` | 512 | Independent perturbed global validation points |
| `validation_nodes` | 9 | Per-parameter ridge centers; each costs 3 validation points |
| `scan_points` | 41 | Preliminary full-domain surrogate grid density (skipped without training when every parameter has an explicit range) |
| `surrogate_optimizer` | hybrid | Batched basin hopping and projected BFGS; `legacy` selects the L-BFGS-B scan with polish and continuation passes |
| `hybrid_walkers` | 8 | Basin-hopping walkers per surrogate node |
| `hybrid_hops` | null | Hops of the hybrid scan at most; null keeps its 12. The scan has converged only after two hops in a row in which no node improved, and a study stops before its true stage when it has not |
| `corrected_tol` | null | Native log-objective gain tolerance; null selects 1e-2 |
| `max_passes`, `gtol` | 15, 1e-4 | Legacy scan pass cap; projected-gradient tolerance for both surrogate methods |
| `maxfun` | 150 | Per true local solve evaluation budget, including interpolation |
| `total_call_budget` | 100000 | Lifetime backend-attempt ceiling, including earlier resumed attempts |
| `audit_points`, `audit_tolerance` | 3, 0.01 | Saved-label checks and absolute Delta-chi-squared tolerance, plus CSV rounding allowance |
| `backend_module` | null | Optional importable backend module, e.g. `classy`, to fingerprint and verify in every worker |
| `backend_sha256` | null | Optional required module-file SHA256 |
| `run_config` | unique run YAML | Explicit run YAML path/name when discovery is ambiguous |

The standard target is the true native-objective profile. Augmentation is optional,
not a prerequisite. Set `retrain: true` and `added_points: 2000` to use the
75/25 broad/targeted design and training settings described below.
`retrain: true, added_points: 0` retrains on existing data and still acquires
independent validation points. An explicit positive `--add-points N` or YAML
`added_points` enables retraining if `retrain` is omitted; an explicit false value
with positive additions is rejected. Use `--retrain` for retraining without additions.
The default skips augmentation, validation acquisition and training entirely.
Old studies resume their recorded settings, including their previous retraining policy.

## Networks trained on boosts

A run with `acquisition.boost.region` ([boost_background_region.md](boost_background_region.md))
saves, with every boost, the part of the box that the posterior points of its boosts reach
(`RUN/boost/region_it_N.npz`). From then on the run's own chains, audit and evidence see the
network only there, and nothing keeps the network honest anywhere else: it has maxima outside
that the truth does not have. A profile is a search for the network's maximum on a slice of
the box, so a profile of the bare network finds them. On a real 49-parameter likelihood
the optima of a bare boost network lay 38 to 46 posterior sigma from the data, 85 to 165 above the true peak in the
network and 600 to 700 below it in truth; all 15 nodes of a pilot ran to their call limit.
Inside the region the same pilot converged at every node, and a study of all 49 parameters
at 15 points converged at 677 of 735 nodes with 51 true calls a node in the median.

The setting `region` says where the network is searched:

| `region` | |
|---|---|
| `run` (default) | inside the region the run saved for this network: the file `region_it_M.npz` with the largest M not above the iteration, with the run's own edge (`acquisition.boost.region`, 2 in the measured runs) |
| a number E | the same region with its edge at E times the reach of the boost points. 1 keeps the search to where the boost points are, which also rules out what the run's audit tolerates between 1 and 2 (a maximum up to 10 above the best) |
| `null` | the bare network over the whole box, which is what a study made before the setting used |

A network of the ordinary loop has no region; it is searched over the whole box whatever
the setting says, and nothing below applies to it. A derived-coordinate view of a network
trained on boosts carries the network's region, and everything below applies to it in the
same way ([derived_parameters.md](derived_parameters.md#profiles-in-derived-parameters)). What the searches are given is a Keras
model at double precision like any other (`model.network.restrict_model`): the network
inside, and outside the region's wall, far below anything inside and falling away from the
region. Every optimiser of the workflow takes it, 2D surfaces included. Profile training
(`retrain: true`) trains the network itself; the searches before and after see it in the
region. The log's first line after the label audit names the region, the number of boost
points and their reach.

**A node outside the region.** A node can be computed from the network only if the
network's best point on its slice lies inside the region. It does not when the slice misses
the region, or when the network still rises where the region ends, which is the same
statement about the network: it cannot be trusted there. After every surrogate scan the
workflow looks at where each node's optimum is. Nodes whose optimum lies beyond 98% of the
edge are cut from the ends of the parameter's range, the range becomes the stretch of
neighbouring nodes inside that holds the best fit, with as many nodes as before, and the
scan is repeated; this happens before any true evaluation of a node. The cut is printed and
recorded: `ranges.json` keeps what was asked for in `requested`, has the range computed in
`actual`, and under `region` the range before, the range kept and every node value found
outside; the scan's own file has the same under `region_cuts`. The full-box scan that chooses
ranges is cut in the same way, so in a wide box it becomes a scan of the region's extent.
The study ends only when fewer than two neighbouring nodes of a parameter are inside, or
when after four scans a node is still pressed against the edge.

How far a profile reaches before this matters: along one parameter the region extends to
the edge times the reach of the boost points, and the reach is the Mahalanobis distance of
the farthest of thousands of posterior points, which grows with the dimension (12.4 sigma
in a real run with 49 parameters; 10.6 and 11.6 for the two modes of a 16-parameter twin).
The nodes of a ±3 sigma profile lay at 0.3 of the reach in the 49-parameter run.

If the search for the best fit itself ends on the region's edge, the network rises above
its value at the data inside the region. The study stops at the surrogate check and says
so; a smaller `region` keeps the search to where the boost points are.

**Two modes.** On the analytic twin of a two-mode likelihood (16 parameters; a region of two
ellipsoids of 10 000 boost points each) three parameters were profiled at 15 points. With
the run's edge of 2 the true profiles are those found with the bare network, to every digit
printed, at the same number of true calls. One of them is the parameter that separates the
modes: its profile runs through the valley between them, with nodes at up to 1.3 times the
reach, and is exact to 0.017 in the log-objective. With the edge at 1 that profile's last
node, at the edge of the box, has its optimum outside: it is cut, the range ends one node
earlier, and all 15 nodes of the shorter range converge. Where two modes are almost equally
high a search can stay in one at a node where the other is 0.08 to 0.15 higher; that is so
with and without the region.

**Resuming.** The manifest records the region a study was made in (file, hash and edge, or
none), and that is what a resumed study must match, not the wording of the setting. A study
made before the setting on a run without boosts resumes with the default. One made on a
run with a region used the bare network: it resumes with `region: null`, or without
`--config`, and is refused with the default.

**Surfaces.** In 2D the network is searched inside the region too, but a rectangle is not
cut. When a node of the surface has its optimum at the region's edge or beyond it, the study
stops after the scan, before any true call of a node, and names the nodes
(`surrogate_surface.json` has every node's optimum); give ranges inside the region. Letting
such a study run on was measured and rejected: the scan certifies those nodes on the region's
wall, and on a three-parameter test the ten of them took about 270 of 324 true calls and all
ended `budget`.

What is not established: the cut has been exercised on analytic networks only; in the real
49-parameter study every node lay well inside. A node just inside the edge (rho above
about 0.9 of it) is let through and can stall in the true stage, where the Newton steps'
probes cross the wall; it is flagged, not converged. A 2D study on a real boost network has no
exact reference to be checked against.

## Objective convention and backend identity

Normal CLiENT targets use the backend's **native objective**. In Cobaya this is
`logposterior`, including prior/auxiliary constraints; the workflow preserves
that convention in training and all three curves. It does **not** silently switch
to the sum of likelihood components. If a source network was trained under a
different convention, adapt it explicitly rather than bypassing the label audit.

Saved metadata bounds govern acquisition and optimization. Calibration Gaussian
prior densities remain unchanged. The workflow does not reapply `n_sigma` using
potentially different reference widths. It checks backend parameter order and
validates a few saved labels before spending on augmentation. This audit can
catch a wrong backend/convention, but is not a proof of numerical accuracy.
The labels checked are those of the best training rows, where the profiles are. The tolerance is absolute, and far from the peak it
tests something else: at a design point with a chi-squared of 1e5 a relative
difference of 1e-5 between two machines' builds of the same backend is a
Delta chi-squared of 1, while the rows near the peak agree to 1e-4.

For CLASS++ use the same environment/PYTHONPATH that resolves the desired module,
and set `backend_module: classy`. The module path and hash enter the manifest;
each worker checks the hash. Optionally pin a known `backend_sha256`. On resume,
a different module identity is rejected. The run YAML alone is not evidence that
CLASS++ rather than another CLASS build has been loaded.

## Method and training selection

For parameter `t` and nuisance coordinates `eta`, the surrogate profile is

\[
\widetilde\ell_p(t)=\max_\eta\widetilde\ell(t,\eta).
\]

The direct curve evaluates `ell(t, eta_surrogate(t))`. The corrected curve uses
that point as the initial condition for a local optimisation of the true native
objective: the surrogate-corrected stage (see [hybrid_optimisers.md](hybrid_optimisers.md)),
which models the truth as the surrogate plus a measured correction. Surrogate derivatives shape that model
and the curvature geometry, but are **never used as true-likelihood gradients**: the
true gradient is always measured by finite differences of the true objective.
All parameters not selected for a particular profile point remain free inside
the saved box.

Broad conditional draws retain full-domain support; targeted draws concentrate
near profile ridges, peaks and interval crossings. The targeted pool itself
contains broad draws: 75/25 is **not** a literal narrow/wide draw fraction. The
quadratic acquisition geometry and mixtures are in `dataset/profile_augmentation.py`.

New training coordinates are made float32-representable **before** true labels
are evaluated. Uniqueness and train/validation disjointness are checked after
this conversion. A collision raises an error before spending calls on that
batch; it is not silently replaced by a mislabeled or duplicate row. Labels
retain full precision in the augmented dataset. Validation costs
`validation_global + 3 * validation_nodes * len(parameters)` calls by design
(674 for six parameters with defaults) and is never included in the training set.

Retraining uses all old and added rows, a fresh Adam optimizer, and the original
network/scalers as a warm start. Scalers are not adapted; their stored weights
are checked unchanged. This is a new profile-training phase, **not** an exact
continuation of the original optimizer trajectory. Two selected checkpoints
come from the same fit: pooled contrast (primary) and pointwise (diagnostic).
The epoch-zero network remains eligible. Selection normalizers are frozen at
epoch zero; matching optimizer slots are saved with each selected network.
All final profile curves use `model_contrast.keras`.

The default Newton method measures residual slopes and learns a curvature correction
in scaled surrogate Hessian geometry. Every free coordinate on a bound receives an
inward true probe before it can be held there; physical bounds are never widened.
Newton refines the common true anchor after the profile nodes. The direct and optimized true curves share that anchor; the surrogate uses its own
anchor. `diagnostics.json` reports unsettled anchors, clipping and nonzero solver
flags, and an anchor whose own solver did not converge (its verdict was in
`true_profiles.json` alone, under `anchor_diagnostics`): every true curve is referred to
that point, so it can still rise. A successful exit indicates a geometric stopping condition, **not** a
global-optimum or absolute Delta-chi-squared accuracy certificate.

## Outputs

| File/directory | Contents |
|---|---|
| `README.md`, `profiles.png`, `profiles.pdf` | Portable three-curve comparison and warnings |
| `profiles.csv` | One row per node: coordinate, all three raw objectives and Delta-chi-squared values, true solver flag |
| `surrogate_profiles.json` | Selected-network grid, nuisance optima and stationarity/stability diagnostics |
| `direct_profiles.json` | True objective at each surrogate optimum |
| `true_profiles.json` | Corrected true grid, nuisance optima, anchor and solver diagnostics |
| `intervals_grid.json` | **Grid-interpolated** 1/2-sigma crossings, not refined true interval endpoints |
| `diagnostics.json`, `ranges.json` | Numerical warnings, actual ranges and fallback/clipping details |
| `augmented_data.csv`, `augmented_data.npy` | Full training set, original rows plus exactly N additions |
| `validation.npz`, `*_design.json` | Independent validation, contrast pairs and proposal provenance |
| `training/` | Primary/secondary/final networks, selected epochs, histories and scaler checks |
| `manifest.json`, `audit.json` | Resolved configuration, source hashes, backend identity, label audit |
| `refused_points.json` | Only with `retrain: true`, when any: design and validation points the likelihood refused, by stage |
| `calls/attempt_*.jsonl`, `counts.json` | Durable events and aggregate cost by stage across resumes |
| `true/node_*.json`, `status.txt`, `failure.json` | Per-node checkpoints and progress/failure diagnostics |
| `complete.json` | All requested stages finished; **not** a numerical-accuracy certificate |

The default command deliberately does not spend additional backend calls refining
interval crossings between the requested nodes. Use a denser grid when that extra cost is needed.
Do not equate interpolated endpoints with refined ones.

## Resume, costs and failure handling

```sh
python client.py profiles results/MY_RUN \
  --output results/MY_RUN_profiles --resume
```

Without a new `--config`, this uses the stored profile configuration. A changed
source model, data, metadata, likelihood YAML, backend fingerprint or configuration
is rejected. Of `region` the resolved region is compared, not the setting
([Networks trained on boosts](#networks-trained-on-boosts)). The run's files are
compared by what they hold (their hashes) and by their place inside the run, so a study is
resumed as well after its run directory was moved or synced elsewhere with the same files
(the command says where the run stood when the study began, and the
manifest is left as it was written). A refusal is shown as a usage error with the place it
was raised in brackets; `CLIENT_PROFILE_TRACEBACK=1` in the environment gives the whole
traceback. Use a new output directory for a different experiment. A process
lock prevents two commands from writing or evaluating the same study concurrently.
Locks are released by the OS after exit; do not remove files to override an active
process. The workflow targets macOS/Linux local filesystems with `flock` support.

Completed stages and true nodes are reused. Successful old objective values are
imported into the exact cache with provenance. An interrupted training phase is
retrained from the original checkpoint using already saved data; it does not
repeat acquisition. A partially completed expensive batch or node is retried
with successful values cached. Failed or interrupted backend attempts remain
in the lifetime count. Journals are never overwritten. A surviving `failure.json`
may describe an earlier failure; `complete.json`/`status.txt` give final state.

A node whose start the likelihood refuses (a prior edge, a Boltzmann failure) has no value:
`logl`, `xopt` and `direct_logl` are null, its diagnostics say `refused start` with flag 1, and its
chain goes on from the state it was given. The study stops only if every node is refused. Its
checkpoint keeps this on a resume, so delete that node's file (`true/node_*.json`, in 2D
`true_surface/node_*.json`) after a change that cures the backend there.

With `retrain: true`, a design or validation point the likelihood refuses is left out of the
training and validation data, and listed in `refused_points.json`: it is not redrawn, so the network
can have fewer new points than `added_points`, and the study's diagnostics say so. A batch with
more than half its points refused stops the study, and so does a validation set that has lost
all its pairs of one kind, since the score that selects the checkpoint needs them. The label audit
stops on a refused row whatever the setting: its rows are the best the run has, so the backend has
changed.

A node of the Newton stage that was interrupted is charged on resume for every call it had
started, the ones in flight when the job ended too: their starts are in the journal and
their values are not, so they are evaluated again. A node stopped near its budget (`maxfun`)
may therefore no longer afford the gradient round it was in and end with the flag for its
budget, where an uninterrupted run would have converged; its value is still an evaluated
one. Give such a study a larger `maxfun` in a new output directory if many nodes end so.

The parent records an attempt **before dispatch**. Counts therefore describe
objective attempts (including submission/startup failures and interrupted work),
not independently measured internal CLASS/theory computations. Finite/nonfinite
returns, exceptions, cache hits, imported records and coalesced batch requests
are separate events. Cache hits/imports are not new backend calls. `counts.json`
breaks costs down into audits, augmentation, validation, anchors, direct values
and local optimization. Original training rows are reported as rows; historical
attempt totals are not inferred from them. The configured budget applies across
all resumes; expensive partial failures do not reset it.

Repeated values at identical inputs may be cached exactly. This assumes a
deterministic backend. For stochastic/noisy objectives, supply an explicit
noise-aware adapter/optimizer instead of treating this workflow as a certified
solution. Deterministic numerical roughness also matters: the real Planck study
found small coordinate changes producing objective jumps despite exact repeated
values. Tightening a trust-region radius alone cannot remove that problem.

## Python API and validation

```python
from profiles.config import ProfileConfig
from profiles.workflow import run_workflow

settings = ProfileConfig(parameters=['omega_b', 'omega_cdm'], added_points=1000, retrain=True)
run_workflow('results/MY_RUN', 'results/MY_RUN_profiles', settings)
```

In Python `added_points` above zero needs `retrain=True` beside it; `ProfileConfig.load`,
which reads a YAML, sets it for you when the file does not say. Without it the workflow
stops at once ("retrain=false requires added_points=0").

Run the analytic end-to-end and accounting tests without expensive likelihoods:

```sh
python -m unittest discover -s tests -p 'test_profiles_workflow.py' -v
```

The integration test creates a saved Gaussian-network iteration, uses two Cobaya
workers, augments/retrains it, verifies all three outputs against analytic profiles,
checks frozen source files and validation separation, and verifies that a resumed
completed computation spends no extra backend calls. It does not establish
accuracy or scaling for a particular production likelihood.

## Two-dimensional profiles

Use `--pair name1,name2` for exactly one joint surface per run. `--points` becomes
the number of nodes **per axis**. Pair-conditioned augmentation, pooled validation
and both-coordinate nuisance optimization are described in
[Create one 2D profile](create_profiles_2d.md), with a complete configuration example.
