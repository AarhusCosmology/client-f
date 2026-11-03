# One two-dimensional profile per run

Use the existing command with **one pair**:

```sh
python client.py profiles results/MY_RUN --iteration 10 \
  --pair omega_b,omega_cdm --points 21 \
  --sigma-range 3.5 --workers 4 --output results/MY_PAIR
```

`--points 21` means **21 × 21 = 441** true-profile nodes. It does not mean 21
points distributed across a plane. Exactly two distinct sampled parameter names
are required. Derived parameters, such as (H0, σ8), become sampled names in a
derived-coordinate view
([derived_parameters.md](derived_parameters.md#profiles-in-derived-parameters)). Each invocation and output directory contains one pair. Run other
pairs sequentially in separate directories. The original run remains untouched.

The same `--dry-run`, `--resume`, native-objective convention, backend identity
guards, worker ceiling (the host's CPUs), source fingerprints and call ledger documented in
[Create profiles](create_profiles.md) apply. Studies use batched hybrid surrogate scans and residual Newton true refinement.
Independent row chains and their probe batches share `workers` backend processes; each node
holds both selected coordinates fixed. For more than about 16 workers, see
[Many workers](create_profiles.md#many-workers) (`chain_length`, default 2). See
[hybrid optimiser details](hybrid_optimisers.md#joint-2d-profiles) for checkpoint
semantics. Use `caffeinate -i`
on macOS. No external `mpirun` is needed.

## Run from a normal saved iteration

The command accepts the same saved CLiENT run as the 1D workflow. Supply
`--iteration` to choose a checkpoint, or omit it to select the latest matching
model/data iteration. Parameter names must match the run's metadata. A previously
profile-trained model is not required: the default `retrain: false, added_points: 0`
uses the saved network directly and optimizes the true native objective. Optional
`retrain: true` enables the augmentation, validation and training stages.
Original models and data remain untouched.

Copy and edit [the example YAML](../input/profiles_2d_example.yaml), then preview
the job without likelihood evaluations or creating the output directory:

```sh
python client.py profiles results/MY_RUN --iteration 5 \
  --config input/profiles_2d_example.yaml --output results/MY_PAIR --dry-run
```

Start the job by removing `--dry-run`. On macOS, keep the machine awake with:

```sh
caffeinate -i python client.py profiles results/MY_RUN --iteration 5 \
  --config input/profiles_2d_example.yaml --output results/MY_PAIR
```

After an interruption, resume the same job using its saved configuration:

```sh
python client.py profiles results/MY_RUN --output results/MY_PAIR --resume
```

Do not change the pair, ranges, training settings or source checkpoint when
resuming; start a separate output directory for a different experiment. CLI
options override YAML values. The manifest preserves the resolved settings.

## Configuration

```yaml
dimension: 2
parameters: [x_30, x_31]   # exactly one pair
iteration: 10
added_points: 0
broad_fraction: 0.75
retrain: false
profile_points: 21        # per axis
design_points: 7          # 7 × 7 coarse surrogate design, no true calls
validation_global: 512
validation_nodes: 9      # 9² independent pair locations, 3 points each
ranges:
  x_30: prior
  x_31: prior
workers: 4
total_call_budget: 100000
```

Explicit ranges `[low, high]` and `prior` have the same meaning as in 1D. Without
an explicit range, preliminary 1D surrogate profiles set marginal spans, which
form a rectangular 2D domain. This is a display/design choice, not a joint
confidence region. Non-two-sided 1D intervals fall back to saved bounds, with
warnings. Full bounds are appropriate for testing degenerate ridge pairs.

For any saved network, the default is `retrain: false` and
`added_points: 0`. This skips acquisition, validation and training and spends only
label audits, the true anchor and profile calls. It isolates the profiling
algorithm from changes to the surrogate. The source run must contain the intended
checkpoint/data pair; never silently substitute a different saved model.

The grid is capped at 2,500 nodes and the coarse design at 625 nodes to prevent
an accidental quadratic workload. Those are safeguards, not performance promises.
Two-dimensional augmentation needs at least one remaining nuisance parameter;
a genuinely two-parameter objective can be directly evaluated with retraining off.

## What is optimized

For selected coordinates \(q=(q_1,q_2)\) and all remaining coordinates \(\eta\),

\[
\widetilde\ell_p(q)=\max_{\eta\in B_\eta}\widetilde\ell(q,\eta),\qquad
\widetilde\eta(q)=\arg\max_{\eta\in B_\eta}\widetilde\ell(q,\eta).
\]

The three outputs are

\[
\widetilde\ell_p(q),\qquad
\ell(q,\widetilde\eta(q)),\qquad
\ell(q,\widehat\eta(q)),
\]

where \(\widehat\eta\) comes from bounded value-only refinement of the true
native objective, initialized at \(\widetilde\eta\). This is **not** a product of
two 1D profiles. Both coordinates remain fixed simultaneously throughout each
nuisance optimization. For 31 parameters, the search has 29 free coordinates.

The default surrogate scan uses float64 batched basin hopping and projected BFGS,
with exchanges between four adjacent grid nodes and the unrestricted anchor.
Projected gradients and second-order checks use only free nuisance coordinates.
`surrogate_optimizer: legacy` selects alternating L-BFGS-B sweeps.
With either scan, a result that fails its stationarity/stability check stops the
run before any true call and is kept as `failed_surrogate_diagnostics.json`. A
global hybrid search can find spurious network maxima far outside the training
data (network values far above every training label, often at box corners); the
legacy scan, seeded from the best training rows, stays in the trained region.

The default true stage measures the residual gradient with finite differences of
native objective values and learns a curvature correction. Each grid row starts
near the surrogate anchor and continues in two independent outward chains
(with `chain_length`, from further seeds along the row as well).
The default `corrected_tol` is 0.01 native log-objective units; `maxfun` defaults
to 150 calls per node solve. No analytic gradient of the backend is needed.
Boundary coordinates are probed inward before being held. Residual transfer
never crosses rows, so worker scheduling cannot change a node's prior state.

### Continued rows (`continuation`)

Opt-in. Every node above starts from the surrogate scan's optimum at that
node; only the residual model travels along a chain. Where the network is far wrong, its optima
are no start. In the low-omega corner of an axion-monodromy run every network was 120 to 280
log-units too low (a long independent MCMC and our likelihood agree there to 0.02), and the
(A_osc, omega) study's nodes at omega = 1, A_osc >= 0.021, started from them, ran out of calls at
delta chi2 34 to 356 where that chain has about 12. With

```yaml
chain_length: null          # one seed a row
continuation:
  seed: 0.0                 # a row's seed: its node nearest this value of the second parameter
  start: best               # from the best training point; scan: from the network's optimum there
```

each row has one seed, at its node nearest `seed` on the second axis, started from the best
training point with the node's two coordinates set (`best`, the default) or from the scan's
optimum there (`scan`); every other node starts from the true optimum of the node before it in
its chain, read from that node's checkpoint. The network still supplies the curvature of the
Newton steps, and the scan still runs and must converge, but its optima are used at the seeds
only. Put the seed where the network is right: for an oscillation's amplitude at zero, where its
frequency and phase do not matter.

A chain follows the branch it starts on. Where the objective has several conditional maxima at a
node (several phases of an oscillation, say), a continued node can end on a lower one than the
scan's optimum would have given. In the laptop run below the stand-in truth had the network's
phase structure, several maxima at low omega, and two of nine rows followed a lower branch from
their second node on, ending 1.5 to 6 log-units below the scan's optima; the other seven matched
them to 0.3. At small amplitude a real oscillation has one phase maximum (its log L is
A g(phi) - A^2 f(phi) / 2 with g a first harmonic), which is why the seed goes at zero amplitude.

The setting, with the best point, enters the true stage's signature, so a resumed study cannot
change it; results depend on it and never on `workers`. It needs `chain_length: null`. Rows run in parallel, a row's nodes one after another. Tested on analytic
surfaces (`tests/test_surface_continuation.py`) and in a laptop run of the corner study through
the workflow with a stand-in truth (network 18 plus a bump in the corner: 81 nodes, 4007 calls, 74
converged and 7 stalled); not yet measured on a real likelihood.

Every true node is saved separately with its successor state; restart imports
completed journal values and resumes unfinished nodes. The node checkpoints are tied to
the surrogate grid's optima, so a resumed study with the hybrid scan reads the grid it finished
(`surrogate_surface.json`) and does not scan again (done again, the scan would have to come out
identical bit for bit, or the study would be refused). Direct and optimised
surfaces share one unrestricted true anchor; the surrogate has its own anchor.
Stop flags and stable values do not certify a global optimum or exploration of
all disconnected branches.

## Pair-conditioned augmentation and training

A coarse grid supplies nuisance anchors over the requested rectangle. Broad-pool
proposals choose its nodes uniformly and jitter both fixed coordinates within
half-cell neighborhoods. Targeted-pool proposals use weights

\[
w(q)=1+3\exp[-((\Delta\widetilde\chi^2(q)-2.30)/1.5)^2]
       +3\exp[-((\Delta\widetilde\chi^2(q)-6.18)/2)^2].
\]

The positive floor preserves support away from contour regions. After jitter,
both pair coordinates are fixed again and the nuisance anchor is reoptimized.
A regularized quadratic fit to the training data supplies the conditional
nuisance precision \(P_{\eta\eta}\), with the pair coordinates removed together.
In standardized nuisance coordinates, broad displacements follow
\(\delta\eta=Lz\), where \(LL^T=P_{\eta\eta}^{-1}\) and
\(z\sim N(0,T I)\), with \(T\in\{0.25,0.5,1\}\).
Near-ridge displacements use \(\|z\|^2=2d\),
\(d\in\{0.05,0.2,0.5,1\}\). Rejection enforces the original box; a recorded
truncated-Gaussian Gibbs fallback handles inefficient rejection. Proposed
coordinates are quantized before labels are evaluated, and collisions rejected.

The 75/25 setting is **pool membership**, not a literal narrow/wide draw fraction:
the targeted pool also includes broad nuisance displacements. Exactly N new rows
are evaluated and added; analytic benchmark information never guides acquisition.
These are practical extensions of the 1D defaults, not an assertion that 2,000
points are optimal or sufficient for every pair.

Independent validation uses random pair locations from the coverage design, with
one optimized center and two nuisance displacements at each location, plus global
held-out points. Its cost is

\[
N_{\rm val}=N_{\rm global}+3n_{\rm val}^2,
\]

or 755 with defaults. Profile contrasts compare pair-center objective differences;
nuisance contrasts compare each displacement with its center. Validation rows are
excluded from training after coordinate quantization. Coordinates are checkpointed
before evaluation for exact reuse after an interrupted batch.

The existing frozen-scaler, two-phase cosine schedule and pooled contrast selector
are reused, including epoch-zero eligibility and a secondary pointwise checkpoint.
No Gaussian/flat-specific loss or selector is introduced.

The configurable defaults are `first_epochs: 400`, `polish_epochs: 1600`,
`initial_lr: 1.0e-5`, `transition_lr: 1.0e-6`, `final_lr: 1.0e-7` and
`batch_size: 128`. Together these request 2,000 additional epochs from the selected
source iteration, with the first cosine phase ending at the transition rate and
the second ending at the final rate. The selected model need not be the last
epoch. These controls appear explicitly in the example YAML.

## Cost and interpretation

For an n×n grid, a planning ceiling for grid solves is roughly
\(n^2\,\mathrm{maxfun}\), plus direct evaluations, audits and unrestricted-anchor
solves. Cache hits can reduce actual calls. The lifetime `total_call_budget` is
the hard accounting limit; inspect `counts.json` by stage rather than interpreting
this ceiling as an expected cost. Count old training and new profiling separately.
A solver budget flag remains visible even when a finite incumbent is saved.

The figures use `chi2.ppf(2*norm.cdf([1,2])-1, df=2)`: levels **2.295749** and
**6.180074** enclose 68.27% and 95.45% for a 2D Gaussian. These are the conventional two-parameter
likelihood-ratio reference levels. They are not 1D levels 1 and 4. They do **not**
assert frequentist coverage for boundary or non-identifiable ridge models.
The heatmap retains linear contours on the raw grid. Separate `isocontours.png`
and `.pdf` use a common tensor cubic spline (quadratic for a 3-node axis), with
no smoothing and a recorded check for dips below the common anchor. The spline is
fitted to Δχ² clipped at 25, so that one very deep node cannot ring through the
whole grid; both levels lie far below the clip. These are
interpolated lines, not additional evaluated points. Cubic interpolation can
reduce grid error on smooth surfaces, but can overshoot on rough surfaces; compare
both renderings and refine the grid when necessary. No extra true calls
refine contours. A curved thin ridge may need a finer grid even
when every computed node is accurate. Use nodal errors and resolution studies
before interpreting a visually smooth interpolated contour.

## Files and tests

- `surrogate_surface.json`: axes, full nuisance optima, values, gradients and pass diagnostics.
- `true_surface.json`: direct and optimized raw true values, common anchor and all flags.
- `surface.csv`, `surface.png`, `surface.pdf`: all three surfaces on identical axes.
- `isocontours.png`, `isocontours.pdf`: separate joint 1σ/2σ spline contour comparisons.
- `surface_diagnostics.json`: clipping, anchor and solver warnings.
- `failed_surrogate_diagnostics.json`: an unconverged surrogate scan, written instead of `surrogate_surface.json` when the run stops.
- `coarse_surface.json`, `acquisition_design.json`, `validation_coordinates.npz`: training provenance when retraining.
- `true_surface/node_*.json`: reusable completed node computations.
- Existing manifest, training histories, cost journals and completion markers remain available.

```sh
python -m unittest discover -s tests -p 'test_profiles_2d.py' -v
```

`tests/test_surface_continuation.py` covers `continuation`: each node's start, both seed starts,
serial = parallel, resume and the signature, and the settings.

Tests include a correlated conditional optimum with two simultaneous constraints,
a saved-network-to-augmented-surface integration, exact requested data size,
validation separation, fixed-coordinate checks, and zero-call resume.
