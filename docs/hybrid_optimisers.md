# Hybrid profile optimisers

New 1D and 2D studies made with `client.py profiles` or `python -m profiles` use the
following default optimisers:

| stage | method | alternative |
|---|---|---|
| surrogate profiles | `surrogate_optimizer: hybrid` — batched basin hopping + certified projected BFGS ([`profiles/hybrid.py`](../profiles/hybrid.py)) | `legacy` — L-BFGS-B + shifted/Newton polish + continuation passes |
| true profiles | residual Newton — surrogate plus a measured residual gradient and learned curvature correction ([`profiles/residual_newton.py`](../profiles/residual_newton.py)) | none |

The surrogate stage was validated on the
31D banana + curved-ridge analytic target (below). The `newton` true stage was developed
and validated on the real CLASS++ likelihood of the massless-SINU study (Planck TT, logGeff
profile at the it10 and it20 networks), on a network-as-truth testbed and on the 31D target.
Joint 2D surfaces use hybrid/Newton refinement (see the 2D section below).
Augmentation/validation nuisance designs use L-BFGS-B. Serial Py-BOBYQA, the comparison in the
tables below, was removed from CLiENT.

## Reported objective

Profile figures label their ordinate `2(ell_max - ell)`, where `ell` is the backend's
native log objective. When priors enter that objective, this is a log-posterior difference,
not likelihood-only chi-squared. CSV keys named `dchi2` hold this same quantity. The 1D
levels 1 and 4 and the 2D Gaussian reference levels
do not certify frequentist coverage or Bayesian credible regions for arbitrary targets.

## Budgets and interrupted studies

Interrupted Newton nodes retain their lifetime call budgets and cached finite incumbents;
nonfinite interior probes produce an explicit unverified result. Torn final journal records
are preserved and conservatively reserved in the budget: an incomplete journal tail with no
recoverable stage reserves one attempt in the study's lifetime budget, and is not charged
again against every later Newton node. Attempts whose stage is known count against that node
on resume, including failures and in-flight calls. None of this alters the optimiser's
decisions on an intact journal, the numerical policy of a fresh run or any physical bound. See
[Continuing and recovering a run](run_recovery.md) for accounting and limitations.

## Saved-network default

New 1D and 2D studies set `retrain: false` and `added_points: 0`. The final target
is the true native-objective profile; profile-focused augmentation is opt-in.
Use `retrain: true` with an explicit added-point budget to use that workflow.
Old manifests preserve their original retraining policy.

## Selecting the optimisers

```yaml
surrogate_optimizer: hybrid     # or legacy
hybrid_walkers: 8               # basin-hopping walkers per grid node
hybrid_hops: null               # hops at most; null = 12
corrected_tol: null             # native log-objective units; null = 1e-2
```

`surrogate_optimizer: legacy` selects L-BFGS-B with polish and continuation passes. On a rugged
network it can do better: on a real 2D surface of an axion-monodromy run, with the same anchor, it
was better than the hybrid scan with 60 hops by more than 1 at 173 of 2500 nodes and worse by more
than 1 at 35.

A study's manifest records these keys, so `--resume` refuses a changed optimiser. With `--resume` and no `--config`, the CLI loads the saved settings automatically. A manifest
that lacks a current key is refused: an older study cannot be continued.

## Surrogate stage: `hybrid`

Goal: the most precise profile *of the surrogate*.

* **Batched.** All grid nodes and the global anchor are optimised together; each step is
  one float64 TensorFlow evaluation of every walker (idea taken from BEST's `Optimiser`).
* **Basin-hopping walkers.** Each node keeps `hybrid_walkers` persistent walkers. A hop
  either perturbs a walker with a Gaussian step shaped by the anchor's Laplace covariance
  (BEST's covariance preconditioning), or copies a neighbouring node's incumbent (CLiENT's
  continuation). Metropolis acceptance at T = 1 in log-likelihood lets walkers leave a
  basin; the node incumbent is elitist; step sizes adapt towards 50% acceptance, capped at
  5 sigma. Hopping stops when no node improves by more than `value_tol` = 1e-4 for 2 hops,
  or after `hybrid_hops` hops (12 when unset). A scan the limit stopped has not converged
  and the study stops before any true call ("surrogate stationarity/stability check
  failed"; `failed_surrogate_diagnostics.json` has `profiles_stable: false`). On a rugged
  network the limit can matter: 2500-node surfaces of a network of an axion-monodromy run
  were still improving at a handful of nodes at hop 12, with every node stationary
  and the anchor above them all; `hybrid_hops: 20` lets such a scan settle. A larger
  limit changes nothing for a scan that settles before 12.
* **Local ascent.** Batched projected BFGS in prior-box coordinates. Its Hessian model
  starts from the exact autodiff Hessian (whitening) and is carried across hops; bounds are
  exact via an epsilon-active set; there is no relative-`ftol` stop. (The legacy
  L-BFGS-B's `ftol=1e-10` is relative, so with log L ~ -5500 it stops at a projected
  gradient ~1e-2 and every node needs the polish.) The steps of all rows are one batched
  solve, which numpy refuses as a whole when a single row's model is singular (measured on
  a real study: a network inside a region of five cells stopped at its scan that way). Such a
  batch is therefore solved row by row, and a
  singular row takes its free gradient (capped as every step is) and restarts its model
  from the identity.
* **Certification.** Every incumbent must pass CLiENT's projected-gradient test
  (`gtol`) and a second-order test on the exact Hessian of the free, inactive coordinates;
  saddles are escaped along the direction of positive curvature. An incumbent that fails
  only the gradient test also counts as stationary when its Newton decrement, the
  predicted remaining gain g^T(-H)^{-1}g/2 over the free coordinates at a local maximum,
  is at most `gain_tol` = 1e-6 log L (`HybridConfig`). Unlike a box-coordinate gradient,
  this does not depend on the parameterisation. It matters on very ill-conditioned
  targets: on the 58D CLOE Gaussian (9.5 decades of eigenvalue spread) converged nodes kept
  box gradients of 1e-4 to 1e-3 along their stiffest directions with remaining gains
  below 4e-11. The test only classifies; it never moves a point or changes a value, and
  nodes that pass the gradient test are not re-evaluated. Diagnostics record
  `remaining_gain` per node and `stationary_by_remaining_gain` in the scan quality.

`optimization_quality` holds
`method`, `hops`, `all_local_max_certified`, `saddle_escapes`, evaluation counts and time.

## True stage: `corrected`

Goal: the fewest true-likelihood calls for a given accuracy. `corrected_method` has one
value, `newton`.

### Method `newton`

[`profiles/residual_newton.py`](../profiles/residual_newton.py). Around an incumbent `u_b`
the truth is modelled as `t(u) ~ s(u) + r(u_b) + g.d + 0.5 d^T Q d` (`d = u - u_b`): the
network `s` with all its nonlinearity, plus a residual whose gradient `g` is **measured**
and whose curvature correction `Q` is **learned**. Coordinates are whitened by the
network's local curvature (sigma capped at half a box, as `profiles.hybrid.curvature_geometry`).

1. **Start** at the surrogate optimum (its true value is the direct call).
2. **Transfer.** The residual gradient at an optimum of `t` is exactly `-grad s` there, so
   each finished node hands its successor that gradient; the chain extrapolates it
   linearly along the profile and, if the model then predicts more than `transfer_min`,
   spends one call at the model's maximum. A transfer that fails pauses the next two.
3. **Gradient round** (one concurrent batch of about n calls): forward differences of
   `t - s`, 0.25 sigma along the network's curvature eigenvectors (at most 0.1 box
   half-widths); inward along the axis for a coordinate on a bound; **central**
   differences with a 0.5 sigma baseline along the 2 flattest directions, which also
   measure the curvature there. Whitened least squares gives `g`.
   Every free coordinate on a bound is probed inward, including coordinates whose
   surrogate or inherited slope points outward. Only the measured slope determines
   the active set for the subsequent step; the full free-coordinate whitening frame
   is retained for the gradient round. A nonfinite inward probe returns
   `unverified_bound`, not convergence. Physical bounds are unchanged. Central
   remeasurement may release a previously held coordinate, and an improving probe
   overrides a stationary local model.
4. **Trust-region steps** on the model, starting at 0.5 sigma and doubling after good steps
   to the edge (then stepping again with `g` carried along by `Q`, without a new round).
   Each step's actual/predicted mismatch corrects `Q` along the step; two gradient rounds
   0.5 sigma or more apart give a PSB secant update (least change of `Q`). `Q` is carried
   along the chain, bounded to at most 90% flattening of the network's curvature. A step
   that fails within 0.5 sigma mirrors every probe (central differences everywhere).
5. **Stop** when a freshly measured gradient predicts at most `corrected_tol` (default
   1e-2 log L), or after a good step when the curvature error bound
   `0.5 (L * distance since the round)^2` is below it; `L` is floored at 0.2 and learned
   from secant and step mismatches beyond the noise.

Each profile starts at its node nearest the surrogate's best fit and runs two chains
outward from it (a node learns only from its chain predecessor, so results do not depend
on `workers`); the global best fit is solved last, from the best node, with every
coordinate free. `chain_length: k` (default 2; null for the plan above alone) adds a seed,
solved cold as the first, after every k chained nodes, so that more nodes can run at once
on many workers. All
seeds start together, and each chain starts as soon as its own seed is done. See
[Many workers](create_profiles.md#many-workers) for costs and measured efficiencies. On `--resume` finished nodes are loaded (each checkpoint holds the state
its successor needs) and an interrupted node re-runs with its journaled calls served from
the cache. Calls that were started and had not returned when the job ended are charged to
the node again (their starts are journaled, their values are not), so a node stopped inside
a round near its call budget may not afford that round on resume and ends flagged. A
gradient round that ends exactly at the budget still gets its verdict, which costs no call. Node checkpoints are `true/node_*.json`; the stages in the call journal are `corrected/anchor` and `corrected/{direct,node}/NAME/J`.

**Why it is built this way** (all established on the real SINU likelihood, see the
evidence below):

* A Gaussian-process residual with random probes was tried and rejected: it spent 2-4x Py-BOBYQA's
  calls there, and its tolerance sat below CLASS++'s value scatter (1e-3 - 2.5e-3 log L at probe
  scale). A full measured gradient per round and a tolerance a few times the gradient-noise
  floor fix both.
* The network's flattest direction at these nodes is the tau_reio - A_s degeneracy, and
  the node optima sit at tau ~ 0.085, near the prior edge 0.1. One network sigma along it
  puts tau on the edge, where the truth falls by 40 - 150 log L that the network does not
  reproduce (measured). Within +-0.5 sigma the residual is smooth
  and nearly linear. Hence the 0.5 sigma initial trust radius and the 0.5 sigma central
  baseline along the flattest directions (a 0.1 sigma baseline measured curvature with
  noise ~0.6, which corrupted `Q`).
* The networks' curvature error relative to the truth is 0.3 - 0.5 at these nodes; a
  floor of 0.1 on `L` let a barrier node stop 0.1 log L short after a 1-sigma move.

**Evidence** (true calls exclude the start value, which every method shares):

| test | Py-BOBYQA | `newton` |
|---|---|---|
| real SINU TT, it20 network, first 29 nodes (development version) | 1,016 calls, median 32 | 621 calls, median 17; dchi2 vs BOBYQA median +0.005 (better), range -0.21 .. +1.09 |
| real SINU TT, it10 network, first 24 nodes (development version) | 1,116 calls, median 41 | 619 calls, median 18; dchi2 vs BOBYQA median +0.017, range -0.05 .. +0.16 |
| testbed it19 -> it20 network, noise 2.5e-3, 51 nodes | 1,788 calls; gap median / max 0.046 / 0.65 | 1,208 calls; 0.005 / 0.025 |
| testbed it10 -> it20 network, noise 2.5e-3, 51 nodes | 3,181 calls; 0.19 / 4.95 | 3,369 calls; 0.016 / 1.46 |
| 31D analytic target, 151 points | 10,870 calls; 0.0035 / 0.024 | 4,923; 0.0007 / 0.019 |

Gaps in the synthetic tests refer to the exact (31D) or local (testbed) node maximum.
The real SINU objective includes priors, so its comparisons are differences in
-2 logposterior, not likelihood-only chi-square.

**Independent validation (2026-09-29).** A final comparison
used 2,670 Newton versus 2,804 BOBYQA node calls on 43 common it10 nodes
(logGeff -5 through -0.8), and 1,040 versus 1,519 on 39 it20 nodes (-5 through -1.2).
Independent cross-start polishing improved the selected Newton values by at
most 0.0869 (it10) and 0.0307 (it20) in -2 logposterior. A further it20 tail at
-1.1, -1.0, -0.9 and -0.8 used 427 Newton versus 535 BOBYQA calls from identical
starts; Newton found the better evaluated value at all four nodes. Some it10 Newton
solves still stalled or exhausted their budget. Tail Newton stalled at -1.0 and -0.9;
BOBYQA exhausted its budget at -1.1 and -1.0. These are local checks, not global
certificates.

The table's testbed rows were measured without the inward probe of every free coordinate on a
bound (step 3 above). That probe prevents an inherited or surrogate slope from falsely holding a
coordinate at a bound without a true inward probe. It is a correctness rule, not an
established speed improvement: with it, noisy proxy calls rise from 1,208 to 1,586 (it19 to it20)
and from 3,369 to 4,129 (it10 to it20), and worst local gaps change from 0.025 to 0.0574
and from 1.455 to 0.870. All physical bounds are kept, including tau_reio <= 0.1.

Validation used 2,772 unique actual objective evaluations, all finite, plus 142 cache
hits and 17,208 import records (including repeated recovery imports, not new calls).
A harness context-manager error was recovered using saved checkpoints; its 403 prior
finite calls are included in the total. All 47 checked solve outputs were audited
against durable journals and pinned manifests. These tests stop at -0.8; the near-zero branch was outside scope.

## Evidence: 31D banana + curved ridge

Target `banana_loglkl_planck31` (27D Planck-shaped Gaussian + banana + parabolic ridge),
network `banana_planck31_wide` at iteration 15, 10 profiled directions
(6 LCDM + x_28..x_31) x 15 nodes. Every node has an exact answer (bounded QP + analytic
ridge block). The harness of these comparisons is not part of the release.

**Surrogate stage**

| | legacy | hybrid |
|---|---|---|
| exact analytic target: worst node vs exact (log L) | 1e-11 | 4e-10 |
| network: stationary / local-max certified | 150/150 / – | 150/150 / 151/151 |
| network: remaining Newton gain, max (log L) | 7.5e-10 | 5.4e-12 |
| network: nodes where the other method is higher by > 1e-6 | 6 | 0 |
| network: wall time (runs overlapped, not a clean timing) | 441 s | 104 s |

The 6 nodes are 0.02–0.11 log L (up to 0.22 Delta chi^2) higher with hybrid; at 5 of them
the *true* likelihood is lower, i.e. the extra optima are network artefacts. The profile
error against the exact answer is dominated by the network (median 0.16, max 0.76 in
Delta chi^2) and identical for both.

### Trust-boundary convergence

A successful step still truncated by the trust radius must be remeasured before declaring
convergence. The curvature error bound also cannot certify an incumbent found by a different
probe. Such steps return to the gradient loop; exhaustion remains flagged. This can
increase call cost and preserves the noise-scale/default tolerances.

A step whose true value is not finite (a point the backend refuses) is a failed step: the
radius shrinks, and the curvature model is left alone. Written into the model, the excess of
-inf enters as the stiffest curvature allowed, which cost the node and its
successors their budget.

## Known limitations and open items

* **Real-likelihood noise and incomplete coverage.** Finite-difference stopping remains
  approximate at the measured noise floor; record stop flags and cross-start checks.
  The main it20 real comparison covers logGeff -5 through -1.2.
  The tail through -0.8 and active-bound release have been validated separately,
  with the remaining stop flags documented above. No bound was widened; tau_reio keeps
  its physical upper limit 0.1.
* **Wall time.** `corrected` keeps up to `workers` backend processes busy, but rounds
  end at a barrier, so its speed-up is below `workers` (modelled 3.5x at 4).
* **Where the calls go.** Most `corrected` calls are verification probes. A sequential
  test (stop after 2 probes when the estimate is clearly small) is the obvious next saving.
* **Judging a change.** Judge it on a network standing in for the truth with added noise of
  2.5e-3 (calls and the *worst* local gap) and on real CLASS++ nodes,
  never on the 31D target alone.
* `max_passes` applies to the legacy surrogate scan only.

## Tests

`tests/test_hybrid.py` (ascent with bounds, saddle escape, basin hopping, concurrent
ledger), `tests/test_residual_newton.py`, and `tests/test_profiles_workflow.py` (end-to-end
toy study for each optimiser pair, resume without new calls, pre-hybrid manifests).


## Reproducing the packaged checks

Use the environment in `environment.yaml`. The test suite uses analytic objectives and needs
no CLASS data:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_*.py'
```

True-stage journals are not overwritten. The raw experiments behind the numbers above are not
bundled with this repository; the numerical summaries
here state their scope and limitations. Evaluate broader examples before assuming the measured SINU call
savings apply to another target.

## Joint 2D profiles

`dimension: 2` studies use the hybrid scan and residual Newton. Both selected
coordinates are held fixed simultaneously; every other coordinate is optimised
within the original physical domain. This is a joint surface, not a product of
separate 1D profiles. The implementation is in `profiles/hybrid_surface.py`.

The surrogate scan batches the full grid and an unrestricted anchor. Candidate
optima can move between the four adjacent grid nodes and the anchor. Projected
first/second-order checks operate only on free nuisance coordinates. Surrogate
point-value, gradient and Hessian evaluation counts are recorded separately from
true objective calls.

True refinement starts each row at the second-coordinate node nearest the
surrogate anchor, then follows two chains outward along that row. Residual state
is shared only with each chain's predecessor; rows never depend on completion
order. `workers` schedules these chains and their probe batches; `chain_length` adds
seeds along each row as in 1D ([Many workers](create_profiles.md#many-workers)).
Both fixed physical values are pinned exactly on every backend request. The
unrestricted true anchor is refined last from the best evaluated grid node.

Each completed node saves its native value, coordinates, direct evaluation,
solver diagnostics and continuation state in `true_surface/node_*.json`. A
checkpoint signature covers the grid, bounds, starts and Newton settings; the
workflow manifest also guards the model and likelihood. Resume imports journaled
finite values into the exact-coordinate cache, preserves lifetime call counts,
and recomputes only unfinished nodes. Imports and cache hits are never counted
as new objective evaluations. A per-node budget limit retains the best evaluated
point and a nonzero diagnostic flag; stopping does not certify a global optimum.

```yaml
dimension: 2
parameters: [x_28, x_29]
workers: 4
# To reuse a prepared model without augmentation or training:
retrain: false
added_points: 0
```

For 2D, `surrogate_optimizer: legacy` selects the alternating L-BFGS-B scan, with the same true stage.
Augmentation/validation nuisance designs use L-BFGS-B. Use a new output directory to compare methods.

The 2D extension is checked against analytic joint profiles, active nuisance
bounds, exact fixed-coordinate calls, serial/parallel equality, interruption and
cache recovery, fully fixed 2D targets, and complete saved-model workflows with
and without retraining. Large 31D 2D reruns were not completed;
no performance or real-likelihood 2D validation claim is made.
