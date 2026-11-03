# Profile likelihoods and Bayesian evidence

## Adaptive profile augmentation

Run on an existing checkpoint, writing a separate experiment directory:

```sh
python benchmarking/augment_profiles.py RUN_DIR -it 5 --reference EXACT_PROFILES.json \
    --budgets 1000,2000 -o OUTPUT_DIR
python benchmarking/compare_profile_augmentation.py OUTPUT_DIR
```

Defaults profile the six cosmological parameters, on 61 points covering the central
half of the metadata prior box. Supply an exact reference on this same grid. Original
run artifacts are read-only. `--params`, `--n-grid`, `--grid-frac`, `--seed` and the
budget list are configurable. Nonempty output directories are refused.

Each round starts from the current surrogate profiles. It retains seven full-box
coverage nodes per parameter and adds the peak coordinate and any 1/2-sigma crossings.
Duplicate nodes are merged. Crossings have allocation weight 4, the peak weight 3,
and broad coverage weight 1; a minimum per-node allocation is retained. Nuisance
optima are recomputed at the selected node coordinates.

Each node gets an explicit anchor, unless that duplicates an existing point. Other
draws alternate between prescribed conditional quadratic deficits of 0.05, 0.2, 0.5
and 1.0, and broader conditional Gaussians at T=0.25, 0.5 and 1.0. Geometry uses the
same global quadratic fit/precision repair as the paper baseline, retaining its 1e-3
eigenvalue floor. No analytic target geometry is used for acquisition.

Proposals outside the prior box are rejected. After 128 failed proposals for a point,
60 sweeps of coordinate Gibbs sampling in the truncated conditional Gaussian provide
a feasible fallback. For a narrow-radius component, fallback temperature is 2D/(d-1),
where D is its requested deficit. Thus fallback points do not have an exact prescribed
radius and are not claimed to be independent equilibrium draws. This is a training
design, not a posterior sample. Fallback use and realized radii are recorded. There
is no clipping; intentional anchors may lie on a constrained maximum's box face.

For budgets 1000,2000, the larger dataset contains the smaller one; its second round
follows the +1000 surrogate. Both final networks start independently from the original
checkpoint with function-preserving scaler compensation. Training uses LR 3e-5 with
the saved plateau factor/patience/minimum/grace and epoch ceiling. Original validation
rows stay fixed between arms; entire augmentation node groups are assigned to
validation. Their membership masks are saved. Held-out original rows may have been
used when the source checkpoint was trained; this is not an independent test set.

Outputs include full augmented CSVs, models, histories, source hashes, per-draw/node
provenance, optimizer diagnostics, profiles and true evaluations at surrogate optima.
The comparison plots all three profiles against the exact reference, reports grid dPM
and profile-shape/raw residual RMS, and refines crossing brackets using nuisance
optimization and Brent roots. Where the exact reference is verified quadratic, its
analytic roots provide the refined dPM denominator and endpoints. Unusable roots are
flagged rather than silently replaced by interpolation. Global optimality remains
uncertified; inspect stationarity diagnostics for grids and root evaluations.

These experiments test the combined changes in design, warm starts and fine-tuning,
not separate causal effects. One seed does not establish a robust improvement.
Unprofiled directions and posterior accuracy require separate checks. The global
quadratic/residual-network and local-correction alternatives have not been added.

## Numerical checks

The profiler writes schema version 2, with the fields `profiles`, `logl`,
`xopt`, `best_logl` and `best_x`. `full_parameter_names` specifies the column
order of the full optimum vectors; `names` remains the profiled subset.

Each profile includes `optimization`, with one diagnostic per node. The selected
point reports `stationary` and `projected_gradient_inf` in box-scaled coordinates,
plus the optimizer status, message and call counts. `attempts` includes all starting
points and local solves from that selection call. The best finite starting point is
retained if a local solve fails or returns a worse value. Up to `--retry-starts 2`
independent starts are tried if stationarity has not been reached.

`optimization_quality` records anchor consistency, stability between complete profile
sweeps, stationarity of selected points, passes completed and pass-cap exhaustion.
`converged` requires all these numerical checks; it does **not** certify a global
maximum. Tolerances are configurable with `--gtol`, `--ftol` and `--value-tol`, whose
defaults are 1e-4, 1e-10 and 1e-4 respectively. Gradient tolerance is in scaled
coordinates; value tolerance is in raw log-likelihood units. Inspect these diagnostics
when using a different output scale or precision. The scorer warns if checks are
incomplete. An older JSON file without these fields remains readable, but has no quality diagnostics.

`--true` evaluates truth **at surrogate optima**; it does not perform
true constrained optimization. The JSON records this as `true_evaluation_kind`.

The metric treats the piecewise-linear confidence set explicitly. Both endpoint
roots are included, disconnected sets are not replaced by their outer envelope, and
empty or zero-width sets are not called unconstrained. These cases have a flagged
NaN single-interval score; the CLI reports excluded counts. `confidence_set()` returns
all connected components for callers needing them. An unconstrained reference with a
constrained surrogate is flagged `spurious constraint` and scored by endpoint error
over the full grid width. A connected set follows the one-/two-sided definition of section 2.

`--floor` fits the Gaussian curvature and off-grid centre in the crossing region.
It returns no estimate when there is no two-sided interval or the region is not
consistent with an unbound quadratic Gaussian profile. Shared interpolation errors
can cancel between curves, so this estimate is not a hard floor on dPM. Refine the
actual crossing brackets when resolving small differences.

The experimental results below were measured before these checks existed and have not been repeated with them.

Two modules extend CLiENT's assessment beyond the posterior:

| module | what it does |
|---|---|
| `benchmarking/profile_likelihood.py` | 1D profile likelihoods from a trained surrogate |
| `profiles/metric.py` | the profile metric `dPM`, and its resolution floor |

They were developed against a 31-dimensional analytic target whose posterior, profile and
evidence are all known in closed form, so every number below was checked against an exact
reference rather than against another sampler.

---

## Why these are separate from the credible metric

The posterior, the profile and the evidence weight the likelihood surface in different
places, and an emulator accurate for one is **not** automatically accurate for another:

- the **posterior** is governed by the bulk of the probability mass;
- the **profile** by a thin ridge of maximum likelihood, which is where training data is
  scarcest;
- the **evidence** by the *typical set*, a shell roughly `d/2` log-units below the peak
  where most of the still-appreciable prior volume lives.

Measured on the 31D target: a network with **6.5× the profile residual rms** of another
gave an evidence **indistinguishable from it** (0.13σ apart). The evidence integral simply
does not sample the region where that network fails. The practical consequence is that
**profiles and evidence need separate validation** — neither result licenses the other.

A second trap worth stating: **validation loss predicts neither.** On the same target it
ranked networks in the *reverse* order of their profile accuracy, because it scores fit to
the training distribution while the profile optimiser deliberately climbs to wherever the
surrogate is most optimistic, which is by construction the least-sampled place.

---

## 1. Profile likelihoods

```bash
python benchmarking/profile_likelihood.py RUN_DIR -it 10 -o profiles_it10.json --true
```

For each parameter this walks a grid, maximising the surrogate over all others at each
point. With `--true` it also evaluates the real likelihood at each profile optimum, which
is what makes the `raw` residuals in the output meaningful.

Gradients come from the network itself, so L-BFGS-B gets analytic derivatives rather than
finite differences — a direct pay-off of emulating the likelihood rather than observables.

### The optimiser is not a plain multi-start, and cannot be

A naive multi-start produces profile points that *beat* the global anchor, i.e. negative
Δχ², because in 30-odd dimensions a random start rarely finds the ridge at all. Three
things fix it and all three are needed:

1. profiles are walked **outward from the anchor**, each grid point warm-started from its
   neighbour, so the optimiser never re-finds the ridge from scratch;
2. **neighbour relaxation** re-optimises each point from its neighbours' solutions until
   nothing improves, repairing points that fell off the ridge;
3. the **anchor is re-polished** whenever any profile point beats it, and the sweep
   repeats.

If the run exhausts `--max-passes` while still improving it says so (a warning that the
numerical checks are incomplete, and `pass_cap_reached` in `optimization_quality`), and the
reported best fit should then be read as a lower bound.

### Scaled coordinates are load-bearing

Every optimisation runs in `u = (x − centre)/halfwidth`, so each free coordinate is O(1) on
[−1, 1]. This is not cosmetic. Cosmological parameter vectors mix `omega_b ~ 0.02` with
`ps_A_100_100 ~ 200`; L-BFGS-B builds its inverse-Hessian approximation from that geometry
and its line search stalls far short of the optimum. On the 27D Gaussian it5 network the
raw-unit search reported −23.3518 after sixty passes, while the scaled search reaches
−22.6676 — the network's actual maximum — from the same 31 starts in 0.1 s.

The failure is quiet and reads like a budget problem: the raw-unit search reports monotone
progress pass after pass and trips the pass-cap warning, which invites more passes
rather than suspicion of the coordinates. Re-measured with scaling, three 31D ridge
anchors moved by +0.41, +0.58 and +1.05 log-units.

It is worth being explicit that this is **conditioning, not float32 precision**, because
the latter is the tempting explanation. The gradients are exact autodiff derivatives of the
network, not finite differences, and their noise floor (~1e-7 relative, so ~3e-6 absolute
near lnL = −23) is four orders of magnitude below the gaps above. Replacing L-BFGS-B with a
gradient-free optimiser would therefore discard good information to treat the wrong cause,
and Nelder-Mead in particular degrades badly beyond ~10 dimensions.

### Two quantities that are easy to conflate

```
below peak = 2[lnL_max      - lnL_true(x̂)]    how far the located optimum is from the truth
bias       = 2[lnL_true(x̂)  - lnL_emul(x̂)]    how wrong the surrogate is AT that point
```

They are independent. On the 31D target one network had a bias of only −0.032 yet its
optimum sat 1.261 in Δχ² below the true maximum: locally faithful, but under-predicting
near the real peak so the optimiser never climbed there. **A small bias is not evidence
that the maximum has been found.**

### Grid choice

`--grid-frac` spans the central fraction of the prior box; the default 1.0 covers all of
it. That is right for seeing where an emulator is optimistic, but wrong for locating Δχ²
crossings — see the floor discussion below.

---

## 2. The profile metric

```bash
python -m profiles.metric profiles_it10.json -r exact_profiles.json
python -m profiles.metric exact_profiles.json --floor
```

`dPM` is the credible metric's formula with the endpoints taken to be the Δχ² = 1 and 4
crossings instead of posterior quantiles:

```
dPM = (|s_hi − r_hi| + |s_lo − r_lo|) / (r_hi − r_lo)
```

The **reference** decides whether a parameter is two-sided, one-tailed or unconstrained,
and the emulator is measured against that choice — exactly as in the credible metric. An
unconstrained level scores 0; a crossing the emulator fails to produce is anchored at the
grid edge, making `dPM` large rather than undefined.

### Always measure the floor first

This is the part most likely to waste your time if skipped. `dPM` is computed by linearly
interpolating a locally parabolic curve between grid points, which systematically
underestimates the interval width. Measured on the exact reference, where the closed form
is known:

| grid | 1σ floor | 2σ floor |
|---|---|---|
| 25 points over ±10σ (`--grid-frac 1.0`) | **0.054** | 0.011 |
| 61 points over ±5σ (`--grid-frac 0.5`) | **0.0016** | 0.0006 |

On the coarse grid the floor was **as large as the metric values themselves**, and
refining the grid *inverted the ranking of the emulators* — one arm went from best to
worst. A ranking that inverts under grid refinement was never measuring the emulators.

Do not narrow the span too far either: at ±3σ one parameter's profile only reached
Δχ² = 2.49, so its 2σ crossing fell outside the grid and the metric became undefined.
**±5σ with 61 points put every crossing of every parameter inside the grid** on the 31D
target, and is a reasonable starting point.

### dPM and residual rms measure different things

One network had 6.5× the residual rms of another yet an essentially equal `dPM` (0.0304
vs 0.0312). The rms is a pointwise average dominated by wherever the emulator is worst;
`dPM` asks only where two specific contours fall. Report both.

### A target-dependent degeneracy

Parameters with exactly flat true profiles are unconstrained at every level and score 0
for every network. On the 31D ridge four of ten parameters are in this class, so the
summaries are taken over the remaining six. This is a property of the target, not of the
metric.

---

## 3. Bayesian evidence

**Use `python client.py evidence RUN_DIR` ([evidence.md](evidence.md)).** It reports ln Z in
the run's own prior convention, Cobaya's or MontePython's, which does not depend on the
box, and flags a box that cuts the posterior. The numbers below were measured by integrating
over a chosen box, with a script that is not part of this repository.

Requires [BEST](https://github.com/AndreasNygaard/best-inference): install it, set `BEST_PATH`
to its checkout, or pass `--best-path`.

### Two conventions decide whether the answer means anything

**The prior offset.** Wrappers do not in general return the bare likelihood. The Cobaya
wrapper returns `log(L · π_hard)`, carrying a constant `−Σ ln(range)` over all parameters
— for the 31D target, −103.301361. The network is trained on it, so every emulator `ln Z`
inherits it and any exact value compared against must carry it too. Getting this wrong
looks like a 103 log-unit catastrophe, which is at least easy to spot.

**The box.** Nested sampling with a uniform prior computes `Z = (1/V_box)∫_box L dx`, so
`ln Z` depends directly on the box and is not a property of the likelihood alone. The
command's default box is the run's domain. `--n-sigma` asks for ±K on the **proposal** widths
— the convention `restrict_prior_bounds` uses throughout this package, *not* K times the
posterior sigma. The two differ by factors of 0.45 to 1.31 per parameter on the 31D target.
Parameters with hard physical limits or no meaningful sigma go in `--keep-bounds`.

**Where the box is centred, which is the trap.** The boxes measured here were `0.5*(lower +
upper) ± n_sigma·sigma` on the run's *metadata* bounds. CLiENT clips those bounds to the
wrapper's hard priors, so for a parameter bounded below at zero the metadata centre is **not** the
likelihood peak. On the 27D Gaussian the offsets reach 9.32 proposal sigma
(`galf_TE_A_100`), 6.11 (`gal545_A_100`) and 4.57 (`A_cib_217`) — larger than a ±4 sigma
box half-width, so the box can nearly exclude the peak along those directions. A ±4 sigma box
so built does not mean "±4 sigma around the best fit" whenever clipping has occurred.
`client.py evidence --n-sigma K` centres on the proposal centre instead, within the domain,
and that need not be the peak either.

This is a property of a box chosen this way rather than a defect: the peak is not known in
general. But it makes one rule mandatory. **Build any exact reference from the
`lower`/`upper` that the evidence records in its own output json, never from the
likelihood mean.** A reference centred on the mean instead produced a spurious 12.58
log-unit (109σ) disagreement on the 27D Gaussian; rebuilt on the recorded box, the same
network agreed at 1.27σ.

Verifying a small box probability needs care too: at P ≈ 8×10⁻⁷ in 27 dimensions, uniform
importance sampling is hopeless (relative se 0.56) and the `V·φ_max` bound is vacuous,
because in standardised units with a near-singular correlation matrix the density exceeds
1. Direct sampling from the Gaussian works — 120M draws gave 107 hits and confirmed Genz
to 0.56σ — and Genz's algorithm is the right tool provided it is cross-checked that way.

### Reading the result

BEST defines its σ as `sqrt(H/n_live)`, so a match between the two holds by
construction and says nothing about convergence. `H` was
stable at 24.6–25.1 nats across the eight runs on the 31D target (four emulators × two
resolutions): the information content is a property of the target, not the emulator.

All eight runs recovered the exact evidence to better than **0.12 in ln Z**, i.e. Z to
within 13%, in 31 dimensions. For context, on the Jeffreys, Kass–Raftery and Trotta scales
alike the threshold below which a Bayes factor says nothing is |ln B| ≈ 1, so at this
accuracy the surrogate is not the limiting factor for model selection. Two caveats: errors
from two separately emulated models need not cancel, and `ln Z` depends on the prior
volume, so a factor-of-two change in an unconstrained parameter's range shifts `ln B` by
0.69 — far larger than the emulator error.

---

## A worked sequence

```bash
RUN=results/<timestamp>_ridge31

# profiles on a grid fine enough for the metric
python benchmarking/profile_likelihood.py $RUN -it 10 --true \
    -n 61 --grid-frac 0.5 -o $RUN/profiles_it10.json

# check what the grid costs before trusting any score
python -m profiles.metric $RUN/exact_profiles.json --floor

# score
python -m profiles.metric $RUN/profiles_it10.json -r $RUN/exact_profiles.json

# evidence
python client.py evidence $RUN -i 10
```

The exact references for an analytic target (`exact_profiles.json`, `exact_evidence.json`)
are target-specific and are not produced by these modules; the scripts that generated them
for the 31D ridge are not part of this repository.

---

## What is not established

- **Convergence.** The drift criterion never reached its nominal 0.01 threshold on the 31D
  target, so no stopping criterion was exercised for these runs.
- **Disconnected credible regions.** The credible metric, scored on a single interval,
  returned bit-identical values for genuinely different emulators on a parameter whose 68%
  region is disconnected. That is a defect of the interval construction; exclude such
  parameters from summaries.
- **Seeds.** One seed per configuration throughout. Differences below about 1σ of the
  sampler's own error bar should not be read as real.
