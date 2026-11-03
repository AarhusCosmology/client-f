# Boost with a quadratic background and a region

Two opt-in keys of `acquisition.boost`, both off by default:

```yaml
acquisition:
  boost:
    n: 2000
    start_iteration: 15
    density_filter: true
    background: quadratic   # the network becomes a fixed quadratic plus a network
    region: 2.0             # chains and audit see the network only where the boosts' points reach
```

They answer what the ten-boost study of 5 October ran into
([posterior_density_boost.md](posterior_density_boost.md)): after a few boosts a network can
rise far above the best true value in places no training point is near, the audit that
repairs those places damages the core, and an evidence over a wide box would be at the mercy
of both.

## What a boost run does with them

1. **First boost** (`start_iteration`): the network of the ordinary loop supplies the boost
   points, as without the keys.
2. **First training that holds boost points**: the quadratic is fitted to the training set and
   this one training starts from random weights. The region is built from the boost points.
   (It is the first training after the boosts' start. When the boost at the start found no mode
   and acquired an ordinary batch instead, that training holds no boost point yet: the
   quadratic is fitted to the training set as it is, and stays.)
   The audit searches inside the region, the tempered chain runs inside it (its walkers start
   inside), and the boost's own chains inherit that. The starting walkers are drawn uniformly
   in the region and kept where they are in the box; along a parameter whose posterior fills
   its range or lies against a bound, the region reaches beyond the box. When 200 draws have
   not found the walkers (0.46% of the draws inside at network 18 of an axion run, whose
   farthest boost row had gone from 9.7 to 18.2 sigma), the draw closes in on the region's
   centre.
3. **Every later boost**: warm start as usual, with the quadratic unchanged; the region is
   rebuilt from all boost points so far.

## What was found, on the two analytic targets

All numbers are errors of the network's log-likelihood on exact posterior draws (24 000 at
T = 1), on the 58-parameter Gaussian and the two-mode analytic twin of SINU (16 parameters).

**The far false maxima belong to the long chain of warm starts, not to the boosts.** Networks
trained once from random weights on the same data have none, in two whole-box searches each
(plain, both targets). Carried through four further boosts with the loop's own warm start
and schedule and no audit at all, they still have none (30 searches). The chain of 16
warm starts had 5 to 11 per search on the Gaussian.

**A cold start is also the larger gain in accuracy.** Twin, same data, plain network:

| boosts in the data | trained once from random weights | the warm-started chain |
|---|---|---|
| 0 | 0.32 | 0.75 |
| 1 | 0.25 | 0.73 |
| 5 | 0.19 (0.19 when warm-started from the cold start through the boosts) | 0.28 |

**The audit's repairs are what damaged the core.** Five refuted maxima, 10⁵ log-units deep,
carry 171% of the whole training loss; fitting them, and fine-tuning at 1e-5 at all, moved the
Gaussian's core error from 0.69 to 1.28. On the network that had them, the first repair of an
audit without the region took the fit near the best from 0.77 to 215.

## The quadratic background

`model/network.py`: `QuadraticBackground`, `fit_background`, `build_model(..., background=)`.

The model is q + network, q(θ) = top − ½ (θ − m)ᵀ P (θ − m). m and the covariance C are those
of the training rows within T·ndim log-units of the best (T the acquisition temperature), P =
κ C⁻¹ with κ the one number that best matches those rows' depths, top the best value.
Deliberately crude, and never refitted: it is stored in the model file as non-trainable
weights, and a later training takes it from the previous model. The network's targets are the
log-likelihood minus q; a warm start keeps the function exactly as before.

| trained once from random weights, first boost's data | plain | with the background |
|---|---|---|
| Gaussian, 58 parameters: error at T = 1 / rms at T = 7 | 0.94 / 11.2 | 0.33 / 4.3 |
| twin, 16 parameters: error at T = 1 / offset between the modes | 0.25 / −0.21 | 0.19 / −0.09 |

Through three further boosts the twin's network went 0.19, 0.23, 0.20, 0.14, with the offset
between the modes at −0.03, means within 0.006 σ and widths within 0.3% of exact.

**It needs the region on a likelihood that is not Gaussian.** On the twin the background
creates false maxima that the plain network does not have: 16 in a search, the highest 2041
above the best where the truth is 5051 below. Along the valley the truth is flatter than the
quadratic at both ends of the logGeff range, the network has to add 14 to 39 log-units there,
and it carries that rising correction into the corners, where there is no data. They grow
with the boosts (69 after three, to +4187). A background of up to 12 Gaussians changes
nothing: the far rows fall faster than any Gaussian.

## The region

`sampling/region.py`. Built from the boost rows alone: the T = 1 posterior points whose true
values are known and lie within T·ndim/2 of the best (`RUN/boost/boost_it_N.csv`). One
Gaussian per mode, in box units; ρ(θ) is the smallest, over the modes, of the Mahalanobis
distance to the mode's Gaussian divided by the largest such distance among that mode's own
rows; inside means ρ ≤ `region`. Saved as `RUN/boost/region_it_N.npz`.

Cloud matching uses verified rows, not mode numbers (which are weight ranks). Each older
cloud joins the current group with the largest fraction of rows covered by either cloud's
ellipsoid, provided the two are one mode: the mean of the one with fewer rows must lie
within the reach of the other, in that other's shape. Otherwise it stays as it is, as does
a mode that is absent from a boost. An older cloud is assigned once; several old clouds may
join one current group when a later boost finds them as one. Distinct groups of the current
boost never share historical rows. This geometric association is not a test of physical
mode connectivity. (Any overlap at all is not enough: one stray row merged a
mode's 3 or 4 rows into the other mode's 10000 in 2 of about 620 trials on the twin's boost
files, and that cloud's reach went from 10.6 to about 29.)

**A sparse cloud takes the shape of a well-sampled one.** With at most twice the dimension
in distinct verified points a cloud has no covariance to speak of. It is given the shape of
the well-sampled cloud with the most rows, its own centre, and it reaches as far as its own
rows do, never farther than the cloud whose shape it has. Without a well-sampled cloud the
shape is the scatter of all the clouds about their own means once that is well sampled;
until then, and for a single verified point, the cloud is left out of the region, counted
in the log line and in the saved file (`left_out`), and the run goes on. For deciding
whether two sets of rows are one mode a sparse set is taken to be as wide as the cloud
whose shape it borrows: its own few rows say where it is, not how far it goes.

Why not a shrunk covariance: shrinkage pulls towards a sphere in box units, and a real cloud is hundreds
of times narrower in some directions than in others (principal widths 0.00025 to 0.15 of
the box on the 58-parameter target, 0.0017 to 0.2 on the twin).

| mode known from | shrunk covariance | shape of the well-sampled cloud |
|---|---|---|
| 58 parameters, up to 116 rows: edge along the narrowest direction, against the region of all 20 000 rows | 64 to 80 times as far | 0.5 to 0.9 times |
| the same: share of the sparse region inside the region of all rows | below 0.5% | 100% |
| twin, network 25, strong mode from a few of its rows: draws in which its T = 1 chain sat where the network is 6300 to 11 600 above the truth | 3 of 9 (5 to 10 rows) | 0 of 11 (5 to 32 rows) |
| the same: the network's weight of the strong mode (0.198 from all rows) | 0.04 to 0.20 | 0.194 to 0.201 |

Modes of one posterior differ in shape, so the borrowed one is an approximation: on the twin
a tenth to a quarter of such a region lies outside the region the mode's own 10 000 rows
give. It holds until the mode has its own rows, which is two boosts for a mode that gets
its share of the points. A single distinct point must not raise an error, which would come
from the saved boost files and so at every restart: it is left out and counted, as above.
Well-sampled clouds, and so every single-mode run with its hundreds or thousands of rows per
boost, keep their own covariance, and saved region files are not rewritten.

Who sees it, from the second boost on:

* the **tempered chain** and through it the **boost's own chains**: the surrogate is the
  network inside and, outside, a value that does not look at the network at all
  (`SurrogateLikelihood(model, metadata, region=)`);
* the **audit**: its searches are projected onto the region (`find_maxima(..., project=)`),
  so only maxima a T = 1 consumer could meet are evaluated and trained on;
* `client.py evidence RUN`, by default (`--region none` for the bare
  network; [evidence.md](evidence.md));
* `client.py profiles`, by default (the profile setting `region`,
  [create_profiles.md](create_profiles.md#networks-trained-on-boosts)): the searches for a
  profile's optima are given the network inside the region and the wall outside, as a Keras
  model at double precision (`model.network.restrict_model`). A profile of the bare network
  of a real 49-parameter run found maxima 40 posterior sigma from the data.

Measured with the edge at 2:

| | Gaussian, 58 parameters | twin, 16 parameters |
|---|---|---|
| exact posterior outside, of 2·10⁶ draws | 0 (also at 1.25) | 0 (1.5·10⁻⁴ at 1.5) |
| known false maxima, in multiples of the reach | 43 to 55 | 3.2 to 11.5; 2.3 after three boosts |
| audit inside the region, on a network that has them | 0 true calls, clear, network unchanged | the same |
| tempered chain inside it: true depth at 5 / 50 / 95% | 127 / 172 / 205 (exact 128 / 173 / 205) | 23 / 45 / 74 (exact the same) |
| evidence over the whole box, nested sampling (BEST, 1000 live points) | not run | −5341.82 ± 0.20 against −5341.50 exact, from the network with maxima at +2041 |

## Settings

| key | default | |
|---|---|---|
| `background` | absent | `quadratic`: see above; the first training with boost points is a cold start |
| `region` | absent | a number above 1, the edge in multiples of the boost points' reach; 2 is the measured setting |
| `region_quantile` | absent | with `region`: a number in (0, 1], each cloud's reach is that quantile of its points' distances instead of the farthest point (below) |
| `train_domain` | absent | with `region`: a number of at least 1, the boost networks are trained only inside that edge of a fixed ellipsoid (below) |

## The training domain and the quantile reach (17 October 2026, opt-in)

Measured on five runs: it is what the axion run needed, and it costs accuracy on
near-Gaussian likelihoods ([boost.md, two recommended settings](boost.md#two-recommended-settings)).

The axion monodromy run (24 parameters, Planck TT at 15 sigma) has structure of a few
log-units in three of its parameters, the oscillation's amplitude, frequency and phase, and
its networks did not learn it. Two things in the run's control branch (chain search on, the
region at edge 2) pointed at the training set's dynamic range and at the region's reach:

* **The far points set the network's output scale.** The network's output is standardised by
  the spread of its targets (the truth minus the quadratic): 52 280 log-units over the whole
  training set of network 17, 22 without the 11% of its points more than 1000 below the best
  (they reach -2.4 million). msre gives those points almost no weight (its effective number of
  points was 40 100 with them and without), but an optimiser's step is fixed in the weights,
  so in units of the scale: the oscillations were two millionths of it. Those points also
  sit up to 420 000 above the quadratic, which falls faster than the truth there: the
  rising correction of "The quadratic background" above, in the directions the region's wall
  then stops.
* **The false maxima sat on the region's wall, and one far point moved the wall.** From
  network 15 on, the 366 false maxima the audits evaluated lay at 2.00 times the reach, the
  posterior's own points within 0.8 to 1.0 of it. At network 18 one far boost point took the
  farthest from 9.7 to 18.2 sigma; the region doubled and took in a false peak (16% of that
  network's posterior, about 780 above the best). The 99.9% point of the same distances was
  7.3, 7.5 and 7.6 at networks 12, 17 and 18.

`train_domain: E` trains every boost network (from the first training after the boosts'
start) only on the points inside an ellipsoid: the mean and covariance of the points within
T * ndim of the best, the ones the quadratic background is fitted to, reaching E times as far
as the farthest of them (training/domain.py). It is built once, at that first training, from
the training set there, saved as `boost/train_domain_it_N.npz`, committed with that training
and read by every later one, so that it does not grow with the boosts as the farthest-point
reach does. The background, the network's input and output scaling, the loss's normalisation,
the validation split and the audit's fine-tunes all see the cut set. The points outside stay
in the training set's files and in its other uses (the acquisition's density, the derived
emulator). In the control branch the domain built at network 9 held, at edge 1, 84% of
network 17's training set with log L from -697 to -408, and the region at edge 1.5 lay wholly
inside it at every network (at edge 2, 8.4% of network 17's region did not).

`region_quantile: q` makes each cloud's reach the q quantile of its points' distances from its
centre; which points are one mode is still decided by the farthest point. A sparse cloud's own reach, capped at
its donor's, is the same quantile.

Both need the region, which keeps every chain, search, evidence and profile inside it: a
network trained on part of the box has nothing to hold it down elsewhere. The loop prints, at
every boost, an estimate of the uniform-volume share of the box-clipped region (to its edge)
outside the training domain. It uses 4000 proposals from the unit box or the inflated
ellipsoids, choosing the proposal with the smaller total volume. Ellipsoids are selected
proportionally to their volumes; draws outside the box are discarded and a point inside
`m` ellipsoids has weight `1/m`. Thus overlaps contribute once to the union volume.
The estimate is `sum(weight * outside) / sum(weight)`, not a posterior mass fraction.
Unlike `Region.start`, this diagnostic never moves draws towards ellipsoid centres.
The log reports the volume-weight Kish ESS; with no hits coverage is unknown, and
below 200 effective samples it is marked poorly resolved rather than assessed against
the 1% warning threshold. Otherwise an estimated share above 1% produces a warning.
This Monte Carlo estimate cannot prove containment, even when it returns zero.
The coverage fractions quoted above for the control branch were measured with equal-cell
initialisation draws, which are not uniform in volume, and should not be interpreted as
uniform-volume measurements. A branch of a
run at its boosts' start leaves the domain behind and builds its own (the file is named after
the network whose training built it).

Without the keys a region reaches to its farthest point: 218 regions of 41 runs' boost files,
built with and without the code for these keys, are identical. `tests/test_train_domain.py`
covers the domain and the reach; `tests/mpi_train_domain_smoke.py` runs a branch with both keys
through the launcher (`hpc/launch.py`) on two ranks, restarted once. What they do for the
networks is in [boost.md](boost.md#two-recommended-settings).

## What is not established

* The two ten-boost runs below are single trajectories on analytic likelihoods. They show
  that the loop with both keys runs ten boosts without a false maximum and reaches its
  accuracy in one or two; they do not establish how many boosts a problem needs.
* The cold start and the quadratic were introduced together in those runs; what each
  contributes was measured on single networks only (the tables above).
* No false maximum appeared in those runs at all, so they did not need the region. That it
  does its work when one is there rests on the separately trained twin network above.
* The evidence with the region is validated on the twin only.
* Profiles in the region are validated on one real 49-parameter run, with the edge at 1, and
  on the twin with the edge at 2. The run's own edge of 2 has not been used for profiles on
  a real likelihood.
* A search cannot find a narrow false maximum inside the region; there the boost points, which
  are posterior samples with true values, are the check.
* The region holds the chains to the modes the first boost found: a boost no longer discovers
  a mode.
* On the twin the lowest far maxima (10 to 20 above the best) had come to 2.3 times the reach
  after three boosts. One that enters the region is the audit's to find; its repair still
  fine-tunes at 1e-5, and a cool-down after a repair is not implemented.
* A warm-started boost at the loop's rate scatters from one training to the next (0.25 to 0.38
  on the Gaussian with the background); restarting at 1e-6 is steady and gains little.

## Ten boosts from the same starting points, 6 to 7 October 2026

The study of 5 October ([posterior_density_boost.md](posterior_density_boost.md)) was repeated
with the two keys and nothing else changed: the same iteration-15 networks and training sets,
2000 accepted density-filtered points per boost, the loop's training schedule (restart at
1e-4, down to 1e-6 on plateaus), the audit with its budget of 200 true calls per iteration,
and the same per-round diagnostics. No CLASS++ evaluation was made.

* **First boost**: from the network of the ordinary loop, as in the study of 5 October.
* **Network 16**: the quadratic is fitted (Gaussian: 27 917 rows within 406 of the best, scale
  6.56; twin: 30 021 rows within 112, scale 5.85) and the network is trained from random
  weights: 50 and 39 minutes, against about 20 for each later warm start.
* **Networks 17 to 25**: warm starts with the quadratic unchanged. Before every audit the
  region is rebuilt from all boost points; the audit, the tempered chain and the boost's
  chains work inside it.

Each cell is *before / after*: the study of 5 October, without the keys, and with the two keys. Round 0
is the common starting network. **T = 1 rms** is the root-mean-square error of the
log-likelihood on 24 000 exact posterior draws; the **centred sd** leaves out its constant
offset, which changes no posterior. **T = 7 rms** is the same on 4000 exact draws at the
acquisition temperature. The credible metric at level p is (|upper − upper_ref| + |lower −
lower_ref|) / (upper_ref − lower_ref); the worst over the parameters is given.

Gaussian, 58 parameters:

| boosts | worst CM 68% | worst CM 95% | T = 1 rms | T = 1 centred sd | T = 7 rms | audit calls |
|---|---|---|---|---|---|---|
| 0 | 0.309 / 0.309 | 0.147 / 0.147 | 6.073 / 6.073 | 1.212 / 1.212 | 15.00 / 15.00 | 13 / 13 |
| 1 | 0.151 / 0.075 | 0.092 / 0.034 | 2.780 / 0.318 | 1.022 / 0.314 | 9.38 / 3.56 | 37 / 0 |
| 2 | 0.085 / 0.040 | 0.048 / 0.031 | 0.574 / 0.260 | 0.571 / 0.251 | 8.00 / 3.45 | 0 / 0 |
| 3 | 0.107 / 0.057 | 0.059 / 0.029 | 0.642 / 0.768 | 0.628 / 0.313 | 7.52 / 3.72 | 1 / 0 |
| 4 | 0.058 / 0.034 | 0.035 / 0.031 | 0.482 / 0.243 | 0.476 / 0.225 | 6.20 / 3.38 | 0 / 0 |
| 5 | 0.733 / 0.048 | 0.367 / 0.028 | 4.196 / 0.258 | 2.806 / 0.219 | 13.60 / 3.56 | 12 (not clear) / 0 |
| 6 | – / 0.054 | – / 0.036 | – / 0.299 | – / 0.257 | – / 3.72 | – / 0 |
| 7 | – / 0.040 | – / 0.026 | – / 0.226 | – / 0.210 | – / 3.53 | – / 0 |
| 8 | – / 0.037 | – / 0.027 | – / 0.313 | – / 0.253 | – / 3.84 | – / 0 |
| 9 | – / 0.037 | – / 0.031 | – / 0.294 | – / 0.267 | – / 3.86 | – / 0 |
| 10 | – / 0.051 | – / 0.028 | – / 0.421 | – / 0.273 | – / 3.73 | – / 0 |

The 5 October run was stopped after its fifth boost: the audit was not clear (a maximum 2040
above the best was left) and the core error had gone from 0.48 to 2.81.

Analytic twin of SInu, 16 parameters, two modes (exact weight of the strong mode 19.03%):

| boosts | worst CM 68% | worst CM 95% | T = 1 rms | T = 1 centred sd | T = 7 rms | strong mode, % | audit calls |
|---|---|---|---|---|---|---|---|
| 0 | 0.121 / 0.121 | 0.084 / 0.084 | 0.994 / 0.994 | 0.745 / 0.745 | 2.56 / 2.56 | 14.5 / 14.5 | 0 / 0 |
| 1 | 0.112 / 0.030 | 0.070 / 0.022 | 2.311 / 0.246 | 0.734 / 0.224 | 3.17 / 1.05 | 14.7 / 17.9 | 8 / 0 |
| 2 | 0.074 / 0.025 | 0.051 / 0.008 | 0.559 / 0.173 | 0.465 / 0.166 | 1.95 / 0.93 | 15.8 / 19.0 | 0 / 0 |
| 3 | 0.030 / 0.014 | 0.077 / 0.014 | 0.833 / 0.179 | 0.356 / 0.178 | 1.88 / 0.96 | 18.6 / 18.6 | 2 / 0 |
| 4 | 0.085 / 0.043 | 0.034 / 0.020 | 0.561 / 0.235 | 0.326 / 0.181 | 1.64 / 1.03 | 16.6 / 17.4 | 0 / 0 |
| 5 | 0.043 / 0.021 | 0.018 / 0.070 | 0.310 / 0.202 | 0.278 / 0.170 | 1.44 / 0.97 | 18.3 / 19.4 | 0 / 0 |
| 6 | 0.076 / 0.036 | 0.025 / 0.019 | 0.329 / 0.235 | 0.291 / 0.175 | 1.41 / 1.03 | 17.5 / 18.1 | 1 / 0 |
| 7 | 0.037 / 0.031 | 0.015 / 0.015 | 0.246 / 0.185 | 0.224 / 0.168 | 1.24 / 0.98 | 17.8 / 17.9 | 0 / 0 |
| 8 | 0.054 / 0.031 | 0.021 / 0.069 | 0.283 / 0.161 | 0.236 / 0.158 | 1.20 / 0.94 | 16.7 / 20.0 | 1 / 0 |
| 9 | 0.037 / 0.049 | 0.016 / 0.022 | 0.218 / 0.170 | 0.213 / 0.168 | 1.22 / 0.96 | 17.4 / 17.6 | 0 / 0 |
| 10 | 0.045 / 0.031 | 0.022 / 0.013 | 0.371 / 0.157 | 0.207 / 0.157 | 1.20 / 0.99 | 18.7 / 20.1 | 6 / 0 |

**The gain comes at once.** After the first boost the centred error is 0.31 on the Gaussian and
0.22 on the twin, against 1.02 and 0.73 at the same round without the keys; the twin is then where the
5 October run arrived after seven boosts. From the second boost on neither run improves by
more than its own scatter: 0.21 to 0.31 on the Gaussian with no trend, 0.16 to 0.18 on the
twin with a slow gain (0.166 after two boosts, 0.157 after ten). The T = 1 rms follows the
centred sd except where a training leaves a constant offset (0.77 at the Gaussian's third
round is an offset of −0.70).

**The credible metric has a noise floor there.** Scoring the two halves of one chain separately
gives, after ten boosts, 0.055 and 0.055 for the Gaussian's worst 68% metric and 0.032 and 0.037
for the twin's; at the twin's fifth round the worst 95% metric is 0.069 in one half and 0.009 in
the other, so the 0.070 of that round is chain noise. Below about 0.03 to 0.05 the worst
credible metric does not tell two networks apart. The error on exact posterior draws has no
such floor and is the better measure of the core.

**Audit and region.** All 20 audits were clear without a true evaluation; the 5 October runs
spent 50 (Gaussian, five boosts, the fifth not clear) and 18 (twin) at the boost stage. A search
of the whole box, without the region, finds no maximum above +10 in the Gaussian's networks 20
and 25 or in the twin's network 25. No boost point was left out as unverified. The Gaussian's
region is steady (reach 10.3 to 10.6 sigma). The twin's grows, from 6.9 and 6.2 sigma for its
two modes to 8.3 and 10.3 at the second boost and 10.6 and 11.6 by the tenth: a few points of
the new, better network lie at up to 1.66 times the first boost's reach with true depths of
only 9 to 16, real posterior the first boost had not reached. The reach is set by the single farthest point; a high quantile
(`region_quantile`) would be the steadier definition.
