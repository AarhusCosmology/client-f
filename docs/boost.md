# Boost: training points from the posterior itself

`acquisition.boost` turns an iteration's acquisition into a *boost*: instead of the tempered
batch, the new training points are posterior samples of the current network (T = 1), the same
number from every mode. It is meant for the end of a run, once the network is good, and above
all for posteriors with separated modes, whose relative weights the loop does not otherwise
get right. The single-mode case is the same step with one mode. Absent from the
configuration, there is no boost.

```yaml
acquisition:
  boost:
    n: 1000              # true evaluations per boost, shared equally among the modes found
    start_iteration: 15  # from this iteration on, every acquisition is a boost
```

```bash
mpirun -n 6 python client.py RUN_DIR -i 2    # a finished run, with the block added to its YAML: two boosts
```

## Two recommended settings

Which boost settings to use depends on how close the likelihood is to a Gaussian, that is, on
how well the quadratic background describes it (the evidence is below).

**Near-Gaussian likelihoods** (cosmological parameter estimation of the Planck kind: the LCDM
runs, the 49-parameter run, HiLLiK). The settings of the reference runs:

```yaml
acquisition:
  audit: {}                  # the audit's defaults (docs/audit_optima.md)
  boost:
    n: 3000                  # posterior points per boost
    start_iteration: 30      # at the latest: with convergence.metric tempered_batch_scatter the
                             # boosts begin when the ordinary loop has converged
    boosts: 10
    density_filter: true
    pool_factor: 20
    background: quadratic
    region: 2.0
```

**Likelihoods known to be non-Gaussian** (oscillations, several modes, curved ridges: the axion
monodromy run, SINU). The settings of the axion run's pilot branch:

```yaml
acquisition:
  audit: {chain: true, tau: 5, posterior: 150}   # chain search, threshold 5, posterior pilot
  boost:
    n: 3000                  # in the axion run 10 000 per boost did better than 4000 for the same
                             # number of true calls
    start_iteration: 30
    boosts: 10
    density_filter: true
    pool_factor: 20
    background: quadratic
    region: 1.5
    region_quantile: 0.999   # the region's reach by a quantile, not the farthest point
    train_domain: 1.25       # train only near the posterior from the first boost on
    mixing: 0.1              # modes decided by measured mixing
    min_share: 0.005         # small groups of walkers reach the mixing test
    mode_depth: auto         # T * ndim / 2: leave out "modes" far below the posterior's
```

Both keep the last weights of a boost-phase training. The setting that decides between
them is the training domain ([its section](boost_background_region.md#the-training-domain-and-the-quantile-reach-17-october-2026-opt-in)).
With a near-Gaussian likelihood the quadratic background is close to exact, and the far points
of the tempered training set fix the network's curvature over a long lever arm; the domain
leaves them out (31% of the 49-parameter run's points) and the network's tails come out too
high. With an oscillating or multimodal likelihood the same far points set an output scale of
tens of thousands of log-units against features of a few, and without the domain the network
does not learn the features at all.

The evidence: each reference run was branched at its hand-over with the second set, everything
else unchanged (`hpc/analysis/branch_run.py --set`), and the error of each network at
its own boost points compared, network by network (S is the hand-over):

| run (parameters) | reference: S+1 / S+3 / S+6 / last | non-Gaussian settings: S+1 / S+3 / S+6 / last |
|---|---|---|
| 49-parameter (49) | 0.79 / 0.34 / 0.27 / 0.21 | 2.40 / 0.81 / 0.63 / 0.52 |
| HiLLiK (82) | 0.47 / 0.29 / 0.24 / 0.22 | 2.56 / 1.63 / 1.44 / 1.44 |
| SINU (16, two modes) | 0.084 / 0.064 / 0.072 / 0.044 | 0.059 / 0.046 / 0.039 / 0.040 |
| ridge (31, analytic) | 0.059 / 0.037 / 0.021 / 0.017 | 0.144 / 0.037 / 0.023 / 0.018 |

(last: S+9, the last boost of a run, except HiLLiK's S+8, 18 October 2026.)

In the 49-parameter branch the error of its first network grew with the depth below the peak
(+7 on average 30 to 45 log-units down, where the reference's stayed near zero), and the branch
stayed 2.2 to 2.5 times the reference from S+3 on. HiLLiK's first branch network needed four
audit rounds for false maxima up to +39 far from any training point and the posterior pilot's
first fine-tune (mean excess +2.25); from S+5 on its error stayed at 1.4 to 1.5, six times the
reference's, with clean audits (highest maximum +0.3): a network less accurate everywhere near
the posterior, not one misled by false maxima. SINU's branch was better than its reference at
every network, the ridge's equal after one network. In the axion run (24
parameters, oscillations of a few log-units) the reference settings never learned the
oscillation: the networks' profile correlated with the truth's at +0.35 at best and the main
peak held 1.6 times an even share of the posterior; with the second set (10 000 points per
boost) the final error was 0.50, the correlation +0.88, and the main peak's mass within 6% of
the truth's by reweighting the boost points.

When in doubt, the first set: the reference runs of the table above all used it. The parts of the second
set other than the training domain and the narrower region (the audit's chain search and
posterior pilot, mixing, `min_share`, `mode_depth`, the quantile reach) did no harm in the
near-Gaussian branches, but they were changed together there and have not been tested one by one.

### What the training domain changes: the output scale (18 October 2026)

The domain changes three things at the first boost training: the points trained on, the points
the input normalisation is adapted to, and the output scale (the mean and standard deviation of
the residual targets, network minus quadratic background). The far design points dominate the
last: on the 49-parameter run the residual's standard deviation is 4651 log-units with every
point and 47 inside the domain. Each run's first boost network was retrained from
random weights, on the same training set, with these chosen separately (no true call). Errors
are network minus truth, rms: near the posterior at the reference run's last boost points (the
axion run: its pilot's boost 17), and on the training's own held-out tenth within 50 log-units
of the best (the same rows for every variant: they all lie inside the domain).

| run (parameters) | no domain | domain, its own scale | domain, all-points scale | domain, all-points normalisation and scale |
|---|---|---|---|---|
| SINU (16) | | **0.058** (0.14) | 0.080 (0.19) | |
| axion (24) | 1.22 (1.49) | **0.37** (0.62) | 1.08 (1.27) | |
| ridge (31) | | 0.132 (0.18) | **0.083** (0.09) | |
| 49-parameter (49) | **0.62** (0.43) | 1.09 (0.81) | 0.70 (0.50) | 0.66 (0.46) |
| HiLLiK (82) | **0.49** (0.40) | 1.83 (1.78) | 1.27 (0.98) | 1.19 (0.99) |

(Near the posterior; in brackets near the best on the held-out rows. One seed each; the "domain,
its own scale" variant reproduces the confirmation branches' own first networks, 0.057 against
0.059 on SINU, 0.146 against 0.144 on the ridge.)

The small output scale lets a network fit its training points much more closely (the axion
run's: 0.08 in-sample, 0.62 held out). That is what an oscillation of a few log-units needs, and
it is overfitting where the likelihood has no such structure and the points are sparse: the
harm grows with the number of parameters. On the 49-parameter run the output scale carries most
of the effect (the far points' information is not needed: 0.66 against 0.62); on HiLLiK, in 82
parameters, the far points' information matters as well (1.19 against 0.49).

In every row the variant with the lowest held-out error near the best is the best near the
posterior. A boost setting that trains the candidates at the first boost training and keeps
that one would choose between the two recommended settings from the run's own data. It is not
built: five likelihoods, one seed each.

### A region the networks miss: the axion run's low-omega corner (18 October 2026)

A long MCMC of a student's on the same likelihood (Planck TT, 24 parameters, 30 000 samples) puts
4.5% of the posterior at omega < 3 and 1.8% at omega < 2 and A_osc > 0.015, as high as the bulk
(about 10 log-units below its best), with n_s 3.3 sigma above the bulk's, omega_cdm 1.2 sigma and
ln A_s 0.6 sigma: at omega 1 to 2 the oscillation spans about one period of the observed range of
k and trades off against the tilt. Our likelihood agrees with his values there to 0.02 (105
points). The networks of the axion run's pilot branch,
minus the truth at his points (medians):

| his points | share | network 8 (hand-over) | 9 | 10 | 13 | 18 (last) |
|---|---|---|---|---|---|---|
| omega >= 5 | 94% | -6 | -1 | -1 | 0 | 0 |
| omega 2 to 3 | 0.9% | -14 | -4 | -5 | -4 | -4 |
| omega < 2, A_osc < 0.015 | 1.8% | -26 | -19 | -24 | -21 | -20 |
| omega < 2, A_osc 0.015 to 0.025 | 0.9% | -121 | -127 | -156 | -133 | -127 |
| omega < 2, A_osc > 0.025 | 0.8% | -235 | -258 | -281 | -233 | -219 |

Network 18's posterior has 0.5% at omega < 3. About 3.5% of the true posterior is therefore
missing from the run's results, and its evidence is low by about 0.04 in ln Z (its quoted error
is 0.16). Nothing that takes its cue from the network could have found it: at the hand-over the
training set had one point there within 30 log-units of the best; the boosts draw from the
network's posterior; network 18's region leaves out 83% of his corner points; the audit looks at
the network's maxima and the true check where the network puts mass; and the true (A_osc, omega)
profile, whose nodes start from the network's optima, ran out of calls there at delta chi2 34 to
356 against about 12 in his chain. A hotter boost does not reach it either: sampled at T = 3,
network 9 put 1 of its 256 000 chain rows in the corner (the network's deficit of 120 to 280 is
still a factor of e^-40 or less). An annealed start of the boosts (T = 3, 2, 1) helps where a
network is off by a few times T, not here.

True evaluations placed without the network did not bring it back either.
- **The profile.** A 2D profile continued along the amplitude from zero, with the profile setting
  `continuation` ([Continued rows](create_profiles_2d.md#continued-rows-continuation)), found the
  ridge on some of its rows.
- **Its points in the full-box run.** Added to that run's training set (`client.py augment`),
  they were learned themselves. The ridge's volume around them was not: the networks stayed 70 to
  160 too low there.

What found the corner is domain splitting ([combine.md](combine.md)). A run on ω in [1, 3] alone
found the ridge with its ordinary tempered loop, without any added points, and after one boost
matched the long MCMC there to 0.15 to 0.18 rms. Combined with the full-box run's network, it
reproduced the MCMC's posterior mass below ω = 3 to about 0.2 percentage points.

## Warm start is the default; random weights are an option (31 October 2026)

In the boosts each network starts from the previous one (`training.warm_start: true`, the
default) and keeps its last weights. This stays the
default. With `training.warm_start: false` each network starts from random weights and keeps the
weights with the lowest validation loss. The quadratic
background and the training domain fitted at the first training with boost points are kept, and
the region is built as usual. Training takes about 1.5 times as long (52 against 35 minutes at
120 000 points).

A run can change to it at any network N, since the setting is not part of the run's identity: edit
the copy of the settings in the run's folder and continue with `-s N --rewind --retrain`, or do
the same in a branch of the run. The networks before N stay as they are.

Random weights are for a likelihood on which warm-started networks stop improving. Four
comparisons on real likelihoods, all with branches from the same network. *Boost points*: the
scatter of network minus truth at the boost points of the last two networks with such points, which
they had not seen. *Posterior*: the scatter at 500 posterior samples of the final network.

| likelihood | branch from network, final network | boost points, warm | boost points, random | posterior, warm | posterior, random |
|---|---|---|---|---|---|
| Axion monodromy, H0WC, omega in [5.5, 60] | 8, 21 | 1.25, 1.29 | 0.14, 0.13 | 1.21 | 0.10 |
| The same, omega in [1, 6.1] | 8, 16 | 0.14, 0.13 | 0.14, 0.14 | 0.08 | 0.12 |
| Axion monodromy, PDP+, omega in [2.5, 60] | 8, 16 | 0.21, 0.22 | 0.39, 0.35 | 0.18 | 0.36 |
| 49 parameters, near-Gaussian | 14, 22 | 0.22, 0.21 | 0.28, 1.2 | 0.21 | 0.30 |

Only the first likelihood gains, and there by a factor of ten. On the other three, random weights
end 1.4 to 2 times less accurate at posterior samples, and they vary more from one network to the
next: on the last likelihood network 21 scatters by 1.33 at posterior samples, is 1.59 too high on average
and leaves 119 effective points of 500, between 0.28 and 0.30 at its neighbours, so that a run stopped there
would have been poor. On those three the evidences agree within 0.2 (one BEST run each).

There is no automatic switch, since no signal measured so far separates the first case from the
others. A warm-started training whose validation loss is lowest in its first few epochs occurs in
the first likelihood, but also in the third, where warm starts work. The audits and the posterior
pilot were clean on the plateau at 1.3 in the first. What identifies the case is the scatter at
the boost points levelling off well above that of comparable runs; try a branch with random
weights from a few networks back and compare.

## Optional posterior density filter

Set `density_filter: true` and optionally `pool_factor: 20` in the boost block to
filter against a balanced T=1 mode mixture using all existing genuine data. The
accepted split may differ between modes; `n` is an accepted-point request and
underfilled batches are not topped up. Filtered reports omit the unweighted true
mode-weight correction because selected rows are not posterior draws. The default
is unfiltered. See [configuration, results and limitations](posterior_density_boost.md).

Keep audit enabled. The native loop can continue after an uncleared audit;
the analytic Gaussian study was stopped for precisely this reason. Monitor the
saved audit status before further acquisition; the loop does not gate on it.
The algorithm and weight-correction discussion below describe the default
unfiltered path.

## Optional quadratic background and region

`background: quadratic` and `region: 2.0` change what a boost run trains and where its chains
and audit look: a fixed quadratic under the network, one cold start, and the region the boost
points reach. Both are off by default. See
[boost_background_region.md](boost_background_region.md) for the layout, the measurements and
what is not established. That document's measurements are on analytic likelihoods; the first
runs with both keys on a real one, ΛCDM on Planck + DESI, are in
[boost_real_lcdm.md](boost_real_lcdm.md).

## Optional: the modes decided by measured mixing

`mixing: 0.1` replaces the fixed `barrier` of step 1 below by a test: maxima are grouped into
candidate basins with a low barrier, a free T = 1 chain of the boost's own length is run, and
two basins are one mode when that chain has forgotten which of them its walkers started in.
How high a barrier a chain crosses depends on the number of parameters (6 log-units with 16,
no longer 4 with 58, measured), and the straight path between two maxima of a curved mode can
dip far more than the chain has to. Off by default. `python client.py modes RUN_DIR` runs the
same analysis on any saved network. See [boost_modes_mixing.md](boost_modes_mixing.md).

## Why

The loop draws its batches from the network at the acquisition temperature T. In n dimensions
such a batch lies about T n / 2 log-units under the peak; the posterior lies n / 2 under it.
At T = 7 and 16 parameters a batch of 2000 puts about 7 points where the posterior is, and the
injected points are drawn at T as well. On the final SINU run (Planck + DESI DR2, 34 272
training points at iteration 15) 226 points lay within 15 log-units of the peak; 153 of them
came from the profile stage and lie on one curve. The loss scale stayed at c = 19.2 log-units,
and the final validation loss corresponds to an error of about 0.6 log-units in the core,
which is what is measured there.

The network's level in the core of a mode is therefore an extrapolation from the shell around
it. For one mode a constant offset changes nothing. For modes that a chain cannot travel
between, it sets their weights.

Write L for the true likelihood, f for the network's log-likelihood, R_k for the region of
mode k, and D = log L - f. The evidence of mode k is

    Z_k = integral over R_k of L,        Zhat_k = integral over R_k of exp(f)    (the network's),

and, with q_k the network's posterior restricted to R_k (density exp(f) / Zhat_k),

    Z_k = Zhat_k <exp(D)>_{q_k}.                                                  (1)

The weight of mode k is Z_k / sum_j Z_j. So the odds of two modes under the network are wrong
by the factor exp(delta), delta being the difference between the two modes of
log <exp(D)>: a difference of the network's error between the two cores. On six good SINU
networks |delta| was 0.03 to 0.81:

| network | strong-mode weight, network | bias between modes, delta | weight by (1) |
|---|---|---|---|
| final run, iteration 15 | 12.7% | +0.81 | 24.7% |
| long training on every iteration, iteration 10 | 38.1% | -0.59 | 25.5% |
| iteration 15 trained once more on the same data | 35.7% | -0.60 | 23.3% |
| the same plus 586 posterior points | 20.3% | +0.31 | 25.8% |
| the same plus 2903 posterior points | 23.2% | -0.03 | 22.8% |
| the same plus 2903, loss scale c = 8 | 23.9% | +0.16 | 26.8% |

Three things follow from the table. The networks disagree by a factor of three and one more
training on identical data moved the weight from 12.7% to 35.7%: the existing data do not pin
the weights down. Identity (1), with 300 true values per mode, brings all six to 24.8% with a
spread of 1.5 points. And true values at posterior samples, added to the training set, remove
the bias itself. The boost does both of the last two in one step.

## What a boost iteration does

1. **Modes** (`sampling/modes.py`; no true calls). An ensemble started on the iteration's
   tempered chain is cooled to T = 1 through a ladder of temperatures. Every walker then climbs
   to its local maximum (projected gradient ascent in box units). Two maxima belong to the same
   mode when the straight path between them never dips more than `barrier` log-units (default
   4, a Delta chi^2 of 8) below the lower of the two; the connected groups are the modes, and a
   walker belongs to the mode of its maximum. Groups with less than `min_share` of the walkers
   are left out, as is a climb that ended more than `barrier` under a maximum it is joined to
   (it stalled on a slope or a saddle). Both are counted in the log.

   The maxima are compared, not the walkers. A T = 1 walker sits about n / 2 log-units under its
   peak, so the path between two walkers of different modes need not dip below them at all; and
   a walker stranded on the barrier is joined to both sides.
2. **Samples.** Each mode gets its own T = 1 chain, started on its walkers and confined to its
   region R_k: the points for which the mode's Gaussian, fitted to its T = 1 walkers, is the
   nearest (Mahalanobis distance plus log-determinant). Between separated modes that boundary
   runs through the barrier, where there is no posterior to speak of.
3. **Points.** `n` divided equally among the modes (the first modes get the remainder), drawn
   at random from those chains, are evaluated with the true likelihood and join the training
   set as a batch does. A boost iteration has no tempered batch and no injection: its true
   calls are the `n` asked for. The split is equal whatever the weights, so that each
   <exp(D)> in (1) is known equally well, and because the mode the network under-rates is the
   one a proportional split would starve.
4. **Weights.** Zhat_k comes from bridge sampling (Meng and Wong) between the mode's chain and
   a Gaussian mixture g fitted to half of it, widened by 1.3 in variance and cut to R_k: with
   l = exp(f) / g, N1 samples x of the chain and N2 draws y of g, s1 = N1 / (N1 + N2) and
   s2 = N2 / (N1 + N2), it is the fixed point of

       Zhat = [ (1/N2) sum_y l(y) / (s1 l(y) + s2 Zhat) ] / [ (1/N1) sum_x 1 / (s1 l(x) + s2 Zhat) ].

   The log reports the weights Zhat_k / sum Zhat as the network has them, the spread of
   log Zhat_k over three random halves, and the efficiency of g as an importance proposal.
   Then (1) with the boost's true values gives the corrected weights, and the effective number
   of points behind each mean, (sum w)^2 / sum w^2 with w = exp(D).

   A point the likelihood refused, one that came back as −inf or with the run's refused
   value, is a point where the true posterior is zero: it has weight zero and stays in the
   denominator of its mode's mean, as in the true check (`hpc/analysis/true_check.py`), and the report counts
   it (`zero_density`, `refused`; the log says how many). A point without any value (NaN,
   +inf) is another matter: what its mode's mean would have been is not known, and one such
   point withholds the corrected weights of every mode: the report has
   `evidence_correction_valid: false`, `correction_withheld_reason` and the count (`failed`),
   and the log says so. Refused points are not dropped: dropping them and normalising the rest
   over-rates a mode with refused points. The pooled error of the boost is reported either
   way, with the points left out counted.

With one mode, steps 1 and 2 are a T = 1 chain and the boost is `n` posterior samples.

### A network without a mode

Step 1 can end with no mode: no group of cooled walkers reaches `min_share`. This is what a
network far from converged looks like, with a surface so rough that nearly every walker
climbs to a maximum of its own; it was met on a real likelihood whose boosts began while the
network was still some 15 log-units off at its tempered batch, two iterations after the
first boost. Such a network has no posterior to draw from. The iteration then acquires as an
ordinary one does (the tempered batch of `acquisition.n_append` points, and the injected points
if configured), which is what such a network needs, and says so:

```
Boost: 3000 posterior samples of the network (T = 1) in place of the tempered batch
   no group of the 512 cooled walkers reaches min_share (0.02): no mode found. The network is not ready for a boost: this iteration acquires as an ordinary one does instead
Selecting 3000 new samples from surrogate chain
```

The next iteration tries a boost again, with the network trained on those points.
`RUN/boost/fallback_it_N.json` records the iteration; it has no `boost_it_N` files, and its
error is reported as that of a tempered batch. In the boost phase a tempered chain runs
inside the region when one is configured, so the batch of such an iteration comes from
there. Any other error in a boost still stops the run.

A boost that falls back is a sign that the boosts began too early: read the error at the
tempered batch before `start_iteration` (README, convergence section) and the error at the
first boosts.

The next network is trained on the combined set in the ordinary way (warm start, ordinary
schedule, the audit if configured).

The weights are not taken from the tempered chain. Its split between the modes, reweighted or
cooled to T = 1, was wrong by about 0.2 in log-odds per network (six estimates of one number
spread by 0.21, although each walker crossed the barrier some fifty times); the per-mode
evidences bring that spread to 0.06, and three independent evaluations of one network gave
13.2%, 13.5% and 13.9%.

## Reading the report

```
Boost: 1000 posterior samples of the network (T = 1) in place of the tempered batch
   cooled 512 walkers from T = 7 through 4.3, 2.65, 1.63, 1 and climbed: 2 modes (barrier 4 log-units); 4 walkers in smaller groups left out
   mode 1: weight 86.5% (log evidence -5340.55 +- 0.06, proposal efficiency 0.44); 397 walkers; highest value -5293.81
   mode 2: weight 13.5% (log evidence -5342.41 +- 0.03, proposal efficiency 0.53); 111 walkers; highest value -5294.53
   1000 boost points: 500, 500 per mode
   mode 1: 500 true values; network minus truth +1.39 (sd 0.48), 406 effective points; weight 86.5% -> 74.6% with the true values
   mode 2: 500 true values; network minus truth +0.79 (sd 0.74), 235 effective points; weight 13.5% -> 25.4% with the true values
```

- **Effective points.** If D is roughly Gaussian with spread s, the relative error of
  <exp(D)> from N points is sqrt((exp(s^2) - 1) / N): 3% for s = 0.5 and N = 300, and growing
  fast beyond s of about 1. When the network is off by more than about a log-unit on its own
  samples a few points carry the average, the corrected weight is not to be trusted, and it
  tends to come out low for that mode. Three iteration-10 SINU networks, with spreads of 1.3 to 4.7, had
  2 to 63 effective points of 300 and corrected weights from 0.6% to 12%. The points are still
  training data, but the boost came early.
- **Proposal efficiency** of the mixture: near zero means the mode is far from a mixture of
  `components` Gaussians and its evidence is uncertain; so does a large spread of the log
  evidence.
- **Walkers** per mode are the tempered chain's split, not the weights.
- **Network minus truth** is the plain mean; the correction uses the mean of exp(D), which a
  larger spread raises.

### The error at the boost points, and when to stop

A boost's points are evaluated with the true likelihood, and the network that drew them had
not been trained on them. After the lines above every boost prints the network against the
truth over all its points, and from the second boost of a run on, the scatter of every boost
so far:

```
   network 4 at its boost points, which it had not seen: network minus truth rms 3.01 (mean +2.14, scatter 2.12, largest 9.76) at 300 points
   scatter at the boost points, boost by boost: 2.17, 23.9, 2.12
```

(the format, from a 16-parameter analytic test whose boosts were started after two
iterations on purpose; the middle value is the network that `background: quadratic` trains
from random weights at the first training with boost points). The same numbers are in `error` of
`RUN/boost/boost_it_N.json` and, as `held_out_*`, in the acquisition entries of the run's
`metrics.json`; `dataset.boost.boost_errors(RUN / "boost")` returns the scatter of every
boost of a run.

This is what there is to decide on whether another boost is worth its true calls. No
boost stops on it: the number of boosts is the `-i` of the run, its `max_iterations`, or
`acquisition.boost.boosts`. The convergence criterion of the ordinary loop
([tempered_convergence.md](tempered_convergence.md)) decides when the boosts begin and is
never applied to a boost.

- **Scatter, not rms.** A constant offset between network and truth changes no posterior of
  a single mode; the rms contains it and the scatter does not. With several modes the offset
  matters between them, and the per-mode lines and corrected weights above are what to read.
- **What a scatter s costs.** For a roughly Gaussian D the network's posterior and the true
  one differ by a Kullback-Leibler divergence of about s^2 / 2, and reweighting N samples of
  the network to the truth leaves about N exp(-s^2) effective ones: 78% for s = 0.5, 37% for
  s = 1, 2% for s = 2. The largest error says whether a few points carry more than that.
- **When it stops falling.** The scatter of boost N belongs to network N, which was trained
  with the boosts before it. A sequence that has levelled off says that further boosts of
  this size no longer change the network where its posterior is.
- **With the density filter** the points are the posterior samples the training set was short
  of, not a posterior draw, so the per-mode correction is withheld (above) while this error is
  still reported: it is the network's error at the points chosen, which lean towards where the
  training set is thin. For the error at plain posterior samples evaluate the truth at a draw
  of a finished network's chain (`hpc/analysis/true_check.py` does this on a cluster).
- A point the likelihood could not evaluate is left out and counted (`... at 299 points, 1
  left out`), also when a launcher gave it a finite value to train on.

## Settings

| key | default | meaning |
|---|---|---|
| `n` | required | true evaluations per boost, shared equally among the modes |
| `start_iteration` | required | first iteration whose acquisition is a boost; with `convergence.metric: tempered_batch_scatter` and no `-i`, the latest one: the boosts begin when the ordinary loop has converged ([tempered_convergence.md](tempered_convergence.md)) |
| `boosts` | null | opt-in: the run ends after this many boosts (ignored when `-i` gives its length) |
| `barrier` | 4.0 | log-units; maxima joined by a straight path that dips less are one mode |
| `min_share` | 0.02 | a mode needs this share of the cooled walkers; with none, the iteration is an ordinary one |
| `walkers` | 512 | ensemble for the cooling and for each mode's chain |
| `cool_steps` | 2000 | steps at each temperature on the way down |
| `steps` | 10000 | steps of each mode's T = 1 chain, after a fifth more as burn-in |
| `components` | 6 | Gaussians in the mixture each mode's evidence is bridged to |
| `mixing` | null | opt-in, 0.1 tested: decide the modes by a free T = 1 chain; basins whose memory after `steps` is at most this are one mode ([boost_modes_mixing.md](boost_modes_mixing.md)) |
| `candidate_barrier` | 1.0 | with `mixing`: log-units; maxima joined by a path that dips less are one basin without a test |
| `mode_depth` | null | opt-in, log-units or `auto` (T * ndim / 2, T = `sampling.temperature`): a mode whose highest maximum lies more than this below the highest mode's is left out before its chain is run (below) |
| `balance_floor` | null | opt-in, a weight in (0, 1]: a mode lighter than this gets its equal share times its weight over the floor; heavier modes share equally, as without it (below) |

`mode_depth`. Every mode gets the same number of points whatever its weight,
so a "mode" far down in the tail costs as much as the posterior. The first boost of a branch of
the axion run (24 parameters; mixing with `min_share` 0.005, before any region) found one 1030
log-units below the posterior's peak, with weight 0, in a corner where the likelihood refuses
every point: 1978 of its 4000 points were refused, and the network after it trained on half
the usual number. A mode that deep has no posterior weight to measure; T * ndim / 2 of the
ordinary loop (84 there) is a natural depth, and `mode_depth: auto` sets it. The info of the mode search lists what was left
out (`deep_modes`: walkers, highest maximum, depth); without the key it has no such entry.

`balance_floor`. With `min_share` 0.005, as in the setting for non-Gaussian
likelihoods, a group of three walkers can become a mode, and its equal share is then as large as
the posterior's. The posterior pilot cannot check such a mode: a 1% mode has one or two of its 150
walkers. In the H0WC axion run (22 parameters) network 8 found two modes of 1.3% and 1.1% (6 and 7
of 512 walkers) beside one of 97.6%; the density filter gave them 4164 and 4376 of the 10 000
points against 1460 for the posterior's mode, the network was 2.0 and 13.4 too high on average at
their points, and network 9, trained on them, was off by 3.5 at its own boost points (spread of
network minus truth) against 1.5 for network 7, before network 10 came back to 2.1. Over the 77 boosts with several modes in the cluster runs of October 2026, the modes
below 5% of the weight (0.8% together, in the median) received a median of 55% of the points.
`balance_floor: f` gives mode k the share min(w_k / f, 1) of an equal one, w_k its weight from
the evidences, scaled so that the largest is 1: modes at or above the floor share equally (the
rule above for modes whose weight the network may get wrong), a lighter one in proportion to its
weight. With 0.1 the two H0WC modes would have had 13% and 11% of an equal share; SINU's strong
mode (10.6% in its least favourable network) and the H0WC low-frequency run's second mode (9.5%
at its first boost, then about 30%) keep, or nearly keep, theirs. In the filtered boost the clouds of `select_posterior_boost` are cut to their
shares (the target is the mixture weighted by them); in the unfiltered one the counts are. Each
mode keeps at least one point. Without the key every mode has the equal share (`tests/test_boost.py`,
`tests/test_posterior_boost.py`), and the record has no
`share`; with it each mode's entry in `boost_it_N.json` has its `share`, and the log a line when a
share is below one.

Files: `RUN/boost/boost_it_N.csv` (the parameters, `loglkl`, the network's value `predicted`,
`mode`) and `boost_it_N.json` (the report, with each mode's peak, mean and spread). The points
are part of `data_it_<N+1>.csv`, so a continued run needs nothing else.

A boost costs the master a cooling, a climb and one chain per mode (three minutes at 16
parameters with two modes), during which the MPI workers sleep, then `n` true evaluations.

## Test: SINU, Planck + DESI DR2

The finished run of 4 October (16 parameters, audit, injection, profile points; networks 0 to
15) was continued on a copy with `boost: {n: 1000, start_iteration: 15}` for two iterations.
The modes were found without naming a parameter: two, with the boundary at
log10 G_eff of about -2.67, and 500 points from each. No evaluation failed, and both audits after the
boosted trainings found no maximum above the threshold.

| | network 15 | network 16 (one boost) | network 17 (two boosts) |
|---|---|---|---|
| strong-mode weight, network | 13.5% | 20.7% | 23.4% |
| corrected by (1) with that boost's true values | 25.4% | 25.1% | |
| network minus truth on its own samples, weak / strong mode | +1.39 / +0.79 | +0.23 / +0.11 | |
| spread of that error, weak / strong | 0.48 / 0.74 | 0.36 / 0.62 | |
| effective points of 500, weak / strong | 406 / 235 | 442 / 320 | |
| posterior drift against the previous network | 0.116 | 0.048 | 0.043 |
| training points within 15 log-units of the peak, weak / strong | 141 / 85 | 634 / 573 | 1125 / 1065 |
| weight at log10 G_eff in [-5, -4.6] over that in [-4.2, -3.8] | 2.2 | 1.5 | 1.3 |

- The corrected weight is the same from both boosts, 25.4% and 25.1%, and agrees with the
  24.8% (spread 1.5 points) of the six networks above.
- The network's own weight moves to it: 13.5%, 20.7%, 23.4%.
- One boost removed most of the network's offset in the cores, from 1.4 and 0.8 log-units to
  0.2 and 0.1.
- The true profile is flat below log10 G_eff = -3.8, so the last row should be near 1.
- Network 16 had a small island on the barrier at the 99% level (0.3% of the weight between
  -2.8 and -2.0); it is gone in network 17 (0.1%). The boost puts no points between the modes.
- Each boost iteration took 40 to 45 minutes on a laptop with six ranks: 1000 true calls, 13
  minutes of training, and the loop's two tempered chains.

So a run can use the boost in two ways: take the corrected weights from one boost, or boost
two or three times until the network's own weights agree with them.

## What is not established

- The test above is one likelihood (SINU, 16 parameters, two modes), one seed, two boosts.
  The single-mode case has been run on a real likelihood, with the density filter, the
  background and the region: ΛCDM on Planck + DESI, five boosts, 0.06 log-units from the truth
  at posterior samples at the end ([boost_real_lcdm.md](boost_real_lcdm.md)).
- The weak mode is a plateau that runs to the lower bound of the box, so its weight depends
  on that bound. That is a property of the posterior, not of the method.
- The barrier test follows a straight line between maxima. Two regions joined only by a
  curved low path with no maximum along it would be reported as two modes. They would each get
  points, which does no harm, but their separate weights would then mean little. The opt-in
  `mixing` decides by a chain instead ([boost_modes_mixing.md](boost_modes_mixing.md)).
- The fixed `barrier` of 4 is not the same statement at every number of parameters: with 58 a
  chain of the default length no longer mixes across it (same document).
- A mode the network gives no weight to is not found: coverage still comes from the tempered
  acquisition, the audit and the profile stage. The axion run's low-omega corner is such a
  region, missed by every network of its run ([the section above](#a-region-the-networks-miss-the-axion-runs-low-omega-corner-18-october-2026)).
  Domain splitting ([combine.md](combine.md)) recovers such a region when the user knows where it is.
- The Gaussian mixture is a poor proposal for a mode far from Gaussian; the efficiency and the
  spread in the log say when. At many more than 16 parameters this has not been tried.
- The weights of a run's final posterior are not reported by the loop after its last
  iteration: `python client.py modes RUN_DIR` gives them for any saved network
  ([boost_modes_mixing.md](boost_modes_mixing.md)).
- When to boost is the user's choice (`start_iteration`). Nothing in the loop decides that the
  network is ready; the effective points say afterwards whether it was.
