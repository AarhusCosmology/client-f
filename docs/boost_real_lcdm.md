# The loop with boosts on a real likelihood: ΛCDM, Planck + DESI

The first runs of the whole pipeline on a real likelihood: ten iterations of the ordinary
loop (tempered batches, the optimum audit, injected points), then five boosts with the
density filter, the quadratic background and the region
([boost.md](boost.md), [posterior_density_boost.md](posterior_density_boost.md),
[boost_background_region.md](boost_background_region.md)). Two runs on 7 October 2026, each on
one 48-core cluster node, identical apart from the injection's window
([injection_window.md](injection_window.md)).

What they show:

- **Both final networks reproduce the true likelihood to 0.06 log-units rms** at 500
  independent samples of their own posterior (largest error 0.2). Reweighting those samples
  by the true likelihood leaves 498 effective points of 500 and moves no mean by more than
  0.01 σ and no width by more than 2%.
- **The two runs give the same posterior**: no mean differs by more than 0.013 σ and no width
  by more than 0.8%, below the noise of one chain.
- **The ordinary loop alone had not got there.** After ten tempered iterations and 23 400
  points the error at posterior samples was 1.0 and 0.7 log-units, and the foreground
  nuisance posteriors were 14 to 32% too narrow. The loop's drift metric had stopped
  falling: between 0.1 and 0.8 from network 5 on, and 0.02 after the boosts.
- **One boost did most of it.** After the first boost the error at posterior points was 0.25
  and 0.14, after the second 0.13 and 0.12.
- **No false maximum from network 4 (3) on**: every later audit, the boosts' included, was
  clear without a true call.
- **The restricted injection made no difference to the final network.** Before the boosts
  that run was the better of the two; one seed each.
- A run took 5.0 hours: 33 100 evaluations in batches (0.9 h on 48 cores) and 303 or 214
  audit calls.

## The runs

| | |
|---|---|
| likelihood | Planck 2018 low-ℓ TT and EE, NPIPE CamSpec TTTEEE, DESI DR2 BAO (all tracers), through Cobaya 3.6.2 |
| theory | CLASS++ at commit 7cc41a51, massless neutrinos (N_eff = 3.045) |
| parameters | 15: ω_b, ω_cdm, H0, ln(10¹⁰A_s), n_s, τ and nine CamSpec nuisances |
| training box | `hpc/runs/lcdm_planck_desi/domain_bounds.json`, applied by the launcher |
| design | 2000 Latin-hypercube points + 100 injected |
| networks 0 to 10 | per iteration 2000 points at T = 7 + 100 injected; the audit at its defaults |
| networks 11 to 15 | five boosts of 2000 posterior points: `density_filter: true`, `background: quadratic`, `region: 2.0`; modes by the fixed barrier |
| network, training | 5 × 512, Alsing; msre with the ESS floor, plateau learning rate, warm starts |
| sampling | AIES, 256 walkers, 100 000 steps |
| seed | 42 in both, so the 2000 design points are the same |

The two runs: **every point** (the injection's quadratic fitted to every
training point, which is no longer a setting) and **window 16** (only the points near the
best, as CLiENT does now). Both use
`hpc/runs/lcdm_planck_desi/cobaya.yaml`; the settings of the second are
`hpc/runs/lcdm_planck_desi/client.yaml`.

Five of the 2000 design points had no finite likelihood; no later point failed.

## How the accuracy was measured

- **Held-out errors.** Network N chose the points of iteration N's acquisition and had not
  seen their true values. Network N at those points against the truth is a test where its
  own tempered posterior is (networks 0 to 9) or where its posterior is (the boost points of
  networks 10 to 14, which are selected by the density filter and so are not plain posterior
  draws).
- **The true check.** 500 samples drawn at random from a T = 1 chain of a network, with the
  true likelihood at each: network 10, the last one trained on tempered points only, and
  network 15, the last. With D = truth minus network: its rms, the effective number of
  points (Σw)² / Σw² for w = exp(D), and every parameter's mean and width with the weights w
  against the network's own.
- **Posteriors.** Every network sampled at T = 1 and compared with the last network of its
  run, and the two last networks with each other. Agreement between networks is
  convergence, not accuracy.

Errors are in log-likelihood units throughout. Pairs of numbers are every point / window 16.

## The final networks against the truth

| run | network | rms | mean | largest | effective points of 500 | largest corrected mean shift | largest corrected width change |
|---|---|---|---|---|---|---|---|
| every point | 10 | 1.03 | +0.82 | 5.0 | 79 | 0.66 σ (amp_217) | 67% (amp_217) |
| every point | 15 | 0.06 | −0.01 | 0.2 | 498 | 0.005 σ | 0.7% |
| window 16 | 10 | 0.69 | −0.18 | 3.1 | 220 | 0.22 σ (calTE) | 35% (amp_217) |
| window 16 | 15 | 0.06 | +0.01 | 0.2 | 498 | 0.007 σ | 1.7% (τ) |

The samples lie a median 6.6 log-units below the best training value (n / 2 = 7.5): this is
a test of the bulk of the posterior. The corrected numbers of network 10 rest on 79 and 220
effective points and are rough.

The two final posteriors, mean ± standard deviation:

| parameter | every point | window 16 | difference of the means (σ) | of the widths |
|---|---|---|---|---|
| omega_b | 0.0223105 ± 0.00012 | 0.0223107 ± 0.00012 | −0.002 | +0.4% |
| omega_cdm | 0.118088 ± 0.000636 | 0.118087 ± 0.000637 | +0.002 | 0.0% |
| H0 | 68.5187 ± 0.287 | 68.5198 ± 0.286 | −0.004 | +0.4% |
| logA | 3.03512 ± 0.0161 | 3.0351 ± 0.0162 | +0.002 | −0.3% |
| n_s | 0.967561 ± 0.00339 | 0.967585 ± 0.00338 | −0.007 | +0.5% |
| tau_reio | 0.0530716 ± 0.00768 | 0.0530692 ± 0.00768 | 0.000 | 0.0% |
| A_planck | 1.00044 ± 0.00248 | 1.00043 ± 0.00249 | +0.004 | −0.6% |
| amp_143 | 17.4549 ± 2.05 | 17.4312 ± 2.04 | +0.012 | +0.7% |
| amp_217 | 11.3258 ± 1.96 | 11.3102 ± 1.95 | +0.008 | +0.8% |
| amp_143x217 | 8.2222 ± 1.97 | 8.19872 ± 1.96 | +0.012 | +0.8% |
| n_143 | 1.02082 ± 0.185 | 1.02331 ± 0.185 | −0.013 | +0.1% |
| n_217 | 1.59917 ± 0.415 | 1.6024 ± 0.414 | −0.008 | +0.2% |
| n_143x217 | 1.83472 ± 0.598 | 1.84026 ± 0.594 | −0.009 | +0.6% |
| calTE | 0.99747 ± 0.00365 | 0.997495 ± 0.00367 | −0.007 | −0.4% |
| calEE | 0.998845 ± 0.00376 | 0.99885 ± 0.00377 | −0.001 | −0.3% |

The largest credible-metric distance between the two is 0.015 (68%) and 0.019 (95%); two
halves of one chain differ by 0.03 and 0.04 to 0.06. The best true values found are −5493.45
and −5493.43, and both runs end with a log evidence of −5540.13 from their last boost.

## The ordinary loop, networks 0 to 9

Rms of network minus truth on the 2000 tempered points the network went on to acquire; the
bands are counted from each run's best true value, and T n / 2 = 52. A dash is a band with
fewer than 20 points.

| network | training points | batch, median depth | error on the batch | 15 to 52 below the best | 52 to 105 below | audit true calls | audit clear | drift |
|---|---|---|---|---|---|---|---|---|
| 0 | 2095 / 2095 | 737 / 426 | 1053 / 554 | | | 190 / 168 | no / no | |
| 1 | 4385 / 4363 | 119 / 84 | 26.4 / 52.7 | - / 16.4 | 31.8 / 28.0 | 86 / 38 | yes / no | 14 / 11 |
| 2 | 6571 / 6501 | 63 / 48 | 41.9 / 13.1 | 24.6 / 12.6 | 41.2 / 12.6 | 16 / 8 | no / yes | 5.5 / 3.3 |
| 3 | 8687 / 8609 | 44 / 43 | 10.6 / 9.7 | 10.1 / 9.4 | 11.1 / 10.0 | 11 / 0 | yes / yes | 2.1 / 0.98 |
| 4 | 10 798 / 10 709 | 46 / 52 | 8.8 / 28.4 | 6.6 / 7.0 | 9.6 / 17.7 | 0 / 0 | yes / yes | 1.1 / 1.0 |
| 5 | 12 898 / 12 809 | 49 / 50 | 10.8 / 8.8 | 4.9 / 5.0 | 9.2 / 10.6 | 0 / 0 | yes / yes | 0.34 / 0.53 |
| 6 | 14 998 / 14 909 | 49 / 51 | 9.5 / 10.7 | 4.7 / 5.5 | 10.0 / 12.3 | 0 / 0 | yes / yes | 0.13 / 0.81 |
| 7 | 17 098 / 17 009 | 49 / 49 | 7.7 / 6.8 | 5.1 / 3.9 | 8.7 / 8.0 | 0 / 0 | yes / yes | 0.24 / 0.34 |
| 8 | 19 198 / 19 109 | 48 / 50 | 7.1 / 5.0 | 4.2 / 2.9 | 8.1 / 6.1 | 0 / 0 | yes / yes | 0.32 / 0.29 |
| 9 | 21 298 / 21 209 | 48 / 49 | 5.9 / 4.5 | 3.4 / 2.9 | 7.1 / 5.4 | 0 / 0 | yes / yes | 0.27 / 0.19 |

From network 4 the error near the tempered posterior falls slowly, by a factor of two over
six iterations and 12 600 points; from network 5 the drift stays between 0.1 and 0.8.

What that left in the posterior, as the T = 1 posterior of network 10 against the run's last
network (mean shift in σ / width error):

| parameter | every point | window 16 |
|---|---|---|
| omega_b | +0.05 / −1% | −0.01 / 0% |
| omega_cdm | −0.03 / +1% | +0.05 / 0% |
| H0 | +0.02 / 0% | −0.04 / +1% |
| logA | +0.07 / +13% | −0.07 / +12% |
| n_s | +0.19 / −7% | −0.10 / −4% |
| tau_reio | +0.07 / +15% | −0.06 / +15% |
| A_planck | −0.02 / +1% | −0.08 / +1% |
| amp_143 | −0.24 / −27% | +0.03 / −19% |
| amp_217 | −0.31 / −32% | −0.01 / −23% |
| amp_143x217 | −0.30 / −31% | −0.02 / −22% |
| n_143 | +0.30 / −14% | +0.15 / −15% |
| n_217 | +0.23 / −24% | −0.02 / −23% |
| n_143x217 | +0.21 / −24% | −0.02 / −25% |
| calTE | +0.06 / +1% | +0.11 / 0% |
| calEE | +0.01 / −1% | +0.09 / 0% |

The same pattern in both runs: the six foreground parameters 14 to 32% too narrow, ln A_s and
τ 12 to 15% too wide, the other cosmological parameters right to about 1% in width. The two
networks before (8 and 9) are no better (foreground widths 16 to 44% too narrow). A tempered
batch puts a handful of its 2000 points within n / 2 of the peak, and the six foreground
parameters have the widest, least Gaussian posteriors.

## The boosts, networks 10 to 15

Network 11 is the first trained on boost points: from random weights, on the difference from
the quadratic background fitted then. Networks 12 to 15 warm-start from it.

**Every point**

| network | boosts trained on | error at its boost points: rms | bias | largest | lowest validation loss | training s | log evidence | CM68 against network 15 | drift |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 0 | 1.01 | −0.82 | 4.0 | 2.2e-3 | 439 | −5541.20 ± 0.05 | 0.31 | 0.12 |
| 11 | 1 | 0.25 | −0.01 | 2.0 | 1.4e-4 | 883 | −5540.17 ± 0.02 | 0.12 | 0.23 |
| 12 | 2 | 0.13 | −0.04 | 1.3 | 5.7e-5 | 500 | −5540.17 ± 0.03 | 0.043 | 0.053 |
| 13 | 3 | 0.10 | +0.01 | 1.2 | 3.6e-5 | 533 | −5540.11 ± 0.01 | 0.030 | 0.022 |
| 14 | 4 | 0.09 | +0.01 | 2.2 | 2.3e-5 | 569 | −5540.13 ± 0.05 | 0.017 | 0.028 |
| 15 | 5 | | | | 1.4e-5 | 596 | | (0.030) | 0.025 |

**Window 16**

| network | boosts trained on | error at its boost points: rms | bias | largest | lowest validation loss | training s | log evidence | CM68 against network 15 | drift |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 0 | 0.67 | +0.16 | 4.2 | 2.2e-3 | 565 | −5540.17 ± 0.03 | 0.30 | 0.098 |
| 11 | 1 | 0.14 | +0.04 | 0.8 | 6.9e-5 | 889 | −5540.11 ± 0.01 | 0.030 | 0.15 |
| 12 | 2 | 0.12 | +0.04 | 2.7 | 4.2e-5 | 491 | −5540.11 ± 0.04 | 0.024 | 0.092 |
| 13 | 3 | 0.06 | +0.01 | 0.5 | 2.0e-5 | 524 | −5540.12 ± 0.01 | 0.015 | 0.023 |
| 14 | 4 | 0.05 | 0.00 | 0.3 | 1.2e-5 | 562 | −5540.13 ± 0.01 | 0.040 | 0.021 |
| 15 | 5 | | | | 7.1e-6 | 600 | | (0.031) | 0.017 |

The CM68 in brackets is the chain noise of network 15 (two halves of its own chain). The log
evidence is the network's own, by bridge sampling inside the region, and is each boost's
by-product, so network 15 has none.

- The posterior is within chain noise of the final one from network 12 (every point) and
  network 11 (window 16): two boosts and one. The error at posterior points keeps falling
  after that, from 0.13 to 0.06 and from 0.12 to 0.06.
- The first run's network 11 still had ln A_s and τ 10 and 11% too wide; its network 12 has
  them within 3%.
- The log evidence moved by 1.03 with the first boost of the first run, which is the bias
  removed (network 10 was 0.82 low at posterior points), and by less than 0.1 afterwards.
- Every audit of networks 10 to 15 was clear without a true call; the highest maximum the
  searches found was 0.6 to 1.3 above the best true value (in the window run 4.9 at network
  10, before its first boost).
- Each boost found one mode. All 2000 candidates passed the density filter every time.
- From the second boost on, 8 to 23 of the 512 walkers of the mode search ended in smaller
  groups and were left out (the fixed barrier of 4 log-units at 15 parameters).
- The region held 2000, 4000, ... verified boost points, the farthest at 8.1 to 9.5 σ of the
  cloud, the edge at twice that.

## The two injections

With every point fitted the injected points lay 4 300 to 9 400 log-units below the peak
through all eleven injections; with the window they were at the tempered batch's own depth
from the third on ([injection_window.md](injection_window.md)).

At the end there is no difference: both final networks are 0.06 from the truth and the two
posteriors are the same. Before the boosts the run with the window was the better one
(network 10 at posterior samples 0.69 against 1.03, 220 effective points against 79; network
9 in the band 15 to 52 below the best 2.9 against 3.4), and it stayed about one boost ahead
(0.14 against 0.25 after one, 0.06 against 0.10 after three). With one seed each that cannot
be told from the scatter between two runs. What is certain is the cost: the first run spent
1100 evaluations where they taught the network nothing about the posterior.

## Cost

One node, 48 cores: 48 MPI ranks with one thread each for the likelihood, eight threads for
the network on rank 0.

| | every point | window 16 |
|---|---|---|
| the loop, 16 networks | 4.97 h | 5.01 h |
| likelihood, 33 100 points in 27 batches | 0.89 h | 0.90 h |
| per point | 4.6 core-seconds | 4.6 core-seconds |
| audit true calls | 303 | 214 |
| training | 2.06 h | 2.12 h |
| acquisition chains | 1.19 h | 1.20 h |
| audit, searches and calls | 0.19 h | 0.15 h |

A tempered iteration took 12 to 26 minutes. An iteration that trains on boost points and
acquires the next boost took 21 to 22, and 27 with the cold start. The likelihood is under a fifth of the wall time: training and
the chains run on one rank while the other 47 wait.

## What is not established

- **One likelihood, 15 parameters, one mode, a posterior close to Gaussian in its
  cosmological parameters.** Two runs, one seed each, which differ in the injection: the
  scatter between two runs of the same configuration was not measured.
- **The tails of the final network.** The true check tests the bulk (median 6.6 below the
  best). Between 15 and 105 below, the last measurement is network 9's (2.9 to 7.1); no
  network after the first boost was tested there.
- **How many boosts are needed, and when to start.** Here one or two gave the posterior and
  five gave 0.06. The loop's errors had stopped falling fast by network 4 or 5; starting the
  boosts there was not tried.
- **Which ingredient did it.** The cold start, the quadratic background, the region and the
  density filter were all on. No false maximum appeared after network 3, so the region was
  not needed against one.
- **The walkers left out** of the mode search from the second boost on (2 to 4%) were not
  looked into. `acquisition.boost.mixing` ([boost_modes_mixing.md](boost_modes_mixing.md)),
  which would join such groups when the chain mixes between them, was off.
- **The evidence.** −5540.13 in both runs is the networks' own, inside the region; it was not
  compared with a nested sampler.
- Wall times are from one cluster node and say nothing about another machine.

## Files

`hpc/runs/lcdm_planck_desi/` holds `cobaya.yaml` and `domain_bounds.json`, the likelihood and the training box of
both runs, and the `client.yaml` of the window-16 run. The launcher that applies the box is `hpc/launch.py`.
