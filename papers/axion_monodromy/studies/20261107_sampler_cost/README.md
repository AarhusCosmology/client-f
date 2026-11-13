# The cost of standard samplers on the PDP+ likelihood, measured on its emulator (7 November 2026)

Standard samplers run on the PDP+ combined emulator of the seven parts (omega <= 200,
results/20261107_084147_axion_pp_combined200, network 0, inside its region; `emulator.py`) in place of CLASS++, counting
likelihood calls. Network only. The counts are what the same samplers would need with CLASS++, if the emulator is a fair
stand-in (true-check rms 0.067 at its posterior samples).

## Correction of the Metropolis-Hastings counts (8 November 2026)

The counts below and in the first versions of both tables were proposals (accepted samples / acceptance rate), not
likelihood calls: a proposal outside the prior box is rejected by Cobaya without a call. `calls_per_proposal.py` measured
0.85 calls per proposal for the PDP+ zone, 0.67 over the PDP+ prior, 0.46 and 0.48 for H0WC. Corrected (summarise.py,
sampler_cost.json): PDP+ zone R-1 = 0.05 / 0.02 / 0.01 after 3.8e5 / 8.9e5 / 1.3e6 calls; PDP+ whole prior R-1 = 0.07
after 2.9e5, ~2e6 for 0.01; H0WC zone R-1 < 0.02 after 8.6e5 and 0.016-0.022 up to 1.3e6, ~2.7e6 for 0.01; H0WC whole
prior R-1 = 0.06 after 3.2e5. Extrapolations now take R-1 proportional to 1 / calls from the last test (it overestimates
the PDP+ zone run by a factor 1.4). Both tables are corrected; the paper's Table tab:cost (build200.py) reads
sampler_cost.json.

## Runs

- `run_mcmc.py zone1b` (Cobaya 3.6.2 Metropolis-Hastings, 4 chains, omega in [38.25, 60.5], the proposal started from the
  emulator posterior's covariance there): R-1 of the means 0.15 after 1.6e5 calls, 0.05 after 4.6e5, below 0.02 after
  about 9e5, about 0.01 after 1.6e6 (0.0097 at 1.67e6), then fluctuating at 0.011-0.016; Cobaya's stop (also R-1 of the
  95% bounds < 0.2) not reached in 2.74e6 calls (stopped). 5.25 calls per accepted step. Its posterior agrees with the
  emulator's own (AIES) chain in that range: means within 0.02 sd (theta_s 0.17), the omega ~ 51 peak's share 0.762
  against 0.761.
- `run_mcmc.py full` (2 chains, omega over [1, 200], the zone run's learnt proposal, starts at random omega, max_tries
  10000): both chains crossed the whole omega range (through small A_osc, where omega does not matter) and sat in the
  same peaks (51, 179, 196); R-1 0.30 after 2.7e5 calls, 0.09 after 4.0e5 (stopped). 8.1 calls per accepted step. With
  the zone run's rate of decrease, R-1 ~ 0.01 needs ~2.5-3e6 calls (extrapolated, a factor 2-3). A first attempt with
  the whole posterior's covariance as the proposal stopped at once (760 rejections in a row).
- `run_nested.py 500 4` (dynesty 3.1.0, random-direction slice sampling with PolyChord's default of 5 d = 95 slices per
  point, multi-ellipsoid bounds, 500 live points): 475-555 calls per new point (9.6e5 calls in 3523 iterations, stopped).
  BEST on the same target needed n_dead = 54 n_live to reach a live fraction of 1e-3 (H = 38.2 nats), so ~2.7e4
  iterations and ~1.4e7 calls for sigma(ln Z) = sqrt(H / n_live) = 0.28, ~6e7 for 0.14 (2048 live points, BEST's).

## CLASS++ and CLiENT for comparison

- One evaluation: 26 core-s for PDP+, 80-83 core-s for H0WC (the kit's true checks, one core per evaluation).
- CLiENT, PDP+: 4.3e5 calls in the training sets after four boosts (posterior stable; ~8 h per part on 96 cores), 9.1e5
  at the end (~20 h per part), plus ~4e4 for the true checks. The parts run at the same time.
- Profiles with the true likelihood: 6.2e5 calls for the 20 000 nodes of the eight finished 2D studies, 31 per node
  (median 19-21).

## H0WC (8 November 2026)

The same runs on the H0WC combined emulator of the eight parts (results/20261107_084828_axion_wc_combined200; 22
parameters, 4.4 ms per call), with `SAMPLER_TAG=wc EMULATOR_RUN=<that run>` (outputs prefixed `wc_`):

- `run_mcmc.py zone1b` (4 chains, omega in [38.25, 60.5]): R-1 below 0.02 after 1.9e6 calls, then 0.016-0.021 up to
  2.8e6 (stopped; PDP+ reached ~0.01 at 1.6e6). 7.6 calls per accepted step (PDP+ 5.25). R-1 ~ 0.01 estimated at ~5e6.
- `run_mcmc.py full` (2 chains, the zone run's learnt proposal, random starts): both chains crossed omega 1-200; R-1
  0.06-0.08 between 5e5 and 6.4e5 calls (stopped); ~6e6 for R-1 ~ 0.01 (extrapolated, a factor 2-3).
- `run_nested.py 500 2`: ~600 calls per new point in the steady state (mean of 300 iterations; slices 5 d = 110);
  BEST on the H0WC emulator: n_dead = 52 n_live (106 368 / 2048), H = 37.1 nats, ln Z = -2921.51 +- 0.13; so ~1.6e7
  calls for sigma(ln Z) = 0.28, ~6.4e7 for 0.14.
- One H0WC evaluation with CLASS++: 82 core-s. CLiENT, H0WC: 5.0e5 / 5.9e5 / 7.7e5 / 1.04e6 training points after
  4 / 5 / 7 / 10 boosts; after five boosts each part's posterior agrees with its final one within 0.07 sd (widths 3%),
  except the lowest part (omega < 6, ~1% of the posterior), which moves by 0.1-0.25 sd until seven. Wall time ~19 h
  (four boosts), ~23 h (five), 40-44 h (final) per part on 96 cores. Profile studies: ~40-54 calls per node so far.
- Table: `sampler_cost_table_wc.tex` / `.pdf`.

## PROSPECT's profile likelihoods (8 November 2026)

PROSPECT 24.0.2 (simulated annealing; Holm, Nygaard, Dakin, Hannestad and Tram, arXiv:2312.02972; installed in `pylib/`)
on the PDP+ emulator of the seven parts in the zone omega in [38.25, 60.5] (`emulator_cobaya.py`, Cobaya kernel
`prospect/cobaya_pp_zone1b.yaml`), profile in A_osc at 0.01, 0.02, 0.03 and 0.04, two repetitions per point, the
settings of PROSPECT's `input/explanatory.yaml` (exponential temperature 0.1 to 0.001 over `max_iterations` 20, adaptive
step size) with 1000, 200, 100, 50 and 25 steps per iteration (`prospect/pp_zone1b_*.yaml`), started from the zone's
Metropolis-Hastings chains (`start_from_mcmc`, `start_bin_fraction` 0.1). Run from this folder with
`PYTHONPATH=pylib:. mpirun -n 5 python pylib/bin/prospect prospect/<yaml>`; analysed with
`prospect/analyse_prospect.py <output folders>` (writes `convergence.json` in each).

- PROSPECT has no stopping rule unless `chi2_tolerance` is set (the explanatory input does not set it): `max_iterations`
  only sets the cooling rate, and the annealing goes on below the lowest temperature. The runs were stopped after 25-40
  iterations; the cost quoted is that of the schedule's 20 iterations, (steps + 1) calls each (PROSPECT's Num_evals).
- Reference: `exact_profile.py` (version 2), L-BFGS-B with the network's gradients in unit coordinates of the box from
  30 MCMC samples and PROSPECT's best points, then L-BFGS-B and Powell until neither gains 1e-6: -6179.1818,
  -6177.6296, -6177.9047, -6180.0032 at omega 51.31-51.40 (`prospect/exact_profile_pp_zone1b.json`). Version 1 ran in the
  parameters themselves, stopped early, and was 0.25-0.47 low in chi^2 (`prospect/exact_profile_v1_raw_coordinates.log`).
  Every start from the MCMC ends at the same value now.
- The MCMC start (the best sample of the chain's bin) is on the right peak, 3.2-3.4 below the profile in chi^2.

| steps per iteration | calls per chain to within 0.1 in chi^2 (range of the 8 chains) | gap after 20 iterations | calls per point (2 repetitions x 20 iterations) |
|---|---|---|---|
| 1000 (explanatory input) | 9 000-11 000 | 0.007-0.010 | 40 040 |
| 200 | 1 800-2 400 | 0.010-0.013 | 8 040 |
| 100 | 1 100-1 400 | 0.013-0.026 | 4 040 |
| 50 | 660-820 | 0.017-0.048 | 2 040 |
| 25 | 360-490 | 0.028-0.058 | 1 040 |

The two repetitions of a point agree with each other to 0.025 in chi^2 or better. The number of iterations to 0.1 falls
only from ~10 to ~17 between 1000 and 25 steps, so the cost is set by the schedule, roughly 400-500 calls per chain at
the least. Not tried below 25 steps.

- Without an MCMC (`prospect/nomcmc/`: A_osc = 0.02, 100 steps, started from the MCMC's start with omega set to 40, 44,
  48, 55 or 59 and phi = 0, the zone's covariance matrix): every chain ends on the maximum in omega of its own basin,
  42.67 from 40 and 44 (11.5 below the profile in chi^2), 57.22 from 55 and 59 (9.0 below), 51.34 from 48 (0.015 below),
  and the two repetitions of a start agree, so they do not reveal it. The PDP+ profile in omega has about 50 local
  maxima over [1, 200] (median spacing 4; `omega200_pp.json` of the paper, 1-wide grid, so a lower bound), 8 of them
  within 5 in chi^2 of the best. A profile point without an MCMC needs a start in every basin.
- Compared with CLiENT's true stage: 31 calls per node on average (median 19-21) over the 20 000 nodes of the eight PDP+
  2D studies, against 1 000-40 000 per point for PROSPECT plus its MCMC (~1.6e6 calls in one zone, ~3e6 over the
  prior). PROSPECT computes 1D profiles only (2D left for future work in its paper).
- Emulator only: the network is smooth, and CLASS++'s value noise (~1e-3 in ln L) is not modelled.
