# Injection: fitting the quadratic to the points near the best

`prior.n_inject` and `acquisition.n_inject` add points drawn from the Gaussian that a
quadratic fit of the training set implies, at the acquisition temperature T. They are meant
to land about T n / 2 log-units below the peak (n parameters), where the tempered chain is.

On real CMB likelihoods in a wide training box they do not when the quadratic is fitted to
every training point: it puts them thousands of log-units below the peak. Both injections
fit only the points near the best, within 16 x T n / 2 log-units (`INJECT_WINDOW` in
`dataset/peak_samples.py`):

```yaml
prior:
  n_inject: 100
acquisition:
  n_inject: 100
```

The window does nothing while both `n_inject` are 0 (their default).

## What goes wrong

Without the window the quadratic is fitted by least squares to every training point. A log-likelihood is
quadratic near its peak, not over a box a hundred posterior widths across, where the training
values span tens of thousands of log-units. On two laptop runs (Planck + DESI, ΛCDM with 15
parameters and self-interacting neutrinos with 16) the fit to every point had:

- a residual of 12 000 to 43 000 log-units;
- its own peak value 7 000 to 48 000 above the best true value;
- 5 to 8 of its 15 or 16 curvatures with the wrong sign;
- its vertex 2 to 15 posterior widths (rms over the parameters) from the posterior mean.

The loop prints this in every such run, for instance

    Injecting 100 points from the quadfit Gaussian at T=7: fitted on 1996 points, vertex value 64583.1, ... fit rms 43214.3
       injected log-likelihood range [-15354.3, -14086.1]

where the best value of that run is about -5300. The same is seen on a cluster run (ΛCDM,
7 October: the injected points of iterations 0 to 2 lie 8812, 4937 and 8267 below the best).
On the analytic banana the fit to every point works, in a box of 10 and of 30 widths.

## What the window does

1. **Which points.** Only those within 16 × T n / 2 log-units of the best
   training value are fitted. If that leaves fewer than three points per coefficient of the
   quadratic (1 + n + n (n + 1) / 2 coefficients), the best three per coefficient are used.
2. **How wide.** The Gaussian is drawn with variance T / curvature along every
   eigenvector of the fitted Hessian, but never wider along an eigenvector than the fitted
   points themselves are spread along it. Where a curvature is near zero or has the wrong sign,
   T / |curvature| is a width the points say nothing about.

The fit routine, the stratified draw and the rejection at the box are the same with and without
the window. The log line names the points fitted, the wrong-sign curvatures and the
directions along which the width was capped.

## Against the true likelihood

Two cluster runs on 7 October, ΛCDM on Planck + DESI with 15 parameters, identical apart
from this key (same seed, so the same 2000 design points; best design point −7261.2 in
both). True values of the 100 injected points, as depth below the peak (−5494.1); a correct
Gaussian fit gives 52 ± 19:

| injection | every point: median | window 16: median | window 16: range | fit rms, every point | fit rms, window 16 | wrong-sign curvatures, window 16 |
|---|---|---|---|---|---|---|
| after the design | 4931 | 2125 | 1022 to 4026 | 37 846 | 1594 | 4 |
| after batch 0 | 8821 | 180 | 68 to 726 | 26 171 | 3.5 | 3 |
| after batch 1 | 4947 | 51 | 21 to 132 | 21 647 | 13.6 | 2 |
| after batch 2 | 8276 | 45 | 14 to 111 | 18 820 | 12.6 | 0 |
| after batch 3 | 8124 | 46 | 18 to 121 | 16 994 | 12.1 | 0 |
| after batch 4 | 7551 | 45 | 11 to 101 | 15 653 | 14.1 | 0 |
| after batch 5 | 7308 | 46 | 15 to 131 | 14 599 | 13.9 | 0 |

With every point fitted the injected points stayed between 4 300 and 9 400 below the peak
through all eleven injections of that run. With the window the first injection already improved on the best design
point by 745 log-units, and from the third on the surface's own peak value was within 8 of
the best training value. From the fourth on the injected points lie where the tempered batch
itself does (medians 45 to 46, the batch's 43 to 51): a hundred more points of the kind the
iteration already has two thousand of.

The same two runs at the same network (every point / window 16):

| network | best training value | tempered batch, median depth | network error on its batch | the same, 15 to 52 below the best | audit true calls | drift |
|---|---|---|---|---|---|---|
| 0 | −7261 / −6516 | 736 / 426 | 1053 / 554 | | 190 / 168 | |
| 1 | −5609 / −5562 | 118 / 84 | 26 / 53 | 79 (3 points) / 16 | 86 / 38 | 14.1 / 11.5 |
| 2 | −5533 / −5509 | 62 / 47 | 42 / 13 | 25 / 12.5 | 16 / 8 | 5.5 / 3.3 |
| 3 | −5504 / −5497 | 44 / 43 | 10.6 / 9.7 | 10.2 / 9.5 | 11 / 0 | 2.1 / 1.0 |
| 4 | −5494 / −5497 | 46 / 51 | 8.8 / 28 | 6.6 / 7.4 | 0 / 0 | 1.1 / 1.0 |
| 5 | −5494 / −5497 | 48 / 49 | 10.8 / 8.8 | 4.9 / 5.3 | 0 / 0 | 0.34 / 0.53 |

Errors are the rms of network minus truth on the 2000 points the network went on to acquire;
the band is counted from each run's own best value.

Over the first three networks the run with the window is about one iteration ahead: the
better best training value, the shallower batch, half the error near the peak at network 2,
and an audit that is clear without a true call from network 3 instead of network 4 (214 audit
calls in all against 303). From network 3 on the two runs are level: 15 to 52 below the best
the errors are 10.2, 6.6, 4.9 against 9.5, 7.4, 5.3. The window run's network 4 was the worse
on its whole batch (28 against 8.8; 122 of its points lay more than 105 below the best, where
its error was 100) and its network 5 is back at 8.8. The run without the window holds the
better best value at network 5 (−5494.1, from its audit of network 3, against −5497.2).

**At the end of the two runs** (ten tempered iterations and five boosts each;
[boost_real_lcdm.md](boost_real_lcdm.md)) there is no difference. Both final networks are
0.06 log-units rms from the truth at 500 samples of their own posterior, and the two
posteriors agree to 0.013 σ in every mean and 0.8% in every width. Before the boosts the run
with the window was the better one: its network 10 was 0.69 from the truth at posterior
samples against 1.03 (220 effective points of 500 against 79), its network 9 had 2.9 against
3.4 in the band 15 to 52 below the best, and it stayed about one boost ahead (0.14 against
0.25 after one boost, 0.06 against 0.10 after three). One seed each: that lead cannot be told
from the scatter between two runs. The run without the window spent its 1100 injected
evaluations 4 300 to 9 400 below the peak.

## Measurements on saved runs

Depth of the 100 injected points below the best true value of the whole run: median (10%,
90%). A correct Gaussian fit gives T n / 2 ± T √(n / 2).

For the points the runs themselves injected the true values are known. Every other depth is
the run's last network evaluated at the drawn points, standing in for the truth: on its own
training points that network is good to 3 to 18 log-units rms within 2000 of the best and
useless beyond, so a depth under 2000 is reliable and a larger one only says "far". The
stand-in reproduces the true depths of the runs' own injections (third against fourth column).

**ΛCDM, Planck + DESI, 15 parameters** (laptop run of 30 September; law 52 ± 19):

| iteration | training points | best point below the run's best | every point, true values | every point, stand-in | window 16, stand-in | points fitted | fit rms | wrong-sign curvatures | draws inside the box |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2095 | 1758 | 7507 (7199, 7749) | 7365 | 1180 (1040, 1297) | 408 | 1059 | 4 | 0% |
| 1 | 4195 | 1488 | 9117 (8348, 9435) | 10 098 | 356 (314, 416) | 1363 | 0.3 | 4 | 52% |
| 2 | 6295 | 725 | 9274 (8900, 9640) | 9633 | 150 (112, 194) | 1998 | 1.0 | 4 | 0% |
| 4 | 10 495 | 140 | 8497 (8059, 8931) | 9047 | 53 (30, 84) | 2606 | 5.6 | 2 | 90% |
| 7 | 16 795 | 1.8 | 16 054 (15 100, 17 491) | 15 228 | 40 (22, 68) | 8118 | 8.2 | 0 | 89% |
| 10 | 23 095 | 0 | | 6320 | 41 (22, 68) | 14 113 | 9.0 | 0 | 88% |

From iteration 4 the restricted fit is what the injection was designed to be: the points
land at the depth of the law and the fit has a residual of a few log-units. From iteration 7
it has no curvature of the wrong sign and its vertex is within 0.8 posterior widths of the
posterior mean. In iterations 0 to 2 there is little near the
peak to fit, and the points land well above the best training point (1180 against 1758, 356
against 1488, 150 against 725): there the injection would work as a step towards the peak.

**Self-interacting neutrinos, Planck + DESI, 16 parameters** (laptop run of 4 October;
law 56 ± 20):

| iteration | training points | best point below the run's best | every point, true values | every point, stand-in | window 16, stand-in | points fitted | fit rms | wrong-sign curvatures | draws inside the box |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2096 | 1179 | 9504 (9082, 9903) | 8247 | 3627 (2981, 4409) | 459 | 1587 | 5 | 0% |
| 1 | 4396 | 248 | 7090 (6899, 7438) | 6445 | 578 (490, 683) | 811 | 7.5 | 4 | 0% |
| 2 | 6622 | 77 | 8313 (8009, 8585) | 8259 | 1352 (429, 3594) | 2293 | 13 | 3 | 11% |
| 4 | 10 954 | 9 | 7862 (7525, 8139) | 7760 | 1069 (173, 2849) | 6364 | 13 | 2 | 67% |
| 7 | 17 465 | 0 | 6999 (6758, 7292) | 6916 | 242 (58, 1591) | 12 566 | 17 | 1 | 48% |
| 10 | 23 765 | 0 | 6100 (5824, 6345) | 5985 | 138 (54, 621) | 18 566 | 16 | 1 | 51% |

Much better than the fit to every point, but not at the law, and with a long tail. This
likelihood is not Gaussian near its peak either: it is flat in log10 G_eff over a wide
stretch and has a second mode, and one to three curvatures keep the wrong sign whatever
points are fitted. A Gaussian cannot do more here.

**Analytic banana, 16 parameters** (true values; law 56 ± 20), from a design of 2000 points:

| box | every point | window 16 |
|---|---|---|
| 10 widths | 40 (23, 59) | 39 (23, 58) |
| 30 widths | 36 (19, 56) | 35 (20, 56) |

No difference where the fit to every point already works. The same holds on the two larger
analytic targets of `input/`: on the 58-parameter
Gaussian the training set stays below three points per coefficient (5310) for the first
iterations and the window then fits every point, identically; on the 31-parameter banana in
a box of 30 widths the injected points are at the same depth or slightly nearer the peak
(true values, medians −179 against −179, −180 against −187, −142 against −175 over three
stages).

## What was tried and not adopted

- **Other windows** (4, 8, 16, 32 times T n / 2; the best 2, 3, 5 points per coefficient).
  Late in a ΛCDM run they all give the same. Early in a run none is best throughout: 16 is
  far better than 4 or 8 at ΛCDM iteration 2 (150 against 770) and SInu iteration 1 (580
  against 3300), and worse than 4 at SInu iteration 2 (1200 against 280). 32 is worse on
  ΛCDM (iteration 2: 1750). Small fixed counts are erratic on SInu.
- **Choosing the window by a hold-out** (fit without the best fifth of the points, predict
  them): it does not pick the window whose points land best.
- **Drawing the points in four rounds with a refit between them.** On ΛCDM it reaches the
  peak faster in the first iterations (best point 430 below the run's best after iteration 0,
  against 934 in one round). On SInu it is erratic: some rounds land 20 000 below
  (emulated with the stand-in).

## What is not established

- **Whether the injected points help at all on a real likelihood.** In the pair of runs above
  the final networks do not differ, and the lead of the run with the window before the boosts
  is within what two runs of one configuration might differ by; that scatter was not
  measured. No run without any injection was made on that likelihood. The evidence
  that injection speeds convergence is from analytic targets, where the fit to every point
  works.
- **One true-likelihood test**, on ΛCDM, one seed. On the non-Gaussian likelihood the
  restricted fit has only been judged with the stand-in network.
- One window (16) on two real likelihoods and one box. A narrower training box makes the fit
  to every point less wrong and the window less needed.
- The curved-degeneracy caveat of `UPGRADING.md` (a quadratic puts its vertex off a parabolic
  ridge) is about the shape near the peak and is not changed by the window.

## Files

- `dataset/peak_samples.py`: `near_best`, `gaussian_peak_samples(..., window=)`.
  `_fit_quadratic` is shared with the profile code: the window selects the points before the
  fit and caps the widths after it.
- `dataset/peak_samples.py`: `INJECT_WINDOW` (16). `client.py`: `draw_peak_samples` passes
  it on and prints what was fitted.
- `tests/test_injection_window.py`: the selection, the default left alone, a target with a
  Gaussian peak and one-sided walls where the fit to every point fails as on the real runs,
  the width cap, the configuration.