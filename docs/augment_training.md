# Adding profile points to a run's training set

`client.py augment` adds true evaluations made outside the training loop to a run's
training set, through CLiENT's acquisition filter. The usual source is a profile study:
every true-likelihood call of its true stage is a point with a known value.

```bash
python client.py profiles RUN --config profile.yaml -o STUDY    # a true profile along the suspect parameter
python client.py augment RUN --profiles STUDY -n 300             # its points, through the filter
python client.py RUN -i 5                                        # continue as usual
```

## When to use it

The acquisition aims for a training density c L^(1/T), with L estimated by the current
network. Where the network is too pessimistic, the region gets too few points, and because
nothing is evaluated there, the network is never corrected. The drift metric does not see
this: successive networks agree with each other.

A true profile along the affected parameter does see it, because it follows the likelihood
itself from where the network is right. So this is a tool for a direction you already
suspect: a parameter whose profile and posterior disagree, a plateau that runs to a domain
edge, a second solution known from the physics. It does not find problems on its own.

Two rules from the runs below:

- **Profile the whole domain of the parameter.** The network extrapolates the profile's
  trend into any part of the domain the profile did not cover, and acquisition then goes
  there first. If part of the domain is unphysical, end the domain where the physics ends.
  The command warns when a profile stops short of the domain.
- **An early network is good enough.** The true stage measures true gradients, so it does
  not need an accurate network, only more calls.

## The filter

The points are not all added. A training set distributed as c L^(1/T) has the same density
wherever L is the same, so the density a point *should* have around it can be read off the
training set:

1. The k-NN number density is computed at every training point (k is
   `acquisition.n_neighbors`, in the selector's whitened coordinates).
2. Its median in bins of the points' true log L, made monotone, is the density law ρ(L). It
   covers the range the acquired cloud spans, 0.5 T (n + 3√(2n)) log-units below the best
   training value. If fewer than 120 training points lie in that range, the law is read
   from the best 120 instead, with a warning: a training set that young need not follow
   L^(1/T) yet, and the filter then finds little to add.
3. A candidate with true value L is short of points if the k-NN density of the training set
   around it is below ρ(L). It is kept with the selector's retention factor
   a = [1 − ρ/ρ(L)]₊, the acceptance rule of the paper (eq. 13) evaluated with the true
   likelihood.
4. The candidates are visited once, in random order. Every point kept raises the density
   its neighbours see.

Both densities come from the same estimator, so its high-dimensional bias cancels, as in
`dataset/acquisition.py`. The consequences:

- A real gap (a near 1) is filled until the local density reaches the law.
- A tight cluster of profile probes contributes about k points, not all of them.
- Where the training set is already at its target, about half the candidates look slightly
  short from k-NN noise alone (a of 0.1 to 0.2); most are passed over.
- Candidates below the law's range, outside the run's domain, or already in the training
  set are left out. Coordinates are compared at the training set's single precision, so
  points injected earlier are recognised. A candidate above the best training value is
  judged against the density of the best training points.

`-n` is a ceiling. On a converged training set the filter stops below it (185 of 2848
candidates at iteration 12 of the first run below); on a young one the ceiling binds (300
of 4186 at iteration 5).

## What is written, and how a run uses it

| file | content |
|---|---|
| `RUN/augmentation/augment_it_N.csv` | the selected points: the run's parameters, then `loglkl` |
| `RUN/augmentation/augment_it_N.json` | sources, counts, the density law, each point's origin and deficit |
| `RUN/augmentation/augment_it_N.trained.json` | written by the run after the points' first training: iteration, epochs, schedule factor |

N is the training set the points were selected against, by default the one a continued run
loads. No existing file is changed. When the run is continued from iteration N, the loop
adds the points right after loading `data_it_N.csv` and says so; they are then part of
`data_it_<N+1>.csv` and every later file. If `model_it_N.keras` already exists, that network
is used as it is and the points first enter the training of iteration N+1.

Derived parameters of the added points are unknown and recorded as NaN, like any point
evaluated without them. To undo the step before continuing, delete the files.

Options: `--points FILE.csv` (any CSV with the run's parameters and `loglkl`, repeatable,
also together with `--profiles`), `-i N`, `--seed`, `--force` (replace an existing file),
`--output DIR` (a trial: write elsewhere, the run does not see it).

The study must have used the run's training objective (`objective: native` in its
manifest) and the run's parameters. It need not have been made from this run.

## The first training after an injection

Warm starting works because the training set changes little from one iteration to the
next. An injection is not such a change, so the first training that includes the points
runs a longer schedule: `lr_patience`, `lr_grace` and `patience` are multiplied by
`training.augment_schedule_factor` (default 3). The run prints a line when it does, and
every later iteration uses the ordinary schedule again. Set the factor to 1 to switch this
off. It has no effect in a run that is never augmented.

```yaml
training:
  augment_schedule_factor: 3   # the default; 1: the ordinary schedule
```

The measurement behind it is in the next section. The factor 3 is the only value measured,
and it was measured five iterations after the injection, not on the injection's own first
training.

## Validation on SINU, Planck + DESI DR2

Self-interacting neutrinos on Planck Commander low-ℓ TT + NPIPE CamSpec TTTEEE + DESI DR2
BAO (CLASS++, 16 parameters, T = 7, 2000 + 100 points per iteration). The true logGeff
profile is flat up to about −3.8, climbs a Δχ² = 13.4 barrier at −2.45 and falls into a
strong-coupling minimum at −1.55 that is level with the plateau (0.2 better). By that
profile about 28% of L^(1/7) lies beyond the barrier.

**The problem.** Twelve ordinary iterations put 5 to
10% of each batch beyond the barrier, none within Δχ² 42 of the minimum; the network stayed
about 80 log-units too pessimistic there from iteration 6 on, with the drift at 0.12.

**Retrospective test.** Iteration 12 retrained with the 185 points the filter selects from
the iteration-10 profile's 2848 calls (none below logGeff −2.30): the error at the minimum
drops from +79 to +3.8 log-units.

**First forward run** (domain logGeff ∈ [−5, 0], profile
on [−5, −0.8]). The first network with the points was right on the profiled ridge, but its
own profile ran away above −0.8, where the profile had no points: 183 in χ² too good at
logGeff = 0, then a false optimum at −1.2, then a corner of the box 3321 too good. Three
iterations did not settle it. Hence the first rule above.

**End-to-end run** (domain logGeff ∈ [−5, −0.8]):
iterations 0 to 5, true profile from network 5, `augment -n 300`, continued.

| | result |
|---|---|
| Network 5 on the true ridge | 40 to 280 log-units too pessimistic; no minimum beyond the barrier; its batch 0.2% beyond the barrier |
| Profile from network 5 | 4186 calls, 48 minutes on 4 workers; within 0.2 in Δχ² of the iteration-10 reference at 19 of 29 nodes, within 1.3 everywhere |
| Filter | 4143 of 4186 candidates short of density; 300 selected (the ceiling), spread over the whole profile |
| Network 6, first with the points | within about 10 log-units on the whole ridge (−1.1 at the minimum); its own profile ran down to −71 at the domain edge |
| Batch of network 6 | 89% beyond the barrier, all truly poor (median Δχ² 100 to 220): the over-extrapolation refuted in one batch |
| Networks 8 to 11 | drift 0.43, 0.30, 0.24, 0.25; batches 15 to 35% beyond the barrier; ridge error 6 to 12 log-units on the strong side |

**What the ordinary schedule left undone.** At iteration 10 the posterior had both modes,
the strong one in the right place but with 1.6% of the weight, visible only at the 99%
contour. The network had its minimum 5.3 in χ² above the weak one. The cause was the fit:
of the 82 training points within Δχ² 30 of the strong minimum, 79 were profile points
(three acquired in five iterations, since the T = 7 cloud sits about Δχ² 100 from a
minimum), and the network misfitted them by 5.5 log-units while fitting the weak side to
0.5. Iteration 11 retrained on the same 25 496 points:

| variant | epochs | misfit of the strong minimum's own training points | error on the true ridge, rms | strong minus weak minimum in the network's profile (truth −0.2) | note |
|---|---|---|---|---|---|
| warm start, ordinary schedule | 405 | +5.7 | 6.7 | +3.0 | as the pipeline |
| warm start, `lr_patience` and `lr_grace` × 3 | 2513 | +0.9 | 3.0 | −3.2 | strong mode 18.4% of the posterior, at logGeff −1.52 ± 0.09; no runaway |
| c lowered from 19.2 to 7.7 | 323 | +1.9 | 5.6 | +1.1 | barrier in the network's profile 12.9 → 6.1 (truth 13.4) |
| c lowered to 3.4 | 1264 | +1.1 | 3.6 | −2.6 | barrier 4.2, far wall 23 (truth 63) |
| cold start, ordinary schedule | 1326 | −0.15 | 0.4 | not usable | runaway optimum 35 210 in χ² too good, 0.5 box widths from any data; the T = 7 chain goes there with every sample |

Hence the longer schedule on the first training with the points, and not a lower c or a
cold start.

**What is not established.**

- The factor 3 was measured at iteration 11, not on the injection's own first training,
  and no other value was tried.
- Even with the longer schedule the network's own profile is good to a few units of χ²
  (rms 6.6 against the true profile). The posterior weight of a second mode depends
  exponentially on the relative height of the two minima, so 18.4% is an estimate of order
  10 to 30%, not a measurement.
- One model, one data combination, one seed.

**Alternatives to the profile that were tested** on the same run at iteration 11 or 12
(error at the strong minimum without any of them: +79 log-units). None is in the code:

| approach | result |
|---|---|
| true-likelihood climbs from the batch points the network under-predicted most (716 calls) | +2.1 on this run; on a synthetic 28-dimensional analogue the climbs reached the main optimum but not the hidden one |
| 300 points chosen by a freshly initialised network from its own chain | +73.7: all of them went to that network's own hallucination, Δχ² in the thousands |
| 300 points from the main network's chain, acceptance weighted by the spread of the main and two fresh networks | +77.6: the spread tracks the main network's error (rank correlation 0.8 to 0.9), but its own chain only offers candidates where it is optimistic |

The two network-only approaches failed: networks say where they are uncertain, not which
way the truth lies. The climbs need an acquired point already inside the hidden region.
