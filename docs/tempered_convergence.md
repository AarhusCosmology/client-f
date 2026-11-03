# Convergence of the ordinary loop: the error at the tempered batch

```yaml
convergence:
  metric: tempered_batch_scatter   # the criterion described here
  threshold: 1.2                   # in units of sampling.temperature; 1.2 when left out
  consecutive: 2                   # networks running below it; 2 when left out
  max_iterations: 30
```

The ordinary loop (train, sample at temperature T, acquire a tempered batch) has converged
when the scatter of network minus truth at the tempered batch was below 1.2 T for two
networks running. It is opt-in: `metric: gaussian_posterior_drift` keeps the drift criterion.

## Why not the drift between chains

The drift criterion compares the chain of a network with the chain of the one before it and
stops below 0.01. On real likelihoods it has a floor far above that, set by the noise between
two trainings: none of five real runs (15 and 49 parameters) came near 0.01, and the run that
was by every other measure finished ended at 0.14. A run with that criterion goes on to
`max_iterations`.

The loop already has a better number. Every ordinary iteration evaluates the true likelihood
at a batch that its network chose and had not been trained on, so the network's error there
is a test against true values that costs nothing (`utils/held_out.py`). The loop prints it,

```
   network 9 at its tempered batch, which it had not seen: network minus truth rms 7.96 (mean -0.34, scatter 7.96, largest 46.7) at 3000 points
```

and it falls steadily as a run goes on.

## The criterion

At iteration N, when network N has been trained and sampled, the scatters at the batches of
networks N − 1 and N − 2 are known. If both are below `threshold` × T the ordinary loop has
converged: network N is the first trained on both batches.

- **Scatter, not rms.** A constant offset between network and truth changes no posterior of
  one mode. The standard deviation leaves it out.
- **In units of T.** The batch sits about T n / 2 below the peak, so the same error at the
  posterior shows as a larger error at the batch when T is larger. A fixed number would be
  wrong at another temperature (measured below).
- **Two networks running.** One low value can be a dip: a real 15-parameter run had 8.3 at
  network 3 and 26.9 at network 4.
- **Refused and non-finite points** are not in the scatter (as in the line the loop prints).
  A batch without a recorded error counts as not below the threshold.

What follows from convergence:

| the run | at convergence |
|---|---|
| has no `acquisition.boost` | it ends; network N is its last |
| has `acquisition.boost` whose `start_iteration` is still ahead | **the boosts begin with network N**; `start_iteration` is then only the latest iteration at which they begin |
| was given its number of iterations (`-i`) | nothing: the verdict is printed and recorded, the run does what it was told |

The criterion is never applied once the boosts have begun, and neither is the drift
criterion, which is printed there and decides nothing. A boost iteration reports the
error of its network at posterior points against true values
([boost.md](boost.md), "The error at the boost points, and when to stop"), and how many boosts
to run is the user's decision: `-i`, `max_iterations`, or

```yaml
acquisition:
  boost:
    n: 3000
    start_iteration: 30   # at the latest
    boosts: 5             # the run ends after five boosts (ignored with -i)
```

A run that hands over writes `RUN/boost/start.json` (the iteration, the two scatters, the
limit); a continued run reads it, with or without `-i`. The log:

```
tempered_batch_scatter: 7.96, 7.86 at the tempered batches of networks 9 to 10, against 1.2 x T = 8.4: converged
The ordinary loop has converged: the boosts begin with this network (acquisition.boost.start_iteration: 30); the run ends after 5 boosts, with network 16
```

The value kept in `metrics.json` (`convergence`, `metric_name: tempered_batch_scatter`) is
the larger of the two scatters in units of T. The drift between chains is also printed, as
information, and the chain summaries are written.

Runs with the other metric are not affected under MPI: the hand-over travels with the
broadcast of the stopping decision, and only in a run with this metric, which also sends one
broadcast at the start to tell the workers a recorded start.

## Where 1.2 comes from

### A suite on analytic likelihoods

The suite itself is not part of the release. Three likelihoods of
`input/cobaya/` (curved, 16 and 31 parameters; Gaussian, 58 parameters), T = 3, 5, 7 and 10,
two seeds: 20 of the 24 cases finished (four of the 58-parameter ones did not). A case is the
ordinary loop for 14 networks with the settings of the real runs and 2000 points per batch,
the true likelihood at 500 posterior samples of every network, and branches with three boosts
from the networks at which the posterior scatter first reached 8, 4, 2 and 1.

A hand-over is counted as working when the network after the three boosts has at least 95%
effective samples at its own posterior (weights exp(truth − network)). For the 31-parameter
likelihood, the highest batch scatter from which it worked and the lowest from which it did
not:

| T | worked from | failed from | threshold / T between |
|---|---|---|---|
| 3 | 2.1 | 4.0 | 0.7 and 1.3 |
| 5 | 7.2 | 8.7 | 1.4 and 1.7 |
| 7 | 8.6 | 9.2 | 1.2 and 1.3 |
| 10 | 11.8 | 13.1 | 1.2 and 1.3 |

The threshold rises from about 3 at T = 3 to about 12 at T = 10: it is proportional to T,
and 1.2 T is inside or at the lower end of every bracket.

- **16 parameters.** The boosts made progress from every hand-over, the very first network
  included (81 to 100% effective samples after three boosts). No threshold can be read off;
  a low dimension is forgiving.
- **58 parameters.** No base run got far enough in 14 networks (posterior scatter 1.4 to 1.7
  at the last), and every branch failed. What it shows is the scaling: the batch scatter at
  a given posterior scatter is proportional to T there too.
- A boost iteration without a mode ([boost.md](boost.md), "A network without a mode")
  occurred nine times, always after a hand-over at a batch scatter of 16 or more, and none
  of those branches recovered in three iterations.

### Real likelihoods

Five runs at T = 7: two of 15 parameters (ΛCDM, Planck and DESI) and three of 49 parameters
(one real likelihood in two training boxes and with two settings), all with boosts from
network 15. Looking back, a threshold anywhere from 8 to 11 log-units (1.15 to 1.6 T) on two
networks running is met before network 15 by exactly the three runs whose boosts then
worked, and not by the two whose boosts did not: those had a batch scatter of 11.5 and 15.4
when their boosts began, ended their five boosts with 12 to 49 effective samples of 500, or
ran into a network without a mode.

The 49-parameter run that worked had a batch scatter of 7.96 and 7.86 at networks 9 and 10,
so 1.2 T = 8.4 is met at network 11. A copy of the run with its boosts from network 11
instead of 15 confirmed it against true values at 500 posterior samples per network:

| boosts trained on | from network 11: scatter | effective of 500 | from network 15: scatter | effective of 500 |
|---|---|---|---|---|
| 0 | 1.96 | 31 | 1.06 | 210 |
| 1 | 1.13 | 134 | 0.57 | 345 |
| 2 | 0.46 | 405 | 0.54 | 349 |
| 3 | 0.67 | 287 | 0.53 | 378 |
| 4 | 0.59 | 355 | 0.56 | 339 |

From the second boost on the two are the same within their own scatter from network to
network, and the earlier hand-over had used four tempered iterations fewer (12 400 true
evaluations).

## What is not established

- Two seeds per temperature, and one real problem with a controlled comparison.
- Nothing above 49 parameters: the 58-parameter runs were too short to give a threshold.
- The 95% of the analytic suite is not a number for real likelihoods, whose networks level
  off lower (85% in the run above); only the comparison between hand-over points carries over.
- The criterion says when the ordinary loop may hand over. It says nothing about the
  posterior being right; that is what the boosts' errors and a check against true values are for.
