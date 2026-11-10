# Moving from the first CLiENT

Nothing is required of a YAML that ran before, apart from what is listed here, but results will differ. Use the saved settings and
manifests when reproducing an older run. [CHANGES.md](CHANGES.md) says what is new.

## What changes results by default

| Setting | Default | To restore the old behaviour |
|---|---|---|
| `training.msre_ess_floor` | `0.1`: `msre`'s constant is raised while the weights collapse | `0.0` |
| `training.warm_start` | `true` | `warm_start: false` |

These have no setting: the raw predictions are kept when the scalers change, a warm-started training of the boost phase keeps its last weights, the
selector commits ten candidates per pass (about 90% of the points differ and their distribution does not), and an injection fits only the points
near the best.

The learning rate stays fixed unless `training.lr_schedule: plateau` is asked for, which real runs have done.

## What is refused now

- An unknown key anywhere in the YAML, or a value out of range. `acquisition.audit: false` is off (it gave an audit with all defaults).
- Fewer walkers than twice the number of parameters, before the initial design.
- A run without an objective identity cannot be continued: start a new run. A reviewed difference that leaves the objective unchanged needs
  `--accept-identity-change` once ([guide](docs/run_recovery.md)).
- `-s N` in a run that has gone further than network N: use `--rewind`, which moves the later files to `interrupted/` and deletes nothing. Do not
  delete files of earlier iterations of a run you mean to continue; committed files are checked at each start.
- A saved profile study whose manifest lacks a current key cannot be continued, or one that used a removed method: start a new study.
- `search_ranges` (10 November 2026) is such a key: a profile study made with an earlier client-f, 1D or 2D, cannot be continued
  with this one. Finish it with the client-f it was started with, or start it again.

## Things that surprise people

- **`-i` disables the convergence test.** On a continued run `-i N` means N more iterations.
- **A continued run restarts at the last iteration**, since the final iteration of any run never acquires. The network is loaded, not trained again.
- **Drift is not an accuracy measure.** It shows that a run is progressing, and carries a factor-of-two scatter on a plateau.
- **A seeded run is statistically, not bitwise, reproducible.** The AIES proposals run under XLA, which discards seeds.
- **`sampler.chain()` returns numpy views** into the sampler's buffer: do not call `.numpy()` on them, or `sampler.reset()` while holding them.
- **Injection interacts with the ESS floor.** Keep `n_inject > alpha * n_samples / (1 - alpha)`, with `alpha` the floor. On curved degeneracies a
  quadratic fit puts its vertex off the ridge: measure before trusting it.
- **Comparing with results at 216 walkers:** 432 walkers have twice the effective sample size, so the same credible-metric value is not measuring the same thing.

## Verifying an install

```bash
python client.py input/_smoke.yaml -n smoke -i 2
```

takes about 20 s on the analytic banana. `python tests/release_smoke.py` runs the whole workflow, with boosts, evidence, modes, a profile and an export, in about three minutes.
