# Density-filtered posterior boosts

`acquisition.boost.density_filter: true` enables a posterior-focused version of CLiENT's
density filter. The default is `false`, the unfiltered boost and its evidence-correction
report; both recommended settings use the filter (`docs/boost.md`, "Two recommended settings"). See
[boost.md](boost.md) for mode finding, sampling and bridge weights.

```yaml
acquisition:
  audit: {}
  boost:
    n: 2000
    start_iteration: 15
    density_filter: true
    pool_factor: 20
```

With that block in a completed iteration-15 run's YAML:

```bash
python client.py RUN_DIR -s 15 -i 10
```

This requests ten acquisitions and trains through iteration 25. `n` is the number
requested **after filtering**, not the candidate count. Audit evaluations are
additional. A finite candidate pool may underfill; it is never topped up with
unfiltered points. No accepted points raises an explicit error. Boosts replace the
ordinary tempered acquisition and injection; they do not disable audit.

**Known audit limitation:** the native loop does not stop on `clear: false`.
Final verification may discover a maximum without sending it back for correction.
Inspect `audit/audit_it_N.json` before further sampling/acquisition from that network;
unattended filtered boosts need an external guard. The
study below used such a guard for SInu after the Gaussian failure.

## Selection and statistical interpretation

The mode finder supplies equal-size T=1 clouds, retaining repeated MCMC states.
For K modes the reference is the balanced mixture

    q_bal(x) = (1/K) sum_k q_k(x).

It is a training-density target, not the physical posterior mixture. Physical mode
weights still come from bridge evidences. Equal candidate supply does not force
equal accepted counts: existing local density determines the deficit.

With `acquisition.boost.balance_floor` (opt-in, docs/boost.md) mode k's cloud is cut
to s_k times the common size, s_k = min(w_k / floor, 1) over the largest such, so
the reference is sum_k s_k q_k / sum_k s_k: modes below the floor contribute in
proportion to their weight. Without it every cloud has the common size and the reference is
q_bal.

`dataset/posterior_boost.py` calls `dataset/acquisition.select_points` with a local
T=1 override and all genuine current training data, including earlier boost and
audit additions. Both current and reference densities use the same k-NN estimator
and whitening. Candidate and reference states are disjoint; sequential updates
(`batch_size=1`) include every accepted point before the next selection. The target
count scales with the existing plus accepted data, using the normal selector's
positive density deficit. Its normalization grows with the dataset: this is not
an absolute cap on core points and does not remove existing overdensities.

All equal float32 coordinates are grouped before selection, retaining residence
multiplicity as sampling weight. Coordinates already stored in the training set
are excluded. The run's ordinary acquisition temperature and global NumPy random
state are unchanged by the filtered selector. Optional RNG/exclusion arguments
leave the ordinary acquisition path unchanged.

With `n: 2000, pool_factor: 20`, the selector requests a 40,000-state candidate
pool and a disjoint reference cloud. When the chains hold too few distinct states
for that pool and a reference of more than `n_neighbors`, the pool is made smaller,
so that the reference keeps min(training points, half the states); the log says so
and the selection record has `pool_factor_requested`, `pool_factor_used` and
`distinct_states`. If that still leaves no such reference the selector raises an
error: extend the posterior chains or reduce `acquisition.boost.pool_factor`.
Larger candidate pools are not guaranteed to fill a density-limited request.

The identity `Z_true/Z_surrogate = mean(exp(true - surrogate))` assumes posterior
draws. Density-selected rows do not satisfy it. Their report sets
`evidence_correction_valid: false`, records selection/accepted counts and finite
true evaluations, and omits `log_mean_exp`, `effective_points` and
`corrected_weight`. Estimating a true mode-weight correction requires a separate
unfiltered diagnostic sample or another justified estimator. Reported errors on
selected points are descriptive, not unbiased posterior error estimates.

## Completed analytic study, 5–6 October 2026

Both studies used full native adaptive training and audit, not shortened pilot
fits. The schedule was **10** boosts of up to 2,000
accepted points. No CLASS++ evaluations were made.

- **Gaussian58:** start at the existing audited iteration-15 model.
- **SInu16:** rebuild with the real SInu settings, run iterations 0–5, compute a
  29-node true logGeff profile on [-5, -0.8], add 93 density-filtered profile
  points (cap 300), continue through iteration 15 with injection and audit, then
  boost ten times.
- Training: 5×512 Alsing, dynamic MSRE with ESS floor 0.1, batch 128, learning rate
  1e-4 down to 1e-6, plateau patience 75, floor grace 100, 5,000-epoch ceiling.
  The first post-augmentation training used the standard ×3 schedule.
- Ordinary acquisition: T=7, 2,000 points plus 100 injected points. SInu used
  256 walkers and 100,000 steps; Gaussian retained 432 walkers and 60,000 steps.
- Pointwise validation: fixed independent 24,000-point true-posterior sample and
  4,000-point T=7 sample, never used in training. Credible metrics use unfiltered
  per-mode chains with bridge-evidence weights, not density-selected rows.

| Study / checkpoint | Max CM68 | Max CM95 | Centred log-objective error SD | Strong-mode weight |
|---|---:|---:|---:|---:|
| Gaussian, before boost | 0.3091 | 0.1470 | 1.212 | — |
| Gaussian, four boosts (best measured) | 0.0577 | 0.0352 | 0.476 | — |
| Gaussian, five boosts, failed audit | 0.7330* | 0.3670* | 2.806 | — |
| SInu, before boost | 0.1211 | 0.0838 | 0.745 | 14.49% |
| SInu, seven boosts | 0.0369 | 0.0155 | 0.224 | 17.85% |
| SInu, ten boosts | 0.0452 | 0.0224 | 0.207 | 18.73% |

The analytic SInu strong-mode probability is 19.0348%. All ten batches filled,
adding 20,000 genuine points plus 18 boost-stage audit evaluations. Every saved
SInu boost audit was clear; the last needed six corrective evaluations after an
initial false maximum +1297 above the best true value.

\* Gaussian's fifth audit remained uncleared, so these chain-based metrics need
not describe its full surrogate posterior. Final verification found a false
maximum +2040 despite an earlier clear search. The native loop acquired a sixth
batch before the process was stopped during iteration-21 training. All outputs
were preserved. Six batches were acquired (12,000 points), five boosted networks
were completed, and the original sequence used 50 audit calls. No continuation
past this scientific stop was performed.

An isolated re-audit corrected that maximum with one additional true evaluation.
Its search became clear, but independent error SD remained 1.279 versus 0.476 at
the four-boost checkpoint. Clearing a search does not itself restore posterior
accuracy. The diagnostic used a documented new validation split; original shuffle
indices had not been saved, so it was not an exact replay.

## What is and is not demonstrated

Repeated boosts improved independent accuracy in these examples, but gains were
not monotonic and optimality of the filter was not established. SInu's final T=1
drift was 0.071, and some earlier checkpoints had better credible metrics. Final
chain-half maximum CM95 values were 0.0739 and 0.0215, with n_s responsible for the
first-half discrepancy; these reuse the fitted mode weights and do not measure
bridge-weight uncertainty. This supports improvement, not established convergence.

CM at level p is `(abs(upper-upper_ref) + abs(lower-lower_ref)) /
(upper_ref-lower_ref)`, with the repository's one-sided prior-bound conventions.
Gaussian references are analytic; SInu intervals use 500,000 independent bounded
analytic draws and reference-half checks. Its negligible softened posterior tail
is omitted. Constant objective offsets affect RMSE but not normalized posterior
shape; centred error SD is therefore reported separately. The final SInu chain
was initialized from preceding posterior states at T=1, whereas intermediate
rounds cooled from T=7; this final discovery step is not an independent global
mode search. Finite audit searches provide no global guarantee.

The SInu profile label-check tolerance alone changed 0.01→0.03 after verifying
float32 rounding on 1,003 points. No objective or optimizer tolerance changed.
Independent profile validation found median Δχ² gap 0.0104 and worst gap 0.442;
the anchor was not certified.

Earlier four-round, 1,000-point pilots used short 40-epoch fits and an immature
iteration-7 twin. Importance-proposal coverage was inadequate in the twin pilot;
its apparent gains did not survive direct sampling. Those pilots are not evidence
against the later full-schedule result. A selection-only two-mode test confirmed
that the filter directs additions toward the locally underfilled mode, including
in 58 dimensions; it does not establish improved neural-network accuracy.

Open, and not implemented: an audit-clear gate specifically for boosting; routing failed
final verification back into the remaining correction budget; a test of the preservation
of core accuracy; resampling selected checkpoints with independent seeds; comparison
against matched retraining-only and unfiltered controls.

## Reproduction and regression checks

Regression coverage in `tests/test_posterior_boost.py` checks duplicate residence
mass, existing-coordinate exclusion, local seeding, rejection of invalid clouds,
mode-dependent filtering, and omission of invalid evidence corrections. The
boost tests (`tests/test_boost.py`) cover the default unfiltered behaviour.
