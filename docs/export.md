# Exporting a network: `client.py export`

A network saved by CLiENT (`trained_models/model_it_N.keras`) holds CLiENT's own Keras layers:
the output scaling, the quadratic background, the region wall, the blend of a combined emulator,
and the activation. Keras can rebuild such a file only with CLiENT's code on the path; without it,
loading fails with a deserialisation error. `client.py export` writes the same function as a
TensorFlow SavedModel, which TensorFlow alone loads and calls. It is the form to hand to anyone
using the emulator outside CLiENT, for example with BEST, another sampler, or a notebook.

```bash
python client.py export RUN OUTPUT [--iteration N] [--no-region] [--dtype float32|float64]
```

- **The network:** N, by default the run's latest. RUN may be a combined emulator
  ([combine.md](combine.md)).
- **The region (default):** the exported function is the network inside the region of its boosts
  (`boost/region_it_M.npz`, M the largest at most N), as CLiENT's own evidence, profiles and chains
  use it. Inside the region it is the network; outside it is the region's wall,
  floor − 10⁴ (ρ − edge)², far below the peak. A network trained on boosts has false maxima outside
  its region.
- **`--no-region`:** the bare network over the whole box.
- **`--dtype`:** the precision of the exported function; float32 by default. float64 evaluates the
  same weights at double precision, which suits optimisers.

## What OUTPUT holds

| file | content |
|---|---|
| `saved_model/` | the SavedModel; its endpoint `serve` takes (n, ndim) at the chosen precision and returns (n, 1) |
| `emulator.json` | the parameters in order, with labels and box; the objective's convention; a reference value; the source (run, network, hashes, region; a combined emulator's record); the check below |
| `README.md` | how to call it, the parameter table, the two caveats below |

```python
import numpy as np, tensorflow as tf
emulator = tf.saved_model.load("OUTPUT/saved_model")   # keep the object: its variables live in it
loglkl = emulator.serve(tf.constant(x)).numpy()[:, 0]   # x: (n, ndim) in emulator.json's order
```

Keep a reference to the loaded object. `tf.saved_model.load(...).serve(x)` in one expression can
fail, because TensorFlow frees the object's variables when the object is dropped.

## The check

Before an export is accepted, it is reloaded in a separate Python process that cannot import
CLiENT (`python -I`, another working directory). It is compared with the source at 512 points of
the network's training set: the 256 best and 256 drawn at random.
- **Allowed:** a difference of 10⁻³ + 2 × 10⁻⁵ |value| at float32, or 10⁻⁸ + 10⁻¹² |value| at
  float64. An export that differs by more is refused and nothing is written: OUTPUT is built beside
  itself and renamed last, so a directory that exists is a whole, checked export.
- **Why the relative part:** TensorFlow orders the operations of a traced function differently
  from eager ones. At values of 10⁵ and more, on a region's wall or in a corner of the box, the
  two then differ by a few float32 steps, 0.3 at −590 510.
- **Recorded:** `emulator.json` holds the largest difference over all points, and over those
  within 50 of the best.

On the axion run's combined emulator (19 October 2026) the largest difference within 50 of the
best was 6 × 10⁻⁵ at float32 and 6 × 10⁻¹⁴ at float64.

## Two things a user must know

- **The output is the run's objective.** With the Cobaya wrapper this is Cobaya's log-posterior:
  the log-likelihood plus the log-prior of the run's Cobaya priors. Outside the box in
  `emulator.json` the network was never trained; treat it as outside the prior.
- **Sums in single precision.** BEST works in single precision, and adding up raw values of a
  real likelihood (−15 656 on the 49-parameter run) was measured to bias ln Z by 0.46
  ([evidence.md](evidence.md)). Subtract `reference` (the best training value) from the
  emulator's values before such a sum, and add it back to the result, as `client.py evidence`
  does.

Tests: `tests/test_export.py`.
