# CLiENT speed and memory changes

The speed and memory work on CLiENT since its first release, with the measurements behind it.

**Scope.** Speed and memory only. The changes to the *iterative training algorithm* — the
ESS-floored `msre` loss and dynamic `c`, Gaussian peak injection, per-iteration
re-injection, the new targets — are algorithm work and are in [CHANGES.md](CHANGES.md).
The optimiser work in §7 sits on the boundary; it is included because the motivation was
cost rather than accuracy, and flagged as such.

---

## 1. Candidate filtering in acquisition (15 September 2026)

`dataset/acquisition.py::select_points`. The loop uses `batch_size=10`, which is not a setting.

With the original selector, generating 32 100 points from a chain took **3 483 s** (58 min), single-threaded. Profiled
at the real sizes — pool 642 000 × 31 dims, k = 20:

| stage | time | share |
|---|---:|---:|
| setup: `_deduplicate`, two k-NN queries, the weighted draw | 67 s | 2% |
| **the greedy selection loop** | **~3 416 s** | **~98%** |

Within one pass:

| operation | ms | share |
|---|---:|---:|
| `new_distance` — `((pool_whitened - pool_whitened[j])**2).sum(1)` | **58.6** | **48%** |
| fancy-index update | 21.2 | 17% |
| `np.max(current_distances, axis=1)` | 14.3 | 12% |
| `np.argmax(current_distances, axis=1)` — *recomputes the same quantity* | 13.3 | 11% |
| building `a`, `np.random.choice`, logs, sums | 14.9 | 12% |

Cost is `O(pool_factor · n_append²)`: `pool_size = pool_factor · n_append` and the loop
runs `n_append` passes each costing `O(pool_size)`. A 32 100-point request therefore cost
~260× a 2 000-point one, not 16×.

### Tier 1 — exact, always on

1. **BLAS distance identity** `‖x−v‖² = ‖x‖² − 2x·v + ‖v‖²`, with `‖xᵢ‖²` precomputed once
   outside the loop: one `gemv` per pass instead of a `pool_size × ndim` broadcast.
   **58.6 ms → 3.5 ms.**
2. **Squared distances throughout.** The estimator needs `ndim·log r_k` = `(ndim/2)·log r_k²`,
   and the eviction test is a comparison `sqrt` cannot affect, so no `sqrt` appears in the
   loop. `max_i sqrt(d²ᵢ) == sqrt(max_i d²ᵢ)` bit-for-bit — `sqrt` is monotonic and
   correctly rounded — so `r_k` is unchanged.
3. **One `argmax`, not `max` *and* `argmax`.** The two calls returned the same element.
4. **Incremental re-max.** A cached `(worst_indices, r_k²)` pair updated by flat scatter,
   re-maximising only rows whose k-th neighbour changed, instead of rescanning all
   `pool_size × k` every pass.

Measured on the real chain, same seed and parameters, only the selector differing:

| | baseline | Tier 1 |
|---|---:|---:|
| wall time | 3 483 s | **878 s** |
| speedup end-to-end / loop-only | — | **3.97×** / 4.21× |
| per pass | 115.8 ms | 25.6 ms |
| points identical to baseline's selection | — | **32 092 / 32 100 (99.98%)** |
| KS(loglkl); KS per param max | — | 0.0000 (p = 1.0); 0.0004 |

The BLAS identity is exact in exact arithmetic and agrees to **8.8e-16 relative** at the
`r_k` scale (`d²` ≈ 16–47 whitened). The cancellation that can reach 1e-2 relative is
confined to `d² → 0` — a point against itself — and the result is clamped at zero.
`sklearn.metrics.pairwise.euclidean_distances` uses the same identity. Bit identity over
32 100 sequential draws is *not* claimed; `blas_distances=False` recovers it. In practice
the perturbation is local, not compounding: a redirected pick lands in the same deficit
region, so the trajectory re-converges — hence 8 changed points in 32 100.

### Tier 2 — `batch_size`, default 10

Commits `batch_size` candidates per pass, distances from one `gemm` instead of B `gemv`s,
turning `n_append` passes into `n_append / batch_size`.

| `batch_size` | wall time | speedup | shared with exact | KS(loglkl) | KS max |
|---:|---:|---:|---:|---:|---:|
| 1 | 878 s | 3.97× | 99.98% | 0.0000 | 0.0004 |
| **10 (default)** | **277 s** | **12.6×** | 11.5% | 0.0062 (p=0.57) | 0.0098 |
| 50 | 198 s | 17.6× | 8.2% | 0.0049 (p=0.83) | 0.0090 |

**An approximation in two separable ways**, both bounded by `batch_size`: *joint drawing*
(members don't see each other's density contribution) and *nearest-only insertion* (a pool
row absorbs only the closest batch member into its k-nearest list). The low overlap is
expected — the selector is stochastic and joint drawing diverges the trajectory
immediately — so the test is whether the draws come from the same density. They do, and
there is no clustering:

| set | median NN spacing | 5th pct | 1st pct |
|---|---:|---:|---:|
| exact | 3.3254 | 0.7931 | 0.2302 |
| B = 10 | 3.3190 | 0.8080 | 0.2307 |
| B = 50 | 3.3126 | 0.8036 | 0.2364 |

Beyond ~B = 50 it regresses: at B = 200 the `pool_size × B` block reaches 1 GB, leaves
cache, and the run got *slower* (4.6 min vs 4.0 at B = 50) while the approximation error
kept growing. **One target, one `n_append`, two batch sizes** — error grows with B and with
how concentrated the deficit landscape is, so repeat the spacing table on a new target.
`batch_size=1` in `select_points` is the exact path, which the density-filtered boost uses.

### Tests

The numbers above were established with six tests that are not in this repository:
bit-identity with `blas_distances=False, batch_size=1` against a frozen baseline of the
original selector, the same across three further parameter combinations, the BLAS
identity's precision and non-negativity, the BLAS default's agreement, the batched
default's distributional equivalence and absence of clustering, and the speedup.

---

## 2. The sampler

`sampling/aies.py`, `sampling/autocorr.py`. **The largest memory change in the codebase.**

### Preallocated chain buffer

A chain accumulated as a list of per-chunk tensors, concatenated with `tf.concat` and then
copied again by `.numpy()`, has all three coexisting, with a high-water mark of **2.93× one
chain**. Each chunk is therefore written straight into a preallocated `np.empty` array, and
`chain()` returns a view of the filled part.

| 216 walkers × 30 000 × 16D (0.386 GiB per copy) | peak RSS |
|---|---:|
| before | 1.589 GiB |
| after | **0.983 GiB** |

At 29D with 232 walkers × 100 000 steps a chain copy is 2.5 GiB, so this is the difference
between ~3.2 GiB and ~6.8 GiB resident; measured peak across the 29D runs was 3.8–4.4 GiB.
`np.empty` is lazily faulted, so preallocating at the maximum costs nothing until written —
a 1.29 GiB buffer with 30 000 of 100 000 rows written is 0.407 GiB resident. The buffer is
`float32`.

**API consequence:** `chain()` and `log_prob()` return **numpy views**, not tf tensors.
Callers must not call `.numpy()` on them, and must not `reset()` the sampler while still
holding them.

### `thin` at write time

`run(..., thin=k)` keeps every k-th step as it goes, so a thinned chain never allocates the
unthinned one. `benchmarking/benchmark.py` uses this rather than slicing an
already-materialised chain.

### `burn_in` inside `run()`

Burn-in is consumed by the sampler rather than sampled and then discarded, so burn-in rows
are never allocated in the chain buffer.

### `max(tau)` always reported

Costs ~0.7 s, 0.4% of a sampling step, and is the only way to see whether a fixed-length
chain is long enough. It exposed that a 16D setting of 30 000 steps delivered only **36 τ**.

---

## 3. Acquisition memory

* **Deduplication.** Duplicate MCMC states are found per walker by
  run-length detection — `np.flatnonzero(np.any(x[1:] != x[:-1], axis=1))` — instead of a
  sort or `np.unique` over the whole chain, and multiplicities are carried as counts so
  probability mass is preserved without materialising duplicates.
* **Reweighting from unique log-posteriors directly**, avoiding an intermediate
  full-length weight array.
* **No repeated distance-matrix reallocation** — the selection loop
  reuses one array rather than allocating per pass.

## 4. Convergence metric memory

`convergence/convergence.py`. `summarize(chain)` consumes the
`(n_steps, n_walkers, ndim)` ensemble array **per walker**, accumulating `sum_x` and
validating finiteness walker by walker, instead of requiring a flattened
`(n_samples, ndim)` copy of the whole chain. With a 2.5 GiB chain that copy would be the
second peak of the iteration.

## 5. MPI and startup

* **No full-array broadcast.** The entire sample array is not sent to every rank:
  `broadcast_and_evaluate` returns `(points, values)` with the master keeping the array and
  only work being distributed.
* **Deferred master-only imports** to cut MPI worker startup overhead.
* **Reduced evaluation verbosity in serial runs**, with progress tracking in
  `broadcast_and_evaluate`.

* **`CLIENT_MPI_SCHEDULE=chunked`** (2 October 2026, opt-in). The static schedule gathers
  after every point, so a batch of `n_ranks` points costs its slowest member. `chunked`
  scatters the whole phase once, one contiguous slice per rank, and gathers once: a slice
  costs the sum of its points. Six ranks, CLASS++ on Planck + DESI DR2, same machine:

  | schedule | evaluations per second | ranks busy |
  |---|---|---|
  | static | 1.25 to 1.41 | 87.5% |
  | chunked | 1.65 to 1.70 | 98.2% |

  Values identical to the static path. No dispatcher rank is given up. A dispatcher that
  hands out points one at a time was measured no faster than the static default and is not in
  the code: over a 640-point A/B on the real Planck likelihood ranks 1-7 took 3.27 to 3.31 s
  per evaluation, homogeneous to ±0.6%, so it has nothing to recover and costs a rank
  (`utils/mpi_utils.py`). Progress lines report rank 0's slice.

## 6. Main-loop lifetime management

* **Explicit release** of `chain`, `logposts`, `surrogate`, `model` and the sampler at the
  end of each iteration. This interacts with §2: `chain` and
  `logposts` are *views into the sampler's buffer*, so the sampler must be dropped too or
  nothing is actually freed — the code says so at the call site.
* **`tf.keras.backend.clear_session()`** between iterations, which also prevents a
  staleness bug.
* **Skip retraining when a checkpoint exists**, so a continued run costs one
  sampling pass plus acquisition at its restart iteration rather than a full retrain.
* **No extra chain/logpost copies** at the sampling boundary.
* **Dtype normalisation at surrogate boundaries**, preventing silent float64
  promotion of `float32` chains.

## 7. Optimiser cost (boundary case)

Motivated by training wall-clock, so included here, but these change what the optimiser does;
[CHANGES.md](CHANGES.md), "Deliberately not here", has the accuracy side. Both were measured and
left out: `training/training.py` uses Adam, and no setting chooses another optimiser.

* **An input-preconditioned Adam** — Adam with a K-FAC style `A`-factor preconditioner on the
  first layer only, taking `cond(A)` from 18.3 to **1.002**, trace-normalised so the
  learning rate keeps its meaning. Cost is one `d×d` inverse and one matmul per step,
  applied to 1 of 22 variables. Worth 1.20–1.41× in validation loss; **not** the
  explanation for whitening's larger gains, which the experiment was designed to test.
* **Muon: rejected on cost.** Keras 3.12.4's Muon is broken on the TensorFlow backend — it
  reads `variable.path` in seven places and the TF backend passes `ResourceVariable`s, which
  lack it; worse, momentum buffers are *written* under `path` and *read* under something
  else, so a naive attribute fix silently pairs each variable with the wrong buffer. With
  that patched properly it measured **10.6× slower per epoch** than Adam at Keras's default 6
  Newton–Schulz steps (1 712 s vs 162 s over 600 epochs) — the orthogonalisation on 512×512
  matrices.

---

## Measured and rejected

Recorded because each cost time to establish and would otherwise be re-proposed.

| idea | verdict |
|---|---|
| `n_jobs=-1` on the acquisition k-NN queries | **1.0×, no gain.** `_WhitenedKNN` builds `NearestNeighbors` without `n_jobs` so both setup queries are single-threaded, and a free multicore win looked available. There isn't one — and setup is 2% of runtime regardless. |
| `np.random.choice(pool_size, p=a)` as the selection-loop bottleneck | **Not a bottleneck** — 4.9 ms, 4% of a pass. The full cumulative sum looked expensive; it is not. Gumbel-max would cut it to 1.4 ms, ~3% of total, which does not justify changing the draw. |
| Parallelising across selections | **Impossible, not merely hard.** Each pick changes the merged density determining the next. The per-pass work *is* parallel, and routing it through BLAS is how cores actually get used — which is what §1 does. |
| Muon optimiser | 10.6× slower per epoch; see §7. |
| `batch_size` ≥ 200 | Slower than B = 50 *and* less accurate; see §1. |

## Memory summary

| | before | after |
|---|---:|---:|
| sampler chain high-water mark | 2.93 × one chain | 1 × one chain (view) |
| 29D, 232 × 100 000 resident | ~6.8 GiB | 3.8–4.4 GiB measured |
| convergence `summarize` | full flat chain copy | per-walker streaming |
| selection-loop distance temporary (pool 642 000 × 31) | 159 MB per pass | ~5 MB at B=1, 51 MB at B=10 |

Steady-state acquisition allocation is **unchanged** — `pool_whitened` (159 MB) and
`current_distances` (103 MB) dominate and neither was touched. `batch_size > 1` slightly
*increases* peak, by the `pool_size × batch_size` block.
