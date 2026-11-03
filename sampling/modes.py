"""The modes of a posterior that a T = 1 chain cannot travel between, and T = 1 samples and the
evidence of each. For a surrogate network: no true calls.

A tempered chain (the loop's acquisition chain) visits every mode; its split between them is
not a good estimate of their weights at T = 1, and a T = 1 chain stays where it starts. So:

  1. an ensemble started on the tempered chain is cooled to T = 1, and every walker then
     climbs to its local maximum (projected gradient ascent, as the audit's search);
  2. two maxima are in the same mode when the straight path between them never dips more than
     `barrier` log-units below the lower of the two; the connected groups are the modes, and a
     walker belongs to the mode of its maximum (groups holding less than `min_share` of the
     walkers are dropped and reported). The maxima, not the walkers, are compared: a walker of
     a T = 1 chain sits some ndim/2 log-units under its peak, which hides a barrier of that
     size, and one stranded on the barrier would join the two sides;
  3. each mode gets its own T = 1 chain, started on its walkers and kept in its region (the
     region of a mode: where its Gaussian, fitted to its T = 1 walkers, is the nearest);
  4. the evidence of each mode, the integral of exp(log-likelihood) over its region, by bridge
     sampling between the mode's chain and a Gaussian mixture fitted to half of it. The weights
     of the modes are the ratios of the evidences.

With one mode, steps 1 and 3 are a T = 1 chain and step 4 its evidence.

Step 2 by measured mixing (`mixing`, opt-in). A fixed `barrier` stands in for "a T = 1 chain
cannot travel between them", and how high a barrier a chain crosses depends on the dimension:
measured with this module's own chain on two equal Gaussians, 512 walkers and 10 000 steps
still mix across 6 log-units with 16 parameters and no longer across 3 to 4 with 58. With
`mixing` set, the maxima are first grouped with a low `candidate_barrier` (dips under it are
the network's own wiggles) into candidate basins, a free T = 1 chain of the same length as the
modes' chains is run over all of them, and two basins are one mode when that chain forgets
which of the two it started in: with a = P(in A at t + lag | in A at t) and b = P(in A at
t + lag | in B at t), both among the walkers found in A or B at t + lag, the memory a - b is 1
for basins that never exchange a walker and 0 for basins that exchange them freely. It is
measured at half the chain's length and squared (what is left after the whole chain); basins
whose memory is at most `mixing` are merged. The chain, the walkers and the number of steps
are the ones the modes will be sampled with, so the answer is about them. Where a walker is, is
decided by letting it climb: it is in the basin of the maximum it reaches.
"""

import time

import numpy as np

_BATCH = 16384


class NoModeFound(RuntimeError):
    """No group of cooled walkers reaches min_share: the network has no mode to draw posterior
    samples from. Seen on a network far from converged, whose surface is so rough that nearly
    every walker climbs to a maximum of its own."""


def split_evenly(n, parts):
    """n as `parts` whole numbers that differ by at most one, the larger ones first."""
    base, extra = divmod(int(n), int(parts))
    return [base + (1 if k < extra else 0) for k in range(parts)]


def as_numpy(predict):
    """A TensorFlow log-likelihood (n, ndim) -> (n,) as a NumPy function, evaluated in batches."""
    import tensorflow as tf

    def values(points):
        points = np.asarray(points, np.float32)
        return np.concatenate([
            np.asarray(predict(tf.constant(points[i:i + _BATCH])), float).reshape(-1)
            for i in range(0, len(points), _BATCH)]) if len(points) else np.zeros(0)

    return values


def climb(predict, points, lower, upper, steps=600, rates=(0.01, 1e-4)):
    """The local maximum each point climbs to: Adam in box units, projected onto the box, the
    step shrinking from rates[0] to rates[1]; the highest point visited is returned."""
    import tensorflow as tf

    lower, width = np.asarray(lower, float), np.asarray(upper, float) - np.asarray(lower, float)
    lo_t, width_t = tf.constant(lower, tf.float32), tf.constant(width, tf.float32)
    u = tf.Variable(np.clip((np.asarray(points, float) - lower) / width, 0.0, 1.0), dtype=tf.float32)
    rate = tf.Variable(rates[0], dtype=tf.float32, trainable=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lambda: rate)
    best_u = tf.Variable(u)
    best_f = tf.Variable(predict(lo_t + u * width_t))

    @tf.function
    def ascend():
        with tf.GradientTape() as tape:
            f = predict(lo_t + u * width_t)
            loss = -tf.reduce_sum(f)
        better = f > best_f
        best_f.assign(tf.where(better, f, best_f))
        best_u.assign(tf.where(better[:, None], u, best_u))
        optimizer.apply_gradients([(tape.gradient(loss, u), u)])
        u.assign(tf.clip_by_value(u, 0.0, 1.0))

    for step in range(steps):
        rate.assign(rates[0] * (rates[1] / rates[0]) ** (step / max(1, steps - 1)))
        ascend()
    f = predict(lo_t + u * width_t)
    better = f > best_f
    best_u.assign(tf.where(better[:, None], u, best_u))
    return (lower + best_u.numpy().astype(float) * width).astype(np.float32)


def _paths(values, points, interior):
    """f at the points, and for every pair (i, j) the lowest value on the straight path between."""
    points = np.asarray(points, float)
    f = values(points)
    i, j = np.triu_indices(len(points), 1)
    lowest = np.full(len(i), np.inf)
    for s in np.linspace(0.0, 1.0, interior + 2)[1:-1]:
        lowest = np.minimum(lowest, values(points[i] + s * (points[j] - points[i])))
    return f, i, j, lowest


def _components(n, f, i, j, lowest, barrier):
    """Maxima joined by paths that dip less than `barrier`: (group of every point, number of
    groups, which points are maxima, the joins before the stalled climbs were taken out)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    near = lowest >= np.minimum(f[i], f[j]) - barrier
    # A climb that stalled on a slope or a saddle is not a maximum, and one on the saddle would
    # join the two sides: a point more than `barrier` under a point it is joined to is left out.
    above = np.full(n, -np.inf)
    np.maximum.at(above, i[near], f[j[near]])
    np.maximum.at(above, j[near], f[i[near]])
    top = f >= above - barrier
    joined = near & top[i] & top[j]
    graph = coo_matrix((np.ones(int(joined.sum())), (i[joined], j[joined])), shape=(n, n))
    count, raw = connected_components(graph, directed=False)
    return raw, count, top, near


def find_modes(values, points, barrier, min_share=0.02, interior=8):
    """Labels 0, 1, ... of `points` (local maxima), by size; -1 for points of groups below
    min_share. Returns (labels, number of modes)."""
    points = np.asarray(points, float)
    n = len(points)
    f, i, j, lowest = _paths(values, points, interior)
    raw, count, top, _ = _components(n, f, i, j, lowest, barrier)
    sizes = np.bincount(raw[top], minlength=count)
    kept = np.flatnonzero(sizes >= max(2, min_share * n))
    kept = kept[np.argsort(-sizes[kept], kind="stable")]
    relabel = np.full(count, -1)
    relabel[kept] = np.arange(len(kept))
    return np.where(top, relabel[raw], -1), len(kept)


def find_basins(values, points, barrier, interior=8):
    """The candidate basins of the mixing test: `points` (local maxima) grouped as in find_modes
    with a low `barrier`, but nothing is dropped. A stalled climb goes to the basin of the
    highest point it is joined to (it lies on that hill; left out of the grouping, it cannot
    join two hills). Returns (labels 0, 1, ... by size, number of basins, dips): dips[a, b]
    is the smallest dip found on a straight path between a maximum of a and one of b."""
    points = np.asarray(points, float)
    n = len(points)
    f, i, j, lowest = _paths(values, points, interior)
    raw, count, top, near = _components(n, f, i, j, lowest, barrier)
    label = np.where(top, raw, -1)
    best = np.full(n, -1)                       # the highest point each stalled climb is joined to
    height = np.full(n, -np.inf)
    for a, b in ((i[near], j[near]), (j[near], i[near])):
        better = f[b] > height[a]
        order = np.argsort(f[b][better], kind="stable")      # the highest written last
        best[a[better][order]] = b[better][order]
        np.maximum.at(height, a, f[b])
    for k in np.argsort(-f, kind="stable"):                   # a higher stalled climb is settled first
        if label[k] < 0:
            label[k] = label[best[k]] if best[k] >= 0 and label[best[k]] >= 0 else count
            count += label[k] == count
    used, label = np.unique(label, return_inverse=True)
    sizes = np.bincount(label)
    order = np.argsort(-sizes, kind="stable")
    rank = np.empty(len(order), int)
    rank[order] = np.arange(len(order))
    label = rank[label]
    dips = np.full((len(order), len(order)), np.inf)
    dip = np.minimum(f[i], f[j]) - lowest
    maxima = top[i] & top[j]
    np.minimum.at(dips, (label[i[maxima]], label[j[maxima]]), dip[maxima])
    dips = np.minimum(dips, dips.T)
    np.fill_diagonal(dips, 0.0)
    return label, len(order), dips


def mixing_memory(labels, lag, basins=None):
    """How much a chain remembers, `lag` stored steps later, which of two basins it was in.

    labels: (stored steps, walkers) basin of every walker at every stored step (-1: none).
    Returns (memory, counts): counts[a, b] is the number of (step, walker) pairs found in a
    and, `lag` steps later, in b; memory[a, b] = counts[a, a] / (counts[a, a] + counts[a, b])
    - counts[b, a] / (counts[b, a] + counts[b, b]), 1 when no walker changes between a and b,
    0 when where a walker is does not depend on where it was (nan when one of the two was
    never occupied)."""
    labels = np.asarray(labels)
    k = int(labels.max()) + 1 if basins is None else int(basins)
    return _memory(labels[:-lag].ravel(), labels[lag:].ravel(), k)


def _memory(before, after, k):
    seen = (before >= 0) & (after >= 0)
    counts = np.zeros((k, k))
    np.add.at(counts, (before[seen], after[seen]), 1.0)
    stay = np.diag(counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        memory = stay[:, None] / (stay[:, None] + counts) - counts.T / (counts.T + stay[None, :])
    np.fill_diagonal(memory, 0.0)
    return memory, counts


def assign_basins(predict, values, points, peaks, basin, lower, upper, rng, representatives=48,
                  interior=8, chunk=1024):
    """The basin each of `points` lies in: every point climbs to its maximum, and takes the
    basin (`basin`, >= 0, of the known maxima `peaks`) to one of whose maxima the straight path
    dips least. Up to `representatives` maxima stand for a basin, the highest among them, so
    that a long ridge is not judged from its summit alone. Returns (labels, dips): the dip to
    the chosen basin, large when the point is on a hill none of the known maxima is on."""
    count = int(basin.max()) + 1
    chosen, owner = [], []
    height = values(peaks)
    for c in range(count):
        members = np.flatnonzero(basin == c)
        take = members if len(members) <= representatives else np.unique(np.r_[
            members[np.argmax(height[members])], rng.choice(members, representatives - 1, replace=False)])
        chosen.append(take)
        owner.append(np.full(len(take), c))
    chosen, owner = np.concatenate(chosen), np.concatenate(owner)
    anchors, anchor_height = np.asarray(peaks, float)[chosen], height[chosen]
    tops = climb(predict, points, lower, upper).astype(float)
    top_height = values(tops)
    labels, dips = np.empty(len(tops), int), np.empty(len(tops))
    for start in range(0, len(tops), chunk):
        block = tops[start:start + chunk]
        lowest = np.full((len(block), len(anchors)), np.inf)
        for t in np.linspace(0.0, 1.0, interior + 2)[1:-1]:
            path = block[:, None, :] + t * (anchors[None] - block[:, None, :])
            lowest = np.minimum(lowest, values(path.reshape(-1, path.shape[2])).reshape(len(block), -1))
        dip = np.minimum(top_height[start:start + chunk, None], anchor_height[None]) - lowest
        best = np.full((len(block), count), np.inf)
        for c in range(count):
            best[:, c] = dip[:, owner == c].min(axis=1)
        labels[start:start + chunk] = best.argmin(axis=1)
        dips[start:start + chunk] = best.min(axis=1)
    return labels, dips


class Regions:
    """The box divided among the modes: a point belongs to the mode whose Gaussian (mean and
    covariance of its T = 1 walkers, in box units) is nearest in Mahalanobis distance plus
    log-determinant. Between well separated modes the boundary runs through the barrier, where
    there is no posterior to speak of, so its exact course does not matter."""

    def __init__(self, clouds, lower, upper, groups=None):
        """groups: optional, the mode of every cloud (several basins that mix are one mode:
        its region is where any of its basins' Gaussians is the nearest)."""
        self.lower, self.width = np.asarray(lower, float), np.asarray(upper, float) - np.asarray(lower, float)
        self.groups = np.arange(len(clouds)) if groups is None else np.asarray(groups, int)
        ndim = len(self.lower)
        self.means, self.whiten, self.logdet = [], [], []
        for cloud in clouds:
            u = (np.asarray(cloud, float) - self.lower) / self.width
            cov = np.cov(u, rowvar=False) if len(u) > ndim + 1 else np.diag(u.var(axis=0))
            chol = np.linalg.cholesky(np.atleast_2d(cov) + 1e-10 * np.eye(ndim))
            self.means.append(u.mean(axis=0))
            self.whiten.append(np.linalg.inv(chol).T)
            self.logdet.append(2.0 * np.log(np.diag(chol)).sum())
        self.means, self.whiten, self.logdet = map(np.array, (self.means, self.whiten, self.logdet))

    def __len__(self):
        return int(self.groups.max()) + 1

    def labels(self, points):
        u = (np.asarray(points, float) - self.lower) / self.width
        z = np.einsum("knd,kde->kne", u[None] - self.means[:, None], self.whiten)
        return self.groups[np.argmin((z ** 2).sum(axis=2) + self.logdet[:, None], axis=0)]

    def labels_tf(self, x):
        import tensorflow as tf

        u = (x - tf.constant(self.lower, tf.float32)) / tf.constant(self.width, tf.float32)
        z = tf.einsum("knd,kde->kne", u[None] - tf.constant(self.means, tf.float32)[:, None],
                      tf.constant(self.whiten, tf.float32))
        nearest = tf.argmin(tf.reduce_sum(z ** 2, axis=2) + tf.constant(self.logdet, tf.float32)[:, None],
                            axis=0, output_type=tf.int32)
        return tf.gather(tf.constant(self.groups, tf.int32), nearest)


def _ensemble(predict, lower, upper, n_walkers, regions=None):
    """An AIES sampler of predict (inside the box, and inside one region when `regions` is
    given) whose temperature and region are set between runs: (sampler, beta, target)."""
    import tensorflow as tf

    from sampling.base import build_sampler

    lo, hi = tf.constant(lower, tf.float32), tf.constant(upper, tf.float32)
    beta = tf.Variable(1.0, dtype=tf.float32, trainable=False)
    target = tf.Variable(0, dtype=tf.int32, trainable=False)

    def log_prob(x):
        ok = tf.reduce_all((x >= lo) & (x <= hi), axis=1)
        if regions is not None:
            ok = ok & (regions.labels_tf(x) == target)
        return tf.where(ok, predict(x), -float("inf")) * beta

    return build_sampler("aies", n_walkers, len(lower), log_prob), beta, target


def _ladder(hot, cold, ratio):
    stages = max(1, int(np.ceil(np.log(hot / cold) / np.log(ratio) - 1e-9)))
    return list(np.geomspace(hot, cold, stages + 1)[1:])


def _cool(sampler, beta, positions, temperatures, steps):
    for temperature in temperatures:
        beta.assign(1.0 / temperature)
        sampler.run(n_steps=steps, initial_positions=positions, progress=False,
                    thin=max(1, steps // 4), chunk_size=5000)
        # The ensemble after the stage's last step. chain()[-1] is the row stored at three
        # quarters of the stage: its last quarter was computed and thrown away, at every
        # temperature of every boost.
        positions = sampler.last_positions()
    return positions


def _fill(cloud, n, lower, upper, accept, rng):
    """n distinct starting walkers from the points of one mode: the points themselves, then
    copies moved by a tenth of the cloud's spread (kept only inside the box and the region)."""
    cloud = np.asarray(cloud, float)
    if len(cloud) >= n:
        return cloud[rng.choice(len(cloud), n, replace=False)].astype(np.float32)
    ndim = cloud.shape[1]
    spread = (np.linalg.cholesky(np.cov(cloud, rowvar=False) + 1e-12 * np.diag((upper - lower) ** 2))
              if len(cloud) > ndim + 1 else np.diag(np.maximum(cloud.std(axis=0), 1e-3 * (upper - lower))))
    out = [cloud]
    need = n - len(cloud)
    for _ in range(50):
        if need <= 0:
            break
        base = cloud[rng.integers(0, len(cloud), 2 * need)]
        trial = (base + 0.1 * rng.normal(size=base.shape) @ spread.T).astype(np.float32)
        good = trial[np.all((trial >= lower) & (trial <= upper), axis=1) & accept(trial)][:need]
        out.append(good)
        need -= len(good)
    if need > 0:
        out.append(cloud[rng.integers(0, len(cloud), need)])
    return np.concatenate(out).astype(np.float32)


def bridge_log_evidence(x, f, values, inside, lower, upper, components, rng, n=20000, widen=1.3,
                        repeats=3):
    """log of the integral of exp(values) over the region `inside`, from samples x of
    exp(values) there (f = values(x)): bridge sampling (Meng and Wong) against a Gaussian
    mixture fitted to one half of the samples, widened, and cut to the region. Returns the mean
    over `repeats` random halves, their spread, and the efficiency of the mixture as an
    importance proposal (1: it is the posterior; near 0: do not trust the number)."""
    from scipy.special import logsumexp
    from sklearn.mixture import GaussianMixture

    x, f = np.asarray(x, float), np.asarray(f, float)
    lower, width = np.asarray(lower, float), np.asarray(upper, float) - np.asarray(lower, float)
    n = min(n, len(x) // 2)
    out, efficiency = [], []
    for repeat in range(repeats):
        order = rng.permutation(len(x))
        fit, use = order[:n], order[n:2 * n]
        unit = lambda y: (y - lower) / width
        mixture = GaussianMixture(min(components, max(1, n // (20 * x.shape[1]))), covariance_type="full",
                                  reg_covar=1e-9, random_state=int(rng.integers(2 ** 31))).fit(unit(x[fit]))
        mixture.covariances_ = mixture.covariances_ * widen
        mixture.precisions_cholesky_ = mixture.precisions_cholesky_ / np.sqrt(widen)
        # scikit-learn returns a mixture's draws component by component: without the shuffle the
        # n kept below are the first components' alone, against the density of the whole mixture.
        draws = rng.permutation(mixture.sample(4 * n)[0]) * width + lower
        keep = inside(draws)
        if not keep.any():                     # no estimate: the callers refuse a non-finite one
            return float("nan"), float("nan"), 0.0
        accepted = max(keep.mean(), 1.0 / len(keep))
        y = draws[keep][:n]
        log_g = lambda z: mixture.score_samples(unit(z)) - np.log(accepted) - np.log(width).sum()
        l1, l2 = f[use] - log_g(x[use]), values(y) - log_g(y)
        n1, n2 = len(l1), len(l2)
        s1, s2 = np.log(n1 / (n1 + n2)), np.log(n2 / (n1 + n2))
        r = np.median(l1)
        for _ in range(1000):
            new = (logsumexp(l2 - np.logaddexp(s1 + l2, s2 + r)) - np.log(n2)
                   - logsumexp(-np.logaddexp(s1 + l1, s2 + r)) + np.log(n1))
            done = abs(new - r) < 1e-10
            r = new
            if done:
                break
        out.append(r)
        w = np.exp(l2 - logsumexp(l2))
        efficiency.append(1.0 / np.sum(w ** 2) / n2)
    return float(np.mean(out)), float(np.std(out, ddof=1)) if repeats > 1 else 0.0, float(np.mean(efficiency))


def mode_log_evidence(k, *arguments):
    """bridge_log_evidence of mode k, or a stop before any true call: all the modes' weights are
    fractions of one sum, so one that is not finite leaves none of them known."""
    log_z, spread, efficiency = bridge_log_evidence(*arguments)
    if not np.isfinite([log_z, spread, efficiency]).all() or efficiency <= 0:
        raise ValueError(f"mode {k + 1}: the bridge estimate of its evidence failed (log evidence {log_z}, "
                         f"proposal efficiency {efficiency}), so the modes' weights are not known")
    return log_z, spread, efficiency


def modes_by_mixing(predict, values, peaks, at_one, lower, upper, *, mixing, candidate_barrier,
                    barrier, min_share, walkers, steps, rng, log=print):
    """Step 2 by measured mixing (see the module text). Returns (labels of the walkers, number
    of modes, clouds of the candidate basins, mode of every basin, report)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    basin, n_basins, dips = find_basins(values, peaks, candidate_barrier)
    sizes = np.bincount(basin, minlength=n_basins)
    large = sizes >= max(2, min_share * len(peaks))
    # A group too small to stand for itself joins the large basin with the lowest barrier to it
    # when that barrier is under `barrier` (one mode by the fixed rule too); else it is left out.
    home = np.where(large, np.arange(n_basins), -1)
    for k in np.flatnonzero(~large):
        if large.any():
            nearest = np.flatnonzero(large)[np.argmin(dips[k, large])]
            if dips[k, nearest] < barrier:
                home[k] = nearest
    kept = np.flatnonzero(large)
    index = np.full(n_basins, -1)
    index[kept] = np.arange(len(kept))
    candidate = np.where(home[basin] >= 0, index[home[basin]], -1)
    k = len(kept)
    peak_values = values(peaks)
    report = dict(threshold=float(mixing), candidate_barrier=float(candidate_barrier),
                  basins=[dict(walkers=int(np.sum(candidate == c)),
                               peak_value=float(peak_values[candidate == c].max())) for c in range(k)],
                  small_groups_joined=int(np.sum((candidate >= 0) & ~large[basin])),
                  left_out=int(np.sum(candidate < 0)), pairs=[])
    if k == 0:
        return candidate, 0, [], None, report
    clouds = [at_one[candidate == c] for c in range(k)]
    groups = np.zeros(k, int)
    if k > 1:
        started = time.monotonic()
        sampler, beta, _ = _ensemble(predict, lower, upper, walkers)
        beta.assign(1.0)
        thin = max(1, steps // 400)
        sampler.run(n_steps=steps, progress=False, thin=thin, chunk_size=5000,
                    initial_positions=_fill(at_one[candidate >= 0], walkers, lower, upper,
                                            lambda y: np.ones(len(y), bool), rng))
        chain = np.array(sampler.chain())
        sampler.reset()
        # Where every walker is at a few times and half the chain later: the hill it climbs
        # to. (A nearest-Gaussian label is not good enough here: fitted to the few walkers of
        # a small basin in many dimensions it gave such a basin away, measured.)
        lag = max(1, len(chain) // 2)
        origins = np.unique(np.linspace(0, len(chain) - lag - 1, min(8, len(chain) - lag)).astype(int))
        before = chain[origins].reshape(-1, len(lower))
        after = chain[origins + lag].reshape(-1, len(lower))
        known = candidate >= 0
        where, dip_to_basin = assign_basins(predict, values, np.concatenate([before, after]), peaks[known],
                                            candidate[known], lower, upper, rng)
        memory, counts = _memory(where[:len(before)], where[len(before):], k)
        # what is left of the memory after the whole chain; a basin never met is not mixed with
        left = np.clip(np.nan_to_num(memory, nan=1.0), 0.0, 1.0) ** (len(chain) / lag)
        a, b = np.triu_indices(k, 1)
        merged = left[a, b] <= mixing
        graph = coo_matrix((np.ones(int(merged.sum())), (a[merged], b[merged])), shape=(k, k))
        _, groups = connected_components(graph, directed=False)
        for i, j, one in zip(a, b, merged):
            report["pairs"].append(dict(
                basins=[int(i), int(j)], dip=float(dips[kept[i], kept[j]]),
                memory_at_half=float(memory[i, j]) if np.isfinite(memory[i, j]) else None,
                memory_after_chain=float(left[i, j]), exchanged=int(counts[i, j] + counts[j, i]),
                pairs=int(counts[i, i] + counts[j, j] + counts[i, j] + counts[j, i]), one_mode=bool(one)))
            log(f"   mixing: basins {i + 1} and {j + 1} ({report['basins'][i]['walkers']} and "
                f"{report['basins'][j]['walkers']} walkers; the straight path between their maxima dips "
                f"{dips[kept[i], kept[j]]:.1f} log-units): memory {left[i, j]:.2f} after {steps} steps "
                f"({int(counts[i, j] + counts[j, i])} of {report['pairs'][-1]['pairs']} walker positions half a chain "
                f"apart had changed basin): {'one mode' if one else 'separate modes'}")
        report.update(steps=int(steps), lag=int(lag * thin), positions=int(len(where)),
                      on_no_known_hill=int(np.sum(dip_to_basin > candidate_barrier)),
                      seconds=time.monotonic() - started)
        log(f"   mixing: the free chain and {len(where)} climbs took {report['seconds']:.0f} s")
        # the regions of the basins from every labelled position, not only the cooled walkers
        clouds = [np.concatenate([clouds[c], np.concatenate([before, after])[where == c]]) for c in range(k)]
    # modes by size, as find_modes numbers them
    size = np.bincount(groups, weights=[len(c) for c in clouds])
    rank = np.empty(len(size), int)
    rank[np.argsort(-size, kind="stable")] = np.arange(len(size))
    groups = rank[groups]
    labels = np.where(candidate >= 0, groups[np.maximum(candidate, 0)], -1)
    return labels, int(groups.max()) + 1, clouds, groups, report


def cooled_peaks(predict, lower, upper, tempered, temperature, walkers=512, cool_steps=2000, seed=None):
    """Where the walkers of a chain at `temperature` end when cooled to T = 1 and climbed, and
    the cooled walkers themselves: (summits, walkers), each (walkers, ndim). The first steps of
    posterior_modes, walker for walker with the same seed, without its grouping, sampling and
    evidences: for the audit's chain search (training/audit.py, audit_optima's `peaks`), which
    asks only whether the chain reaches a maximum that the audit's own climbs do not."""
    rng = np.random.default_rng(seed)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    tempered = np.asarray(tempered, np.float32).reshape(-1, len(lower))
    start = tempered[rng.choice(len(tempered), walkers, replace=len(tempered) < walkers)]
    sampler, beta, _ = _ensemble(predict, lower, upper, walkers)
    down = _ladder(max(float(temperature), 1.0), 1.0, 1.7) if temperature > 1.0 else [1.0]
    at_one = _cool(sampler, beta, start, down, cool_steps)
    sampler.reset()
    return climb(predict, at_one, lower, upper), np.asarray(at_one, float)


def posterior_modes(predict, lower, upper, tempered, temperature, *, barrier=4.0, min_share=0.02,
                    walkers=512, cool_steps=2000, steps=10000, components=6, seed=None, log=print,
                    mixing=None, candidate_barrier=1.0, mode_depth=None):
    """The modes of exp(predict) inside the box, with T = 1 samples and the evidence of each.

    predict: TensorFlow function (n, ndim) -> (n,). tempered: positions from a chain at
    `temperature` (at least `walkers` of them are used). Returns (modes, info): modes is a list,
    heaviest first, of dicts with x (samples, float32), f (predict at them), log_evidence,
    log_evidence_spread, efficiency, weight, walkers; info has the temperatures, the number of
    walkers dropped and the shares of dropped groups.

    mixing: None (the default) groups the maxima with the fixed `barrier`. A number in (0, 1)
    decides instead by a free T = 1 chain between candidate basins (`candidate_barrier`): two
    basins are one mode when the memory left after `steps` steps is at most `mixing`; info then
    has the test under "mixing".

    mode_depth: None (the default) keeps every mode. A number of log-units: a mode whose highest
    maximum lies more than that below the highest mode's is left out before its chain is run,
    and counted in info["deep_modes"] (acquisition.boost.mode_depth). In the axion run the first
    boost of a branch, searched over the whole box, found a "mode" 1030 below the posterior's,
    in a corner where the likelihood refuses every point, and spent half the boost there.
    "auto": temperature * ndim / 2, the depth of the tempered chain's own reach (84 for the
    axion run's 24 parameters at T = 7)."""
    from scipy.special import logsumexp

    rng = np.random.default_rng(seed)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if isinstance(mode_depth, str):
        if mode_depth != "auto":
            raise ValueError(f"mode_depth must be a number, 'auto' or None, not {mode_depth!r}")
        mode_depth = 0.5 * float(temperature) * len(lower)
    values = as_numpy(predict)
    tempered = np.asarray(tempered, np.float32).reshape(-1, len(lower))
    start = tempered[rng.choice(len(tempered), walkers, replace=len(tempered) < walkers)]
    sampler, beta, _ = _ensemble(predict, lower, upper, walkers)
    down = _ladder(max(float(temperature), 1.0), 1.0, 1.7) if temperature > 1.0 else [1.0]
    at_one = _cool(sampler, beta, start, down, cool_steps)
    sampler.reset()
    peaks = climb(predict, at_one, lower, upper)
    test = None
    if mixing is None:
        labels, count = find_modes(values, peaks, barrier, min_share)
        clouds, groups = [at_one[labels == k] for k in range(count)], None
    else:
        labels, count, clouds, groups, test = modes_by_mixing(
            predict, values, peaks, at_one, lower, upper, mixing=mixing,
            candidate_barrier=candidate_barrier, barrier=barrier, min_share=min_share,
            walkers=walkers, steps=steps, rng=rng, log=log)
    if count == 0:
        raise NoModeFound(f"no group of the {walkers} cooled walkers reaches min_share ({min_share:g}): no mode found")
    dropped = int(np.count_nonzero(labels < 0))
    deep = []
    if mode_depth is not None and count > 1:
        # By the modes' highest maxima, before any chain is run on them. The highest mode is
        # never deep, so at least one stays.
        tops = np.array([values(peaks[labels == k]).max() for k in range(count)])
        keep = tops >= tops.max() - mode_depth
        if not keep.all():
            deep = [dict(walkers=int(np.sum(labels == k)), peak_value=float(tops[k]),
                         below=float(tops.max() - tops[k])) for k in np.flatnonzero(~keep)]
            new = np.full(count, -1)
            new[keep] = np.arange(int(keep.sum()))
            labels = np.where(labels >= 0, new[np.maximum(labels, 0)], -1)
            if groups is None:
                clouds = [at_one[labels == k] for k in range(int(keep.sum()))]
            else:
                basins = keep[groups]
                clouds = [c for c, kept in zip(clouds, basins) if kept]
                groups = new[groups[basins]]
            count = int(keep.sum())
    depths = ", ".join(f"{d['below']:.0f}" for d in deep)
    log(f"   cooled {walkers} walkers from T = {temperature:g} through {', '.join(f'{t:.3g}' for t in down)} and climbed: "
        f"{count} mode{'s' if count > 1 else ''} "
        + (f"(barrier {barrier:g} log-units)" if mixing is None else
           f"of {len(clouds)} candidate basin{'s' if len(clouds) > 1 else ''} (measured mixing, memory at most {mixing:g})")
        + (f"; {dropped} walkers in smaller groups left out" if dropped else "")
        + (f"; {len(deep)} mode{'s' if len(deep) > 1 else ''} of {sum(d['walkers'] for d in deep)} walkers left out, "
           f"whose highest maximum lies {depths} "
           f"below the highest mode's (acquisition.boost.mode_depth {mode_depth:g})" if deep else ""))
    regions = Regions(clouds, lower, upper, groups) if count > 1 else None
    sampler, beta, target = _ensemble(predict, lower, upper, walkers, regions)
    modes = []
    for k in range(count):
        cloud = at_one[labels == k]
        in_region = (lambda y, k=k: regions.labels(y) == k) if regions is not None else (lambda y: np.ones(len(y), bool))
        inside = lambda y, in_region=in_region: np.all((y >= lower) & (y <= upper), axis=1) & in_region(y)
        own = cloud[inside(cloud)]
        beta.assign(1.0)
        target.assign(k)
        sampler.run(n_steps=steps, initial_positions=_fill(own if len(own) else cloud, walkers, lower, upper, in_region, rng),
                    burn_in=max(1, steps // 5), progress=False, thin=20, chunk_size=5000)
        x = np.array(sampler.chain()).reshape(-1, len(lower))
        f = np.array(sampler.log_prob(), float).reshape(-1)
        log_z, spread, efficiency = mode_log_evidence(k, x, f, values, inside, lower, upper, components, rng)
        peak = peaks[labels == k][np.argmax(values(peaks[labels == k]))]
        modes.append(dict(x=x, f=f, log_evidence=log_z, log_evidence_spread=spread, efficiency=efficiency,
                          walkers=int(len(cloud)), peak=peak, peak_value=float(values(peak[None])[0]),
                          acceptance=float(np.mean(sampler.acceptance_fraction().numpy()))))
        modes[-1].update(max_tau=float(sampler.max_tau), steps=int(steps),
                         chain_walkers=int(walkers), stored_samples=int(len(x)))
    sampler.reset()
    order = np.argsort([-m["log_evidence"] for m in modes], kind="stable")
    modes = [modes[k] for k in order]
    total = logsumexp([m["log_evidence"] for m in modes])
    for m in modes:
        m["weight"] = float(np.exp(m["log_evidence"] - total))
    for k, m in enumerate(modes):
        log(f"   mode {k + 1}: weight {100 * m['weight']:.1f}% (log evidence {m['log_evidence']:.2f} +- {m['log_evidence_spread']:.2f}, "
            f"proposal efficiency {m['efficiency']:.2f}); {m['walkers']} walkers; highest value {m['peak_value']:.2f}")
    return modes, dict(temperatures=[float(t) for t in down], dropped_walkers=dropped,
                       walkers=int(walkers), barrier=float(barrier),
                       **({} if test is None else {"mixing": test}),
                       **({} if mode_depth is None else {"mode_depth": float(mode_depth), "deep_modes": deep}))
