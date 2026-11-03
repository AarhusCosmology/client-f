"""acquisition.boost.region: the part of the box that the posterior points of the boosts reach.

A network is only as good as its training points allow, and far from them it can rise above
the best true value (training/audit.py). Whatever uses the network at T = 1 (the chains of the
next boost, a nested sampler, a profile) has no business there: a region that no posterior
chain touches carries no posterior weight. The region makes that a property of the surrogate
instead of something every consumer has to hope for.

It is built from the boost rows alone, the T = 1 posterior points whose true values are known
(RUN/boost/boost_it_N.csv): one Gaussian per mode, in box units, from the rows of that mode.
rho(theta) is the smallest, over the modes, of the Mahalanobis distance to the mode's Gaussian
divided by the largest such distance among the mode's own rows; inside means rho <= inflate.
No network enters and nothing is assumed about the shape beyond one ellipsoid per mode.

`restrict` gives the surrogate a consumer should see: the network inside, and outside a value
that does not look at the network at all, far below anything inside and falling away from the
region, so that a search or a sampler started anywhere in the box is led to it.

Measured on the analytic targets (docs/boost.md): with inflate 2 none of 2e6 exact posterior
draws lies outside (58 parameters, and the two-mode twin with 16), all known false maxima lie at
3 to 55 times the reach, and a nested sampler over the whole box returns the exact evidence
within its error from a network with false maxima 2000 log-units above the best.
"""

import json
from pathlib import Path

import numpy as np

# Outside the region the restricted surrogate is  floor - STRENGTH (rho - inflate)^2,  with
# floor = best true value - FLOOR.
FLOOR = 1000.0
STRENGTH = 1.0e4


# A cloud with at most this many times ndim distinct rows is sparse: it has no covariance of
# its own to speak of. Shrunk towards a sphere in box units (as it was for a day, 12 October
# 2026), its ellipsoid reached 64 to 80 times too far along the narrow directions of a
# 58-parameter posterior, and on a network of the two-mode twin a T = 1 chain inside the region
# of a mode known from 5 to 10 rows sat, in 3 of 12 draws, where the network is 6000 to 12000
# above the truth. With the shape of the well-sampled cloud it did not, in any.
SPARSE = 2

def _distinct(u):
    return len(np.unique(u, axis=0))


def whitener(cov, jitter=1e-14):
    """W with W' C W = 1 for the covariance C (plus `jitter` on its diagonal): the inverse of
    C's Cholesky factor, transposed. A point's Mahalanobis distance is the norm of ``(x - mean) W``."""
    cov = np.atleast_2d(cov)
    return np.linalg.inv(np.linalg.cholesky(cov + jitter * np.eye(len(cov)))).T


def lengths(z):
    """Euclidean length along the last axis (the same sums, in the same order, as
    sqrt((z ** 2).sum(axis=-1)))."""
    return np.linalg.norm(z, axis=-1)


def _reach(distances, quantile=None):
    """How far a cloud reaches: its farthest point, or with `quantile`
    (acquisition.boost.region_quantile) that quantile of its points' distances."""
    return float(distances.max()) if quantile is None else float(np.quantile(distances, quantile))


def _donor(units, distinct, quantile=None):
    """The shape a sparse cloud takes, (whiten, reach) in box units: that of the well-sampled
    cloud with the most rows; without one, of the scatter of all the clouds about their own
    means when together they are well-sampled; else None. Modes of one posterior differ in
    shape, but far less than a shape guessed from a handful of rows differs from either."""
    ndim = units[0].shape[1]
    full = [k for k, (u, n) in enumerate(zip(units, distinct)) if n > SPARSE * ndim]
    if full:
        u = units[max(full, key=lambda k: len(units[k]))]
        mean = u.mean(axis=0)
        w = whitener(np.cov(u, rowvar=False))
        return w, _reach(lengths((u - mean) @ w), quantile)
    scatter = [u - u.mean(axis=0) for u, n in zip(units, distinct) if n >= 2]
    freedom = sum(n - 1 for n in distinct if n >= 2)
    if freedom <= SPARSE * ndim:
        return None
    r = np.concatenate(scatter)
    w = whitener(r.T @ r / freedom)
    return w, _reach(lengths(r @ w), quantile)


class Region:
    def __init__(self, means, whiten, reach, lower, upper, inflate, floor, rows=None):
        self.means = np.atleast_2d(np.asarray(means, float))
        self.whiten = np.asarray(whiten, float).reshape(len(self.means), self.means.shape[1], -1)
        self.reach = np.asarray(reach, float).ravel()
        self.lower = np.asarray(lower, float)
        self.width = np.asarray(upper, float) - self.lower
        self.inflate = float(inflate)
        self.floor = float(floor)
        self.rows = [int(n) for n in (rows if rows is not None else [0] * len(self.means))]

    @classmethod
    def from_clouds(cls, clouds, lower, upper, inflate, best, quantile=None):
        """One ellipsoid per verified cloud of points (each (n, ndim)), reaching as far as
        its farthest point, or with `quantile` as that quantile of its points' distances
        (acquisition.boost.region_quantile; which rows are one mode is still decided by the farthest).

        A cloud with more than SPARSE * ndim distinct rows has its own covariance. A sparser
        one has no covariance to speak of: it takes the shape of a well-sampled cloud
        (`_donor`), its own centre, and reaches as far as its own rows do, never farther than
        the cloud whose shape it has. A cloud that cannot be given a shape (fewer than two
        distinct rows, or sparse with nothing to take a shape from) raises ValueError:
        `shaped` says which clouds can, and `boost_region` leaves the others out.
        """
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        width = upper - lower
        units = [(np.asarray(cloud, float) - lower) / width for cloud in clouds]
        distinct = [_distinct(u) for u in units]
        donor = _donor(units, distinct, quantile) if any(n <= SPARSE * u.shape[1] for u, n in zip(units, distinct)) else None
        means, whiten, reach = [], [], []
        for u, n in zip(units, distinct):
            mean = u.mean(axis=0)
            if n > SPARSE * u.shape[1]:
                w = whitener(np.cov(u, rowvar=False))
                far = _reach(lengths((u - mean) @ w), quantile)
            elif n >= 2 and donor is not None:
                w = donor[0]
                far = min(_reach(lengths((u - mean) @ w), quantile), donor[1])
            else:
                raise ValueError(
                    "a cloud of the boost region has no shape: "
                    + ("fewer than two distinct verified points" if n < 2 else
                       f"{n} distinct verified points in {u.shape[1]} dimensions and no well-sampled cloud to take a shape from")
                    + " (Region.shaped says which clouds have one)")
            means.append(mean)
            whiten.append(w)
            reach.append(far)
        return cls(means, whiten, reach, lower, upper, inflate, best - FLOOR,
                   rows=[len(c) for c in clouds])

    @staticmethod
    def shaped(clouds, lower, upper):
        """Which clouds `from_clouds` can give an ellipsoid, and whose shape is borrowed:
        (flags, borrowed), two lists of booleans as long as `clouds`."""
        lower = np.asarray(lower, float)
        width = np.asarray(upper, float) - lower
        units = [(np.asarray(cloud, float) - lower) / width for cloud in clouds]
        distinct = [_distinct(u) for u in units]
        sparse = [n <= SPARSE * u.shape[1] for u, n in zip(units, distinct)]
        donor = _donor(units, distinct) if any(sparse) else None
        flags = [not thin or (n >= 2 and donor is not None) for thin, n in zip(sparse, distinct)]
        return flags, [thin and flag for thin, flag in zip(sparse, flags)]

    def __len__(self):
        return len(self.means)

    def radii(self, points):
        """(modes, n): Mahalanobis distance to every mode's Gaussian over that mode's reach."""
        u = (np.asarray(points, float) - self.lower) / self.width
        z = np.einsum("knd,kde->kne", u[None] - self.means[:, None], self.whiten)
        return lengths(z) / self.reach[:, None]

    def rho(self, points):
        return self.radii(points).min(axis=0)

    def inside(self, points):
        return self.rho(points) <= self.inflate

    def labels(self, points):
        """Disjoint nearest-ellipsoid cells, including overlaps (ties go to the first)."""
        return self.radii(points).argmin(axis=0)

    def labels_tf(self, x):
        import tensorflow as tf
        u = (x - tf.constant(self.lower, tf.float32)) / tf.constant(self.width, tf.float32)
        z = tf.einsum('knd,kde->kne', u[None] - tf.constant(self.means, tf.float32)[:, None],
                      tf.constant(self.whiten, tf.float32))
        return tf.argmin(tf.reduce_sum(z * z, axis=2) / tf.constant(self.reach ** 2, tf.float32)[:, None],
                         axis=0, output_type=tf.int32)

    def rho_tf(self, x):
        import tensorflow as tf

        u = (x - tf.constant(self.lower, tf.float32)) / tf.constant(self.width, tf.float32)
        z = tf.einsum("knd,kde->kne", u[None] - tf.constant(self.means, tf.float32)[:, None],
                      tf.constant(self.whiten, tf.float32))
        radius = tf.sqrt(tf.reduce_sum(z * z, axis=2) + 1e-12)
        return tf.reduce_min(radius / tf.constant(self.reach, tf.float32)[:, None], axis=0)

    def project_unit(self, u):
        """Box-normalised points outside the region moved onto its surface, straight towards
        the centre of the nearest mode; points inside stay. For gradient searches
        (training/audit.py find_maxima), which a wall would only throw about."""
        import tensorflow as tf

        means = tf.constant(self.means, tf.float32)
        z = tf.einsum("knd,kde->kne", u[None] - means[:, None], tf.constant(self.whiten, tf.float32))
        radius = (tf.sqrt(tf.reduce_sum(z * z, axis=2) + 1e-12)
                  / tf.constant(self.reach, tf.float32)[:, None])
        centre = tf.gather(means, tf.argmin(radius, axis=0))
        scale = self.inflate / tf.reduce_min(radius, axis=0)
        return tf.where((scale < 1.0)[:, None], centre + (u - centre) * scale[:, None], u)

    def restrict(self, predict):
        """predict: (n, ndim) -> (n,). The same inside the region; outside, the wall."""
        import tensorflow as tf

        edge = tf.constant(self.inflate, tf.float32)
        floor = tf.constant(self.floor, tf.float32)

        def restricted(x):
            rho = self.rho_tf(x)
            return tf.where(rho <= edge, predict(x), floor - STRENGTH * tf.square(rho - edge))

        return restricted

    def start(self, n, rng, component=None):
        """n points inside the region and the box: starting walkers for a chain. With
        `component`, inside that ellipsoid's cell (the points nearer to it than to any other).
        A cell can be a small part of its ellipsoid, one that lies inside another's or is cut
        by one: the draw then closes in on the ellipsoid's centre, which is in its own cell.

        Without `component` the region can also reach far beyond the box, along parameters
        whose posterior fills their range or lies against a bound: 0.46% of the draws were
        in the box at network 18 of an axion run, whose farthest boost row had gone from 9.7
        to 18.2 sigma, and 200 draws left the chain 70 walkers short. When 200 draws have not
        found n points, the draw closes in on the centres the same way. A draw that finds its
        points in those 200 is the one it always was, number for number."""
        out = []
        need = n
        upper = self.lower + self.width
        scale = 1.0
        unwhiten = np.linalg.inv(self.whiten)
        for attempt in range(200 if component is not None else 400):
            k = (rng.integers(0, len(self.means), 2 * need) if component is None
                 else np.full(2 * need, component, dtype=int))
            z = rng.normal(size=(2 * need, len(self.lower)))
            z *= (self.reach[k] * rng.random(2 * need) ** (1.0 / len(self.lower))
                  / np.linalg.norm(z, axis=1))[:, None]
            u = self.means[k] + scale * np.einsum("nd,nde->ne", z, unwhiten[k])
            x = self.lower + u * self.width
            keep = np.all((x >= self.lower) & (x <= upper), axis=1)
            if component is not None:
                keep &= self.labels(x) == component
                if keep.sum() < 0.1 * len(keep):
                    scale *= 0.5
            elif attempt >= 200 and keep.sum() < 0.1 * len(keep):
                scale *= 0.5
            good = x[keep][:need]
            out.append(good)
            need -= len(good)
            if need <= 0:
                break
        if need > 0:
            raise RuntimeError("the region lies outside the box" if component is None else
                               f"no starting point in cell {component} of the region: its ellipsoid's centre lies "
                               "outside the box or in another ellipsoid's cell")
        return np.concatenate(out).astype(np.float32)

    def save(self, path, **extra):
        np.savez(path, means=self.means, whiten=self.whiten, reach=self.reach, lower=self.lower,
                 upper=self.lower + self.width, inflate=self.inflate, floor=self.floor,
                 rows=np.asarray(self.rows), extra=json.dumps(extra))

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            return cls(f["means"], f["whiten"], f["reach"], f["lower"], f["upper"],
                       float(f["inflate"]), float(f["floor"]), rows=f["rows"].tolist())


def _matching(sets, lower, width):
    """For deciding which sets of rows are one mode: (mean, whiten, reach) of each set in box
    units, or None. A well-sampled set has its own. A sparse one is taken to be as wide as the
    well-sampled set whose shape it borrows (`_donor`): its own few rows say where it is, not
    how far it goes, and with their own reach two rows of a mode never met the next boost's two."""
    units = [(np.asarray(points, float) - lower) / width for points in sets]
    distinct = [_distinct(u) for u in units]
    ndim = len(lower)
    donor = _donor(units, distinct) if units and any(n <= SPARSE * ndim for n in distinct) else None
    out = []
    for u, n in zip(units, distinct):
        mean = u.mean(axis=0)
        if n > SPARSE * ndim:
            w = whitener(np.cov(u, rowvar=False))
            out.append((u, mean, w, float(lengths((u - mean) @ w).max())))
        else:
            out.append(None if donor is None else (u, mean, donor[0], donor[1]))
    return out


def _share(a, b):
    """The fraction of a's rows inside b's ellipsoid (0 without shapes)."""
    if a is None or b is None:
        return 0.0
    return float(np.mean(lengths((a[0] - b[1]) @ b[2]) <= b[3]))


def _one_mode(a, b):
    """Whether two overlapping sets may be joined: the mean of the one with fewer rows lies
    within the reach of the other, in that other's shape. By the mean, as the rule before 12
    October 2026 had it, and not by single rows: one stray row of overlap merged a mode's 3 or
    4 rows into the other mode's 10000, whose reach went from 10.6 to about 29, while a bound
    on the farthest row split clouds of 50 rows in 16 dimensions that are one mode (their
    covariance is noisy and new rows lie beyond the reach of the old)."""
    if a is None or b is None:
        return False
    small, large = (a, b) if len(a[0]) < len(b[0]) else (b, a)
    return bool(lengths((small[1] - large[1]) @ large[2]) <= large[3])


def boost_clouds(directory, names, before, best, depth, lower, upper):
    """The verified posterior points of the boosts before iteration `before`, one cloud per mode.

    A boost row counts when its true value is finite and within `depth` of the best: a row the
    network placed in its posterior and the truth did not is no posterior point. The modes of
    one boost are numbered heaviest first, which need not be the same order in the next: the
    first boost's modes start the clouds. After that an old cloud joins the group of the
    current boost with which it shares the largest fraction of rows (its rows inside the
    group's ellipsoid, or the group's inside its own), provided the two are one mode
    (`_one_mode`: the smaller one's mean inside the larger one's reach); otherwise it stays as it is, as does a mode absent from this boost. Rows are
    assigned once, and several old clouds can join one group: a later boost can find as one
    mode what earlier ones found as two. Returns (clouds, info). Every cloud is returned,
    however few its rows: `boost_region` decides which can be given an ellipsoid.
    """
    import pandas as pd

    directory = Path(directory)
    lower = np.asarray(lower, float)
    width = np.asarray(upper, float) - lower
    files = sorted((int(p.stem.rsplit("_", 1)[1]), p) for p in directory.glob("boost_it_*.csv"))
    clouds, used, dropped, merged = [], [], 0, 0
    for iteration, path in files:
        if iteration >= before:
            continue
        frame = pd.read_csv(path)
        value = frame["loglkl"].to_numpy(float)
        good = np.isfinite(value) & (best - value <= depth)
        dropped += int(np.count_nonzero(~good))
        used.append(iteration)
        groups = [frame.loc[good & (frame["mode"] == label).to_numpy(), names].to_numpy(float)
                  for label in sorted(frame["mode"].unique())]
        # largest first, as the clouds of a first boost have always been ordered
        groups = sorted((g for g in groups if len(g)), key=len, reverse=True)
        shapes = _matching(clouds + groups, lower, width)
        old_shapes, new_shapes = shapes[:len(clouds)], shapes[len(clouds):]
        assigned = [[] for _ in groups]
        unmatched = []
        for index, old in enumerate(clouds):
            scores = [max(_share(old_shapes[index], shape), _share(shape, old_shapes[index])) for shape in new_shapes]
            target = int(np.argmax(scores)) if scores else None
            if target is not None and scores[target] > 0 and _one_mode(old_shapes[index], new_shapes[target]):
                assigned[target].append(index)
            else:
                unmatched.append((index, old))
        combined = list(unmatched)
        for k, (group, predecessors) in enumerate(zip(groups, assigned)):
            merged += max(0, len(predecessors) - 1)
            order = min(predecessors) if predecessors else len(clouds) + k
            combined.append((order, np.concatenate([clouds[j] for j in predecessors] + [group])))
        clouds = [c for _, c in sorted(combined, key=lambda item: item[0])]
    return clouds, dict(boosts=used, rows=[len(c) for c in clouds], dropped=dropped, merged_clouds=merged)


def boost_region(directory, names, lower, upper, inflate, before, best, depth, quantile=None):
    """The Region of the boosts before iteration `before`, or (None, info) when there is none yet.

    A cloud without a shape is left out and counted (info["left_out"], its rows): a single
    verified point, or a sparse cloud while no cloud is well-sampled. It is not an error: a
    run whose boost has one verified point in a mode must be able to go on. (For a day a
    single point raised here, in the loop, from the saved boost files: at every restart.)
    info["small_clouds"] counts the sparse clouds that are in, with a borrowed shape.
    """
    clouds, info = boost_clouds(directory, names, before, best, depth, lower, upper)
    flags, borrowed = Region.shaped(clouds, lower, upper) if clouds else ([], [])
    kept = [c for c, flag in zip(clouds, flags) if flag]
    info.update(rows=[len(c) for c in kept], left_out=[len(c) for c, flag in zip(clouds, flags) if not flag],
                small_clouds=int(sum(b for b, flag in zip(borrowed, flags) if flag)))
    if not kept:
        return None, info
    region = Region.from_clouds(kept, lower, upper, inflate, best, quantile)
    info.update(reach=region.reach.tolist(), inflate=float(inflate))
    if quantile is not None:
        info.update(quantile=float(quantile))
    return region, info
