"""acquisition.boost.train_domain: the part of the box the boost networks are trained on.

A boost network is used only inside the region of the boosts (sampling/region.py): its
chains, the audit's searches, the evidence and the profiles all stay there. Its training set
reaches much farther: the initial design and the early tempered batches cover the whole box,
with values thousands to millions of log-units below the best. Those points do little in the
loss (msre weighs a point by its depth), but they set the network's output scale. In the axion
run (24 parameters, Planck TT at 15 sigma) the targets minus the quadratic background had a
spread of 52 280 log-units over the whole training set and 22 without the 11% of points more
than 1000 below the best, while the oscillations to be learned are a few log-units: two
millionths of the scale, which the optimiser's steps (fixed in the weights, so in units of the
scale) cannot resolve. They also demand corrections of up to +420 000 over the quadratic in
directions where the truth falls more slowly than it, which the network carries towards the
region's wall, where every false maximum of the late networks sat.

The domain is an ellipsoid around the points within T * ndim of the best, the points the
quadratic background is fitted to (model/network.py fit_background): their mean and covariance,
reaching `edge` times as far as the farthest of them. It is built once, at the first training
after the boosts' start, saved as boost/train_domain_it_N.npz (committed with that training)
and read by every later training: rebuilt each time it would grow with every far point the
boosts verify, as the region's reach did. Points outside it stay in the training set's files
and in every other use of the training set (the acquisition's density, the derived
emulator); only the network's training, its background, its scaling and the audit's
fine-tunes see the cut. In the axion run's control branch the region at edge 1.5 lay wholly
inside the domain at edge 1 at every network; the loop reports the share of the region
outside the domain at every boost.
"""
import json

import numpy as np

from run.run import iterations
from sampling.region import lengths, whitener


class TrainDomain:
    """mean and scale (per parameter, physical units), whiten (of the standardised rows) and
    far (the farthest row, in whitened units): rho(x) = |((x - mean) / scale) whiten| / far."""

    def __init__(self, mean, scale, whiten, far, depth, top, rows):
        self.mean = np.asarray(mean, float)
        self.scale = np.asarray(scale, float)
        self.whiten = np.asarray(whiten, float)
        self.far = float(far)
        self.depth = float(depth)
        self.top = float(top)
        self.rows = int(rows)

    @classmethod
    def build(cls, inputs, targets, depth):
        """The domain of a training set: its rows within `depth` of the best (fit_background's)."""
        x = np.asarray(inputs, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64).ravel()
        top = float(y.max())
        near = (top - y) < depth
        ndim = x.shape[1]
        if near.sum() <= 2 * ndim:
            raise ValueError(f"training domain: {int(near.sum())} rows within {depth:g} of the best, "
                             f"more than {2 * ndim} needed")
        mean = x[near].mean(axis=0)
        scale = x[near].std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        z = (x[near] - mean) / scale
        whiten = whitener(np.cov(z, rowvar=False), jitter=1e-12)
        far = float(lengths(z @ whiten).max())
        return cls(mean, scale, whiten, far, depth, top, int(near.sum()))

    def rho(self, x):
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return lengths(z @ self.whiten) / self.far

    def inside(self, x, edge):
        return self.rho(x) <= edge

    def kept(self, x, edge, name):
        """The rows of a training set a training uses. A domain is never rebuilt, and the rows
        that built it stay in the set (more than 2 ndim, inside at any edge of at least 1), so
        too few means that the file does not belong to this training set."""
        keep = self.inside(x, edge)
        needed = 2 * np.shape(x)[1]
        if keep.sum() < needed:
            raise ValueError(f"training domain {name}: {int(keep.sum())} of {len(keep)} points inside {edge:g} times "
                             f"its reach, at least {needed} needed. The file does not belong to this training set")
        return keep

    def save(self, path, **extra):
        np.savez(path, mean=self.mean, scale=self.scale, whiten=self.whiten, far=self.far,
                 depth=self.depth, top=self.top, rows=self.rows, extra=json.dumps(extra))

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            return cls(f["mean"], f["scale"], f["whiten"], float(f["far"]), float(f["depth"]),
                       float(f["top"]), int(f["rows"]))


def saved_domain(directory, iteration):
    """(path, TrainDomain) of the domain a training of network `iteration` uses: the earliest
    one saved in `directory` (RUN/boost) for a network up to `iteration`, or (None, None)."""
    found = {n: p for n, p in iterations(directory, "train_domain_it_*.npz").items() if n <= iteration}
    if not found:
        return None, None
    path = found[min(found)]
    return path, TrainDomain.load(path)


def region_outside(domain, edge, region, rng, n=4000, *, diagnostics=False):
    """Estimate the uniform-volume fraction of the box-clipped region outside the domain.

    Draw exactly n proposals from the box, or from full ellipsoids selected in
    proportion to volume (whichever has smaller total volume). For the latter,
    inverse overlap multiplicity corrects the mixture density. Reject box-exterior
    proposals, never shrink toward centres. The ratio of weighted sums estimates
    the union's volume fraction; it is not a posterior probability. With no hits
    it is unknown (NaN). Optional diagnostics report proposal count and Kish ESS,
    so tiny intersections are not mistaken for a well-resolved measurement.
    """
    import math

    n = int(n)
    if n <= 0:
        raise ValueError("region volume diagnostic needs a positive proposal count")
    ndim = len(region.lower)
    radius = region.reach * region.inflate
    log_volume = (ndim * np.log(radius) - np.linalg.slogdet(region.whiten)[1]
                  + 0.5 * ndim * np.log(np.pi) - math.lgamma(0.5 * ndim + 1))
    log_total = np.logaddexp.reduce(log_volume)
    ellipsoids = log_total < 0.0             # the unit box has volume one
    weights, outside = [], []
    for start in range(0, n, 512):
        size = min(512, n - start)
        if ellipsoids:
            labels = rng.choice(len(region), size=size, p=np.exp(log_volume - log_total))
            z = rng.normal(size=(size, ndim))
            z /= np.linalg.norm(z, axis=1)[:, None]
            z *= rng.uniform(size=size)[:, None] ** (1.0 / ndim)
            u = np.empty_like(z)
            for k in np.unique(labels):
                chosen = labels == k
                u[chosen] = region.means[k] + np.linalg.solve(
                    region.whiten[k].T, (radius[k] * z[chosen]).T).T
        else:
            u = rng.uniform(size=(size, ndim))
        x = region.lower + region.width * u
        multiplicity = (region.radii(x) <= region.inflate).sum(axis=0)
        valid = np.all((u >= 0) & (u <= 1), axis=1) & (multiplicity > 0)
        w = np.zeros(size)
        w[valid] = 1.0 / multiplicity[valid] if ellipsoids else 1.0
        weights.append(w)
        outside.append(~domain.inside(x, edge))
    weights, outside = np.concatenate(weights), np.concatenate(outside)
    total = weights.sum()
    fraction = float(weights @ outside / total) if total else float("nan")
    ess = float(total ** 2 / (weights @ weights)) if total else 0.0
    result = dict(fraction=fraction, proposals=n, hits=int(np.count_nonzero(weights)),
                  effective_samples=ess, proposal="ellipsoids" if ellipsoids else "box")
    return result if diagnostics else fraction


def region_coverage_message(coverage):
    """Human-readable coverage, with sampling limitations visible in the training log."""
    outside, ess = coverage["fraction"], coverage["effective_samples"]
    if not np.isfinite(outside):
        return ("Region volume coverage unknown: no proposals reached the box-clipped region; "
                "this does not establish containment in the training domain")
    message = (f"Estimated {outside:.1%} of the region volume (to its edge, inside the box) "
               f"lies outside the training domain; volume-weight ESS {ess:.0f} "
               f"from {coverage['proposals']} proposals")
    if ess < 200:
        return message + "; poorly resolved: too few effective samples to assess 1% coverage"
    if outside > 0.01:
        message += ("; WARNING: this part of the region is outside the training domain. "
                    "Review acquisition.boost.region and acquisition.boost.train_domain")
    return message
