"""Pair-conditioned training and independent pooled contrast validation.

Uniform coverage and contour-weighted anchors share one nuisance-space rule;
no named Gaussian/flat directions and no analytic target information are used.
"""

import numpy as np
from dataset.peak_samples import _fit_quadratic
from dataset.profile_augmentation import _truncated_gibbs


def surface_design(data, surface, bounds, count, broad_fraction, seed):
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, float)
    indices = surface["indices"]
    anchors = np.array(surface["xopt"]).reshape(-1, len(bounds))
    y = np.array(surface["logl"]).ravel()
    # A floor gives every design node support, including outside visible contours.
    delta = 2 * (surface["best_logl"] - y)
    weights = (
        1
        + 3 * np.exp(-(((delta - 2.30) / 1.5) ** 2))
        + 3 * np.exp(-(((delta - 6.18) / 2) ** 2))
    )
    fit = _fit_quadratic(data[:, :-1], data[:, -1], ridge_eig=1e-3)
    precision = -(fit["V"] * fit["w_neg"]) @ fit["V"].T
    mask = np.ones(len(bounds), bool)
    mask[indices] = False
    P = precision[np.ix_(mask, mask)]
    L = np.linalg.cholesky(np.linalg.inv(P))
    std = fit["std"][mask]
    rows = []
    records = []
    broad = int(round(count * broad_fraction))
    for k in range(count):
        pool = "broad" if k < broad else "targeted"
        node = int(
            rng.choice(
                len(anchors), p=None if pool == "broad" else weights / weights.sum()
            )
        )
        anchor = anchors[node]
        point = anchor.copy()
        # Broad pool scans the whole rectangular domain, targeted pool jitters
        # around its selected node. Reoptimize both at the new pair below.
        axes = [np.asarray(a) for a in surface["axes"]]
        for d, index in enumerate(indices):
            width = (axes[d][-1] - axes[d][0]) / (len(axes[d]) - 1)
            low = max(axes[d][0], anchor[index] - width / 2)
            high = min(axes[d][-1], anchor[index] + width / 2)
            point[index] = rng.uniform(low, high)
        rows.append(point)
        records.append(dict(pool=pool, node=node))
    return np.array(rows).reshape(count, len(bounds)), records, (P, L, std, mask)


def nuisance_draw(anchor, geometry, bounds, rng, mode):
    P, L, std, mask = geometry
    if not mask.any():
        return anchor.copy(), False
    deficit = float(rng.choice([0.05, 0.2, 0.5, 1.0]))
    temperature = float(rng.choice([0.25, 0.5, 1.0]))
    for _ in range(128):
        z = rng.normal(size=len(P))
        z *= (
            np.sqrt(temperature)
            if mode == "broad"
            else np.sqrt(2 * deficit) / np.linalg.norm(z)
        )
        point = anchor.copy()
        point[mask] += std * (L @ z)
        if np.all(point >= bounds[:, 0]) and np.all(point <= bounds[:, 1]):
            return point, False
    low = (bounds[mask, 0] - anchor[mask]) / std
    high = (bounds[mask, 1] - anchor[mask]) / std
    point = anchor.copy()
    point[mask] += std * _truncated_gibbs(
        P, low, high, temperature if mode == "broad" else 2 * deficit / len(P), rng
    )
    return point, True
