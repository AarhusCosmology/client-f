"""ln Z over a box with the BEST nested sampler, and the diagnostics that go with it."""

import os
import sys
import time

import numpy as np

# Posterior mass within this fraction of the box width of a face counts as touching it.
EDGE_FRACTION = 0.02
EDGE_WARNING = 1e-3


def import_best(best_path=None):
    """BEST's NestedSampler, from `best_path`, $BEST_PATH, or an installed package."""
    for path in (best_path, os.environ.get("BEST_PATH")):
        if path and path not in sys.path:
            sys.path.insert(0, str(path))
    try:
        from best import NestedSampler
    except ImportError as exc:
        raise ImportError(
            "could not import BEST (https://github.com/AndreasNygaard/best-inference): "
            "install it, set BEST_PATH, or pass --best-path to its checkout") from exc
    return NestedSampler


def run_nested(log_prob, lower, upper, n_live=2048, n_live_updates=10, tolerance=1e-3,
               n_max_iter=100000, seed=42, best_path=None, verbose=True):
    """Nested sampling of a uniform prior on [lower, upper] times exp(log_prob).

    `log_prob` maps a (n, d) float32 tensor to (n,) log-likelihood values. BEST adds ln Z up
    in single precision, so the values near the peak should be near zero: at -15656, where
    single-precision numbers are 1e-3 apart, part of every addition is rounded away (a run
    taken to the end came out 0.46 too low with 8000 live points on a 4-parameter Gaussian).
    The command subtracts a reference for that reason.

    `tolerance` is the fraction of the evidence left in the live points at which the run
    stops: the live points' estimate of what remains, over what the dead points have added
    up. Returns a dict with ln Z_box, its error, the information H, the weighted samples,
    the live points' share of the evidence (`live_fraction`) and timing.
    """
    import tensorflow as tf

    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance is a fraction of the evidence: above 0 and below 1")
    if n_max_iter < 1:
        raise ValueError("n_max_iter must be at least 1")
    NestedSampler = import_best(best_path)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    tf.keras.utils.set_random_seed(seed)
    device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    t0 = time.monotonic()
    with tf.device(device):
        # BEST's own tolerance is compared with ln(Z_live / Z_dead). Any positive value, its
        # default 1e-3 among them, stops the run with half the evidence still in the live
        # points, which sigma = sqrt(H / n_live) knows nothing about.
        sampler = NestedSampler(log_prob, (lower.tolist(), upper.tolist()), n_live,
                                n_live_updates=n_live_updates, tolerance=float(np.log(tolerance)),
                                n_max_iter=n_max_iter, seed=seed)
        # BEST reads its loop variable after a loop over n_max_iter // update_interval chunks,
        # and with no chunk there is none (an UnboundLocalError below 50 iterations).
        result = sampler.run(verbose=verbose, update_interval=min(50, n_max_iter), display_param_idx=0)
    seconds = time.monotonic() - t0
    points = np.asarray(result.all_points, float)
    weights = np.asarray(result.posterior_weights, float)
    n_dead = len(points) - n_live
    return dict(lnZ_box=float(np.asarray(result.logZ)), sigma=float(np.asarray(result.sigma_logZ)),
                H=float(np.asarray(result.KLDivergence)), points=points, weights=weights,
                logL=np.asarray(result.all_logL, float), n_dead=n_dead,
                iteration_cap_reached=n_dead >= n_max_iter * n_live_updates,
                live_fraction=float(weights[n_dead:].sum()), seconds=seconds, device=device)


def edge_mass(points, weights, lower, upper, fraction=EDGE_FRACTION):
    """Posterior mass within `fraction` of the box width of each face: (n, 2) array."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    width = upper - lower
    near_lower = points < (lower + fraction * width)
    near_upper = points > (upper - fraction * width)
    return np.stack([weights @ near_lower, weights @ near_upper], axis=1)
