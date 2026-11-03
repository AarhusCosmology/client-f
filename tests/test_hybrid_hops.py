"""hybrid_hops: the limit on the hybrid scan's hops, as a profile setting (opt-in)."""
from collections import Counter
from types import SimpleNamespace
import unittest

import numpy as np

from profiles.config import ProfileConfig
from profiles.hybrid import HybridConfig, hybrid_settings
from profiles.hybrid_surface import scan_surface_hybrid

quiet = lambda *a, **k: None


class Rugged:
    """BatchedSurrogate's interface for a Gaussian with ripples in the free coordinates whose
    phase moves with the fixed pair (x0, x1): every node has many local maxima of unequal
    height, and a node's best ripple is not its neighbour's. The 5 x 5 surface below settles at
    hop 4 to 12 depending on the seed (6 or later for 70 of 100 seeds), where the axion network's
    2500-node surfaces had not settled by 12. Which hop it is moves with the platform's rounding:
    seed 42 settles at 12 on macOS and at 10 on the Linux runners, and relative changes of 1e-13
    to 1e-11 in the ripples' amplitude move it anywhere from 4 to 13."""

    def __init__(self, n=6, a=1.0, w=8.0, s=3.0, p=5.0):
        bounds = np.array([[-4.0, 4.0]] * n)
        self.bounds = bounds
        self.lo, self.hi = bounds[:, 0], bounds[:, 1]
        self.centre, self.half = (self.lo + self.hi) / 2, (self.hi - self.lo) / 2
        self.a, self.w, self.s, self.d = a, w, s, p * np.array([1.7, 2.3])
        self.counts, self.seconds = Counter(), Counter()

    def physical(self, U):
        return np.clip(self.centre + self.half * np.asarray(U, float), self.lo, self.hi)

    def box(self, X):
        return np.clip((np.asarray(X, float) - self.centre) / self.half, -1, 1)

    def value_grad_hess(self, U):
        x = self.physical(np.atleast_2d(U))
        r, n = x.shape
        z = self.w * x[:, 2:] + (x[:, :2] @ self.d)[:, None]
        f = -0.5 * np.sum(x ** 2, 1) / self.s ** 2 + self.a * np.sum(np.cos(z), 1)
        sin, cos = np.sin(z), np.cos(z)
        g = -x / self.s ** 2
        g[:, 2:] -= self.a * self.w * sin
        g[:, :2] -= self.a * np.sum(sin, 1)[:, None] * self.d
        H = np.zeros((r, n, n))
        H[:, np.arange(n), np.arange(n)] = -1 / self.s ** 2
        c = -self.a * cos
        H[:, 2:, 2:] += np.einsum("rk,kl->rkl", c * self.w ** 2, np.eye(n - 2))
        H[:, :2, 2:] += np.einsum("rk,i->rik", c * self.w, self.d)
        H[:, 2:, :2] += np.einsum("rk,i->rki", c * self.w, self.d)
        H[:, :2, :2] += np.einsum("r,i,j->rij", c.sum(1), self.d, self.d)
        return f, g * self.half, H * np.outer(self.half, self.half)

    def value_grad(self, U):
        f, g, _ = self.value_grad_hess(U)
        return f, g

    def value(self, U):
        return self.value_grad_hess(U)[0]


def scan(hops, seed=42):
    sb = Rugged()
    rng = np.random.default_rng(3)
    X = rng.uniform(-4, 4, (16, len(sb.bounds)))
    data = np.column_stack([X, sb.value(sb.box(X))])
    axes = [np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)]
    config = SimpleNamespace(hybrid_walkers=2, seed=seed, gtol=1e-4, hybrid_hops=hops)
    return scan_surface_hybrid(None, sb.bounds, data, [0, 1], axes, config, surrogate=sb, log=quiet)


class HybridHopsTests(unittest.TestCase):
    def test_setting_validated(self):
        self.assertIsNone(ProfileConfig(parameters=["a"]).hybrid_hops)
        ProfileConfig(parameters=["a"], hybrid_hops=20).validate()
        for bad in (3, True, 12.0, "20"):
            with self.assertRaises(ValueError):
                ProfileConfig(parameters=["a"], hybrid_hops=bad).validate()

    def test_unset_is_the_scans_own_configuration(self):
        settings = SimpleNamespace(hybrid_walkers=8, seed=42, gtol=1e-4)
        self.assertEqual(hybrid_settings(settings), HybridConfig(walkers=8, seed=42, gtol=1e-4))
        self.assertEqual(hybrid_settings(SimpleNamespace(**vars(settings), hybrid_hops=None)),
                         HybridConfig(walkers=8, seed=42, gtol=1e-4))
        self.assertEqual(hybrid_settings(SimpleNamespace(**vars(settings), hybrid_hops=20)).hops_max, 20)

    def test_limit_stops_an_unsettled_scan_and_a_larger_one_changes_nothing_once_settled(self):
        # The hop a scan settles at moves with the platform's rounding (see Rugged), so a generous
        # limit finds it and the other limits are set relative to it. The scan stops after
        # `patience` hops without a gain, so `patience` hops before it settles a node still
        # improves; that limit must be one a study can set (hybrid_hops >= 4). About a third of
        # the seeds settle earlier, so the test takes the first seed from 42 that settles later.
        patience = HybridConfig().patience
        for seed in range(42, 62):
            long = scan(40, seed)
            settled = long["quality"]["hops"]
            if settled - patience >= 4:
                break
        else:
            self.fail("no seed from 42 to 61 settles after hop 5")
        self.assertLess(settled, 40)
        self.assertTrue(long["quality"]["converged"])
        self.assertEqual(long["quality"]["configuration"]["hops_max"], 40)
        unsettled = settled - patience
        # Too few hops: a node still improves at the last hop, the scan has not converged
        # (the check that stopped the axion studies at 12).
        short = scan(unsettled, seed)
        self.assertEqual(short["quality"]["hops"], unsettled)
        self.assertFalse(short["quality"]["profiles_stable"])
        self.assertFalse(short["quality"]["converged"])
        self.assertTrue(short["quality"]["all_selected_points_stationary"])
        # A limit at the hop it settles at: a larger one changes nothing, number for number.
        exact = scan(settled, seed)
        self.assertEqual(exact["quality"]["hops"], settled)
        self.assertTrue(exact["quality"]["converged"])
        np.testing.assert_array_equal(long["logl"], exact["logl"])
        np.testing.assert_array_equal(long["xopt"], exact["xopt"])


if __name__ == "__main__":
    unittest.main()
