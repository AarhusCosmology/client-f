"""hybrid_training_starts: walkers of the hybrid scan that start from training points near their node (opt-in)."""
from collections import Counter
from types import SimpleNamespace
import unittest

import numpy as np

from profiles.config import ProfileConfig
from profiles.hybrid import HybridConfig, hybrid_settings, scan_hybrid, training_starts

quiet = lambda *a, **k: None


class TwoBranches:
    """BatchedSurrogate's interface for the logarithm of two Gaussians in three coordinates.

    The profile in x0 has two branches. The central one (x1 = 0) holds the best point and is narrow
    in x0; the far one (x1 = 3.5) is 3 lower at its top and wide in x0, so that it is the higher one
    for |x0| > 1.24. The far branch is 0.05 wide in x1 and the two meet at x1 near 1.7, more than
    ten of the best point's widths away: steps with the size and shape of that maximum, at most five
    widths, do not get there.
    """

    means = np.array([[0.0, 0.0, 0.0], [0.0, 3.5, 0.0]])
    widths = np.array([[0.5, 0.15, 0.15], [3.0, 0.05, 0.15]])
    heights = np.array([0.0, -3.0])

    def __init__(self):
        bounds = np.array([[-4.0, 4.0]] * 3)
        self.bounds = bounds
        self.lo, self.hi = bounds[:, 0], bounds[:, 1]
        self.centre, self.half = (self.lo + self.hi) / 2, (self.hi - self.lo) / 2
        self.counts, self.seconds = Counter(), Counter()

    def physical(self, U):
        return np.clip(self.centre + self.half * np.asarray(U, float), self.lo, self.hi)

    def box(self, X):
        return np.clip((np.asarray(X, float) - self.centre) / self.half, -1, 1)

    def value_grad_hess(self, U):
        x = self.physical(np.atleast_2d(U))
        d = (x[:, None, :] - self.means) / self.widths ** 2                    # (rows, branches, 3): -gradient of each
        q = self.heights - 0.5 * np.sum((x[:, None, :] - self.means) ** 2 / self.widths ** 2, axis=2)
        top = q.max(axis=1, keepdims=True)
        f = top[:, 0] + np.log(np.sum(np.exp(q - top), axis=1))
        p = np.exp(q - f[:, None])
        g = -np.einsum("rm,rmi->ri", p, d)
        H = np.einsum("rm,rmi,rmj->rij", p, d, d) - np.einsum("ri,rj->rij", g, g)
        H -= np.einsum("rm,mi,ij->rij", p, 1 / self.widths ** 2, np.eye(3))
        return f, g * self.half, H * np.outer(self.half, self.half)

    def value_grad(self, U):
        f, g, _ = self.value_grad_hess(U)
        return f, g

    def value(self, U):
        return self.value_grad_hess(U)[0]

    def branches(self, x0):
        """The two branches' heights at x0: the profile is the higher one (to exp(-gap))."""
        x0 = np.asarray(x0, float)[:, None]
        return self.heights - 0.5 * (x0 - self.means[:, 0]) ** 2 / self.widths[:, 0] ** 2


def training_set(sb, n=300, seed=5):
    """Points on both branches with their values, as a run that sampled the posterior would hold."""
    rng = np.random.default_rng(seed)
    X = np.vstack([sb.means[m] + sb.widths[m] * rng.standard_normal((n, 3)) for m in (0, 1)])
    X = np.clip(X, -4, 4)
    return np.column_stack([X, sb.value(sb.box(X))])


GRID = np.linspace(-2.5, 2.5, 11)


def scan(starts):
    sb = TwoBranches()
    md = SimpleNamespace(param_names=["a", "b", "c"], bounds=dict(zip("abc", sb.bounds)))
    result, _ = scan_hybrid(None, md, training_set(sb), ["a"], [GRID],
                            hybrid=HybridConfig(walkers=4, training_starts=starts), surrogate=sb, log=quiet)
    return sb, result


class TrainingStartsTests(unittest.TestCase):
    def test_setting_validated(self):
        self.assertEqual(ProfileConfig(parameters=["a"]).hybrid_training_starts, 0)
        ProfileConfig(parameters=["a"], hybrid_training_starts=7).validate()
        for bad in (-1, 8, True, 2.0, "2", None):
            with self.assertRaises(ValueError):
                ProfileConfig(parameters=["a"], hybrid_training_starts=bad).validate()

    def test_setting_reaches_the_scan(self):
        settings = SimpleNamespace(hybrid_walkers=8, seed=3, gtol=1e-4, hybrid_hops=None)
        self.assertEqual(hybrid_settings(settings).training_starts, 0)                      # an older caller's settings
        self.assertEqual(hybrid_settings(SimpleNamespace(**vars(settings), hybrid_training_starts=3)).training_starts, 3)
        self.assertEqual(HybridConfig().training_starts, 0)

    def test_starts_are_the_best_points_near_the_node(self):
        sb = TwoBranches()
        data = training_set(sb)
        U, y = sb.box(data[:, :-1]), data[:, -1]
        nodes = sb.box(np.column_stack([GRID, np.zeros((len(GRID), 2))]))[:, 0]
        mask = np.zeros((1 + len(nodes), 3), bool)
        mask[1:, 0] = True
        fixed = np.zeros((1 + len(nodes), 3))
        fixed[1:, 0] = nodes
        starts = training_starts(sb, data, mask, fixed, 2)
        self.assertEqual(starts.shape, (1 + len(nodes), 2, 3))
        self.assertTrue(np.isnan(starts[0]).all())                                           # the anchor has no node
        spacing = nodes[1] - nodes[0]
        for k, u in enumerate(nodes, start=1):
            self.assertTrue(np.all(starts[k, :, 0] == u))                                    # moved to the node's value
            near = np.flatnonzero(np.abs(U[:, 0] - u) <= spacing)
            if len(near) >= 8:
                best = near[np.argsort(-y[near], kind="stable")[:2]]
                np.testing.assert_array_equal(starts[k, :, 1:], U[best, 1:])
        # a node with few points near it takes the nearest ones, and a label without a value is never a start
        data[:, -1] = np.where(np.arange(len(data)) % 2 == 0, data[:, -1], -np.inf)
        far = np.zeros((2, 3))
        far[1, 0] = 0.99
        mask = np.zeros((2, 3), bool)
        mask[1, 0] = True
        lone = training_starts(sb, data, mask, far, 2)
        self.assertTrue(np.isfinite(lone[1]).all())
        rows = [int(np.flatnonzero((U[:, 1:] == lone[1, j, 1:]).all(axis=1))[0]) for j in range(2)]
        self.assertTrue(all(r % 2 == 0 for r in rows))

    def test_far_branch_found_only_with_training_starts(self):
        sb, standard = scan(0)
        _, started = scan(2)
        heights = sb.branches(GRID)
        profile = heights.max(axis=1)
        far = heights[:, 1] > heights[:, 0] + 1                                              # the far branch is the profile here
        self.assertEqual(int(far.sum()), 6)
        missed = standard["profiles"]["a"]["logl"]
        found = started["profiles"]["a"]["logl"]
        # without the setting the scan stays on the central branch: every one of those nodes too low
        self.assertTrue(np.all(missed[far] < profile[far] - 1))
        np.testing.assert_allclose(missed[far], heights[far, 0], atol=1e-3)
        # with it, the profile at every node, and the points on the far branch
        np.testing.assert_allclose(found, profile, atol=1e-3)
        np.testing.assert_allclose(started["profiles"]["a"]["xopt"][far, 1], 3.5, atol=1e-3)
        np.testing.assert_allclose(found[~far], missed[~far], atol=1e-6)
        self.assertEqual(started["optimization_quality"]["configuration"]["training_starts"], 2)
        self.assertEqual(standard["optimization_quality"]["configuration"]["training_starts"], 0)

    def test_without_the_setting_the_scan_is_unchanged(self):
        # training_starts() is not called and the walkers' starts are those of the scan as it was
        from unittest import mock

        with mock.patch("profiles.hybrid.training_starts", side_effect=AssertionError("called")):
            _, first = scan(0)
        _, second = scan(0)
        np.testing.assert_array_equal(first["profiles"]["a"]["logl"], second["profiles"]["a"]["logl"])


if __name__ == "__main__":
    unittest.main()
