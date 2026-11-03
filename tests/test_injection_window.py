"""The injection's quadratic fitted to the points near the best only (dataset.peak_samples.INJECT_WINDOW;
window=None fits every point).

Analytic targets, no Boltzmann code. docs/injection_window.md has the measurements on real runs.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config import AcquisitionConfig  # noqa: E402
from dataset.peak_samples import gaussian_peak_samples, near_best  # noqa: E402

D, T, WIDTH = 6, 7.0, 0.35
CENTRE = np.array([1.0, -0.5, 0.3, 0.0, 0.8, -1.2])
LOWER, UPPER = -10.0 * np.ones(D), 10.0 * np.ones(D)


def target(x):
    """A Gaussian of width WIDTH with its peak at CENTRE (value 0) in a box sixty widths
    across, and one-sided quartic walls in three coordinates that no quadratic over the whole
    box can follow. Fitted to every point of training_set(), the injection's surface has a
    residual of 22 000, its vertex 6 or more from CENTRE and a vertex value of 21 000: what
    the real CMB runs show."""
    r = np.asarray(x, float) - CENTRE
    return -0.5 * np.sum(r ** 2, axis=1) / WIDTH ** 2 - 30.0 * np.sum(np.maximum(r[:, :3] - 1.5, 0.0) ** 4, axis=1)


def training_set(seed=0, near=600):
    """A design over the whole box and points near the peak, as a run has after a few iterations."""
    rng = np.random.default_rng(seed)
    x = np.vstack([rng.uniform(LOWER, UPPER, size=(2000, D)), CENTRE + 0.8 * rng.normal(size=(near, D))])
    x = np.clip(x, LOWER, UPPER)
    return x, target(x)


class Selection(unittest.TestCase):
    def test_within_the_depth(self):
        y = -np.arange(100.0)
        pick = near_best(y, 9.5, 5)
        self.assertEqual(pick.sum(), 10)
        self.assertTrue(pick[:10].all())

    def test_never_fewer_than_the_minimum(self):
        y = -np.arange(100.0) ** 2
        pick = near_best(y, 3.0, 20)                      # only two points are within 3
        self.assertEqual(pick.sum(), 20)
        self.assertTrue(pick[:20].all())

    def test_all_when_there_are_fewer_than_the_minimum(self):
        self.assertTrue(near_best(-np.arange(7.0), 1.0, 20).all())


class Window(unittest.TestCase):
    def test_without_the_key_every_point_is_fitted(self):
        x, y = training_set()
        new, info = gaussian_peak_samples(x, y, 50, temperature=T, bounds=np.c_[LOWER, UPPER], seed=3)
        same, _ = gaussian_peak_samples(x, y, 50, temperature=T, bounds=np.c_[LOWER, UPPER], seed=3, window=None)
        np.testing.assert_array_equal(new, same)
        self.assertIsNone(info["window_depth"])
        self.assertEqual((info["n_fit"], info["n_all"], info["n_capped"]), (len(y), len(y), 0))

    def test_the_restricted_fit_lands_at_the_peak_where_the_fit_to_everything_does_not(self):
        x, y = training_set()
        law = 0.5 * T * D
        everything, info_all = gaussian_peak_samples(x, y, 200, temperature=T, bounds=np.c_[LOWER, UPPER], seed=3)
        near, info = gaussian_peak_samples(x, y, 200, temperature=T, bounds=np.c_[LOWER, UPPER], seed=3, window=16)
        deep, close = -target(everything), -target(near)          # depth below the true peak (0)
        self.assertGreater(np.median(deep), 5 * law)              # the fit to everything is nowhere near
        self.assertLess(np.median(close), 2 * law)                # the restricted one is where a T chain would be
        self.assertGreater(np.median(close), 0.3 * law)           # and not piled on the peak
        self.assertLess(np.linalg.norm(info["x_star"] - CENTRE), 0.6)
        self.assertGreater(np.linalg.norm(info_all["x_star"] - CENTRE), 2.0)
        self.assertLess(info["n_fit"], info["n_all"])
        self.assertAlmostEqual(info["window_depth"], 16 * law)
        self.assertLess(abs(info["model_value"]), 5.0)            # the surface's own peak value against the true 0
        self.assertGreater(abs(info_all["model_value"]), 100.0)

    def test_too_few_points_near_the_best_takes_the_best_three_per_coefficient(self):
        x, y = training_set(near=0)                               # a design only: nothing near the peak
        ncoef = 1 + D + D * (D + 1) // 2
        _, info = gaussian_peak_samples(x, y, 50, temperature=T, bounds=np.c_[LOWER, UPPER], seed=3, window=0.01)
        self.assertEqual(info["n_fit"], 3 * ncoef)

    def test_the_reported_depth_is_that_of_the_points_drawn(self):
        # Where the window caps a width the points lie nearer the vertex than the tempered law
        # says. The log gave the law's depth all the same: 20 where the points were 10 down.
        rng = np.random.default_rng(5)
        d, temperature = 6, 7.0
        a = rng.normal(size=(d, d))
        precision, centre = a @ a.T / d + 0.5 * np.eye(d), rng.uniform(-1, 1, d)
        surface = lambda x: -0.5 * np.einsum("ni,ij,nj->n", x - centre, precision, x - centre)
        x = rng.uniform(-6, 6, (3000, d))
        bounds = np.c_[np.full(d, -6.0), np.full(d, 6.0)]
        for window, capped in ((None, 0), (0.5, d)):
            new, info = gaussian_peak_samples(x, surface(x), 400, temperature=temperature, bounds=bounds, seed=1, window=window)
            self.assertEqual((info["n_capped"], info["n_clipped"]), (capped, 0))
            # an exact quadratic: the fit is the surface, so its drop at a point is the true one
            np.testing.assert_allclose(info["deficit"], -surface(new), atol=1e-6)
            np.testing.assert_allclose(info["predicted_loglkl"], surface(new), atol=1e-6)
        self.assertLess(info["deficit"].mean(), 0.6 * info["deficit_mean_expected"])     # capped: well inside the law

    def test_the_width_is_capped_by_the_spread_of_the_fitted_points(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(-3, 3, size=(1500, 4))
        y = -0.5 * np.sum(x[:, :3] ** 2, axis=1)                  # flat in the last coordinate: no curvature to read a width from
        free, _ = gaussian_peak_samples(x, y, 400, temperature=T, seed=5)
        capped, info = gaussian_peak_samples(x, y, 400, temperature=T, seed=5, window=1e6)   # every point, but capped
        self.assertGreaterEqual(info["n_capped"], 1)
        self.assertEqual(info["n_fit"], info["n_all"])
        self.assertLess(capped[:, 3].std(), 1.3 * x[:, 3].std())
        self.assertGreater(free[:, 3].std(), 5 * x[:, 3].std())


class RefusedPoints(unittest.TestCase):
    """Injected points where the backend returns no finite value (a Boltzmann code that
    refuses part of the box) are left out; the loop must survive all of them failing."""

    class Likelihood:
        n_derived = 0

        def __init__(self, refuse):
            self.refuse = refuse
            self.prior_bounds = {f"p{i}": (LOWER[i], UPPER[i]) for i in range(D)}

        def evaluate_with_derived(self, x):
            return np.array([-np.inf if self.refuse(np.asarray(x)) else float(target(np.atleast_2d(x))[0])])

        def split_evaluations(self, values):
            values = np.asarray(values, dtype=float).reshape(-1, 1)
            return values[:, 0], values[:, 1:]

    def draw(self, refuse):
        import contextlib
        import io
        from types import SimpleNamespace

        import client

        cfg = SimpleNamespace(acquisition=SimpleNamespace(target_temperature=T))
        x, y = training_set()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = client.draw_peak_samples(cfg, self.Likelihood(refuse), x, y, True, n_draw=40, seed=3)
        return result, out.getvalue()

    def test_every_injected_point_refused(self):
        (inputs, targets, derived), log = self.draw(lambda x: True)
        self.assertEqual((inputs.shape, targets.shape, derived.shape), ((0, D), (0,), (0, 0)))
        self.assertIn("40 of the 40 injected points have no finite log-likelihood", log)
        self.assertNotIn("injected log-likelihood range", log)

    def test_some_refused(self):
        (inputs, targets, _), log = self.draw(lambda x: x[0] > CENTRE[0])
        self.assertTrue(0 < len(targets) < 40)
        self.assertTrue(np.isfinite(targets).all())
        self.assertTrue((inputs[:, 0] <= CENTRE[0]).all())
        self.assertIn("injected log-likelihood range", log)
        self.assertIn(f"{40 - len(targets)} of the 40 injected points have no finite log-likelihood", log)

    def test_none_refused_prints_what_it_did_before(self):
        (_, targets, _), log = self.draw(lambda x: False)
        self.assertEqual(len(targets), 40)
        self.assertNotIn("no finite log-likelihood", log)


class Configuration(unittest.TestCase):
    BASE = dict(n_append=10, n_neighbors=5, target_temperature=7.0, pool_factor=5)

    def test_the_window_is_not_a_setting(self):
        from dataset.peak_samples import INJECT_WINDOW
        self.assertEqual(INJECT_WINDOW, 16.0)
        with self.assertRaises(ValueError):
            AcquisitionConfig.from_dict(dict(self.BASE, inject_window=16))


if __name__ == "__main__":
    unittest.main()
