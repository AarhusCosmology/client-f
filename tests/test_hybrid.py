from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profiles.hybrid import ConcurrentLedger, NodeLedger, ascend, certify, scan_hybrid, HybridConfig
from profiles.evaluation import EvaluationLedger, CallBudgetExceeded


class Analytic:
    """BatchedSurrogate's interface for a numpy objective with gradient and Hessian."""

    def __init__(self, bounds, f, g, h):
        bounds = np.asarray(bounds, float)
        self.lo, self.hi = bounds[:, 0], bounds[:, 1]
        self.centre, self.half = (self.lo + self.hi) / 2, (self.hi - self.lo) / 2
        self.n = len(bounds)
        self.f, self.g, self.h = f, g, h
        self.counts, self.seconds = Counter(), Counter()

    def physical(self, U):
        return np.clip(self.centre + self.half * np.asarray(U, float), self.lo, self.hi)

    def box(self, X):
        return np.clip((np.asarray(X, float) - self.centre) / self.half, -1, 1)

    def value(self, U):
        return np.array([self.f(x) for x in self.physical(np.atleast_2d(U))])

    def value_grad(self, U):
        X = self.physical(np.atleast_2d(U))
        return np.array([self.f(x) for x in X]), np.array([self.g(x) for x in X]) * self.half

    def value_grad_hess(self, U):
        X = self.physical(np.atleast_2d(U))
        f, g = self.value_grad(U)
        return f, g, np.array([self.h(x) for x in X]) * np.outer(self.half, self.half)


def quadratic(P, mu):
    return (lambda x: -0.5 * (x - mu) @ P @ (x - mu), lambda x: -P @ (x - mu), lambda x: -P)


class TestAscent(unittest.TestCase):
    def test_a_singular_model_in_one_row_does_not_stop_the_batch(self):
        # numpy's batched solve raises for the whole batch when one matrix is singular; the
        # scan of a real study stopped that way. The singular row climbs on, the other rows
        # are what they are alone.
        P = np.array([[4.0, 3.0, 0.0], [3.0, 4.0, 1.0], [0.0, 1.0, 2.0]])
        mu = np.array([0.5, 0.2, -0.3])
        sb = Analytic([[-1, 1], [0, 1], [-1, 1]], *quadratic(P, mu))
        starts = sb.box([[0.9, 0.9, 0.9], [-0.8, 0.1, 0.7], [0.0, 0.5, 0.0]])
        fixed = np.zeros((3, 3), bool)
        B = np.stack([np.eye(3), np.zeros((3, 3)), 2 * np.eye(3)])
        with self.assertRaises(np.linalg.LinAlgError):
            np.linalg.solve(B, np.ones((3, 3, 1)))
        r = ascend(sb, starts, fixed, B=B, gtol=1e-8)
        self.assertTrue(np.all(np.isin(r.status, ["stationary", "stalled", "max_iter"])))
        self.assertTrue(np.all(np.isfinite(r.f)))
        np.testing.assert_allclose(sb.physical(r.U)[1], mu, atol=1e-5)
        for i in (0, 2):
            alone = ascend(sb, starts[i:i + 1], fixed[i:i + 1], B=B[i:i + 1], gtol=1e-8)
            np.testing.assert_allclose(r.U[i], alone.U[0], rtol=0, atol=1e-12)
            self.assertEqual(r.iterations[i], alone.iterations[0])

    def test_correlated_bounded_quadratic_with_fixed_coordinate(self):
        P = np.array([[4.0, 3.0, 0.0], [3.0, 4.0, 1.0], [0.0, 1.0, 2.0]])
        mu = np.array([0.5, 0.2, -0.3])
        sb = Analytic([[-1, 1], [0, 1], [-1, 1]], *quadratic(P, mu))
        start = sb.box([[0.9, 0.9, 0.9]])
        fixed = np.array([[True, False, False]])
        r = ascend(sb, start, fixed, gtol=1e-10)
        x = sb.physical(r.U)[0]
        # Fixing x0 = 0.9 pushes x1 below 0: the solution sits on the x1 = 0 bound, with
        # x2 then the unconstrained conditional optimum given (x0, x1).
        self.assertEqual(r.status[0], "stationary")
        self.assertAlmostEqual(x[0], 0.9, places=14)
        self.assertEqual(x[1], 0.0)
        d0, d1 = x[0] - mu[0], x[1] - mu[1]
        self.assertAlmostEqual(x[2], mu[2] - (P[2, 0] * d0 + P[2, 1] * d1) / P[2, 2], places=9)

    def test_reaches_gtol_below_the_objective_noise(self):
        # As at the SINU TT it10 network: log L ~ -3240, curvature 4e5 in box units and
        # value noise ~1e-10. Near the optimum the gain a correct Newton step makes (~1e-12)
        # is below the noise, so value tests reject it about half the time.
        rng = np.random.default_rng(7)
        Q = np.linalg.qr(rng.normal(size=(4, 4)))[0]
        P = Q @ np.diag([4e5, 3e3, 50.0, 2.0]) @ Q.T
        mu = np.array([0.1, -0.2, 0.3, 0.05])
        noise = lambda x: 1e-10 * np.sin(np.dot(x, [3.1e13, 1.7e13, 2.3e13, 4.1e13]))
        sb = Analytic([[-1, 1]] * 4, lambda x: -3240.0 - 0.5 * (x - mu) @ P @ (x - mu) + noise(x),
                      lambda x: -P @ (x - mu), lambda x: -P)
        starts = mu + Q[:, 0] * rng.uniform(1e-9, 1e-8, (20, 1))  # pg ~ 4e-4 .. 4e-3
        r = ascend(sb, sb.box(starts), np.zeros((20, 4), bool), gtol=1e-4, max_iter=50)
        self.assertTrue(np.all(r.status == "stationary"), list(r.status))
        self.assertTrue(np.all(r.pg <= 1e-4))
        self.assertTrue(np.all(r.f >= -3240.0 - 1e-9))

    def test_certify_escapes_ridge_saddle(self):
        # f = -0.5 (x1 - x0^2)^2 with x1 = 0.5 fixed: x0 = 0 is a stationary saddle.
        f = lambda x: -0.5 * (x[1] - x[0] ** 2) ** 2
        g = lambda x: np.array([2 * x[0] * (x[1] - x[0] ** 2), -(x[1] - x[0] ** 2)])
        h = lambda x: np.array([[2 * (x[1] - x[0] ** 2) - 4 * x[0] ** 2, 2 * x[0]], [2 * x[0], -1.0]])
        sb = Analytic([[-2, 2], [0, 1]], f, g, h)
        u = sb.box([[0.0, 0.5]])
        fixed = np.array([[False, True]])
        plain = ascend(sb, u, fixed)
        self.assertEqual(plain.status[0], "stationary")  # first order cannot see the saddle
        self.assertAlmostEqual(plain.f[0], -0.125)
        r, info = certify(sb, u, fixed)
        self.assertEqual(info["escapes"][0], 1)
        self.assertTrue(info["local_max"][0])
        # gtol = 1e-4 in box units and curvature 8 there bound the error by ~2.5e-5 in x0.
        self.assertAlmostEqual(abs(sb.physical(r.U)[0, 0]), np.sqrt(0.5), delta=2.5e-5)
        self.assertLessEqual(r.pg[0], 1e-4)
        self.assertAlmostEqual(r.f[0], 0.0, delta=1e-9)

    def test_certify_accepts_a_negligible_remaining_gain_along_a_stiff_direction(self):
        # Box curvatures 1e10 and 1 (the 58D CLOE-Gaussian networks span 9.5 decades). Row 0
        # sits 1e-13 off along the stiff direction: box gradient 1e-3 > gtol, remaining gain
        # 5e-17. Row 1 has a gradient of 2e-3 along the flat direction: remaining gain 2e-6.
        P = np.diag([1e10, 1.0])
        sb = Analytic([[-1, 1], [-1, 1]], lambda x: -0.5 * x @ P @ x, lambda x: -P @ x, lambda x: -P)
        u = np.array([[1e-13, 0.0], [0.0, 2e-3]])
        free = np.zeros((2, 2), bool)
        for gain_tol, expected in [(1e-6, [True, False]), (1e-5, [True, True]), (0.0, [False, False])]:
            r, info = certify(sb, u, free, gtol=1e-4, max_iter=0, newton_rounds=0, gain_tol=gain_tol)
            np.testing.assert_array_equal(r.U, u)  # classification only: points never move
            self.assertTrue(np.all(r.pg > 1e-4))
            np.testing.assert_allclose(info["remaining_gain"], [5e-17, 2e-6], rtol=1e-6)
            np.testing.assert_array_equal(info["stationary"], expected)
            np.testing.assert_array_equal(info["local_max"], expected)
        # A row that passes the gradient test is not re-evaluated; a saddle never passes by gain.
        r, info = certify(sb, np.zeros((1, 2)), np.zeros((1, 2), bool))
        self.assertTrue(info["stationary"][0] and np.isnan(info["remaining_gain"][0]))
        S = np.diag([1e10, -1.0])
        saddle = Analytic([[-1, 1], [-1, 1]], lambda x: -0.5 * x @ S @ x, lambda x: -S @ x, lambda x: -S)
        r, info = certify(saddle, np.array([[1e-13, 0.0]]), np.zeros((1, 2), bool), max_iter=0,
                          newton_rounds=0, rounds=0, gain_tol=1.0)
        self.assertFalse(info["local_max"][0])


class TestBasinHopping(unittest.TestCase):
    def test_walkers_leave_the_basin_of_the_starting_points(self):
        # Nuisance x1 has a shallow well at -0.7 (reached from every start) and a deeper
        # one at +0.7, separated by a barrier. x0 couples weakly.
        def f(x):
            w = x[1] ** 2 - 0.49
            return -4 * w ** 2 + 0.3 * x[1] - 0.5 * (x[0] - 0.1 * x[1]) ** 2

        def g(x):
            w = x[1] ** 2 - 0.49
            return np.array([-(x[0] - 0.1 * x[1]), -16 * w * x[1] + 0.3 + 0.1 * (x[0] - 0.1 * x[1])])

        def h(x):
            return np.array([[-1.0, 0.1], [0.1, -16 * (3 * x[1] ** 2 - 0.49) - 0.01]])

        bounds = np.array([[-2.0, 2.0], [-1.5, 1.5]])
        sb = Analytic(bounds, f, g, h)
        data = np.array([[0.0, -0.7, 0.0], [0.2, -0.75, 0.0], [-0.1, -0.65, 0.0]])
        md = SimpleNamespace(param_names=["a", "b"], bounds={"a": (-2.0, 2.0), "b": (-1.5, 1.5)})
        grid = np.linspace(-1, 1, 5)
        result, _ = scan_hybrid(None, md, data, ["a"], [grid], surrogate=sb,
                                hybrid=HybridConfig(walkers=6, seed=1), log=lambda *a, **k: None)
        xs = np.array(result["profiles"]["a"]["xopt"])
        self.assertTrue(np.all(xs[:, 1] > 0.5), xs[:, 1])
        self.assertGreater(result["best_x"][1], 0.5)
        self.assertTrue(result["optimization_quality"]["converged"])
        for v, x, logl in zip(grid, xs, result["profiles"]["a"]["logl"]):
            self.assertEqual(x[0], v)
            self.assertAlmostEqual(logl, f(x), places=12)
            self.assertLess(np.abs(g(x)[1]) * 1.5, 1e-4)  # CLiENT's gtol, in box units


class TestConcurrentTrueStage(unittest.TestCase):
    def test_ledger_counts_cache_budget_and_per_node_views(self):
        with tempfile.TemporaryDirectory() as d:
            L = EvaluationLedger(lambda x: -float(x @ x), [[-1, 1], [-1, 1]],
                                 journal=Path(d) / "calls.jsonl", max_calls=3)
            shared = ConcurrentLedger(L)
            a, b = NodeLedger(shared, "node/a"), NodeLedger(shared, "node/b")
            self.assertAlmostEqual(a([0.1, 0.2]), -0.05, places=15)
            self.assertEqual(b([0.1, 0.2]), a([0.1, 0.2]))
            self.assertEqual((a.counts["backend_calls"], b.counts["cache_hits"]), (1, 1))
            self.assertEqual(a([2.0, 0.0]), -np.inf)
            b([0.3, 0.0])
            b([0.4, 0.0])
            with self.assertRaises(CallBudgetExceeded):
                a([0.5, 0.0])
            shared.close()
            self.assertEqual(L.counts["backend_calls"], 3)
            self.assertEqual(L.by_stage["node/b"]["backend_calls"], 2)
            events = [json.loads(line) for line in L.journal.read_text().splitlines()]
            self.assertEqual([e["event"] for e in events].count("start"), 3)
            self.assertEqual([e["event"] for e in events].count("return"), 3)
            self.assertEqual({e["stage"] for e in events if e["event"] == "cache_hit"}, {"node/a", "node/b"})
            self.assertNotIn("_event", vars(L))  # the per-instance journal writer is removed

if __name__ == "__main__":
    unittest.main()
