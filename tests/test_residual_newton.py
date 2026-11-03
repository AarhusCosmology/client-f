from collections import Counter
from concurrent.futures import Future
import json
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest
import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profiles.evaluation import CallBudgetExceeded, EvaluationLedger
from profiles.residual_newton import NewtonConfig, NewtonRefiner, refine_profiles_newton


class Quadratic:
    """BatchedSurrogate's interface for s(x) = -0.5 (x - mu)^T P (x - mu)."""

    def __init__(self, bounds, P, mu):
        bounds = np.asarray(bounds, float)
        self.lo, self.hi = bounds[:, 0], bounds[:, 1]
        self.centre, self.half = (self.lo + self.hi) / 2, (self.hi - self.lo) / 2
        self.n, self.P, self.mu = len(bounds), P, mu
        self.counts, self.seconds = Counter(), Counter()

    def physical(self, U):
        return np.clip(self.centre + self.half * np.asarray(U, float), self.lo, self.hi)

    def box(self, X):
        return np.clip((np.asarray(X, float) - self.centre) / self.half, -1, 1)

    def value(self, U):
        d = self.physical(np.atleast_2d(U)) - self.mu
        return -0.5 * np.einsum("ri,ij,rj->r", d, self.P, d)

    def value_grad(self, U):
        d = self.physical(np.atleast_2d(U)) - self.mu
        return -0.5 * np.einsum("ri,ij,rj->r", d, self.P, d), -(d @ self.P) * self.half

    def value_grad_hess(self, U):
        f, g = self.value_grad(U)
        H = -self.P * np.outer(self.half, self.half)
        return f, g, np.broadcast_to(H, (len(f),) + H.shape).copy()


def study(curvature_error=0.0, noise=0.0, shift=None, n=6, seed=3):
    """Correlated 6D Gaussian truth; the surrogate has a shifted optimum and, optionally, a
    curvature wrong by `curvature_error` along its flattest direction. Three profiles."""
    rng = np.random.default_rng(seed)
    V = np.linalg.qr(rng.normal(size=(n, n)))[0]
    lam = np.geomspace(1, 50, n)
    P_true = V @ np.diag(lam) @ V.T
    lam_s = lam.copy()
    lam_s[0] *= 1 + curvature_error
    P_surr = V @ np.diag(lam_s) @ V.T
    mu_true = np.full(n, 0.2)
    mu_surr = mu_true + (np.array([0.0, 0.3, -0.2, 0.25, -0.3, 0.2]) if shift is None else shift)
    bounds = np.array([[-3.0, 3.0]] * n)
    Sigma_s, Sigma_t = np.linalg.inv(P_surr), np.linalg.inv(P_true)
    names = [f"p{i}" for i in range(n)]
    grid = np.linspace(-0.4, 0.8, 5)
    source = dict(names=names[:3], full_parameter_names=names, best_x=mu_surr, profiles={
        names[i]: dict(grid=grid, xopt=[mu_surr + Sigma_s[:, i] / Sigma_s[i, i] * (v - mu_surr[i]) for v in grid])
        for i in range(3)})
    exact = {names[i]: -0.5 * (grid - mu_true[i]) ** 2 / Sigma_t[i, i] for i in range(3)}

    def truth(x):
        x = np.asarray(x, float)
        eps = np.random.default_rng(zlib.crc32(x.astype("<f8").tobytes())).standard_normal() if noise else 0.0
        return float(-0.5 * (x - mu_true) @ P_true @ (x - mu_true)) + noise * eps

    return Quadratic(bounds, P_surr, mu_surr), truth, source, exact, bounds


class Pool:
    """A backend pool for WorkflowLedger, with jittered timing."""

    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        time.sleep(random.random() * 1e-3)
        return self.f(np.asarray(x))

    def submit(self, x):
        future = Future()
        future.set_result(self(x))
        return future


def run(sb, truth, source, bounds, d, workers=1, config=None, max_calls=20000, chain_length=None):
    ledger = EvaluationLedger(Pool(truth), bounds, journal=Path(d) / "calls.jsonl", max_calls=max_calls)
    refine_profiles_newton(None, source, bounds, ledger, d, config, workers=workers, surrogate=sb,
                           log=lambda *a, **k: None, chain_length=chain_length)
    return json.loads((Path(d) / "true_profiles.json").read_text()), ledger


def refusing(truth, *points):
    """`truth` that returns -inf at the given points, as the cluster backend does for an error."""
    def objective(x):
        return -np.inf if any(np.allclose(x, p, atol=1e-12, rtol=0) for p in points) else truth(x)
    return objective


class RefusedStart(unittest.TestCase):
    def test_a_refused_start_is_recorded_and_the_rest_of_the_study_goes_on(self):
        sb, truth, source, exact, bounds = study()
        start = np.clip(np.array(source["profiles"]["p0"]["xopt"][4], float), bounds[:, 0], bounds[:, 1])
        for workers in (1, 3):
            with tempfile.TemporaryDirectory() as d:
                result, _ = run(sb, refusing(truth, start), source, bounds, d, workers=workers)
                record = result["profiles"]["p0"]
                self.assertEqual((record["logl"][4], record["xopt"][4]), (None, None))
                self.assertEqual(record["optimization"][4]["reason"], "refused start")
                self.assertEqual(record["optimization"][4]["flag"], 1)
                np.testing.assert_allclose(record["logl"][:4], exact["p0"][:4] - 0.0, atol=2e-3)
                for name in ("p1", "p2"):
                    np.testing.assert_allclose(result["profiles"][name]["logl"], exact[name], atol=2e-3)
                self.assertIsNotNone(result["best_logl"])
                direct = json.loads((Path(d) / "direct_profiles.json").read_text())["profiles"]["p0"]["logl"]
                self.assertIsNone(direct[4])
                # A resume reads the node as it is, and asks the likelihood nothing.
                ledger = EvaluationLedger(Pool(refusing(truth, start)), bounds, journal=Path(d) / "resumed.jsonl", max_calls=20000)
                refine_profiles_newton(None, source, bounds, ledger, d, None, workers=workers, surrogate=sb, log=lambda *a, **k: None)
                again = json.loads((Path(d) / "true_profiles.json").read_text())
                self.assertEqual(again["profiles"]["p0"]["logl"], record["logl"])
                self.assertEqual(ledger.counts["backend_calls"], 0)

    def test_the_report_of_a_study_with_a_refused_node(self):
        from profiles.io import save
        from profiles.reporting import write_outputs

        sb, truth, source, exact, bounds = study()
        start = np.clip(np.array(source["profiles"]["p0"]["xopt"][4], float), bounds[:, 0], bounds[:, 1])
        with tempfile.TemporaryDirectory() as d:
            run(sb, refusing(truth, start), source, bounds, d)
            names = source["names"]
            save(Path(d) / "surrogate_profiles.json", dict(source, best_logl=0.0, profiles={
                n: dict(source["profiles"][n], logl=exact[n]) for n in names}))
            save(Path(d) / "ranges.json", {n: dict(clipped=False, kind="explicit physical range") for n in names})
            save(Path(d) / "counts.json", dict(backend_attempts=0))
            write_outputs(d)
            warnings = json.loads((Path(d) / "diagnostics.json").read_text())["warnings"]
            self.assertTrue(any("p0: 1 node(s) without a true value" in w for w in warnings), warnings)
            self.assertFalse(any("p1: " in w and "without a true value" in w for w in warnings), warnings)
            self.assertTrue((Path(d) / "profiles.png").exists() and (Path(d) / "profiles.csv").exists())
            self.assertIn("p0", json.loads((Path(d) / "intervals_grid.json").read_text())["parameters"])

    def test_crossings_read_over_the_nodes_with_a_value(self):
        from profiles.metric import crossings
        from profiles.reporting import crossings_with_values

        grid, values = np.array([0.0, 1.0, 2.0, 3.0]), np.array([3.0, 0.0, 3.0, 6.0])
        np.testing.assert_array_equal(crossings_with_values(grid, values, 1.0), crossings(grid, values, 1.0))
        np.testing.assert_allclose(crossings_with_values(grid, np.array([3.0, 0.0, np.nan, 6.0]), 1.0), [2 / 3, 1 + 1 / 6 * 2])
        self.assertEqual(crossings_with_values(grid, np.array([np.nan, np.nan, 1.0, np.nan]), 1.0), [])

    def test_a_refused_seed_leaves_its_chains_a_cold_start(self):
        sb, truth, source, exact, bounds = study()
        grid = np.asarray(source["profiles"]["p0"]["grid"])
        seed = int(np.argmin(np.abs(grid - np.asarray(source["best_x"])[0])))
        start = np.clip(np.array(source["profiles"]["p0"]["xopt"][seed], float), bounds[:, 0], bounds[:, 1])
        with tempfile.TemporaryDirectory() as d:
            result, _ = run(sb, refusing(truth, start), source, bounds, d)
            logl = result["profiles"]["p0"]["logl"]
            self.assertIsNone(logl[seed])
            others = [j for j in range(len(grid)) if j != seed]
            np.testing.assert_allclose(np.array(logl, float)[others], exact["p0"][others], atol=2e-3)

    def test_a_study_with_every_start_refused_stops_and_says_so(self):
        sb, truth, source, exact, bounds = study()
        with tempfile.TemporaryDirectory() as d, self.assertRaisesRegex(RuntimeError, "refuses the start of every node"):
            run(sb, lambda x: -np.inf, source, bounds, d)

    def test_a_refused_point_is_called_once(self):
        from profiles.hybrid import ConcurrentLedger, NodeLedger

        calls = []
        def truth(x):
            calls.append(tuple(x))
            return -np.inf if x[0] > 0 else 0.0
        bounds = np.array([[-1.0, 1.0]] * 2)
        with tempfile.TemporaryDirectory() as d:
            shared = ConcurrentLedger(EvaluationLedger(Pool(truth), bounds, journal=Path(d) / "calls.jsonl", max_calls=10), workers=2)
            try:
                node = NodeLedger(shared, "node")
                refused, finite = [0.5, 0.0], [-0.5, 0.0]
                self.assertEqual(node.missing_calls([refused]), 1)
                np.testing.assert_array_equal([node(refused), node(refused), node(finite)], [-np.inf, -np.inf, 0.0])
                self.assertEqual(len(calls), 2)
                self.assertEqual(node.missing_calls([refused, finite]), 0)
                values = [v for v, _ in shared.call_many([[0.25, 0.0], [0.25, 0.0], [0.25, 0.0]], "node")]
                self.assertEqual(values, [-np.inf] * 3)
                self.assertEqual(len(calls), 3)
            finally:
                shared.close()


class TestResidualNewton(unittest.TestCase):
    def test_linear_residual_is_removed_exactly(self):
        sb, truth, source, exact, bounds = study()
        with tempfile.TemporaryDirectory() as d:
            record, ledger = run(sb, truth, source, bounds, d, config=NewtonConfig(tol=1e-6))
            for name in source["names"]:
                r = record["profiles"][name]
                np.testing.assert_allclose(r["logl"], exact[name], atol=1e-6)
                for v, x in zip(r["logl"], r["xopt"]):
                    self.assertAlmostEqual(v, truth(np.array(x)), places=12)  # evaluated values
                for diag in r["optimization"]:
                    self.assertEqual(diag["flag"], 0)
                    # the model is exact: a round, steps to the optimum, a confirming round
                    self.assertLessEqual(diag["gradient_rounds"], 2)
                    self.assertLessEqual(diag["failed_steps"], 0)
            self.assertAlmostEqual(record["best_logl"], 0.0, places=6)
            self.assertTrue(record["anchor_consistent"])

    def test_curvature_error_and_noise(self):
        # The network's flattest direction is 40% too flat, values carry 1e-3 noise.
        sb, truth, source, exact, bounds = study(curvature_error=-0.4, noise=1e-3)
        with tempfile.TemporaryDirectory() as d:
            record, ledger = run(sb, truth, source, bounds, d, config=NewtonConfig(tol=5e-3))
            for name in source["names"]:
                r = record["profiles"][name]
                # a lower bound on the profile, within a few tolerances of it
                self.assertTrue(np.all(np.array(r["logl"]) <= exact[name] + 5e-3))
                np.testing.assert_allclose(r["logl"], exact[name], atol=0.03)
                self.assertTrue(all(diag["flag"] == 0 for diag in r["optimization"]))
            self.assertLess(ledger.counts["backend_calls"], 15 * 30)

    def test_optimum_beyond_a_bound(self):
        # Both optima lie outside the box in p5 (upper bound 0): the node ends on the bound.
        rng = np.random.default_rng(3)
        V = np.linalg.qr(rng.normal(size=(6, 6)))[0]
        P = V @ np.diag(np.geomspace(1, 50, 6)) @ V.T
        mu_s = np.full(6, 0.2)
        mu_t = mu_s + np.array([0.0, 0.1, -0.1, 0.1, -0.1, 0.3])
        bounds = np.array([[-3.0, 3.0]] * 5 + [[-3.0, 0.0]])
        sb = Quadratic(bounds, P, mu_s)
        truth = lambda x: float(-0.5 * (np.asarray(x) - mu_t) @ P @ (np.asarray(x) - mu_t))

        def constrained(mu):
            x = mu.copy()
            x[0], x[5] = 0.1, 0.0
            f, F = [1, 2, 3, 4], [0, 5]
            x[f] = mu[f] - np.linalg.solve(P[np.ix_(f, f)], P[np.ix_(f, F)] @ (x[F] - mu[F]))
            self.assertGreater(-(P @ (x - mu))[5], 0)  # the bound is active
            return x

        with tempfile.TemporaryDirectory() as d:
            ledger = EvaluationLedger(truth, bounds, journal=Path(d) / "calls.jsonl", max_calls=5000)
            v, x, diag, state = NewtonRefiner(sb, ledger).solve(sb.box(constrained(mu_s)), 0, 0.1, stage="node")
        self.assertEqual(x[5], 0.0)
        self.assertEqual(diag["flag"], 0)
        self.assertLessEqual(v, truth(constrained(mu_t)) + 1e-12)
        self.assertAlmostEqual(v, truth(constrained(mu_t)), delta=1e-2)

    def test_bound_release_from_wrong_surrogate_and_inherited_slopes(self):
        # Both signs, both all-bound and mixed interior/bound free sets.  The
        # inherited prediction deliberately reinforces the wrong active set.
        for sign in (-1., 1.):
            for mixed in (False, True):
                for inherited in (False, True):
                    with self.subTest(sign=sign, mixed=mixed, inherited=inherited):
                        n = 3 if mixed else 2
                        bounds = np.array([[-1., 1.]] * n)
                        mu_s = np.zeros(n); mu_s[1] = 2 * sign
                        mu_t = np.zeros(n); mu_t[1] = .2 * sign
                        if mixed: mu_t[2] = .35
                        sb = Quadratic(bounds, np.eye(n), mu_s)
                        seen = []
                        def truth(x):
                            x = np.asarray(x)
                            seen.append(x.copy())
                            return float(-.5 * ((x - mu_t) @ (x - mu_t)))
                        start = np.zeros(n); start[1] = sign
                        prior = None
                        if inherited:
                            g = np.zeros(n); g[1] = 3 * sign
                            prior = dict(path=[(0., g.tolist())], transfer_skip=1)
                        with tempfile.TemporaryDirectory() as d:
                            ledger = EvaluationLedger(truth, bounds, journal=Path(d)/"calls.jsonl", max_calls=150)
                            v, x, diag, _ = NewtonRefiner(sb, ledger, NewtonConfig(tol=1e-6)).solve(
                                start, 0, 0., prior=prior)
                        self.assertEqual(diag["flag"], 0)
                        self.assertGreater(diag["gradient_rounds"], 0)
                        self.assertLess(-2 * v, 1e-5)
                        np.testing.assert_allclose(x, mu_t, atol=.003)
                        self.assertTrue(all(p[0] == 0 and np.all(abs(p) <= 1) for p in seen))
                        self.assertTrue(any(abs(p[1]) < 1 for p in seen))

    def test_unverified_bound_does_not_converge_when_budget_is_exhausted(self):
        bounds = np.array([[-1., 1.]] * 2)
        sb = Quadratic(bounds, np.eye(2), np.array([0., 2.]))
        truth = lambda x: float(-.5 * (np.asarray(x)[1] - .2)**2)
        with tempfile.TemporaryDirectory() as d:
            ledger = EvaluationLedger(truth, bounds, journal=Path(d)/"calls.jsonl", max_calls=150)
            v, x, diag, _ = NewtonRefiner(sb, ledger, NewtonConfig(maxcalls=1)).solve([0., 1.], 0, 0.)
        self.assertEqual(diag["reason"], "budget")
        self.assertEqual(diag["flag"], 1)
        self.assertEqual(diag["backend_calls"], 1)

    def test_step_pinned_to_a_bound_cannot_cycle_on_cached_points(self):
        # Found by a random search. A step clipped at a bound is longer than the trust radius
        # in the whitened metric; when it fails, norm / 4 gives the same radius back, and the
        # same point (a cache hit, no budget) followed without end, as at the it30 SINU TT 2D
        # node (14, 1), where one point was requested 79,000 times.
        P = np.array([[1.0, 0.046119219684511506, 0.28281806272957816],
                      [0.046119219684511506, 1341.2951207924364, -978.1005729974498],
                      [0.28281806272957816, -978.1005729974498, 716.696469788557]])
        mu = np.array([-0.712080412767336, 1.5547431567193657, 1.0537494015421012])
        tilt = np.array([3.080194596103163, 2.3058799580730214, -0.8493974695950111])
        w = np.array([0.17414758274471914, 0.7043174487610269, -0.6881929604369181])
        bounds = np.array([[-1.0, 1.0]] * 3)
        sb = Quadratic(bounds, P, mu)
        truth = lambda x: float(-0.5 * (x - mu) @ P @ (x - mu) + tilt @ x - 11.24209405156911 * (w @ (x - mu)) ** 2)
        with tempfile.TemporaryDirectory() as d:
            ledger = EvaluationLedger(truth, bounds, journal=Path(d) / "calls.jsonl", max_calls=1000)
            v, x, diag, _ = NewtonRefiner(sb, ledger, NewtonConfig(), log=lambda *a, **k: None).solve(
                sb.box(np.clip(mu, -1, 1)), 0, 0.1461576127282748)
        self.assertTrue(any(h.get("repeat") for h in diag["history"]))  # the cycle state was reached
        self.assertLessEqual(diag["backend_calls"], 150)
        self.assertLess(ledger.counts["cache_hits"], 10)
        self.assertEqual(x[0], 0.1461576127282748)
        self.assertTrue(np.isfinite(v))

    def test_nonfinite_inward_probe_cannot_certify_bound(self):
        bounds = np.array([[-1., 1.]] * 2)
        sb = Quadratic(bounds, np.eye(2), np.array([0., 2.]))
        truth = lambda x: 0. if np.asarray(x)[1] == 1. else -np.inf
        with tempfile.TemporaryDirectory() as d:
            ledger = EvaluationLedger(truth, bounds, journal=Path(d)/"calls.jsonl", max_calls=150)
            v, x, diag, _ = NewtonRefiner(sb, ledger).solve([0., 1.], 0, 0.)
        self.assertEqual(diag["reason"], "unverified_bound")
        self.assertEqual(diag["flag"], 1)
        self.assertEqual(v, 0.)

    def test_a_step_without_a_finite_value_is_saved_as_a_failed_step(self):
        # The cluster backend turns an error at a point into -inf. At a trust-region step that
        # is a failed step, but its gain went into the node's history as -inf, the checkpoint
        # (JSON without infinities) could not be written, and every resume stopped there again.
        sb, truth, source, exact, bounds = study(curvature_error=-0.4)
        name = Path("true") / "node_000_00002.json"
        with tempfile.TemporaryDirectory() as d:
            run(sb, truth, source, bounds, d, chain_length=2)
            node = json.loads((Path(d) / name).read_text())
            first = next(h for h in node["diagnostics"]["history"] if h["kind"] == "round")
            events = [json.loads(line) for line in (Path(d) / "calls.jsonl").read_text().splitlines()]
            call = next(e["call"] for e in events if e["event"] == "return" and e["stage"] == "corrected/node/p0/2"
                        and abs(e["value"] - (node["direct_logl"] + first["actual"])) < 1e-9)
            step = np.array(next(e["x"] for e in events if e["event"] == "start" and e["call"] == call))
        seen = []
        def failing(x):
            if np.array_equal(np.asarray(x, float), step):
                seen.append(1)
                return -np.inf
            return truth(x)
        with tempfile.TemporaryDirectory() as d:
            record, _ = run(sb, failing, source, bounds, d, chain_length=2)
            node = json.loads((Path(d) / name).read_text())
            rounds = [h for h in node["diagnostics"]["history"] if h["kind"] == "round"]
            self.assertIsNone(rounds[0]["actual"])
            self.assertTrue(all(np.isfinite(v) for n in source["names"] for v in record["profiles"][n]["logl"]))
            self.assertGreaterEqual(node["diagnostics"]["failed_steps"], 1)       # and the node went on from there
            self.assertGreater(node["logl"], node["direct_logl"])
            self.assertEqual(len(seen), 1)
            # Saving an improving value alone did not catch false convergence: the old
            # trust-boundary exit returned this node 3.013 log-units below its exact peak.
            gap = exact["p0"][2] - node["logl"]
            self.assertTrue(gap <= 0.01 or node["diagnostics"]["flag"] != 0)
            self.assertLess(gap, 3.01347)
            self.assertGreater(node["diagnostics"]["gradient_rounds"], 1)
            # The failed step must not enter the curvature model: as an excess of -inf it wrote
            # the stiffest curvature allowed along the step, and this node and four started from
            # its state ended on their budget (722 calls for the study in place of 289).
            self.assertEqual((node["diagnostics"]["reason"], node["diagnostics"]["flag"]), ("converged", 0))
            self.assertLess(gap, 1e-3)
            flagged = [p.name for p in (Path(d) / "true").glob("node_*.json")
                       if json.loads(p.read_text())["diagnostics"]["flag"] != 0]
            self.assertEqual(flagged, [])

    def test_a_round_that_ends_at_the_budget_still_gets_its_verdict(self):
        # The verdict of a fresh gradient round costs no call. A node whose certifying round
        # ended exactly at its call budget was reported out of budget, with a colder state for
        # its successor; at 49 parameters, rounds of 51 calls and a budget of 150 that happens.
        sb, truth, source, exact, bounds = study(curvature_error=-0.4, noise=1e-3)
        x_s = np.array(source["profiles"]["p0"]["xopt"][2], float)
        x_s[0] = source["profiles"]["p0"]["grid"][2]

        def solve(maxcalls):
            with tempfile.TemporaryDirectory() as d:
                ledger = EvaluationLedger(truth, bounds, journal=Path(d) / "calls.jsonl", max_calls=10 ** 6)
                start = float(ledger(x_s))
                return NewtonRefiner(sb, ledger, NewtonConfig(maxcalls=maxcalls)).solve(
                    sb.box(x_s), 0, float(x_s[0]), stage="node", t_start=start)

        value, x, diag, state = solve(150)
        calls = diag["backend_calls"]
        self.assertEqual((diag["reason"], diag["history"][-1]["kind"]), ("converged", "round"))
        self.assertNotIn("actual", diag["history"][-1])                 # certified by the round, without a step
        exact_value, exact_x, exact_diag, exact_state = solve(calls)    # the same calls and not one to spare
        self.assertEqual((exact_diag["reason"], exact_diag["flag"], exact_diag["backend_calls"]), ("converged", 0, calls))
        self.assertEqual(exact_value, value)
        np.testing.assert_array_equal(exact_x, x)
        self.assertEqual(exact_state, state)
        self.assertEqual(exact_diag["history"], diag["history"])
        short = solve(calls - 1)[2]                                       # one call fewer: the round cannot be made
        self.assertEqual((short["reason"], short["flag"]), ("budget", 1))

    def test_start_outside_the_box_is_clipped_not_rejected(self):
        # A scan can return a coordinate a few ulps outside its bound (seen with the legacy
        # scan: tau_reio = 0.01 - 5e-18). The true stage must start from the clipped point.
        sb, truth, source, exact, bounds = study()
        source = json.loads(json.dumps(source, default=lambda a: np.asarray(a).tolist()))
        source["profiles"]["p0"]["xopt"][2][5] = float(np.nextafter(3.0, np.inf))
        source["profiles"]["p1"]["xopt"][0][4] = float(np.nextafter(-3.0, -np.inf))
        with tempfile.TemporaryDirectory() as d:
            record, ledger = run(sb, truth, source, bounds, d)
            self.assertEqual(ledger.counts["invalid_requests"], 0)
            node = json.loads((Path(d) / "true" / "node_000_00002.json").read_text())
            self.assertEqual(node["surrogate_x"][5], 3.0)
            self.assertTrue(np.isfinite(node["direct_logl"]))
            self.assertTrue(all(np.isfinite(v) for n in source["names"] for v in record["profiles"][n]["logl"]))

    def test_clip_profile_starts(self):
        from profiles.evaluation import clip_profile_starts

        source = dict(full_parameter_names=["a", "b"], best_x=[0.5, 1.0],
                      profiles=dict(a=dict(grid=[0.0, 0.5], xopt=[[0.0, float(np.nextafter(1.0, 2.0))], [0.5, 0.2]])))
        clipped, changes = clip_profile_starts(source, [[0.0, 1.0], [-1.0, 1.0]])
        self.assertEqual(clipped["profiles"]["a"]["xopt"], [[0.0, 1.0], [0.5, 0.2]])
        self.assertEqual(clipped["best_x"], [0.5, 1.0])
        self.assertEqual([(c["where"], c["coordinate"], c["clipped_to"]) for c in changes], [("a/0", "b", 1.0)])
        self.assertEqual(source["profiles"]["a"]["xopt"][0][1], float(np.nextafter(1.0, 2.0)))  # input untouched

    def test_parallel_chains_reproduce_serial(self):
        sb, truth, source, exact, bounds = study(curvature_error=-0.4, noise=1e-3)
        results = {}
        for workers in (1, 4):
            with tempfile.TemporaryDirectory() as d:
                record, ledger = run(sb, truth, source, bounds, d, workers=workers)
                results[workers] = (record, ledger.counts["backend_calls"])
        (serial, n1), (parallel, n4) = results[1], results[4]
        self.assertEqual(n1, n4)
        self.assertEqual(serial["best_logl"], parallel["best_logl"])
        for name in source["names"]:
            self.assertEqual(serial["profiles"][name]["logl"], parallel["profiles"][name]["logl"])
            self.assertEqual(serial["profiles"][name]["xopt"], parallel["profiles"][name]["xopt"])

    def test_chain_plan(self):
        from profiles.residual_newton import chain_plan

        side = [1, 2, 3, 4, 5, 6, 7]  # nearest the seed first
        self.assertEqual(chain_plan(side, None), (side, []))
        self.assertEqual(chain_plan(side, 2), ([1, 2], [(3, [4, 5]), (6, [7])]))
        self.assertEqual(chain_plan(side, 0), ([], [(j, []) for j in side]))
        self.assertEqual(chain_plan(side, 10), (side, []))
        self.assertEqual(chain_plan([], 1), ([], []))

    def test_short_chains_reproduce_serial_and_stay_accurate(self):
        # More seeds, shorter chains: more concurrent work, and still a function of the
        # plan alone, never of workers or timing.
        sb, truth, source, exact, bounds = study(curvature_error=-0.4, noise=1e-3)
        for chain_length in (1, 0):
            results = {}
            for workers in (1, 4):
                with tempfile.TemporaryDirectory() as d:
                    record, ledger = run(sb, truth, source, bounds, d, workers=workers,
                                         config=NewtonConfig(tol=5e-3), chain_length=chain_length)
                    results[workers] = (record, ledger.counts["backend_calls"])
            (serial, n1), (parallel, n4) = results[1], results[4]
            self.assertEqual(n1, n4)
            self.assertEqual(serial["best_logl"], parallel["best_logl"])
            for name in source["names"]:
                r = serial["profiles"][name]
                self.assertEqual(r["logl"], parallel["profiles"][name]["logl"])
                self.assertEqual(r["xopt"], parallel["profiles"][name]["xopt"])
                self.assertTrue(np.all(np.array(r["logl"]) <= exact[name] + 5e-3))
                np.testing.assert_allclose(r["logl"], exact[name], atol=0.03)
                self.assertTrue(all(diag["flag"] == 0 for diag in r["optimization"]))
                if chain_length == 0:  # every node starts cold: nothing to transfer
                    self.assertTrue(all(diag["transfer"] == 0 for diag in r["optimization"]))

    def test_interrupted_stage_resumes_without_repeating_calls(self):
        from profiles.evaluation import WorkflowLedger

        sb, truth, source, exact, bounds = study(curvature_error=-0.4, noise=1e-3)
        with tempfile.TemporaryDirectory() as d:
            reference, ledger = run(sb, truth, source, bounds, d, workers=3)
            total = ledger.counts["backend_calls"]
        with tempfile.TemporaryDirectory() as d:
            calls = Path(d) / "calls"
            with self.assertRaises(CallBudgetExceeded):
                refine_profiles_newton(None, source, bounds, WorkflowLedger(Pool(truth), bounds, calls, total // 2),
                                       d, workers=3, surrogate=sb, log=lambda *a, **k: None)
            done = len(list((Path(d) / "true").glob("node_*.json")))
            self.assertLess(done, 15)
            ledger = WorkflowLedger(Pool(truth), bounds, calls, 20000)
            refine_profiles_newton(None, source, bounds, ledger, d, workers=3, surrogate=sb,
                                   log=lambda *a, **k: None)
            record = json.loads((Path(d) / "true_profiles.json").read_text())
            # Nothing is repeated but the calls the interruption abandoned before they ran.
            abandoned = sum('"event": "exception"' in line for path in calls.glob("attempt_*.jsonl")
                            for line in path.read_text().splitlines())
            self.assertEqual(ledger.prior_calls + ledger.counts["backend_calls"], total + abandoned)
            for name in source["names"]:
                self.assertEqual(record["profiles"][name]["logl"], reference["profiles"][name]["logl"])


if __name__ == "__main__":
    unittest.main()
