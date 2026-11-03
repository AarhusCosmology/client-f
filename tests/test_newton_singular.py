"""A node whose Newton step's linear algebra fails ends that node, not the study."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiles.evaluation import EvaluationLedger
import profiles.hybrid
from profiles.residual_newton import NewtonRefiner
from test_residual_newton import Quadratic

quiet = lambda *a, **k: None


def problem():
    rng = np.random.default_rng(3)
    V = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    P = V @ np.diag(np.geomspace(1, 50, 6)) @ V.T
    mu_s = np.full(6, 0.2)
    mu_t = mu_s + np.array([0.0, 0.1, -0.1, 0.1, -0.1, 0.2])
    bounds = np.array([[-3.0, 3.0]] * 6)
    truth = lambda x: float(-0.5 * (np.asarray(x) - mu_t) @ P @ (np.asarray(x) - mu_t))
    return Quadratic(bounds, P, mu_s), truth, bounds, mu_s


def solve(prior=None):
    sb, truth, bounds, mu_s = problem()
    with tempfile.TemporaryDirectory() as d:
        ledger = EvaluationLedger(truth, bounds, journal=Path(d) / "calls.jsonl", max_calls=2000)
        start = sb.box(mu_s.copy())
        t0 = truth(sb.physical(start))
        return (*NewtonRefiner(sb, ledger, log=quiet).solve(start, 0, 0.2, stage="node", prior=prior), t0)


class SingularNode(unittest.TestCase):
    def test_a_singular_step_ends_the_node_with_its_best_value_and_passes_its_prior_on(self):
        v, _, diag, _, _ = solve()
        self.assertEqual(diag["reason"], "converged")
        prior = dict(path=[(0.2, [0.0] * 6)], Q=(0.5 * np.eye(6)).tolist(), transfer_skip=1, L=1.0)
        with mock.patch.object(profiles.hybrid, "ascend", side_effect=np.linalg.LinAlgError("Singular matrix")):
            v_s, x_s, diag_s, state_s, t0 = solve(prior)
        self.assertEqual((diag_s["reason"], diag_s["flag"]), ("singular", 1))
        self.assertGreaterEqual(v_s, t0)                  # the best true point found stands
        self.assertLessEqual(v_s, v + 1e-12)
        self.assertEqual(x_s[0], 0.2)                     # the profiled coordinate is the node's
        self.assertEqual(state_s["Q"], prior["Q"])        # what it was given goes on along the chain
        self.assertEqual(state_s["path"], [(0.2, [0.0] * 6)])

    def test_a_transfer_that_cannot_be_maximised_is_skipped(self):
        real = profiles.hybrid.ascend
        calls = []

        def first_fails(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:                           # the transfer's step
                raise np.linalg.LinAlgError("Singular matrix")
            return real(*args, **kwargs)

        prior = dict(path=[(0.1, [0.3, -0.2, 0.1, 0.0, 0.2, -0.1])], Q=np.zeros((6, 6)).tolist())
        with mock.patch.object(profiles.hybrid, "ascend", side_effect=first_fails):
            v, _, diag, _, t0 = solve(prior)
        self.assertGreater(len(calls), 1)
        self.assertEqual((diag["transfer"], diag["reason"]), (0, "converged"))
        self.assertGreater(v, t0)


if __name__ == "__main__":
    unittest.main()
