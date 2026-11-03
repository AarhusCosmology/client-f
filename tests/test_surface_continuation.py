"""The 2D setting `continuation`: rows of true optima continued from a chosen seed, every node
after the seed started from its predecessor's true optimum instead of the network's."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiles.config import ProfileConfig
from profiles.hybrid_surface import refine_surface_newton
from test_hybrid_surface import run, study


def direct_calls(d):
    """{(i, j): the point of node (i, j)'s direct call} from the study's call journal."""
    out = {}
    for journal in sorted((Path(d) / "calls").glob("attempt_*.jsonl")):
        for line in journal.read_text().splitlines():
            e = json.loads(line)
            if e["event"] == "start" and e["stage"].startswith("corrected/direct/surface/"):
                i, j = map(int, e["stage"].split("/")[-2:])
                out[i, j] = np.array(e["x"])
    return out


def continued(seed, start="scan"):
    sb, truth, bounds, cfg, source, exact = study()
    cfg.chain_length = None
    cfg.continuation = dict(seed=seed, start=start)
    return sb, truth, bounds, cfg, source, exact


class Continuation(unittest.TestCase):
    def test_each_node_after_the_seed_starts_at_its_predecessors_true_optimum(self):
        sb, truth, bounds, cfg, source, exact = continued(seed=-0.7)       # the first node of every row
        with tempfile.TemporaryDirectory() as d:
            result, _ = run(d, sb, truth, bounds, cfg, source)
            np.testing.assert_allclose(result["logl"], exact, atol=1e-6)
            starts, xopt = direct_calls(d), np.asarray(result["xopt"], float)
            axes = source["axes"]
            for i in range(3):
                seed = np.array(source["xopt"], float)[i, 0].copy()
                seed[:2] = axes[0][i], axes[1][0]
                np.testing.assert_array_equal(starts[i, 0], np.clip(seed, bounds[:, 0], bounds[:, 1]))
                for j in (1, 2):                     # one chain, upward from the seed
                    expected = xopt[i, j - 1].copy()
                    expected[:2] = axes[0][i], axes[1][j]
                    np.testing.assert_array_equal(starts[i, j], expected)
            method = json.loads((Path(d) / "true_surface/method.json").read_text())
            self.assertEqual(method["continuation"], dict(seed=-0.7, start="scan"))
            self.assertEqual(result["continuation"], method["continuation"])

    def test_a_seed_inside_the_row_continues_both_ways(self):
        sb, truth, bounds, cfg, source, exact = continued(seed=0.25)       # nearest 0.2, the middle node
        with tempfile.TemporaryDirectory() as d:
            result, _ = run(d, sb, truth, bounds, cfg, source)
            starts, xopt = direct_calls(d), np.asarray(result["xopt"], float)
            for i in range(3):
                for j, before in ((0, 1), (2, 1)):
                    expected = xopt[i, before].copy()
                    expected[:2] = source["axes"][0][i], source["axes"][1][j]
                    np.testing.assert_array_equal(starts[i, j], expected)

    def test_seed_from_the_best_training_point(self):
        sb, truth, bounds, cfg, source, exact = continued(seed=-0.7, start="best")
        best = np.array([0.5, -0.5, 0.9])
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "seed_start"):
                run(d, sb, truth, bounds, cfg, source)
        source = dict(source, seed_start=best.tolist())
        with tempfile.TemporaryDirectory() as d:
            result, _ = run(d, sb, truth, bounds, cfg, source)
            np.testing.assert_allclose(result["logl"], exact, atol=1e-6)
            starts = direct_calls(d)
            for i in range(3):
                np.testing.assert_array_equal(starts[i, 0], [source["axes"][0][i], source["axes"][1][0], 0.9])
            method = json.loads((Path(d) / "true_surface/method.json").read_text())
            self.assertEqual(method["continuation"]["seed_start"], best.tolist())

    def test_serial_parallel_resume_and_signature(self):
        sb, truth, bounds, cfg, source, exact = continued(seed=-0.7)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            serial, _ = run(a, sb, truth, bounds, cfg, source)
            cfg.workers = 3
            parallel, _ = run(b, sb, truth, bounds, cfg, source)
            for key in ["logl", "xopt", "best_logl", "best_x", "diagnostics"]:
                self.assertEqual(serial[key], parallel[key])
            _, resumed = run(b, sb, truth, bounds, cfg, source)
            self.assertEqual(resumed.counts["backend_calls"], 0)
            cfg.continuation = dict(seed=0.8, start="scan")               # another plan
            with self.assertRaisesRegex(ValueError, "checkpoint inputs"):
                run(b, sb, truth, bounds, cfg, source)
            cfg.continuation = None                                     # and the ordinary one
            with self.assertRaisesRegex(ValueError, "checkpoint inputs"):
                run(b, sb, truth, bounds, cfg, source)
        cfg.continuation, cfg.workers = None, 1                         # without it: no key
        with tempfile.TemporaryDirectory() as c:
            run(c, sb, truth, bounds, cfg, source)
            self.assertNotIn("continuation", json.loads((Path(c) / "true_surface/method.json").read_text()))

    def test_a_refused_node_hands_on_the_start_it_was_given(self):
        sb, truth, bounds, cfg, source, exact = continued(seed=-0.7)
        with tempfile.TemporaryDirectory() as first:
            run(first, sb, truth, bounds, cfg, source)
            refused = direct_calls(first)[0, 1]
        objective = lambda x: -np.inf if np.allclose(x, refused, atol=1e-12, rtol=0) else truth(x)
        with tempfile.TemporaryDirectory() as d:
            result, _ = run(d, sb, objective, bounds, cfg, source)
            self.assertIsNone(result["logl"][0][1])
            starts = direct_calls(d)
            np.testing.assert_array_equal(starts[0, 2][2:], np.asarray(result["xopt"][0][0])[2:])
            self.assertIsNotNone(result["logl"][0][2])

    def test_settings(self):
        ok = dict(dimension=2, parameters=["x", "y"], chain_length=None, continuation=dict(seed=0.0))
        ProfileConfig(**ok).validate()
        ProfileConfig(**dict(ok, continuation=dict(seed=1, start="scan"))).validate()
        for bad in (dict(seed="low"), dict(seed=True), dict(seed=float("nan")), dict(start="best"),
                    dict(seed=0.0, start="network"), dict(seed=0.0, step=1), [0.0]):
            with self.assertRaisesRegex(ValueError, "continuation must be"):
                ProfileConfig(**dict(ok, continuation=bad)).validate()
        for other in (dict(chain_length=2), dict(dimension=1, parameters=["x"])):
            with self.assertRaisesRegex(ValueError, "continuation requires"):
                ProfileConfig(**dict(ok, **other)).validate()

    def test_a_seed_outside_the_axis_and_a_plan_with_further_seeds_are_refused(self):
        sb, truth, bounds, cfg, source, exact = continued(seed=0.9)        # the axis ends at 0.8
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "outside the second axis"):
                run(d, sb, truth, bounds, cfg, source)
        cfg.continuation, cfg.chain_length = dict(seed=0.0), 2
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "chain_length: null"):
                run(d, sb, truth, bounds, cfg, source)


if __name__ == "__main__":
    unittest.main()
