"""The report (hpc/analysis/report.py): the posterior summary, the
comparison with published values and the run's refused value. Its tables on saved runs are
exercised on the cluster."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hpc" / "analysis"))
import report  # noqa: E402


class ErrorStatistics(unittest.TestCase):
    def test_what_a_scatter_costs_at_posterior_points(self):
        # A Gaussian error of spread s: reweighting to the truth leaves exp(-s^2) of the points
        # and the two posteriors differ by a Kullback-Leibler divergence of s^2 / 2.
        error = 0.3 + 0.5 * np.random.default_rng(3).standard_normal(200000)
        found = report.stats(error)
        self.assertAlmostEqual(found["effective_fraction"], np.exp(-0.25), delta=0.01)
        self.assertAlmostEqual(found["kl"], 0.125, delta=0.003)
        self.assertAlmostEqual(found["robust_sd"], 0.5, delta=0.005)
        self.assertAlmostEqual(found["bias"], 0.3, delta=0.005)
        shifted = report.stats(error + 4.0)                      # a constant offset costs nothing
        self.assertAlmostEqual(shifted["effective_fraction"], found["effective_fraction"], places=9)
        self.assertAlmostEqual(shifted["kl"], found["kl"], places=9)
        self.assertEqual(report.stats([np.nan]), dict(n=0))


class RefusedValue(unittest.TestCase):
    """The value a run's refused points carry, and what knowing it does to the held-out errors."""

    def journal(self, folder, name, values):
        file = Path(folder) / name
        file.write_text("".join(json.dumps(dict(batch=k + 1, points=48, not_finite=0, refused=3, refused_value=value)) + "\n"
                                for k, value in enumerate(values)) + "\n")
        return file

    def test_from_the_run_itself_and_from_the_journals_otherwise(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "20261008_143804_run"
            run.mkdir()
            self.assertIsNone(report.refused_value_of(run))                       # run by hand: no journal, no record
            design = self.journal(tmp, "evaluations_1.jsonl", [None, -30700.0, -30700.0])   # the design, then the loop
            other = self.journal(tmp, "evaluations_2.jsonl", [-250000.0])
            self.assertEqual(report.refused_value_of(run, [design]), -30700.0)    # as before the run kept a record
            self.assertIsNone(report.refused_value_of(run, [design, other]))      # journals that do not agree
            (run / "refused_value.json").write_text(json.dumps(dict(value=-30700.0, setting="design", job="6842481")))
            self.assertEqual(report.refused_value_of(run), -30700.0)              # by hand, as the header shows
            self.assertEqual(report.refused_value_of(run, [other]), -30700.0)     # the run's own record first
            for damaged in ("{", "[]", json.dumps(dict(setting="design")), json.dumps(dict(value="x"))):
                (run / "refused_value.json").write_text(damaged)
                self.assertEqual(report.refused_value_of(run, [design]), -30700.0, damaged)
                self.assertIsNone(report.refused_value_of(run), damaged)

    def run_with_refused_rows(self, folder):
        """A stand-in for report.Run: 60 batch rows, 12 of them refused and carrying the run's value,
        and a network within 0.2 of the truth that predicts -1040 where the likelihood refuses."""
        rng = np.random.default_rng(8)
        truth = -1000.0 - 20.0 * rng.random(60)
        refused = np.arange(60) % 5 == 0
        predicted = np.where(refused, -1040.0, truth + 0.2 * rng.standard_normal(60))
        new = pd.DataFrame(dict(a=rng.random(60), b=rng.random(60), loglkl=np.where(refused, -30700.0, truth)))
        return SimpleNamespace(dir=Path(folder), names=["a", "b"], ndim=2, temperature=7.0, best=-1000.0, models={4: None},
                               acquired=lambda iteration: (new, np.full(60, "batch", dtype=object)),
                               predict=lambda iteration, points: predicted)

    def test_refused_rows_are_left_out_of_the_held_out_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.run_with_refused_rows(tmp)
            run.refused_value = report.refused_value_of(run.dir)                  # no record: the rows are taken as values
            taken = report.held_out(run, 4)["batch"]
            self.assertEqual((taken["n"], taken["refused"]), (60, 0))
            self.assertGreater(taken["rms"], 10000.0)
            (run.dir / "refused_value.json").write_text(json.dumps(dict(value=-30700.0)))
            run.refused_value = report.refused_value_of(run.dir)                  # what main() does, without --journal
            found = report.held_out(run, 4)["batch"]
            self.assertEqual((found["n"], found["refused"]), (48, 12))
            self.assertLess(found["rms"], 0.3)
            self.assertLess(found["largest"], 1.0)
            self.assertEqual(sum(band.get("n", 0) for band in found["bands"]), 48)


class SummaryTests(unittest.TestCase):
    def test_summary_is_in_double_precision(self):
        # 100 theta is known to 4 parts in 10 000. Averaged in single precision along the
        # sample axis, as the samples are stored, its mean came out two widths off and its
        # width doubled (8 October 2026), which read as a posterior against the box edge.
        rng = np.random.default_rng(5)
        truth = np.column_stack([1.0405 + 0.00037 * rng.standard_normal(85504),
                                 3.04 + 0.015 * rng.standard_normal(85504)])
        samples = truth.astype(np.float32)
        single = samples.mean(axis=0)
        self.assertGreater(abs(single[0] - truth[:, 0].mean()) / truth[:, 0].std(), 0.5)  # the error this guards against
        summary = report.summarise(samples)
        np.testing.assert_allclose(summary["mean"], truth.mean(axis=0), rtol=0, atol=2e-7)
        np.testing.assert_allclose(summary["sd"], truth.std(axis=0), rtol=1e-3)
        self.assertEqual(summary["q"].shape, (5, 2))
        self.assertLess(abs(summary["q"][2, 0] - np.median(truth[:, 0])), 2e-7)

    def test_posterior_against_published_values(self):
        run = SimpleNamespace(names=["a", "b", "c", "d"], lower=np.array([0.0, 0.0, -1.0, 0.0]), upper=np.array([2.0, 10.0, 1.0, 5.0]))
        posterior = dict(mean=np.array([1.25, 9.0, 0.2, 2.0]), sd=np.array([0.1, 0.5, 0.3, 1.0]))
        reference = {
            "a": dict(centre=1.0, minus=0.2, plus=0.1, prior=[0.0, 4.0]),       # asymmetric: above the centre, the upper error
            "b": dict(centre=8.0, minus=1.0, plus=1.0, prior=[0.0, 20.0]),      # two widths from a box edge inside the prior
            "c": dict(centre=0.0, minus=0.3, plus=0.3, prior=[-1.0, 1.0]),      # both box edges are prior edges
            "d": dict(centre=0.0, minus=0.0, plus=2.0, prior=[0.0, 50.0]),      # an upper limit: one-sided
        }
        rows = {r["name"]: r for r in report.against_reference(run, posterior, reference)}
        self.assertAlmostEqual(rows["a"]["shift"], 2.5)
        self.assertAlmostEqual(rows["a"]["width"], 0.1 / 0.15)
        self.assertAlmostEqual(rows["a"]["edge"], 7.5)             # the upper box edge; the lower one is the prior's
        self.assertEqual(rows["a"]["flag"], "")
        self.assertAlmostEqual(rows["b"]["shift"], 1.0)
        self.assertAlmostEqual(rows["b"]["edge"], 2.0)
        self.assertEqual(rows["b"]["flag"], "box edge")
        self.assertIsNone(rows["c"]["edge"])
        self.assertEqual(rows["c"]["flag"], "")
        self.assertAlmostEqual(rows["d"]["shift"], 1.0)             # in units of the upper error
        self.assertAlmostEqual(rows["d"]["width"], 0.5)
        posterior["mean"][0] = 1.35                               # 3.5 published sigmas
        self.assertEqual(report.against_reference(run, posterior, reference)[0]["flag"], "shift")
        self.assertEqual([r["name"] for r in report.against_reference(run, posterior, {"c": reference["c"]})], ["c"])


if __name__ == "__main__":
    unittest.main()
