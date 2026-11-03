"""The error of a network at the points it acquired (utils/held_out.py), and its place in a
run's metrics."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run.metrics import AcquisitionMetrics, MetricsTracker
from utils.held_out import describe, held_out_error


class HeldOut(unittest.TestCase):
    def test_error_is_network_minus_truth(self):
        truth = np.array([-10.0, -12.0, -11.0, -13.0])
        error = held_out_error(truth + np.array([1.0, -1.0, 3.0, -3.0]), truth)
        self.assertEqual((error["n"], error["left_out"]), (4, 0))
        self.assertAlmostEqual(error["bias"], 0.0)
        self.assertAlmostEqual(error["sd"], np.sqrt(5.0))
        self.assertAlmostEqual(error["rms"], np.sqrt(5.0))
        self.assertAlmostEqual(error["largest"], 3.0)
        shifted = held_out_error(truth + 2.0, truth)        # a constant offset is all bias
        self.assertAlmostEqual(shifted["bias"], 2.0)
        self.assertAlmostEqual(shifted["sd"], 0.0)
        self.assertAlmostEqual(shifted["rms"], 2.0)

    def test_points_without_a_true_value_are_left_out(self):
        # A point the likelihood refused: not finite, or carrying the launcher's fixed value.
        truth = np.array([-10.0, -np.inf, -30000.0, -11.0, np.nan])
        predicted = np.array([-9.0, -5.0, -12.0, -12.0, -11.0])
        error = held_out_error(predicted, truth, ignore=-30000.0)
        self.assertEqual((error["n"], error["left_out"]), (2, 3))
        self.assertAlmostEqual(error["rms"], 1.0)
        self.assertEqual(held_out_error(predicted, truth)["n"], 3)   # without the mark it counts
        nothing = held_out_error(predicted[:0], truth[:0])
        self.assertEqual(nothing, dict(n=0, left_out=0))
        self.assertEqual(describe(nothing), "no point with a true value")
        self.assertIn("rms 1 (mean +0, scatter 1, largest 1) at 2 points, 3 left out", describe(error))

    def test_in_the_metrics_of_a_run(self):
        with tempfile.TemporaryDirectory() as d:
            tracker = MetricsTracker(results_dir=Path(d))
            tracker.add_acquisition_metrics(0, 10, 10, 1.0, 110, held_out=dict(n=10, rms=2.0, bias=0.5, sd=1.9, largest=4.0))
            tracker.add_acquisition_metrics(1, 10, 9, 1.0, 119)      # nothing measured: stays empty
            tracker.save_all_metrics()
            saved = json.loads((Path(d) / "metrics.json").read_text())["acquisition"]
            self.assertEqual((saved[0]["held_out_rms"], saved[0]["held_out_n"], saved[1]["held_out_rms"]), (2.0, 10, None))
            # a run recorded before the fields existed loads as before
            old = {k: v for k, v in saved[0].items() if not k.startswith("held_out")}
            self.assertIsNone(AcquisitionMetrics(**old).held_out_rms)
            report = (Path(d) / "metrics.md").read_text()
            self.assertIn("| it | evaluated | added | dataset size | time | held-out rms | held-out scatter |", report)
            self.assertIn("| 2 | 1.9 |", report)
            self.assertIn("| - | - |", report)       # the iteration without a measurement, and the totals

    def test_a_run_without_it_keeps_its_table(self):
        with tempfile.TemporaryDirectory() as d:
            tracker = MetricsTracker(results_dir=Path(d))
            tracker.add_acquisition_metrics(0, 10, 10, 1.0, 110)
            tracker.save_all_metrics()
            report = (Path(d) / "metrics.md").read_text()
            self.assertIn("| it | evaluated | added | dataset size | time |\n", report)
            self.assertNotIn("held-out", report)


if __name__ == "__main__":
    unittest.main()
