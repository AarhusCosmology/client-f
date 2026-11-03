"""convergence.metric: tempered_batch_scatter. The ordinary loop converges on the error of its
networks at their own tempered batches, and hands over to the boosts when the run has any."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from config.config import BoostConfig, ConvergenceConfig
from convergence.convergence import TEMPERED_BATCH_SCATTER, build_convergence_metric, tempered_verdict
from dataset.boost import record_start, recorded_start
from run.metrics import MetricsTracker


class Verdict(unittest.TestCase):
    def test_below_the_threshold_for_two_networks_running(self):
        scatters = {5: 11.0, 6: 8.1, 7: 7.9, 8: 9.0}
        got = tempered_verdict(scatters, 8, temperature=7.0)             # networks 6 and 7, against 1.2 x 7 = 8.4
        self.assertTrue(got["converged"])
        self.assertEqual((got["used"], got["limit"]), ([8.1, 7.9], 8.4))
        self.assertAlmostEqual(got["value"], 8.1 / 7.0)                  # the larger of the two, in units of T
        self.assertFalse(tempered_verdict(scatters, 7, 7.0)["converged"])   # network 5 was at 11
        self.assertFalse(tempered_verdict(scatters, 9, 7.0)["converged"])   # network 8 went back up: one dip is not enough

    def test_the_threshold_is_in_units_of_the_temperature(self):
        scatters = {3: 8.1, 4: 7.9}
        self.assertTrue(tempered_verdict(scatters, 5, 7.0)["converged"])
        self.assertFalse(tempered_verdict(scatters, 5, 5.0)["converged"])   # 1.2 x 5 = 6
        self.assertTrue(tempered_verdict(scatters, 5, 5.0, threshold=2.0)["converged"])
        self.assertFalse(tempered_verdict({3: 8.4, 4: 1.0}, 5, 7.0)["converged"])   # below, not at, the limit

    def test_a_batch_without_an_error_is_not_known(self):
        for scatters in ({4: 1.0}, {3: None, 4: 1.0}, {3: float("nan"), 4: 1.0}, {}):
            got = tempered_verdict(scatters, 5, 7.0)
            self.assertFalse(got["converged"])
            self.assertIsNone(got["value"])
        self.assertTrue(tempered_verdict({4: 1.0}, 5, 7.0, consecutive=1)["converged"])
        self.assertTrue(tempered_verdict({2: 1.0, 3: 2.0, 4: 3.0}, 5, 7.0, consecutive=3)["converged"])
        self.assertFalse(tempered_verdict({2: 9.0, 3: 2.0, 4: 3.0}, 5, 7.0, consecutive=3)["converged"])


class Configuration(unittest.TestCase):
    def test_the_metric_and_its_defaults(self):
        got = ConvergenceConfig.from_dict(dict(metric=TEMPERED_BATCH_SCATTER, max_iterations=26))
        self.assertEqual((got.threshold, got.consecutive, got.max_iterations), (1.2, 2, 26))
        got = ConvergenceConfig.from_dict(dict(metric=TEMPERED_BATCH_SCATTER, threshold=1.5, consecutive=3, max_iterations=9))
        self.assertEqual((got.threshold, got.consecutive), (1.5, 3))
        for bad in (dict(threshold=0.0), dict(threshold=-1), dict(consecutive=0)):
            with self.assertRaises(ValueError):
                ConvergenceConfig.from_dict(dict(metric=TEMPERED_BATCH_SCATTER, max_iterations=9, **bad))

    def test_the_drift_metric_is_as_it_was(self):
        got = ConvergenceConfig.from_dict(dict(metric="gaussian_posterior_drift", threshold=0.01, max_iterations=20))
        self.assertEqual((got.threshold, got.metric, got.max_iterations, got.consecutive), (0.01, "gaussian_posterior_drift", 20, 2))
        with self.assertRaises(KeyError):                                # its threshold has no default
            ConvergenceConfig.from_dict(dict(metric="gaussian_posterior_drift", max_iterations=20))
        self.assertEqual(build_convergence_metric("gaussian_posterior_drift").name, "gaussian_posterior_drift")
        with self.assertRaises(ValueError):
            build_convergence_metric("something_else")
        # the new criterion still summarises chains, and prints their drift, with the drift's class
        self.assertEqual(build_convergence_metric(TEMPERED_BATCH_SCATTER).name, "gaussian_posterior_drift")

    def test_the_number_of_boosts(self):
        self.assertIsNone(BoostConfig.from_dict(dict(n=10, start_iteration=3)).boosts)
        self.assertEqual(BoostConfig.from_dict(dict(n=10, start_iteration=3, boosts=5)).boosts, 5)
        self.assertIsNone(BoostConfig.from_dict(dict(n=10, start_iteration=3, boosts=None)).boosts)
        for bad in (0, -1, True):
            with self.assertRaises(ValueError):
                BoostConfig.from_dict(dict(n=10, start_iteration=3, boosts=bad))


class InTheLoop(unittest.TestCase):
    """client.tempered_convergence: what follows from the verdict."""

    def decide(self, scatters, iteration, use_convergence=True, boosts_follow=False):
        import client

        cfg = SimpleNamespace(sampling=SimpleNamespace(temperature=7.0),
                              convergence=ConvergenceConfig.from_dict(dict(metric=TEMPERED_BATCH_SCATTER, max_iterations=30)))
        with tempfile.TemporaryDirectory() as d:
            tracker = MetricsTracker(results_dir=Path(d))
            for k, scatter in scatters.items():
                tracker.add_acquisition_metrics(k, 100, 100, 1.0, 1000,
                                                held_out=None if scatter is None else dict(n=100, rms=scatter, bias=0.0, sd=scatter, largest=5 * scatter))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                stop, handover, verdict = client.tempered_convergence(cfg, tracker, iteration, use_convergence, boosts_follow)
            saved = json.loads((Path(d) / "metrics.json").read_text())["convergence"] if (Path(d) / "metrics.json").exists() else []
        return stop, handover, verdict, out.getvalue(), saved

    def test_not_converged(self):
        stop, handover, verdict, log, saved = self.decide({8: 12.0, 9: 8.0}, 10)
        self.assertEqual((stop, handover), (False, False))
        self.assertIn("tempered_batch_scatter: 12, 8 at the tempered batches of networks 8 to 9, against 1.2 x T = 8.4: not converged", log)
        self.assertEqual((saved[0]["metric_name"], saved[0]["converged"], saved[0]["iteration"]), (TEMPERED_BATCH_SCATTER, False, 10))
        self.assertAlmostEqual(saved[0]["metric_value"], 12.0 / 7.0)

    def test_converged_without_boosts_stops_the_run(self):
        stop, handover, _, log, saved = self.decide({8: 8.3, 9: 8.0}, 10)
        self.assertEqual((stop, handover), (True, False))
        self.assertIn(": converged", log)
        self.assertIn("Convergence criterion met, stopping...", log)
        self.assertTrue(saved[0]["converged"])

    def test_converged_with_boosts_still_to_begin_hands_over(self):
        stop, handover, verdict, log, _ = self.decide({8: 8.3, 9: 8.0}, 10, boosts_follow=True)
        self.assertEqual((stop, handover), (False, True))
        self.assertEqual(verdict["used"], [8.3, 8.0])
        self.assertNotIn("stopping", log)

    def test_a_run_of_given_length_only_prints_it(self):
        for boosts_follow in (False, True):
            stop, handover, _, log, saved = self.decide({8: 8.3, 9: 8.0}, 10, use_convergence=False, boosts_follow=boosts_follow)
            self.assertEqual((stop, handover), (False, False))
            self.assertIn("nothing follows from it: the number of iterations was given (-i)", log)
            self.assertTrue(saved[0]["converged"])                       # recorded all the same

    def test_too_early_or_without_a_recorded_error(self):
        stop, handover, _, log, saved = self.decide({0: 50.0}, 1)
        self.assertEqual((stop, handover, saved), (False, False, []))
        self.assertIn("not yet, it takes the tempered batches of 2 networks", log)
        stop, handover, _, log, saved = self.decide({8: None, 9: 8.0}, 10)      # a run recorded before the error was kept
        self.assertEqual((stop, handover, saved), (False, False, []))
        self.assertIn("not known, 8 at the tempered batches", log)


class EarlyStart(unittest.TestCase):
    def test_the_record_of_the_hand_over(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d) / "boost"
            self.assertIsNone(recorded_start(directory))
            record_start(directory, 11, 20, tempered_verdict({9: 8.3, 10: 8.0}, 11, 7.0), 7.0, 1.2)
            self.assertEqual(recorded_start(directory), 11)
            saved = json.loads((directory / "start.json").read_text())
        self.assertEqual((saved["start_iteration"], saved["configured"], saved["scatter_at_the_tempered_batches"]), (11, 20, [8.3, 8.0]))
        self.assertAlmostEqual(saved["limit"], 8.4)


if __name__ == "__main__":
    unittest.main()
