"""client.py evidence on a log-likelihood of thousands: BEST adds ln Z up in single precision,
so it is given the network relative to a reference, which is added back in double precision.

A network that is a normalised Gaussian times exp(-15656) exactly, as a real run's values
are: its evidence in Cobaya's convention is -15656 + ln P(box).
"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_evidence import best_available, cobaya_run, command, quadratic_model  # noqa: E402

OFFSET = -15656.0


def gaussian_run(directory, mu, sd, training=True):
    """The run, its box, the network's peak value and its exact evidence. The peak is a
    single-precision number, and the evidence is that of the peak the network has."""
    mu, sd = np.asarray(mu, float), np.asarray(sd, float)
    norm = -np.sum(np.log(sd)) - 0.5 * len(mu) * math.log(2 * math.pi)
    lower, upper = np.full(len(mu), -1.0), np.full(len(mu), 1.0)
    # The training set: the peak, and a point 40 below it.
    points = np.stack([mu, mu + sd * math.sqrt(80.0 / len(mu))])
    run = cobaya_run(directory, quadratic_model(mu, sd, OFFSET + norm), lower, upper,
                     training=(points, [OFFSET + norm, OFFSET + norm - 40.0]) if training else None)
    mass = sum(math.log(0.5 * (math.erf((b - m) / (s * math.sqrt(2))) - math.erf((a - m) / (s * math.sqrt(2)))))
               for a, b, m, s in zip(lower, upper, mu, sd))
    peak = float(np.float32(OFFSET + norm))
    return run, (lower, upper), peak, peak - norm + mass


class ReferenceTests(unittest.TestCase):
    def quadrature(self, log_prob, lower, upper, **settings):
        """In place of BEST: the midpoint rule on a grid, exact enough in two parameters. It
        keeps what it was given."""
        import tensorflow as tf

        self.settings = settings
        edges = [np.linspace(a, b, 301) for a, b in zip(lower, upper)]
        grid = np.stack(np.meshgrid(*[0.5 * (e[1:] + e[:-1]) for e in edges], indexing="ij"), axis=-1)
        self.points = grid.reshape(-1, len(lower))
        self.values = log_prob(tf.constant(self.points, tf.float32)).numpy()
        self.outside = log_prob(tf.constant([np.asarray(upper) + 0.5], tf.float32)).numpy()
        logL = self.values.astype(float)
        weights = np.exp(logL - logL.max())
        self.lnZ_box = float(np.log(weights.mean()) + logL.max())
        return dict(lnZ_box=self.lnZ_box, sigma=0.0, H=1.0, points=self.points, weights=weights / weights.sum(),
                    logL=logL, n_dead=len(logL) - settings["n_live"], iteration_cap_reached=False,
                    live_fraction=0.0, seconds=0.0, device="/CPU:0")

    def run_command(self, run, *words):
        with mock.patch("evidence.sampler.import_best"), mock.patch("evidence.sampler.run_nested", self.quadrature):
            printed = command(run, "--n-live", "100", "--quiet", *words)
        stem = "evidence_it_2_nlive100" + (f"_{words[-1]}" if words else "")
        return printed, json.loads((run / "evidence" / f"{stem}.json").read_text()), \
            np.loadtxt(run / "evidence" / f"{stem}.txt")

    def test_best_gets_the_network_relative_to_the_training_best(self):
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as d:
            run, _, peak, exact = gaussian_run(d, [0.3, -0.2], [0.10, 0.20])
            network = tf.keras.models.load_model(run / "trained_models" / "model_it_2.keras", compile=False)
            printed, report, chain = self.run_command(run)

            # The reference is the best training value, here the peak, as a single-precision number.
            self.assertEqual((report["lnL_reference"], report["lnL_reference_from"]),
                             (peak, "training_data/data_it_2.csv"))
            self.assertIn(f"ln L relative to {peak:.10g}, the best of training_data/data_it_2.csv",
                          printed.splitlines()[0])
            # What the sampler got: the network minus the reference, to the last bit, so near
            # zero at the peak where the network itself is near -15656; -inf outside the domain.
            raw = network(tf.constant(self.points, tf.float32)).numpy()[:, 0]
            self.assertLess(abs(raw.max() - peak), 0.01)
            np.testing.assert_array_equal(self.values, raw - np.float32(peak))
            self.assertLess(abs(self.values.max()), 0.01)
            self.assertEqual(self.outside[0], -np.inf)
            self.assertEqual(self.settings["tolerance"], 1e-3)
            # What is reported and written has the reference back, added in double precision.
            self.assertEqual(report["lnZ_box"], self.lnZ_box + peak)
            self.assertEqual(report["lnZ"], self.lnZ_box + peak + report["lnV_box"])
            self.assertLess(abs(report["lnZ"] - exact), 1e-3)
            self.assertIn(f"ln Z      = {report['lnZ']:+.4f}", printed)
            self.assertIn(f"ln Z_box  = {report['lnZ_box']:+.4f}", printed)
            kept = self.values.astype(float) > -700  # the rows with a weight that is not zero
            np.testing.assert_allclose(chain[:, 1], -(self.values.astype(float)[kept] + peak), rtol=1e-9)
            self.assertLess(abs(chain[:, 1].min() + peak), 0.01)

    def test_the_default_number_of_live_points_is_a_power_of_two(self):
        # BEST adds the prior volume up in single precision, n_live_updates / n_live at every
        # iteration. With 2048 live points every partial sum is a single-precision number; with
        # 2000 it is not, and the sum drifts (docs/evidence.md).
        steps = 20000                                    # a 49-parameter run has about this many
        for n_live, exact in ((2048, True), (2000, False)):
            summed = np.cumsum(np.full(steps, 10 / n_live, np.float32), dtype=np.float32)[-1]
            self.assertEqual(float(summed) == steps * 10 / n_live, exact, n_live)
        with tempfile.TemporaryDirectory() as d:
            run, _, _, _ = gaussian_run(d, [0.3, -0.2], [0.10, 0.20])
            with mock.patch("evidence.sampler.import_best"), mock.patch("evidence.sampler.run_nested", self.quadrature):
                printed = command(run, "--quiet")
                self.assertEqual(self.settings["n_live"], 2048)
                self.assertTrue((run / "evidence" / "evidence_it_2_nlive2048.json").exists())
                self.assertNotIn("not a power of two", printed)
                self.assertIn("--n-live 2000 is not a power of two", command(run, "--quiet", "--n-live", "2000"))

    def test_without_the_training_set_nothing_is_subtracted(self):
        with tempfile.TemporaryDirectory() as d:
            run, _, peak, exact = gaussian_run(d, [0.3, -0.2], [0.10, 0.20], training=False)
            printed, report, chain = self.run_command(run)
            self.assertEqual((report["lnL_reference"], report["lnL_reference_from"]), (0.0, None))
            self.assertIn("no training set in the run for a reference", printed.splitlines()[0])
            self.assertIn("Warning: BEST added ln Z up in single precision at ln L near", printed)
            self.assertLess(abs(self.values.max() - peak), 0.01)
            self.assertEqual(report["lnZ_box"], self.lnZ_box)
            self.assertLess(abs(report["lnZ"] - exact), 1e-3)

    def test_reference_is_a_finite_training_value(self):
        from evidence.__main__ import reference_value

        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / "training_data").mkdir()
            self.assertEqual(reference_value(run, 4), (0.0, None))
            (run / "training_data" / "data_it_4.csv").write_text("a,b,loglkl\n0,0,-inf\n1,2,nan\n")
            self.assertEqual(reference_value(run, 4), (0.0, None))
            (run / "training_data" / "data_it_4.csv").write_text(
                "a,b,loglkl\n0,0,-15660.25\n1,2,-15656.123456789\n3,4,nan\n5,6,-inf\n")
            value, source = reference_value(run, 4)
            self.assertEqual(source, "training_data/data_it_4.csv")
            self.assertEqual(value, float(np.float32(-15656.123456789)))
            self.assertNotEqual(value, -15656.123456789)
            # The network's own file gone (a run tidied up): the nearest iteration's serves, the
            # earlier of two equally near.
            (run / "training_data" / "data_it_4.csv").rename(run / "training_data" / "data_it_7.csv")
            self.assertEqual(reference_value(run, 4), (value, "training_data/data_it_7.csv"))
            (run / "training_data" / "data_it_1.csv").write_text("a,b,loglkl\n0,0,-15700.5\n")
            self.assertEqual(reference_value(run, 4), (-15700.5, "training_data/data_it_1.csv"))
            self.assertEqual(reference_value(run, 6), (value, "training_data/data_it_7.csv"))


@unittest.skipUnless(best_available(), "BEST is not installed or on $BEST_PATH")
class SinglePrecisionTests(unittest.TestCase):
    def test_evidence_of_a_likelihood_of_thousands(self):
        # Four parameters and 8000 live points, about 45 s for the two runs. Measured on four
        # seeds with the sums taken to 1e-3 of the evidence: the network as it is gave ln Z 0.45
        # to 0.47 too low with a quoted error of 0.030, the network minus the reference was off
        # by -0.008 to +0.016. (Stopped at one half, as runs were, three seeds were within their
        # error without the reference, -0.016 to +0.004: the two changes belong together.)
        import tensorflow as tf
        from evidence.sampler import run_nested

        with tempfile.TemporaryDirectory() as d:
            run, (lower, upper), peak, exact = gaussian_run(d, [0.3, -0.2, 0.1, 0.0], [0.10, 0.05, 0.20, 0.08])
            command(run, "--n-live", "8000", "--quiet")
            report = json.loads((run / "evidence" / "evidence_it_2_nlive8000.json").read_text())
            self.assertLess(abs(report["lnZ"] - exact), 4 * report["sigma_lnZ"])
            self.assertEqual(report["lnL_reference"], peak)
            self.assertFalse(report["iteration_cap_reached"])

            # The same network without the reference, as BEST was given it before, to the same end.
            network = tf.keras.models.load_model(run / "trained_models" / "model_it_2.keras", compile=False)
            raw = run_nested(lambda x: tf.squeeze(network(tf.cast(x, tf.float32), training=False), axis=1),
                             lower, upper, n_live=8000, verbose=False)
            self.assertLess(raw["lnZ_box"] + float(np.sum(np.log(upper - lower))) - exact, -5 * raw["sigma"])


if __name__ == "__main__":
    unittest.main()
