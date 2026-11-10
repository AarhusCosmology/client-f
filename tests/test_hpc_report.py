"""The report (hpc/analysis/report.py): the posterior summary, the
comparison with published values and the run's refused value. Its tables on saved runs are
exercised on the cluster."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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



ROOT = Path(__file__).resolve().parents[1]
LN10 = float(np.log(10.0))


def saved_run(folder):
    """A run on disk with one network, number 2: the Gaussian 0 - ((x0 / 0.2)^2 + (x1 / 0.2)^2) / 2 in the box
    [-1, 1]^2, sampled flat in x0 and x1 (test_evidence.cobaya_run), with what report.Run and its T = 1 chain need."""
    import yaml
    from test_evidence import cobaya_run, quadratic_model

    run = cobaya_run(folder, quadratic_model([0.0, 0.0], [0.2, 0.2], 0.0), [-1.0, -1.0], [1.0, 1.0])
    settings = yaml.safe_load((run / "run.yaml").read_text())
    settings["sampling"] = dict(temperature=1.0)
    (run / "run.yaml").write_text(yaml.safe_dump(settings))
    (run / "convergence_stats").mkdir()
    np.savez(run / "convergence_stats" / "chain_summary_it_2.npz", mean=np.zeros(2), cov=np.diag([0.04, 0.04]))
    return run


class LinearPrior(unittest.TestCase):
    """--linear-prior: parameters p that the run sampled with a prior flat in p (log10 of something) get a prior flat
    in 10^p instead. The network learned the log posterior with the run's prior, so that adds ln(10) p to it."""

    def test_the_named_parameters_get_a_prior_flat_in_ten_to_the_p(self):
        # A Gaussian of width s times 10^x = exp(ln(10) x) is the Gaussian moved by ln(10) s^2: 0.092 for s = 0.2.
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as tmp:
            folder = saved_run(tmp)
            for linear, shift in (((), 0.0), (["x0"], LN10 * 0.04)):
                with self.subTest(linear_prior=linear):
                    run = report.Run(folder, ROOT, linear_prior=linear)
                    tf.keras.utils.set_random_seed(3)
                    chain, acceptance = report.sample_posterior(run, 2, 4000, 32, np.random.default_rng(1))
                    x = chain.reshape(-1, 2)
                    self.assertAlmostEqual(x[:, 0].mean(), shift, delta=0.02)
                    self.assertAlmostEqual(x[:, 1].mean(), 0.0, delta=0.02)      # a parameter not named keeps its prior
                    self.assertAlmostEqual(x[:, 0].std(), 0.2, delta=0.02)
                    self.assertTrue(0 < acceptance < 1)
                    # the network itself stays the network: the true check compares it with the truth
                    np.testing.assert_allclose(run.predict(2, [[0.5, 0.0]], region=True), [-3.125], rtol=1e-6)

    def test_named_in_the_run_s_order_once_and_only_parameters_of_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = saved_run(tmp)
            self.assertEqual(report.Run(folder, ROOT).linear_prior, ())
            self.assertEqual(report.Run(folder, ROOT, linear_prior=["x1", "x0", "x1"]).linear_prior, ("x0", "x1"))
            with self.assertRaises(SystemExit) as refused:
                report.Run(folder, ROOT, linear_prior=["x0", "log10Gamma"])
            self.assertIn("log10Gamma", str(refused.exception))
            self.assertIn("x0, x1", str(refused.exception))

    def test_the_sampled_target_records_it_and_a_chain_of_one_prior_is_never_the_other_s(self):
        from sampling.provenance import cached_samples, save_samples
        from sampling.targets import load_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            folder = saved_run(tmp)
            plain = report.Run(folder, ROOT).target_identity(2)
            self.assertEqual(plain, load_snapshot(folder, 2)[1])          # the chains made before stay valid
            self.assertNotIn("linear_prior", plain)
            linear = report.Run(folder, ROOT, linear_prior=["x0"]).target_identity(2)
            self.assertEqual(linear, dict(plain, linear_prior=["x0"]))
            x = np.zeros((10, 2))
            for saved, other in ((plain, linear), (linear, plain)):
                save_samples(Path(tmp) / "chain.npz", x, ["x0", "x1"], saved)
                self.assertIsNotNone(cached_samples(Path(tmp) / "chain.npz", saved))
                self.assertIsNone(cached_samples(Path(tmp) / "chain.npz", other))

    def test_its_files_go_to_a_folder_of_their_own(self):
        run = Path("results/20261106_121906_run")
        self.assertEqual(report.analysis_folder(run, ()), run / "analysis")
        self.assertEqual(report.analysis_folder(run, ("log10Gamma",)), run / "analysis" / "linear_prior_log10Gamma")
        self.assertEqual(report.analysis_folder(run, ("log10Gamma", "log10m")), run / "analysis" / "linear_prior_log10Gamma+log10m")

    def test_a_region_of_several_ellipsoids_weighs_its_cells_with_it(self):
        # Two equal Gaussian modes of width 0.1 at x = -0.5 and 0.5. Times 10^x the second weighs 10^(0.5 + 0.5) = 10
        # times the first: shares 1/11 and 10/11, and that only if the cells' evidences are those of the new target.
        import tensorflow as tf
        from sampling.region import Region

        rng = np.random.default_rng(42)
        clouds = [rng.normal([m, 0.0], 0.1, (500, 2)) for m in (-0.5, 0.5)]
        region = Region.from_clouds(clouds, [-2, -2], [2, 2], 2, 0)

        def predict(x):
            a = -0.5 * tf.reduce_sum(((x - [-0.5, 0.0]) / 0.1) ** 2, axis=1)
            b = -0.5 * tf.reduce_sum(((x - [0.5, 0.0]) / 0.1) ** 2, axis=1)
            return tf.reduce_logsumexp(tf.stack([a, b]), axis=0)

        surrogate = SimpleNamespace(region=region, logpost=region.restrict(predict))
        run = SimpleNamespace(ndim=2, names=["x", "y"], linear_prior=("x",), lower=np.array([-2.0, -2.0]),
                              upper=np.array([2.0, 2.0]), surrogate=lambda *a, **k: surrogate)
        tf.keras.utils.set_random_seed(3)
        chain, _ = report.sample_posterior(run, 7, 4000, 32, np.random.default_rng(1))
        self.assertAlmostEqual(np.mean(chain[:, :, 0] < 0), 1 / 11, delta=0.02)
        self.assertAlmostEqual(run.mixtures[7]["cells"][0]["weight"], 1 / 11, delta=0.02)
        self.assertEqual(run.mixtures[7]["unreliable_cells"], [])

    def test_the_report_writes_into_that_folder_and_says_which_prior(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = saved_run(tmp)
            argv = ["report.py", str(folder), "--client", str(ROOT), "--steps", "400", "--walkers", "8", "--linear-prior", "x0"]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                report.main()
            out = folder / "analysis" / "linear_prior_x0"
            self.assertTrue((out / "posterior_it_2.npz").exists())
            self.assertTrue((out / "report.json").exists())
            self.assertFalse((folder / "analysis" / "posterior_it_2.npz").exists())
            self.assertFalse((folder / "analysis" / "report.json").exists())
            self.assertEqual(json.loads((out / "report.json").read_text())["linear_prior"], ["x0"])
            self.assertIn("prior flat in 10^x0", (out / "report.md").read_text())
            with np.load(out / "posterior_it_2.npz") as saved:
                self.assertEqual(json.loads(str(saved["target_json"].item()))["linear_prior"], ["x0"])


if __name__ == "__main__":
    unittest.main()
