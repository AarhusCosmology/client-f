"""The derived-parameter emulator: configuration, loss, fit, persistence and command."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.config import Config, DerivedConfig
from derived.emulator import (
    load_derived_model,
    masked_mse,
    model_path,
    predict_derived,
    resolve_input_indices,
    train_derived_model,
)

NAMES = ["a", "b", "c", "d", "e"]


def truth(x):
    """Two smooth derived parameters of x[:, :3]; x[:, 3:] are nuisance-like."""
    return np.column_stack([
        60.0 + 10.0 * x[:, 0] - 4.0 * x[:, 1] ** 2,
        2e-9 * np.exp(0.5 * x[:, 2]) * (1.0 + 0.1 * x[:, 0]),
    ])


def small_config(**overrides):
    settings = dict(n_layers=2, n_neurons=32, n_epochs=400, batch_size=64,
                    lr_patience=30, lr_grace=40, inputs=["a", "b", "c"])
    settings.update(overrides)
    return DerivedConfig.from_dict(settings)


class DerivedConfigTests(unittest.TestCase):
    def test_absent_section_means_no_emulator(self):
        base = yaml.safe_load((Path(__file__).resolve().parents[1] / "input/_smoke.yaml").read_text())
        self.assertIsNone(Config.from_dict(base).derived)
        base["derived"] = None
        self.assertEqual(Config.from_dict(base).derived, DerivedConfig())
        base["derived"] = {"inputs": ["omega_b", "H0"], "n_neurons": 64}
        derived = Config.from_dict(base).derived
        self.assertEqual((derived.inputs, derived.n_neurons), (("omega_b", "H0"), 64))
        # as acquisition.audit and .boost: false is off (it was an error), true every default
        base["derived"] = False
        self.assertIsNone(Config.from_dict(base).derived)
        base["derived"] = True
        self.assertEqual(Config.from_dict(base).derived, DerivedConfig())
        for inputs in (3, [], [1, 2]):               # a number gave a bare TypeError
            with self.subTest(inputs=inputs), self.assertRaisesRegex(ValueError, "derived.inputs must be"):
                DerivedConfig.from_dict({"inputs": inputs})

    def test_input_selection(self):
        self.assertEqual(resolve_input_indices("auto", NAMES, ("c", "a")), [0, 2])
        self.assertEqual(resolve_input_indices("auto", NAMES, ()), [0, 1, 2, 3, 4])
        self.assertEqual(resolve_input_indices("all", NAMES, ("a",)), [0, 1, 2, 3, 4])
        self.assertEqual(resolve_input_indices(("e", "b"), NAMES, ()), [1, 4])
        with self.assertRaises(ValueError):
            resolve_input_indices(("z",), NAMES, ())
        with self.assertRaises(ValueError):
            resolve_input_indices("some", NAMES, ())


class MaskedLossTests(unittest.TestCase):
    def test_nan_targets_are_ignored(self):
        y_true = tf.constant([[1.0, np.nan], [np.nan, np.nan], [0.0, 2.0]])
        y_pred = tf.Variable([[2.0, 5.0], [7.0, 7.0], [1.0, 0.0]])
        with tf.GradientTape() as tape:
            loss = masked_mse(y_true, y_pred)
            total = tf.reduce_sum(loss)
        np.testing.assert_allclose(loss.numpy(), [1.0, 0.0, 2.5])
        gradient = tape.gradient(total, y_pred).numpy()
        self.assertTrue(np.isfinite(gradient).all())
        np.testing.assert_allclose(gradient[1], [0.0, 0.0])


class EmulatorFitTests(unittest.TestCase):
    def test_fit_uses_only_selected_inputs_and_round_trips(self):
        tf.keras.utils.set_random_seed(7)
        rng = np.random.default_rng(7)
        x = rng.uniform(-1.0, 1.0, (600, 5)).astype(np.float32)
        d = truth(x.astype(np.float64))
        d[:40, 1] = np.nan          # partially known rows stay in, masked
        d[40:50] = np.nan           # rows with nothing known are dropped
        loglkl = -0.5 * (x.astype(np.float64) ** 2).sum(axis=1)

        model, history, metrics, report = train_derived_model(
            x, d, loglkl, [0, 1, 2], ["H0ish", "A_sish"], small_config(),
            rng=np.random.default_rng(1),
        )
        self.assertEqual(metrics["n_train"] + metrics["n_validation"], 590)
        self.assertEqual(model.input_shape, (None, 5))
        self.assertTrue((report["rms_over_spread"] < 0.05).all(), report)

        # The nuisance-like columns cannot move the prediction.
        probe = rng.uniform(-1.0, 1.0, (8, 5)).astype(np.float32)
        shifted = probe.copy()
        shifted[:, 3:] += 0.7
        np.testing.assert_array_equal(predict_derived(model, probe),
                                      predict_derived(model, shifted))

        with tempfile.TemporaryDirectory() as tmp:
            path = model_path(tmp, 3)
            model.save(path)
            loaded = load_derived_model(path)
            np.testing.assert_array_equal(predict_derived(loaded, probe),
                                          predict_derived(model, probe))


class ChainTests(unittest.TestCase):
    def test_getdist_reads_the_chain_with_derived_marked(self):
        from getdist import loadMCSamples

        from derived.chain import write_getdist_chain

        rng = np.random.default_rng(5)
        chain = rng.normal(size=(50, 4, 2)).astype(np.float32)       # steps, walkers, ndim
        log_prob = -0.5 * (chain.astype(float) ** 2).sum(axis=-1)
        derived = np.stack([chain[..., 0] + chain[..., 1], np.exp(chain[..., 0])], axis=-1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "posterior_chains" / "surrogate_it_0"
            write_getdist_chain(root, chain, log_prob, ["100*theta_s", "b"], ["t", "b"],
                                [(-5, 5), (-5, 5)], derived, ["H0", "S8"], ["H_0", "S_8"])
            samples = loadMCSamples(str(root), settings={"ignore_rows": 0})
            flags = {p.name: p.isDerived for p in samples.paramNames.names}
            self.assertEqual(flags, {"100theta_s": False, "b": False, "H0": True, "S8": True})
            np.testing.assert_allclose(samples["H0"], samples["100theta_s"] + samples["b"],
                                       rtol=1e-6, atol=1e-6)
            self.assertEqual(len(samples["H0"]), 200)
            np.testing.assert_allclose(np.sort(samples.loglikes),
                                       np.sort(-log_prob.ravel()), rtol=1e-6)

    def test_exact_comparison_when_no_true_value_is_finite(self):
        from derived.chain import compare_exact, format_exact

        rng = np.random.default_rng(3)
        derived_chain = rng.normal(10.0, 2.0, size=(100, 3, 1))
        exact = np.full((5, 1), np.nan)
        rows, summary = compare_exact(["p"], np.full((5, 1), 10.0), exact, derived_chain,
                                      np.zeros(5), np.full(5, -np.inf))
        self.assertEqual((rows[0]["n"], summary["n"]), (0, 0))
        self.assertTrue(np.isnan(rows[0]["rms_error_over_sigma"]) and np.isnan(summary["importance_ess_fraction"]))
        self.assertIn("no true value", summary["importance_unavailable"])
        self.assertIn("no true value", format_exact(rows, summary))

    def test_exact_comparison_with_some_true_values_missing(self):
        from derived.chain import compare_exact, format_exact

        draws = np.array([-1.0, 0.0, 1.0])
        chain, exact, surrogate = draws.reshape(3, 1, 1), draws.reshape(3, 1), np.zeros(3)
        # Without any value the weight is not known: nothing is reported from weights.
        for last in (np.nan, np.inf):
            rows, summary = compare_exact(["p"], exact.copy(), exact, chain, surrogate, np.array([0.0, 0.0, last]))
            self.assertTrue(np.isnan(rows[0]["is_shift_over_sigma"]) and np.isnan(summary["importance_ess_fraction"]))
            self.assertIn("1 of the 3 true evaluations failed", summary["importance_unavailable"])
            self.assertEqual((summary["failed"], summary["refused"]), (1, 0))
            self.assertAlmostEqual(rows[0]["rms_error_over_sigma"], 0.0)              # pointwise: stays
            self.assertIn("no importance weights", format_exact(rows, summary))
        # A refused draw has weight zero and stays in the baseline, with its emulated value.
        refused = exact.copy()
        refused[2] = np.nan
        rows, summary = compare_exact(["p"], exact.copy(), refused, chain, surrogate, np.array([0.0, 0.0, -np.inf]))
        self.assertAlmostEqual(rows[0]["is_shift_over_sigma"], -0.5 / draws.std())    # -0.6124, not zero
        self.assertAlmostEqual(summary["importance_ess_fraction"], 2 / 3)
        self.assertEqual(rows[0]["baseline_emulated"], 1)
        self.assertEqual((summary["refused"], summary["failed"]), (1, 0))
        self.assertNotIn("importance_unavailable", summary)
        self.assertIn("1 more refused", format_exact(rows, summary))
        # A reason already given is kept with the new one.
        _, summary = compare_exact(["p"], exact.copy(), exact, chain, surrogate, np.array([0.0, 0.0, np.nan]), weights_problem="cells")
        self.assertTrue(summary["importance_unavailable"].startswith("cells; 1 of the 3"))

    def test_exact_comparison_of_a_parameter_without_exact_values(self):
        from derived.chain import compare_exact

        rng = np.random.default_rng(4)
        derived_chain = rng.normal(10.0, 2.0, size=(100, 3, 2))
        exact = np.column_stack([rng.normal(10.0, 2.0, size=40), np.full(40, np.nan)])
        rows, summary = compare_exact(["p", "q"], exact + 0.1, exact, derived_chain, np.zeros(40), rng.normal(size=40))
        self.assertEqual([r["n"] for r in rows], [40, 0])
        self.assertGreater(summary["importance_ess_fraction"], 0)

    def test_exact_comparison(self):
        from derived.chain import compare_exact

        rng = np.random.default_rng(2)
        derived_chain = rng.normal(10.0, 2.0, size=(1000, 3, 1))
        exact = rng.normal(10.0, 2.0, size=(400, 1))
        emulated = exact + 0.02          # a bias of 1% of the posterior width
        surrogate = rng.normal(size=400)
        rows, summary = compare_exact(["p"], emulated, exact, derived_chain,
                                      surrogate, surrogate + 3.0)
        self.assertAlmostEqual(rows[0]["mean_error_over_sigma"], 0.01, delta=2e-4)
        self.assertAlmostEqual(rows[0]["rms_error_over_sigma"], 0.01, delta=2e-4)
        # A constant offset between true and surrogate is not an error in the
        # posterior: equal weights, full ESS, no shift.
        self.assertAlmostEqual(summary["mean_true_minus_surrogate"], 3.0)
        self.assertAlmostEqual(summary["importance_ess_fraction"], 1.0)
        self.assertAlmostEqual(rows[0]["is_shift_over_sigma"], 0.0)
        self.assertNotIn("importance_unavailable", summary)

    def test_no_importance_shifts_from_a_chain_whose_cell_weights_are_not_reliable(self):
        # A region of several ellipsoids is mixed by the network's evidence in each cell. Where
        # that estimate is marked as not reliable the shares of the modes are uncertain, and
        # the exact check still reported shifts and an effective fraction from them.
        import json
        import tempfile
        from pathlib import Path

        from derived.chain import compare_exact, format_exact, weights_problem
        from sampling.provenance import sampler_identity

        rng = np.random.default_rng(2)
        derived_chain = rng.normal(10.0, 2.0, size=(1000, 3, 1))
        exact = rng.normal(10.0, 2.0, size=(400, 1))
        surrogate = rng.normal(size=400)
        cells = [dict(cell=0, efficiency=0.6), dict(cell=1, efficiency=0.03)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "surrogate_it_3"
            sidecar = root.with_suffix(".sampling.json")
            self.assertIn("no readable record", weights_problem(root))
            sidecar.write_text(json.dumps(dict(sampler_identity(None), target={}, acceptance=0.3)))
            self.assertIsNone(weights_problem(root))                                   # one ensemble
            sidecar.write_text(json.dumps(dict(sampler_identity([0, 1]), target={}, cells=cells[:1] * 2, unreliable_cells=[])))
            self.assertIsNone(weights_problem(root))
            sidecar.write_text(json.dumps(dict(sampler_identity([0, 1]), target={}, cells=cells, unreliable_cells=[1])))
            problem = weights_problem(root)
        self.assertIn("unreliable region-cell mixture weights (cells [1])", problem)
        rows, summary = compare_exact(["p"], exact + 0.02, exact, derived_chain, surrogate, surrogate + rng.normal(size=400),
                                      weights_problem=problem)
        self.assertAlmostEqual(rows[0]["mean_error_over_sigma"], 0.01, delta=2e-4)     # pointwise: stays
        self.assertTrue(np.isnan(rows[0]["is_shift_over_sigma"]) and np.isnan(summary["importance_ess_fraction"]))
        self.assertEqual(summary["importance_unavailable"], problem)
        text = format_exact(rows, summary)
        self.assertIn(f"no importance weights ({problem}): the IS shifts are not given", text)
        self.assertNotIn("WARNING", text)


class CommandTests(unittest.TestCase):
    def test_trains_from_a_run_directory(self):
        from derived.__main__ import main
        from derived.metadata import DerivedMetadata
        from likelihood.base import ParameterInfo
        from likelihood.surrogate import SurrogateMetadata

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            (run / "training_data").mkdir(parents=True)
            config = yaml.safe_load((root / "input/_smoke.yaml").read_text())
            config["seed"] = 3
            config["derived"] = {"n_layers": 2, "n_neurons": 16, "n_epochs": 60,
                                 "lr_patience": 10, "lr_grace": 10}
            (run / "run.yaml").write_text(yaml.safe_dump(config))
            params = [ParameterInfo(name=n, label=n, lower=-1.0, upper=1.0) for n in NAMES]
            SurrogateMetadata(parameters=tuple(params)).save(run / "metadata.json")
            DerivedMetadata(
                parameters=(ParameterInfo(name="H0ish", label="H_0"),
                            ParameterInfo(name="A_sish", label="A_s")),
                inputs=("a", "b", "c"),
            ).save(run / "derived_data")

            x = np.random.default_rng(0).uniform(-1, 1, (200, 5)).astype(np.float32)
            names = SimpleNamespace(param_names=NAMES, derived_names=["H0ish", "A_sish"])
            from dataset.dataset import TrainingDataset
            dataset = TrainingDataset(x, -0.5 * (x**2).sum(axis=1), names, n_neighbors=5,
                                      target_temperature=1.0, derived=truth(x.astype(float)))
            dataset.save(run / "training_data/data_it_2.csv")
            dataset.save_derived(run / "derived_data/derived_it_2.csv")

            main([str(run)])

            self.assertTrue(model_path(run / "derived_data", 2).exists())
            report = pd.read_csv(run / "derived_data/derived_report_it_2.csv")
            self.assertEqual(list(report["name"]), ["H0ish", "A_sish"])
            self.assertTrue((run / "derived_data/derived_history_it_2.csv").exists())


if __name__ == "__main__":
    unittest.main()
