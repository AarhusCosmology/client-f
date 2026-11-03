"""Which weights a training keeps: a warm-started training of the boost phase keeps the weights
at the end of its schedule, every other training those of the epoch with the lowest validation
loss. The callbacks, and the loop through client.py (client.py has the measurement behind it).
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.config import TrainingConfig  # noqa: E402
from training import training  # noqa: E402

BASE = dict(learning_rate=1e-3, loss="mse", sigma_level=3, n_epochs=40, batch_size=16, validation_split=0.25, patience=3)


class Setting(unittest.TestCase):
    def test_there_is_no_setting(self):
        with self.assertRaises(ValueError):
            TrainingConfig.from_dict(dict(BASE, warm_start_weights="best"))


class Default(unittest.TestCase):
    def test_warm_start_is_the_default_and_random_weights_an_option(self):
        self.assertTrue(TrainingConfig.from_dict(dict(BASE)).warm_start)
        self.assertFalse(TrainingConfig.from_dict(dict(BASE, warm_start=False)).warm_start)


class Callbacks(unittest.TestCase):
    """Whichever epoch is best, the model holds the weights of the epoch train_model reports."""

    def train(self, schedule, restore_best):
        tf.keras.utils.set_random_seed(3)
        rng = np.random.default_rng(3)
        x = rng.normal(size=(200, 3)).astype(np.float32)
        y = (x @ np.array([1.0, -2.0, 0.5], np.float32) + 0.3 * rng.normal(size=200)).astype(np.float32)
        model = tf.keras.Sequential([tf.keras.Input((3,)), tf.keras.layers.Dense(8, activation="tanh"),
                                     tf.keras.layers.Dense(1)])
        seen = []

        class Record(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                seen.append([w.copy() for w in self.model.get_weights()])

        original = training.build_callbacks
        training.build_callbacks = lambda *a, **k: original(*a, **k) + [Record()]
        try:
            history, metrics = training.train_model(
                model, x, y, "mse", 1e-2, 400, 16, 0.25, 3, lr_schedule=schedule, lr_factor=0.5,
                lr_patience=2, lr_min=1e-4, lr_grace=3, restore_best=restore_best)
        finally:
            training.build_callbacks = original
        return model, history, metrics, seen

    def test_both_schedules(self):
        for schedule in ("plateau", "none"):
            with self.subTest(schedule=schedule):
                model, history, metrics, seen = self.train(schedule, True)
                self.assertEqual(metrics["epoch"], int(np.argmin(history.history["val_loss"])) + 1)
                for a, b in zip(model.get_weights(), seen[metrics["epoch"] - 1]):
                    np.testing.assert_array_equal(a, b)

                model, history, metrics, seen = self.train(schedule, False)
                self.assertEqual(metrics["epoch"], len(history.history["val_loss"]))
                self.assertEqual(metrics["val_loss"], history.history["val_loss"][-1])
                for a, b in zip(model.get_weights(), seen[-1]):
                    np.testing.assert_array_equal(a, b)


class Loop(unittest.TestCase):
    """The documentation's Gaussian example with boosts from iteration 1, networks 0 to 2:
    network 0 from random weights, network 1 warm-started in the ordinary loop, network 2
    warm-started in the boost phase. Only network 2 keeps its last weights."""

    def run_client(self, root, **training_settings):
        config = yaml.safe_load((ROOT / "docs/site/examples/gaussian_client.yaml").read_text())
        config["likelihood"]["input"] = str(ROOT / "docs/site/examples/gaussian_likelihood.yaml")
        config["training"].update(lr_schedule="plateau", lr_patience=2, lr_min=1e-4, lr_grace=2, n_epochs=2000)
        config["training"].update(training_settings)
        config["acquisition"]["boost"] = dict(n=24, start_iteration=1, walkers=32, cool_steps=50, steps=200, components=1)
        path = root / "gaussian.yaml"
        path.write_text(yaml.safe_dump(config))
        result = subprocess.run([sys.executable, str(ROOT / "client.py"), str(path), "-i", "3", "-o", str(root / "runs")],
                                cwd=ROOT, capture_output=True, text=True, timeout=900,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1"))
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
        return result.stdout

    def test_last_applies_to_warm_starts_of_the_boost_phase_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            said = self.run_client(root)
            loop, boost = said.split("--- Iteration 2/")
            self.assertNotIn("Keeping the last weights", loop)                 # networks 0 and 1
            self.assertGreaterEqual(loop.count("Restoring best weights"), 2)
            self.assertIn("the last weights are kept (boost phase)", boost)
            self.assertIn("Keeping the last weights", boost)                   # network 2
            self.assertNotIn("Restoring best weights", boost.split("Training derived")[0])

    def test_random_weights_keep_their_best_weights_in_the_boost_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            said = self.run_client(Path(tmp), warm_start=False)
            self.assertNotIn("Warm start: inherited", said)
            self.assertNotIn("Keeping the last weights", said)
            self.assertIn("Restoring best weights", said.split("--- Iteration 2/")[1])


if __name__ == "__main__":
    unittest.main()
