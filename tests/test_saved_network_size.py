"""run.storage.save_model leaves the optimizer's state out and changes nothing else."""
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

try:
    import tensorflow as tf
except ImportError:                                           # pragma: no cover
    tf = None


def _trained():
    rng = np.random.default_rng(0)
    x = rng.random((128, 4)).astype("float32")
    y = (x ** 2).sum(1, keepdims=True)
    inputs = tf.keras.Input((4,))
    hidden = tf.keras.layers.Dense(32, activation="gelu")(inputs)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(1)(hidden))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(x, y, epochs=2, verbose=0)
    return model, x, y


@unittest.skipIf(tf is None, "needs tensorflow")
class SavedNetworkSize(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.dir = Path(self.temporary.name)

    def test_no_optimizer_state_and_same_network(self):
        import h5py

        from model.network import load_model
        from run.storage import save_model

        model, x, y = _trained()
        optimizer = model.optimizer
        save_model(model, self.dir / "lean.keras")
        model.save(self.dir / "full.keras")                   # what was written before
        self.assertIs(model.optimizer, optimizer)             # put back after the save
        model.fit(x, y, epochs=1, verbose=0)                  # and the model still trains

        with zipfile.ZipFile(self.dir / "lean.keras") as archive:
            archive.extract("model.weights.h5", self.dir / "lean")
        with h5py.File(self.dir / "lean" / "model.weights.h5") as weights:
            self.assertNotIn("optimizer", weights)
        self.assertLess((self.dir / "lean.keras").stat().st_size, (self.dir / "full.keras").stat().st_size)

        lean, full = load_model(self.dir / "lean.keras"), load_model(self.dir / "full.keras")
        self.assertTrue(np.array_equal(lean.predict(x, verbose=0), full.predict(x, verbose=0)))

    def test_uncompiled_model(self):
        from model.network import load_model
        from run.storage import save_model

        inputs = tf.keras.Input((2,))
        model = tf.keras.Model(inputs, tf.keras.layers.Dense(1)(inputs))
        save_model(model, self.dir / "m.keras")
        self.assertEqual(load_model(self.dir / "m.keras").count_params(), model.count_params())


if __name__ == "__main__":
    unittest.main()
