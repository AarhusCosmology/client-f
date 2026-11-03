import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.network import build_model, load_model, transfer_warm_start, copy_model_dtype
from config.config import TrainingConfig


class WarmStartTests(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(123)
        self.rng = np.random.default_rng(42)
        self.x = self.rng.normal(size=(100, 3)).astype('float32')
        self.y = (self.x**2).sum(1)*2-20

    def models(self, layers=2, activation='alsing'):
        old = build_model(self.x, self.y, layers, 12, activation)
        new = build_model(self.x*[1.3, .8, 2]+[.4, -.2, .3], self.y*3+7, layers, 12, activation)
        return old, new

    @staticmethod
    def value_grad(model, x):
        x = tf.convert_to_tensor(x)
        with tf.GradientTape() as tape:
            tape.watch(x)
            value = model(x)
        return value.numpy(), tape.gradient(value, x).numpy()

    def test_values_gradients_and_save_reload(self):
        old, new = self.models()
        test = self.rng.normal(size=(50, 3)).astype('float32')
        expected = self.value_grad(old, test)
        norm = next(l for l in new.layers if isinstance(l, tf.keras.layers.Normalization))
        new_mean = norm.mean.numpy().copy()
        transfer_warm_start(old, new)
        np.testing.assert_array_equal(norm.mean.numpy(), new_mean)
        for a, b in zip(self.value_grad(new, test), expected):
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=3e-6)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'model.keras'; new.save(p)
            for a, b in zip(self.value_grad(load_model(p), test), expected):
                np.testing.assert_allclose(a, b, rtol=2e-5, atol=3e-6)

    def test_linear_model_compensates_both_ends(self):
        old, new = self.models(layers=0)
        transfer_warm_start(old, new)
        np.testing.assert_allclose(new(self.x), old(self.x), rtol=1e-5, atol=3e-6)

    def test_architecture_rejection_does_not_partially_mutate(self):
        old, new = self.models()
        other = build_model(self.x, self.y, 1, 8, 'alsing')
        before = other.get_weights()
        with self.assertRaises(ValueError):
            transfer_warm_start(old, other)
        for a, b in zip(before, other.get_weights()):
            np.testing.assert_array_equal(a, b)

    def test_changed_activation_rejected(self):
        old, _ = self.models(activation='tanh')
        _, new = self.models(activation='relu')
        with self.assertRaises(ValueError):
            transfer_warm_start(old, new)

    def test_constant_feature_uses_normalization_epsilon(self):
        self.x[:, 0] = 0
        old = build_model(self.x, self.y, 1, 6, 'alsing')
        new = build_model(self.x, self.y*2+1, 1, 6, 'alsing')
        transfer_warm_start(old, new)
        np.testing.assert_allclose(new(self.x), old(self.x), rtol=1e-5, atol=3e-6)

    def test_configuration_default(self):
        config = dict(learning_rate=1e-4, loss='msre', sigma_level=5,
                      n_epochs=10, batch_size=10, validation_split=.1, patience=5)
        self.assertTrue(TrainingConfig.from_dict(config).warm_start)

    def test_double_evaluation_retains_coefficients_and_normalization(self):
        old,_=self.models()
        old.compile(optimizer='adam',loss=lambda y,p:tf.reduce_mean((y-p)**2))
        double=copy_model_dtype(old)
        self.assertEqual(double.compute_dtype,'float64')
        for a,b in zip(old.get_weights(),double.get_weights()):
            np.testing.assert_array_equal(a.astype(b.dtype),b)
        np.testing.assert_allclose(old(self.x),double(self.x.astype('float64')),rtol=1e-5,atol=3e-6)
        x=self.x[:1].astype('float64')
        value,grad=self.value_grad(double,x)
        for j in range(x.shape[1]):
            left,right=x.copy(),x.copy();left[0,j]-=1e-5;right[0,j]+=1e-5
            numerical=(double(right).numpy()-double(left).numpy())/(2e-5)
            self.assertAlmostEqual(float(numerical[0,0]),float(grad[0,j]),places=7)


if __name__ == '__main__':
    unittest.main()
