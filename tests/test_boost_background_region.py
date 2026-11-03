"""acquisition.boost.background and acquisition.boost.region: the quadratic background of the
network and the region that the boosts' posterior points reach."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.config import AuditConfig, BoostConfig
from likelihood.surrogate import ParameterInfo, SurrogateLikelihood, SurrogateMetadata
from model.network import (build_model, background_values, copy_model_dtype, fit_background,
                           load_model, model_background, transfer_warm_start)
from sampling.region import FLOOR, STRENGTH, Region, boost_clouds, boost_region
from training.audit import audit_optima
from training.losses import build_loss

NDIM = 4
LOWER, UPPER = np.zeros(NDIM), np.ones(NDIM)
CENTRE = np.full(NDIM, 0.4)
NAMES = [f"p{k}" for k in range(NDIM)]


def truth(points):
    return -0.5 * np.sum(((np.atleast_2d(points) - CENTRE) / 0.05) ** 2, axis=1)


def tempered(n, temperature, seed):
    rng = np.random.default_rng(seed)
    x = np.clip(CENTRE + 0.05 * np.sqrt(temperature) * rng.normal(size=(n, NDIM)), 0, 1)
    return x.astype(np.float32), truth(x).astype(np.float32)


class BumpModel(tf.keras.Model):
    """The truth plus a false optimum of height `height` at the vertex (1, 1, 1, 1)."""

    def __init__(self, height):
        super().__init__()
        self.height = height

    def call(self, x, training=False):
        peak = -0.5 * tf.reduce_sum(((x - 0.4) / 0.05) ** 2, axis=1, keepdims=True)
        bump = self.height * tf.exp(-0.5 * tf.reduce_sum(((x - 1.0) / 0.3) ** 2, axis=1, keepdims=True))
        return tf.maximum(peak, bump - 1.0e-3 * tf.reduce_sum((x - 1.0) ** 2, axis=1, keepdims=True))


class Background(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(5)
        self.x, self.y = tempered(3000, 7.0, 1)
        self.background = fit_background(self.x, self.y, depth=7.0 * NDIM)

    def test_fit_follows_the_cloud_and_refuses_too_few_rows(self):
        # the rows are at T = 7 and cut at 28: one scale brings the cloud's covariance back
        self.assertGreater(self.background["kappa"], 2.0)
        self.assertLess(self.background["kappa"], 8.0)
        self.assertAlmostEqual(self.background["top"], float(self.y.max()), places=5)
        np.testing.assert_allclose(self.background["mean"], CENTRE, atol=0.02)
        near = (self.y.max() - self.y) < 20
        residual = self.y[near] - background_values(self.background, self.x[near])
        self.assertLess(np.abs(residual).max(), 12.0)
        self.assertTrue(np.all(np.linalg.eigvalsh(self.background["precision"]) > 0))
        with self.assertRaises(ValueError):
            fit_background(self.x[:10], self.y[:10], depth=28.0)

    def test_model_is_background_plus_network_and_survives_saving(self):
        model = build_model(self.x, self.y, 2, 12, "alsing", background=self.background)
        plain = build_model(self.x, self.y, 2, 12, "alsing")
        self.assertIsNone(model_background(plain))
        kept = model_background(model)
        np.testing.assert_allclose(kept["precision"], self.background["precision"], rtol=1e-6)
        far = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]], np.float32)
        # far from the data the untrained network part is small beside the quadratic
        q = background_values(self.background, far)
        self.assertLess(np.abs(model(far).numpy().ravel() - q).max(), 0.05 * np.abs(q).min())
        with tempfile.TemporaryDirectory() as d:
            model.save(Path(d) / "m.keras")
            loaded = load_model(Path(d) / "m.keras")
        np.testing.assert_array_equal(loaded(self.x[:200]).numpy(), model(self.x[:200]).numpy())
        np.testing.assert_allclose(model_background(loaded)["mean"], self.background["mean"], rtol=1e-6)
        double = copy_model_dtype(model)
        np.testing.assert_allclose(double(self.x[:200].astype(np.float64)).numpy(), model(self.x[:200]).numpy(),
                                   rtol=1e-5, atol=1e-3)

    def test_warm_start_keeps_the_function_and_needs_the_same_architecture(self):
        old = build_model(self.x, self.y, 2, 12, "alsing", background=self.background)
        more_x, more_y = tempered(400, 1.0, 2)
        x, y = np.concatenate([self.x, more_x]), np.concatenate([self.y, more_y])
        new = build_model(x, y, 2, 12, "alsing", background=model_background(old))
        transfer_warm_start(old, new)
        np.testing.assert_allclose(new(x[:500]).numpy(), old(x[:500]).numpy(), rtol=1e-5, atol=2e-4)
        plain = build_model(x, y, 2, 12, "alsing")
        with self.assertRaises(ValueError):  # the loop then trains from random weights
            transfer_warm_start(plain, new)
        with self.assertRaises(ValueError):
            transfer_warm_start(old, plain)


class Regions(unittest.TestCase):
    def setUp(self):
        self.points, _ = tempered(600, 1.0, 3)
        self.region = Region.from_clouds([self.points], LOWER, UPPER, 2.0, best=0.0)

    def test_own_points_are_inside_and_far_points_are_not(self):
        rho = self.region.rho(self.points)
        self.assertAlmostEqual(rho.max(), 1.0, places=6)
        self.assertTrue(self.region.inside(self.points).all())
        self.assertFalse(self.region.inside(np.ones((1, NDIM))).any())
        tempered_points, _ = tempered(2000, 7.0, 4)
        np.testing.assert_allclose(self.region.rho_tf(tf.constant(tempered_points)).numpy(),
                                   self.region.rho(tempered_points), rtol=1e-4, atol=1e-4)

    def test_restricted_surrogate_is_the_network_inside_and_a_wall_outside(self):
        predict = lambda x: tf.reduce_sum(x, axis=1)
        restricted = self.region.restrict(predict)
        inside = self.points[:50]
        np.testing.assert_array_equal(restricted(tf.constant(inside)).numpy(), predict(tf.constant(inside)).numpy())
        corner = np.ones((1, NDIM), np.float32)
        rho = self.region.rho(corner)[0]
        self.assertAlmostEqual(float(restricted(tf.constant(corner)).numpy()[0]),
                               -FLOOR - STRENGTH * (rho - 2.0) ** 2, delta=1e-3 * STRENGTH * rho ** 2)

    def test_projection_brings_outside_points_to_the_surface(self):
        rng = np.random.default_rng(8)
        u = rng.uniform(size=(500, NDIM)).astype(np.float32)          # box units = physical here
        moved = self.region.project_unit(tf.constant(u)).numpy()
        rho_before, rho_after = self.region.rho(u), self.region.rho(moved)
        inside = rho_before <= 2.0
        self.assertTrue(inside.any() and (~inside).any())
        np.testing.assert_array_equal(moved[inside], u[inside])
        np.testing.assert_allclose(rho_after[~inside], 2.0, rtol=1e-4)
        # straight towards the centre: the direction from the cloud's mean is kept
        mean = self.region.means[0]
        a, b = u[~inside] - mean, moved[~inside] - mean
        cosine = np.sum(a * b, axis=1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
        np.testing.assert_allclose(cosine, 1.0, atol=1e-5)

    def test_start_points_and_files(self):
        start = self.region.start(300, np.random.default_rng(0))
        self.assertEqual(start.shape, (300, NDIM))
        self.assertTrue(self.region.inside(start).all())
        self.assertTrue(np.all((start >= LOWER) & (start <= UPPER)))
        with tempfile.TemporaryDirectory() as d:
            self.region.save(Path(d) / "region.npz", iteration=3)
            loaded = Region.load(Path(d) / "region.npz")
        np.testing.assert_array_equal(loaded.rho(self.points), self.region.rho(self.points))
        self.assertEqual((loaded.inflate, loaded.floor, loaded.rows), (2.0, -FLOOR, [600]))

    def test_start_is_the_draw_it_always_was_where_that_found_its_points(self):
        def as_before(region, n, rng):            # Region.start without a component, to 16 October 2026
            out, need, upper = [], n, region.lower + region.width
            for _ in range(200):
                k = rng.integers(0, len(region.means), 2 * need)
                z = rng.normal(size=(2 * need, len(region.lower)))
                z *= (region.reach[k] * rng.random(2 * need) ** (1.0 / len(region.lower))
                      / np.linalg.norm(z, axis=1))[:, None]
                u = region.means[k] + np.einsum("nd,nde->ne", z, np.linalg.inv(region.whiten)[k])
                x = region.lower + u * region.width
                good = x[np.all((x >= region.lower) & (x <= upper), axis=1)][:need]
                out.append(good)
                need -= len(good)
                if need <= 0:
                    return np.concatenate(out).astype(np.float32)
            raise RuntimeError("the region lies outside the box")

        np.testing.assert_array_equal(self.region.start(300, np.random.default_rng(5)),
                                      as_before(self.region, 300, np.random.default_rng(5)))
        # A region reaching far beyond the box (the axion run's network 18: 0.46% of the draws
        # inside): 200 draws over the whole of it leave the walkers short; it then closes in.
        wide = Region(self.region.means, self.region.whiten, 120 * self.region.reach, LOWER, UPPER, 2.0, -FLOOR)
        with self.assertRaises(RuntimeError):
            as_before(wide, 300, np.random.default_rng(5))
        start = wide.start(300, np.random.default_rng(5))
        self.assertEqual(start.shape, (300, NDIM))
        self.assertTrue(wide.inside(start).all())
        self.assertTrue(np.all((start >= LOWER) & (start <= UPPER)))
        # A region whose centre is outside the box has no start, then as before
        away = Region(self.region.means + 3.0, self.region.whiten, self.region.reach, LOWER, UPPER, 2.0, -FLOOR)
        with self.assertRaisesRegex(RuntimeError, "the region lies outside the box"):
            away.start(50, np.random.default_rng(5))

    def test_clouds_follow_the_modes_not_their_numbers(self):
        rng = np.random.default_rng(6)
        a = lambda n: np.clip([0.3, 0.3, 0.3, 0.3] + 0.02 * rng.normal(size=(n, NDIM)), 0, 1)
        b = lambda n: np.clip([0.7, 0.7, 0.7, 0.7] + 0.02 * rng.normal(size=(n, NDIM)), 0, 1)

        def write(directory, iteration, groups, values=None):
            rows = np.concatenate([g for g in groups])
            frame = pd.DataFrame(rows.astype(np.float32), columns=NAMES)
            frame["loglkl"] = -1.0 if values is None else values
            frame["predicted"] = -1.0
            frame["mode"] = np.concatenate([np.full(len(g), k) for k, g in enumerate(groups)])
            frame.to_csv(Path(directory) / f"boost_it_{iteration}.csv", index=False)

        with tempfile.TemporaryDirectory() as d:
            write(d, 5, [a(60), b(40)])
            write(d, 6, [b(50), a(30)])            # the heavier mode is now the other one
            values = np.full(40, -1.0)
            values[:3] = [-500.0, np.nan, -np.inf]   # not posterior points
            write(d, 7, [a(40)], values)
            write(d, 8, [a(1000)])                 # not before iteration 8
            clouds, info = boost_clouds(d, NAMES, 8, best=0.0, depth=20.0, lower=LOWER, upper=UPPER)
            self.assertEqual(info["boosts"], [5, 6, 7])
            self.assertEqual(info["rows"], [60 + 30 + 37, 40 + 50])
            self.assertEqual(info["dropped"], 3)
            self.assertLess(np.abs(clouds[0].mean(axis=0) - 0.3).max(), 0.02)
            self.assertLess(np.abs(clouds[1].mean(axis=0) - 0.7).max(), 0.02)
            region, info = boost_region(d, NAMES, LOWER, UPPER, 2.0, before=8, best=0.0, depth=20.0)
            self.assertEqual(len(region), 2)
            self.assertTrue(region.inside(np.array([[0.3] * 4, [0.7] * 4])).all())
            self.assertFalse(region.inside(np.array([[0.5] * 4])).any())   # between the modes
            none, info = boost_region(d, NAMES, LOWER, UPPER, 2.0, before=5, best=0.0, depth=20.0)
            self.assertIsNone(none)


class Consumers(unittest.TestCase):
    def setUp(self):
        self.points, _ = tempered(600, 1.0, 3)
        self.region = Region.from_clouds([self.points], LOWER, UPPER, 2.0, best=0.0)

    def test_surrogate_with_a_region(self):
        metadata = SurrogateMetadata(parameters=tuple(
            ParameterInfo(name=n, label=n, scale=1.0, lower=0.0, upper=1.0, center=0.4, sigma=0.05) for n in NAMES))
        model = BumpModel(50.0)
        corner = tf.ones((1, NDIM))
        free = SurrogateLikelihood(model, metadata)
        held = SurrogateLikelihood(model, metadata, region=self.region)
        self.assertGreater(float(free.loglkl(corner)[0]), 40.0)       # the false optimum
        self.assertLess(float(held.loglkl(corner)[0]), -FLOOR)         # the wall
        self.assertGreater(float(held.network(corner)[0]), 40.0)       # the network is still there
        inside = tf.constant(self.points[:20])
        np.testing.assert_array_equal(held.loglkl(inside).numpy(), free.loglkl(inside).numpy())

    def test_audit_restricted_to_the_region_leaves_a_far_optimum_alone(self):
        rng = np.random.default_rng(0)
        x = np.concatenate([rng.uniform(size=(500, NDIM)), np.clip(CENTRE + 0.08 * rng.normal(size=(1000, NDIM)), 0, 1)])
        x = x[np.max(x, axis=1) < 0.8].astype(np.float32)
        y = truth(x).astype(np.float32)
        loss = lambda values: build_loss("msre", sigma_level=3, chi2_dof=NDIM, max_loglkl=float(np.max(values)), targets=values)
        calls = []

        def evaluate(points):
            calls.append(len(points))
            return np.asarray(points, np.float32), truth(points), None

        config = AuditConfig.from_dict(dict(starts=64, steps=300, rounds=1, k=2, epochs=1))
        arguments = dict(inputs=x, targets=y, validation_split=0.1, lower=LOWER, upper=UPPER, evaluate=evaluate,
                         build_loss=loss, config=config, batch_size=64, n_append=20, seed=1, log=lambda *a: None)
        *_, report = audit_optima(BumpModel(50.0), project=self.region.project_unit, **arguments)
        self.assertTrue(report["clear"])
        self.assertTrue(report["restricted"])
        self.assertEqual(calls, [])
        # what is left is the skirt of the bump at the region's edge, not its 50 at the vertex
        self.assertLess(report["highest_excess"], 5.0)
        self.assertGreater(report["highest_excess"], -1.0)
        *_, free = audit_optima(BumpModel(50.0), **arguments)            # as before without it
        self.assertFalse(free["restricted"])
        self.assertGreater(free["rounds"][0]["highest_excess"], 40.0)
        self.assertGreater(sum(calls), 0)


class Configuration(unittest.TestCase):
    def test_both_are_off_by_default(self):
        plain = BoostConfig.from_dict(dict(n=10, start_iteration=2))
        self.assertIsNone(plain.background)
        self.assertIsNone(plain.region)
        both = BoostConfig.from_dict(dict(n=10, start_iteration=2, background="quadratic", region=2))
        self.assertEqual((both.background, both.region), ("quadratic", 2.0))
        self.assertIsInstance(both.region, float)

    def test_values_are_checked(self):
        for bad in (dict(background="cubic"), dict(background=True), dict(region=1), dict(region=0.5), dict(region=True)):
            with self.assertRaises(ValueError, msg=str(bad)):
                BoostConfig.from_dict(dict(n=10, start_iteration=2, **bad))


if __name__ == "__main__":
    unittest.main()
