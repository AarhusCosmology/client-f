"""acquisition.boost.mixing: the modes decided by a measured T = 1 chain, and `client.py modes`."""
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.config import BoostConfig
from dataset.boost import draw_boost
from sampling.modes import (Regions, as_numpy, assign_basins, find_basins, find_modes, mixing_memory,
                            posterior_modes)

QUIET = lambda *_: None


def two_gaussians(ndim, half, second=0.5):
    """Two unit Gaussians at -half and +half on the first axis; the one at +half holds the
    share `second` of the mass. The straight path between the maxima dips about
    half^2 / 2 - log 2 below them."""
    shift = tf.constant(np.r_[half, np.zeros(ndim - 1)], tf.float32)
    lw = tf.constant(np.log([1.0 - second, second]), tf.float32)

    def predict(x):
        a = lw[0] - 0.5 * tf.reduce_sum((x + shift) ** 2, axis=1)
        b = lw[1] - 0.5 * tf.reduce_sum((x - shift) ** 2, axis=1)
        return tf.reduce_logsumexp(tf.stack([a, b]), axis=0)

    return predict


def hot_sample(ndim, half, n=3000, seed=0, second=0.5):
    """What a tempered chain leaves: both Gaussians, widened by sqrt(7)."""
    rng = np.random.default_rng(seed)
    x = np.sqrt(7.0) * rng.normal(size=(n, ndim))
    x[:, 0] += np.where(rng.random(n) < second, half, -half)
    return np.clip(x, -12.0, 12.0).astype(np.float32)


def box(ndim):
    return -12.0 * np.ones(ndim), 12.0 * np.ones(ndim)


class Basins(unittest.TestCase):
    def setUp(self):
        self.ndim, self.half = 4, 4.0                       # a dip of about 7.3
        self.values = as_numpy(two_gaussians(self.ndim, self.half))
        rng = np.random.default_rng(1)
        a = np.zeros((30, self.ndim)); a[:, 0] = -self.half
        b = np.zeros((12, self.ndim)); b[:, 0] = self.half
        self.maxima = np.concatenate([a, b]) + 0.01 * rng.normal(size=(42, self.ndim))

    def test_nothing_is_dropped_and_a_stalled_climb_stays_on_its_hill(self):
        stalled = np.zeros((1, self.ndim)); stalled[0, 0] = -self.half + 2.0   # 2 log-units under the peak at -half
        third = np.zeros((1, self.ndim)); third[0, 1] = 8.0                    # a hill of its own, one climb on it
        centre = tf.constant(third[0], tf.float32)
        two = two_gaussians(self.ndim, self.half)

        def three(x):
            return tf.reduce_logsumexp(tf.stack([two(x), -3.0 - 0.5 * tf.reduce_sum((x - centre) ** 2, axis=1)]), axis=0)

        values = as_numpy(three)
        points = np.concatenate([self.maxima, stalled, third])
        labels, count, dips = find_basins(values, points, barrier=1.0)
        self.assertEqual(count, 3)
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels[:30] == 0) and np.all(labels[30:42] == 1))   # by size
        self.assertEqual(labels[42], 0)                    # the stalled climb: with the hill it lies on
        self.assertEqual(labels[43], 2)                    # the small hill is kept, as a basin of one
        # 8 points on the path, none exactly midway: a little under half^2 / 2 - log 2 = 7.3
        self.assertAlmostEqual(dips[0, 1], 6.3, delta=0.3)
        np.testing.assert_array_equal(dips, dips.T)
        # the fixed rule leaves both the stalled climb and the hill of one out
        fixed, modes = find_modes(values, points, barrier=1.0)
        self.assertEqual((modes, fixed[42], fixed[43]), (2, -1, -1))

    def test_a_position_is_in_the_basin_it_climbs_to(self):
        predict = two_gaussians(self.ndim, self.half)
        labels, _, _ = find_basins(self.values, self.maxima, barrier=1.0)
        rng = np.random.default_rng(4)
        left = rng.normal(size=(40, self.ndim)); left[:, 0] -= self.half
        right = rng.normal(size=(40, self.ndim)); right[:, 0] += self.half
        lower, upper = box(self.ndim)
        got, dip = assign_basins(predict, self.values, np.concatenate([left, right]).astype(np.float32), self.maxima, labels, lower, upper, rng)
        np.testing.assert_array_equal(got, np.r_[np.zeros(40, int), np.ones(40, int)])
        self.assertLess(dip.max(), 0.1)

    def test_a_stalled_climb_does_not_join_two_hills(self):
        saddle = np.zeros((1, self.ndim))                   # midway: joined to both, far under both
        labels, count, _ = find_basins(self.values, np.concatenate([self.maxima, saddle]), barrier=1.0)
        self.assertEqual(count, 2)
        self.assertTrue(np.all(labels[:30] == 0) and np.all(labels[30:42] == 1))


class Memory(unittest.TestCase):
    def test_frozen_free_and_in_between(self):
        rng = np.random.default_rng(2)
        frozen = np.repeat((np.arange(200) % 2)[None], 100, axis=0)
        memory, counts = mixing_memory(frozen, 50)
        self.assertEqual(memory[0, 1], 1.0)
        self.assertEqual(counts[0, 1] + counts[1, 0], 0)
        free = rng.integers(0, 2, size=(100, 4000))
        self.assertLess(abs(mixing_memory(free, 50)[0][0, 1]), 0.02)
        flip = 0.02                                          # a two-state chain: memory (1 - 2 flip)^lag
        state = np.empty((100, 20000), int)
        state[0] = rng.integers(0, 2, 20000)
        for t in range(1, 100):
            state[t] = np.where(rng.random(20000) < flip, 1 - state[t - 1], state[t - 1])
        for lag in (10, 40):
            self.assertAlmostEqual(mixing_memory(state, lag)[0][0, 1], (1 - 2 * flip) ** lag, delta=0.02)
        memory = mixing_memory(state, 10)[0]
        self.assertAlmostEqual(memory[0, 1], memory[1, 0], places=12)

    def test_unoccupied_and_unlabelled(self):
        labels = np.zeros((20, 10), int)
        labels[:, :3] = -1                                    # walkers outside every basin are not counted
        memory, counts = mixing_memory(labels, 5, basins=3)
        self.assertEqual(counts.shape, (3, 3))
        self.assertEqual(counts[0, 0], 15 * 7)
        self.assertTrue(np.isnan(memory[0, 1]) and np.isnan(memory[1, 2]))


class GroupedRegions(unittest.TestCase):
    def test_basins_that_mix_share_a_region(self):
        rng = np.random.default_rng(3)
        centres = np.array([[0.2, 0.2], [0.35, 0.2], [0.8, 0.8]])
        clouds = [c + 0.02 * rng.normal(size=(200, 2)) for c in centres]
        regions = Regions(clouds, np.zeros(2), np.ones(2), groups=[0, 0, 1])
        self.assertEqual(len(regions), 2)
        points = np.array([[0.2, 0.2], [0.36, 0.21], [0.79, 0.8], [0.28, 0.2]])
        np.testing.assert_array_equal(regions.labels(points), [0, 0, 1, 0])
        np.testing.assert_array_equal(regions.labels_tf(tf.constant(points, tf.float32)).numpy(), [0, 0, 1, 0])
        self.assertEqual(len(Regions(clouds, np.zeros(2), np.ones(2))), 3)


class Decision(unittest.TestCase):
    SETTINGS = dict(walkers=64, cool_steps=300, steps=2000, components=1, seed=5, log=QUIET)

    def test_far_apart_stays_two_modes_with_the_right_weights(self):
        ndim, half = 4, 6.5                                   # a dip of about 20: a walker in a thousand crosses
        lower, upper = box(ndim)
        modes, info = posterior_modes(two_gaussians(ndim, half, second=0.3), lower, upper, hot_sample(ndim, half, second=0.3),
                                      7.0, mixing=0.1, **self.SETTINGS)
        self.assertEqual(len(modes), 2)
        pair = info["mixing"]["pairs"][0]
        self.assertFalse(pair["one_mode"])
        # (the stretch move flings a walker in the inner tail away from its own mode, so a few
        # of 64 walkers do get across in 2000 steps even here: the memory is high, not 1)
        self.assertGreater(pair["memory_after_chain"], 0.7)
        self.assertLess(pair["exchanged"], 0.05 * pair["pairs"])
        self.assertGreater(pair["dip"], 15.0)
        self.assertAlmostEqual(modes[0]["weight"], 0.7, delta=0.04)
        self.assertLess(modes[0]["x"][:, 0].mean(), 0)

    def test_a_barrier_the_chain_crosses_is_one_mode_where_the_fixed_rule_splits(self):
        ndim, half = 4, 3.0                                   # a dip of about 3.8
        lower, upper = box(ndim)
        predict, hot = two_gaussians(ndim, half), hot_sample(ndim, half)
        fixed, _ = posterior_modes(predict, lower, upper, hot, 7.0, barrier=2.0, **self.SETTINGS)
        self.assertEqual(len(fixed), 2)
        modes, info = posterior_modes(predict, lower, upper, hot, 7.0, barrier=2.0, mixing=0.1, **self.SETTINGS)
        self.assertEqual(len(modes), 1)
        pair = info["mixing"]["pairs"][0]
        self.assertTrue(pair["one_mode"])
        self.assertLess(pair["memory_after_chain"], 0.1)
        self.assertGreater(pair["exchanged"], 100)
        self.assertAlmostEqual(float(np.mean(modes[0]["x"][:, 0] > 0)), 0.5, delta=0.08)   # the one chain has both

    def test_a_barrier_the_chain_does_not_cross_is_two_modes_where_the_fixed_rule_joins(self):
        ndim, half = 40, 2.9                                  # a dip of about 3.5, under the fixed barrier of 4
        lower, upper = box(ndim)
        predict, hot = two_gaussians(ndim, half), hot_sample(ndim, half)
        settings = dict(self.SETTINGS, walkers=128, steps=1500)
        fixed, _ = posterior_modes(predict, lower, upper, hot, 7.0, **settings)
        self.assertEqual(len(fixed), 1)
        modes, info = posterior_modes(predict, lower, upper, hot, 7.0, mixing=0.1, **settings)
        self.assertEqual(len(modes), 2)
        pair = info["mixing"]["pairs"][0]
        self.assertLess(pair["dip"], 4.0)
        self.assertGreater(pair["memory_after_chain"], 0.1)

    def test_one_peak_needs_no_chain(self):
        ndim = 4
        lower, upper = box(ndim)
        modes, info = posterior_modes(two_gaussians(ndim, 0.0), lower, upper, hot_sample(ndim, 0.0), 7.0, mixing=0.1, **self.SETTINGS)
        self.assertEqual(len(modes), 1)
        self.assertEqual(info["mixing"]["pairs"], [])
        self.assertEqual(len(info["mixing"]["basins"]), 1)

    def test_default_is_the_fixed_barrier(self):
        ndim, half = 4, 5.0
        lower, upper = box(ndim)
        _, info = posterior_modes(two_gaussians(ndim, half), lower, upper, hot_sample(ndim, half), 7.0, **self.SETTINGS)
        self.assertNotIn("mixing", info)


class Configuration(unittest.TestCase):
    def test_off_by_default_and_checked(self):
        plain = BoostConfig.from_dict(dict(n=10, start_iteration=2))
        self.assertIsNone(plain.mixing)
        self.assertEqual(plain.candidate_barrier, 1.0)
        got = BoostConfig.from_dict(dict(n=10, start_iteration=2, mixing=0.1, candidate_barrier=0.5))
        self.assertEqual((got.mixing, got.candidate_barrier), (0.1, 0.5))
        for bad in (dict(mixing=0), dict(mixing=1), dict(mixing=True), dict(candidate_barrier=0)):
            with self.assertRaises(ValueError, msg=str(bad)):
                BoostConfig.from_dict(dict(n=10, start_iteration=2, **bad))

    def test_a_boost_passes_it_on(self):
        ndim, half = 4, 5.0
        lower, upper = box(ndim)
        config = BoostConfig.from_dict(dict(n=20, start_iteration=0, walkers=64, cool_steps=300, steps=2000, components=1, mixing=0.1))
        points, draw = draw_boost(two_gaussians(ndim, half), lower, upper, hot_sample(ndim, half), 7.0, config, seed=4, log=QUIET)
        self.assertEqual(len(points), 20)
        self.assertEqual(len(draw["modes"]), 2)
        self.assertIn("mixing", draw["info"])
        json.dumps(draw["info"])                              # the report must be writable


class Command(unittest.TestCase):
    def test_modes_of_a_saved_network(self):
        from sampling.detect import main

        centre, sigma, offset = (np.array([0.3, 0.3]), np.array([0.7, 0.7])), (0.04, 0.02), 1.0
        inputs = tf.keras.Input(shape=(2,))
        hills = []
        for c, s, shift in zip(centre, sigma, (0.0, -offset)):
            scaled = tf.keras.layers.Dense(2)(inputs)
            square = tf.keras.layers.Dot(axes=1)([scaled, scaled])
            hills.append(tf.keras.layers.Dense(1)(square))
        model = tf.keras.Model(inputs, tf.keras.layers.Maximum()(hills))
        dense = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
        for k, (c, s, shift) in enumerate(zip(centre, sigma, (0.0, -offset))):
            dense[k].set_weights([np.eye(2, dtype=np.float32) / s, (-c / s).astype(np.float32)])
            dense[2 + k].set_weights([np.array([[-0.5]], np.float32), np.array([shift], np.float32)])
        np.testing.assert_allclose(model(np.array([[0.3, 0.3], [0.7, 0.7]], np.float32)).numpy().ravel(), [0.0, -offset], atol=1e-4)
        cfg = yaml.safe_load((ROOT / "input/_smoke.yaml").read_text())
        cfg["sampling"].update(n_walkers=32, n_steps=1500, burn_in=200, thin=1)
        cfg["acquisition"]["boost"] = dict(n=10, start_iteration=0, walkers=64, cool_steps=300, steps=2000, components=1, mixing=0.2)
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            (run / "trained_models").mkdir(parents=True)
            (run / "run.yaml").write_text(yaml.safe_dump(cfg))
            (run / "metadata.json").write_text(json.dumps(dict(parameters=[
                dict(name=n, label=n, scale=1.0, lower=0.0, upper=1.0, center=0.5, sigma=0.1) for n in ("x", "y")])))
            model.save(run / "trained_models" / "model_it_3.keras")
            from run.storage import digest
            from sampling import modes as search

            network = run / "trained_models" / "model_it_3.keras"
            recorded = digest(network)
            original = search.posterior_modes

            def replaced_meanwhile(*args, **kwargs):
                # a training that ends while the search runs writes another network to the file
                result = original(*args, **kwargs)
                network.write_bytes(b"another network")
                return result

            with mock.patch.object(search, "posterior_modes", side_effect=replaced_meanwhile):
                main([str(run), "--seed", "1", "--tag", "check"])
            report = json.loads((run / "modes" / "modes_it_3_check.json").read_text())
            samples = np.load(run / "modes" / "modes_it_3_check.npz")
            # the result is recorded for the network that was searched, not the one on disk at the end
            self.assertNotEqual(digest(network), recorded)
            self.assertEqual(report["target"]["model_sha256"], recorded)
            self.assertEqual(report["target"]["region"], None)
            self.assertEqual(json.loads(str(samples["target_json"].item())), report["target"])
        self.assertEqual((report["iteration"], report["decided_by"], len(report["modes"])), (3, "measured mixing", 2))
        expected = 1.0 / (1.0 + np.exp(-offset) * (sigma[1] / sigma[0]) ** 2)        # the weight of the wider peak
        self.assertAlmostEqual(report["modes"][0]["weight"], expected, delta=0.03)
        np.testing.assert_allclose(report["modes"][0]["mean"], centre[0], atol=0.01)
        np.testing.assert_allclose(samples["weights"], [m["weight"] for m in report["modes"]])
        self.assertEqual(samples["x0"].shape[1], 2)
        self.assertFalse(report["search"]["mixing"]["pairs"][0]["one_mode"])
        self.assertEqual(report["search"]["mixing"]["threshold"], 0.2)       # the run's own setting, no --mixing given


if __name__ == "__main__":
    unittest.main()
