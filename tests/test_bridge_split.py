"""sampling.modes.bridge_log_evidence on a chain: the mixture is fitted to one half of the walkers
and bridged with the other. Random halves of the rows of an autocorrelated chain are not
independent, and the evidence came out low (0.27 to 0.29 in ln Z on two 44-parameter Euclid
networks, 21 November 2026)."""
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import tensorflow as tf

import sampling.modes as modes
from sampling.modes import _halves, bridge_log_evidence


def ar1_chain(rng, ndim, walkers, steps, rho, sigma, centre):
    """(steps, walkers, ndim): every walker an AR(1) process with stationary law N(centre, sigma^2),
    started in it; its autocorrelation time is (1 + rho) / (1 - rho) steps."""
    z = rng.normal(size=(walkers, ndim))
    chain = np.empty((steps, walkers, ndim))
    for t in range(steps):
        z = rho * z + np.sqrt(1 - rho ** 2) * rng.normal(size=z.shape)
        chain[t] = z
    return centre + sigma * chain


class Halves(unittest.TestCase):
    def test_the_two_halves_share_no_walker_and_change_with_the_repeat(self):
        walkers, steps = 7, 5
        walker = np.arange(walkers * steps) % walkers      # rows as chain().reshape(-1, ndim) has them
        rng = np.random.default_rng(0)
        seen = set()
        for repeat in range(6):
            fit, use = _halves(walkers * steps, walkers, repeat, rng)
            self.assertEqual(sorted(np.concatenate([fit, use])), list(range(walkers * steps)))
            self.assertFalse(set(walker[fit]) & set(walker[use]))
            self.assertEqual(len(set(walker[fit])), walkers // 2)
            seen.add(tuple(sorted(set(walker[fit]))))
        self.assertGreater(len(seen), 1)

    def test_one_walker_is_halved_in_time_in_turn(self):
        rng = np.random.default_rng(0)
        first = _halves(11, 1, 0, rng)
        np.testing.assert_array_equal(first[0], np.arange(5))
        np.testing.assert_array_equal(first[1], np.arange(5, 11))
        second = _halves(11, 1, 1, rng)
        np.testing.assert_array_equal(second[0], first[1])
        np.testing.assert_array_equal(second[1], first[0])

    def test_samples_that_are_not_a_chain_of_that_many_walkers_are_refused(self):
        x = np.zeros((30, 2))
        values = lambda y: np.zeros(len(y))
        inside = lambda y: np.ones(len(y), bool)
        for walkers in (0, 4, 30):
            with self.assertRaisesRegex(ValueError, "not a chain"):
                bridge_log_evidence(x, values(x), values, inside, [-1, -1], [1, 1], 1, np.random.default_rng(0),
                                    walkers=walkers)


class AutocorrelatedChain(unittest.TestCase):
    def test_halves_by_walker_are_unbiased_where_random_halves_are_low(self):
        # 16 parameters, 64 walkers of 1000 stored steps with an autocorrelation time of 199 steps:
        # some 320 independent states in 64 000. Over 20 seeds the random halves gave
        # -0.159 +- 0.019 (none above -0.12), the halves by walker +0.004 +- 0.019, and the spread
        # of the walker halvings (0.019 on average) the scatter between seeds.
        ndim, walkers, steps, sigma = 16, 64, 1000, 0.05
        rng = np.random.default_rng(0)
        x = ar1_chain(rng, ndim, walkers, steps, 0.99, sigma, 0.5).reshape(-1, ndim)
        values = lambda y: -0.5 * np.sum((np.asarray(y) - 0.5) ** 2, axis=1) / sigma ** 2
        inside = lambda y: np.all((y >= 0) & (y <= 1), axis=1)
        exact = 0.5 * ndim * np.log(2 * np.pi * sigma ** 2)
        box = (np.zeros(ndim), np.ones(ndim), 2, rng)
        rows, _, _ = bridge_log_evidence(x, values(x), values, inside, *box, n=10000)
        by_walker, spread, efficiency = bridge_log_evidence(x, values(x), values, inside, *box, n=10000,
                                                            walkers=walkers)
        self.assertLess(rows - exact, -0.09)
        self.assertAlmostEqual(by_walker, exact, delta=0.07)
        self.assertGreater(spread, 0.0)
        self.assertGreater(efficiency, 0.3)

    def test_one_long_walker(self):
        ndim, sigma = 4, 0.05
        rng = np.random.default_rng(1)
        x = ar1_chain(rng, ndim, 1, 40000, 0.9, sigma, 0.5)[:, 0]
        values = lambda y: -0.5 * np.sum((np.asarray(y) - 0.5) ** 2, axis=1) / sigma ** 2
        inside = lambda y: np.all((y >= 0) & (y <= 1), axis=1)
        got, _, _ = bridge_log_evidence(x, values(x), values, inside, np.zeros(ndim), np.ones(ndim), 1, rng,
                                        n=10000, walkers=1)
        self.assertAlmostEqual(got, 0.5 * ndim * np.log(2 * np.pi * sigma ** 2), delta=0.05)


class Callers(unittest.TestCase):
    """posterior_modes (the boosts) and sample_region_mixture (cells of a region) give the bridge
    their chain's walkers."""

    def asked(self):
        calls = []

        def recorded(*arguments, **options):
            calls.append(options)
            return bridge_log_evidence(*arguments, **options)
        return calls, mock.patch.object(modes, "bridge_log_evidence", recorded)

    def test_the_boosts_mode_chains(self):
        calls, patch = self.asked()
        tempered = np.clip(0.3 + 0.1 * np.random.default_rng(0).normal(size=(2000, 4)), 0, 1).astype(np.float32)
        peak = lambda x: -0.5 * tf.reduce_sum(((x - 0.3) / 0.04) ** 2, axis=1)
        with patch:
            found, _ = modes.posterior_modes(peak, np.zeros(4), np.ones(4), tempered, 7.0, seed=3, log=lambda *_: None,
                                             walkers=64, cool_steps=300, steps=1000, components=1)
        self.assertEqual([c.get("walkers") for c in calls], [64])
        self.assertEqual(found[0]["stored_samples"] % 64, 0)

    def test_the_cells_of_a_region(self):
        from sampling.mixture import sample_region_mixture
        from sampling.region import Region

        rng = np.random.default_rng(42)
        region = Region.from_clouds([rng.normal([m, 0.0], 0.1, (500, 2)) for m in (-0.5, 0.5)], [-2, -2], [2, 2], 2, 0)

        def predict(x):
            a = -0.5 * tf.reduce_sum(((x - [-0.5, 0.0]) / 0.1) ** 2, axis=1)
            b = -0.5 * tf.reduce_sum(((x - [0.5, 0.0]) / 0.1) ** 2, axis=1)
            return tf.reduce_logsumexp(tf.stack([a, b]), axis=0)

        surrogate = SimpleNamespace(region=region, logpost=region.restrict(predict))
        sampling = SimpleNamespace(n_walkers=32, burn_in=200, chunk_size=1000, progress=False)
        calls, patch = self.asked()
        tf.keras.utils.set_random_seed(3)
        with patch:
            sample_region_mixture(surrogate, [[-2, 2], [-2, 2]], sampling, 1000, 1, seed=1)
        self.assertEqual([c.get("walkers") for c in calls], [32, 32])


if __name__ == "__main__":
    unittest.main()
