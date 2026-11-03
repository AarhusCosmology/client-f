"""acquisition.boost: the modes of a posterior, their T = 1 samples and evidences, and the boost points."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from config.config import AcquisitionConfig, AuditConfig, BoostConfig
from dataset.boost import boost_report, draw_boost, finish_boost, mode_shares, split_by_shares
from sampling.modes import (NoModeFound, Regions, as_numpy, bridge_log_evidence, climb, find_modes,
                            posterior_modes, split_evenly)

NDIM = 4
LOWER, UPPER = np.zeros(NDIM), np.ones(NDIM)
A, B = np.full(NDIM, 0.3), np.full(NDIM, 0.7)
SIGMA_A, SIGMA_B = 0.04, 0.02


def two_peaks(offset=1.0):
    """log of exp(-|x-A|^2 / 2 sA^2) + exp(-offset) exp(-|x-B|^2 / 2 sB^2): the peak at B is
    `offset` lower; the valley between them is hundreds of log-units deep."""
    def predict(x):
        a = -0.5 * tf.reduce_sum(((x - 0.3) / SIGMA_A) ** 2, axis=1)
        b = -offset - 0.5 * tf.reduce_sum(((x - 0.7) / SIGMA_B) ** 2, axis=1)
        return tf.reduce_logsumexp(tf.stack([a, b]), axis=0)
    return predict


def one_peak(x):
    return -0.5 * tf.reduce_sum(((x - 0.3) / SIGMA_A) ** 2, axis=1)


def tempered(n=4000, seed=0):
    """Points spread widely around both peaks, as a hot chain leaves them."""
    rng = np.random.default_rng(seed)
    centres = np.where(rng.random((n, 1)) < 0.7, A, B)
    return np.clip(centres + 0.1 * rng.normal(size=(n, NDIM)), 0.0, 1.0).astype(np.float32)


SETTINGS = dict(walkers=64, cool_steps=300, steps=2000, components=1)


class CooledPeaks(unittest.TestCase):
    """sampling.modes.cooled_peaks: where a tempered chain's walkers end when cooled and climbed,
    for the audit's chain search (acquisition.audit.chain)."""

    def test_the_summits_of_both_peaks_and_the_walkers_that_climbed_to_them(self):
        from sampling.modes import as_numpy, cooled_peaks

        tf.keras.utils.set_random_seed(4)
        predict = two_peaks()
        summits, walkers = cooled_peaks(predict, LOWER, UPPER, tempered(), 7.0, walkers=64, cool_steps=300, seed=2)
        self.assertEqual((summits.shape, walkers.shape), ((64, NDIM), (64, NDIM)))
        to_a, to_b = (np.abs(summits - A).max(axis=1) < 0.01), (np.abs(summits - B).max(axis=1) < 0.01)
        self.assertTrue(np.all(to_a | to_b) and to_a.any() and to_b.any())            # every walker climbed to one of the two
        values = as_numpy(predict)
        self.assertTrue(np.all(values(summits) >= values(walkers) - 1e-4))            # and no lower than where it stood
        self.assertGreater(np.abs(walkers - summits).max(), 0.01)                     # the walkers are T = 1 points, not summits
        # walker for walker what the boost's mode search starts from, with the same seed
        again, _ = cooled_peaks(predict, LOWER, UPPER, tempered(), 7.0, walkers=64, cool_steps=300, seed=2)
        self.assertEqual(again.shape, summits.shape)


class Configuration(unittest.TestCase):
    def test_absent_by_default_and_needs_n_and_start(self):
        base = dict(n_append=10, n_neighbors=5, target_temperature=7.0, pool_factor=5)
        self.assertIsNone(AcquisitionConfig.from_dict(base).boost)
        got = AcquisitionConfig.from_dict(dict(base, boost=dict(n=1000, start_iteration=15, barrier=3.5))).boost
        self.assertEqual((got.n, got.start_iteration, got.barrier), (1000, 15, 3.5))
        # min_share: 0 (every group of walkers a mode) is allowed, as BoostConfig says; the
        # general check of the settings refused it
        self.assertEqual(BoostConfig.from_dict(dict(n=10, start_iteration=3, min_share=0)).min_share, 0.0)
        for bad in (dict(n=1000), dict(start_iteration=3), dict(n=0, start_iteration=3), dict(n=10, start_iteration=3, min_share=1),
                    dict(n=10, start_iteration=3, walkers=7), dict(n=10, start_iteration=3, other=1)):
            with self.assertRaises(ValueError):
                BoostConfig.from_dict(bad)

    def test_false_switches_an_opt_in_section_off(self):
        # `audit: false` used to give the audit with every default, the reverse of what it says.
        base = dict(n_append=10, n_neighbors=5, target_temperature=7.0, pool_factor=5)
        off = AcquisitionConfig.from_dict(dict(base, audit=False, boost=False))
        self.assertIsNone(off.audit)
        self.assertIsNone(off.boost)
        for on in (None, True, {}):
            self.assertEqual(AcquisitionConfig.from_dict(dict(base, audit=on)).audit, AuditConfig())
        for bad in (0, "", "false", []):
            with self.subTest(audit=bad), self.assertRaises(ValueError):
                AcquisitionConfig.from_dict(dict(base, audit=bad))

    def test_mode_depth(self):
        base = dict(n=10, start_iteration=2)
        self.assertIsNone(BoostConfig.from_dict(base).mode_depth)
        self.assertEqual(BoostConfig.from_dict(dict(base, mode_depth=84)).mode_depth, 84.0)
        self.assertEqual(BoostConfig.from_dict(dict(base, mode_depth="auto")).mode_depth, "auto")
        for wrong in (0, -5, True, "deep", "Auto"):
            with self.assertRaisesRegex(ValueError, "mode_depth"):
                BoostConfig.from_dict(dict(base, mode_depth=wrong))

    def test_balance_floor(self):
        base = dict(n=10, start_iteration=3)
        self.assertIsNone(BoostConfig.from_dict(base).balance_floor)
        self.assertEqual(BoostConfig.from_dict(dict(base, balance_floor=0.1)).balance_floor, 0.1)
        self.assertEqual(BoostConfig.from_dict(dict(base, balance_floor=1)).balance_floor, 1.0)
        for wrong in (0, -0.1, 1.5, True, "auto"):
            with self.assertRaises(ValueError):
                BoostConfig.from_dict(dict(base, balance_floor=wrong))

    def test_shares(self):
        self.assertIsNone(mode_shares([0.976, 0.013, 0.011], None))
        # the H0WC axion run's network 8 (30 October 2026): its two light modes at 13% and 11%
        np.testing.assert_allclose(mode_shares([0.976, 0.013, 0.011], 0.1), [1, 0.13, 0.11])
        np.testing.assert_array_equal(mode_shares([0.6, 0.3, 0.1], 0.1), [1, 1, 1])
        np.testing.assert_allclose(mode_shares([0.6, 0.4], 1.0), [1, 0.4 / 0.6])        # floor 1: by weight
        self.assertEqual(split_by_shares(101, [1, 1]), split_evenly(101, 2))
        self.assertEqual(split_by_shares(100, [1, 0.13, 0.11]), [81, 10, 9])
        self.assertEqual(split_by_shares(10, [1, 1e-9, 1e-9]), [8, 1, 1])               # one point at least
        with self.assertRaises(ValueError):
            mode_shares([0.0, 0.0], 0.1)

    def test_split(self):
        self.assertEqual(split_evenly(1000, 2), [500, 500])
        self.assertEqual(split_evenly(1000, 1), [1000])
        self.assertEqual(split_evenly(13, 3), [5, 4, 4])


class Modes(unittest.TestCase):
    def test_climb_reaches_the_peaks(self):
        start = np.array([A + 0.05, B - 0.03], np.float32)
        end = climb(two_peaks(), start, LOWER, UPPER)
        np.testing.assert_allclose(end, [A, B], atol=3e-3)

    def test_two_peaks_are_two_modes_and_a_stalled_point_does_not_join_them(self):
        values = as_numpy(two_peaks())
        rng = np.random.default_rng(1)
        peaks = np.concatenate([A + 0.002 * rng.normal(size=(30, NDIM)), B + 0.001 * rng.normal(size=(10, NDIM))])
        labels, count = find_modes(values, peaks, barrier=4.0)
        self.assertEqual(count, 2)
        self.assertTrue(np.all(labels[:30] == 0) and np.all(labels[30:] == 1))
        # a point on the slope between them, joined to a peak but far below it: left out
        slope = (A + 0.6 * SIGMA_A * 5 * (B - A) / np.linalg.norm(B - A))[None]
        labels, count = find_modes(values, np.concatenate([peaks, slope]), barrier=4.0)
        self.assertEqual((count, labels[-1]), (2, -1))

    def test_one_peak_is_one_mode_and_a_low_barrier_joins(self):
        values = as_numpy(one_peak)
        peaks = A + 0.002 * np.random.default_rng(2).normal(size=(20, NDIM))
        self.assertEqual(find_modes(values, peaks, barrier=4.0)[1], 1)
        # two bumps 2 sigma apart along one axis: the dip between them is under a log-unit
        shifted = np.array(A); shifted[0] += 2 * SIGMA_A
        def ridge(x):
            a = -0.5 * tf.reduce_sum(((x - 0.3) / SIGMA_A) ** 2, axis=1)
            b = -0.5 * tf.reduce_sum(((x - tf.constant(shifted, tf.float32)) / SIGMA_A) ** 2, axis=1)
            return tf.reduce_logsumexp(tf.stack([a, b]), axis=0)
        both = np.concatenate([peaks, shifted + 0.002 * np.random.default_rng(3).normal(size=(20, NDIM))])
        self.assertEqual(find_modes(as_numpy(ridge), both, barrier=4.0)[1], 1)

    def test_cooling_hands_on_the_ensemble_of_its_last_step(self):
        # Each stage runs `steps` steps and stores every quarter of them, counted from the
        # first: the last stored row is the ensemble at three quarters, which the next stage
        # used to start from.
        from sampling.modes import _cool, _ensemble

        sampler, beta, _ = _ensemble(one_peak, LOWER, UPPER, 16)
        start = np.clip(A + 0.05 * np.random.default_rng(1).normal(size=(16, NDIM)), 0.0, 1.0).astype(np.float32)
        end = _cool(sampler, beta, start, [4.0, 1.0], 400)
        self.assertEqual(sampler._n_proposals, 400)                       # the last stage, in full
        np.testing.assert_array_equal(end, sampler.last_positions())
        self.assertEqual(len(sampler.chain()), 4)
        self.assertFalse(np.array_equal(end, sampler.chain()[-1]))        # the row of step 301, not the end
        self.assertEqual(float(beta.numpy()), 1.0)

    def test_regions_divide_the_box(self):
        rng = np.random.default_rng(4)
        regions = Regions([A + SIGMA_A * rng.normal(size=(200, NDIM)), B + SIGMA_B * rng.normal(size=(200, NDIM))], LOWER, UPPER)
        points = np.array([A, B, A + 0.05, B - 0.03])
        np.testing.assert_array_equal(regions.labels(points), [0, 1, 0, 1])
        np.testing.assert_array_equal(regions.labels_tf(tf.constant(points, tf.float32)).numpy(), [0, 1, 0, 1])

    def test_bridge_gives_a_gaussian_integral(self):
        rng = np.random.default_rng(5)
        x = A + SIGMA_A * rng.normal(size=(20000, NDIM))
        values = as_numpy(one_peak)
        inside = lambda y: np.all((y >= LOWER) & (y <= UPPER), axis=1)
        got, spread, efficiency = bridge_log_evidence(x, values(x), values, inside, LOWER, UPPER, 1, rng, n=5000)
        self.assertAlmostEqual(got, 0.5 * NDIM * np.log(2 * np.pi * SIGMA_A ** 2), delta=0.03)
        self.assertGreater(efficiency, 0.3)

    def test_bridge_proposal_is_drawn_from_the_whole_mixture(self):
        # Two lobes in one region, weights 3 to 1, a mixture of two components. scikit-learn
        # returns a mixture's draws component by component, and the first n of them used to
        # be the proposal sample: one lobe's points, weighed by the density of both.
        rng = np.random.default_rng(6)
        a, b, sigma = np.full(NDIM, 0.3), np.full(NDIM, 0.7), 0.03
        def values(y):
            y = np.asarray(y, float)
            return np.logaddexp(np.log(0.75) - 0.5 * np.sum((y - a) ** 2, axis=1) / sigma ** 2,
                                np.log(0.25) - 0.5 * np.sum((y - b) ** 2, axis=1) / sigma ** 2)
        x = np.where(rng.random((20000, 1)) < 0.75, a, b) + sigma * rng.normal(size=(20000, NDIM))
        asked = []
        def recorded(y):
            asked.append(np.asarray(y, float))
            return values(y)
        inside = lambda y: np.all((y >= 0.0) & (y <= 1.0), axis=1)
        got, _, efficiency = bridge_log_evidence(x, values(x), recorded, inside, np.zeros(NDIM), np.ones(NDIM), 2, rng, n=5000)
        self.assertAlmostEqual(got, 0.5 * NDIM * np.log(2 * np.pi * sigma ** 2), delta=0.03)
        for proposal in asked:
            self.assertAlmostEqual(np.mean(np.linalg.norm(proposal - a, axis=1) < np.linalg.norm(proposal - b, axis=1)), 0.75, delta=0.03)

    def test_weights_of_two_modes(self):
        offset = 1.0
        modes, info = posterior_modes(two_peaks(offset), LOWER, UPPER, tempered(), 7.0, seed=3, log=lambda *_: None, **SETTINGS)
        self.assertEqual(len(modes), 2)
        expected = 1.0 / (1.0 + np.exp(-offset) * (SIGMA_B / SIGMA_A) ** NDIM)  # the weight of the peak at A
        self.assertAlmostEqual(modes[0]["weight"], expected, delta=0.02)
        np.testing.assert_allclose(modes[0]["x"].mean(axis=0), A, atol=0.01)
        np.testing.assert_allclose(modes[1]["x"].mean(axis=0), B, atol=0.01)
        np.testing.assert_allclose(modes[1]["x"].std(axis=0), SIGMA_B, rtol=0.2)

    def test_a_mode_far_below_the_highest_is_left_out_with_mode_depth(self):
        deep = two_peaks(20.0)                                  # the peak at B is 20 below the one at A
        modes, info = posterior_modes(deep, LOWER, UPPER, tempered(), 7.0, seed=3, log=lambda *_: None, **SETTINGS)
        self.assertEqual(len(modes), 2)
        self.assertNotIn("deep_modes", info)                    # without the setting, as before
        said = []
        for extra in ({}, {"mixing": 0.1}):
            modes, info = posterior_modes(deep, LOWER, UPPER, tempered(), 7.0, seed=3, log=said.append,
                                          mode_depth=10.0, **extra, **SETTINGS)
            self.assertEqual(len(modes), 1)
            self.assertEqual(modes[0]["weight"], 1.0)
            np.testing.assert_allclose(modes[0]["x"].mean(axis=0), A, atol=0.01)
            (gone,) = info["deep_modes"]
            self.assertAlmostEqual(gone["below"], 20.0, delta=0.5)
            self.assertGreater(gone["walkers"], 0)
            self.assertTrue(any("left out, whose highest maximum lies 20 below" in line for line in said), said)
        modes, info = posterior_modes(deep, LOWER, UPPER, tempered(), 7.0, seed=3, log=lambda *_: None,
                                      mode_depth=30.0, **SETTINGS)
        self.assertEqual((len(modes), info["deep_modes"]), (2, []))

    def test_mode_depth_auto_is_temperature_times_ndim_over_two(self):
        # T = 7, four parameters: 14. A mode 20 below is left out, one 12 below is kept.
        quiet = lambda *_: None
        modes, info = posterior_modes(two_peaks(20.0), LOWER, UPPER, tempered(), 7.0, seed=3, log=quiet,
                                      mode_depth="auto", **SETTINGS)
        self.assertEqual((len(modes), info["mode_depth"], len(info["deep_modes"])), (1, 14.0, 1))
        self.assertAlmostEqual(info["deep_modes"][0]["below"], 20.0, delta=0.5)
        modes, info = posterior_modes(two_peaks(12.0), LOWER, UPPER, tempered(), 7.0, seed=3, log=quiet,
                                      mode_depth="auto", **SETTINGS)
        self.assertEqual((len(modes), info["mode_depth"], info["deep_modes"]), (2, 14.0, []))
        with self.assertRaisesRegex(ValueError, "mode_depth"):
            posterior_modes(two_peaks(20.0), LOWER, UPPER, tempered(), 7.0, seed=3, log=quiet,
                            mode_depth="deep", **SETTINGS)

    def test_one_mode(self):
        modes, _ = posterior_modes(one_peak, LOWER, UPPER, tempered(), 7.0, seed=3, log=lambda *_: None, **SETTINGS)
        self.assertEqual(len(modes), 1)
        self.assertEqual(modes[0]["weight"], 1.0)
        np.testing.assert_allclose(modes[0]["x"].std(axis=0), SIGMA_A, rtol=0.2)


class Boost(unittest.TestCase):
    def config(self, n):
        return BoostConfig.from_dict(dict(n=n, start_iteration=0, **SETTINGS))

    def test_same_number_from_every_mode(self):
        points, draw = draw_boost(two_peaks(), LOWER, UPPER, tempered(), 7.0, self.config(101), seed=1, log=lambda *_: None)
        self.assertEqual(len(points), 101)
        self.assertEqual(np.bincount(draw["label"]).tolist(), [51, 50])
        self.assertEqual(len(np.unique(points, axis=0)), 101)
        near_a = np.linalg.norm(points - A, axis=1) < np.linalg.norm(points - B, axis=1)
        np.testing.assert_array_equal(near_a, draw["label"] == 0)
        np.testing.assert_allclose(draw["predicted"], as_numpy(two_peaks())(points), atol=1e-3)

    def test_a_light_mode_gets_its_weight_over_the_floor(self):
        """The peak at B holds about 2.3% of the posterior: with a floor of 0.1 its share is about
        0.23 of A's, where without it the two have the same number."""
        config = BoostConfig.from_dict(dict(n=101, start_iteration=0, balance_floor=0.1, **SETTINGS))
        lines = []
        points, draw = draw_boost(two_peaks(), LOWER, UPPER, tempered(), 7.0, config, seed=1, log=lines.append)
        counts = np.bincount(draw["label"]).tolist()
        self.assertEqual(sum(counts), 101)
        weight_b = draw["modes"][1]["weight"]
        self.assertAlmostEqual(draw["modes"][1]["share"], weight_b / 0.1, places=6)
        self.assertAlmostEqual(counts[1] / counts[0], weight_b / 0.1, delta=0.05)
        self.assertTrue(any("balance floor 0.1" in line for line in lines))

    def test_a_floor_below_every_weight_changes_nothing(self):
        """On the same modes (the mode search's chains are not repeatable between calls), a floor
        under every weight draws the same points as no floor; without the key the record is as before."""
        from unittest import mock
        found = posterior_modes(two_peaks(), LOWER, UPPER, tempered(), 7.0, seed=1, log=lambda *_: None, **SETTINGS)
        floored_config = BoostConfig.from_dict(dict(n=101, start_iteration=0, balance_floor=0.01, **SETTINGS))
        with mock.patch("dataset.boost.posterior_modes", return_value=found):
            default, draw = draw_boost(two_peaks(), LOWER, UPPER, tempered(), 7.0, self.config(101), seed=1, log=lambda *_: None)
            floored, draw_floored = draw_boost(two_peaks(), LOWER, UPPER, tempered(), 7.0, floored_config, seed=1,
                                               log=lambda *_: None)
        np.testing.assert_array_equal(default, floored)
        np.testing.assert_array_equal(draw["label"], draw_floored["label"])
        self.assertEqual(np.bincount(draw["label"]).tolist(), [51, 50])
        self.assertNotIn("share", draw["modes"][0])
        self.assertEqual([m["share"] for m in draw_floored["modes"]], [1.0, 1.0])

    def test_one_mode_gets_all(self):
        points, draw = draw_boost(one_peak, LOWER, UPPER, tempered(), 7.0, self.config(60), seed=1, log=lambda *_: None)
        self.assertEqual((len(points), len(draw["modes"])), (60, 1))

    def test_a_deep_mode_gets_none_with_mode_depth(self):
        config = BoostConfig.from_dict(dict(n=60, start_iteration=0, mode_depth=10, **SETTINGS))
        points, draw = draw_boost(two_peaks(20.0), LOWER, UPPER, tempered(), 7.0, config, seed=1, log=lambda *_: None)
        self.assertEqual((len(points), len(draw["modes"])), (60, 1))
        self.assertTrue(np.all(np.linalg.norm(points - A, axis=1) < np.linalg.norm(points - B, axis=1)))
        self.assertEqual(len(draw["info"]["deep_modes"]), 1)

    def test_report_corrects_the_weights(self):
        """A network 0.7 too low at B: the true weight of B is exp(0.7) larger in odds."""
        points, draw = draw_boost(two_peaks(), LOWER, UPPER, tempered(), 7.0, self.config(200), seed=2, log=lambda *_: None)
        truth = draw["predicted"] + np.where(draw["label"] == 1, 0.7, 0.0)
        report = boost_report(draw, truth)
        w = draw["modes"][1]["weight"]
        expected = w * np.exp(0.7) / (w * np.exp(0.7) + 1 - w)
        self.assertAlmostEqual(report["modes"][1]["corrected_weight"], expected, places=6)
        self.assertAlmostEqual(report["modes"][1]["error_mean"], -0.7, places=5)
        self.assertAlmostEqual(report["modes"][1]["effective_points"], 100, places=3)

    def test_files_and_failed_points(self):
        points, draw = draw_boost(two_peaks(), LOWER, UPPER, tempered(), 7.0, self.config(40), seed=2, log=lambda *_: None)
        truth = draw["predicted"].copy()
        truth[0] = np.nan
        order = np.random.default_rng(0).permutation(len(points))  # the evaluation may return them in any order
        with tempfile.TemporaryDirectory() as tmp:
            report = finish_boost(Path(tmp) / "boost", 7, list("abcd"), draw, points[order], truth[order], log=lambda *_: None)
            import pandas as pd
            frame = pd.read_csv(Path(tmp) / "boost" / "boost_it_7.csv")
            self.assertTrue((Path(tmp) / "boost" / "boost_it_7.json").exists())
        self.assertEqual(list(frame.columns), list("abcd") + ["loglkl", "predicted", "mode"])
        self.assertEqual((report["failed"], report["modes"][0]["failed"]), (1, 1))
        self.assertEqual((report["error"]["n"], report["error"]["left_out"]), (39, 1))
        np.testing.assert_allclose(frame["loglkl"].to_numpy()[1:], truth[1:], rtol=1e-6)
        self.assertEqual(frame["mode"].value_counts().to_dict(), {1: 20, 2: 20})

    def test_a_bridge_without_a_finite_estimate_stops_before_any_weight_is_made(self):
        from sampling.modes import bridge_log_evidence, mode_log_evidence

        rng = np.random.default_rng(0)
        x = 0.5 + 0.1 * rng.normal(size=(4000, 2))
        values = lambda y: -50.0 * np.sum((np.asarray(y) - 0.5) ** 2, axis=1)
        everywhere = lambda y: np.ones(len(y), bool)
        nowhere = lambda y: np.zeros(len(y), bool)
        box = ([0.0, 0.0], [1.0, 1.0], 2)
        self.assertTrue(np.isfinite(mode_log_evidence(0, x, values(x), values, everywhere, *box, rng)).all())
        log_z, _, efficiency = bridge_log_evidence(x, values(x), values, nowhere, *box, rng)
        self.assertTrue(np.isnan(log_z) and efficiency == 0)
        with self.assertRaisesRegex(ValueError, "mode 2: the bridge estimate"):
            mode_log_evidence(1, x, values(x), values, nowhere, *box, rng)
        broken = lambda y: np.full(len(y), np.nan)
        with self.assertRaisesRegex(ValueError, "mode 1: the bridge estimate"):
            mode_log_evidence(0, x, values(x), broken, everywhere, *box, rng)

    def test_a_point_a_launcher_moved_by_the_last_digit_is_the_drawn_point(self):
        """The launcher (hpc/launch.py) rounds a batch to single precision and keeps it inside the
        box as single precision holds it: a boost point in the last digit before an edge is
        evaluated one step inside. Matched bit for bit, it was "unexpected" and the run stopped
        at its first boost; on a network pressed against its box, at every one."""
        points, draw = draw_boost(two_peaks(), LOWER, UPPER, tempered(), 7.0, self.config(40), seed=2, log=lambda *_: None)
        truth = draw["predicted"] + 0.25
        evaluated = np.array(points, np.float32)
        evaluated[3, 1] = np.nextafter(evaluated[3, 1], np.float32(-np.inf))          # one step, as the launcher's clip
        evaluated[17, 0] = np.nextafter(evaluated[17, 0], np.float32(np.inf))
        order = np.random.default_rng(1).permutation(len(points))
        lines = []
        with tempfile.TemporaryDirectory() as tmp:
            report = finish_boost(Path(tmp) / "boost", 7, list("abcd"), draw, evaluated[order].astype(float), truth[order],
                                  log=lines.append)
            import pandas as pd
            frame = pd.read_csv(Path(tmp) / "boost" / "boost_it_7.csv")
            self.assertEqual(report["failed"], 0)
            np.testing.assert_allclose(frame["loglkl"].to_numpy(), truth, rtol=1e-6)               # each value at its own point
            # recorded where it was evaluated, which is where the training set has it
            np.testing.assert_array_equal(frame[list("abcd")].to_numpy(np.float32), evaluated)
            self.assertIn("2 of the 40 boost points were evaluated one step of single precision", "\n".join(lines))
            # a point that is not a drawn one, by more than the last digit, still stops the run
            far = evaluated.copy()
            far[5, 2] += np.float32(1e-3)
            with self.assertRaisesRegex(ValueError, "unexpected or extra boost evaluation coordinate"):
                finish_boost(Path(tmp) / "far", 7, list("abcd"), draw, far, truth, log=lambda *_: None)
            # and so does one more point than was drawn
            with self.assertRaisesRegex(ValueError, "unexpected or extra boost evaluation coordinate"):
                finish_boost(Path(tmp) / "extra", 7, list("abcd"), draw, np.vstack([points, points[:1]]),
                             np.append(truth, 0.0), log=lambda *_: None)
            # without a moved point nothing is said and the file is the draw, as before
            quiet = []
            finish_boost(Path(tmp) / "same", 7, list("abcd"), draw, points, truth, log=quiet.append)
            self.assertNotIn("single precision", "\n".join(quiet))
            pd.DataFrame(draw["points"], columns=list("abcd")).to_csv(Path(tmp) / "draw.csv", index=False)
            written = pd.read_csv(Path(tmp) / "same" / "boost_it_7.csv")[list("abcd")]
            written.to_csv(Path(tmp) / "written.csv", index=False)
            self.assertEqual((Path(tmp) / "written.csv").read_text(), (Path(tmp) / "draw.csv").read_text())

    def test_a_point_the_likelihood_could_not_evaluate_is_left_out_of_the_error(self):
        """A launcher may give such a point a fixed finite value; the network's error at the
        boost points must not be the distance to that value."""
        points, draw = draw_boost(two_peaks(), LOWER, UPPER, tempered(), 7.0, self.config(40), seed=2, log=lambda *_: None)
        truth = draw["predicted"] + 0.25
        truth[3] = -30000.0
        lines = []
        with tempfile.TemporaryDirectory() as tmp:
            report = finish_boost(Path(tmp) / "boost", 7, list("abcd"), draw, points, truth, log=lines.append, ignore=-30000.0)
            counted = finish_boost(Path(tmp) / "other", 7, list("abcd"), draw, points, truth, log=lambda *_: None)
        self.assertEqual((report["error"]["n"], report["error"]["left_out"]), (39, 1))
        self.assertAlmostEqual(report["error"]["rms"], 0.25, places=6)
        self.assertGreater(counted["error"]["rms"], 1000)
        self.assertTrue(any("rms 0.25" in line and "at 39 points, 1 left out" in line for line in lines), lines)


class Fallback(unittest.TestCase):
    """A network without a mode to draw posterior samples from: the boost iteration acquires
    as an ordinary one does (client.draw_boost_points)."""

    def draw(self, directory, **boost):
        import contextlib
        import io
        from types import SimpleNamespace

        import client

        cfg = SimpleNamespace(seed=5, sampling=SimpleNamespace(temperature=7.0),
                              acquisition=SimpleNamespace(boost=BoostConfig.from_dict(dict(n=40, start_iteration=0, **SETTINGS, **boost))))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = client.draw_boost_points(cfg, two_peaks(), LOWER, UPPER, tempered(), 3, None, directory, True)
        return result, out.getvalue()

    def test_no_group_reaches_min_share(self):
        # The hot points are 70% and 30% at the two peaks: neither group is 90% of the walkers.
        with self.assertRaises(NoModeFound) as raised:
            posterior_modes(two_peaks(), LOWER, UPPER, tempered(), 7.0, seed=3, log=lambda *_: None, min_share=0.9, **SETTINGS)
        self.assertIsInstance(raised.exception, RuntimeError)       # what it was before it had a name
        self.assertIn("no mode found", str(raised.exception))

    def test_an_iteration_without_a_mode_is_an_ordinary_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "boost"
            (points, draw, boosting), log = self.draw(directory, min_share=0.9)
            self.assertEqual((points, draw, boosting), (None, None, False))
            import json
            record = json.loads((directory / "fallback_it_3.json").read_text())
            self.assertEqual((record["iteration"], record["acquired"]), (3, "tempered batch"))
            self.assertIn("no mode found", record["reason"])
            self.assertEqual([p.name for p in directory.iterdir()], ["fallback_it_3.json"])
        self.assertIn("no mode found", log)
        self.assertIn("not ready for a boost: this iteration acquires as an ordinary one does instead", log)

    def test_a_boost_that_finds_its_modes_is_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "boost"
            (points, draw, boosting), log = self.draw(directory)
            self.assertFalse(directory.exists())
        self.assertTrue(boosting)
        self.assertEqual((len(points), np.bincount(draw["label"]).tolist()), (40, [20, 20]))
        self.assertNotIn("not ready", log)

    def test_the_seed_of_a_boost_is_the_one_the_loop_used(self):
        from types import SimpleNamespace
        from unittest import mock

        import client

        cfg = SimpleNamespace(seed=5, sampling=SimpleNamespace(temperature=7.0), acquisition=SimpleNamespace(boost=self_config()))
        with mock.patch("dataset.boost.draw_boost", return_value=("points", "draw")) as drawn:
            self.assertEqual(client.draw_boost_points(cfg, "predict", LOWER, UPPER, "tempered", 3, "dataset", None, True),
                             ("points", "draw", True))
        self.assertEqual(drawn.call_args.args, ("predict", LOWER, UPPER, "tempered", 7.0, cfg.acquisition.boost))
        self.assertEqual((drawn.call_args.kwargs["seed"], drawn.call_args.kwargs["dataset"]), (5 + 104729 * 4, "dataset"))

    def test_another_error_is_not_swallowed(self):
        from types import SimpleNamespace
        from unittest import mock

        import client

        cfg = SimpleNamespace(seed=None, sampling=SimpleNamespace(temperature=7.0), acquisition=SimpleNamespace(boost=self_config()))
        with tempfile.TemporaryDirectory() as tmp, mock.patch("dataset.boost.draw_boost", side_effect=RuntimeError("something else")):
            with self.assertRaises(RuntimeError):
                client.draw_boost_points(cfg, two_peaks(), LOWER, UPPER, tempered(), 0, None, Path(tmp) / "boost", True)
            self.assertFalse((Path(tmp) / "boost").exists())


def self_config():
    return BoostConfig.from_dict(dict(n=40, start_iteration=0, **SETTINGS))


if __name__ == "__main__":
    unittest.main()
