"""acquisition.audit: the search for a network's maxima and the true-evaluation loop."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from config.config import AcquisitionConfig, AuditConfig
from model.network import build_model
from training.audit import audit_optima, find_maxima, fine_tune, inside_float32, save_audit, summit_maxima, weights_digest
from training.losses import build_loss

LOWER, UPPER = np.zeros(4), np.ones(4)
CENTRE = np.full(4, 0.4)


def truth(points):
    """A Gaussian peak at CENTRE: log L = -0.5 * sum((x - 0.4)^2) / 0.05^2."""
    points = np.atleast_2d(points)
    return -0.5 * np.sum(((points - CENTRE) / 0.05) ** 2, axis=1)


class BumpModel(tf.keras.Model):
    """The truth plus a false optimum of height `height` at the vertex (1, 1, 1, 1)."""

    def __init__(self, height):
        super().__init__()
        self.height = height

    def call(self, x, training=False):
        peak = -0.5 * tf.reduce_sum(((x - 0.4) / 0.05) ** 2, axis=1, keepdims=True)
        bump = self.height * tf.exp(-0.5 * tf.reduce_sum(((x - 1.0) / 0.3) ** 2, axis=1, keepdims=True))
        return tf.maximum(peak, bump - 1.0e-3 * tf.reduce_sum((x - 1.0) ** 2, axis=1, keepdims=True))


def training_set(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.uniform(size=(n // 3, 4)), np.clip(CENTRE + 0.08 * rng.normal(size=(n - n // 3, 4)), 0, 1)])
    x = x[np.max(x, axis=1) < 0.8]  # nothing near the far vertex
    return x.astype(np.float32), truth(x).astype(np.float32)


class FindMaxima(unittest.TestCase):
    def test_finds_a_false_optimum_at_a_vertex(self):
        x, y = training_set()
        maxima = find_maxima(BumpModel(50.0), LOWER, UPPER, x, y, starts=64, steps=300, seed=1)
        top = maxima[0]
        self.assertGreater(top["excess"], 40.0)
        self.assertEqual(top["at_bound"], 4)
        np.testing.assert_allclose(top["x"], 1.0, atol=1e-5)
        self.assertGreater(top["distance"], 0.3)
        self.assertEqual(sum(m["excess"] > 10 for m in maxima), 1)
        # 64 uniform + 64 vertex + the best training points; most starts far out end here
        self.assertEqual(top["starts"], 64 + 64 + 128)
        self.assertGreater(top["hits"], 20)
        self.assertLessEqual(sum(m["hits"] for m in maxima), top["starts"])
        counts = {}
        find_maxima(BumpModel(50.0), LOWER, UPPER, x, y, starts=64, steps=300, seed=1, keep=1, counts=counts)
        self.assertEqual(sum(counts["hits"]), counts["starts"])  # every start is counted once
        self.assertEqual(sum(e > 10 for e in counts["excess"]), 1)
        self.assertEqual(counts["excess"], sorted(counts["excess"], reverse=True))

    def test_default_starts_scale_with_the_dimension(self):
        from training.audit import default_starts

        self.assertEqual((default_starts(16), default_starts(31)), (2048, 3968))
        x, y = training_set()
        maxima = find_maxima(BumpModel(-1.0e4), LOWER, UPPER, x, y, steps=5, seed=1)
        self.assertEqual(maxima[0]["starts"], 2 * 128 * 4 + 128)

    def test_a_faithful_network_has_none(self):
        x, y = training_set()
        maxima = find_maxima(BumpModel(-1.0e4), LOWER, UPPER, x, y, starts=64, steps=300, seed=1)
        self.assertLess(maxima[0]["excess"], 1.0)
        self.assertLess(maxima[0]["distance"], 0.1)

    def test_points_are_single_precision_inside_the_box(self):
        lower, upper = np.array([0.1, -0.8]), np.array([0.7, 0.3])
        p = inside_float32([[0.1, 0.3], [0.7000001, -0.8000001]], lower, upper)
        self.assertTrue(np.all(p >= lower) and np.all(p <= upper))
        np.testing.assert_array_equal(p, p.astype(np.float32).astype(float))


class ManyBumps(tf.keras.Model):
    """The truth plus a false optimum at each of six vertices."""

    VERTICES = np.array([[1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]], np.float32)

    def call(self, x, training=False):
        value = -0.5 * tf.reduce_sum(((x - 0.4) / 0.05) ** 2, axis=1, keepdims=True)
        for j, v in enumerate(self.VERTICES):
            d2 = tf.reduce_sum((x - v) ** 2, axis=1, keepdims=True)
            value = tf.maximum(value, (40.0 + 5.0 * j) * tf.exp(-0.5 * d2 / 0.15 ** 2) - 1.0e-3 * d2)
        return value


class AuditLoop(unittest.TestCase):
    def test_quick_search_is_enough_while_maxima_are_plentiful(self):
        rng = np.random.default_rng(0)
        x = np.clip(CENTRE + 0.08 * rng.normal(size=(800, 4)), 0, 1).astype(np.float32)
        y = truth(x).astype(np.float32)
        seen = []

        def evaluate(points):
            seen.append(len(points))
            return points, truth(points), None

        config = AuditConfig(tau=10.0, k=3, rounds=1, epochs=1, starts=512, steps=300)
        nx, ny, nd, report = audit_optima(
            ManyBumps(), x, y, 0.1, LOWER, UPPER, evaluate, lambda t: "mse", config,
            batch_size=64, n_append=100, seed=1, log=lambda *_: None)
        first = report["rounds"][0]
        self.assertEqual((first["search"], first["starts"]), ("quick", 64 + 64 + 128))
        self.assertGreaterEqual(first["flagged"], 3)
        self.assertEqual(first["distinct_above_tau"], first["flagged"])  # below the cap of 50 they agree
        self.assertGreaterEqual(first["distinct_maxima"], first["distinct_above_tau"])
        self.assertTrue(0 < first["share_of_starts_above_tau"] <= 1)
        self.assertEqual(seen, [3])
        # a single flagged maximum is enough for the quick search to be used
        nx, ny, nd, report = audit_optima(
            BumpModel(50.0), x, y, 0.1, LOWER, UPPER, evaluate, lambda t: "mse", config,
            batch_size=64, n_append=100, seed=1, log=lambda *_: None)
        self.assertEqual((report["rounds"][0]["search"], report["rounds"][0]["flagged"]), ("quick", 1))
        # nothing flagged by the quick search: the full one gives the verdict
        nx, ny, nd, report = audit_optima(
            BumpModel(-1.0e4), x, y, 0.1, LOWER, UPPER, evaluate, lambda t: "mse", config,
            batch_size=64, n_append=100, seed=1, log=lambda *_: None)
        self.assertEqual((report["rounds"][0]["search"], report["rounds"][0]["starts"], report["clear"]),
                         ("full", 512 + 512 + 128, True))

    def test_evaluations_per_round_follow_the_batch(self):
        rng = np.random.default_rng(0)
        x = np.clip(CENTRE + 0.08 * rng.normal(size=(800, 4)), 0, 1).astype(np.float32)
        y = truth(x).astype(np.float32)
        seen = []

        def evaluate(points):
            seen.append(len(points))
            return points, truth(points), None

        # k 0 (the default): fraction * n_append / rounds, here 0.1 * 40 / 2
        config = AuditConfig(rounds=2, epochs=1, starts=512, steps=300)
        nx, ny, nd, report = audit_optima(
            ManyBumps(), x, y, 0.1, LOWER, UPPER, evaluate, lambda t: "mse", config,
            batch_size=64, n_append=40, seed=1, log=lambda *_: None)
        self.assertEqual((seen[0], report["k"]), (2, 2))
        self.assertLessEqual(report["true_calls"], 4)

    def test_keeps_the_best_round_not_the_last(self):
        from unittest import mock

        class Wobble(tf.keras.Model):
            """BumpModel whose height is a weight, so a fine-tune can be scripted."""

            def __init__(self):
                super().__init__()
                self.height = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(50.0), trainable=False)

            def call(self, x, training=False):
                peak = -0.5 * tf.reduce_sum(((x - 0.4) / 0.05) ** 2, axis=1, keepdims=True)
                bump = self.height * tf.exp(-0.5 * tf.reduce_sum(((x - 1.0) / 0.3) ** 2, axis=1, keepdims=True))
                return tf.maximum(peak, bump - 1.0e-3 * tf.reduce_sum((x - 1.0) ** 2, axis=1, keepdims=True))

        x, y = training_set()
        model = Wobble()
        model(x[:2])
        heights = iter([30.0, 80.0])  # the first fine-tune helps, the second makes it worse

        def scripted(model, *arguments):
            model.height.assign(next(heights))

        log = []
        config = AuditConfig(tau=10.0, k=1, rounds=2, epochs=1, starts=64, steps=300)
        with mock.patch("training.audit.fine_tune", scripted):
            nx, ny, nd, report = audit_optima(
                model, x, y, 0.1, LOWER, UPPER, lambda points: (points, truth(points), None),
                lambda t: "mse", config, batch_size=64, n_append=100, seed=1, log=log.append)
        highest = [round(r["highest_excess"]) for r in report["rounds"]]
        self.assertEqual(highest, [50, 30, 80])
        self.assertEqual((report["kept_round"], round(report["highest_excess"]), report["clear"]), (1, 30, False))
        self.assertAlmostEqual(float(model.height.numpy()), 30.0)
        self.assertEqual(len(ny), 2)  # the points of both rounds are returned
        self.assertTrue(any("the network of round 1 is kept" in line for line in log))

    def test_settling_rechecks_the_final_network_without_more_true_calls(self):
        from unittest import mock

        model = mock.Mock()
        state = [0]
        model.get_weights.side_effect = lambda: [state[0]]
        model.set_weights.side_effect = lambda weights: state.__setitem__(0, weights[0])
        searches = []

        def search(model, lower, upper, inputs, targets, **kwargs):
            searches.append((state[0], kwargs["starts"]))
            excess = [50.0, 5.0, 40.0][state[0]]
            kwargs["counts"].update(excess=[excess], hits=[1], starts=1)
            return [dict(x=np.ones(4), excess=excess, distance=1.0,
                         at_bound=4, starts=1, hits=1)]

        def tune(*args):
            state[0] += 1

        evaluate = mock.Mock(return_value=(np.ones((1, 4)), np.array([-1.0]), None))
        config = AuditConfig(rounds=1, k=1, starts=512, steps=1)
        with mock.patch("training.audit.find_maxima", side_effect=search), \
             mock.patch("training.audit.fine_tune", side_effect=tune), \
             mock.patch("training.audit._predict", side_effect=lambda m, x: np.full(len(x), [1., 3., 1.][state[0]])):
            _, _, _, report = audit_optima(
                model, np.zeros((10, 4)), np.zeros(10), 0.1, LOWER, UPPER,
                evaluate, lambda t: "mse", config, 64, 100, seed=1, log=lambda *_: None)
        self.assertEqual(state[0], 2)  # settling improved the fit, but regrew a maximum
        self.assertEqual(report["settling_rounds"], 1)
        self.assertFalse(report["clear"])
        self.assertEqual(report["highest_excess"], 40.0)
        self.assertEqual(report["final_search"]["distinct_above_tau"], 1)
        self.assertEqual(searches[-1], (2, 512))  # full search on the saved weights
        evaluate.assert_called_once()
        self.assertEqual(report["true_calls"], 1)

    def trained(self, seed=3):
        """A small network trained on the peak only, then given a false optimum at the far
        vertex by a few steps on fake high targets there."""
        tf.keras.utils.set_random_seed(seed)
        x, y = training_set(seed=seed)
        model = build_model(inputs=x, targets=y, n_layers=2, n_neurons=64, activation="alsing")
        loss = build_loss("msre", sigma_level=3, chi2_dof=4, max_loglkl=float(y.max()), targets=y)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss)
        model.fit(x, y, epochs=60, batch_size=64, verbose=0)
        corner = np.clip(1.0 - 0.02 * np.abs(np.random.default_rng(seed).normal(size=(40, 4))), 0, 1).astype(np.float32)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        model.fit(np.concatenate([x, corner]), np.concatenate([y, np.full(40, 200.0, np.float32)]),
                  epochs=40, batch_size=64, verbose=0)
        return model, x, y

    def test_refutes_a_false_optimum_and_keeps_the_points(self):
        model, x, y = self.trained()
        before = find_maxima(model, LOWER, UPPER, x, y, starts=64, steps=300, seed=1)[0]
        self.assertGreater(before["excess"], 50.0)
        calls = []

        def evaluate(points):
            calls.append(len(points))
            return points, truth(points), np.zeros((len(points), 2))

        config = AuditConfig(tau=10.0, k=4, rounds=8, epochs=40, learning_rate=3e-4, starts=64, steps=300, fit_window=50.0)
        log = []
        nx, ny, nd, report = audit_optima(
            model, x, y, 0.1, LOWER, UPPER, evaluate,
            lambda t: build_loss("msre", sigma_level=3, chi2_dof=4, max_loglkl=float(np.max(t)), targets=t),
            config, batch_size=64, n_append=100, seed=5, log=log.append)
        self.assertEqual(report["true_calls"], sum(calls))
        self.assertEqual(len(ny), report["added"])
        self.assertEqual(nd.shape, (len(ny), 2))
        np.testing.assert_allclose(ny, truth(nx))
        self.assertTrue(np.all(ny < y.max()))  # every flagged maximum was false here
        after = find_maxima(model, LOWER, UPPER, np.concatenate([x, nx]), np.concatenate([y, ny]), starts=64, steps=300, seed=9)[0]
        self.assertLess(after["excess"], before["excess"] / 5)
        if report["clear"]:
            self.assertLessEqual(report["highest_excess"], config.tau)
        # the network's value at the audited points is no longer above the best true value
        self.assertTrue(np.all(model(nx.astype(np.float32), training=False).numpy().ravel() < y.max() + config.tau))

    def test_failed_evaluations_are_not_added_and_end_the_loop(self):
        model, x, y = self.trained()
        calls = []

        def evaluate(points):
            calls.append(len(points))
            return points, np.full(len(points), -np.inf), None

        config = AuditConfig(tau=10.0, k=4, rounds=6, epochs=5, starts=64, steps=300)
        nx, ny, nd, report = audit_optima(
            model, x, y, 0.1, LOWER, UPPER, evaluate,
            lambda t: build_loss("msre", sigma_level=3, chi2_dof=4, max_loglkl=float(np.max(t)), targets=t),
            config, batch_size=64, n_append=100, seed=5, log=lambda *_: None)
        self.assertEqual(len(ny), 0)
        self.assertEqual(report["failed"], sum(calls))
        self.assertIsNone(nd)
        # points set aside are not evaluated twice
        self.assertLess(len(calls), config.rounds + 1)

    def test_nothing_flagged_costs_no_true_call(self):
        tf.keras.utils.set_random_seed(0)
        x, y = training_set()

        def evaluate(points):
            raise AssertionError("no evaluation expected")

        config = AuditConfig(starts=64, steps=200)
        nx, ny, nd, report = audit_optima(
            BumpModel(-1.0e4), x, y, 0.1, LOWER, UPPER, evaluate, lambda t: "mse", config,
            batch_size=64, n_append=100, seed=1, log=lambda *_: None)
        self.assertEqual((len(ny), report["true_calls"], report["clear"]), (0, 0, True))


class Needle(tf.keras.Model):
    """The truth, a false optimum at the far vertex of height `bump` (which every climb finds),
    and a needle at SPOT of height `needle` above the peak: so narrow that no climb from a
    start of the audit's reaches it. A chain does (SINU, 13 October 2026: 67 above the best
    true value, the audit clear, a third of two boosts spent on it)."""
    SPOT = np.full(4, 0.7)

    def __init__(self, needle=30.0, bump=-1.0e4):
        super().__init__()
        constant = tf.keras.initializers.Constant
        self.needle = self.add_weight(shape=(), initializer=constant(needle), trainable=False)
        self.bump = self.add_weight(shape=(), initializer=constant(bump), trainable=False)

    def call(self, x, training=False):
        peak = -0.5 * tf.reduce_sum(((x - 0.4) / 0.05) ** 2, axis=1, keepdims=True)
        bump = self.bump * tf.exp(-0.5 * tf.reduce_sum(((x - 1.0) / 0.3) ** 2, axis=1, keepdims=True))
        needle = (500.0 + self.needle) * tf.exp(-0.5 * tf.reduce_sum(((x - 0.7) / 0.02) ** 2, axis=1, keepdims=True)) - 500.0
        return tf.maximum(tf.maximum(peak, bump - 1.0e-3 * tf.reduce_sum((x - 1.0) ** 2, axis=1, keepdims=True)), needle)


class ChainSearch(unittest.TestCase):
    """acquisition.audit.chain: a round whose climbs flag nothing looks where a tempered chain's
    cooled walkers climbed to (audit_optima's `peaks`)."""

    def chain(self, calls):
        """What the loop's hook returns: 20 cooled walkers at the needle and 30 at the peak, with
        the summits they climbed to."""
        rng = np.random.default_rng(0)

        def peaks(model):
            calls.append(1)
            summits = np.concatenate([np.tile(Needle.SPOT, (20, 1)), np.tile(CENTRE, (30, 1))])
            return summits, summits + 0.002 * rng.normal(size=summits.shape)

        return peaks

    def audit(self, model, config, tunes=(), evaluate=None, **options):
        from unittest import mock

        x, y = training_set()
        model(x[:2])
        script = iter(tunes)
        evaluated, log = [], []

        def true_values(points):
            evaluated.append(np.array(points))
            return points, truth(points), None

        with mock.patch("training.audit.fine_tune", lambda network, *arguments: next(script)(network)):
            nx, ny, nd, report = audit_optima(
                model, x, y, 0.1, LOWER, UPPER, evaluate or true_values, lambda t: "mse", config,
                batch_size=64, n_append=100, seed=1, log=log.append, **options)
        return report, evaluated, log

    def test_the_summits_are_taken_where_they_are(self):
        x, y = training_set()
        model = Needle()
        model(x[:2])
        counts = {}
        summits = np.concatenate([np.tile(Needle.SPOT, (20, 1)), np.tile(CENTRE, (30, 1))])
        maxima = summit_maxima(model, summits, LOWER, UPPER, x, y, counts=counts)
        self.assertEqual([(round(m["excess"]), m["hits"], m["starts"]) for m in maxima], [(30, 20, 50), (0, 30, 50)])
        np.testing.assert_allclose(maxima[0]["x"], Needle.SPOT, atol=1e-6)
        self.assertEqual((counts["starts"], counts["hits"]), (50, [20, 30]))
        # the audit's own search, handed nothing, does not find the needle
        self.assertLess(find_maxima(model, LOWER, UPPER, x, y, starts=64, steps=300, seed=1)[0]["excess"], 1.0)

    def test_without_it_the_needle_stands_and_with_it_it_is_evaluated(self):
        config = AuditConfig(tau=10.0, k=6, rounds=3, epochs=1, starts=64, steps=300)
        report, evaluated, _ = self.audit(Needle(), config)
        self.assertEqual((report["true_calls"], report["clear"], [r["search"] for r in report["rounds"]]), (0, True, ["full"]))

        calls, model = [], Needle()
        report, evaluated, log = self.audit(model, config, tunes=[lambda network: network.needle.assign(-100.0)],
                                            peaks=self.chain(calls), cloud=True)
        self.assertEqual([r["search"] for r in report["rounds"]], ["chain", "chain"])
        self.assertEqual([round(r["highest_excess"]) for r in report["rounds"]], [30, 0])
        self.assertEqual((len(calls), report["true_calls"], report["clear"], report["kept_round"]), (2, 6, True, 1))
        # the summit itself, and cooled walkers that climbed to it, up to k
        np.testing.assert_allclose(evaluated[0][0], Needle.SPOT, atol=1e-6)
        self.assertEqual((len(evaluated[0]), report["rounds"][0]["cloud"]), (6, 5))
        self.assertTrue(np.all(np.abs(evaluated[0] - Needle.SPOT) < 0.02))
        self.assertTrue(any("5 cooled walkers at them" in line for line in log))
        self.assertTrue(any("(chain search)" in line for line in log))
        # the summit alone
        calls = []
        report, evaluated, _ = self.audit(Needle(), config, tunes=[lambda network: network.needle.assign(-100.0)],
                                          peaks=self.chain(calls))
        self.assertEqual((report["true_calls"], len(evaluated[0]), report["clear"]), (1, 1, True))

    def test_the_chain_is_not_run_while_the_climbs_still_find_something(self):
        calls = []
        config = AuditConfig(tau=10.0, k=1, rounds=3, epochs=1, starts=64, steps=300)
        tunes = [lambda network: network.bump.assign(-1.0e4), lambda network: network.needle.assign(-100.0)]
        report, evaluated, _ = self.audit(Needle(bump=50.0), config, tunes=tunes, peaks=self.chain(calls))
        self.assertEqual([r["search"] for r in report["rounds"]], ["full", "chain", "chain"])
        self.assertEqual((len(calls), report["true_calls"], report["clear"]), (2, 2, True))

    def test_its_rounds_have_an_allowance_of_their_own(self):
        # One round for the climbs, and they use it: the chain search first looks in what was the
        # last round allowed, found the maximum, and could not act on it (a 16-parameter run, 13
        # October 2026: the boost then met it with all its weight).
        calls = []
        config = AuditConfig(tau=10.0, k=1, rounds=1, epochs=1, starts=64, steps=300)
        tunes = [lambda network: network.bump.assign(-1.0e4), lambda network: network.needle.assign(-100.0)]
        report, evaluated, _ = self.audit(Needle(bump=50.0), config, tunes=tunes, peaks=self.chain(calls))
        self.assertEqual([r["search"] for r in report["rounds"]], ["full", "chain", "chain"])
        self.assertEqual((report["true_calls"], report["call_budget"], report["clear"]), (2, 2, True))
        # and it is an allowance: a needle that every fine-tune leaves standing ends the audit
        calls = []
        report, evaluated, _ = self.audit(Needle(), config, tunes=[lambda network: None], peaks=self.chain(calls))
        self.assertEqual(([r["search"] for r in report["rounds"]], report["true_calls"], report["clear"]),
                         (["chain", "chain"], 1, False))
        # without the chain search the number of rounds is what it was
        report, _, _ = self.audit(Needle(bump=50.0), config, tunes=[lambda network: network.bump.assign(40.0)])
        self.assertEqual(([r["search"] for r in report["rounds"]], report["call_budget"]), (["full", "full"], 1))

    def test_a_round_the_chain_looked_at_is_compared_with_such_rounds_only(self):
        # The climbs' rounds report what the climbs found. Against them a round with the deeper
        # look lost, and the network of the first round was kept: the fine-tunes were undone.
        calls = []
        config = AuditConfig(tau=10.0, k=1, rounds=2, epochs=1, starts=64, steps=300)
        tunes = [lambda network: network.bump.assign(-1.0e4)]

        def evaluate(points):                         # the needle's summit cannot be evaluated: it stays
            points = np.asarray(points, float)
            values = np.where(np.all(np.abs(points - Needle.SPOT) < 0.05, axis=1), np.nan, truth(points))
            return points, values, None

        model = Needle(needle=80.0, bump=50.0)
        report, _, log = self.audit(model, config, tunes=tunes, evaluate=evaluate, peaks=self.chain(calls))
        self.assertEqual([r["search"] for r in report["rounds"]], ["full", "chain", "chain"])
        self.assertEqual([round(r["highest_excess"]) for r in report["rounds"]], [50, 80, 80])
        self.assertIn(report["kept_round"], (1, 2))                        # a round of the chain's, not round 0
        self.assertAlmostEqual(float(model.bump.numpy()), -1.0e4)          # the fine-tune stands
        self.assertFalse(any("the network of round 0 is kept" in line for line in log))

    def test_a_fingerprint_of_the_weights(self):
        model = Needle()
        model(np.zeros((1, 4), np.float32))
        before = weights_digest(model)
        self.assertEqual(before, weights_digest(model))
        model.needle.assign(29.0)
        self.assertNotEqual(before, weights_digest(model))


class Plateau(tf.keras.Model):
    """The truth, raised by `lift` over a broad patch around PATCH: a network too high where its
    posterior is, with no maximum above the best true value (the axion run's networks 10, 17
    October 2026: some 10 too high over a third of a boost)."""
    PATCH = np.full(4, 0.45)

    def __init__(self, lift=10.0):
        super().__init__()
        self.lift = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(lift), trainable=False)

    def call(self, x, training=False):
        peak = -0.5 * tf.reduce_sum(((x - 0.4) / 0.05) ** 2, axis=1, keepdims=True)
        return peak + self.lift * tf.exp(-0.5 * tf.reduce_sum(((x - 0.45) / 0.05) ** 2, axis=1, keepdims=True))


class PosteriorPilot(unittest.TestCase):
    """acquisition.audit.posterior: cooled T = 1 walkers of the network evaluated after the
    search for maxima, and a fine-tune when the network is too high there on average."""

    def walkers(self, calls):
        rng = np.random.default_rng(0)

        def peaks(model):
            calls.append(weights_digest(model))
            walkers = np.clip(Plateau.PATCH + 0.03 * rng.normal(size=(60, 4)), 0, 1)
            return np.tile(CENTRE, (60, 1)), walkers

        return peaks

    def audit(self, model, config, tunes=(), values_fn=None, **options):
        from unittest import mock

        x, y = training_set()
        model(x[:2])
        script = iter(tunes)
        evaluated, log = [], []

        def true_values(points):
            evaluated.append(np.array(points))
            return points, (truth if values_fn is None else values_fn)(points), None

        with mock.patch("training.audit.fine_tune", lambda network, *arguments: next(script)(network)):
            nx, ny, nd, report = audit_optima(
                model, x, y, 0.1, LOWER, UPPER, true_values, lambda t: "mse", config,
                batch_size=64, n_append=100, seed=1, log=log.append, **options)
        return nx, ny, report, evaluated, log

    def test_a_network_too_high_where_its_posterior_is_is_fine_tuned(self):
        calls, model = [], Plateau(lift=10.0)
        config = AuditConfig(tau=10.0, k=4, rounds=2, epochs=1, starts=64, steps=300, chain=True,
                             posterior=20, posterior_tolerance=2.0, posterior_rounds=2)
        nx, ny, report, evaluated, log = self.audit(model, config, tunes=[lambda network: network.lift.assign(0.0)],
                                                    peaks=self.walkers(calls))
        self.assertEqual(report["clear"], True)                         # no maximum above the best: the search saw nothing
        pilot = report["posterior"]
        self.assertEqual([r["fine_tuned"] for r in pilot["rounds"]], [True, False])
        self.assertGreater(pilot["rounds"][0]["mean_excess"], 2.0)
        self.assertAlmostEqual(pilot["rounds"][1]["mean_excess"], 0.0, places=4)
        self.assertTrue(pilot["tuned"])
        self.assertEqual(report["true_calls"], 40)
        self.assertEqual(len(ny), 40)                                   # the pilot's points are trained on
        self.assertEqual([len(e) for e in evaluated], [20, 20])
        self.assertTrue(np.all(np.abs(evaluated[0] - Plateau.PATCH) < 0.2))
        self.assertEqual(len(set(calls)), len(calls))                   # every chain looked at another network
        self.assertTrue(any("posterior pilot 0: 20 cooled T = 1 walkers evaluated" in line and "fine-tuned" in line
                            for line in log), log)

    def test_a_last_fine_tune_is_followed_by_a_look_at_the_new_network(self):
        calls, model = [], Plateau(lift=10.0)
        config = AuditConfig(tau=10.0, k=4, rounds=2, epochs=1, starts=64, steps=300, chain=True,
                             posterior=10, posterior_rounds=1)
        _, ny, report, evaluated, log = self.audit(model, config, tunes=[lambda network: network.lift.assign(0.0)],
                                                   peaks=self.walkers(calls))
        pilot = report["posterior"]
        self.assertEqual(([r["fine_tuned"] for r in pilot["rounds"]], report["true_calls"]), ([True], 10))
        self.assertAlmostEqual(pilot["highest_excess_after"], 0.0, delta=0.5)
        # the last chain is of the network the audit hands on, which the sampling step then reuses
        self.assertEqual(calls[-1], weights_digest(model))
        self.assertTrue(any("after the last fine-tune" in line for line in log))

    def test_a_network_right_where_its_posterior_is_is_left_alone(self):
        calls, model = [], Plateau(lift=0.0)
        config = AuditConfig(tau=10.0, k=4, rounds=2, epochs=1, starts=64, steps=300, chain=True, posterior=20)
        _, ny, report, evaluated, _ = self.audit(model, config, tunes=[], peaks=self.walkers(calls))
        self.assertEqual([r["fine_tuned"] for r in report["posterior"]["rounds"]], [False])
        self.assertFalse(report["posterior"]["tuned"])
        self.assertEqual((report["true_calls"], len(ny), len(calls)), (20, 20, 1))   # the chain search's look, reused

    def test_without_it_the_audit_is_as_it_was(self):
        calls = []
        config = AuditConfig(tau=10.0, k=4, rounds=2, epochs=1, starts=64, steps=300, chain=True)
        _, ny, report, evaluated, _ = self.audit(Plateau(lift=10.0), config, tunes=[], peaks=self.walkers(calls))
        self.assertNotIn("posterior", report)
        self.assertEqual((report["true_calls"], len(ny), report["call_budget"]), (0, 0, 16))

    def test_nonfinite_pilot_returns_are_counted_without_training_on_them(self):
        config = AuditConfig(tau=10., k=4, rounds=2, epochs=1, starts=64, steps=50,
                             chain=True, posterior=10)
        for finite in (0, 3):
            with self.subTest(finite=finite):
                def values(points):
                    y = truth(points)
                    y[finite:] = -np.inf
                    return y
                _, ny, report, _, log = self.audit(
                    Plateau(lift=0.), config, values_fn=values, peaks=self.walkers([]))
                pilot = report["posterior"]
                self.assertEqual((report["true_calls"], report["failed"], len(ny)), (10, 10-finite, finite))
                self.assertFalse(pilot["tuned"])
                self.assertEqual(pilot["rounds"][0]["finite"], finite)
                if not finite:
                    self.assertEqual(pilot["rounds"][0]["status"], "no_finite_returns")
                    self.assertIsNone(pilot["rounds"][0]["mean_excess"])
                    self.assertTrue(any("accuracy is unassessed" in line for line in log))

    def test_clearance_is_recomputed_after_a_pilot_makes_the_model_worse(self):
        config = AuditConfig(tau=10., k=4, rounds=2, epochs=1, starts=64, steps=300,
                             chain=True, posterior=10, posterior_rounds=1)
        calls, model = [], Plateau(lift=10.)
        _, _, report, _, log = self.audit(
            model, config, tunes=[lambda m: m.lift.assign(100.)], peaks=self.walkers(calls))
        self.assertEqual(report["posterior"]["pre_pilot_status"]["distinct_above_tau"], 0)
        self.assertFalse(report["clear"])
        self.assertGreater(report["highest_excess"], config.tau)
        self.assertEqual(report["final_search"]["stage"], "post_pilot")
        self.assertEqual(report["final_search"]["weights_digest"], weights_digest(model))
        self.assertEqual(calls[-1], weights_digest(model))
        self.assertEqual(report["true_calls"], 10)
        self.assertNotIn("nothing above", log[-1])


class Configuration(unittest.TestCase):
    BASE = dict(n_append=10, n_neighbors=5, target_temperature=7.0, pool_factor=20)

    def test_the_posterior_pilot_is_off_unless_asked_and_needs_the_chain_search(self):
        self.assertEqual(AuditConfig().posterior, 0)
        audit = AcquisitionConfig.from_dict(dict(self.BASE, audit=dict(chain=True, posterior=150))).audit
        self.assertEqual((audit.posterior, audit.posterior_tolerance, audit.posterior_rounds), (150, 2.0, 2))
        with self.assertRaisesRegex(ValueError, "posterior needs acquisition.audit.chain"):
            AcquisitionConfig.from_dict(dict(self.BASE, audit=dict(posterior=150)))
        for wrong in (dict(posterior=-1), dict(posterior_tolerance=0.0), dict(posterior_rounds=0)):
            with self.assertRaisesRegex(ValueError, "posterior"):
                AcquisitionConfig.from_dict(dict(self.BASE, audit=dict(chain=True, **wrong)))

    def test_the_chain_search_is_off_unless_asked(self):
        self.assertFalse(AuditConfig().chain)
        self.assertFalse(AcquisitionConfig.from_dict(dict(self.BASE, audit={})).audit.chain)
        self.assertTrue(AcquisitionConfig.from_dict(dict(self.BASE, audit=dict(chain=True))).audit.chain)
        with self.assertRaisesRegex(ValueError, "chain must be a YAML boolean"):
            AcquisitionConfig.from_dict(dict(self.BASE, audit=dict(chain="yes")))

    def test_absent_means_off(self):
        self.assertIsNone(AcquisitionConfig.from_dict(self.BASE).audit)

    def test_empty_mapping_takes_the_defaults(self):
        self.assertEqual(AcquisitionConfig.from_dict(dict(self.BASE, audit=None)).audit, AuditConfig())
        self.assertEqual(AcquisitionConfig.from_dict(dict(self.BASE, audit={})).audit, AuditConfig())

    def test_values_and_unknown_keys(self):
        audit = AcquisitionConfig.from_dict(dict(self.BASE, audit=dict(tau=5, k=12, start_iteration=2))).audit
        self.assertEqual((audit.tau, audit.k, audit.start_iteration), (5.0, 12, 2))
        with self.assertRaises(ValueError):
            AcquisitionConfig.from_dict(dict(self.BASE, audit=dict(taus=5)))
        with self.assertRaises(ValueError):
            AcquisitionConfig.from_dict(dict(self.BASE, audit=dict(k=-1)))
        with self.assertRaises(ValueError):
            AcquisitionConfig.from_dict(dict(self.BASE, audit=dict(fraction=0)))

    def test_fine_tunes_are_not_compiled_with_xla(self):
        # Every XLA compilation leaves memory mappings for the life of the process and the
        # audit compiles on every round: runs of 21 iterations ended at Linux's limit.
        x, y = training_set(300)
        model = build_model(x, y, 2, 16, "tanh")
        fine_tune(model, x, y, "mse", (x[:30], y[:30]), 1, 1e-5, 64)
        self.assertIs(model.jit_compile, False)
        with self.assertRaises(ValueError):
            AcquisitionConfig.from_dict(dict(self.BASE, audit=dict(jit_compile=True)))

    def test_budget_is_a_fraction_of_the_batch(self):
        # the default: a tenth of the batch over five rounds
        self.assertEqual((AuditConfig().fraction, AuditConfig().rounds), (0.1, 5))
        self.assertEqual(AuditConfig().per_round(2000), 40)
        self.assertEqual(AuditConfig().per_round(32100), 642)
        self.assertEqual(AuditConfig().per_round(10), 1)  # never none
        self.assertEqual(AuditConfig(fraction=0.05, rounds=10).per_round(2000), 10)
        self.assertEqual(AuditConfig(k=6).per_round(2000), 6)  # an explicit k wins


class Files(unittest.TestCase):
    def test_derived_values_survive_restart_and_old_audits_remain_readable(self):
        import pandas as pd
        from training.audit import load_audit
        from dataset.dataset import TrainingDataset
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmp:
            names = ["x", "y"]
            x = np.arange(12, dtype=float).reshape(6, 2) / 10
            values = np.column_stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]])
            values[1, 1] = np.nan
            save_audit(tmp, 3, names, x, -np.arange(6), {},
                       derived_names=["d", "e"], derived=values)
            rx, ry, rd = load_audit(tmp, 3, names, ["e", "d", "new"])
            np.testing.assert_allclose(rd[:, :2], values[:, ::-1], equal_nan=True)
            self.assertTrue(np.isnan(rd[:, 2]).all())
            # Exercise the same dataset append used by client.py on continuation.
            backend = SimpleNamespace(param_names=names, derived_names=["e", "d", "new"])
            dataset = TrainingDataset(x, -np.arange(6), backend, n_neighbors=2, target_temperature=7)
            dataset.add_data(rx, ry, derived=rd)
            np.testing.assert_allclose(dataset.derived[-6:], rd, equal_nan=True)
            sidecar = Path(tmp) / "audit_derived_it_3.csv"
            frame = pd.read_csv(sidecar)
            frame.iloc[::-1].to_csv(sidecar, index=False)
            with self.assertRaisesRegex(ValueError, "do not match"):
                load_audit(tmp, 3, names, ["d"])
            sidecar.unlink()  # pre-sidecar audit
            self.assertTrue(np.isnan(load_audit(tmp, 3, names, ["d"])[2]).all())
            save_audit(tmp, 4, names, np.empty((0, 2)), [], {}, derived_names=["d"])
            self.assertEqual(load_audit(tmp, 4, names, ["d"])[2].shape, (0, 1))

    def test_csv_is_a_training_data_file(self):
        from dataset.augment import read_augmentation

        with tempfile.TemporaryDirectory() as tmp:
            names = ["a", "b", "c", "d"]
            x = np.random.default_rng(0).uniform(size=(3, 4))
            save_audit(tmp, 7, names, x, [-1.0, -2.0, -3.0], dict(true_calls=3))
            rx, ry = read_augmentation(Path(tmp) / "audit_it_7.csv", names)
            np.testing.assert_allclose(rx, x.astype(np.float32))
            np.testing.assert_allclose(ry, [-1.0, -2.0, -3.0])
            save_audit(tmp, 8, names, np.empty((0, 4)), np.empty(0), dict(true_calls=0))
            self.assertEqual(len(read_augmentation(Path(tmp) / "audit_it_8.csv", names)[1]), 0)


if __name__ == "__main__":
    unittest.main()
