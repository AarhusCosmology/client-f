"""client.py evidence: prior conventions per backend, user ranges, box handling, and (with
BEST on $BEST_PATH or installed) nested sampling of an analytic target and the command."""

import contextlib
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence import priors
from evidence.sampler import edge_mass


def best_available():
    from evidence.sampler import import_best

    try:
        import_best()
        return True
    except ImportError:
        return False


def gaussian_3d():
    """A normalised 3D Gaussian as BEST takes it, and its mean."""
    import tensorflow as tf

    mu = tf.constant([0.3, -0.2, 0.1], tf.float32)
    sd = tf.constant([0.10, 0.05, 0.20], tf.float32)

    def log_prob(x):
        z = (tf.cast(x, tf.float32) - mu) / sd
        return -0.5 * tf.reduce_sum(z * z, axis=1) - tf.reduce_sum(tf.math.log(sd)) - 1.5 * math.log(2 * math.pi)

    return log_prob, mu.numpy()


def quadratic_model(mu, sd, top, rise=None):
    """A network that is top - sum(((x - mu) / sd)^2) / 2 at single precision, from stock Keras
    layers. `rise` = (direction, start, strength) adds strength * relu(direction . x - start)^2."""
    import tensorflow as tf

    mu, sd = np.asarray(mu, np.float32), np.asarray(sd, np.float32)
    inputs = tf.keras.Input((len(mu),))
    z = tf.keras.layers.Dense(len(mu), name="whiten")(inputs)
    output = tf.keras.layers.Dense(1, name="value")(tf.keras.layers.Dot(axes=1)([z, z]))
    if rise is not None:
        ramp = tf.keras.layers.Dense(1, activation="relu", name="ramp")(inputs)
        square = tf.keras.layers.Multiply()([ramp, ramp])
        output = tf.keras.layers.Add()([output, tf.keras.layers.Dense(1, use_bias=False, name="rise")(square)])
    model = tf.keras.Model(inputs, output)
    model.get_layer("whiten").set_weights([np.diag(1 / sd), -mu / sd])
    model.get_layer("value").set_weights([np.array([[-0.5]], np.float32), np.array([top], np.float32)])
    if rise is not None:
        direction, start, strength = rise
        model.get_layer("ramp").set_weights([np.array(direction, np.float32).reshape(-1, 1),
                                             np.array([-start], np.float32)])
        model.get_layer("rise").set_weights([np.array([[strength]], np.float32)])
    return model


def cobaya_run(root, model, lower, upper, iteration=2, training=None):
    """A run directory of the cobaya wrapper around `model`, with uniform priors on the box
    [lower, upper]; `training` = (points, values) becomes the network's training set."""
    names = [f"x{i}" for i in range(len(lower))]
    run = Path(root) / "run"
    (run / "likelihood_input").mkdir(parents=True)
    (run / "trained_models").mkdir()
    (run / "likelihood_input" / "toy.yaml").write_text(yaml.safe_dump(dict(params={
        n: dict(prior=dict(min=float(a), max=float(b))) for n, a, b in zip(names, lower, upper)})))
    (run / "run.yaml").write_text(yaml.safe_dump(dict(likelihood=dict(wrapper="cobaya", input="elsewhere/toy.yaml"))))
    (run / "metadata.json").write_text(json.dumps(dict(parameters=[
        dict(name=n, label=n, scale=1.0, lower=float(a), upper=float(b), center=0.5 * (a + b), sigma=0.1 * (b - a))
        for n, a, b in zip(names, lower, upper)])))
    model.save(run / "trained_models" / f"model_it_{iteration}.keras")
    if training is not None:
        (run / "training_data").mkdir()
        np.savetxt(run / "training_data" / f"data_it_{iteration}.csv", np.column_stack(training),
                   delimiter=",", header=",".join(names + ["loglkl"]), comments="")
    return run


def command(*words):
    """The evidence command with these words; what it printed."""
    from evidence.__main__ import main

    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        main([str(word) for word in words])
    return printed.getvalue()


PARAM = """# MontePython input
data.experiments=['toy']
data.parameters['omega_b']     = [ 2.2377, None, None, 0.015, 0.01, 'cosmo']
data.parameters['100*theta_s'] = [ 1.0411,  1.0,  1.1, 0.0003,   1, 'cosmo']
data.parameters['tau_reio']    = [ 0.0543, 0.004, None, 0.008,   1, 'cosmo']  # one-sided
data.parameters['N_ur']        = [ 3.046,  None, None,   0,     1, 'cosmo']  # fixed
#data.parameters['A_s']        = [ 2.1,    None, None,  0.03, 1e-9, 'cosmo']
"""

COBAYA = dict(params=dict(
    a=dict(prior=dict(min=0.0, max=2.0)),
    b=dict(prior=dict(dist="uniform", loc=-1.0, scale=4.0)),
    c=dict(prior=[0.5, 1.5]),
    d=dict(prior=dict(dist="norm", loc=0.0, scale=1.0)),
    e=dict(value=1.0),
))


class PriorTests(unittest.TestCase):
    def test_montepython_ranges_are_physical(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "toy.param"
            path.write_text(PARAM)
            ranges = priors.montepython_ranges(path, {"omega_b": 0.01, "100*theta_s": 1.0, "tau_reio": 1.0})
            self.assertEqual(ranges, {"omega_b": (None, None), "100*theta_s": (1.0, 1.1), "tau_reio": (0.004, None)})
            with self.assertRaisesRegex(ValueError, "A_s"):  # commented out
                priors.montepython_ranges(path, {"A_s": 1e-9})
            path.write_text(PARAM.replace("0.004, None", "0.4, 0.8").replace("1.0,  1.1", "100, 110"))
            ranges = priors.montepython_ranges(path, {"100*theta_s": 0.01, "tau_reio": 1.0})
            self.assertEqual(ranges["100*theta_s"], (1.0, 1.1))  # MontePython units times scale

    def test_cobaya_ranges(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "toy.yaml"
            path.write_text(yaml.safe_dump(COBAYA))
            self.assertEqual(priors.cobaya_ranges(path, ["a", "b", "c", "d"]),
                             {"a": (0.0, 2.0), "b": (-1.0, 3.0), "c": (0.5, 1.5), "d": None})

    def test_user_ranges(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ranges.yaml"
            path.write_text("omega_b: [0.019, 0.025]\n'100*theta_s': [1.03, 1.05]\n")
            ranges = priors.parse_ranges("omega_b=0.02:0.024, tau_reio=0.01:0.1", path)
        self.assertEqual(ranges, {"omega_b": (0.02, 0.024), "100*theta_s": (1.03, 1.05), "tau_reio": (0.01, 0.1)})
        self.assertEqual(priors.parse_ranges("100*theta_s=1.0:1.1"), {"100*theta_s": (1.0, 1.1)})
        for bad in ("omega_b=0.02", "omega_b:0.02:0.03", "omega_b=0.03:0.02", "=1:2"):
            with self.assertRaises(ValueError):
                priors.parse_ranges(bad)

    def test_montepython_prior_uses_the_domain_where_unbounded(self):
        names = ["omega_b", "100*theta_s", "tau_reio"]
        domain = ([0.020, 1.03, 0.01], [0.025, 1.05, 0.10])
        param = {"omega_b": (None, None), "100*theta_s": (1.0, 1.1), "tau_reio": (0.004, None)}
        box, table, correction = priors.resolve("montepython", names, domain, domain, param)
        self.assertEqual(box, ([0.020, 1.03, 0.01], [0.025, 1.05, 0.10]))
        self.assertEqual([r["source"] for r in table], ["run domain", ".param", ".param (upper: run domain)"])
        self.assertEqual([(r["lower"], r["upper"]) for r in table], [(0.020, 0.025), (1.0, 1.1), (0.004, 0.10)])
        self.assertAlmostEqual(correction, -math.log(0.005) - math.log(0.1) - math.log(0.096))
        # A user range replaces the prior range, and shrinks the box where narrower.
        box, table, correction = priors.resolve("montepython", names, domain, domain, param,
                                                {"omega_b": (0.021, 0.030)})
        self.assertEqual(box[0][0], 0.021)
        self.assertEqual(box[1][0], 0.025)
        self.assertEqual((table[0]["lower"], table[0]["upper"], table[0]["source"]), (0.021, 0.030, "user"))
        self.assertAlmostEqual(correction, -math.log(0.009) - math.log(0.1) - math.log(0.096))
        with self.assertRaisesRegex(ValueError, "unknown"):
            priors.resolve("montepython", names, domain, domain, param, {"H0": (60, 70)})
        with self.assertRaisesRegex(ValueError, "overlap"):
            priors.resolve("montepython", names, domain, domain, param, {"omega_b": (0.03, 0.04)})

    def test_cobaya_prior_is_already_in_the_target(self):
        names = ["a", "d"]
        domain = ([0.0, -3.0], [2.0, 3.0])
        ranges = {"a": (0.0, 2.0), "d": None}
        box, table, correction = priors.resolve("cobaya", names, domain, domain, ranges)
        self.assertEqual(correction, 0.0)
        self.assertEqual([r["source"] for r in table], ["Cobaya prior", "Cobaya prior (not uniform), normalised by Cobaya"])
        # A user range renormalises Cobaya's uniform density: range 2 -> 1.
        box, table, correction = priors.resolve("cobaya", names, domain, domain, ranges, {"a": (0.5, 1.5)})
        self.assertAlmostEqual(correction, math.log(2.0))
        self.assertEqual(box, ([0.5, -3.0], [1.5, 3.0]))
        with self.assertRaisesRegex(ValueError, "not uniform"):
            priors.resolve("cobaya", names, domain, domain, ranges, {"d": (-1.0, 1.0)})

    def test_changed_ranges_leave_out_a_prior_without_a_range(self):
        # The summary lists the ranges that are not the backend's own. A Gaussian Cobaya prior
        # has none (lower = upper = None) and is the backend's own: not listed, not formatted.
        names = ["a", "d"]
        domain = ([0.0, -3.0], [2.0, 3.0])
        ranges = {"a": (0.0, 2.0), "d": None}
        _, table, _ = priors.resolve("cobaya", names, domain, domain, ranges)
        self.assertEqual((table[1]["lower"], table[1]["upper"]), (None, None))
        self.assertEqual(priors.changed_ranges(table), [])
        _, table, _ = priors.resolve("cobaya", names, domain, domain, ranges, {"a": (0.5, 1.5)})
        self.assertEqual(priors.changed_ranges(table), ["a [0.5, 1.5] (user (replaces Cobaya's uniform prior))"])
        param = {"x": (None, None), "y": (-1.0, 1.0), "z": (0.0, None)}
        domain = ([0.0, -0.5, 0.1], [1.0, 0.5, 0.9])
        _, table, _ = priors.resolve("montepython", ["x", "y", "z"], domain, domain, param)
        self.assertEqual(priors.changed_ranges(table),
                         ["x [0, 1] (run domain)", "z [0, 0.9] (.param (upper: run domain))"])

    def test_edge_mass(self):
        points = np.array([[0.005, 0.5], [0.5, 0.995], [0.5, 0.5]])
        weights = np.array([0.1, 0.2, 0.7])
        np.testing.assert_allclose(edge_mass(points, weights, [0, 0], [1, 1]), [[0.1, 0.0], [0.0, 0.2]])


class ChainFileTests(unittest.TestCase):
    def test_getdist_loads_the_chain_of_a_montepython_run(self):
        # MontePython calls CLASS's angular scale 100*theta_s, and GetDist refuses a name with
        # a space, * or ?: the chain written beside every such evidence could not be loaded.
        from getdist import loadMCSamples

        from evidence.__main__ import write_chain

        rng = np.random.default_rng(1)
        points = rng.normal([1.04, 0.022], [0.001, 0.0002], size=(500, 2))
        with tempfile.TemporaryDirectory() as d:
            root = str(Path(d) / "evidence_it_3_nlive500")
            write_chain(root, ["100*theta_s", "omega b?"], ["100\\theta_s", "\\omega_b"], points, np.full(500, 1 / 500),
                        -0.5 * rng.chisquare(2, 500), ([1.03, 0.021], [1.05, 0.023]))
            self.assertEqual([line.split("\t")[0] for line in Path(root + ".paramnames").read_text().splitlines()],
                             ["100theta_s", "omegab"])
            self.assertEqual([line.split("\t")[0] for line in Path(root + ".ranges").read_text().splitlines()],
                             ["100theta_s", "omegab"])
            samples = loadMCSamples(root, settings=dict(ignore_rows=0))
            self.assertEqual([p.name for p in samples.getParamNames().names], ["100theta_s", "omegab"])
            np.testing.assert_allclose(samples.mean("100theta_s"), points[:, 0].mean(), rtol=1e-6)
            self.assertEqual(samples.ranges.getLower("omegab"), 0.021)


@unittest.skipUnless(best_available(), "BEST is not installed or on $BEST_PATH")
class NestedSamplingTests(unittest.TestCase):
    def test_n_sigma_box_refuses_a_parameter_without_a_centre_or_sigma(self):
        from evidence.__main__ import sampling_box

        parameter = lambda name, center, sigma: SimpleNamespace(name=name, center=center, sigma=sigma)
        md = SimpleNamespace(param_names=["a", "b"], bounds=dict(a=(0.0, 10.0), b=(0.0, 10.0)),
                             parameters=[parameter("a", 5.0, 1.0), parameter("b", None, 1.0)])
        with self.assertRaisesRegex(ValueError, "no proposal centre or sigma for b; name them in --keep-bounds"):
            sampling_box(md, 2.0)
        (lo, hi), _ = sampling_box(md, 2.0, keep=("b",))
        np.testing.assert_array_equal(lo, [3.0, 0.0])
        np.testing.assert_array_equal(hi, [7.0, 10.0])

    def test_gaussian_evidence_on_two_boxes(self):
        # A normalised 3D Gaussian inside the prior: Cobaya's convention gives
        # ln Z = ln Z_box + ln V_box = ln P(box) = 0 on any box holding the mass.
        from evidence.sampler import run_nested

        log_prob, mu = gaussian_3d()
        values = []
        for half in (1.0, 2.0):
            lo, hi = np.full(3, -half), np.full(3, half)
            r = run_nested(log_prob, lo, hi, n_live=300, seed=3, verbose=False)
            values.append(r["lnZ_box"] + float(np.sum(np.log(hi - lo))))
            self.assertLess(abs(values[-1]), 4 * r["sigma"] + 0.02)
            self.assertFalse(r["iteration_cap_reached"])
            self.assertAlmostEqual(r["weights"].sum(), 1.0, places=5)
            np.testing.assert_allclose(r["weights"] @ r["points"], mu, atol=0.03)

    def test_tolerance_is_the_fraction_left_in_the_live_points(self):
        # BEST compares its own tolerance with ln(Z_live / Z_dead), so that every positive value
        # (1e-3 was the command's) stopped the run with half the evidence still in the live
        # points. As a fraction, 1e-3 goes on until that is what the live points hold.
        from evidence.sampler import run_nested

        log_prob, _ = gaussian_3d()
        lo, hi = np.full(3, -1.0), np.full(3, 1.0)
        early = run_nested(log_prob, lo, hi, n_live=300, tolerance=0.5, seed=3, verbose=False)
        late = run_nested(log_prob, lo, hi, n_live=300, seed=3, verbose=False)
        self.assertTrue(0.25 < early["live_fraction"] < 0.334, early["live_fraction"])
        self.assertTrue(5e-4 < late["live_fraction"] < 1e-3, late["live_fraction"])
        self.assertGreater(late["n_dead"] - early["n_dead"], 4 * 300)
        self.assertLess(abs(late["lnZ_box"] + float(np.sum(np.log(hi - lo)))), 4 * late["sigma"] + 0.02)
        for bad in (0.0, 1.0, 1.5, -6.9, float("nan")):
            with self.assertRaisesRegex(ValueError, "fraction"):
                run_nested(log_prob, lo, hi, n_live=300, tolerance=bad, verbose=False)

    def test_fewer_iterations_than_the_display_interval(self):
        # Below 50 iterations BEST had no chunk to run and stopped with an UnboundLocalError.
        from evidence.sampler import run_nested

        log_prob, _ = gaussian_3d()
        with contextlib.redirect_stdout(io.StringIO()):  # BEST says that it reached the cap
            r = run_nested(log_prob, np.full(3, -1.0), np.full(3, 1.0), n_live=100, n_max_iter=20, seed=3,
                           verbose=False)
        self.assertTrue(r["iteration_cap_reached"])
        self.assertEqual(r["n_dead"], 200)
        self.assertGreater(r["live_fraction"], 0.9)  # nearly everything is still to come
        with self.assertRaisesRegex(ValueError, "n_max_iter"):
            run_nested(log_prob, np.full(3, -1.0), np.full(3, 1.0), n_live=100, n_max_iter=0, verbose=False)


@unittest.skipUnless(best_available(), "BEST is not installed or on $BEST_PATH")
class CommandTests(unittest.TestCase):
    def test_montepython_run(self):
        # The command end to end on a tiny MontePython-style run with a random network:
        # outputs, the prior table and ln Z = ln Z_box + ln V_box - sum ln(range).
        import tensorflow as tf
        from evidence.__main__ import main

        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            (run / "likelihood_input").mkdir(parents=True)
            (run / "trained_models").mkdir()
            (run / "likelihood_input" / "toy.param").write_text(
                "data.parameters['x'] = [0.0, None, None, 0.1, 1, 'cosmo']\n"
                "data.parameters['y'] = [0.0, -1.0, 1.0, 0.1, 1, 'nuisance']\n")
            (run / "run.yaml").write_text(yaml.safe_dump(dict(likelihood=dict(wrapper="montepython",
                                                                              input="elsewhere/toy.param"))))
            (run / "metadata.json").write_text(json.dumps(dict(parameters=[
                dict(name="x", label="x", scale=1.0, lower=-0.5, upper=0.5, center=0.0, sigma=0.1),
                dict(name="y", label="y", scale=1.0, lower=-0.5, upper=0.5, center=0.0, sigma=0.1)])))
            tf.keras.utils.set_random_seed(1)
            inputs = tf.keras.Input(shape=(2,))
            hidden = tf.keras.layers.Dense(8, activation="tanh")(inputs)
            model = tf.keras.Model(inputs, tf.keras.layers.Dense(1)(hidden))
            model.save(run / "trained_models" / "model_it_2.keras")
            main([str(run), "--n-live", "100", "--quiet", "--prior", "x=-1:1"])
            report = json.loads((run / "evidence" / "evidence_it_2_nlive100.json").read_text())
            self.assertEqual(report["iteration"], 2)
            self.assertEqual([(r["name"], r["source"], r["lower"], r["upper"]) for r in report["priors"]],
                             [("x", "user", -1.0, 1.0), ("y", ".param", -1.0, 1.0)])
            self.assertAlmostEqual(report["lnV_box"], 0.0)  # the domain box, 1 x 1
            self.assertAlmostEqual(report["lnZ"], report["lnZ_box"] - 2 * math.log(2.0), places=6)
            # the box's faces lie inside both priors: an evidence check would flag real mass there
            self.assertEqual(len(report["priors"]), 2)
            for suffix in (".txt", ".paramnames", ".ranges"):
                self.assertTrue((run / "evidence" / f"evidence_it_2_nlive100{suffix}").exists())

    def test_cobaya_run_with_a_gaussian_prior(self):
        # A sampled parameter with a Gaussian Cobaya prior has no range in the prior table;
        # the command must get through its closing summary, and not list that parameter.
        import contextlib
        import io

        import tensorflow as tf
        from evidence.__main__ import main

        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            (run / "likelihood_input").mkdir(parents=True)
            (run / "trained_models").mkdir()
            (run / "likelihood_input" / "toy.yaml").write_text(yaml.safe_dump(dict(params=dict(
                x=dict(prior=dict(min=-1.0, max=1.0)),
                y=dict(prior=dict(dist="norm", loc=0.0, scale=0.1))))))
            (run / "run.yaml").write_text(yaml.safe_dump(dict(likelihood=dict(wrapper="cobaya",
                                                                              input="elsewhere/toy.yaml"))))
            (run / "metadata.json").write_text(json.dumps(dict(parameters=[
                dict(name="x", label="x", scale=1.0, lower=-0.5, upper=0.5, center=0.0, sigma=0.1),
                dict(name="y", label="y", scale=1.0, lower=-0.5, upper=0.5, center=0.0, sigma=0.1)])))
            tf.keras.utils.set_random_seed(1)
            inputs = tf.keras.Input(shape=(2,))
            hidden = tf.keras.layers.Dense(8, activation="tanh")(inputs)
            model = tf.keras.Model(inputs, tf.keras.layers.Dense(1)(hidden))
            model.save(run / "trained_models" / "model_it_2.keras")

            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                main([str(run), "--n-live", "100", "--quiet"])
            report = json.loads((run / "evidence" / "evidence_it_2_nlive100.json").read_text())
            self.assertEqual([(r["name"], r["source"], r["lower"], r["upper"]) for r in report["priors"]],
                             [("x", "Cobaya prior", -1.0, 1.0),
                              ("y", "Cobaya prior (not uniform), normalised by Cobaya", None, None)])
            self.assertAlmostEqual(report["lnZ"], report["lnZ_box"] + report["lnV_box"], places=6)
            self.assertIn("wrote ", printed.getvalue())
            self.assertNotIn("prior ranges not taken from the backend", printed.getvalue())
            # A run without boosts and without its training set: the whole box, ln L as it is.
            first = printed.getvalue().splitlines()[0]
            self.assertIn("no boost region for this network: the bare network", first)
            self.assertIn("BEST stops with 0.001 of the evidence left in the live points", first)
            self.assertIn("no training set in the run for a reference", first)
            self.assertIsNone(report["region"])
            self.assertEqual((report["lnL_reference"], report["lnL_reference_from"]), (0.0, None))
            self.assertEqual(report["tolerance"], 1e-3)
            # A sixteenth of the live points is replaced per iteration (128 of the default 2048)
            self.assertEqual(report["n_live_updates"], 100 // 16)
            self.assertIn("100 live points, 6 replaced per iteration", first)
            self.assertTrue(5e-4 < report["live_fraction"] < 1e-3, report["live_fraction"])
            self.assertEqual(report["model_sha256"], hashlib.sha256(
                (run / "trained_models" / "model_it_2.keras").read_bytes()).hexdigest())
            # Asking for the region of a run that has none stops, as does a tolerance that is
            # no fraction (BEST's own was a difference of ln Z, where -6.9 made sense).
            for words in (["--region"], ["--region", "run"], ["--region", "1"], ["--region", "0"], ["--region", "-1"],
                          ["--region", "nan"], ["--region", "everywhere"], ["--n-live-updates", "0"],
                          ["--n-live-updates", "51"], ["--tolerance", "0"], ["--tolerance", "-6.9"],
                          ["--tolerance", "1"], ["--max-iterations", "0"]):
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()) as refused:
                    main([str(run), "--n-live", "100", "--quiet", "--tag", "refused"] + words)
                self.assertIn(words[0], refused.getvalue())
            self.assertFalse(list((run / "evidence").glob("*refused*")))

            # With a user range the summary lists that parameter, and still not the Gaussian one.
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                main([str(run), "--n-live", "100", "--quiet", "--prior", "x=-0.25:0.25", "--tag", "user"])
            listed = printed.getvalue().split("prior ranges not taken from the backend as written:")[1]
            self.assertIn("x [-0.25, 0.25] (user (replaces Cobaya's uniform prior))", listed)
            self.assertNotIn("not uniform", listed)
            self.assertIn("wrote ", printed.getvalue())

    def test_region_of_the_boosts_is_the_default(self):
        # A unit Gaussian of evidence 1 in the box [-4, 4]^2 and, beyond x0 = 3.5, a rise that
        # the truth lacks (+50 at the face). The region of the boosts, of radius 3.3 to 3.6 and
        # ending at x0 = 3.43, holds none of the rise and all but 0.3% of the Gaussian (summed
        # on a grid: ln of the mass inside is -0.0027).
        from sampling.region import Region

        with tempfile.TemporaryDirectory() as d:
            model = quadratic_model([0.0, 0.0], [1.0, 1.0], -math.log(2 * math.pi), rise=([1.0, 0.0], 3.5, 200.0))
            run = cobaya_run(d, model, [-4.0, -4.0], [4.0, 4.0])
            cloud = np.random.default_rng(5).normal(size=(4000, 2))
            cloud = cloud[np.hypot(*cloud.T) <= 2.2][:600]
            angle = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            cloud = np.vstack([cloud, 2.2 * np.column_stack([np.cos(angle), np.sin(angle)])])
            (run / "boost").mkdir()
            Region.from_clouds([cloud], [-4, -4], [4, 4], 1.5, best=-math.log(2 * math.pi)).save(
                run / "boost" / "region_it_2.npz")
            entry = dict(file="boost/region_it_2.npz", edge=1.5, sha256=hashlib.sha256(
                (run / "boost" / "region_it_2.npz").read_bytes()).hexdigest())
            inside = -0.0027

            # No option: the region, as profiles and derived chains have it.
            printed = command(run, "--n-live", "300", "--quiet")
            report = json.loads((run / "evidence" / "evidence_it_2_nlive300.json").read_text())
            self.assertIn("network inside the region of boost/region_it_2.npz (edge 1.5)", printed.splitlines()[0])
            self.assertEqual(report["region"], entry)
            self.assertLess(abs(report["lnZ"] - inside), 4 * report["sigma_lnZ"] + 0.02)

            # The bare option where the cluster's job script puts it, before the user's words.
            command(run, "-i", "2", "--region", "--n-live", "300", "--quiet", "--tag", "flag")
            flagged = json.loads((run / "evidence" / "evidence_it_2_nlive300_flag.json").read_text())
            self.assertEqual(flagged["region"], entry)
            self.assertLess(abs(flagged["lnZ"] - inside), 4 * flagged["sigma_lnZ"] + 0.02)

            # --region none: the bare network over the box, rise and all.
            printed = command(run, "--region", "none", "--n-live", "300", "--quiet", "--tag", "bare")
            bare = json.loads((run / "evidence" / "evidence_it_2_nlive300_bare.json").read_text())
            self.assertIn("--region none: the bare network, not the region of boost/region_it_2.npz",
                          printed.splitlines()[0])
            self.assertIsNone(bare["region"])
            self.assertGreater(bare["lnZ"], 20.0)

            # --region E: the same region with its edge at E reaches, beside the default output. At 2
            # it reaches past the box's face and holds the rise; at 1 it holds no rise and less of the
            # Gaussian than at 1.5 (its mass inside summed on a grid).
            printed = command(run, "--region", "2", "--n-live", "300", "--quiet")
            wide = json.loads((run / "evidence" / "evidence_it_2_nlive300_edge2.json").read_text())
            self.assertIn("network inside the region of boost/region_it_2.npz (edge 2)", printed.splitlines()[0])
            self.assertEqual(wide["region"], dict(entry, edge=2.0))
            self.assertEqual(wide["target"]["region"], dict(entry, edge=2.0))
            self.assertGreater(wide["lnZ"], 20.0)
            command(run, "--region", "1", "--n-live", "300", "--quiet", "--tag", "seed42")
            tight = json.loads((run / "evidence" / "evidence_it_2_nlive300_seed42_edge1.json").read_text())
            self.assertEqual(tight["region"]["edge"], 1.0)
            region = Region.load(run / "boost" / "region_it_2.npz")
            region.inflate = 1.0
            grid = np.linspace(-4, 4, 801)
            points = np.column_stack([a.ravel() for a in np.meshgrid(grid, grid)])
            density = np.exp(-0.5 * (points ** 2).sum(axis=1)) / (2 * math.pi) * (grid[1] - grid[0]) ** 2
            exact = math.log(density[region.inside(points)].sum())
            self.assertLess(exact, inside - 0.03)
            self.assertLess(abs(tight["lnZ"] - exact), 4 * tight["sigma_lnZ"] + 0.02)
            # the default output is the one the run's own edge made
            self.assertEqual(json.loads((run / "evidence" / "evidence_it_2_nlive300.json").read_text())["region"], entry)

            # A region saved after the network is not the network's.
            (run / "boost" / "region_it_2.npz").rename(run / "boost" / "region_it_3.npz")
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                command(run, "--region", "run", "--n-live", "300", "--quiet", "--tag", "later")


if __name__ == "__main__":
    unittest.main()
