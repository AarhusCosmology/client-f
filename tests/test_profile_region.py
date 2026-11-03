"""The profile setting `region`: a boost network is searched inside the region of its boosts.

Analytic: the two-parameter toy run of test_profiles_workflow, given a region file and a
network that rises where the truth does not. No CLASS or Planck resources.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from profiles import region as boosts  # noqa: E402
from profiles.config import ProfileConfig  # noqa: E402
from test_profiles_workflow import make_toy_run  # noqa: E402

RADIUS = 2.2     # the toy's boost points lie within this distance of the peak


def add_region(run, inflate=1.0, iteration=2):
    """RUN/boost/region_it_N.npz from points of the toy's posterior within RADIUS of its peak."""
    from sampling.region import Region

    cloud = np.random.default_rng(5).normal(size=(4000, 2))
    cloud = cloud[np.hypot(*cloud.T) <= RADIUS][:600]
    angle = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    cloud = np.vstack([cloud, RADIUS * np.column_stack([np.cos(angle), np.sin(angle)])])
    region = Region.from_clouds([cloud], [-4, -4], [4, 4], inflate, best=-6.0)
    (run / "boost").mkdir(exist_ok=True)
    region.save(run / "boost" / f"region_it_{iteration}.npz")
    return region


def add_rise(run, direction, start, strength):
    """The toy's network plus strength * relu(direction . x - start)^2, which the truth lacks."""
    import tensorflow as tf
    from model.network import load_model

    path = run / "trained_models/model_it_2.keras"
    base = load_model(path)
    ramp = tf.keras.layers.Dense(1, activation="relu", name="ramp")(base.input)
    square = tf.keras.layers.Multiply(name="ramp_squared")([ramp, ramp])
    rise = tf.keras.layers.Dense(1, use_bias=False, name="rise")(square)
    model = tf.keras.Model(base.input, tf.keras.layers.Add(name="with_rise")([base.output, rise]))
    model.get_layer("ramp").set_weights([np.array(direction, np.float32).reshape(-1, 1),
                                         np.array([-start], np.float32)])
    model.get_layer("rise").set_weights([np.array([[strength]], np.float32)])
    model.save(path)


def study(out, name):
    return json.loads((out / name).read_text())


def calls(out):
    from profiles.evaluation import aggregate_counts

    return aggregate_counts(out / "calls")["backend_attempts"]


class Wall(unittest.TestCase):
    def setUp(self):
        import tensorflow as tf
        from sampling.region import Region

        rng = np.random.default_rng(0)
        cloud = rng.normal(0.5, 0.05, (400, 3))
        self.region = Region.from_clouds([cloud, cloud + [0.3, 0, 0]], np.zeros(3), np.ones(3), 1.5, best=-100.0)
        inputs = tf.keras.Input((3,))
        hidden = tf.keras.layers.Dense(8, activation="tanh")(
            tf.keras.layers.Normalization(mean=[0.5] * 3, variance=[0.01] * 3)(inputs))
        self.network = tf.keras.Model(inputs, tf.keras.layers.Dense(1)(hidden))
        self.x = rng.uniform(0, 1, (4000, 3))

    def test_the_network_inside_and_the_wall_outside(self):
        import tensorflow as tf
        from model.network import copy_model_dtype, restrict_model
        from sampling.region import STRENGTH

        restricted = restrict_model(self.network, self.region)
        self.assertEqual(restricted.compute_dtype, "float64")     # the batched surrogate takes it as it is
        rho = self.region.rho(self.x)
        self.assertTrue(0.05 < np.mean(rho <= 1.5) < 0.5)
        bare = copy_model_dtype(self.network, "float64")(self.x, training=False).numpy()[:, 0]
        expected = np.where(rho <= 1.5, bare, self.region.floor - STRENGTH * (rho - 1.5) ** 2)
        values = restricted(tf.constant(self.x), training=False).numpy()
        self.assertEqual(values.shape, (len(self.x), 1))
        np.testing.assert_allclose(values[:, 0], expected, rtol=1e-12)
        np.testing.assert_array_equal(values[rho <= 1.5, 0], bare[rho <= 1.5])
        self.assertLess(values[rho > 1.5].max(), self.region.floor + 1e-9)
        # What the node optimiser does with a model (profiles/surrogate.py).
        copy = copy_model_dtype(restricted, "float64")
        np.testing.assert_array_equal(copy(tf.constant(self.x), training=False).numpy(), values)

    def test_derivatives_at_the_centre_of_a_mode(self):
        import tensorflow as tf
        from model.network import restrict_model

        restricted = restrict_model(self.network, self.region)
        centre = tf.constant(self.region.lower + self.region.means * self.region.width)
        with tf.GradientTape() as outer:
            outer.watch(centre)
            with tf.GradientTape() as tape:
                tape.watch(centre)
                value = restricted(centre, training=False)[:, 0]
            gradient = tape.gradient(value, centre)
        hessian = outer.batch_jacobian(gradient, centre)
        self.assertTrue(np.isfinite(gradient.numpy()).all() and np.isfinite(hessian.numpy()).all())

    def test_only_the_run_s_own_model_is_restricted(self):
        from model.network import restrict_model

        with self.assertRaisesRegex(TypeError, "leave the region to the profile code"):
            restrict_model(lambda x, training=False: x, self.region)


class Setting(unittest.TestCase):
    def test_values(self):
        self.assertEqual(ProfileConfig(parameters=["x"]).region, "run")
        for good in ("run", 1, 1.0, 2.5, None):
            ProfileConfig(parameters=["x"], region=good).validate()
        for bad in (0, -1.0, True, False, "Run", "off", float("inf"), float("nan"), [2.0]):
            with self.subTest(region=bad), self.assertRaisesRegex(ValueError, "region must be run"):
                ProfileConfig(parameters=["x"], region=bad).validate()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.yaml"
            for text, expected in [("region: run", "run"), ("region: 1.5", 1.5), ("region: null", None), ("", "run")]:
                path.write_text(f"parameters: [x]\n{text}\n")
                self.assertEqual(ProfileConfig.load(path).region, expected)

    def test_the_region_that_belongs_to_a_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            (run / "boost").mkdir(parents=True)
            self.assertIsNone(boosts.resolve(run, 30, "run"))            # a run without boosts
            for iteration, inflate in [(20, 2.0), (25, 3.0)]:
                add_region(run, inflate=inflate, iteration=iteration)
            (run / "boost" / "region_it_x.npz").write_bytes(b"")           # not a region of a network
            self.assertIsNone(boosts.region_file(run, 15))                # the ordinary loop's network
            self.assertEqual(boosts.region_file(run, 22).name, "region_it_20.npz")
            self.assertEqual(boosts.region_file(run, 25).name, "region_it_25.npz")
            self.assertEqual(boosts.region_file(run, 30).name, "region_it_25.npz")
            self.assertIsNone(boosts.resolve(run, 15, "run"))
            self.assertIsNone(boosts.resolve(run, 30, None))              # the bare network, asked for
            own = boosts.resolve(run, 22, "run")
            self.assertEqual((own["file"], own["edge"]), ("boost/region_it_20.npz", 2.0))
            self.assertEqual(len(own["sha256"]), 64)
            tight = boosts.resolve(run, 30, 1)
            self.assertEqual((tight["file"], tight["edge"]), ("boost/region_it_25.npz", 1.0))
            self.assertIsInstance(tight["edge"], float)
            region = boosts.load(run, tight)
            self.assertEqual(region.inflate, 1.0)                         # the file says 3
            self.assertIn("inside region_it_25.npz only: 1 mode(s)", boosts.describe(region, tight))


class Cutting(unittest.TestCase):
    """Nodes whose optimum is not inside the region leave the ends of a parameter's range."""

    @staticmethod
    def result(grid, best):
        return dict(full_parameter_names=["a", "b"], best_x=[best, 0.0],
                    profiles=dict(a=dict(grid=list(grid), xopt=[[v, 0.0] for v in grid])))

    def test_the_nodes_around_the_best_fit_are_kept(self):
        grid = np.linspace(0.0, 8.0, 9)
        flags = lambda *outside: dict(a=np.isin(np.arange(9), outside))   # noqa: E731
        grids, cuts = boosts.shorten(self.result(grid, 4.2), flags(0, 1, 8), [grid], ["a"])
        np.testing.assert_allclose(grids[0], np.linspace(2.0, 7.0, 9))    # as many nodes as before
        self.assertEqual(cuts, dict(a=dict(asked=[0.0, 8.0], kept=[2.0, 7.0], outside=[0.0, 1.0, 8.0])))
        # Nothing outside: the grid is the one given, and nothing is recorded.
        grids, cuts = boosts.shorten(self.result(grid, 4.2), flags(), [grid], ["a"])
        np.testing.assert_array_equal(grids[0], grid)
        self.assertEqual(cuts, {})
        # A stretch outside in the middle: the side of the best fit is kept, the other is not
        # reached through a part the network does not know.
        grids, cuts = boosts.shorten(self.result(grid, 6.6), flags(3, 4), [grid], ["a"])
        self.assertEqual(cuts["a"]["kept"], [5.0, 8.0])
        grids, cuts = boosts.shorten(self.result(grid, 1.1), flags(3, 4), [grid], ["a"])
        self.assertEqual(cuts["a"]["kept"], [0.0, 2.0])
        # A best fit outside every node: the nearest node inside decides.
        grids, cuts = boosts.shorten(self.result(grid, 20.0), flags(3, 4), [grid], ["a"])
        self.assertEqual(cuts["a"]["kept"], [5.0, 8.0])

    def test_fewer_than_two_nodes_inside_ends_the_study(self):
        grid = np.linspace(0.0, 4.0, 5)
        for outside in ([0, 1, 2, 3, 4], [0, 1, 3, 4], [0, 2, 4]):
            with self.subTest(outside=outside), self.assertRaisesRegex(ValueError, "a: fewer than two neighbouring nodes"):
                boosts.shorten(self.result(grid, 2.0), dict(a=np.isin(np.arange(5), outside)), [grid], ["a"])

    def test_scanning_until_every_node_is_inside(self):
        from sampling.region import Region

        cloud = np.random.default_rng(1).normal(0.0, 1.0, (500, 2))
        region = Region.from_clouds([cloud], [-10, -10], [10, 10], 1.0, best=0.0)
        limit = float(region.reach[0] * cloud[:, 0].std(ddof=1))          # rho = 1 along the first axis, about
        scanned, printed = [], io.StringIO()

        def scan_once(grids):
            scanned.append(np.asarray(grids[0]))
            return self.result(grids[0], 0.0)

        grid = np.linspace(-9.0, 9.0, 13)
        with contextlib.redirect_stdout(printed):
            result = boosts.scan(scan_once, region, [grid], ["a"])
        self.assertEqual(len(scanned), 2)
        self.assertFalse(boosts.outside(region, result)["a"].any())
        cut = result["region_cuts"]["a"]
        self.assertEqual(cut["asked"], [-9.0, 9.0])
        self.assertLess(max(map(abs, cut["kept"])), limit)
        self.assertGreater(max(map(abs, cut["kept"])), limit - 1.5)        # to the last node inside, a spacing at most
        self.assertTrue(all(abs(v) > boosts.INSIDE * limit - 0.1 for v in cut["outside"]))
        self.assertIn("of a have their optimum outside the region", printed.getvalue())
        # Every node inside: one scan, nothing recorded.
        scanned.clear()
        result = boosts.scan(scan_once, region, [np.linspace(-1.0, 1.0, 5)], ["a"])
        self.assertEqual((len(scanned), "region_cuts" in result), (1, False))

        # A network that rises to the edge wherever the range ends: the study gives up.
        def pressed(grids):
            result = self.result(grids[0], 0.0)
            result["profiles"]["a"]["xopt"][-1] = [limit * 1.2, 0.0]
            return result

        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, rf"after {boosts.PASSES} scans .*a \(1\)"):
            boosts.scan(pressed, region, [np.linspace(-1.0, 1.0, 5)], ["a"])

    def test_a_node_pressed_against_the_edge(self):
        # The network rises towards one corner, from inside the region on: on the slice
        # x_1 = 1.95 its best point in the region is on the region's edge, which no search can
        # certify. That node is cut; the others, and the best fit, are untouched.
        from likelihood.surrogate import SurrogateMetadata
        from model.network import load_model, restrict_model
        from profiles.hybrid import HybridConfig, scan_hybrid

        with tempfile.TemporaryDirectory() as temporary:
            run = make_toy_run(Path(temporary))
            add_rise(run, [1.0, 1.0], 2.4, 3.0)
            region = add_region(run)
            metadata = SurrogateMetadata.load(run / "metadata.json")
            data = np.loadtxt(run / "training_data/data_it_2.csv", delimiter=",", skiprows=1)
            model = restrict_model(load_model(run / "trained_models/model_it_2.keras"), region)
            first = []

            def scan_once(grids):
                result, _ = scan_hybrid(model, metadata, data, ["x_1"], grids, hybrid=HybridConfig(seed=42),
                                        log=lambda *a, **k: None)
                first.append(result) if not first else None
                return result

            with contextlib.redirect_stdout(io.StringIO()):
                result = boosts.scan(scan_once, region, [np.linspace(-1.95, 1.95, 5)], ["x_1"])
        record = first[0]["profiles"]["x_1"]
        rho = region.rho(np.asarray(record["xopt"]))
        np.testing.assert_array_equal(rho > boosts.INSIDE, [False, False, False, False, True])
        self.assertAlmostEqual(rho[-1], 1.0, places=3)
        self.assertFalse(record["optimization"][-1]["stationary"])
        self.assertFalse(first[0]["optimization_quality"]["converged"])
        self.assertEqual(result["region_cuts"]["x_1"]["outside"], [1.95])
        np.testing.assert_allclose(result["region_cuts"]["x_1"]["kept"], [-1.95, 0.975])
        self.assertTrue(result["optimization_quality"]["converged"])
        np.testing.assert_allclose(result["best_x"], [0.0, 0.0], atol=1e-6)
        final = result["profiles"]["x_1"]
        np.testing.assert_allclose(2 * (result["best_logl"] - np.asarray(final["logl"])), np.asarray(final["grid"]) ** 2, atol=1e-6)


class Study(unittest.TestCase):
    def test_a_boost_network_is_searched_inside_its_region(self):
        # The network has a maximum the truth lacks, 3.25 above the peak at the box edge, far
        # outside the region of the boosts (a real 49-parameter run: hpc/README.md).
        from profiles.workflow import run_workflow

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = make_toy_run(root)
            add_rise(run, [1.0, 0.0], 2.5, 5.0)
            add_region(run)

            def config(**settings):
                return ProfileConfig(parameters=["x_1"], ranges={"x_1": [-2.0, 2.0]}, profile_points=5,
                                     scan_points=9, workers=2, **settings)

            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                run_workflow(run, root / "region", config())
                run_workflow(run, root / "bare", config(region=None))
            self.assertIn("region: the network is searched inside region_it_2.npz only", printed.getvalue())
            manifest = study(root / "region", "manifest.json")
            self.assertEqual((manifest["configuration"]["region"], manifest["region"]["file"], manifest["region"]["edge"]),
                             ("run", "boost/region_it_2.npz", 1.0))
            self.assertIsNone(study(root / "bare", "manifest.json")["region"])
            inside, bare = (study(root / name, "surrogate_profiles.json") for name in ("region", "bare"))
            np.testing.assert_allclose(inside["best_x"], [0.0, 0.0], atol=1e-6)
            np.testing.assert_allclose(bare["best_x"], [4.0, 0.0], atol=1e-6)     # the maximum the truth lacks
            self.assertAlmostEqual(bare["best_logl"] - inside["best_logl"], 3.25, places=4)
            truth = study(root / "region", "true_profiles.json")
            row = truth["profiles"]["x_1"]
            np.testing.assert_allclose(2 * (truth["best_logl"] - np.array(row["logl"])), np.array(row["grid"]) ** 2, atol=0.03)
            self.assertTrue(all(node["reason"] == "converged" for node in row["optimization"]))
            self.assertLess(calls(root / "region"), calls(root / "bare"))
            self.assertFalse("region" in study(root / "region", "ranges.json")["x_1"])   # nothing was cut

            # What a study depends on is the region, not the words that asked for it.
            for out, good, bad in [(root / "region", config(region=1.0), config(region=None)),
                                   (root / "bare", config(region=None), config(region=2.0))]:
                before = calls(out)
                (out / "complete.json").unlink()
                with self.assertRaisesRegex(ValueError, "resume refused"):
                    run_workflow(run, out, bad, resume=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    run_workflow(run, out, good, resume=True)
                self.assertEqual(calls(out), before)

    def test_a_range_beyond_the_region_is_cut_and_recorded(self):
        from profiles.workflow import run_workflow

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, out = make_toy_run(root), root / "profiles"
            add_region(run)
            # x_0: an explicit range to the box's edge. x_1: no range, so a scan of the whole
            # box chooses 3.5 sigma, beyond the toy's boost points too.
            c = ProfileConfig(parameters=["x_0", "x_1"], ranges={"x_0": [-3.9, 3.9]}, profile_points=5,
                              scan_points=9, workers=2)
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                run_workflow(run, out, c)
            self.assertIn("2 node(s) of x_0 have their optimum outside the region", printed.getvalue())
            baseline = study(out, "baseline_profiles.json")
            self.assertEqual(baseline["region_cuts"]["x_1"]["asked"], [-4.0, 4.0])
            self.assertEqual(baseline["region_cuts"]["x_1"]["outside"], [-4.0, -3.0, 3.0, 4.0])
            np.testing.assert_allclose(baseline["profiles"]["x_1"]["grid"], np.linspace(-2.0, 2.0, 9))
            ranges = study(out, "ranges.json")
            self.assertEqual(ranges["x_0"]["requested"], [-3.9, 3.9])
            np.testing.assert_allclose(ranges["x_0"]["actual"], [-1.95, 1.95])
            self.assertEqual(ranges["x_0"]["region"]["outside"], [-3.9, 3.9])
            self.assertEqual(ranges["x_1"]["kind"], "surrogate 1-sigma widths")
            np.testing.assert_allclose(ranges["x_1"]["requested"], [-3.5, 3.5])
            np.testing.assert_allclose(ranges["x_1"]["actual"], [-1.75, 1.75])
            surrogate, truth = study(out, "surrogate_profiles.json"), study(out, "true_profiles.json")
            self.assertEqual(sorted(surrogate["region_cuts"]), ["x_0", "x_1"])
            region = boosts.load(run, study(out, "manifest.json")["region"])
            for name in ("x_0", "x_1"):
                row = truth["profiles"][name]
                np.testing.assert_allclose(row["grid"], np.linspace(*ranges[name]["actual"], 5))
                self.assertLess(region.rho(np.asarray(surrogate["profiles"][name]["xopt"])).max(), boosts.INSIDE)
                np.testing.assert_allclose(2 * (truth["best_logl"] - np.array(row["logl"])), np.array(row["grid"]) ** 2, atol=0.03)
                self.assertTrue(all(node["reason"] == "converged" for node in row["optimization"]))
            # Resumed, the study keeps its cut ranges and evaluates nothing.
            before = calls(out)
            (out / "complete.json").unlink()
            with contextlib.redirect_stdout(io.StringIO()):
                run_workflow(run, out, c, resume=True)
            self.assertEqual(calls(out), before)
            self.assertEqual(study(out, "ranges.json"), ranges)

    def test_a_surface_is_searched_inside_the_region_too(self):
        from profiles.workflow import run_workflow

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = make_toy_run(root)
            add_rise(run, [1.0, 0.0], 2.5, 5.0)
            add_region(run)
            out = root / "surface"
            c = ProfileConfig(dimension=2, parameters=["x_0", "x_1"], ranges={"x_0": [-1.2, 1.2], "x_1": [-1.2, 1.2]},
                              profile_points=3, workers=2)
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                run_workflow(run, out, c)
            self.assertIn("region: the network is searched inside region_it_2.npz only", printed.getvalue())
            self.assertEqual(study(out, "manifest.json")["region"]["edge"], 1.0)
            surrogate, truth = study(out, "surrogate_surface.json"), study(out, "true_surface.json")
            np.testing.assert_allclose(surrogate["best_x"], [0.0, 0.0], atol=1e-5)   # not the box edge
            xx, yy = np.meshgrid(*truth["axes"], indexing="ij")
            np.testing.assert_allclose(2 * (truth["best_logl"] - np.array(truth["logl"])), xx ** 2 + yy ** 2, atol=0.03)

    def test_a_resumed_surface_reads_its_finished_scan(self):
        # The true stage's node checkpoints are tied to the scan's optima bit for bit. The
        # scan was done again on every resume and had to come out identical; the finished
        # one is read instead.
        from unittest import mock
        from profiles.evaluation import aggregate_counts
        from profiles.workflow import run_workflow

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, out = make_toy_run(root), root / "surface"
            c = ProfileConfig(dimension=2, parameters=["x_0", "x_1"], ranges={"x_0": [-1.2, 1.2], "x_1": [-1.2, 1.2]},
                              profile_points=3, workers=2)
            run_workflow(run, out, c)
            scan, truth = study(out, "surrogate_surface.json"), study(out, "true_surface.json")
            before = aggregate_counts(out / "calls")["backend_attempts"]
            (out / "complete.json").unlink()
            printed = io.StringIO()
            with mock.patch("profiles.hybrid_surface.scan_surface_hybrid", side_effect=AssertionError("scanned again")), \
                    contextlib.redirect_stdout(printed):
                run_workflow(run, out, c, resume=True)
            self.assertIn("surrogate grid of the earlier start of this study is used again", printed.getvalue())
            self.assertEqual(study(out, "surrogate_surface.json"), scan)
            self.assertEqual(study(out, "true_surface.json")["logl"], truth["logl"])
            self.assertEqual(aggregate_counts(out / "calls")["backend_attempts"], before)
            # a scan that was not finished and checked is not taken: it is done again
            (out / "complete.json").unlink()
            unfinished = dict(scan)
            unfinished.pop("names")
            (out / "surrogate_surface.json").write_text(json.dumps(unfinished))
            run_workflow(run, out, c, resume=True)
            self.assertEqual(study(out, "surrogate_surface.json")["names"], ["x_0", "x_1"])
            self.assertEqual(aggregate_counts(out / "calls")["backend_attempts"], before)

    def test_a_resumed_surface_reads_its_range_scan(self):
        # The axes go into the node checkpoints' signature: a scan done again need not give the
        # same last digits on another device or build, and the true stage would then be refused.
        from unittest import mock
        from profiles.workflow import run_workflow

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, out = make_toy_run(root), root / "surface"
            c = ProfileConfig(dimension=2, parameters=["x_0", "x_1"], profile_points=3, scan_points=9, workers=2)
            run_workflow(run, out, c)
            axes = study(out, "surrogate_surface.json")["axes"]
            (out / "complete.json").unlink()
            with mock.patch("profiles.hybrid.scan_hybrid", side_effect=AssertionError("scanned again")):
                run_workflow(run, out, c, resume=True)
            self.assertEqual(study(out, "surrogate_surface.json")["axes"], axes)

    def test_a_surface_beyond_the_region_stops_before_any_true_call(self):
        # A range is cut at its ends; a rectangle is not. Its nodes outside used to reach the
        # true stage with a point of the wall as their start.
        from profiles.workflow import run_workflow

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, out = make_toy_run(root), root / "profiles"
            add_region(run)
            c = ProfileConfig(dimension=2, parameters=["x_0", "x_1"], ranges={"x_0": [-3.0, 3.0], "x_1": [-1.2, 1.2]},
                              profile_points=3, workers=2)
            for resume in (False, True):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(
                        RuntimeError, r"6 of 9 nodes of the surface in x_0 and x_1 .* outside the region .*\(-3, -1\.2\) rho"):
                    run_workflow(run, out, c, resume=resume)
                self.assertFalse((out / "true_surface.json").exists())
                stages = {json.loads(line)["stage"] for file in (out / "calls").glob("*.jsonl")
                          for line in file.read_text().splitlines()}
                # the check of the saved rows alone (and, resumed, the reading of its journal)
                self.assertEqual(stages - {"resume/import"}, {"audit"})


if __name__ == "__main__":
    unittest.main()
