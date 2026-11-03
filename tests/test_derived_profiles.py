"""Profiles in derived parameters through derived-coordinate views (derived/coordinates.py).

Analytic: a unit Gaussian whose derived parameter d = x_0 + x_1 / 2 has the exact profile
Delta chi^2(d) = d^2 / 1.25, run through the unchanged profile workflow on the view.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.config import DerivedConfig
from derived.coordinates import (
    PENALTY,
    choose_pivots,
    emulator_parameters,
    make_view,
    pivot_map_model,
    to_physical,
    view_model,
)
from derived.emulator import build_derived_model, train_derived_model
from model.network import copy_model_dtype

NAMES = ["x_0", "x_1", "x_2"]
BOUNDS = [(-4.0, 4.0)] * 3


def curved(x):
    """Monotone in x_0, not in x_1."""
    return (x[:, 0] + 0.3 * x[:, 1] ** 2 + 0.2 * np.sin(x[:, 2]))[:, None]


def trained_emulator(seed=4):
    tf.keras.utils.set_random_seed(seed)
    x = np.random.default_rng(seed).uniform(-4, 4, (1500, 3)).astype(np.float32)
    cfg = DerivedConfig.from_dict(dict(n_layers=2, n_neurons=32, n_epochs=300, lr_patience=20, lr_grace=20))
    model, *_ = train_derived_model(x, curved(x.astype(float)), -0.5 * (x**2).sum(1), [0, 1, 2],
                                    ["d"], cfg, rng=np.random.default_rng(seed))
    return model


class PivotSolveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.emulator = trained_emulator()
        cls.pmap = pivot_map_model(3, [0], [BOUNDS[0]], emulator_parameters(cls.emulator, [0]))

    def test_round_trip_and_float64_clone(self):
        x = np.random.default_rng(1).uniform(-3, 3, (200, 3))
        x[:, 0] = np.clip(x[:, 0], -1, 1)             # keep every target reachable
        d = self.emulator.predict(x.astype(np.float32), verbose=0).astype(float)[:, 0]
        z = x.copy()
        z[:, 0] = d
        back, residual = to_physical(self.pmap, z)
        self.assertLess(np.abs(back - x).max(), 1e-4)
        self.assertLess(np.abs(residual).max(), 1e-8)
        # The float32 model agrees with the float64 clone to float32 precision.
        out32 = self.pmap(tf.constant(z, tf.float32)).numpy()
        np.testing.assert_allclose(out32[:, :3], back, atol=1e-4)

    def test_unreachable_target_is_clamped_and_penalised(self):
        z = np.array([[50.0, 0.0, 0.0]])             # d = 50 needs x_0 far above 4
        x, residual = to_physical(self.pmap, z)
        self.assertAlmostEqual(x[0, 0], 4.0)
        self.assertGreater(PENALTY * float(np.sum(residual**2)), 100.0)

    def test_pivot_choice_requires_monotonicity(self):
        anchor = np.zeros(3)
        pivots, report = choose_pivots(self.emulator, NAMES, BOUNDS, ["d"], ["d"], anchor, np.ones(3))
        self.assertEqual(pivots, ["x_0"])
        self.assertFalse(report["d"]["candidates"]["x_1"]["monotone"])
        with self.assertRaises(ValueError):
            choose_pivots(self.emulator, NAMES, BOUNDS, ["d"], ["d"], anchor, np.ones(3), exclude=["x_0"])


LINEAR = {"d": [1.0, 0.5, 0.0], "e": [0.0, -0.3, 1.0]}


def exact_linear_emulator(coefficients):
    """The derived emulator architecture representing d = A x exactly (Alsing with
    gamma = 1 is the identity)."""
    A = np.asarray(coefficients, np.float32)                     # (k, 3)
    k = len(A)
    x = np.random.default_rng(0).uniform(-3, 3, (64, 3)).astype(np.float32)
    model, _, _, _ = build_derived_model(x, (x @ A.T).astype(float), [0, 1, 2], 1, 3, "alsing")
    for layer in model.layers:
        kind = type(layer).__name__
        if kind == "Normalization":
            layer.set_weights([np.zeros(3, np.float32), np.ones(3, np.float32), np.array(64, np.int64)])
            layer.finalize_state()
        elif kind == "Dense" and layer.name != "standardized_derived":
            layer.set_weights([np.eye(3, dtype=np.float32), np.zeros(3, np.float32)])
        elif kind == "Alsing":
            layer.set_weights([np.ones(3, np.float32), np.ones(3, np.float32)])
        elif kind == "Dense":
            layer.set_weights([A.T.copy(), np.zeros(k, np.float32)])
        elif kind == "DerivedDenormalization":
            layer.set_weights([np.zeros(k, np.float32), np.ones(k, np.float32)])
    return model


def make_linear_run(root, derived=("d",)):
    """A saved CLiENT iteration: unit Gaussian truth, exact network, exact emulator of
    the linear derived parameters LINEAR[name] . x."""
    from derived.metadata import DerivedMetadata
    from likelihood.base import ParameterInfo
    from likelihood.cobaya import CobayaLikelihood
    from likelihood.surrogate import SurrogateMetadata

    run = root / "run"
    for name in ["likelihood_input", "trained_models", "training_data", "derived_data"]:
        (run / name).mkdir(parents=True)
    info = dict(
        likelihood={"gaussian_mixture": dict(means=[[0.0] * 3], covs=[np.eye(3).tolist()],
                                             input_params_prefix="x")},
        params={**{n: dict(prior=dict(min=-4.0, max=4.0)) for n in NAMES},
                **{name: dict(derived="lambda x_0, x_1, x_2: " + " + ".join(
                    f"{c} * x_{i}" for i, c in enumerate(LINEAR[name])), latex=name)
                   for name in derived}},
    )
    (run / "likelihood_input/toy.yaml").write_text(yaml.safe_dump(info, sort_keys=False))
    (run / "run.yaml").write_text(yaml.safe_dump(dict(likelihood=dict(wrapper="cobaya", input="input/toy.yaml"))))
    truth = CobayaLikelihood(str(run / "likelihood_input/toy.yaml"))
    SurrogateMetadata.from_likelihood(truth).save(run / "metadata.json")
    DerivedMetadata(parameters=tuple(ParameterInfo(name=n, label=n) for n in derived),
                    inputs=tuple(NAMES)).save(run / "derived_data")
    x = np.random.default_rng(91).uniform(-3, 3, (120, 3)).astype(np.float32)
    y = np.array([truth.loglkl(p) for p in x])
    peak = truth.loglkl([0.0, 0.0, 0.0])
    truth.cobaya_model.close()
    np.savetxt(run / "training_data/data_it_2.csv", np.column_stack([x, y]), delimiter=",",
               header="x_0,x_1,x_2,loglkl", comments="")
    inputs = tf.keras.Input((3,))
    norm = tf.keras.layers.Normalization(mean=[0.0] * 3, variance=[1.0] * 3)(inputs)
    hidden = tf.keras.layers.Dense(3)(norm)
    q = tf.keras.layers.Dot(axes=1)([hidden, hidden])
    output = tf.keras.layers.Dense(1)(q)
    network = tf.keras.Model(inputs, output)
    network.layers[2].set_weights([np.eye(3, dtype=np.float32), np.zeros(3, np.float32)])
    network.layers[-1].set_weights([np.array([[-0.5]], np.float32), np.array([peak], np.float32)])
    network.save(run / "trained_models/model_it_2.keras")
    exact_linear_emulator([LINEAR[n] for n in derived]).save(run / "derived_data/derived_model_it_2.keras")
    return run


class DerivedProfileWorkflowTests(unittest.TestCase):
    def test_view_network_loads_in_a_fresh_process(self):
        # The profile workflow's master loads the view network through
        # model.network.load_model without importing derived.coordinates itself.
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_view(make_linear_run(root), root / "view", ["d"])
            repo = Path(__file__).resolve().parents[1]
            code = ("import sys; sys.path.insert(0, sys.argv[1]); from model.network import load_model; "
                    "print(type(load_model(sys.argv[2]).layers[1]).__name__)")
            result = subprocess.run([sys.executable, "-c", code, str(repo),
                                     str(root / "view/trained_models/model_it_2.keras")],
                                    capture_output=True, text=True, cwd=repo)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            self.assertEqual(result.stdout.strip().splitlines()[-1], "PivotSolve")

    def test_view_profile_matches_the_exact_constrained_profile(self):
        from derived.view_cli import evaluate_nodes
        from profiles.config import ProfileConfig
        from profiles.workflow import run_workflow, sha

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = make_linear_run(root)
            before = {p: sha(p) for p in run.rglob("*") if p.is_file()}
            summary = make_view(run, root / "view", ["d"])
            # the view names the files it was made from: the network and metadata it loaded,
            # the derived emulator and the training set it read
            self.assertEqual(summary["source_target"]["model_sha256"], sha(run / "trained_models/model_it_2.keras"))
            self.assertEqual(summary["source_target"]["metadata_sha256"], sha(run / "metadata.json"))
            self.assertIsNone(summary["source_target"]["region"])
            self.assertIsNone(summary["region"])
            self.assertEqual(summary["source_files"], {
                "derived_data/derived_model_it_2.keras": sha(run / "derived_data/derived_model_it_2.keras"),
                "training_data/data_it_2.csv": sha(run / "training_data/data_it_2.csv")})
            self.assertEqual(json.loads((root / "view/view.json").read_text())["source_files"], summary["source_files"])
            self.assertEqual(summary["pivots"], {"d": "x_0"})
            self.assertEqual(summary["coordinates"], ["d", "x_1", "x_2"])
            self.assertLess(summary["roundtrip"]["max_residual"], 1e-10)

            # The view's network is the original one, composed with the pivot solve.
            from model.network import load_model
            view_net = copy_model_dtype(load_model(root / "view/trained_models/model_it_2.keras"), "float64")
            z = np.array([[1.0, 0.4, -0.3]])
            x = np.array([[1.0 - 0.2, 0.4, -0.3]])
            net = copy_model_dtype(load_model(run / "trained_models/model_it_2.keras"), "float64")
            np.testing.assert_allclose(view_net(tf.constant(z)).numpy(), net(tf.constant(x)).numpy(), rtol=1e-12)

            config = ProfileConfig(parameters=["d"], workers=2, profile_points=5,
                                   ranges={"d": [-2.0, 2.0]}, total_call_budget=3000)
            config.validate()
            out = root / "profile"
            run_workflow(root / "view", out, config)
            records = json.loads((out / "true_profiles.json").read_text())
            profile = records["profiles"]["d"]
            grid = np.array(profile["grid"])
            np.testing.assert_allclose(2 * (records["best_logl"] - np.array(profile["logl"])),
                                       grid**2 / 1.25, atol=0.03)

            report = evaluate_nodes(out, workers=1, log=lambda *_: None)
            self.assertLess(report["d"]["max_abs_offset"], 1e-8)
            self.assertLess(report["max_objective_reevaluation_difference"], 1e-9)
            # The source run is untouched.
            self.assertEqual(before, {p: sha(p) for p in run.rglob("*") if p.is_file()})

    def test_surface_in_two_derived_parameters(self):
        """(d, e) jointly, with two pivots. Exact: min over x_1 of
        (d - x_1/2)^2 + x_1^2 + (e + 0.3 x_1)^2."""
        from derived.view_cli import evaluate_nodes
        from profiles.config import ProfileConfig
        from profiles.workflow import run_workflow

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = make_linear_run(root, derived=("d", "e"))
            summary = make_view(run, root / "view", ["d", "e"])
            self.assertEqual(summary["pivots"], {"d": "x_0", "e": "x_2"})
            self.assertEqual(summary["coordinates"], ["d", "x_1", "e"])

            config = ProfileConfig(dimension=2, parameters=["d", "e"], workers=2, profile_points=3,
                                   ranges={"d": [-1.5, 1.5], "e": [-1.5, 1.5]}, total_call_budget=3000)
            config.validate()
            out = root / "surface"
            run_workflow(root / "view", out, config)
            surface = json.loads((out / "true_surface.json").read_text())
            d, e = np.meshgrid(surface["axes"][0], surface["axes"][1], indexing="ij")
            x1 = (0.5 * d - 0.3 * e) / 1.34
            exact = (d - 0.5 * x1) ** 2 + x1**2 + (e + 0.3 * x1) ** 2
            np.testing.assert_allclose(2 * (surface["best_logl"] - np.array(surface["logl"])), exact, atol=0.03)

            # Offsets are the emulator's error: here the float32 rounding of its weights
            # (-0.3 is not exact in float32), ~1e-8.
            report = evaluate_nodes(out, workers=1, log=lambda *_: None)
            self.assertLess(report["d"]["max_abs_offset"], 1e-6)
            self.assertLess(report["e"]["max_abs_offset"], 1e-6)

def add_view_region(run, radius=2.2, inflate=1.0):
    """RUN/boost/region_it_2.npz for make_linear_run: the boost points of its unit Gaussian
    within `radius` of the peak (test_profile_region.add_region, in three parameters)."""
    from sampling.region import Region

    rng = np.random.default_rng(5)
    cloud = rng.normal(size=(6000, 3))
    cloud = cloud[np.linalg.norm(cloud, axis=1) <= radius][:900]
    rim = rng.normal(size=(40, 3))
    cloud = np.vstack([cloud, radius * rim / np.linalg.norm(rim, axis=1, keepdims=True)])
    region = Region.from_clouds([cloud], [-4.0] * 3, [4.0] * 3, inflate, best=-1.5 * np.log(2 * np.pi))
    (run / "boost").mkdir()
    region.save(run / "boost/region_it_2.npz")
    return region


def boosted_view(root, derived=("d",)):
    """A view of a network trained on boosts: the linear run's network rises along x_2 beyond
    2.5, which the truth does not (3.25 above the peak at the box edge), far outside the
    region of its boosts."""
    from test_profile_region import add_rise

    run = make_linear_run(root, derived=derived)
    add_rise(run, [0.0, 0.0, 1.0], 2.5, 5.0)
    region = add_view_region(run)
    return run, region, make_view(run, root / "view", list(derived))


class ViewRegionTests(unittest.TestCase):
    """A view carries the region of its run's boosts, and its profiles search the network there."""

    def test_the_view_carries_the_region(self):
        from derived.coordinates import ViewRegion, restrict_view
        from model.network import load_model
        from profiles import region as boosts
        from run.storage import digest
        from sampling.region import STRENGTH

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, region, summary = boosted_view(root)
            view = root / "view"
            source = run / "boost/region_it_2.npz"
            self.assertEqual(summary["source_region"], "boost/region_it_2.npz")
            self.assertEqual(summary["region"], dict(file="region/region_it_2.npz", sha256=digest(source),
                                                     source="boost/region_it_2.npz"))
            self.assertEqual(summary["source_target"]["region"]["sha256"], digest(source))
            self.assertEqual((view / "region/region_it_2.npz").read_bytes(), source.read_bytes())
            # Not where a run keeps its own: the run's chains and evidence would read it in z.
            self.assertFalse((view / "boost").exists())

            entry = boosts.resolve(view, 2, "run")
            self.assertEqual((entry["file"], entry["edge"], entry["coordinates"]["file"]),
                             ("region/region_it_2.npz", 1.0, "likelihood_input/pivot_map.keras"))
            self.assertEqual(boosts.resolve(view, 2, 1.5)["edge"], 1.5)
            self.assertIsNone(boosts.resolve(view, 2, None))
            seen = boosts.load(view, entry)
            self.assertIsInstance(seen, ViewRegion)
            self.assertEqual((len(seen), seen.inflate, seen.rows), (1, 1.0, region.rows))

            # A point of the view is measured where the pivot solve takes it.
            rng = np.random.default_rng(3)
            x = rng.uniform(-3.5, 3.5, (3000, 3))
            x[:, 0] = np.clip(x[:, 0], -2.0, 2.0)
            z = x.copy()
            z[:, 0] = x[:, 0] + 0.5 * x[:, 1]
            rho = region.rho(x)
            self.assertTrue(0.1 < np.mean(rho <= 1.0) < 0.9)
            np.testing.assert_allclose(seen.rho(z), rho, rtol=1e-9)

            # The network inside, the region's wall outside; the feasibility penalty everywhere.
            network = load_model(view / "trained_models/model_it_2.keras")
            restricted = boosts.restrict(network, seen)
            self.assertEqual(restricted.compute_dtype, "float64")
            bare = copy_model_dtype(network, "float64")(tf.constant(z), training=False).numpy()[:, 0]
            values = restricted(tf.constant(z), training=False).numpy()[:, 0]
            np.testing.assert_allclose(values, np.where(rho <= 1.0, bare, region.floor - STRENGTH * (rho - 1.0) ** 2),
                                       rtol=1e-12)
            top = tf.constant([[0.0, 0.0, 4.0]], tf.float64)                         # the bare view's maximum
            peak = copy_model_dtype(network, "float64")(tf.zeros((1, 3), tf.float64)).numpy()[0, 0]
            self.assertAlmostEqual(copy_model_dtype(network, "float64")(top).numpy()[0, 0] - peak, 3.25, places=5)
            self.assertLess(restricted(top).numpy()[0, 0], region.floor)
            far = np.array([[9.0, 0.0, 0.0]])                                        # d = 9 needs x_0 = 9, beyond the box
            x_far, r_far = to_physical(seen.pivot_map, far)
            wall = region.floor - STRENGTH * (region.rho(x_far)[0] - 1.0) ** 2
            np.testing.assert_allclose(restricted(tf.constant(far)).numpy()[0, 0],
                                       wall - PENALTY * float(np.sum(r_far ** 2)), rtol=1e-12)

            # What the optimisers do with a model: copy it, differentiate it twice.
            copy = copy_model_dtype(restricted, "float64")
            np.testing.assert_array_equal(copy(tf.constant(z), training=False).numpy()[:, 0], values)
            point = tf.constant([[0.3, 0.2, -0.1]], tf.float64)
            with tf.GradientTape() as outer:
                outer.watch(point)
                with tf.GradientTape() as tape:
                    tape.watch(point)
                    value = restricted(point, training=False)[:, 0]
                gradient = tape.gradient(value, point)
            hessian = outer.batch_jacobian(gradient, point).numpy()[0]
            # -(x_0^2 + x_1^2 + x_2^2) / 2 with x_0 = d - x_1 / 2
            np.testing.assert_allclose(hessian, [[-1.0, 0.5, 0.0], [0.5, -1.25, 0.0], [0.0, 0.0, -1.0]], atol=1e-6)

            with self.assertRaisesRegex(ValueError, "derived-coordinate view"):
                restrict_view(load_model(run / "trained_models/model_it_2.keras"), region)

            # The copy and the pivot map are what the view was made with, and stay so.
            pivots = view / "likelihood_input/pivot_map.keras"
            kept = pivots.read_bytes()
            pivots.write_bytes(kept + b"changed")
            with self.assertRaisesRegex(ValueError, "region changed"):
                boosts.load(view, entry)
            pivots.write_bytes(kept)
            copy = view / "region/region_it_2.npz"
            copy.write_bytes(copy.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "not the copy the view was made with"):
                boosts.resolve(view, 2, "run")
            with self.assertRaisesRegex(ValueError, "region changed"):
                boosts.load(view, entry)

    def test_a_profile_of_the_view_is_searched_inside_the_region(self):
        from profiles.config import ProfileConfig
        from profiles.evaluation import aggregate_counts
        from profiles.workflow import run_workflow

        def config(**settings):
            c = ProfileConfig(parameters=["d"], workers=2, profile_points=5, ranges={"d": [-2.0, 2.0]},
                              total_call_budget=3000, **settings)
            c.validate()
            return c

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            boosted_view(root)
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                run_workflow(root / "view", root / "region", config())
                run_workflow(root / "view", root / "bare", config(region=None))
            self.assertIn("region: the network is searched inside region_it_2.npz only, in the coordinates of the "
                          "view's source run", printed.getvalue())
            manifest = json.loads((root / "region/manifest.json").read_text())
            self.assertEqual((manifest["region"]["file"], manifest["region"]["coordinates"]["file"]),
                             ("region/region_it_2.npz", "likelihood_input/pivot_map.keras"))
            inside, bare = (json.loads((root / name / "surrogate_profiles.json").read_text()) for name in ("region", "bare"))
            np.testing.assert_allclose(inside["best_x"], [0.0, 0.0, 0.0], atol=1e-5)
            np.testing.assert_allclose(bare["best_x"], [0.0, 0.0, 4.0], atol=1e-5)     # the maximum the truth lacks
            self.assertAlmostEqual(bare["best_logl"] - inside["best_logl"], 3.25, places=4)
            truth = json.loads((root / "region/true_profiles.json").read_text())
            row = truth["profiles"]["d"]
            np.testing.assert_allclose(2 * (truth["best_logl"] - np.array(row["logl"])), np.array(row["grid"]) ** 2 / 1.25,
                                       atol=0.03)
            self.assertTrue(all(node["reason"] == "converged" for node in row["optimization"]))

            # A resumed study must match the region it was made in, pivot map included.
            calls = aggregate_counts(root / "region/calls")["backend_attempts"]
            (root / "region/complete.json").unlink()
            with self.assertRaisesRegex(ValueError, "resume refused"):
                run_workflow(root / "view", root / "region", config(region=None), resume=True)
            with contextlib.redirect_stdout(io.StringIO()):
                run_workflow(root / "view", root / "region", config(region=1.0), resume=True)
            self.assertEqual(aggregate_counts(root / "region/calls")["backend_attempts"], calls)

    def test_a_surface_of_the_view_is_searched_inside_the_region(self):
        from profiles.config import ProfileConfig
        from profiles.workflow import run_workflow

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            boosted_view(root, derived=("d", "e"))
            config = ProfileConfig(dimension=2, parameters=["d", "e"], workers=2, profile_points=3,
                                   ranges={"d": [-1.0, 1.0], "e": [-1.0, 1.0]}, total_call_budget=3000)
            config.validate()
            with contextlib.redirect_stdout(io.StringIO()):
                run_workflow(root / "view", root / "surface", config)
            self.assertEqual(json.loads((root / "surface/manifest.json").read_text())["region"]["file"],
                             "region/region_it_2.npz")
            surface = json.loads((root / "surface/true_surface.json").read_text())
            d, e = np.meshgrid(surface["axes"][0], surface["axes"][1], indexing="ij")
            x1 = (0.5 * d - 0.3 * e) / 1.34
            exact = (d - 0.5 * x1) ** 2 + x1**2 + (e + 0.3 * x1) ** 2
            np.testing.assert_allclose(2 * (surface["best_logl"] - np.array(surface["logl"])), exact, atol=0.03)

    def test_nodes_of_the_view_outside_the_region_are_cut(self):
        from profiles.config import ProfileConfig
        from profiles.workflow import run_workflow

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, region, _ = boosted_view(root)
            # The optimum at d lies at x = d (1, 1/2, 0) / 1.25: inside the region at d = 1.8, not at 3.6.
            along = np.array([[1.0, 0.5, 0.0]]) / 1.25
            self.assertLess(region.rho(1.8 * along)[0], 0.9)
            self.assertGreater(region.rho(3.6 * along)[0], 1.2)
            config = ProfileConfig(parameters=["d"], workers=2, profile_points=5, ranges={"d": [-3.6, 3.6]},
                                   total_call_budget=3000)
            config.validate()
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                run_workflow(root / "view", root / "cut", config)
            self.assertIn("region: 2 node(s) of d have their optimum outside the region", printed.getvalue())
            ranges = json.loads((root / "cut/ranges.json").read_text())["d"]
            np.testing.assert_allclose(ranges["actual"], [-1.8, 1.8])
            np.testing.assert_allclose(ranges["region"]["outside"], [-3.6, 3.6])
            truth = json.loads((root / "cut/true_profiles.json").read_text())
            row = truth["profiles"]["d"]
            np.testing.assert_allclose(row["grid"], np.linspace(-1.8, 1.8, 5), atol=1e-12)
            np.testing.assert_allclose(2 * (truth["best_logl"] - np.array(row["logl"])), np.array(row["grid"]) ** 2 / 1.25,
                                       atol=0.03)

    def test_a_view_made_before_views_carried_the_region(self):
        from profiles import region as boosts

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            boosted_view(root)
            view = root / "view"
            made = json.loads((view / "view.json").read_text())
            del made["region"]
            (view / "view.json").write_text(json.dumps(made))
            # Its network is the bare one: the default would search another surrogate.
            with self.assertRaisesRegex(ValueError, "made before views carried the region"):
                boosts.resolve(view, 2, "run")
            self.assertIsNone(boosts.resolve(view, 2, None))

        with tempfile.TemporaryDirectory() as tmp:
            # A view of a network without a region has none, whatever the setting.
            root = Path(tmp)
            make_view(make_linear_run(root), root / "view", ["d"])
            self.assertIsNone(boosts.resolve(root / "view", 2, "run"))


class SurfaceReportingTests(unittest.TestCase):
    def test_one_penalised_node_does_not_ring_through_the_isocontours(self):
        # A 2D surface whose corner target is out of reach carries the view's penalty
        # there. On the Planck (H0, sigma8) surface that was 2.5e5 in Delta chi^2, and
        # an unclipped spline dipped to -1.2e4 across the grid.
        from profiles.surface_reporting import write_surface_outputs

        axes = [np.linspace(-3, 3, 7), np.linspace(-3, 3, 7)]
        u, v = np.meshgrid(*axes, indexing="ij")
        logl = -0.5 * (u**2 + v**2)
        logl[-1, -1] = -1.25e5
        surface = dict(axes=[a.tolist() for a in axes], names=["d", "e"], best_logl=0.0,
                       logl=logl.tolist(), quality={})
        diagnostics = [[dict(flag=0)] * 7 for _ in range(7)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "surrogate_surface.json").write_text(json.dumps(surface))
            (out / "true_surface.json").write_text(json.dumps(dict(
                surface, direct_logl=surface["logl"], diagnostics=diagnostics,
                anchor_consistent=True)))
            (out / "ranges.json").write_text(json.dumps(
                {n: dict(clipped=False) for n in ("d", "e")}))
            (out / "counts.json").write_text(json.dumps(dict(backend_attempts=0)))
            write_surface_outputs(out)
            report = json.loads((out / "surface_diagnostics.json").read_text())
        self.assertEqual(report["warnings"], [])
        self.assertGreater(min(report["spline_min_delta_chi2"]), -0.01)


if __name__ == "__main__":
    unittest.main()
