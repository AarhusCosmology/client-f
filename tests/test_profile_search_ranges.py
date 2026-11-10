"""The 2D setting `search_ranges`: a parameter that is not on the grid is searched over a narrower
range than the network's box, by the scan and by the true stage.

Analytic: a three-parameter Cobaya Gaussian mixture with two modes in x_2 (weights 0.8 at
x_2 = +3 and 0.2 at x_2 = -3), and a network that is the larger of the two modes' quadratics.
On the whole box every node's optimum is the high mode; with x_2 in [-5, 0] it is the low one,
and the surface is the true profile over that range. No CLASS or Planck resources.
"""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiles.config import ProfileConfig  # noqa: E402

MEANS = [[0.0, 0.0, 3.0], [0.0, 0.0, -3.0]]
WEIGHTS = [0.8, 0.2]
LOW = [-5.0, 0.0]
BOX = [[-4.0, 4.0], [-4.0, 4.0], [-6.0, 6.0]]


def make_run(root):
    """A saved three-parameter iteration (no boosts, so no region) with the two-mode truth."""
    import tensorflow as tf
    from likelihood.cobaya import CobayaLikelihood
    from likelihood.surrogate import SurrogateMetadata

    run = root / "run"
    for folder in ["likelihood_input", "training_data", "trained_models"]:
        (run / folder).mkdir(parents=True)
    info = dict(likelihood={"gaussian_mixture": dict(means=MEANS, covs=[np.eye(3).tolist()] * 2, weights=WEIGHTS,
                                                     input_params_prefix="x")},
                params={f"x_{i}": dict(prior=dict(min=lo, max=hi)) for i, (lo, hi) in enumerate(BOX)})
    path = run / "likelihood_input/toy.yaml"
    path.write_text(yaml.safe_dump(info))
    (run / "run.yaml").write_text(yaml.safe_dump(dict(likelihood=dict(wrapper="cobaya", input="toy.yaml"))))
    truth = CobayaLikelihood(str(path))
    SurrogateMetadata.from_likelihood(truth).save(run / "metadata.json")
    x = np.random.default_rng(3).uniform([-3, -3, -5], [3, 3, 5], (120, 3)).astype(np.float32)
    x[:4] = [[0.1, -0.1, 2.9], [-0.1, 0.1, 3.1], [0.1, 0.1, -2.9], [-0.1, -0.1, -3.1]]
    y = np.array([truth.loglkl(p) for p in x])
    peaks = [truth.loglkl(m) for m in MEANS]
    truth.cobaya_model.close()
    np.savetxt(run / "training_data/data_it_0.csv", np.column_stack([x, y]), delimiter=",",
               header="x_0,x_1,x_2,loglkl", comments="")
    # The network: the larger of the two modes' quadratics (exact away from the midpoint).
    inputs = tf.keras.Input((3,))
    branches = []
    for k, (mean, peak) in enumerate(zip(MEANS, peaks)):
        shifted = tf.keras.layers.Dense(3, name=f"shift_{k}")(inputs)
        square = tf.keras.layers.Dot(axes=1, name=f"square_{k}")([shifted, shifted])
        branches.append(tf.keras.layers.Dense(1, name=f"mode_{k}")(square))
    model = tf.keras.Model(inputs, tf.keras.layers.Maximum(name="larger")(branches))
    for k, (mean, peak) in enumerate(zip(MEANS, peaks)):
        model.get_layer(f"shift_{k}").set_weights([np.eye(3, dtype=np.float32), -np.array(mean, np.float32)])
        model.get_layer(f"mode_{k}").set_weights([np.array([[-0.5]], np.float32), np.array([peak], np.float32)])
    model.save(run / "trained_models/model_it_0.keras")
    return run, path


def true_profile(path, axes, low, high):
    """The truth maximised over x_2 in [low, high] at every node, on a fine grid and then polished."""
    from scipy.optimize import minimize_scalar
    from likelihood.cobaya import CobayaLikelihood

    truth = CobayaLikelihood(str(path))
    out = np.empty((len(axes[0]), len(axes[1])))
    for i, a in enumerate(axes[0]):
        for j, b in enumerate(axes[1]):
            z = np.linspace(low, high, 301)
            k = int(np.argmax([truth.loglkl([a, b, v]) for v in z]))
            lo, hi = z[max(k - 1, 0)], z[min(k + 1, len(z) - 1)]
            best = minimize_scalar(lambda v: -truth.loglkl([a, b, v]), bounds=(lo, hi), method="bounded",
                                   options=dict(xatol=1e-8))
            out[i, j] = -best.fun
    truth.cobaya_model.close()
    return out


def config(**overrides):
    c = ProfileConfig(dimension=2, parameters=["x_0", "x_1"], ranges={"x_0": [-1.0, 1.0], "x_1": [-1.0, 1.0]},
                      retrain=False, added_points=0, profile_points=3, workers=2, total_call_budget=3000,
                      hybrid_hops=8, **overrides)
    c.validate()
    return c


class Settings(unittest.TestCase):
    def test_values(self):
        ok = dict(dimension=2, parameters=["x", "y"], retrain=False, ranges={"x": [0, 1], "y": "prior"})
        self.assertIsNone(ProfileConfig(**ok).search_ranges)
        ProfileConfig(**ok, search_ranges={"z": [0.0, 1.0]}).validate()
        ProfileConfig(**ok, search_ranges={"z": [0, 1], "w": [-2, 3.5]}, surrogate_optimizer="legacy").validate()
        for bad in ({}, [["z", 0, 1]], {"z": [1.0, 0.0]}, {"z": [0.0]}, {"z": [0.0, float("inf")]},
                    {"z": [True, 1.0]}, {"z": "prior"}, {"z": (0.0, 1.0)}):
            with self.subTest(search_ranges=bad), self.assertRaisesRegex(ValueError, "search.range"):
                ProfileConfig(**ok, search_ranges=bad).validate()
        with self.assertRaisesRegex(ValueError, "not on the grid"):
            ProfileConfig(**ok, search_ranges={"x": [0.0, 0.5]}).validate()
        for other, message in ((dict(dimension=1, parameters=["x"], ranges={}), "2D study"),
                               (dict(retrain=True), "retrain: false"),
                               (dict(ranges={"x": [0, 1]}), "ranges for both")):
            with self.subTest(**{k: str(v) for k, v in other.items()}), self.assertRaisesRegex(ValueError, message):
                ProfileConfig(**dict(ok, **other), search_ranges={"z": [0.0, 1.0]}).validate()

    def test_in_every_manifest(self):
        # client-f fills no keys for studies made before them: a new study's manifest holds the key, None
        # without it, and a study whose manifest lacks it is not resumed.
        self.assertIn("search_ranges", ProfileConfig(dimension=2, parameters=["x", "y"]).as_dict())
        self.assertIsNone(ProfileConfig(dimension=2, parameters=["x", "y"]).as_dict()["search_ranges"])

    def test_yaml(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.yaml"
            path.write_text("dimension: 2\nparameters: [x, y]\nretrain: false\nranges:\n  x: [0, 1]\n  y: [0, 1]\n"
                            "search_ranges:\n  omega: [176.5, 185.5]\n")
            self.assertEqual(ProfileConfig.load(path).search_ranges, {"omega": [176.5, 185.5]})


class Workflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.run_dir, cls.likelihood = make_run(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def study(self, name, c, **kw):
        from profiles.workflow import run_workflow

        out = self.root / name
        run_workflow(self.run_dir, out, c, **kw)
        return out, json.loads((out / "true_surface.json").read_text())

    def test_the_surface_over_a_narrower_range(self):
        out, result = self.study("narrow", config(search_ranges={"x_2": LOW}))
        x = np.array(result["xopt"])
        self.assertTrue(np.all((x[..., 2] >= LOW[0]) & (x[..., 2] <= LOW[1])))
        np.testing.assert_allclose(x[..., 2], -3.0, atol=0.05)                 # the low mode
        expected = true_profile(self.likelihood, result["axes"], *LOW)
        np.testing.assert_allclose(result["logl"], expected, atol=1e-2)
        scan = json.loads((out / "surrogate_surface.json").read_text())
        self.assertTrue(np.all(np.array(scan["xopt"])[..., 2] <= LOW[1]))
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["configuration"]["search_ranges"], {"x_2": LOW})
        method = json.loads((out / "true_surface/method.json").read_text())
        self.assertEqual(method["bounds"][2], LOW)                              # in the true stage's signature
        # Resumed with the same setting it makes no call; with another (or none) it is refused.
        from profiles.evaluation import aggregate_counts

        before = aggregate_counts(out / "calls")["backend_attempts"]
        (out / "complete.json").unlink()
        self.study("narrow", config(search_ranges={"x_2": LOW}), resume=True)
        self.assertEqual(aggregate_counts(out / "calls")["backend_attempts"], before)
        for other in ({"x_2": [-5.0, 0.5]}, None):
            with self.subTest(search_ranges=other), self.assertRaisesRegex(ValueError, "resume refused"):
                self.study("narrow", config(search_ranges=other), resume=True)

    def test_the_older_scan_takes_the_narrower_range_too(self):
        out, result = self.study("narrow_legacy", config(search_ranges={"x_2": LOW}, surrogate_optimizer="legacy"))
        np.testing.assert_allclose(np.array(result["xopt"])[..., 2], -3.0, atol=0.05)
        np.testing.assert_allclose(result["logl"], true_profile(self.likelihood, result["axes"], *LOW), atol=1e-2)

    def test_without_the_key_the_whole_box(self):
        out, result = self.study("whole", config())
        x = np.array(result["xopt"])
        np.testing.assert_allclose(x[..., 2], 3.0, atol=0.05)                  # the high mode
        np.testing.assert_allclose(result["logl"], true_profile(self.likelihood, result["axes"], *BOX[2]), atol=1e-2)
        self.assertIsNone(json.loads((out / "manifest.json").read_text())["configuration"]["search_ranges"])

    def test_refusals_before_anything_is_computed(self):
        from profiles.workflow import preflight

        for ranges, message in (({"x_9": [0.0, 1.0]}, "unknown parameter"),
                                ({"x_2": [-7.0, 0.0]}, "outside saved metadata domain")):
            with self.subTest(search_ranges=ranges), self.assertRaisesRegex(ValueError, message):
                preflight(self.run_dir, config(search_ranges=ranges))
        with self.assertRaisesRegex(ValueError, "no training point"):
            self.study("empty", replace(config(), search_ranges={"x_2": [5.9, 6.0]}))


if __name__ == "__main__":
    unittest.main()
