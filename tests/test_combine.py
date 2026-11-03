"""client.py combine (run/combine.py): runs of one objective on adjacent boxes in one parameter
as one emulator, blended in a band, each network inside its own region."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tensorflow as tf  # noqa: E402

from likelihood.surrogate import ParameterInfo, SurrogateMetadata  # noqa: E402
from model.network import SmoothStep, build_model, fit_background, load_model, restrict_model  # noqa: E402
from run.combine import Part, combine, rebox  # noqa: E402
from run import combine as combining  # noqa: E402
from run.storage import digest  # noqa: E402
from sampling.region import Region  # noqa: E402
from sampling.targets import load_snapshot  # noqa: E402

NAMES = ["a", "w", "c"]


def identity(boxes, files=None):
    return dict(schema=2, wrapper="cobaya", convention=1, objective="native backend return; priors follow backend convention",
                parameters=[dict(name=n, scale=1.0, lower=boxes[n][0], upper=boxes[n][1], center=None, sigma=None) for n in NAMES],
                files=files or {"input": "abc"})


def make_run(root, name, w_box, seed, iteration=3, refused=None, files=None, c_box=(-2.0, 2.0)):
    """A finished run of a small analytic network on the box a in [-2, 2], w in w_box, c in c_box."""
    run = Path(root) / name
    for sub in ("trained_models", "training_data", "boost", "convergence_stats", "likelihood_input"):
        (run / sub).mkdir(parents=True)
    boxes = dict(a=(-2.0, 2.0), w=tuple(w_box), c=tuple(c_box))
    SurrogateMetadata(parameters=tuple(ParameterInfo(name=n, label=n, scale=1.0, lower=boxes[n][0], upper=boxes[n][1],
                                                     center=None, sigma=None) for n in NAMES)).save(run / "metadata.json")
    rng = np.random.default_rng(seed)
    x = rng.uniform([boxes[n][0] for n in NAMES], [boxes[n][1] for n in NAMES], (400, 3))
    y = -0.5 * ((x[:, 0] - 0.3) ** 2 + (x[:, 2] + 0.1) ** 2) * 4 - 0.1 * x[:, 1]
    tf.random.set_seed(seed)
    model = build_model(x, y, 1, 8, "alsing", background=fit_background(x, y, depth=10.0))
    model.save(run / "trained_models" / f"model_it_{iteration}.keras")
    data = pd.DataFrame(x, columns=NAMES).assign(loglkl=y)
    if refused is not None:
        data.loc[:4, "loglkl"] = refused
        (run / "refused_value.json").write_text(json.dumps(dict(value=refused)))
    data.loc[5, "loglkl"] = -np.inf
    data.to_csv(run / "training_data" / f"data_it_{iteration}.csv", index=False)
    lower, upper = np.array([boxes[n][0] for n in NAMES]), np.array([boxes[n][1] for n in NAMES])
    cloud = x[y > np.percentile(y, 80)]
    Region.from_clouds([cloud], lower, upper, 1.0, float(y.max())).save(run / "boost" / f"region_it_{iteration}.npz")
    (run / "client.yaml").write_text(f"likelihood:\n  wrapper: cobaya\n  input: runs/{name}/cobaya.yaml\nsampling:\n  temperature: 7.0\n")
    (run / "likelihood_input" / "cobaya.yaml").write_text("likelihood: {}\n")
    (run / "objective_identity.json").write_text(json.dumps(dict(identity=identity(boxes, files))))
    np.savez(run / "convergence_stats" / f"chain_summary_it_{iteration}.npz", mean=x.mean(0), cov=np.cov(x.T))
    return run


def values(model, x):
    return model(tf.constant(np.asarray(x, np.float32)), training=False).numpy().ravel()


class Combine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.low = make_run(self.root, "low", (1.0, 3.0), seed=1)
        self.full = make_run(self.root, "full", (1.0, 10.0), seed=2, iteration=5, refused=-777.0)

    def tearDown(self):
        self.tmp.cleanup()

    def combined(self, **kw):
        out = self.root / "combined"
        combine(out, [Part(self.low), Part(self.full)], "w", [(2.4, 2.8)], log=lambda *a: None, **kw)
        return out

    def test_each_network_in_its_range_and_the_blend_in_the_band(self):
        out = self.combined()
        c, lo, hi = (load_model(p) for p in (out / "trained_models/model_it_0.keras",
                                             self.low / "trained_models/model_it_3.keras",
                                             self.full / "trained_models/model_it_5.keras"))
        r_lo, r_hi = Region.load(self.low / "boost/region_it_3.npz"), Region.load(self.full / "boost/region_it_5.npz")
        rng = np.random.default_rng(9)
        x = rng.uniform([-2, 1, -2], [2, 10, 2], (4000, 3)).astype(np.float32)
        fc, fl, fh = values(c, x), values(lo, x), values(hi, x)
        w = x[:, 1]
        below = (w <= 2.4) & (r_lo.rho(x) <= r_lo.inflate)
        above = (w >= 2.8) & (r_hi.rho(x) <= r_hi.inflate)
        self.assertGreater(below.sum(), 20)
        np.testing.assert_array_equal(fc[below], fl[below])          # exactly the lower network
        np.testing.assert_array_equal(fc[above], fh[above])          # exactly the upper network
        band = (w > 2.4) & (w < 2.8) & (r_lo.rho(x) <= r_lo.inflate) & (r_hi.rho(x) <= r_hi.inflate)
        t = (w[band] - 2.4) / 0.4
        s = t ** 3 * (t * (6 * t - 15) + 10)
        np.testing.assert_allclose(fc[band], (1 - s) * fl[band] + s * fh[band], rtol=1e-5, atol=1e-4)
        outside = (w <= 2.4) & (r_lo.rho(x) > r_lo.inflate * 1.1)    # the lower network's own wall
        self.assertTrue(np.all(fc[outside] < r_lo.floor))
        record = json.loads((out / "combine.json").read_text())
        self.assertEqual((record["parameter"], record["bands"], record["walls"]), ("w", [[2.4, 2.8]], True))
        self.assertEqual(record["box"], [1.0, 10.0])
        bare = (w <= 2.4) & (r_lo.rho(x) > r_lo.inflate)                # without walls: the network there too
        self.assertGreater(bare.sum(), 20)
        np.testing.assert_array_equal(values(load_model(self.combined_without_walls()), x)[bare], fl[bare])

    def combined_without_walls(self):
        out = self.root / "nowalls"
        combine(out, [Part(self.low), Part(self.full)], "w", [(2.4, 2.8)], walls=False, log=lambda *a: None)
        return out / "trained_models/model_it_0.keras"

    def test_files_metadata_region_data_and_consumers(self):
        out = self.combined()
        md = SurrogateMetadata.load(out / "metadata.json")
        self.assertEqual(md.bounds["w"], (1.0, 10.0))
        self.assertEqual(md.bounds["a"], (-2.0, 2.0))
        union = Region.load(out / "boost/region_it_0.npz")
        parts = [Region.load(self.low / "boost/region_it_3.npz"), Region.load(self.full / "boost/region_it_5.npz")]
        self.assertEqual(len(union), 2)
        x = np.random.default_rng(3).uniform([-2, 1, -2], [2, 10, 2], (2000, 3))
        np.testing.assert_allclose(union.rho(x), np.minimum(*[r.rho(x) for r in parts]), atol=1e-12)
        data = pd.read_csv(out / "training_data/data_it_0.csv")
        self.assertEqual(list(data.columns), NAMES + ["loglkl"])
        self.assertEqual(len(data), 400 - 1 + 400 - 6)               # the -inf rows and the refused ones left out
        self.assertTrue(np.isfinite(data["loglkl"]).all() and not (data["loglkl"] == -777.0).any())
        self.assertEqual(json.loads((out / "refused_value.json").read_text())["value"], -777.0)   # the widest part's
        identity = json.loads((out / "objective_identity.json").read_text())["identity"]
        self.assertEqual([p for p in identity["parameters"] if p["name"] == "w"][0]["lower"], 1.0)
        self.assertTrue((out / "client.yaml").read_text().startswith("# A combined emulator"))
        self.assertTrue((out / "convergence_stats/chain_summary_it_0.npz").exists())
        target, ident = load_snapshot(out, 0)                        # the evidence's and the report's way in
        self.assertIsNotNone(target.region)
        restricted = restrict_model(load_model(out / "trained_models/model_it_0.keras"), union)   # the profiles' way in
        inside = union.rho(x) <= union.inflate
        np.testing.assert_allclose(restricted(tf.constant(x), training=False).numpy().ravel()[inside],
                                   values(load_model(out / "trained_models/model_it_0.keras"), x)[inside], atol=2e-3)

    def test_what_is_refused(self):
        def attempt(parts, bands, name="x", parameter="w"):
            with self.assertRaises(ValueError) as caught:
                combine(self.root / name, parts, parameter, bands, log=lambda *a: None)
            return str(caught.exception)
        low, full = Part(self.low), Part(self.full)
        self.assertIn("inside both boxes", attempt([low, full], [(2.6, 3.2)]))         # beyond the lower part's box
        self.assertIn("increasing", attempt([full, low], [(2.4, 2.8)], "y"))
        self.assertIn("band(s)", attempt([low, full], []))
        self.assertIn("not a parameter", attempt([low, full], [(2.4, 2.8)], "z", parameter="q"))
        other = make_run(self.root, "other", (1.0, 10.0), seed=3, files={"input": "different"})
        self.assertIn("another objective", attempt([low, Part(other)], [(2.4, 2.8)], "v"))
        wider_c = make_run(self.root, "widerc", (1.0, 10.0), seed=4, c_box=(-3.0, 2.0))
        self.assertIn("box differs in c", attempt([low, Part(wider_c)], [(2.4, 2.8)], "u"))
        (self.root / "taken").mkdir()
        (self.root / "taken" / "something").write_text("")
        self.assertIn("not empty", attempt([low, full], [(2.4, 2.8)], "taken"))

    def test_the_loop_does_not_continue_a_combined_emulator(self):
        out = self.combined()
        run = subprocess.run([sys.executable, str(ROOT / "client.py"), str(out)], capture_output=True, text=True, cwd=ROOT)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("combined emulator", run.stderr + run.stdout)

    def test_mixed_region_support_is_rejected_without_publishing(self):
        (self.full / "boost/region_it_5.npz").unlink()
        for walls in (True, False):
            with self.assertRaisesRegex(ValueError, "with and without regions"):
                self.combined(walls=walls)
            self.assertFalse((self.root / "combined").exists())
        (self.low / "boost/region_it_3.npz").unlink()
        out = self.combined()
        target, _ = load_snapshot(out, 0)
        self.assertIsNone(target.region)
        self.assertFalse((out / "boost/region_it_0.npz").exists())

    def test_changes_during_snapshot_are_rejected(self):
        paths = ["trained_models/model_it_3.keras", "boost/region_it_3.npz",
                 "metadata.json", "training_data/data_it_3.csv",
                 "convergence_stats/chain_summary_it_3.npz"]
        original_copy = combining.shutil.copy2
        for name in paths:
            with self.subTest(artifact=name):
                source = self.low / name
                original = source.read_bytes()

                def mutate(src, dst):
                    result = original_copy(src, dst)
                    if Path(src) == source:
                        source.write_bytes(original + b"\n")
                    return result

                try:
                    with mock.patch.object(combining.shutil, "copy2", side_effect=mutate):
                        with self.assertRaisesRegex(ValueError, "changed while taking"):
                            self.combined()
                    self.assertFalse((self.root / "combined").exists())
                finally:
                    source.write_bytes(original)

    def test_snapshot_selection_detects_a_new_region(self):
        source = self.low / "boost/region_it_3.npz"
        source.rename(self.low / "boost/region_it_2.npz")
        original_copy = combining.shutil.copy2

        def mutate(src, dst):
            result = original_copy(src, dst)
            if Path(src) == self.low / "boost/region_it_2.npz":
                original_copy(src, source)
            return result

        with mock.patch.object(combining.shutil, "copy2", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed while taking"):
                self.combined()
        self.assertFalse((self.root / "combined").exists())

    def test_later_source_changes_cannot_change_output_or_recorded_bytes(self):
        source = self.low / "trained_models/model_it_3.keras"
        original_hash = digest(source)
        x = [[0.3, 1.5, -0.1]]
        expected = values(load_model(source), x)
        build = combining.combined_model

        def replace_source(parts, *args, **kwargs):
            result = build(parts, *args, **kwargs)
            self.assertNotEqual(parts[0].model_path, source)
            # Replace the original after its immutable copy has been consumed.
            source.write_bytes(b"replaced after snapshot")
            return result

        nested = self.full / "likelihood_input/subdirectory/helper.py"
        nested.parent.mkdir()
        nested.write_text("VALUE = 1\n")
        with mock.patch.object(combining, "combined_model", side_effect=replace_source):
            out = self.combined(walls=False)
        record = json.loads((out / "combine.json").read_text())
        self.assertEqual(record["parts"][0]["model_sha256"], original_hash)
        self.assertEqual(record["parts"][0]["artifacts"]["trained_models/model_it_3.keras"], original_hash)
        self.assertEqual(record["parts"][0]["run"], str(self.low))
        np.testing.assert_array_equal(values(load_model(out / "trained_models/model_it_0.keras"), x), expected)
        self.assertEqual((out / "likelihood_input/subdirectory/helper.py").read_text(), "VALUE = 1\n")

    def test_failure_never_leaves_a_partially_published_run(self):
        out = self.root / "combined"
        for existing in (False, True):
            if existing:
                out.mkdir()
            with mock.patch.object(combining, "combined_model", side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    self.combined()
            self.assertEqual(out.exists(), existing)
            if existing:
                self.assertEqual(list(out.iterdir()), [])

    def test_concurrent_output_is_not_overwritten(self):
        out = self.root / "combined"
        build = combining.combined_model

        def concurrent_writer(*args, **kwargs):
            result = build(*args, **kwargs)
            out.mkdir()
            (out / "user_file").write_text("keep")
            return result

        with mock.patch.object(combining, "combined_model", side_effect=concurrent_writer):
            with self.assertRaises(OSError):
                self.combined()
        self.assertEqual([p.name for p in out.iterdir()], ["user_file"])
        self.assertEqual((out / "user_file").read_text(), "keep")

    def test_identical_reboxed_cells_have_one_sampleable_union_cell(self):
        # Same physical ellipsoid, represented in two different training boxes.
        low = Region([[.5, .5, .5]], [np.diag([10., 10., 10.])], [1.],
                     [-2., 1., -2.], [2., 3., 2.], 1.5, -1000., rows=[100])
        means, whiten = rebox(low, [-2., 1., -2.], [2., 10., 2.])
        high = Region(means, whiten, [1.], [-2., 1., -2.], [2., 10., 2.],
                      1.5, -1000., rows=[200])
        low.save(self.low / "boost/region_it_3.npz")
        high.save(self.full / "boost/region_it_5.npz")
        out = self.combined()
        target, _ = load_snapshot(out, 0)
        union = target.region
        self.assertEqual(len(union), 1)
        self.assertEqual(union.rows, [300])
        samples = union.start(64, np.random.default_rng(5), component=0)
        np.testing.assert_array_equal(union.labels(samples), np.zeros(64, int))
        np.testing.assert_allclose(union.rho(samples), low.rho(samples), atol=1e-12)
        record = json.loads((out / "combine.json").read_text())
        self.assertEqual(record["region_cell_sources"], [[dict(part=0, cell=0), dict(part=1, cell=0)]])


class Blend(unittest.TestCase):
    def test_smoothstep_ends_sum_and_smoothness(self):
        x = np.zeros((7, 2), np.float64)
        x[:, 1] = [0.0, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0]
        rising = SmoothStep(1, 1.0, 2.0, dtype="float64")(tf.constant(x)).numpy().ravel()
        falling = SmoothStep(1, 1.0, 2.0, falling=True, dtype="float64")(tf.constant(x)).numpy().ravel()
        np.testing.assert_array_equal(rising[[0, 1, 5, 6]], [0, 0, 1, 1])
        np.testing.assert_array_equal(falling[[0, 1, 5, 6]], [1, 1, 0, 0])
        np.testing.assert_allclose(rising + falling, 1.0, atol=1e-15)
        self.assertAlmostEqual(rising[3], 0.5)
        layer = SmoothStep(1, 1.0, 2.0, dtype="float64")
        for end in (1.0, 2.0):                                       # first and second derivatives vanish at the ends
            z = tf.Variable([[0.0, end]], dtype=tf.float64)
            with tf.GradientTape() as outer:
                with tf.GradientTape() as inner:
                    s = layer(z)
                first = inner.gradient(s, z)[0, 1]
            second = outer.gradient(first, z)[0, 1]
            self.assertAlmostEqual(float(first), 0.0)
            self.assertAlmostEqual(float(second), 0.0)
        with self.assertRaises(ValueError):
            SmoothStep(0, 1.0, 1.0)

    def test_rebox_keeps_distances(self):
        rng = np.random.default_rng(5)
        cloud = rng.normal([0.5, 2.0, -0.2], [0.3, 0.4, 0.2], (200, 3))
        r = Region.from_clouds([cloud], [-2, 1, -2], [2, 3, 2], 1.5, 0.0)
        means, whiten = rebox(r, [-2, 0.5, -2], [2, 12, 2])
        moved = Region(means, whiten, r.reach, [-2, 0.5, -2], [2, 12, 2], r.inflate, r.floor)
        x = rng.uniform([-2, 1, -2], [2, 3, 2], (500, 3))
        np.testing.assert_allclose(moved.rho(x), r.rho(x), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
