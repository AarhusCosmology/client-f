"""client.py export (run/export.py): a run's network as a TensorFlow SavedModel that loads
without CLiENT, inside its region by default."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from unittest import mock  # noqa: E402

import tensorflow as tf  # noqa: E402

from model.network import load_model, restrict_model  # noqa: E402
from run.combine import Part, combine  # noqa: E402
from run.export import export  # noqa: E402
from sampling.region import Region  # noqa: E402
from test_combine import make_run  # noqa: E402

READ = """
import json, sys
import numpy as np
import tensorflow as tf
served = tf.saved_model.load(sys.argv[1])        # kept: a loaded object dropped mid-call frees its variables
print(json.dumps(served.serve(tf.constant(np.load(sys.argv[2]))).numpy().ravel().tolist()))
"""


def without_client(saved, x, workdir):
    """The export's values at x, from a Python process that cannot import CLiENT (-I, another
    working directory)."""
    np.save(Path(workdir) / "x.npy", x)
    done = subprocess.run([sys.executable, "-I", "-c", READ, str(saved), str(Path(workdir) / "x.npy")],
                          capture_output=True, text=True, cwd=workdir, check=True)
    return np.asarray(json.loads(done.stdout.strip().splitlines()[-1]))


class Export(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run = make_run(self.root, "full", (1.0, 10.0), seed=2, iteration=5, refused=-777.0)
        self.x = np.random.default_rng(4).uniform([-2, 1, -2], [2, 10, 2], (300, 3))

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_network_inside_its_region_without_client(self):
        out = self.root / "exported"
        record = export(self.run, out, log=lambda *a: None)
        region = Region.load(self.run / "boost/region_it_5.npz")
        source = restrict_model(load_model(self.run / "trained_models/model_it_5.keras"), region, dtype="float32")
        expected = source(tf.constant(self.x.astype(np.float32)), training=False).numpy().ravel()
        got = without_client(out / "saved_model", self.x.astype(np.float32), self.root)
        np.testing.assert_allclose(got, expected, rtol=2e-5, atol=1e-3)
        self.assertTrue(np.any(region.rho(self.x) > region.inflate))      # some points meet the wall
        self.assertEqual([p["name"] for p in record["parameters"]], ["a", "w", "c"])
        self.assertEqual(record["parameters"][1]["upper"], 10.0)
        self.assertEqual(record["region"]["file"], "boost/region_it_5.npz")
        self.assertEqual(record["source"]["iteration"], 5)
        self.assertEqual(record["dtype"], "float32")
        self.assertLess(record["check"]["largest_difference_near_the_peak"], 1e-3)
        data = np.loadtxt(self.run / "training_data/data_it_5.csv", delimiter=",", skiprows=1)
        self.assertAlmostEqual(record["reference"], float(np.max(data[np.isfinite(data[:, -1]), -1])), places=12)
        self.assertEqual(json.loads((out / "emulator.json").read_text())["source"]["run"], str(self.run))
        self.assertIn("tf.saved_model.load", (out / "README.md").read_text())

    def test_bare_network_and_double_precision(self):
        out = self.root / "bare64"
        record = export(self.run, out, region=False, dtype="float64", log=lambda *a: None)
        self.assertIsNone(record["region"])
        bare = load_model(self.run / "trained_models/model_it_5.keras")
        expected = bare(tf.constant(self.x.astype(np.float32)), training=False).numpy().ravel()
        got = without_client(out / "saved_model", self.x, self.root)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-4)   # the float32 network, evaluated at float64

    def test_a_combined_emulator_and_what_is_refused(self):
        low = make_run(self.root, "low", (1.0, 3.0), seed=1)
        combined = self.root / "combined"
        combine(combined, [Part(low), Part(self.run)], "w", [(2.4, 2.8)], log=lambda *a: None)
        out = self.root / "exported_combined"
        record = export(combined, out, log=lambda *a: None)
        self.assertEqual(record["source"]["combined"]["bands"], [[2.4, 2.8]])
        source = restrict_model(load_model(combined / "trained_models/model_it_0.keras"),
                                Region.load(combined / "boost/region_it_0.npz"), dtype="float32")
        np.testing.assert_allclose(without_client(out / "saved_model", self.x.astype(np.float32), self.root),
                                   source(tf.constant(self.x.astype(np.float32)), training=False).numpy().ravel(),
                                   rtol=2e-5, atol=1e-3)
        with self.assertRaisesRegex(ValueError, "not empty"):
            export(self.run, out, log=lambda *a: None)
        with self.assertRaisesRegex(ValueError, "no network 7"):
            export(self.run, self.root / "other", iteration=7, log=lambda *a: None)
        with self.assertRaisesRegex(ValueError, "dtype"):
            export(self.run, self.root / "other2", dtype="float16", log=lambda *a: None)

    def assertNothingWritten(self, out):
        self.assertFalse(out.exists())
        self.assertEqual([p.name for p in self.root.glob(f".{out.name}*")], [])

    def test_a_refused_export_and_a_run_without_its_training_set_leave_nothing(self):
        import run.export as module

        out = self.root / "refused"
        with mock.patch.dict(module.TOLERANCE, float32=(-1.0, 0.0)), self.assertRaisesRegex(RuntimeError, "refused, and nothing is written"):
            export(self.run, out, log=lambda *a: None)
        self.assertNothingWritten(out)
        (self.run / "training_data" / "data_it_5.csv").rename(self.root / "aside.csv")
        with self.assertRaises(FileNotFoundError):
            export(self.run, out, log=lambda *a: None)
        self.assertNothingWritten(out)
        (self.root / "aside.csv").rename(self.run / "training_data" / "data_it_5.csv")
        self.assertTrue((Path(export(self.run, out, log=lambda *a: None)["source"]["run"])).exists())
        self.assertTrue((out / "emulator.json").exists())

    def test_the_command_with_relative_paths(self):
        # An ordinary command line. The reload runs in the staging directory, and every test
        # until this one used absolute paths.
        done = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "client.py"), "export",
                               os.path.relpath(self.run, self.root), "exported"],
                              capture_output=True, text=True, cwd=self.root)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        record = json.loads((self.root / "exported" / "emulator.json").read_text())
        self.assertTrue(Path(record["source"]["run"]).is_absolute())
        self.assertTrue((self.root / "exported" / "saved_model").exists())

    def test_the_recorded_hashes_are_of_what_was_exported(self):
        from run.storage import digest

        out = self.root / "hashes"
        expected = digest(self.run / "trained_models" / "model_it_5.keras")
        record = export(self.run, out, log=lambda *a: None)
        self.assertEqual(record["source"]["model_sha256"], expected)
        self.assertEqual(record["region"]["sha256"], digest(self.run / "boost" / "region_it_5.npz"))
        self.assertIn("cells", record["region"])


if __name__ == "__main__":
    unittest.main()
