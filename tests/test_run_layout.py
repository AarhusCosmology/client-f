"""Training sets written whole or as increments, read the same (run/layout.py), and the run index.

storage.training_sets: increments writes only the rows an iteration added; the set read back
must be row for row the whole file the cumulative layout writes, and a run of whole files must
read exactly as before.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import StorageConfig  # noqa: E402
from dataset.dataset import TrainingDataset  # noqa: E402
from run import layout  # noqa: E402
from run.run import latest_iteration  # noqa: E402
from run.storage import later_files, phase_outputs, set_later_aside  # noqa: E402


def likelihood(derived=("d1", "d2")):
    return SimpleNamespace(param_names=["a", "b", "c"], derived_names=list(derived))


def growing(seed=0, sizes=(20, 7, 0, 11), derived=True):
    """Successive (inputs, targets, derived) blocks, with NaN and -inf where runs have them."""
    rng = np.random.default_rng(seed)
    blocks = []
    for n in sizes:
        x = rng.normal(size=(n, 3))
        y = rng.normal(-50, 10, n)
        if n > 3:
            y[1], y[2] = np.nan, -np.inf
        d = rng.normal(size=(n, 2)) if derived else None
        if derived and n > 3:
            d[0, 1] = np.nan
        blocks.append((x, y, d))
    return blocks


def build(run, increments, blocks, lk=None):
    """Write sets 0..len(blocks)-1 as the loop does: one dataset that grows."""
    lk = lk or likelihood()
    x, y, d = blocks[0]
    dataset = TrainingDataset(x, y, lk, n_neighbors=3, target_temperature=1.0, derived=d)
    written = [dataset.save_set(run, 0, increments=increments)]
    for n, (x, y, d) in enumerate(blocks[1:], start=1):
        dataset.add_data(x, y, d)
        written.append(dataset.save_set(run, n, increments=increments))
    return dataset, written


def body(path):
    lines = Path(path).read_bytes().splitlines(keepends=True)
    return lines[0], b"".join(lines[1:])


class Increments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.whole, self.parts = self.root / "whole", self.root / "parts"
        self.blocks = growing()
        build(self.whole, False, self.blocks)
        _, self.written = build(self.parts, True, self.blocks)

    def tearDown(self):
        self.tmp.cleanup()

    def test_files_written(self):
        self.assertEqual([p.name for p in self.written[0]], ["data_it_0.csv", "derived_it_0.csv"])
        for n in (1, 2, 3):
            self.assertEqual([p.name for p in self.written[n]], [f"added_it_{n}.csv", f"derived_added_it_{n}.csv"])
        self.assertEqual(layout.set_iterations(self.parts), {0: "whole", 1: "added", 2: "added", 3: "added"})
        self.assertEqual(sorted(p.name for p in (self.whole / "training_data").iterdir()),
                         [f"data_it_{n}.csv" for n in range(4)])

    def test_increments_put_together_are_the_whole_file_byte_for_byte(self):
        for kind in (layout.TRAINING, layout.DERIVED):
            for n in range(4):
                files = layout.set_files(self.parts, n, kind)
                header, joined = body(files[0])
                for f in files[1:]:
                    h, b = body(f)
                    self.assertEqual(h, header)
                    joined += b
                self.assertEqual(header + joined, kind.whole_path(self.whole, n).read_bytes())

    def test_read_set_and_rows(self):
        for n in range(4):
            a = layout.read_set(self.parts, n)
            b = pd.read_csv(self.whole / "training_data" / f"data_it_{n}.csv")
            pd.testing.assert_frame_equal(a, b)
            self.assertEqual(layout.set_rows(self.parts, n), len(b))
            self.assertEqual(layout.set_rows(self.whole, n), len(b))
        self.assertEqual(layout.training_set(self.parts, 3, usecols=["loglkl"]).shape, (38, 1))

    def test_load_is_the_same_and_continues_with_an_increment(self):
        for n in range(4):
            a = TrainingDataset.load(self.parts / "training_data", likelihood(), 3, 1.0, iteration=n,
                                     derived_data_dir=self.parts / "derived_data")
            b = TrainingDataset.load(self.whole / "training_data", likelihood(), 3, 1.0, iteration=n,
                                     derived_data_dir=self.whole / "derived_data")
            np.testing.assert_array_equal(a.inputs, b.inputs)
            np.testing.assert_array_equal(a.targets, b.targets)
            np.testing.assert_array_equal(a.derived, b.derived)
        latest = TrainingDataset.load(self.parts / "training_data", likelihood(), 3, 1.0)
        self.assertEqual(latest.iteration, 3)
        x, y, d = growing(seed=5, sizes=(4,))[0]
        latest.add_data(x, y, d)
        self.assertEqual([p.name for p in latest.save_set(self.parts, 4, increments=True)],
                         ["added_it_4.csv", "derived_added_it_4.csv"])
        self.assertEqual(layout.set_rows(self.parts, 4), 42)
        self.assertEqual(latest_iteration(self.parts), 4)

    def test_changed_rows_give_a_whole_file(self):
        dataset, _ = build(self.root / "relabel", True, self.blocks[:2])
        dataset.targets[0, 0] += 1.0                       # an earlier row's value changed (a relabel)
        dataset.add_data(*self.blocks[2])
        names = [p.name for p in dataset.save_set(self.root / "relabel", 2, increments=True)]
        self.assertEqual(names, ["data_it_2.csv", "derived_added_it_2.csv"])   # derived rows are unchanged
        x, y, d = growing(seed=9, sizes=(5,))[0]
        dataset.add_data(x, y, d)
        dataset.save_set(self.root / "relabel", 3, increments=True)
        self.assertEqual([p.name for p in layout.set_files(self.root / "relabel", 3)], ["data_it_2.csv", "added_it_3.csv"])
        self.assertEqual(np.float32(layout.read_set(self.root / "relabel", 3)["loglkl"].iloc[0]), dataset.targets[0, 0])

    def test_a_set_not_read_here_is_never_extended(self):
        x, y, d = self.blocks[0]
        fresh = TrainingDataset(x, y, likelihood(), n_neighbors=3, target_temperature=1.0, derived=d)
        self.assertEqual(fresh.save_set(self.root / "other", 5, increments=True)[0].name, "data_it_5.csv")

    def test_missing_derived_set_gives_a_whole_derived_file(self):
        dataset, _ = build(self.root / "nod", True, self.blocks[:2])
        layout.DERIVED.added_path(self.root / "nod", 1).unlink()       # a run from before derived values
        dataset.add_data(*self.blocks[3])
        names = [p.name for p in dataset.save_set(self.root / "nod", 2, increments=True)]
        self.assertEqual(names, ["added_it_2.csv", "derived_it_2.csv"])

    def test_no_derived_parameters(self):
        blocks = growing(sizes=(6, 3), derived=False)
        _, written = build(self.root / "plain", True, blocks, lk=likelihood(derived=()))
        self.assertEqual([[p.name for p in w] for w in written], [["data_it_0.csv"], ["added_it_1.csv"]])

    def test_a_missing_link_is_refused(self):
        (self.parts / "training_data" / "added_it_2.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            layout.set_files(self.parts, 3)
        self.assertTrue(layout.has_set(self.parts, 1))
        self.assertFalse(layout.has_set(self.parts, 3))
        (self.parts / "training_data" / "data_it_0.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            layout.set_files(self.parts, 1)

    def test_recovery_and_rewind_know_the_increments(self):
        self.assertIn(self.parts / "training_data" / "added_it_3.csv", phase_outputs(self.parts, "acquisition_it_2"))
        self.assertIn(self.parts / "derived_data" / "derived_added_it_3.csv", phase_outputs(self.parts, "acquisition_it_2"))
        later = {str(p.relative_to(self.parts)) for p, _ in later_files(self.parts, 1)}
        self.assertEqual(later, {"training_data/added_it_2.csv", "training_data/added_it_3.csv",
                                 "derived_data/derived_added_it_2.csv", "derived_data/derived_added_it_3.csv"})
        aside = set_later_aside(self.parts, 1, log=lambda *_: None)
        self.assertEqual(layout.latest_set(self.parts), 1)
        self.assertTrue((aside / "training_data" / "added_it_3.csv").exists())
        self.assertEqual(layout.set_rows(self.parts, 1), 27)


class Index(unittest.TestCase):
    def test_index_of_a_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "20261125_120000_test"
            build(run, True, growing())
            (run / "trained_models").mkdir()
            for n in range(3):
                (run / "trained_models" / f"model_it_{n}.keras").write_bytes(b"x" * 10)
            (run / "interrupted" / "x").mkdir(parents=True)
            (run / "interrupted" / "x" / "data_it_9.csv").write_text("a\n")
            index = layout.write_index(run)
            again = json.loads((run / "run.json").read_text())
            self.assertEqual(index["schema"], 1)
            self.assertEqual(again["iterations"], index["iterations"])
            self.assertEqual([e["iteration"] for e in index["iterations"]], [0, 1, 2, 3])
            self.assertEqual([e["training_set"]["rows"] for e in index["iterations"]], [20, 27, 27, 38])
            self.assertEqual([e["training_set"]["stored"] for e in index["iterations"]], ["whole"] + ["added"] * 3)
            self.assertEqual(index["iterations"][3]["training_set"]["files"],
                             ["training_data/data_it_0.csv", "training_data/added_it_1.csv",
                              "training_data/added_it_2.csv", "training_data/added_it_3.csv"])
            self.assertIn("network", index["iterations"][2])
            self.assertNotIn("network", index["iterations"][3])
            self.assertNotIn("interrupted", index["folders"])
            self.assertEqual(index["folders"]["training_data"]["files"], 4)
            # an unchanged file's rows come from the earlier index, a changed one is counted again
            with (run / "training_data" / "added_it_3.csv").open("a") as stream:
                stream.write("0,0,0,0\n")
            self.assertEqual(layout.write_index(run)["iterations"][3]["training_set"]["rows"], 39)


class Settings(unittest.TestCase):
    def test_storage_section(self):
        self.assertEqual(StorageConfig.from_dict(None).training_sets, "increments")      # the default
        self.assertEqual(StorageConfig.from_dict({}).training_sets, "increments")
        self.assertEqual(StorageConfig.from_dict({"training_sets": "cumulative"}).training_sets, "cumulative")
        with self.assertRaises(ValueError):
            StorageConfig.from_dict({"training_sets": "deltas"})
        with self.assertRaises(ValueError):
            StorageConfig.from_dict({"training_set": "increments"})


if __name__ == "__main__":
    unittest.main()
