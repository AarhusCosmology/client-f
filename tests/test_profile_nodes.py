"""profiles/nodes.py: the finished nodes of a true stage, in one journal or one file per node."""
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiles.io import save  # noqa: E402
from profiles.nodes import JOURNAL, NodeStore, has_nodes, read_nodes  # noqa: E402


def record(k):
    return dict(logl=np.float64(-10.0 - k), x=np.arange(3.0) + k, diagnostics=dict(reason="converged", calls=np.int64(7)),
                state=dict(path=[(0.0, [1.0, 2.0])]), direct_logl=None)


class NodeJournal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "true"

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_new_stage_writes_one_journal_and_resumes_from_it(self):
        store = NodeStore(self.dir)
        calls = []
        first = store.checkpoint("node_000_00001", lambda: calls.append(1) or record(1))
        store.checkpoint("anchor", lambda: record(9))
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), [JOURNAL])
        # what a node file read back gave: JSON types, tuples as lists
        save(Path(self.tmp.name) / "file.json", record(1))
        self.assertEqual(first, json.loads((Path(self.tmp.name) / "file.json").read_text()))
        again = NodeStore(self.dir)
        self.assertEqual(again.checkpoint("node_000_00001", lambda: self.fail("computed again")), first)
        self.assertEqual(calls, [1])
        self.assertEqual(sorted(read_nodes(self.dir)), ["anchor", "node_000_00001"])
        self.assertTrue(has_nodes(self.dir))

    def test_a_stage_with_node_files_goes_on_writing_files(self):
        self.dir.mkdir()
        save(self.dir / "node_000_00000.json", record(0))
        store = NodeStore(self.dir)
        self.assertTrue(store.files)
        store.checkpoint("node_000_00001", lambda: record(1))
        self.assertFalse((self.dir / JOURNAL).exists())
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["node_000_00000.json", "node_000_00001.json"])
        self.assertEqual(store.get("node_000_00000")["x"], [0.0, 1.0, 2.0])

    def test_a_cut_last_line_loses_that_node_only(self):
        store = NodeStore(self.dir)
        store.checkpoint("node_000_00000", lambda: record(0))
        store.checkpoint("node_000_00001", lambda: record(1))
        raw = (self.dir / JOURNAL).read_bytes()
        (self.dir / JOURNAL).write_bytes(raw[:-25])                        # killed inside the second write
        self.assertEqual(sorted(read_nodes(self.dir)), ["node_000_00000"])  # read only: the cut line is skipped
        store = NodeStore(self.dir)
        self.assertEqual(len(store), 1)
        aside = list(self.dir.glob("nodes.truncated-*.json"))
        self.assertEqual(len(aside), 1)
        store.checkpoint("node_000_00001", lambda: record(1))
        self.assertEqual(sorted(NodeStore(self.dir).records), ["node_000_00000", "node_000_00001"])

    def test_a_damaged_line_inside_is_refused(self):
        store = NodeStore(self.dir)
        for k in range(3):
            store.checkpoint(f"node_000_{k:05d}", lambda k=k: record(k))
        lines = (self.dir / JOURNAL).read_bytes().splitlines(keepends=True)
        (self.dir / JOURNAL).write_bytes(lines[0] + b"{broken\n" + lines[2])
        with self.assertRaises(ValueError):
            NodeStore(self.dir)

    def test_nan_is_refused_as_before(self):
        with self.assertRaises(ValueError):
            NodeStore(self.dir).checkpoint("node_000_00000", lambda: dict(logl=float("nan")))
        self.assertEqual(read_nodes(self.dir), {})

    def test_threads(self):
        store = NodeStore(self.dir)
        threads = [threading.Thread(target=lambda k=k: store.checkpoint(f"node_{k % 4:03d}_{k:05d}", lambda: record(k)))
                   for k in range(64)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        lines = (self.dir / JOURNAL).read_text().splitlines()
        self.assertEqual(len(lines), 64)
        self.assertEqual(len(NodeStore(self.dir)), 64)


if __name__ == "__main__":
    unittest.main()
