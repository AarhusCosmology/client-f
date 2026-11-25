"""The finished nodes of a true stage, for resume: one journal per stage, or one file per node.

The true stages (profiles/residual_newton.py `true/`, profiles/hybrid_surface.py
`true_surface/`) keep every finished node so that a study stopped at its time limit goes on
where it was. They wrote one JSON file per node (`node_KKK_NNNNN.json`, `node_IIII_JJJJ.json`,
`anchor.json`): 2,500 files for a 50 x 50 surface, 34,693 of the 35,184 files of the studies of
October and November 2026, each a few kB, and the final true_profiles.json / true_surface.json
repeats all of them but the Newton state. A stage begun now writes one journal instead,
`nodes.jsonl`: a line per finished node, {"event": "node", "node": NAME, "record": RECORD}, NAME
the file name it had (without .json), written whole and fsync'd before the node counts as done.

A stage that already holds node files goes on writing node files, so a study is never mixed
and old studies continue as they were; with no node file and no journal a stage begins a
journal. Either way a node's record is what reading its file back gave (JSON, NaN refused), so
a resumed study sees exactly what it saw before.

A journal whose last line was cut (a job killed inside the write) loses that node only: the
fragment is kept beside it (`nodes.truncated-<id>.json`) and the line is removed before the next
write, so that it never ends up inside the journal. A damaged line before the last is refused.
"""
import json
import os
import threading
import uuid
from pathlib import Path

JOURNAL = "nodes.jsonl"


def _normalised(value):
    """The record as reading its JSON back gives it (profiles/io.py save)."""
    import numpy as np

    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)

    return json.loads(json.dumps(value, default=convert, allow_nan=False))


def _read_journal(path):
    """({name: record}, offset of a cut last line or None). First record of a name wins."""
    raw = Path(path).read_bytes()
    found, offset = {}, 0
    lines = raw.splitlines(keepends=True)
    for k, line in enumerate(lines):
        try:
            item = json.loads(line)
            if not isinstance(item, dict) or item.get("event") != "node" or not isinstance(item.get("node"), str):
                raise ValueError("not a node record")
        except (ValueError, UnicodeDecodeError) as error:
            if k == len(lines) - 1 and not line.endswith(b"\n"):
                return found, offset
            raise ValueError(f"{path}: damaged node record on line {k + 1}") from error
        found.setdefault(item["node"], item["record"])
        offset += len(line)
    return found, None


def read_nodes(directory):
    """{name: record} of the finished nodes of a stage, from its node files and its journal
    (a node file wins); read only, so it can be called while the stage runs."""
    directory = Path(directory)
    found = {}
    journal = directory / JOURNAL
    if journal.exists():
        found.update(_read_journal(journal)[0])
    for path in sorted(directory.glob("*.json")):
        if path.stem.startswith("node_") or path.name == "anchor.json":
            try:
                found[path.stem] = json.loads(path.read_text())
            except ValueError:            # being written by a stage of the old kind
                continue
    return found


def has_nodes(directory):
    directory = Path(directory)
    return ((directory / JOURNAL).exists() or (directory / "anchor.json").exists()
            or any(directory.glob("node_*.json")))


class NodeStore:
    """`checkpoint(name, compute)`: the record of node `name`, computed and kept the first time.
    Thread safe; the computation runs outside the lock."""

    def __init__(self, directory, files=None):
        from profiles.io import save

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._save = save
        self.lock = threading.Lock()
        # Node files where the stage has them already; a journal otherwise (files=True keeps files).
        legacy = any(self.directory.glob("node_*.json")) or (self.directory / "anchor.json").exists()
        self.files = legacy if files is None else bool(files)
        self.journal = self.directory / JOURNAL
        self.records = {}
        if self.journal.exists():
            self.records, cut = _read_journal(self.journal)
            if cut is not None:
                self._set_aside_cut_line(cut)
        for path in self.directory.glob("*.json"):
            if path.stem.startswith("node_") or path.name == "anchor.json":
                self.records[path.stem] = json.loads(path.read_text())

    def _set_aside_cut_line(self, offset):
        raw = self.journal.read_bytes()
        fragment = raw[offset:]
        aside = self.directory / f"nodes.truncated-{uuid.uuid4().hex[:8]}.json"
        aside.write_text(json.dumps(dict(offset=offset, fragment_hex=fragment.hex()), indent=1))
        with self.journal.open("r+b") as stream:
            stream.truncate(offset)
            stream.flush()
            os.fsync(stream.fileno())

    def get(self, name):
        with self.lock:
            return self.records.get(name)

    def __len__(self):
        return len(self.records)

    def checkpoint(self, name, compute):
        with self.lock:
            if name in self.records:
                return self.records[name]
        record = _normalised(compute())
        with self.lock:
            if name in self.records:                  # done meanwhile by another thread
                return self.records[name]
            if self.files:
                self._save(self.directory / f"{name}.json", record)
            else:
                line = json.dumps(dict(event="node", node=name, record=record), allow_nan=False) + "\n"
                with self.journal.open("a", encoding="utf-8") as stream:
                    stream.write(line)
                    stream.flush()
                    os.fsync(stream.fileno())
            self.records[name] = record
        return record
