"""Where a run keeps its training sets, and an index of what a run holds (run.json).

A training set N is the points network N is trained on. Runs have always written it whole,
``training_data/data_it_N.csv``, every iteration, so every point is written again at every later
iteration (on the Grendel runs of October and November 2026 the files of a run added up to 10 to 30
times its last one). With ``storage.training_sets: increments`` an iteration writes only the rows
it added, ``training_data/added_it_N.csv``: the rows of set N after those of set N - 1, in the same
CSV format, header included. A set is then the last whole file at or before N followed by the
increments after it, read in order. The two kinds mix in one run (a whole file is written when the
rows before do not stay as they were, and by a run without the setting), and a run that only ever
wrote whole files is read exactly as before. The derived parameters follow the same rule in
``derived_data/`` (``derived_it_N.csv``, ``derived_added_it_N.csv``).

Read training sets through this module, never by opening ``data_it_N.csv``: for a run with
increments it is there for the first set only.

``python client.py index RUN`` prints what a run holds per iteration; ``--write`` also saves that
as ``RUN/run.json`` (the loop rewrites it after every acquisition). The index is built from the
files and can always be built again: no program needs it to read a run.
"""
from dataclasses import dataclass
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

INDEX_SCHEMA = 1
INDEX_NAME = "run.json"


@dataclass(frozen=True)
class SetKind:
    folder: str
    whole: str          # file name of a whole set, with {} for the iteration
    added: str          # file name of the rows an iteration added

    def whole_path(self, root, iteration):
        return Path(root) / self.folder / self.whole.format(iteration)

    def added_path(self, root, iteration):
        return Path(root) / self.folder / self.added.format(iteration)

    def pattern(self, name):
        return re.compile(re.escape(name).replace(r"\{\}", r"(\d+)"))


TRAINING = SetKind("training_data", "data_it_{}.csv", "added_it_{}.csv")
DERIVED = SetKind("derived_data", "derived_it_{}.csv", "derived_added_it_{}.csv")


def _root(run_or_folder, kind):
    """A run directory, or the set's own folder (as TrainingDataset.load is given it)."""
    path = Path(run_or_folder)
    return path.parent if path.name == kind.folder else path


def set_iterations(run, kind=TRAINING):
    """{N: 'whole' | 'added'} of the set files on disk; a whole file wins where both exist."""
    folder = _root(run, kind) / kind.folder
    found = {}
    if not folder.is_dir():
        return found
    added, whole = kind.pattern(kind.added), kind.pattern(kind.whole)
    for path in folder.iterdir():
        if match := added.fullmatch(path.name):
            found.setdefault(int(match.group(1)), "added")
        elif match := whole.fullmatch(path.name):
            found[int(match.group(1))] = "whole"
    return found


def set_files(run, iteration, kind=TRAINING):
    """The files that make up set `iteration`, in the order their rows come: the last whole
    file at or before it, then the increments after that. FileNotFoundError when the set
    cannot be put together (no such set, or a missing link)."""
    root = _root(run, kind)
    files = []
    for n in range(int(iteration), -1, -1):
        whole = kind.whole_path(root, n)
        if whole.exists():
            return [whole, *reversed(files)]
        added = kind.added_path(root, n)
        if not added.exists():
            if files:
                raise FileNotFoundError(f"{root}: set {iteration} of {kind.folder} needs {kind.whole.format(n)} "
                                        f"or {kind.added.format(n)}, and neither is there")
            raise FileNotFoundError(f"{root}: no {kind.folder}/{kind.whole.format(iteration)} "
                                    f"or {kind.added.format(iteration)}")
        files.append(added)
    raise FileNotFoundError(f"{root}: {kind.folder} set {iteration} has increments but no whole set before them")


def has_set(run, iteration, kind=TRAINING):
    try:
        set_files(run, iteration, kind)
    except FileNotFoundError:
        return False
    return True


def latest_set(run, kind=TRAINING):
    """The highest iteration with a set file, or None."""
    found = set_iterations(run, kind)
    return max(found) if found else None


def read_set(run, iteration, kind=TRAINING, usecols=None):
    """Set `iteration` as one DataFrame, row for row what the whole file of it holds."""
    import pandas as pd

    frames = [pd.read_csv(path, usecols=usecols) for path in set_files(run, iteration, kind)]
    # An iteration that added no rows has a header-only increment, which pandas reads with
    # object columns: joined in, it would turn every column into objects.
    rows = [frame for frame in frames[1:] if len(frame)]
    return frames[0] if not rows else pd.concat([frames[0], *rows], ignore_index=True)


def training_set(run, iteration, usecols=None):
    return read_set(run, iteration, TRAINING, usecols)


def _rows(path):
    with Path(path).open("rb") as stream:
        return max(0, sum(1 for _ in stream) - 1)


def set_rows(run, iteration, kind=TRAINING, counted=None):
    """The number of rows of set `iteration` (0 when there is none). `counted`, a dict
    {path: (size, rows)}, saves counting a file again."""
    try:
        files = set_files(run, iteration, kind)
    except FileNotFoundError:
        return 0
    total = 0
    for path in files:
        size = path.stat().st_size
        cached = None if counted is None else counted.get(str(path))
        if cached is not None and cached[0] == size:
            total += cached[1]
            continue
        rows = _rows(path)
        if counted is not None:
            counted[str(path)] = (size, rows)
        total += rows
    return total


# ---- the index ----

_ITERATION = re.compile(r"_it_(\d+)(?!\d)")
_NOT_INDEXED = ("interrupted",)


def client_commit():
    """The git commit of this checkout and whether it has changes, or None outside git."""
    here = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return dict(commit=commit, modified=bool(dirty))


def describe(run, previous=None):
    """What a run holds: per iteration its files, the size of its training set and how it is
    stored; per folder the number of files and bytes. `previous`, an index written before,
    saves counting the rows of files that have not changed."""
    root = Path(run)
    counted = {}
    if previous:
        for entry in previous.get("iterations", []):
            for name, (size, rows) in entry.get("_counted", {}).items():
                counted[str(root / name)] = (size, rows)
    folders, per_iteration = {}, {}
    for directory, dirs, files in os.walk(root):
        relative = Path(directory).relative_to(root)
        if relative.parts and relative.parts[0] in _NOT_INDEXED:
            dirs[:] = []
            continue
        for name in files:
            if name.startswith("."):
                continue
            path = Path(directory) / name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            top = relative.parts[0] if relative.parts else "."
            folder = folders.setdefault(top, dict(files=0, bytes=0))
            folder["files"] += 1
            folder["bytes"] += size
            match = next((m for part in (*relative.parts, name) if (m := _ITERATION.search(part))), None)
            if match:
                entry = per_iteration.setdefault(int(match.group(1)), dict(files=[], bytes=0))
                entry["files"].append(str(path.relative_to(root)))
                entry["bytes"] += size
    stored = set_iterations(root, TRAINING)
    iterations = []
    for n in sorted(set(per_iteration) | set(stored)):
        entry = dict(iteration=n, **per_iteration.get(n, dict(files=[], bytes=0)))
        entry["files"].sort()
        if n in stored:
            files = set_files(root, n, TRAINING) if has_set(root, n) else []
            entry["training_set"] = dict(
                stored=stored[n], files=[str(p.relative_to(root)) for p in files],
                rows=set_rows(root, n, TRAINING, counted))
            entry["_counted"] = {str(p.relative_to(root)): list(counted[str(p)]) for p in files if str(p) in counted}
        network = root / "trained_models" / f"model_it_{n}.keras"
        if network.exists():
            entry["network"] = str(network.relative_to(root))
        iterations.append(entry)
    return dict(schema=INDEX_SCHEMA, run=root.name, written=time.strftime("%Y-%m-%d %H:%M:%S"),
                client=client_commit(), folders=dict(sorted(folders.items())), iterations=iterations,
                note="Built from the files (run/layout.py); it can always be built again, and nothing "
                     "needs it to read the run. _counted: the row counts behind training_set.rows.")


def write_index(run):
    """Write RUN/run.json from the files; a run's earlier index saves counting rows again."""
    from run.storage import write_json

    root = Path(run)
    previous = None
    try:
        previous = json.loads((root / INDEX_NAME).read_text())
    except (OSError, ValueError):
        pass
    index = describe(root, previous if isinstance(previous, dict) else None)
    write_json(root / INDEX_NAME, index)
    return index


def _size(n):
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000


def main(argv=None):
    parser = argparse.ArgumentParser(prog="client.py index", description="What a run holds, per iteration and per folder.")
    parser.add_argument("run", type=Path, help="run directory")
    parser.add_argument("--write", action="store_true", help=f"also save the index as RUN/{INDEX_NAME}")
    parser.add_argument("--json", action="store_true", help="print the index as JSON")
    args = parser.parse_args(argv)
    if not args.run.is_dir():
        raise SystemExit(f"{args.run}: not a run directory")
    index = write_index(args.run) if args.write else describe(args.run)
    if args.json:
        print(json.dumps(index, indent=1))
        return
    total = sum(f["files"] for f in index["folders"].values()), sum(f["bytes"] for f in index["folders"].values())
    print(f"{index['run']}: {total[0]} files, {_size(total[1])}")
    for name, folder in sorted(index["folders"].items(), key=lambda item: -item[1]["bytes"]):
        print(f"  {name + '/':28s} {folder['files']:6d} files  {_size(folder['bytes']):>9s}")
    print(f"  {'iteration':>9s} {'files':>6s} {'bytes':>9s} {'training set':>13s}  stored  network")
    for entry in index["iterations"]:
        ts = entry.get("training_set")
        print(f"  {entry['iteration']:9d} {len(entry['files']):6d} {_size(entry['bytes']):>9s} "
              f"{ts['rows'] if ts else '':>13} {ts['stored'] if ts else '':>7s}  {'yes' if 'network' in entry else ''}")
    if args.write:
        print(f"written: {args.run / INDEX_NAME}")


if __name__ == "__main__":
    main()
