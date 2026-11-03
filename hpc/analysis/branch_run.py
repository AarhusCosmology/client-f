"""Branch a run at network N: a new run directory that holds the run up to that network and
starts its boosts there.

    python analysis/branch_run.py results/<timestamp>_<RUN> --at 11

For the question "would the boosts have worked from an earlier network?". The new directory,
results/<now>_<RUN>, gets every file of the source that belongs to iterations 0 to N (the
training sets, the networks, the audit's points) and a copy of the run's settings with
acquisition.boost.start_iteration set to N. Nothing in the source is touched. Continue it as
any run; network N is the source's, and its acquisition becomes the first boost:

    sbatch submit_client.sh results/<now>_<RUN> -i 5         # five boosts: networks N+1 to N+5

with the refused value the source was started with, if it had one (launch.py checks; the
command this prints has it when the source recorded it).

For "would the boosts have gone better with another setting?", `--set SECTION.KEY=VALUE`
(repeatable) gives the branch that setting from its start, for example

    python analysis/branch_run.py results/<timestamp>_<RUN> --at 12 --set training.warm_start=false

The line is changed, or added as the section's first, in the branch's copy of the settings,
which is written once, here; the comments stay, and BRANCHED_FROM.txt names the change. A key
of a subsection is set the same way (any of the boosts' but their start, which is --at):

    python analysis/branch_run.py results/<timestamp>_<RUN> --at 8 --set acquisition.boost.region=1.5

For "would the rest of the boosts have gone better with another setting?", `--train-from N`
branches inside the boosts (N after their start) and keeps the source's start: the branch
gets the source's networks 0 to N - 1 and everything their boosts made, including the
training set of N, and trains network N and every later one itself, with its --set settings
(30 October 2026: the H0WC axion high part, whose warm-started networks grew false maxima):

    python analysis/branch_run.py results/<timestamp>_<RUN> --train-from 8 --set training.warm_start=false
    sbatch submit_client.sh results/<now>_<RUN> -i 8        # networks 8 to 16: the boosts the source has left

Network N is trained as the source trained it, with the training domain, background and
region of the source's boosts, except for the settings changed.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import struct
import time
from pathlib import Path

import yaml

# The iteration in a file's name. Not followed by a digit, whatever else follows: with a dot
# required, evidence_it_25_nlive2000.json and phases/history/training_it_25-<stamp>.json had
# no iteration and went into every branch.
ITERATION = re.compile(r"_it_(\d+)(?!\d)")
# The source's own, whatever the iteration: its reports and plots, its evidences and mode
# searches (a branch's evidence/ must hold what the branch computed), and what its unfinished
# phases left.
SOURCE_ONLY = ("analysis", "evidence", "modes", "interrupted", "objective_calls", "branch_provenance")
# The record that the points of `client.py augment` have had their first training. Its name
# carries the training set the points were chosen for (augment_it_K), its content the network
# trained on them first: K + 1 at the earliest.
TRAINED = re.compile(r"augment_it_\d+\.trained\.json")


def as_yaml(value):
    """`value` as a YAML scalar or flow collection that reads back as itself. Not JSON: 1e-06
    in JSON reads back from YAML as a string."""
    text = yaml.safe_dump(value, default_flow_style=True, width=1 << 16).strip()
    return text[:-3].rstrip() if text.endswith("...") else text


def set_setting(text, keys, value):
    """`text` (a run's settings) with the setting `keys` (section, key or section, subsection,
    key) set to `value`: its line replaced where the (sub)section has one, added as the
    (sub)section's first line where it has none. Every other line, comments included, stays
    as it is."""
    lines = text.splitlines(keepends=True)
    start, end, indent = 0, len(lines), ""
    for depth, name in enumerate(keys[:-1]):
        header = next((i for i in range(start, end)
                       if re.fullmatch(rf"{re.escape(indent)}{re.escape(name)}:\s*(#.*)?\n?", lines[i])), None)
        if header is None:
            raise SystemExit(f"--set {'.'.join(keys)}: the settings have no "
                             + ("top-level section" if depth == 0 else "section") + f" {'.'.join(keys[:depth + 1])}:")
        # The (sub)section's lines: those after its header indented deeper than it; comments
        # and blank lines do not end it.
        start, end = header + 1, next((i for i in range(header + 1, end)
                                       if lines[i].strip() and not lines[i].lstrip().startswith("#")
                                       and not lines[i].startswith(indent + " ") and not lines[i].startswith(indent + "\t")), end)
        body = [i for i in range(start, end) if lines[i].strip() and not lines[i].lstrip().startswith("#")]
        indent = re.match(r"\s*", lines[body[0]]).group(0) if body else indent + "  "
    line = f"{indent}{keys[-1]}: {as_yaml(value)}\n"
    existing = [i for i in range(start, end) if re.match(rf"{indent}{re.escape(keys[-1])}:(\s|$)", lines[i])]
    if existing:
        # The key's own block, when its value is one (lines indented deeper), goes with it.
        i = last = existing[0]
        for j in range(i + 1, end):
            if lines[j].strip() and not lines[j].lstrip().startswith("#"):
                if not lines[j].startswith(indent + " ") and not lines[j].startswith(indent + "\t"):
                    break
                last = j
        lines[i:last + 1] = [line]
    else:
        lines.insert(start, line)
    return "".join(lines)


def trained_at(marker):
    """The network a marker of the augmentation names, or None when it cannot be read (CLiENT
    then takes the points as not yet trained on: run/storage.py, dataset/augment.py)."""
    try:
        iteration = json.loads(marker.read_text()).get("iteration")
    except (OSError, ValueError, AttributeError):
        return None
    return iteration if type(iteration) is int and iteration >= 0 else None


def inherited_values(source, iteration, checkpoint=None):
    """Inventory the native values needed at the checkpoint, not cumulative older copies.

    This is a data import, not an objective evaluation or a backend cache fill. Audit and
    optional augmentation additions are restored separately by the continuation.

    checkpoint: the last network taken over, `iteration` but for --train-from, where network
    `iteration` is trained again and its audit's points are not taken over.
    """
    checkpoint = iteration if checkpoint is None else checkpoint
    names = [p["name"] for p in json.loads((source / "metadata.json").read_text())["parameters"]]
    refused_file = source / "refused_value.json"
    refused = json.loads(refused_file.read_text()).get("value") if refused_file.exists() else None
    files, unique = {}, set()
    for relative in (f"training_data/data_it_{iteration}.csv",
                     *([f"audit/audit_it_{iteration}.csv"] if checkpoint == iteration else []),
                     f"augmentation/augment_it_{iteration}.csv"):
        path = source / relative
        if not path.exists():
            continue
        rows = finite = 0
        with path.open() as stream:
            table = csv.DictReader(stream)
            if not set(names + ["loglkl"]) <= set(table.fieldnames or []):
                raise ValueError(f"Cannot inventory inherited native values: {path}")
            for row in table:
                rows += 1
                value = float(row["loglkl"])
                coords = [float(row[n]) for n in names]
                if math.isfinite(value) and value != refused and all(map(math.isfinite, coords)):
                    finite += 1
                    unique.add((struct.pack(f"{len(coords)}f", *coords), value))
        files[relative] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=rows, finite=finite)
    return dict(schema=1, source=str(source), iteration=iteration, files=files,
                **({} if checkpoint == iteration else dict(checkpoint=checkpoint, trained_again_from=iteration)),
                checkpoint_sha256=hashlib.sha256((source / "trained_models" / f"model_it_{checkpoint}.keras").read_bytes()).hexdigest(),
                imports=len(unique), cache_hits=0,
                note="Distinct finite native coordinate/value pairs inherited from the checkpoint, its audit and optional augmentation. "
                     "No new objective attempts or backend cache fills; earlier cumulative training files are not counted again.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", type=Path)
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--at", type=int, help="the network at which the boosts start")
    start.add_argument("--train-from", type=int, metavar="N",
                       help="a branch inside the boosts: the source's networks 0 to N - 1 and the training set of N; "
                            "network N and later trained by the branch; the boosts' start stays the source's")
    parser.add_argument("--out", type=Path, help="parent of the new directory (default: the source's)")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="a setting of the branch other than the source's, e.g. training.warm_start=false "
                             "or acquisition.boost.region=1.5 (repeatable; the value is read as YAML)")
    args = parser.parse_args()
    changes = []
    for item in args.set:
        name, equals, value = item.partition("=")
        keys = name.split(".")
        if (keys[:2] == ["acquisition", "boost"] and len(keys) <= 3
                and (len(keys) == 2 or keys[2] == "start_iteration")):
            raise SystemExit("--set: the boost's start is --at")
        if not equals or not all(keys) or len(keys) not in (2, 3):
            raise SystemExit(f"--set {item}: give SECTION.KEY=VALUE or SECTION.SUBSECTION.KEY=VALUE "
                             "(e.g. acquisition.boost.region=1.5)")
        changes.append((keys, yaml.safe_load(value)))

    source = args.source.resolve()
    name = re.sub(r"^\d{8}_\d{6}_", "", source.name)
    target = (args.out or source.parent) / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}"
    fresh = args.train_from is not None   # --train-from: network N is trained again by the branch
    at = args.train_from if fresh else args.at
    checkpoint = at - 1 if fresh else at     # the last network the branch takes over
    if fresh and at < 1:
        raise SystemExit("--train-from: give a network after the boosts' start")
    for needed in (f"trained_models/model_it_{checkpoint}.keras", f"training_data/data_it_{at}.csv"):
        if not (source / needed).exists():
            raise SystemExit(f"{source} has no {needed}: it cannot be branched at network {at}")
    settings = [*source.glob("*.yaml"), *source.glob("*.yml")]
    if len(settings) != 1:
        raise SystemExit(f"expected one settings file in {source}, found {[p.name for p in settings]}")
    text = settings[0].read_text()
    boost = (yaml.safe_load(text).get("acquisition") or {}).get("boost")
    if not boost:
        raise SystemExit(f"{settings[0]} has no acquisition.boost section to start earlier")
    # Where the source's boosts began: its settings say the latest, and a run whose ordinary
    # loop converged before that (convergence.metric: tempered_batch_scatter) recorded the
    # hand-over in boost/start.json. A branch at a network after it would start "its boosts"
    # at a network already trained on the source's.
    source_start, recorded = boost["start_iteration"], source / "boost" / "start.json"
    if recorded.exists():
        try:
            source_start = min(source_start, int(json.loads(recorded.read_text())["start_iteration"]))
        except (ValueError, KeyError, TypeError) as error:
            raise SystemExit(f"{recorded} cannot be read ({error}): where the source's boosts began is not known") from error
    if fresh:
        # The branch keeps the source's start (and boost/start.json): its network N is trained
        # as a network of the boosts, with their training domain, background and region.
        if at <= source_start:
            raise SystemExit(f"--train-from {at}: the source's boosts start at {source_start}; a network at or before "
                             f"the start is trained again by a branch made with --at {at - 1}")
        new_text = text
    else:
        if args.at > source_start:
            raise SystemExit(f"the source's boosts start at {source_start}, before network {args.at}"
                             + ("" if source_start == boost["start_iteration"] else
                                f" (its ordinary loop converged there; its settings say {boost['start_iteration']})"))
        # The boost's own start_iteration: the first one after the line "boost:" (the audit has one too).
        head, mark, tail = text.partition("\n  boost:")
        changed, count = re.subn(r"(\n\s+start_iteration:\s*)\d+", rf"\g<1>{args.at}", tail, count=1)
        new_text = head + mark + changed
        if not mark or count != 1 or yaml.safe_load(new_text)["acquisition"]["boost"]["start_iteration"] != args.at:
            raise SystemExit(f"could not set acquisition.boost.start_iteration in {settings[0].name}")
        other = yaml.safe_load(new_text); other["acquisition"]["boost"]["start_iteration"] = boost["start_iteration"]
        assert other == yaml.safe_load(text), "only the boost's start may differ"
    expected = yaml.safe_load(new_text)
    for keys, value in changes:
        new_text = set_setting(new_text, keys, value)
        place = expected
        for depth, name in enumerate(keys[:-1]):
            if not isinstance(place.get(name), dict):
                raise SystemExit(f"--set {'.'.join(keys)}: {'.'.join(keys[:depth + 1])} is not a section of mappings "
                                 "in the settings")
            place = place[name]
        place[keys[-1]] = value
    if yaml.safe_load(new_text) != expected:
        raise SystemExit("--set: the branch's settings did not come out as asked; nothing was written")

    def later(relative):
        """Whether a file belongs to an iteration after the branch point."""
        if relative.parts[0] == "augmentation" and TRAINED.fullmatch(relative.name):
            # By what it records, not by its name: with the marker of a training the branch
            # does not hold, the branch's first training on those points would run the
            # ordinary schedule (training.augment_schedule_factor is for that training).
            trained = trained_at(source / relative)
            return trained is None or trained > checkpoint
        match = ITERATION.search(relative.name)
        if fresh:
            # The training set of N (with its derived values and augmentation) is where the
            # branch starts; every other file of N belongs to network N or what came after it.
            if match and int(match.group(1)) == at and (
                    relative.parts[0] in ("training_data", "augmentation")
                    or (relative.parts[0] == "derived_data" and relative.name.startswith("derived_it_"))):
                return False
            return bool(match) and int(match.group(1)) >= at
        # a chain summary of network N is written again when the branch samples it, and so is
        # the record of its acquisition when the source's boosts began at N
        again = relative.parts[0] == "convergence_stats" or (
            relative.parts[0] == "boost" and relative.name.startswith(("boost_it_", "fallback_it_")))
        return bool(match) and int(match.group(1)) > (args.at - 1 if again else args.at)

    def beyond(path):
        """A phase record the branch does not keep: a later iteration's, or the acquisition of
        network N itself, which the branch does again as its first boost."""
        return later(path.relative_to(source)) or (not fresh and path.name == f"acquisition_it_{args.at}.json")

    # Validate the retained history before creating a destination. Merely omitting an
    # incomplete record would turn its copied, unaudited network into an apparently usable
    # checkpoint. Later phases may still be running; they do not belong to this branch, and
    # neither does the acquisition of network N: a source stopped inside it is branched at N
    # like any other.
    phase_records = {}
    phase_paths = sorted((source / "phases").glob("*.json"))
    for path in phase_paths:
        if beyond(path):
            continue
        try:
            record = json.loads(path.read_text())
            if not isinstance(record, dict) or record.get("status") != "complete":
                raise ValueError("phase is incomplete")
            artifacts = record["artifacts"]
            if not isinstance(artifacts, dict) or not artifacts:
                raise ValueError("phase has no committed artifacts")
            for name, expected in artifacts.items():
                artifact = source / name
                if later(Path(name)):
                    continue
                if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"committed artifact changed or missing: {name}")
        except (ValueError, KeyError, OSError) as error:
            raise SystemExit(f"Cannot branch at network {at}: {path.name}: {error}. "
                             "Finish or recover the source phase first, or choose an earlier checkpoint.") from error
        phase_records[path] = record
    # A run begun before the phase records existed has them from its first continuation on:
    # network N needs its record only when the records reach back to it, which is when the
    # branch keeps any. Earlier networks of such a run are as checkable as those of a run
    # without any record, which is branched as it is.
    needed = ([f"training_it_{checkpoint}", f"acquisition_it_{checkpoint}"] if fresh else [f"training_it_{at}"])
    for phase in needed:
        if phase_records and source / f"phases/{phase}.json" not in phase_records:
            raise SystemExit(f"Cannot branch at network {at}: no committed {phase.rpartition('_it_')[0]} phase of "
                             f"network {phase.rpartition('_it_')[2]}")

    inherited = inherited_values(source, at, checkpoint)
    if target.exists():
        raise SystemExit(f"{target} exists")
    # Made under another name and given its own at the end: a copy that stops half-way (a
    # quota, a job's time limit) leaves no directory that reads as a run, and the record of
    # what was inherited, which CLiENT reads after every acquisition, is there whole.
    final, target = target, target.with_name(target.name + ".partial")
    target.mkdir(parents=True, exist_ok=False)
    try:
        copied, left = fill(source, target, at, settings[0], new_text, later, phase_records, inherited,
                            boost["start_iteration"], source_start, changes, fresh)
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    target = target.rename(final)
    if fresh:
        print(f"{target}: {copied} files of networks 0 to {checkpoint} and the training set of {at} ({left} later ones "
              f"left out); network {at} on trained by the branch; boosts from network {source_start} as the source's")
    else:
        print(f"{target}: {copied} files of iterations 0 to {args.at} ({left} later ones left out); boosts from network {args.at} "
              f"(the source: {source_start})")
    # The boosts the source's settings leave from N on (networks N to the last), or five.
    count = max(1, boost["boosts"] - (at - source_start)) if fresh and boost.get("boosts") else 5
    record = target / "refused_value.json"          # written by launch.py; a continued run must be given the same value
    refused = f"CLIENT_REFUSED_VALUE={json.loads(record.read_text())['setting']} " if record.exists() else ""
    print(f"continue it with: {refused}sbatch submit_client.sh "
          f"{target.relative_to(Path.cwd()) if target.is_relative_to(Path.cwd()) else target} -i {count}"
          + ("" if refused else "     (with the CLIENT_REFUSED_VALUE the source was started with, if it had one)"))


def fill(source, target, at, settings, new_text, later, phase_records, inherited, configured_start, source_start,
         changes=(), fresh=False):
    """Copy the source's files of iterations 0 to `at` into the new directory and write the
    branch's own records: (files copied, files left out)."""
    copied = left = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if path.is_dir() or relative.parts[0] in SOURCE_ONLY or relative.parts[:2] == ("phases", "history"):
            continue
        if relative.parts == ("boost", "start.json") and not fresh:
            # the source's hand-over: the branch starts its boosts at N by its settings
            left += 1
            continue
        if later(relative):
            left += 1
            continue
        if relative.parts[0] == "phases" and len(relative.parts) == 2 and path.suffix == ".json":
            # A phase record commits files by hash (CLiENT's run/storage.py), and a continued
            # run checks every one. The acquisition of network N committed the training set of
            # N + 1, which the branch leaves out: that record stays behind with it.
            record = phase_records.get(path)
            artifacts = [Path(name) for name in record["artifacts"]] if record else []
            if record is None or any(later(a) or not (source / a).exists() for a in artifacts):
                left += 1
                continue
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        if path == settings:
            (target / relative).write_text(new_text)
        else:
            shutil.copy2(path, target / relative)
        copied += 1
    # Keep source journals byte-for-byte, including work beyond the branch point, outside
    # the active ledger glob. A branch of a branch keeps that ancestor's provenance here too.
    provenance = target / "branch_provenance"
    provenance.mkdir()
    for folder in ("objective_calls", "branch_provenance"):
        if (source / folder).exists():
            shutil.copytree(source / folder, provenance / "source" / folder)
    inherited["source_history"] = {
        str(path.relative_to(provenance)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((provenance / "source").rglob("*")) if path.is_file()}
    (provenance / "imports.json").write_text(json.dumps(inherited, indent=2) + "\n")
    (target / "BRANCHED_FROM.txt").write_text(
        f"{source}\n"
        + (f"networks 0 to {at - 1} and the training set of {at}; network {at} on trained by the branch; "
           f"acquisition.boost.start_iteration {configured_start} (the source's)\n" if fresh else
           f"networks 0 to {at}; acquisition.boost.start_iteration {configured_start} -> {at}\n")
        + ("" if source_start == configured_start else f"the source's boosts began at {source_start} (boost/start.json)\n")
        + "".join(f"{'.'.join(keys)}: {json.dumps(value)} (--set; the source's settings differ)\n" for keys, value in changes)
        + f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return copied, left


if __name__ == "__main__":
    main()
