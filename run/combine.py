"""Domain splitting: runs of one objective on adjacent boxes in one parameter, as one emulator.

    python client.py combine OUTPUT --parameter omega --part RUN_A[:N] --part RUN_B[:N] --band LO:HI

A mode that a run on the whole box never finds (the axion run's low-omega corner: a ridge holding
2% of the posterior, 120 to 280 log-units under every network of the full-box run) is the main
feature of the posterior in a box cut down to it, and a run there finds it by itself. Each part
is an ordinary CLiENT run of the same objective (the same likelihood and prior; only the training
box differs, in `parameter` alone). Listed in increasing order of that parameter, with one band
between each pair: below a band the lower part's network is used, above it the upper part's,
and inside it the two are blended with SmoothStep (model/network.py), whose first and second
derivatives vanish at the band's ends. Each band must lie inside both parts' boxes.

Both networks emulate the objective itself, not a normalised posterior, so the parts' relative
weights need no evidence ratio: they follow from the values, as far as each network is right
near its own posterior. Each part's network is seen only inside its own boost region
(model.network.RegionWall, inside the combined model): a network has maxima beyond its region,
and the region of one part may reach into the range where another part's network is used.

OUTPUT is a new directory that reads as a finished run with one network, 0:
  trained_models/model_it_0.keras    the combined network: one flat Keras graph (the profile
                                     code copies it layer by layer at double precision)
  boost/region_it_0.npz              every part's region cells, in the combined box's units
  training_data/data_it_0.csv        every part's training points with a value (refused ones left out)
  metadata.json                      the combined box; every other range as the parts'
  <the widest part's YAML>, likelihood_input/, refused_value.json, objective_identity.json
  convergence_stats/chain_summary_it_0.npz   pooled from the parts' (hpc/analysis/report.py needs one)
  combine.json                       the parts (run, network, hashes, boxes), the bands, the walls
The evidence, the true check, the report, the profiles and `client.py modes` read it as any run.
It is not a run to continue: the loop refuses a directory with combine.json.
"""
import argparse
import copy
import dataclasses
import json
import re
import shutil
import time
import tempfile
from pathlib import Path

import numpy as np

PREFIX = "part{}_"


def parse_part(text):
    """RUN or RUN:N."""
    path, _, number = text.rpartition(":") if re.fullmatch(r".+:\d+", text) else (text, "", "")
    return Path(path), (int(number) if number else None)


def parse_band(text):
    lo, hi = (float(v) for v in text.split(":"))
    if not hi > lo:
        raise ValueError(f"band {text}: the upper end must be above the lower")
    return lo, hi


class Part:
    """One run and the network of it that goes into the combination."""

    def __init__(self, run, iteration=None):
        from likelihood.surrogate import SurrogateMetadata
        from run.run import iterations, settings_file
        from sampling.targets import region_file

        self.run = Path(run)
        self.requested_iteration = iteration
        self.source_run = self.run
        models = iterations(self.run / "trained_models", "model_it_*.keras")
        if not models:
            raise ValueError(f"{self.run}: no trained network")
        self.iteration = max(models) if iteration is None else int(iteration)
        if self.iteration not in models:
            raise ValueError(f"{self.run}: no network {self.iteration}")
        self.model_path = models[self.iteration]
        self.data_path = self.run / "training_data" / f"data_it_{self.iteration}.csv"
        if not self.data_path.exists():
            raise ValueError(f"{self.run}: no training set data_it_{self.iteration}.csv")
        self.metadata = SurrogateMetadata.load(self.run / "metadata.json")
        self.names = self.metadata.param_names
        self.yaml = settings_file(self.run)
        identity = self.run / "objective_identity.json"
        if not identity.exists():
            raise ValueError(f"{self.run}: no objective_identity.json; the parts' objectives cannot be compared")
        self.identity = json.loads(identity.read_text())["identity"]
        refused = self.run / "refused_value.json"
        self.refused = json.loads(refused.read_text())["value"] if refused.exists() else None
        self.region_path = region_file(self.run, self.iteration)

    def box(self, name):
        return tuple(float(v) for v in self.metadata.bounds[name])

    def hash_of(self, path):
        """The hash of one of this part's files, taken when its snapshot was verified."""
        return self.artifact_hashes[str(Path(path).relative_to(self.run))]


def check_parts(parts, parameter, bands):
    """The parts must be one objective on adjacent boxes in `parameter`, with a band inside both
    boxes between each pair. Raises ValueError with what is wrong."""
    from run.identity import differences

    if len(parts) < 2:
        raise ValueError("combine needs at least two parts")
    if len(bands) != len(parts) - 1:
        raise ValueError(f"{len(parts)} parts need {len(parts) - 1} band(s), not {len(bands)}")
    restricted = [p.region_path is not None for p in parts]
    if any(restricted) and not all(restricted):
        raise ValueError("cannot combine parts with and without regions: the outer region would "
                         "truncate the unrestricted part; use consistently restricted or unrestricted parts")
    names = parts[0].names
    if parameter not in names:
        raise ValueError(f"{parameter} is not a parameter of the runs")
    for part in parts[1:]:
        if part.names != names:
            raise ValueError(f"{part.source_run}: other parameters, or another order, than {parts[0].source_run}")
        for name in names:
            if name != parameter and part.box(name) != parts[0].box(name):
                raise ValueError(f"{part.source_run}: the box differs in {name} ({part.box(name)} against "
                                 f"{parts[0].box(name)}); only {parameter} may differ")

    def without_parameter(identity):
        out = copy.deepcopy(identity)
        for p in out.get("parameters") or []:
            if p.get("name") == parameter:
                p.pop("lower", None)
                p.pop("upper", None)
        return out

    for part in parts[1:]:
        found = differences(without_parameter(parts[0].identity), without_parameter(part.identity))
        if found:
            raise ValueError(f"{part.source_run} has another objective than {parts[0].source_run}: " + "; ".join(found))
    for j, (lo, hi) in enumerate(bands):
        below, above = parts[j].box(parameter), parts[j + 1].box(parameter)
        if not (above[0] <= lo and hi <= below[1]):
            raise ValueError(f"band [{lo:g}, {hi:g}] is not inside both boxes: {parameter} in "
                             f"[{below[0]:g}, {below[1]:g}] ({parts[j].source_run}) and [{above[0]:g}, {above[1]:g}] "
                             f"({parts[j + 1].source_run}); list the parts in increasing {parameter}")
        if j and lo < bands[j - 1][1]:
            raise ValueError("the bands overlap or are not in increasing order")
    # Part j is used from the band below it (or the bottom of the combined box) to the band above
    # it (or the top): that whole range must be inside its own box.
    whole = (min(p.box(parameter)[0] for p in parts), max(p.box(parameter)[1] for p in parts))
    for j, part in enumerate(parts):
        use = (whole[0] if j == 0 else bands[j - 1][0], whole[1] if j == len(parts) - 1 else bands[j][1])
        box = part.box(parameter)
        if not (box[0] <= use[0] and use[1] <= box[1]):
            raise ValueError(f"{part.source_run} would be used for {parameter} in [{use[0]:g}, {use[1]:g}], outside its box "
                             f"[{box[0]:g}, {box[1]:g}]; list the parts in increasing {parameter}")


def rebuild(model, x, prefix):
    """`model`'s graph, every layer copied under prefix + its name, on tensor `x`: the copies
    hold the same weights, and the result is part of one flat graph with x's other parts."""
    import tensorflow as tf

    mapping = {id(t): x for t in model.inputs}
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        config = layer.get_config()
        config["name"] = prefix + layer.name
        clone = type(layer).from_config(config)
        inputs = layer.input
        out = clone([mapping[id(t)] for t in inputs] if isinstance(inputs, (list, tuple)) else mapping[id(inputs)])
        clone.set_weights(layer.get_weights())
        if isinstance(clone, tf.keras.layers.Normalization):
            clone.finalize_state()
        mapping[id(layer.output)] = out
    return mapping[id(model.output)]


def combined_model(parts, parameter, bands, walls=True):
    """One Keras model: part k's network (inside its own region with `walls`) where `parameter`
    is in part k's range, SmoothStep blends in the bands."""
    import tensorflow as tf
    from model.network import RegionWall, SmoothStep, load_model
    from sampling.region import STRENGTH, Region

    index = parts[0].names.index(parameter)
    x = tf.keras.Input(shape=(len(parts[0].names),), name="parameters")
    outs = []
    for k, part in enumerate(parts):
        out = rebuild(load_model(part.model_path), x, PREFIX.format(k))
        if walls and part.region_path is not None:
            r = Region.load(part.region_path)
            out = RegionWall(r.means, r.whiten, r.reach, r.lower, r.width, r.inflate, r.floor, STRENGTH,
                             name=PREFIX.format(k) + "region_wall")([out, x])
        outs.append(out)
    out = outs[0]
    for j, (lo, hi) in enumerate(bands):
        # (1 - s) A + s B, each weight exactly 0 or 1 outside the band: there the value is exactly
        # one network's, however far below the other one's wall lies (A + s (B - A) rounded it).
        below = tf.keras.layers.Multiply(name=f"blend_{j}_below")(
            [SmoothStep(index, lo, hi, falling=True, name=f"blend_{j}_falling")(x), out])
        above = tf.keras.layers.Multiply(name=f"blend_{j}_above")(
            [SmoothStep(index, lo, hi, name=f"blend_{j}_rising")(x), outs[j + 1]])
        out = tf.keras.layers.Add(name=f"blend_{j}_value")([below, above])
    return tf.keras.Model(x, out, name="combined")


def rebox(region, lower, upper):
    """The same ellipsoids in another box's units: (means, whiten). A point's distance from a
    centre is unchanged (u = (x - lower) / width in either box)."""
    lower, width = np.asarray(lower, float), np.asarray(upper, float) - np.asarray(lower, float)
    scale = width / region.width                               # D: u_own = a + D u_new
    shift = (lower - region.lower) / region.width              # a
    means = (region.means - shift) / scale
    whiten = scale[None, :, None] * region.whiten              # diag(D) W for every cell
    return means, whiten


def combined_region(parts, lower, upper, cell_sources=None):
    """Every part's cells in one Region over the combined box, or None when no part has one."""
    from sampling.region import Region

    regions = [Region.load(p.region_path) for p in parts if p.region_path is not None]
    if not regions:
        return None
    edges = {r.inflate for r in regions}
    if len(edges) != 1:
        raise ValueError(f"the parts' regions have different edges {sorted(edges)}; combine needs one")
    means, whiten, reach, rows = [], [], [], []
    groups, metrics = [], []
    # Identical ellipsoids have identical labels everywhere: the second cell is empty.
    # Compare their normalised quadratic forms, allowing only floating-point reboxing noise.
    tol = 32 * np.finfo(float).eps
    for part_index, r in enumerate(regions):
        m, w = rebox(r, lower, upper)
        for j in range(len(r)):
            factor = w[j] / r.reach[j]
            metric = factor @ factor.T
            same = next((k for k in range(len(means))
                         if np.allclose(m[j], means[k], rtol=tol, atol=tol)
                         and np.allclose(metric, metrics[k], rtol=tol, atol=0.0)), None)
            source = dict(part=part_index, cell=j)
            if same is not None:
                rows[same] += r.rows[j]
                groups[same].append(source)
            else:
                means.append(m[j])
                whiten.append(w[j])
                reach.append(r.reach[j])
                rows.append(r.rows[j])
                metrics.append(metric)
                groups.append([source])
    if cell_sources is not None:
        cell_sources.extend(groups)
    return Region(means, whiten, reach, lower, upper,
                  regions[0].inflate, min(r.floor for r in regions), rows=rows)


def _source_files(part):
    """All bytes consumed by combination, including optional files and recursive inputs."""
    files = [part.model_path, part.data_path, part.yaml, part.run / "metadata.json",
             part.run / "objective_identity.json"]
    if part.region_path is not None:
        files.append(part.region_path)
    for path in (part.run / "refused_value.json",
                 part.run / "convergence_stats" / f"chain_summary_it_{part.iteration}.npz"):
        if path.exists():
            files.append(path)
    files.extend(p for p in (part.run / "likelihood_input").rglob("*") if p.is_file())
    return sorted(p.relative_to(part.run) for p in files)


def snapshot_part(part, destination):
    """Freeze one source; refuse changed selections/bytes during copying. Read only the copy."""
    from run.storage import digest

    source = Part(part.run, part.requested_iteration)
    files = _source_files(source)
    before = {str(p): digest(source.run / p) for p in files}
    for path in files:
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.run / path, target)
    after = Part(source.run, part.requested_iteration)
    if (_source_files(after) != files or after.iteration != source.iteration
            or any(digest(source.run / p) != before[str(p)]
                   or digest(destination / p) != before[str(p)] for p in files)):
        raise ValueError(f"{source.run}: source changed while taking combination snapshot; retry when stable")
    frozen = Part(destination, source.iteration)
    frozen.source_run = source.run
    frozen.artifact_hashes = before
    return frozen


def combine(output, parts, parameter, bands, walls=True, log=print):
    """Snapshot inputs and publish a complete emulator by a single directory rename."""
    output = Path(output)
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError(f"{output} exists and is not empty (or is a symlink)")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".client-combine-", dir=output.parent) as temporary:
        root = Path(temporary)
        frozen = [snapshot_part(part, root / f"source{k}") for k, part in enumerate(parts)]
        staged = root / "output"
        record = _write_combined(staged, frozen, parameter, bands, walls, log)
        # POSIX rename replaces an empty directory only, never a nonempty output.
        # Refuse a symlink even if one appeared while staging.
        if output.is_symlink():
            raise ValueError(f"{output}: output became a symlink while combining")
        staged.rename(output)
    return record


def _write_combined(output, parts, parameter, bands, walls=True, log=print):
    """Write the combined emulator to `output` (a new directory). Returns its combine.json."""
    import pandas as pd
    from likelihood.surrogate import SurrogateMetadata
    from run.storage import atomic_path, save_model, write_json

    output = Path(output)
    check_parts(parts, parameter, bands)
    names = parts[0].names
    spans = [p.box(parameter) for p in parts]
    whole = (min(s[0] for s in spans), max(s[1] for s in spans))
    base = max(parts, key=lambda p: p.box(parameter)[1] - p.box(parameter)[0])   # the widest box
    index = names.index(parameter)
    lower = np.array([parts[0].box(n)[0] for n in names], float)
    upper = np.array([parts[0].box(n)[1] for n in names], float)
    lower[index], upper[index] = whole

    cell_sources = []
    region = combined_region(parts, lower, upper, cell_sources)

    for sub in ("trained_models", "training_data", "boost", "convergence_stats", "likelihood_input"):
        (output / sub).mkdir(parents=True, exist_ok=True)
    metadata = SurrogateMetadata(parameters=tuple(
        dataclasses.replace(p, lower=whole[0], upper=whole[1]) if p.name == parameter else p
        for p in base.metadata.parameters))
    metadata.save(output / "metadata.json")

    model = combined_model(parts, parameter, bands, walls=walls)
    save_model(model, output / "trained_models" / "model_it_0.keras")
    log(f"combined network: {len(parts)} parts in {parameter}, bands "
        + ", ".join(f"[{lo:g}, {hi:g}]" for lo, hi in bands)
        + (", each part inside its own region" if walls else ", no walls"))

    if region is not None:
        with atomic_path(output / "boost" / "region_it_0.npz") as temporary:
            region.save(temporary, combined=[str(p.source_run) for p in parts])
        log(f"region: {len(region)} cell(s) from {sum(p.region_path is not None for p in parts)} part(s), edge {region.inflate:g}")

    frames = []
    for p in parts:
        d = pd.read_csv(p.data_path)[names + ["loglkl"]]
        keep = np.isfinite(d["loglkl"]) & ((d["loglkl"] != p.refused) if p.refused is not None else True)
        frames.append(d[keep])
        log(f"training points of {p.source_run.name} (network {p.iteration}): {int(keep.sum())} of {len(d)}"
            + (f" ({int((~keep).sum())} refused or without a value left out)" if (~keep).any() else ""))
    union = pd.concat(frames, ignore_index=True)
    with atomic_path(output / "training_data" / "data_it_0.csv") as temporary:
        union.to_csv(temporary, index=False)

    with atomic_path(output / base.yaml.name) as temporary:
        temporary.write_text(f"# A combined emulator (client.py combine, combine.json): the settings of {base.source_run.name}, the\n"
                             f"# part with the widest box in {parameter}. Not a run to continue.\n" + base.yaml.read_text())
    if (base.run / "likelihood_input").exists():
        shutil.copytree(base.run / "likelihood_input", output / "likelihood_input", dirs_exist_ok=True)
    if (base.run / "refused_value.json").exists():
        shutil.copy2(base.run / "refused_value.json", output / "refused_value.json")
    identity = copy.deepcopy(base.identity)
    for p in identity.get("parameters") or []:
        if p.get("name") == parameter:
            p["lower"], p["upper"] = whole
    write_json(output / "objective_identity.json", dict(identity=identity, historical_values_verified=False,
                                                        combined=[str(p.source_run) for p in parts]))

    summaries = [np.load(p.run / "convergence_stats" / f"chain_summary_it_{p.iteration}.npz")
                 for p in parts if (p.run / "convergence_stats" / f"chain_summary_it_{p.iteration}.npz").exists()]
    if summaries:
        means = np.array([s["mean"] for s in summaries])
        cov = np.mean([s["cov"] for s in summaries], axis=0)
        if len(means) > 1:                                     # the spread of the parts' means
            cov = cov + np.atleast_2d(np.cov(means.T, bias=True))
        with atomic_path(output / "convergence_stats" / "chain_summary_it_0.npz") as temporary:
            with open(temporary, "wb") as stream:
                np.savez(stream, mean=means.mean(axis=0), cov=cov)

    record = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"), parameter=parameter,
        bands=[list(b) for b in bands], walls=bool(walls), box=list(whole), settings_from=str(base.source_run),
        region_cell_sources=cell_sources,
        parts=[dict(run=str(p.source_run), artifacts=p.artifact_hashes, iteration=p.iteration, box=list(p.box(parameter)),
                    model_sha256=p.hash_of(p.model_path), data_sha256=p.hash_of(p.data_path),
                    region=None if p.region_path is None else dict(file=f"boost/{p.region_path.name}",
                                                                    sha256=p.hash_of(p.region_path)))
               for p in parts],
        training_points=int(len(union)))
    write_json(output / "combine.json", record)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(prog="client.py combine", description=__doc__.split("\n\n")[0])
    p.add_argument("output", type=Path, help="the new directory of the combined emulator")
    p.add_argument("--parameter", required=True, help="the parameter in which the parts' boxes differ")
    p.add_argument("--part", action="append", required=True, metavar="RUN[:N]",
                   help="a run and its network (default: its latest), in increasing order of the parameter; repeat")
    p.add_argument("--band", action="append", required=True, metavar="LO:HI",
                   help="where two adjacent parts are blended; one fewer than the parts, in increasing order")
    p.add_argument("--no-walls", action="store_true",
                   help="use each part's network also outside its own region (not recommended)")
    args = p.parse_args(argv)
    parts = [Part(*parse_part(text)) for text in args.part]
    try:
        combine(args.output, parts, args.parameter, [parse_band(b) for b in args.band], walls=not args.no_walls)
    except ValueError as error:
        raise SystemExit(f"client.py combine: {error}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
