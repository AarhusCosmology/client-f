"""Resolve the saved surrogate region consistently across consumers."""
from pathlib import Path

from run.run import iterations
from run.storage import digest


def region_file(run, iteration):
    """RUN/boost/region_it_M.npz with the largest M <= iteration, or None: the region the
    network of that iteration was last sampled in (profiles, the evidence, derived chains and
    `client.py modes` all take it by this rule)."""
    found = {n: path for n, path in iterations(Path(run) / "boost", "region_it_*.npz").items()
             if n <= iteration}
    return found[max(found)] if found else None


def resolve(run, iteration, setting):
    """What a study's results depend on: None (the bare network), or the region's file, its
    hash and the edge in units of the boost points' reach. This goes into the manifest."""
    if setting is None:
        return None
    path = region_file(run, iteration)
    if path is None:
        return None
    from sampling.region import Region

    edge = Region.load(path).inflate if setting == "run" else float(setting)
    return dict(file=f"boost/{path.name}", sha256=digest(path),
                edge=float(edge))


def load(run, entry):
    """The Region of a manifest's entry, with the entry's edge."""
    from sampling.region import Region

    path = Path(run) / entry["file"]
    if digest(path) != entry["sha256"]:
        raise ValueError(f"Surrogate region changed: {path}")
    region = Region.load(path)
    region.inflate = float(entry["edge"])
    return region



def load_snapshot(run, iteration, region="run"):
    """Load immutable copies and return the identity of those exact bytes.

    The source may be actively training/auditing. Refuse a change while copying; once
    copied, subsequent replacements cannot change either the loaded target or its identity.
    No phase records are required, so saved runs from before phase commits still work.
    """
    import shutil
    import tempfile
    from likelihood.surrogate import SurrogateLikelihood, SurrogateMetadata
    from model.network import load_model
    from sampling.provenance import target_identity

    run = Path(run)
    entry = resolve(run, iteration, region)
    paths = [Path("metadata.json"), Path("trained_models") / f"model_it_{iteration}.keras"]
    if entry is not None:
        paths.append(Path(entry["file"]))
    before = {str(path): digest(run / path) for path in paths}
    with tempfile.TemporaryDirectory(prefix="client-target-") as directory:
        snapshot = Path(directory)
        for path in paths:
            (snapshot / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run / path, snapshot / path)
        for root in (snapshot, run):
            after = {str(path): digest(root / path) for path in paths}
            if after != before or resolve(root, iteration, region) != entry:
                raise RuntimeError("Surrogate target changed while taking its snapshot; retry after training finishes")
        md = SurrogateMetadata.load(snapshot / "metadata.json")
        identity = target_identity(snapshot, iteration, md, entry)
        target = SurrogateLikelihood(load_model(snapshot / "trained_models" / f"model_it_{iteration}.keras"),
                                    md, region=None if entry is None else load(snapshot, entry))
    return target, identity


def load_surrogate(run, iteration, region="run"):
    target, identity = load_snapshot(run, iteration, region)
    return target, identity["region"]
