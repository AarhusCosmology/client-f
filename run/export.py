"""A run's network as a TensorFlow SavedModel that loads without CLiENT.

    python client.py export RUN OUTPUT [--iteration N] [--no-region] [--dtype float32|float64]

A network saved by the loop (trained_models/model_it_N.keras) holds CLiENT's own layers
(model/network.py: TargetDenormalization, QuadraticBackground, RegionWall, SmoothStep; the
activations of model/activations.py), and Keras can only rebuild it with their Python code. The
export is the same function as a TensorFlow SavedModel, which TensorFlow alone loads and calls:

    served = tf.saved_model.load("OUTPUT/saved_model")
    loglkl = served.serve(x)       # x: (n, ndim), the parameters in emulator.json's order

By default the exported function is the network inside the region of its boosts
(RUN/boost/region_it_M.npz, M the largest at most N; the region CLiENT's own consumers use), as
model.network.restrict_model builds it: the network where the region holds, the region's wall
outside. A network trained on boosts has maxima beyond its region; `--no-region` exports the bare
network. A combined emulator (client.py combine) carries each part's walls inside its network.

OUTPUT is a new directory:
  saved_model/      the SavedModel; its endpoint `serve` takes (n, ndim) at `--dtype`, returns (n, 1)
  emulator.json     parameters (order, labels, box), the objective's convention, a reference value
                    near the peak, the source (run, network, hashes, region), the check below
  README.md         how to call it, in a few lines
The export is reloaded in a separate Python process that does not import CLiENT and compared with
the source at points of the run's training set; the largest difference is in emulator.json, and an
export that differs by more than the tolerance is refused.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

# Allowed |export - source|: absolute + relative * |source|. TensorFlow orders the operations of a
# traced function differently from eager ones, so at float32 values of 1e5 and more (a region's
# wall, a corner of the box) the two differ by a few units in the last place: 0.3 at -590 510.
TOLERANCE = dict(float32=(1e-3, 2e-5), float64=(1e-8, 1e-12))
NEAR = 50.0                                      # the check's "near the peak": within this of the best

CHECK = """
import json, sys
import numpy as np
import tensorflow as tf
served = tf.saved_model.load(sys.argv[1])
x = np.load(sys.argv[2])
print(json.dumps(served.serve(tf.constant(x)).numpy().ravel().tolist()))
"""


def source_model(snapshot, target, dtype="float32"):
    """(model, region entry or None): a load_snapshot's network at `dtype`, inside its region."""
    from model.network import copy_model_dtype, restrict_model

    if snapshot.region is None:
        return copy_model_dtype(snapshot.model, dtype), None
    return restrict_model(snapshot.model, snapshot.region, dtype=dtype), dict(target["region"], cells=len(snapshot.region))


def check_points(run, iteration, names, n=512, seed=1):
    """Points to compare the export with the source: the best training points of network
    `iteration` and some spread over the run's training set."""
    import pandas as pd

    data = pd.read_csv(Path(run) / "training_data" / f"data_it_{iteration}.csv")
    data = data[np.isfinite(data["loglkl"])]
    best = data.nlargest(n // 2, "loglkl")
    rest = data.sample(min(n - len(best), len(data)), random_state=seed)
    return pd.concat([best, rest])[names].to_numpy(np.float64), float(data["loglkl"].max())


def export(run, output, iteration=None, region=True, dtype="float32", log=print):
    """Write the export of network `iteration` (default: the latest) of `run` to `output`."""
    from run.run import iterations
    from sampling.targets import load_snapshot

    # Absolute: the check below runs in the staging directory.
    run, output = Path(run).absolute(), Path(output).absolute()
    if dtype not in TOLERANCE:
        raise ValueError(f"dtype must be one of {sorted(TOLERANCE)}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} exists and is not empty")
    models = iterations(run / "trained_models", "model_it_*.keras")
    if not models:
        raise ValueError(f"{run}: no trained network")
    iteration = max(models) if iteration is None else int(iteration)
    if iteration not in models:
        raise ValueError(f"{run}: no network {iteration}")
    snapshot, target = load_snapshot(run, iteration, "run" if region else None)
    metadata = snapshot.metadata
    names = metadata.param_names
    x, best = check_points(run, iteration, names)
    model, region_entry = source_model(snapshot, target, dtype=dtype)
    # Built beside the output and renamed last: an output that exists is a whole, checked export.
    staging = output.with_name(f".{output.name}.partial-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    try:
        record = write_checked(model, staging, run, iteration, x.astype(dtype), best, names, metadata, region_entry, target, dtype)
        staging.replace(output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    check = record["check"]
    log(f"exported network {iteration} of {run.name} ({'inside its region' if region_entry else 'bare'}, {dtype}) to "
        f"{output}: largest difference from the source "
        f"{check['largest_difference_near_the_peak'] if check['largest_difference_near_the_peak'] is not None else float('nan'):.2g} "
        f"within {NEAR:g} of the best ({check['points_near_the_peak']} points), {check['largest_difference']:.2g} over all {check['points']}")
    return record


def write_checked(model, output, run, iteration, x, best, names, metadata, region_entry, target, dtype):
    """The export in `output`, once reloaded without CLiENT and compared with the source."""
    import tensorflow as tf
    from run.storage import write_json

    model.export(str(output / "saved_model"), verbose=False,
                 input_signature=[tf.TensorSpec((None, len(names)), dtype)])
    source = np.concatenate([model(tf.constant(x[i:i + 4096]), training=False).numpy().ravel()
                             for i in range(0, len(x), 4096)]).astype(float)
    np.save(output / "check_points.npy", x)
    try:
        done = subprocess.run([sys.executable, "-I", "-c", CHECK, str(output / "saved_model"), str(output / "check_points.npy")],
                              capture_output=True, text=True, cwd=output, check=True)
    finally:
        (output / "check_points.npy").unlink(missing_ok=True)
    exported = np.asarray(json.loads(done.stdout.strip().splitlines()[-1]), float)
    difference = np.abs(exported - source)
    absolute, relative = TOLERANCE[dtype]
    excess = difference - (absolute + relative * np.abs(source))
    near = source >= best - NEAR
    worst = float(difference.max())
    worst_near = float(difference[near].max()) if near.any() else None
    if not np.all(excess <= 0):
        k = int(np.argmax(excess))
        raise RuntimeError(f"the export differs from the source by {difference[k]:.3g} at a value of {source[k]:.6g} "
                           f"(allowed {absolute:g} + {relative:g} x |value|); refused, and nothing is written")

    identity = json.loads((run / "objective_identity.json").read_text())["identity"] if (run / "objective_identity.json").exists() else {}
    combined = json.loads((run / "combine.json").read_text()) if (run / "combine.json").exists() else None
    record = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        call="tf.saved_model.load(DIR / 'saved_model').serve(x): x of shape (n, ndim) in `parameters` order, "
             f"dtype {dtype}; returns (n, 1)",
        parameters=[dict(name=p.name, label=p.label, lower=p.lower, upper=p.upper) for p in metadata.parameters],
        dtype=dtype,
        output="the run's objective, as CLiENT trained on it: " + str(identity.get("objective", "not recorded"))
               + f" (likelihood wrapper: {identity.get('wrapper', 'not recorded')}). With the Cobaya wrapper this is "
                 "Cobaya's log-posterior: the log-likelihood plus the log-prior of the run's Cobaya priors.",
        reference=best,
        reference_note="the best value in the network's training set: subtract it before summing many values in single "
                       "precision (BEST does), and add it back afterwards",
        region=region_entry,
        region_note=("inside the region the network's value; outside it the region's wall, "
                     "floor - 1e4 (rho - edge)^2, far below the peak (the network is not used there)")
        if region_entry else "the bare network over the whole box: a network trained on boosts can have false maxima "
                             "outside the region of its boosts",
        source=dict(run=str(run), iteration=iteration, model_sha256=target["model_sha256"],
                    metadata_sha256=target["metadata_sha256"], combined=combined),
        check=dict(points=int(len(x)), points_near_the_peak=int(near.sum()), near=NEAR,
                   largest_difference_near_the_peak=worst_near, largest_difference=worst,
                   tolerance=dict(absolute=absolute, relative=relative),
                   note="reloaded with TensorFlow alone, in a process that does not import CLiENT; near the peak: "
                        f"within {NEAR:g} of the reference value"))
    write_json(output / "emulator.json", record)
    (output / "README.md").write_text(readme(record, names))
    return record


def readme(record, names):
    near = record["check"]["largest_difference_near_the_peak"]
    near_text = "(no check point there)" if near is None else f"{near:.2g}"
    box = "\n".join(f"| {i} | `{p['name']}` | {p['label']} | {p['lower']:g} | {p['upper']:g} |"
                    for i, p in enumerate(record["parameters"]))
    return f"""# A CLiENT emulator, exported

Network {record['source']['iteration']} of `{Path(record['source']['run']).name}`, as a TensorFlow SavedModel.
It needs TensorFlow only, not CLiENT.

```python
import numpy as np, tensorflow as tf
emulator = tf.saved_model.load("saved_model")
x = np.array([[...]], dtype=np.{record['dtype']})   # shape (n, {len(names)}), the parameters in the order below
loglkl = emulator.serve(tf.constant(x)).numpy()[:, 0]
```

**Output.** {record['output'][0].upper() + record['output'][1:]}

**Region.** {record['region_note'].capitalize()}.

**Box.** Outside the box below the emulator was never trained: treat it as outside the prior.

**Sums in single precision.** Subtract the reference value {record['reference']:.6g} before
summing many values in single precision, as BEST does, and add it back to the result.

| # | name | label | lower | upper |
|---|---|---|---|---|
{box}

`emulator.json` has the same information, the source's hashes and the check against the source:
largest difference {near_text} within {record['check']['near']:g} of the reference value, reloaded
with TensorFlow alone.
"""


def main(argv=None):
    p = argparse.ArgumentParser(prog="client.py export", description=__doc__.split("\n\n")[0])
    p.add_argument("run", type=Path, help="the run directory (a combined emulator too)")
    p.add_argument("output", type=Path, help="the new directory of the export")
    p.add_argument("--iteration", type=int, help="the network (default: the latest)")
    p.add_argument("--no-region", action="store_true", help="the bare network over the whole box")
    p.add_argument("--dtype", choices=sorted(TOLERANCE), default="float32")
    args = p.parse_args(argv)
    try:
        export(args.run, args.output, args.iteration, region=not args.no_region, dtype=args.dtype)
    except ValueError as error:
        raise SystemExit(f"client.py export: {error}")


if __name__ == "__main__":
    main()
