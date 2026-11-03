"""Profiles in derived parameters, through a derived-coordinate view of a run.

    # 1. The view: a run-shaped directory in which pivots are replaced by derived ones.
    python client.py derived-view RUN_DIR --params theta_s_100 -o VIEW_DIR
    python client.py derived-view RUN_DIR --params S8,Omega_m --pivots S8=logA,Omega_m=omega_cdm -o VIEW_DIR
    python client.py derived-view RUN_DIR --params theta_s_100 --keep H0 -o VIEW_DIR   # (theta_s, H0) pairs

    # 2. Ordinary profiles of the view, 1D or 2D, with every usual option.
    python client.py profiles VIEW_DIR --params theta_s_100 --output PROFILE_DIR

    # 3. The exact derived values at the final nodes.
    python client.py derived-view --nodes PROFILE_DIR [--workers 4]

See derived/coordinates.py for the construction and docs/derived_parameters.md for its use.
"""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_BACKEND = None


def _initialize(spec):
    global _BACKEND
    from likelihood.base import build_likelihood

    _BACKEND = build_likelihood("derived_coordinates", spec)


def _evaluate(z):
    loglkl, derived, x, residual = _BACKEND.exact(np.asarray(z, float))
    return loglkl, derived.tolist(), x.tolist(), residual.tolist()


def _names(spec):
    from likelihood.base import build_likelihood

    backend = build_likelihood("derived_coordinates", spec)
    try:
        return backend.param_names, backend._inner.derived_names, backend._spec["physical_names"]
    finally:
        backend.close()


def evaluate_nodes(profile_dir, workers=1, log=print):
    """Exact derived values, and the re-evaluated objective, at a profile study's nodes."""
    import pandas as pd

    out = Path(profile_dir).resolve()
    view = Path(json.loads((out / "manifest.json").read_text())["run"])
    summary = json.loads((view / "view.json").read_text())
    spec = str(view / "likelihood_input" / "derived_coordinates.json")

    nodes = []  # (label dict, z, recorded logl)
    if (out / "true_profiles.json").exists():
        result = json.loads((out / "true_profiles.json").read_text())
        for name, record in result["profiles"].items():
            for j, (g, z, logl) in enumerate(zip(record["grid"], record["xopt"], record["logl"])):
                if z is not None:
                    nodes.append(({"profile": name, "node": j, name: g}, z, logl))
    elif (out / "true_surface.json").exists():
        result = json.loads((out / "true_surface.json").read_text())
        a, b = result["names"]
        for i, ga in enumerate(result["axes"][0]):
            for j, gb in enumerate(result["axes"][1]):
                z = result["xopt"][i][j]
                if z is not None:
                    nodes.append(({"profile": f"{a},{b}", "node": [i, j], a: ga, b: gb}, z, result["logl"][i][j]))
    else:
        raise FileNotFoundError(f"no true_profiles.json or true_surface.json in {out}")

    view_names, derived_names, physical_names = _names(spec)
    log(f"Evaluating the exact derived parameters at {len(nodes)} nodes with {workers} worker(s)")
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=_initialize, initargs=(spec,)) as pool:
        values = list(pool.map(_evaluate, [z for _, z, _ in nodes]))

    rows = []
    for (label, z, recorded), (logl, derived, x, residual) in zip(nodes, values):
        row = dict(label)
        row.update(logl_recorded=recorded, logl_reevaluated=logl,
                   max_pivot_residual=float(np.max(np.abs(residual))))
        row.update({f"{n}_exact": v for n, v in zip(derived_names, derived)})
        for target in summary["targets"]:
            if target in label:
                row[f"{target}_offset"] = row[f"{target}_exact"] - label[target]
        row.update({f"x_{n}": v for n, v in zip(physical_names, x)})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "derived_nodes.csv", index=False)

    report = dict(view=str(view), targets=summary["targets"], pivots=summary["pivots"], nodes=len(rows),
                  max_objective_reevaluation_difference=float(np.nanmax(np.abs(
                      table["logl_reevaluated"] - table["logl_recorded"]))))
    for target in summary["targets"]:
        column = f"{target}_offset"
        if column in table:
            offsets = table[column].dropna().to_numpy()
            report[target] = dict(max_abs_offset=float(np.max(np.abs(offsets))),
                                  rms_offset=float(np.sqrt(np.mean(offsets**2))))
    (out / "derived_nodes.json").write_text(json.dumps(report, indent=2))
    log(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", nargs="?", type=Path, help="source run directory")
    parser.add_argument("--params", help="comma-separated derived parameters to make coordinates")
    parser.add_argument("--pivots", help="explicit pivots, e.g. S8=logA,Omega_m=omega_cdm")
    parser.add_argument("--keep", default="", help="sampled parameters that must stay coordinates")
    parser.add_argument("-i", "--iteration", type=int)
    parser.add_argument("-o", "--output", type=Path, help="view directory to create")
    parser.add_argument("--nodes", type=Path, help="profile output of a view: evaluate exact derived values")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    if args.nodes is not None:
        evaluate_nodes(args.nodes, workers=args.workers)
        return
    if args.run is None or args.params is None or args.output is None:
        parser.error("give RUN_DIR, --params and -o (or --nodes PROFILE_DIR)")

    from derived.coordinates import make_view

    targets = [t for t in args.params.split(",") if t]
    pivots = None
    if args.pivots:
        given = dict(item.split("=", 1) for item in args.pivots.split(","))
        if set(given) != set(targets):
            parser.error("--pivots must name a pivot for every --params entry")
        pivots = [given[t] for t in targets]
    summary = make_view(args.run, args.output, targets, pivots=pivots,
                        exclude=[k for k in args.keep.split(",") if k], iteration=args.iteration)
    print(json.dumps({k: summary[k] for k in ("targets", "pivots", "coordinates", "roundtrip")}, indent=2))
    if summary.get("region"):
        print(f"Network {summary['iteration']} of this run was trained on boosts. The region of its boosts "
              f"({summary['source_region']}) is copied into the view as {summary['region']['file']}, and profiles "
              "of the view search the network inside it, as the run's own profiles do (the profile setting "
              "`region`; docs/create_profiles.md, \"Networks trained on boosts\").")
    print(f"View written to {args.output}. Profile it with:\n"
          f"  python client.py profiles {args.output} --params {args.params} --output PROFILE_DIR\n"
          f"then evaluate the exact derived values with:\n"
          f"  python client.py derived-view --nodes PROFILE_DIR")


if __name__ == "__main__":
    main()
