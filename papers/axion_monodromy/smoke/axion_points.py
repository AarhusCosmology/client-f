"""True values of a run's likelihood at given points, compared with values given beside them.

    sbatch smoke/submit_axion_points.sh e2e_axion_tt15 runs/e2e_axion_tt15/student_points.csv

The CSV has one column per sampled parameter of the run (any order), a `group` label and a
`reference` value per row: the value the row is to be compared with (another chain's log-likelihood
in the run's convention, or the run's own true value at a training point, which this job should
reproduce). Every row is evaluated with the run's likelihood and domain, as smoke/axion_probe.py
does; the job prints, per group, the median and spread of (true - reference), and writes
logs/axion_points_<job>.json. True evaluations only, no network. Nothing in runs/ is written.

    python smoke/axion_points.py --from-json logs/axion_points_<job>.json      # the summary again, offline
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent


def read_points(path, names):
    """Rows in the run's parameter order, their groups and references, from the CSV."""
    import pandas as pd

    frame = pd.read_csv(path)
    missing = [n for n in names if n not in frame.columns]
    if missing:
        raise SystemExit(f"{path}: no column for {', '.join(missing)}")
    for column in ("group", "reference"):
        if column not in frame.columns:
            raise SystemExit(f"{path}: no column {column!r}")
    return (frame[names].to_numpy(float), frame["group"].astype(str).tolist(),
            frame["reference"].to_numpy(float))


def summarise(groups, values, reference):
    """Per group, in the order the groups first appear: the rows, the finite true values, and the
    median, 16-84% range, smallest and largest of (true - reference) over them."""
    groups = np.asarray(groups)
    values, reference = np.asarray(values, float), np.asarray(reference, float)
    out = []
    for g in dict.fromkeys(groups.tolist()):
        m = groups == g
        d = (values - reference)[m & np.isfinite(values)]
        row = dict(group=g, rows=int(m.sum()), finite=int(d.size))
        if d.size:
            lo, med, hi = np.percentile(d, [16, 50, 84])
            row.update(median=float(med), p16=float(lo), p84=float(hi), smallest=float(d.min()), largest=float(d.max()))
        out.append(row)
    return out


def report_lines(summary):
    lines = [f"{'group':<28} {'rows':>4} {'finite':>6}   true - reference: {'median':>8} {'16%':>8} {'84%':>8} {'min':>8} {'max':>8}"]
    for s in summary:
        if s["finite"]:
            lines.append(f"{s['group']:<28} {s['rows']:>4} {s['finite']:>6}   {'':>18}{s['median']:>8.2f} {s['p16']:>8.2f} "
                         f"{s['p84']:>8.2f} {s['smallest']:>8.2f} {s['largest']:>8.2f}")
        else:
            lines.append(f"{s['group']:<28} {s['rows']:>4} {s['finite']:>6}   no finite true value")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?")
    p.add_argument("points", nargs="?", help="the CSV of points (see above)")
    p.add_argument("--cobaya", help="a Cobaya input instead of the run's (for a test on an analytic likelihood)")
    p.add_argument("--from-json", help="the summary of a job already made; no evaluation")
    args = p.parse_args()
    if args.from_json:
        d = json.loads(Path(args.from_json).read_text())
        values = np.array([np.nan if v is None else v for v in d["values"]], float)
        print("\n".join(report_lines(summarise(d["groups"], values, d["reference"]))))
        return
    if not (args.run and args.points):
        p.error("a run and a CSV of points, or --from-json")

    import yaml
    from mpi4py import MPI

    sys.path.insert(0, str(HERE))
    import client_paths  # noqa: E402,F401  CLiENT and its hpc/ tools (domain, loading)
    from domain import apply_domain, read_domain  # noqa: E402
    from loading import build_in_turn  # noqa: E402
    from likelihood.cobaya import CobayaLikelihood  # noqa: E402
    from utils.mpi_utils import broadcast_and_evaluate  # noqa: E402

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    job = os.environ.get("SLURM_JOB_ID", "local")
    run = HERE / "runs" / args.run
    cfg = yaml.safe_load((run / "client.yaml").read_text())
    source = Path(args.cobaya) if args.cobaya else HERE / cfg["likelihood"]["input"]
    lik = build_in_turn(comm, lambda: CobayaLikelihood(str(source)), label="[axion points]")
    if (run / "domain_bounds.json").exists() and not args.cobaya:
        apply_domain(lik, read_domain(run / "domain_bounds.json"))
    names = list(lik.param_names)

    rows = groups = reference = None
    if rank == 0:
        rows, groups, reference = read_points(HERE / args.points if not Path(args.points).is_absolute() else Path(args.points), names)
        box = np.array([lik.prior_bounds[n] for n in names], float)
        outside = ~np.all((rows >= box[:, 0]) & (rows <= box[:, 1]), axis=1)
        print(f"== {args.run}: {len(rows)} points from {args.points} on {comm.Get_size()} ranks; "
              f"{int(outside.sum())} outside the run's box", flush=True)

    def protected(target):
        def f(x):
            try:
                return float(target.loglkl(x))
            except Exception:
                return float("nan")
        return f

    t0 = time.perf_counter()
    rows, values = broadcast_and_evaluate(rows, protected(lik))
    if rank == 0:
        values = np.asarray(values, float).reshape(-1)
        print(f"{time.perf_counter() - t0:.0f} s; {int(np.isfinite(values).sum())} of {len(values)} finite\n", flush=True)
        print("\n".join(report_lines(summarise(groups, values, reference))), flush=True)
        out = HERE / "logs" / f"axion_points_{job}.json"
        out.write_text(json.dumps(dict(run=args.run, points=args.points, names=names, rows=np.asarray(rows, float).tolist(),
                                       groups=list(groups), reference=np.asarray(reference, float).tolist(),
                                       values=[None if not np.isfinite(v) else float(v) for v in values]), indent=1) + "\n")
        print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
