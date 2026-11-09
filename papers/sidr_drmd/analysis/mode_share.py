"""How a run's training points divide between two sides of one parameter, network by network:

    python analysis/mode_share.py results/<timestamp>_<RUN> --parameter logGeff --split -2.5

For each iteration N, the points it added (training_data/data_it_N.csv minus data_it_{N-1}.csv: the
design, a tempered batch, or a boost) are divided at `--split`. Counted on each side: the points
within `--depth` of the run's best true value so far (default T d / 2, where a tempered batch at
temperature T sits in d parameters), and within 10 of it; the best true value on each side; the
upper side's share of the points near the posterior.

Made for the SIDR runs (20 October 2026): the criterion that hands over to the boosts measures
the network at its own tempered batch, which is mostly in the main mode, and cannot see a second
mode that gets too few points. In the kit's SINU run (results/20261013_134541_e2e_sinu16) the strong
mode (logGeff > -2.5) had 6 to 18% of the near-posterior points of the batches of networks 5 to
16, and 25 to 54% from network 17 on (its boosts, which split their points over the modes: 49 to
52%). Reads only the run's files; no likelihood, no network.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", type=Path)
    p.add_argument("--parameter", required=True)
    p.add_argument("--split", type=float, required=True, help="the upper side is parameter > split")
    p.add_argument("--depth", type=float, help="near the posterior: within this of the best (default T d / 2)")
    args = p.parse_args()

    files = sorted(args.run.glob("training_data/data_it_*.csv"), key=lambda f: int(re.findall(r"\d+", f.name)[-1]))
    if not files:
        raise SystemExit(f"{args.run}: no training_data/data_it_*.csv")
    meta = json.loads((args.run / "metadata.json").read_text())
    ndim = len(meta["param_names"]) if "param_names" in meta else len(meta["parameters"])
    import yaml

    temperature = float(yaml.safe_load((args.run / "client.yaml").read_text())["sampling"]["temperature"])
    depth = args.depth if args.depth is not None else temperature * ndim / 2
    print(f"{args.run.name}: {args.parameter} > {args.split:g} against <= {args.split:g}; near the posterior: "
          f"within {depth:g} of the best so far (T = {temperature:g}, d = {ndim})")
    print(f"{'net':>3} {'added':>6} {'up':>6} {'up near':>7} {'up <10':>6} {'low near':>8} {'low <10':>7} "
          f"{'up share':>8} {'best up':>10} {'best low':>10}")
    previous = 0
    best = -np.inf
    for f in files:
        n = int(re.findall(r"\d+", f.name)[-1])
        data = pd.read_csv(f)
        new = data.iloc[previous:]
        previous = len(data)
        value = pd.to_numeric(data["loglkl"], errors="coerce")
        finite = value[np.isfinite(value) & (value > -1e29)]
        best = max(best, float(finite.max()))
        v = pd.to_numeric(new["loglkl"], errors="coerce")
        up = new[args.parameter] > args.split
        near, close = v > best - depth, v > best - 10
        un, ln = int((up & near).sum()), int((~up & near).sum())
        whole = data[args.parameter] > args.split
        best_up = value[whole].max()
        best_low = value[~whole].max()
        print(f"{n:3d} {len(new):6d} {int(up.sum()):6d} {un:7d} {int((up & close).sum()):6d} {ln:8d} "
              f"{int((~up & close).sum()):7d} {un / max(un + ln, 1):8.2f} {best_up:10.1f} {best_low:10.1f}")


if __name__ == "__main__":
    main()
