"""A profile YAML for `client.py profiles` with an explicit range for every parameter.

    python true_profiles/make_config.py results/<run> --iteration 25 --out runs/<RUN>/profiles_all.yaml
        [--parameters a b c] [--points 15] [--sigmas 3] [--workers 46] [--chain FILE] [--bounds FILE]
        [--range NAME LOW HIGH ...] [--region run|EDGE|none]

Why explicit ranges: without them the profile code first scans the network over the whole
training box to find where each profile falls by one sigma. A network trained on boosts is
only good where the posterior is (acquisition.boost.region), so that scan would be of the
part it does not know. Here the range of a parameter is read off the T = 1 chain of the
run's report instead (analysis/posterior_it_N.npz of --iteration, unless --chain): the quantiles
that +-`sigmas` are for a Gaussian, widened by 5% of their distance each side, cut at the
training box, and moved onto an edge of the box that is nearer than 3% of the range.
A chain of a region with several ellipsoids that does not record how it was sampled is not
used (the shares of its modes, and with them the quantiles, may be those of its start:
analysis/chain_sampling.py); the report samples the network anew.
--range NAME LOW HIGH sets one by hand (CAMB's result at a neutrino mass of exactly zero is
not the limit of small masses: a range that would start on that edge starts just inside).

--region is the profile setting `region` (CLiENT's profiles/region.py): where a network
trained on boosts is searched. `run`, the default, is inside the region the run saved for it;
a number puts the edge at that many times the reach of the boost points (1: where the boost
points are; the nodes of a profile over +-3 sigma lie at about a third of the reach in 49
parameters); `none` is the bare network over the whole box, which has maxima the truth lacks.

The other settings are for a likelihood of about 50 parameters on one 48-core node; the
comments in the file it writes say what each is. Away from the cluster, where only the run's
analysis/ folder is at hand, --bounds runs/<RUN>/domain_bounds.json stands in for the run's
metadata.json (the launcher makes the training box of a run from that file).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from chain_sampling import chain_sampling  # noqa: E402


def network_chain(analysis, iteration):
    """The chain of the network the profiles start from: another network's posterior can be
    shifted or narrower, and the nodes would miss part of this one's."""
    chain = analysis / f"posterior_it_{iteration}.npz"
    if not chain.exists():
        raise SystemExit(f"no {chain.name} in {analysis}: run the report first (analysis/report.py), or name a chain with --chain")
    return chain


def ranges(chain, names, bounds, sigmas, parameters):
    saved = np.load(chain, allow_pickle=True)
    if [str(n) for n in saved["names"]] != names:
        raise SystemExit(f"{chain}: its parameters are not the run's")
    x = np.asarray(saved["x"], np.float64)
    tail = 0.5 * math.erfc(sigmas / math.sqrt(2))
    out, notes = {}, {}
    for name in parameters:
        k = names.index(name)
        low, high = np.quantile(x[:, k], [tail, 1 - tail])
        pad = 0.05 * (high - low)
        low, high = max(bounds[k, 0], low - pad), min(bounds[k, 1], high + pad)
        at = []
        if low - bounds[k, 0] < 0.03 * (high - low):
            low, at = bounds[k, 0], at + ["lower edge of the box"]
        if bounds[k, 1] - high < 0.03 * (high - low):
            high, at = bounds[k, 1], at + ["upper edge of the box"]
        if not high > low:
            raise SystemExit(f"{name}: the chain's {sigmas:g} sigma range has no width (at {low:g}); give it with --range {name} LOW HIGH")
        digits = 4 - int(math.floor(math.log10(high - low)))             # five significant digits of the range
        low = max(bounds[k, 0], math.floor(low * 10 ** digits) / 10 ** digits) if "lower edge of the box" not in at else low
        high = min(bounds[k, 1], math.ceil(high * 10 ** digits) / 10 ** digits) if "upper edge of the box" not in at else high
        out[name] = [float(low), float(high)]
        notes[name] = (float(x[:, k].mean()), float(x[:, k].std()), at)
    return out, notes


def region_setting(text):
    """--region as the YAML has it: `run`, a positive number, or `null` for none."""
    text = str(text).strip().lower()
    if text == "run":
        return "run"
    if text in ("none", "null"):
        return "null"
    try:
        edge = float(text)
    except ValueError:
        edge = float("nan")
    if not (math.isfinite(edge) and edge > 0):
        raise SystemExit(f"--region {text}: run, a positive number of reaches, or none")
    return f"{edge:g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path, help="results/<run>: its metadata.json and analysis/ are read")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True, help="the network the profiles start from")
    parser.add_argument("--parameters", nargs="*", help="default: every parameter of the run")
    parser.add_argument("--points", type=int, default=15)
    parser.add_argument("--sigmas", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=46)
    parser.add_argument("--chain", type=Path)
    parser.add_argument("--bounds", type=Path, help="the run's domain_bounds.json, instead of its metadata.json")
    parser.add_argument("--range", nargs=3, action="append", default=[], metavar=("NAME", "LOW", "HIGH"),
                        help="a range given by hand, inside the training box (may be repeated)")
    parser.add_argument("--region", default="run",
                        help="where a boost network is searched: run (default), a number of reaches, or none")
    parser.add_argument("--maxfun", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--tolerance", type=float, help="corrected_tol; default: the method's own, 1e-2")
    parser.add_argument("--audit-tolerance", type=float, default=0.01,
                        help="Delta chi-squared by which the best saved training values may differ from now (0.01); larger "
                             "for a study whose objective is deliberately not the run's (CLIENT_CAMB_ACCURACY_TARGET)")
    args = parser.parse_args()
    region = region_setting(args.region)
    chain = args.chain or network_chain(args.run / "analysis", args.iteration)
    stop, warning = chain_sampling(chain, args.run)
    if stop:
        raise SystemExit(f"{chain} {stop}. Its quantiles may not be the posterior's: run the report again "
                         "(analysis/report.py), which samples the network anew.")
    if warning:
        print(f"WARNING: {chain.name} {warning}. The ranges are read off it as it is.")
    if args.bounds:
        box = json.loads(args.bounds.read_text())
        names = [str(n) for n in np.load(chain, allow_pickle=True)["names"]]
        if set(names) != set(box):
            raise SystemExit(f"{args.bounds}: its parameters are not the chain's")
        bounds = np.array([box[n] for n in names], float)
    else:
        metadata = json.loads((args.run / "metadata.json").read_text())
        names = [p["name"] for p in metadata["parameters"]]
        bounds = np.array([[p["lower"], p["upper"]] for p in metadata["parameters"]], float)
    parameters = args.parameters or names
    unknown = sorted(set(parameters) - set(names))
    if unknown:
        raise SystemExit(f"not parameters of the run: {unknown}")
    found, notes = ranges(chain, names, bounds, args.sigmas, parameters)
    for name, low, high in args.range:
        if name not in found:
            raise SystemExit(f"--range {name}: not one of the parameters profiled")
        k = names.index(name)
        low, high = float(low), float(high)
        if not bounds[k, 0] <= low < high <= bounds[k, 1]:
            raise SystemExit(f"--range {name}: [{low:g}, {high:g}] is not inside the training box [{bounds[k, 0]:g}, {bounds[k, 1]:g}]")
        found[name], notes[name] = [low, high], notes[name][:2] + (["range set by hand"],)
    n = len(names)
    maxfun = args.maxfun or 6 * (n + 2)
    nodes = len(parameters) * args.points
    budget = args.budget or int(math.ceil(nodes * 4 * (n + 2) / 1000) * 1000)
    lines = [
        f"# True 1D profiles of {len(parameters)} of the {n} parameters of {args.run.name},",
        f"# starting from network {args.iteration}. Written by true_profiles/make_config.py; the ranges are the",
        f"# +-{args.sigmas:g} sigma quantiles of {chain.name}, cut at the training box.",
        "#",
        "#   sbatch true_profiles/submit_profiles.sh results/<run> <this file> results_profiles/<name>",
        "#",
        f"iteration: {args.iteration}",
        f"profile_points: {args.points}          # nodes per parameter, ends included: {nodes} nodes in all",
        f"workers: {args.workers}                 # backend processes on the node; may be changed when resuming",
        "retrain: false              # the saved network as it is",
        "added_points: 0",
        "surrogate_optimizer: hybrid",
        f"corrected_tol: {args.tolerance if args.tolerance is not None else 'null'}"
        "         # null: 1e-2 in the log-objective. Not below twice the noise floor of the likelihood",
        "chain_length: 2             # a cold start after every two chained nodes: keeps the processes busy",
        f"maxfun: {maxfun}                 # true calls a node may use: six gradient rounds of {n + 1} (the default 150 is for ~16 parameters)",
        f"total_call_budget: {budget}   # of the whole study, resumed attempts included (four gradient rounds a node)",
        "audit_points: 3             # best saved training values recomputed before anything else",
        f"audit_tolerance: {args.audit_tolerance:g}",
        f"region: {region}" + " " * max(1, 17 - len(region)) + "# a network trained on boosts is searched inside the region of its boosts"
        + {"run": " (the run's own edge)", "null": ": NOT applied, the bare network"}.get(region, f", edge at {region} times their reach"),
        "hybrid_walkers: 8",
        "seed: 42",
        "parameters:",
    ]
    lines += [f"  - {name}" for name in parameters]
    lines.append("ranges:")
    width = max(len(name) for name in parameters)
    for name in parameters:
        mean, sd, at = notes[name]
        low, high = found[name]
        lines.append(f"  {name + ':':{width + 1}s} [{low:.10g}, {high:.10g}]   # posterior {mean:.5g} +- {sd:.2g}"
                     + (f"; {' and '.join(at)}" if at else ""))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"written: {args.out} ({len(parameters)} parameters x {args.points} points = {nodes} nodes; maxfun {maxfun}, budget {budget})")


if __name__ == "__main__":
    main()
