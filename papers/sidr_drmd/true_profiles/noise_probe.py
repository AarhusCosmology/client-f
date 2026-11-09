"""How noisy is the true objective at the scale of the profile stage's probes?

    python true_profiles/noise_probe.py results/<run> [--workers 46] [--points 21] [--span 0.25]

The residual Newton stage of the true profiles measures the gradient of (truth - network) by
differences over 0.25 posterior sigma, in every free direction, and stops when that gradient
predicts a gain below `corrected_tol`. Noise eps in the true value puts a floor of about
n eps^2 / step^2 under that prediction (n free parameters), so the stage cannot stop at a
tolerance below it and spends its budget on noise instead. Its defaults (step 0.25, tolerance
1e-2 in the log-objective) were set for a noise of 1e-3 to 2.5e-3 in 16 parameters.

This evaluates the truth along short lines through the posterior mean: one per parameter
(the others fixed) and a few random directions in which every parameter moves, each
+-span posterior sigma in `points` steps. Along a line that short the objective is a smooth
curve plus the noise, so the rms about a quartic is the noise. It also recomputes the best
saved training values, which is the first thing a profile study does.

Needs the run's report (analysis/report.json: the posterior mean and width). Writes
results/<run>/analysis/noise_probe.json. Run it through submit_probe.sh: the backend processes
are the profile study's own (cluster_backend.py).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import client_paths  # noqa: E402,F401  CLiENT's hpc/ tools (cluster_backend)


def read_run(run, iteration=None):
    """names, bounds, wrapper, backend input and the best saved rows of a run directory."""
    import re

    import yaml

    metadata = json.loads((run / "metadata.json").read_text())
    names = [p["name"] for p in metadata["parameters"]]
    bounds = np.array([[p["lower"], p["upper"]] for p in metadata["parameters"]], float)
    configs = list(run.glob("*.yaml")) + list(run.glob("*.yml"))
    if len(configs) != 1:
        raise SystemExit(f"{run}: expected one run YAML, found {len(configs)}")
    settings = yaml.safe_load(configs[0].read_text())
    backend = run / "likelihood_input" / Path(settings["likelihood"]["input"]).name
    if iteration is None:
        iteration = max(int(m.group(1)) for p in (run / "training_data").glob("data_it_*.csv")
                        if (m := re.fullmatch(r"data_it_(\d+)\.csv", p.name)))
    data = np.loadtxt(run / "training_data" / f"data_it_{iteration}.csv", delimiter=",", skiprows=1, ndmin=2, dtype=np.float32).astype(float)
    return names, bounds, settings["likelihood"]["wrapper"], backend, data, iteration


def posterior(run, names):
    """(mean, sd, network) of the last network the run's report sampled."""
    report = run / "analysis" / "report.json"
    if not report.exists():
        raise SystemExit(f"no {report}: run the report first (analysis/submit_report.sh), it has the posterior's mean and width")
    report = json.loads(report.read_bytes().rstrip(b"\x00"))
    if report["parameters"] != names:
        raise SystemExit("the report's parameters are not the run's")
    last = max(report["posterior"]["networks"], key=lambda n: n["iteration"])
    return np.array(last["mean"], float), np.array(last["sd"], float), last["iteration"]


def line_noise(s, values):
    """rms about a quartic in the line coordinate, the largest step between neighbours beyond
    that fit, and the variation along the line; None where a value is not finite."""
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        return dict(noise=None, largest_step=None, variation=None, not_finite=int(np.sum(~np.isfinite(values))))
    residual = values - np.polyval(np.polyfit(s, values, 4), s)
    return dict(noise=float(np.sqrt(np.sum(residual ** 2) / (len(s) - 5))), largest_step=float(np.abs(np.diff(residual)).max()),
                variation=float(values.max() - values.min()), not_finite=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--iteration", type=int, help="whose training file supplies the saved values to recompute (default: the latest)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--points", type=int, default=21)
    parser.add_argument("--span", type=float, default=0.25, help="half-length of a line in posterior sigmas (the probes' step)")
    parser.add_argument("--random", type=int, default=4, help="random directions in which every parameter moves")
    parser.add_argument("--labels", type=int, default=3, help="best saved training values to recompute")
    parser.add_argument("--step", type=float, default=0.25, help="the Newton stage's forward-difference step, for the floor")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    names, bounds, wrapper, backend, data, iteration = read_run(run, args.iteration)
    mean, sd, network = posterior(run, names)
    n = len(names)
    s = np.linspace(-args.span, args.span, args.points)
    rng = np.random.default_rng(args.seed)
    directions = [(name, np.eye(n)[k]) for k, name in enumerate(names)]
    for k in range(args.random):
        d = rng.normal(size=n)
        directions.append((f"random {k + 1}", d / np.linalg.norm(d)))        # unit length: the move a probe makes
    centre = np.clip(mean, bounds[:, 0], bounds[:, 1])
    points = np.array([centre + x * d * sd for _, d in directions for x in s])
    inside = np.all((points >= bounds[:, 0]) & (points <= bounds[:, 1]), axis=1)
    best = np.argsort(-data[:, -1], kind="stable")[:args.labels]
    saved = np.clip(data[best, :-1], bounds[:, 0], bounds[:, 1])

    import cluster_backend

    cluster_backend.install()
    from profiles.evaluation import BackendPool

    target = cluster_backend.camb_accuracy_target()
    print(f"[probe] {run.name}: {n} parameters, posterior of network {network}; {len(directions)} lines of {args.points} points over "
          f"+-{args.span:g} sigma and {len(best)} saved values, on {args.workers} backend processes"
          + (f"; CAMB AccuracyTarget {target} (CLIENT_CAMB_ACCURACY_TARGET)" if target is not None else ""), flush=True)
    started = time.time()
    pool = BackendPool(args.workers, wrapper=wrapper, input_path=str(backend), names=names, bounds=bounds, module=None, expected_hash=None)
    try:
        futures = [pool.submit(x) for x in saved] + [pool.submit(x) if ok else None for x, ok in zip(points, inside)]
        values = np.array([np.nan if f is None else f.result() for f in futures], float)
    finally:
        pool.close()
    seconds = time.time() - started
    recomputed, values = values[:len(best)], values[len(best):].reshape(len(directions), args.points)

    print(f"\n[probe] {len(futures)} evaluations in {seconds:.0f} s, building the likelihoods included")
    print("[probe] saved training values against the same points now (the profile study's audit allows 0.005 here):")
    labels = []
    for row, old, new in zip(best, data[best, -1], recomputed):
        labels.append(dict(row=int(row), saved=float(old), now=float(new), difference=float(new - old)))
        print(f"   row {row:6d}: saved {old:.4f}, now {new:.4f}, difference {new - old:+.2e}")
    print(f"\n[probe] noise of the true value along each line (rms about a quartic), in log-objective units:")
    print(f"   {'line':28s} {'variation':>10s} {'noise':>10s} {'largest step':>13s}")
    lines = {}
    for (label, _), row, ok in zip(directions, values, inside.reshape(len(directions), args.points)):
        lines[label] = dict(line_noise(s, row), values=[None if not np.isfinite(v) else float(v) for v in row]) if ok.all() else dict(
            noise=None, largest_step=None, variation=None, not_finite=0, outside_box=int(np.sum(~ok)))
        entry = lines[label]
        print(f"   {label:28s} " + (f"{entry['variation']:10.3f} {entry['noise']:10.2e} {entry['largest_step']:13.2e}" if entry["noise"] is not None
                                    else ("    line leaves the training box" if entry.get("outside_box") else f"    {entry['not_finite']} values not finite")))
    together = [lines[label]["noise"] for label, _ in directions[n:] if lines[label]["noise"] is not None]
    alone = {label: lines[label]["noise"] for label, _ in directions[:n] if lines[label]["noise"] is not None}
    summary = dict(noise_all_parameters_moving=float(np.median(together)) if together else None,
                   noisiest_single_parameters=sorted(alone, key=alone.get, reverse=True)[:6])
    if together:
        eps = float(np.max(together))
        floor = (n - 1) * eps ** 2 / args.step ** 2
        summary.update(noise_used=eps, step=args.step, floor_of_predicted_gain=floor, default_tolerance=1e-2,
                       default_settings_can_stop=bool(2 * floor <= 1e-2))
        print(f"\n[probe] with every parameter moving the noise is {np.median(together):.2e} (median of {len(together)} lines; largest {eps:.2e}).")
        print(f"[probe] the residual Newton stage then predicts a gain of about {floor:.2e} from noise alone "
              f"({n - 1} free parameters, step {args.step:g} sigma): corrected_tol should be at least twice that, {2 * floor:.1e}.")
        print("[probe] the default corrected_tol of 1e-2 " + ("is above that: the default settings can stop." if 2 * floor <= 1e-2 else
              "is BELOW that: with the default settings every node would run to its call limit. See the README (a longer step, a larger tolerance)."))
    print("[probe] noisiest single parameters: " + ", ".join(f"{k} {alone[k]:.1e}" for k in summary["noisiest_single_parameters"]))
    out = args.out or run / "analysis" / ("noise_probe.json" if target is None else f"noise_probe_camb_target{target}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(run=run.name, network=network, training_file=iteration, span=args.span, points=args.points,
                                   camb_accuracy_target=target,
                                   s=s.tolist(), seconds=seconds, workers=args.workers, labels=labels, lines=lines, summary=summary), indent=1))
    print(f"[probe] written: {out}")


if __name__ == "__main__":
    main()
