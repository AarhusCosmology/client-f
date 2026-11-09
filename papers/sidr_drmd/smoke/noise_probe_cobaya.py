"""How noisy is the true log-posterior of a Cobaya model along short lines through a given point? (23 October 2026)

    python smoke/noise_probe_cobaya.py smoke/noise_variants.json LABEL [--workers 46] [--only LIKELIHOOD ...]

The measurement of true_profiles/noise_probe.py, which needs a CLiENT run, made for any Cobaya file and any point. It
evaluates the true log-posterior at `points` equally spaced points along lines of +-`span` sigma through the centre.
There is one line per varied parameter, with the others fixed, and `random` lines in random directions in which every
varied parameter moves. The centre itself is evaluated `repeats` times, to show whether the value is deterministic.

The noise of a line is the rms of its residuals about a polynomial of degree 8 in the line coordinate, with the fit's
degrees of freedom removed, as in the SIDR + DRMD paper (Appendix, "The noise floor of the true likelihood"). The
quartic of noise_probe.py is reported as well.

smoke/noise_variants.json is made on the laptop by results/20261023_noise_variants/make_variants.py. For each variant
label it holds the Cobaya file, the centre (every sampled parameter), the sigma of every varied parameter and their
names. A line that leaves the prior is not evaluated. Writes logs/noise_probe_<label>_<job>.json.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

_MODEL = None


def _init(info):
    global _MODEL
    from cobaya.model import get_model
    _MODEL = get_model(info)


def _evaluate(point):
    r = _MODEL.logposterior(point, make_finite=False)
    return float(r.logpost)


def line_noise(s, values, degree):
    """rms about a polynomial of the given degree in the line coordinate; None if a value is not finite."""
    values = np.asarray(values, float)
    if not np.isfinite(values).all() or len(s) <= degree + 1:
        return None
    residual = values - np.polyval(np.polyfit(s, values, degree), s)
    return float(np.sqrt(np.sum(residual ** 2) / (len(s) - degree - 1)))


def lines_of(spec, args):
    """[(name, unit direction over the varied parameters)]: one per parameter, then the random ones."""
    vary = spec["vary"]
    n = len(vary)
    rng = np.random.default_rng(args.seed)
    out = [(name, np.eye(n)[k]) for k, name in enumerate(vary)]
    for k in range(args.random):
        d = rng.normal(size=n)
        out.append((f"random {k + 1}", d / np.linalg.norm(d)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("variants", type=Path)
    p.add_argument("label")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--points", type=int, default=21)
    p.add_argument("--span", type=float, default=0.25, help="half-length of a line in sigmas")
    p.add_argument("--random", type=int, default=4, help="random directions in which every varied parameter moves")
    p.add_argument("--repeats", type=int, default=3, help="evaluations of the centre itself")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--only", nargs="+", help="keep only these likelihoods (a test on the laptop)")
    p.add_argument("--lines", type=int, help="the first N lines only (a test)")
    args = p.parse_args()

    spec = json.loads(args.variants.read_text())["variants"][args.label]
    info = yaml.safe_load(Path(spec["cobaya"]).read_text())
    info.pop("sampler", None)
    info.pop("output", None)
    if args.only:
        info["likelihood"] = {k: v for k, v in info["likelihood"].items() if k in args.only}
    sampled = {k: v["prior"] for k, v in info["params"].items() if isinstance(v, dict) and "prior" in v}
    missing = sorted(set(sampled) - set(spec["centre"]))
    if missing:
        raise SystemExit(f"{args.label}: the centre has no value for {missing}")
    centre = {k: float(spec["centre"][k]) for k in sampled}
    vary = spec["vary"]
    sigma = np.array([spec["sigma"][k] for k in vary])
    lower = np.array([sampled[k].get("min", -np.inf) for k in vary])
    upper = np.array([sampled[k].get("max", np.inf) for k in vary])
    c0 = np.array([centre[k] for k in vary])

    s = np.linspace(-args.span, args.span, args.points)
    lines = lines_of(spec, args)[:args.lines] if args.lines else lines_of(spec, args)
    points, where = [dict(centre) for _ in range(args.repeats)], []
    for label, d in lines:
        xs = [c0 + x * d * sigma for x in s]
        if all(np.all(x >= lower) and np.all(x <= upper) for x in xs):
            where.append((label, len(points)))
            points += [dict(centre, **dict(zip(vary, map(float, x)))) for x in xs]
        else:
            where.append((label, None))

    import classy
    print(f"== {args.label}: {spec['cobaya']}; {len(vary)} varied parameters, {len(lines)} lines of {args.points} points over "
          f"+-{args.span:g} sigma, the centre {args.repeats} times: {len(points)} evaluations on {args.workers} processes; "
          f"classy from {classy.__file__}", flush=True)
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(info,)) as pool:
        values = list(pool.map(_evaluate, points, chunksize=1))
    seconds = time.time() - started

    repeats = values[:args.repeats]
    out_lines = {}
    for label, at in where:
        if at is None:
            out_lines[label] = dict(outside_prior=True)
            continue
        v = values[at:at + args.points]
        out_lines[label] = dict(values=v, noise=line_noise(s, v, 8), noise_quartic=line_noise(s, v, 4),
                                variation=float(np.nanmax(v) - np.nanmin(v)) if np.isfinite(v).any() else None)
    rand = [out_lines[k]["noise"] for k in out_lines if k.startswith("random") and out_lines[k].get("noise") is not None]
    summary = dict(noise_all_parameters_moving=float(np.median(rand)) if rand else None,
                   noise_random=rand, centre_spread=float(np.ptp(repeats)) if len(repeats) > 1 else None,
                   centre=float(np.mean(repeats)) if repeats else None)
    print(f"\n== {args.label}: {len(points)} evaluations in {seconds:.0f} s; the centre {summary['centre']:.4f} "
          f"(spread over {args.repeats} evaluations {summary['centre_spread']})")
    print(f"   {'line':24s} {'variation':>10s} {'noise (deg 8)':>14s} {'quartic':>10s}")
    for label, e in out_lines.items():
        if e.get("outside_prior"):
            print(f"   {label:24s} leaves the prior: not evaluated")
        else:
            fmt = lambda x: f"{x:.2e}" if x is not None else "not finite"
            print(f"   {label:24s} {e['variation'] if e['variation'] is not None else float('nan'):10.3f} "
                  f"{fmt(e['noise']):>14s} {fmt(e['noise_quartic']):>10s}")
    if rand:
        print(f"   with every varied parameter moving: {summary['noise_all_parameters_moving']:.3f} (median of {len(rand)})")
    Path("logs").mkdir(exist_ok=True)
    out = Path(f"logs/noise_probe_{args.label}_{os.environ.get('SLURM_JOB_ID', 'local')}.json")
    out.write_text(json.dumps(dict(label=args.label, cobaya=spec["cobaya"], description=spec.get("description"),
                                   centre=centre, vary=vary, sigma=spec["sigma"], span=args.span, points=args.points,
                                   s=s.tolist(), seconds=seconds, workers=args.workers, classy=classy.__file__,
                                   only=args.only, repeats=repeats, lines=out_lines, summary=summary), indent=1))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
