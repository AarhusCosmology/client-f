"""chi^2 of every likelihood at many points, in parallel: one Cobaya model in each worker process.

    python smoke/chi2_points_mp.py POINTS SET [--workers N] [--only LIKELIHOOD ...] [--limit N]

POINTS is a JSON file with sets {SET: {cobaya, job_env, points: [{label, params, logl?, ...}]}}, made by
results/20261024_chi2_compare/make_points.py: smoke/chi2_compare_points.json (four points per data set in four
set-ups: why the preferred self-interaction moves on the H0 World Cup data) and smoke/chi2_profile_points.json
(every node of the earlier runs' true profiles). The shell must have sourced the set's job_env, which puts its
CLASS++ build first on PYTHONPATH; the build is checked with boltzmann_build.location_problem, as launch.py does.
--only keeps the named likelihoods (a test on the laptop; keep CamSpec, or its nuisance parameters go to CLASS).

Each point: chi^2 = -2 log L of every likelihood, their sum, -2 log prior, the log-posterior, and the log-posterior
minus the point's logl where it has one (a profile node must reproduce its own value). Also minus2_logprior_gaussian:
-2 ln pdf of every parameter with a Gaussian prior, one by one (a data prior such as the SH0ES M_B prior of the World
Cup runs is not a likelihood here, and its pull is told from the CamSpec calibrations' only by this). A set with "optimise": [NAME,
...] first maximises the log-posterior over those parameters at each point (Powell, within the prior), the others
fixed; for likelihood nuisance parameters Cobaya keeps the spectra, so that costs no Boltzmann call. Writes
logs/chi2_points_<SET>_<job>.json and prints a table. Made 24 October 2026 (smoke/chi2_breakdown.py, in parallel).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
MODEL = None


def load_info(cobaya, only):
    from cobaya.yaml import yaml_load_file

    info = yaml_load_file(str(HERE / cobaya))
    info.pop("sampler", None)
    info.pop("output", None)
    if only:
        missing = set(only) - set(info["likelihood"])
        if missing:
            raise SystemExit(f"not likelihoods of {cobaya}: {sorted(missing)}")
        info["likelihood"] = {k: v for k, v in info["likelihood"].items() if k in only}
    return info


def start(cobaya, only):
    """Worker initializer: one thread, one Cobaya model."""
    global MODEL
    os.environ["OMP_NUM_THREADS"] = "1"
    from cobaya.model import get_model

    MODEL = get_model(load_info(cobaya, only))


def optimise(params, names, free):
    """Maximise the log-posterior over the parameters `free` at fixed others (Powell, within the prior; a Gaussian
    prior is bounded at +-10 sigma). For likelihood parameters such as CamSpec's nuisances Cobaya keeps the theory
    code's spectra, so this costs no Boltzmann call. Returns the optimised parameters and the number of calls."""
    import numpy as np
    from scipy.optimize import minimize

    bounds = []
    for n in free:
        prior = MODEL.prior.pdf[MODEL.prior.params.index(n)]
        lo, hi = prior.interval(1.0)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            m, sd = prior.mean(), prior.std()
            lo, hi = m - 10 * sd, m + 10 * sd
        bounds.append((float(lo), float(hi)))
    x0 = np.clip([params[n] for n in free], [b[0] for b in bounds], [b[1] for b in bounds])
    calls = [0]

    def minus_logpost(x):
        calls[0] += 1
        q = dict(params, **dict(zip(free, map(float, x))))
        v = MODEL.logposterior({n: q[n] for n in names}, make_finite=True).logpost
        return -float(v)

    best = minimize(minus_logpost, x0, method="Powell", bounds=bounds, options=dict(xtol=1e-4, ftol=1e-6, maxfev=4000))
    return dict(params, **dict(zip(free, map(float, best.x)))), calls[0]


def gaussian_priors(params):
    """-2 ln pdf of each sampled parameter whose prior is a normal distribution, at params."""
    out = {}
    for n, pdf in zip(MODEL.prior.params, MODEL.prior.pdf):
        if getattr(getattr(pdf, "dist", None), "name", None) == "norm":
            out[n] = -2 * float(pdf.logpdf(params[n]))
    return out


def evaluate(task):
    index, point, free = task
    names = list(MODEL.parameterization.sampled_params())
    missing = [n for n in names if n not in point["params"]]
    if missing:
        return dict(index=index, label=point["label"], error=f"no value for {missing}")
    t = time.time()
    params = dict(point["params"])
    extra = {}
    if free:
        try:
            before = MODEL.logposterior({n: params[n] for n in names}, make_finite=False)
            chi2_before = {k: -2 * float(v) for k, v in zip(MODEL.likelihood, before.loglikes)}
            params, calls = optimise(params, names, free)
            extra = dict(optimised=free, optimised_values={n: params[n] for n in free}, optimiser_calls=calls,
                         chi2_before=chi2_before, logpost_before=float(before.logpost))
        except Exception as error:                 # noqa: BLE001 - recorded with the point
            return dict(index=index, label=point["label"], error=f"optimisation: {type(error).__name__}: {error}")
    try:
        r = MODEL.logposterior({n: params[n] for n in names}, make_finite=False)
    except Exception as error:                     # noqa: BLE001 - recorded with the point
        return dict(index=index, label=point["label"], error=f"{type(error).__name__}: {error}")
    if r.loglikes is None or len(r.loglikes) == 0:
        return dict(index=index, label=point["label"], error="outside the prior")
    chi2 = {k: -2 * float(v) for k, v in zip(MODEL.likelihood, r.loglikes)}
    out = dict(index=index, label=point["label"], chi2=chi2, chi2_total=sum(chi2.values()),
               minus2_logprior=-2 * float(sum(r.logpriors)), minus2_logprior_gaussian=gaussian_priors(params),
               logpost=float(r.logpost), seconds=time.time() - t, params={n: float(params[n]) for n in names}, **extra)
    if point.get("logl") is not None:
        out["logl_point"] = float(point["logl"])
        out["logpost_minus_point"] = out["logpost"] - float(point["logl"])
    if point.get("network_logpost") is not None:          # a network fit: truth minus network there
        out["logpost_minus_network"] = out["logpost"] - float(point["network_logpost"])
    for key in ("profile", "coordinate", "note", "network_logpost"):
        if key in point:
            out[key] = point[key]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("points", type=Path)
    p.add_argument("set")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--only", nargs="+")
    p.add_argument("--limit", type=int)
    a = p.parse_args()

    spec = json.loads(a.points.read_text())["sets"][a.set]
    import classy
    sys.path.insert(0, str(HERE))
    import client_paths  # noqa: E402,F401  CLiENT's hpc/ tools
    from boltzmann_build import location_problem

    problem = location_problem("classy", classy.__file__)
    if problem:
        raise SystemExit(f"FATAL: {problem} (source {spec['job_env']} first)")
    points = spec["points"][:a.limit] if a.limit else spec["points"]
    workers = max(1, min(a.workers, len(points)))
    print(f"== {a.set}: {spec['cobaya']}, {len(points)} points on {workers} workers; classy from {classy.__file__}"
          + (f"; optimising {', '.join(spec['optimise'])} at each point" if spec.get("optimise") else ""), flush=True)
    t = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=start, initargs=(spec["cobaya"], a.only)) as pool:
        free = spec.get("optimise") or []
        results = sorted(pool.map(evaluate, [(i, q, free) for i, q in enumerate(points)], chunksize=1),
                         key=lambda r: r["index"])
    likelihoods = list(next(r["chi2"] for r in results if "chi2" in r)) if any("chi2" in r for r in results) else []
    print(f"   {len(results)} points in {time.time() - t:.0f} s", flush=True)

    Path("logs").mkdir(exist_ok=True)
    out = HERE / "logs" / f"chi2_points_{a.set}_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    # the network a set's points were fitted on, when the points file names it (analysis/aic/: the AIC points)
    made_on = {k: spec[k] for k in ("network", "model_sha256", "run") if k in spec}
    out.write_text(json.dumps(dict(set=a.set, cobaya=spec["cobaya"], classy=classy.__file__, likelihoods=likelihoods,
                                   **made_on, results=results), indent=1))

    short = [n.split(".")[-1][:12] for n in likelihoods]
    print(f"   {'point':24s} " + " ".join(f"{s:>12s}" for s in short) + f" {'total':>10s} {'post-point':>10s}")
    for r in results:
        if "error" in r:
            print(f"   {r['label']:24s} ERROR {r['error']}")
            continue
        d = r.get("logpost_minus_point")
        print(f"   {r['label']:24s} " + " ".join(f"{r['chi2'][n]:12.2f}" for n in likelihoods)
              + f" {r['chi2_total']:10.2f} " + (f"{d:+10.3f}" if d is not None else ""))
    good = [r for r in results if "chi2" in r]
    for r in good:
        if "chi2_before" in r:
            print(f"   {r['label']:24s} optimised ({r['optimiser_calls']} calls): chi^2 change "
                  + " ".join(f"{r['chi2'][n] - r['chi2_before'][n]:+12.2f}" for n in likelihoods)
                  + f" {r['chi2_total'] - sum(r['chi2_before'].values()):+10.2f}")
    checked = [r["logpost_minus_point"] for r in good if "logpost_minus_point" in r]
    if checked:
        print(f"   log-posterior minus the point's own value: largest |difference| {max(map(abs, checked)):.4f} "
              f"over {len(checked)} points")
    for r in good:
        if "logpost_minus_network" in r:
            print(f"   {r['label']:24s} truth minus network {r['logpost_minus_network']:+.3f}")
    if good and len(good) <= 8:
        ref = good[0]
        print(f"\n   chi^2 minus that at {ref['label']}:")
        for r in good[1:]:
            print(f"   {r['label']:24s} " + " ".join(f"{r['chi2'][n] - ref['chi2'][n]:+12.2f}" for n in likelihoods)
                  + f" {r['chi2_total'] - ref['chi2_total']:+10.2f}")
    print("written:", out, flush=True)


if __name__ == "__main__":
    main()
