"""Minimise chi^2 with the true likelihood alone, from several starts in parallel (26 October 2026).

    python smoke/minimise_true.py SPEC NAME [--workers 48] [--starts LABEL ...]

Made to settle the best fit of DRMD on the World Cup data with SH0ES (the paper's Sec. 5.5: the combined model is 17
units of chi^2 below DRMD, and DRMD's best fit is an emulator's), but general. SPEC (JSON) gives the Cobaya file, the
job.env (CLASS++ build), parameters held fixed ("fixed", e.g. {"logGeff": -5.5}), the fast nuisance parameters
("nuisance"), a scale for every other parameter ("scales": posterior standard deviations), optionally their covariance
("covariance": {names, matrix}; the search then runs in the coordinates it whitens, which a strongly correlated valley
needs: with scales alone the toy test stalled in its banana) and the starts ("starts": [{label, params}]).

Every evaluation computes the spectra once at a point of the slow parameters and maximises the log-posterior over the
nuisance parameters at fixed spectra (Powell within the prior, as smoke/chi2_points_mp.py --optimise; Cobaya keeps the
theory's output, so this costs no Boltzmann call). The outer search is L-BFGS-B over the slow parameters in whitened
coordinates (clipped into the prior box), with central-difference gradients of STEP (forward differences failed the toy
test's line searches far from the fit), evaluated in parallel: one batch of (1 + 2 x slow parameters) evaluations per
gradient, every start in its own thread on the shared pool, so that four starts
of 11 slow parameters take two rounds of 48 workers. Every evaluation is logged (logs/minimise_<NAME>_<job>.json, rewritten as it
goes), with its chi^2 per likelihood, so a stopped job keeps what it found. The log-posterior includes the priors, as the
paper's best fits do; chi^2 of the SH0ES prior on M_B is reported with the likelihoods.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chi2_points_mp as cp  # noqa: E402

STEP = 0.1          # the finite-difference step, in units of each slow parameter's scale
MAX_RUNS = 8        # L-BFGS-B runs per start (restarts from the best point)
RESTART_GAIN = 0.02 # a run that gains less than this in ln L ends the start


def evaluate(task):
    """Worker: the log-posterior at the slow parameters of `params`, maximised over the nuisance parameters."""
    params, nuisance = task
    names = list(cp.MODEL.parameterization.sampled_params())
    t = time.time()
    try:
        best, calls = cp.optimise(dict(params), names, nuisance) if nuisance else (dict(params), 0)
        r = cp.MODEL.logposterior({n: best[n] for n in names}, make_finite=False)
    except Exception as error:                     # noqa: BLE001 - recorded, the point counts as -inf
        return dict(params=params, logpost=-np.inf, error=f"{type(error).__name__}: {error}")
    if r.loglikes is None or len(r.loglikes) == 0 or not np.isfinite(r.logpost):
        return dict(params=params, logpost=-np.inf, error="outside the prior or not finite")
    chi2 = {k: -2 * float(v) for k, v in zip(cp.MODEL.likelihood, r.loglikes)}
    gauss = cp.gaussian_priors(best)
    return dict(params={n: float(best[n]) for n in names}, logpost=float(r.logpost), chi2=chi2,
                minus2_logprior=-2 * float(sum(r.logpriors)), minus2_logprior_gaussian=gauss,
                nuisance_calls=calls, seconds=time.time() - t)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("spec", type=Path)
    p.add_argument("name")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--maxiter", type=int, default=60)
    p.add_argument("--starts", nargs="+", help="only these starts (labels) of SPEC")
    a = p.parse_args()
    spec = json.loads(a.spec.read_text())
    if a.starts:
        missing = set(a.starts) - {s["label"] for s in spec["starts"]}
        if missing:
            raise SystemExit(f"no starts {sorted(missing)} in {a.spec}")
        spec["starts"] = [s for s in spec["starts"] if s["label"] in a.starts]
    if "classy" in (HERE / spec["cobaya"]).read_text():
        import classy
        sys.path.insert(0, str(HERE))
        import client_paths  # noqa: E402,F401  CLiENT's hpc/ tools
        from boltzmann_build import location_problem
        problem = location_problem("classy", classy.__file__)
        if problem:
            raise SystemExit(f"FATAL: {problem} (source {spec['job_env']} first)")
    fixed, nuisance, scales = spec["fixed"], spec["nuisance"], spec["scales"]
    slow = list(scales)
    from cobaya.yaml import yaml_load_file
    prior = yaml_load_file(str(HERE / spec["cobaya"]))["params"]
    lo = np.array([float(prior[n]["prior"]["min"]) for n in slow])
    hi = np.array([float(prior[n]["prior"]["max"]) for n in slow])
    sc = np.array([float(scales[n]) for n in slow])
    # the search coordinates z: x = x_start + L z, L the Cholesky factor of the covariance (or the scales)
    if "covariance" in spec:
        order = [spec["covariance"]["names"].index(n) for n in slow]
        C = np.array(spec["covariance"]["matrix"], float)[np.ix_(order, order)]
        L = np.linalg.cholesky(C)
    else:
        L = np.diag(sc)
    out = HERE / "logs" / f"minimise_{a.name}_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    out.parent.mkdir(exist_ok=True)
    log = dict(spec=str(a.spec), name=a.name, cobaya=spec["cobaya"], fixed=fixed, slow=slow, nuisance=nuisance,
               step=STEP, starts={})
    lock = threading.Lock()

    def save():
        with lock:
            out.write_text(json.dumps(log, indent=1))

    print(f"== {a.name}: {len(spec['starts'])} starts, {len(slow)} slow parameters ({', '.join(slow)}), "
          f"{len(nuisance)} nuisance, fixed {fixed}; {a.workers} workers", flush=True)
    with mp.get_context("spawn").Pool(a.workers, initializer=cp.start, initargs=(spec["cobaya"], None)) as pool:

        def run(start):
            label = start["label"]
            entry = log["starts"].setdefault(label, dict(note=start.get("note", ""), evaluations=[], best=None))
            base = dict(start["params"], **fixed)
            state = dict(nuis={n: base[n] for n in nuisance})

            x_start = np.array([float(base[n]) for n in slow])

            def to_params(z):
                x = np.clip(x_start + L @ z, lo, hi)
                return dict(base, **state["nuis"], **dict(zip(slow, map(float, x))))

            def fun_and_grad(z):
                e = np.eye(len(slow))
                tasks = ([to_params(z)] + [to_params(z + STEP * e[i]) for i in range(len(slow))]
                         + [to_params(z - STEP * e[i]) for i in range(len(slow))])
                res = pool.map(evaluate, [(t, nuisance) for t in tasks], chunksize=1)
                with lock:
                    entry["evaluations"] += [dict(kind="centre" if i == 0 else f"{'+' if i <= len(slow) else '-'}z{(i - 1) % len(slow)}", **r)
                                             for i, r in enumerate(res)]
                    if np.isfinite(res[0]["logpost"]) and (entry["best"] is None or res[0]["logpost"] > entry["best"]["logpost"]):
                        entry["best"] = res[0]
                        state["nuis"] = {n: res[0]["params"][n] for n in nuisance}
                save()
                f0 = -res[0]["logpost"]
                if not np.isfinite(f0):
                    return 1e10, np.zeros(len(slow))
                n = len(slow)
                fp = np.array([-r["logpost"] for r in res[1:n + 1]])
                fm = np.array([-r["logpost"] for r in res[n + 1:]])
                # central differences; at a clipped or refused side the one-sided difference that exists
                g = np.where(np.isfinite(fp) & np.isfinite(fm), (fp - fm) / (2 * STEP),
                             np.where(np.isfinite(fp), (fp - f0) / STEP, np.where(np.isfinite(fm), (f0 - fm) / STEP, 0.0)))
                print(f"   {label}: -lnL {f0:.3f}  |grad| {np.linalg.norm(g):.3f}  ({len(entry['evaluations'])} evaluations)", flush=True)
                return f0, g

            from scipy.optimize import minimize
            # the prior box is not aligned with z: points are clipped into it (to_params). A run whose line search
            # fails (noise, or a start far from the fit where the covariance is a poor metric) is restarted from the
            # best point with a fresh curvature model, until a run gains less than RESTART_GAIN in ln L
            runs = []
            for k in range(MAX_RUNS):
                before = entry["best"]["logpost"] if entry["best"] else -np.inf
                if entry["best"]:
                    x_start[:] = [entry["best"]["params"][n] for n in slow]
                r = minimize(fun_and_grad, np.zeros(len(slow)), jac=True, method="L-BFGS-B",
                             options=dict(maxiter=a.maxiter, ftol=1e-8, gtol=1e-3, maxls=8))
                runs.append(dict(message=str(r.message), iterations=int(r.nit)))
                if entry["best"]["logpost"] - before < RESTART_GAIN:
                    break
            with lock:
                entry.update(message=runs[-1]["message"], runs=runs)
            save()
            b = entry["best"]
            print(f"== {label}: best -2 ln posterior {-2 * b['logpost']:.3f} after {len(entry['evaluations'])} evaluations; "
                  f"{r.message}", flush=True)

        threads = [threading.Thread(target=run, args=(s,)) for s in spec["starts"]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    print(f"written: {out}")
    for label, e in log["starts"].items():
        b = e["best"]
        if b is not None:
            shoes = b["minus2_logprior_gaussian"].get("Mb")
            data = sum(b["chi2"].values()) + (shoes - np.log(2 * np.pi * 0.027 ** 2) if shoes is not None else 0)
            print(f"   {label:28s} chi2 (likelihoods + SH0ES) {data:.2f}   -2 ln posterior {-2 * b['logpost']:.2f}")


if __name__ == "__main__":
    main()
