"""True best fits of the LambdaCDM pair (make_inputs.py): Py-BOBYQA on Cobaya's log posterior in whitened coordinates,
from six starts per input, in parallel, one Cobaya model per process. 9 November 2026.

    python runs/dncdm_proxy_lcdm/minimize_lcdm.py [--workers N] [--only pp_m005 wc_m0 ...]     # submit_minimize.sh

Why not Cobaya's own minimizer: it rescales each parameter by its conditional sigma but leaves the correlations, and on
a toy with this run's PDP+ posterior covariance (15 parameters, deterministic noise 0.005 in log L) it stopped 0.4-2.0
above the minimum in chi^2 (150-220 evaluations; 0.04-0.08 in ~500 with a full quadratic model). Here BOBYQA works in
x = x0 + L z (L L^T = covmat_<ds>.txt, the CLiENT posterior's covariance), in two stages (task() says which). On that
toy through this very code (the run's own priors, which include Gaussian ones on A_planck, calTE and calEE), five
starts ended 0.001-0.010 above the true optimum in -2 log posterior, in 150-190 evaluations; with the toy's optimum
below amp_143x217 = 0 (H0WC's best fit is at that edge), 0.003-0.007 in 330-440 (laptop, Py-BOBYQA 1.5.0 as on
Grendel). A step outside the prior box is evaluated at the point clipped into the box plus 50 per squared sigma outside
it; a Gaussian prior's box is its +-5 sigma (Cobaya's bounds at 0.9999995).

Every (input, start) writes logs/lcdm_minimize/<input>/start<k>.json (the best point, its -log posterior, chi^2 of every
likelihood, -2 log prior, evaluations, BOBYQA's message) and start<k>_trace.jsonl (every evaluation). A start with a
json is not run again, so a resubmitted job continues. Then logs/lcdm_minimize_<job>.json and a table: per input the
best -log posterior and the gap to the second-best start; per data set Delta chi2 = 2 (best m005 - best m0), next to
the proxy profile's own (its plateau node at log10Gamma -2 minus the node m0 started from); and each LambdaCDM best fit
against the proxy node it started from: its log posterior minus ln 16 (the proxy's flat log10Gamma prior over [-2, 14],
which LambdaCDM does not have; CLiENT's objective is Cobaya's log posterior with every prior's normalisation).
The shell must have sourced env.sh and runs/dncdm_proxy_pp/job.env (CLASS++ cb2255f4 first on PYTHONPATH).
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

KIT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INPUTS = ["pp_m005", "pp_m0", "wc_m005", "wc_m0"]
LN_GAMMA_PRIOR = math.log(16.0)
RHOEND = 0.003          # in posterior sigmas
MAXFUN = 600
OUTSIDE = 50.0          # -log posterior per squared sigma outside the box


def task(job):
    name, k, start, names = job
    os.chdir(KIT)                                  # the yamls' python_path (likelihoods/) is relative to this folder (papers/neutrino_decay/)
    out = KIT / "logs" / "lcdm_minimize" / name
    done = out / f"start{k}.json"
    if done.exists():
        return dict(json.loads(done.read_text()), note="from an earlier job")
    t = time.time()
    try:
        import numpy as np
        import pybobyqa
        from cobaya.model import get_model
        from cobaya.yaml import yaml_load_file

        info = yaml_load_file(str(HERE / f"cobaya_{name}.yaml"))
        info.pop("sampler", None)
        info.pop("output", None)
        model = get_model(info)
        sampled = list(model.parameterization.sampled_params())
        if sorted(sampled) != sorted(names):
            raise ValueError(f"sampled parameters {sampled} are not the starts' {names}")
        ds = name.split("_")[0]
        cov_file = HERE / f"covmat_{ds}.txt"
        cov_names = cov_file.read_text().splitlines()[0].lstrip("#").split()
        if cov_names != names:
            raise ValueError(f"{cov_file.name} is for {cov_names}, not {names}")
        cov = np.loadtxt(cov_file)
        L = np.linalg.cholesky(cov)
        sd = np.sqrt(np.diag(cov))
        b = np.array(model.prior.bounds(confidence_for_unbounded=0.9999995))
        lo, hi = b[[sampled.index(n) for n in names]].T
        x0 = np.asarray(start, float)
        out.mkdir(parents=True, exist_ok=True)
        trace = (out / f"start{k}_trace.jsonl").open("w")
        best = [math.inf, None]
        stages = []

        def stage(xs, free, label, rhobeg=1.0):
            """BOBYQA over the parameters `free` (indices), the others fixed at xs, in coordinates whitened by the
            covariance of `free` conditional on the others; a step outside the box is the clipped point + OUTSIDE
            per squared sigma outside it."""
            fixed = [i for i in range(len(names)) if i not in free]
            c = cov[np.ix_(free, free)]
            if fixed:
                c = c - cov[np.ix_(free, fixed)] @ np.linalg.solve(cov[np.ix_(fixed, fixed)], cov[np.ix_(fixed, free)])
            Lf = np.linalg.cholesky(c)

            def objective(z):
                x = xs.copy()
                x[free] = xs[free] + Lf @ z
                xc = np.clip(x, lo, hi)
                r = model.logposterior(dict(zip(names, xc)), make_finite=True)
                v = -float(r.logpost) + OUTSIDE * float(np.sum(((x - xc) / sd) ** 2))
                trace.write(json.dumps(dict(stage=label, x=xc.tolist(), outside=float(np.max(np.abs(x - xc) / sd)),
                                            minuslogpost=-float(r.logpost), t=time.time() - t)) + "\n")
                trace.flush()
                if v < best[0]:
                    best[:] = [v, xc.copy()]
                return v

            res = pybobyqa.solve(objective, np.zeros(len(free)), rhobeg=rhobeg, rhoend=RHOEND, maxfun=MAXFUN)
            x = xs.copy()
            x[free] = xs[free] + Lf @ res.x
            stages.append(dict(stage=label, evaluations=int(res.nf), message=str(res.msg),
                               fixed=[names[i] for i in fixed]))
            return np.clip(x, lo, hi)

        # 1: every parameter free. 2: a parameter that ended within 0.05 sigma of a bound is fixed there and the others
        # minimized again (BOBYQA's quadratic model cannot follow the clipped kink: on a toy with this covariance and
        # its optimum below amp_143x217 = 0, stage 1 alone ended 0.8 above the constrained optimum in chi^2); without
        # one, a restart from stage 1's point at a radius of 0.3 sigma (the toy: 0.01-0.04 above the optimum after
        # stage 1, which the restart brings down; Gaussian priors make the posterior's covariance differ from covmat's).
        x = stage(x0, list(range(len(names))), "all free")
        edge = [i for i in range(len(names)) if min(x[i] - lo[i], hi[i] - x[i]) < 0.05 * sd[i]]
        if edge:
            for i in edge:
                x[i] = lo[i] if x[i] - lo[i] < hi[i] - x[i] else hi[i]
            x = stage(x, [i for i in range(len(names)) if i not in edge], "edge fixed")
        else:
            x = stage(x, list(range(len(names))), "restart", rhobeg=0.3)
        trace.close()
        if best[1] is not None and best[0] < -float(model.logposterior(dict(zip(names, x))).logpost):
            x = best[1]                                  # never report worse than the best point evaluated
        r = model.logposterior(dict(zip(names, x)), make_finite=False)
        row = dict(name=name, start=k, minuslogpost=-float(r.logpost),
                   chi2={lk: -2 * float(v) for lk, v in zip(model.likelihood, r.loglikes)},
                   minus2_logprior=-2 * float(sum(r.logpriors)),
                   params={n: float(v) for n, v in zip(names, x)},
                   at_edge=[n for n, v, l_, h in zip(names, x, lo, hi) if v <= l_ or v >= h],
                   start_x={n: float(v) for n, v in zip(names, x0)}, stages=stages,
                   evaluations=sum(s["evaluations"] for s in stages),
                   message=" / ".join(s["message"] for s in stages), seconds=time.time() - t)
        done.write_text(json.dumps(row, indent=1))
        return row
    except Exception as error:                     # noqa: BLE001 - recorded with the start
        return dict(name=name, start=k, error=f"{type(error).__name__}: {error}", seconds=time.time() - t)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--only", nargs="+", choices=INPUTS)
    a = p.parse_args()
    import classy
    sys.path.insert(0, str(KIT))
    import client_paths  # noqa: F401  CLiENT's hpc/ tools
    from boltzmann_build import location_problem

    problem = location_problem("classy", classy.__file__)
    if problem:
        raise SystemExit(f"FATAL: {problem} (source runs/dncdm_proxy_pp/job.env first)")
    inputs = a.only or INPUTS
    starts = {n: json.loads((HERE / f"starts_{n}.json").read_text()) for n in inputs}
    jobs = [(n, k, s, starts[n]["names"]) for n in inputs for k, s in enumerate(starts[n]["starts"])]
    print(f"== {len(jobs)} minimizations ({', '.join(inputs)}) on {a.workers} workers, OMP_NUM_THREADS "
          f"{os.environ.get('OMP_NUM_THREADS')}; classy from {classy.__file__}", flush=True)
    results = []
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(task, jobs, chunksize=1):
            results.append(r)
            if "error" in r:
                print(f"   {r['name']} start {r['start']}: ERROR {r['error']} ({r['seconds']:.0f} s)", flush=True)
            else:
                print(f"   {r['name']} start {r['start']}: -logpost {r['minuslogpost']:.3f} in {r['evaluations']} "
                      f"evaluations, {r['seconds'] / 3600:.2f} h  {r['message'][:50]}"
                      + (f"  at the edge: {r['at_edge']}" if r.get("at_edge") else ""), flush=True)
    results.sort(key=lambda r: (r["name"], r["start"]))

    summary = {}
    for n in inputs:
        ok = [r for r in results if r["name"] == n and "error" not in r]
        if not ok:
            continue
        v = sorted(r["minuslogpost"] for r in ok)
        best = min(ok, key=lambda r: r["minuslogpost"])
        node = starts[n]["profile_node"]
        summary[n] = dict(best=best["minuslogpost"], starts_ok=len(ok), all=v, gap=v[1] - v[0] if len(v) > 1 else None,
                          best_start=best["start"], chi2=best["chi2"], params=best["params"],
                          profile_node=node, logpost_minus_ln16=-best["minuslogpost"] - LN_GAMMA_PRIOR,
                          vs_proxy_node=(-best["minuslogpost"] - LN_GAMMA_PRIOR) - node["logl"])
    print("\n   input      best -logpost  starts  2nd-best gap  all starts (-logpost - best)            vs proxy node")
    for n, s in summary.items():
        print(f"   {n:9s} {s['best']:14.3f}  {s['starts_ok']:6d}  {s['gap'] if s['gap'] is not None else float('nan'):12.3f}  "
              + " ".join(f"{u - s['best']:.3f}" for u in s["all"]).ljust(40)
              + f"  {s['vs_proxy_node']:+.3f} (log10Gamma {s['profile_node']['log10Gamma']:+.2f})")
    for ds in ("pp", "wc"):
        if f"{ds}_m005" in summary and f"{ds}_m0" in summary:
            lcdm = 2 * (summary[f"{ds}_m005"]["best"] - summary[f"{ds}_m0"]["best"])
            prof = 2 * (summary[f"{ds}_m0"]["profile_node"]["logl"] - summary[f"{ds}_m005"]["profile_node"]["logl"])
            c1, c0 = summary[f"{ds}_m005"]["chi2"], summary[f"{ds}_m0"]["chi2"]
            print(f"   {ds}: Delta chi2 LambdaCDM (nu3 0.05 eV - massless) {lcdm:.3f};  proxy profile (plateau - node "
                  f"{summary[f'{ds}_m0']['profile_node']['log10Gamma']:+.2f}) {prof:.3f};  by likelihood: "
                  + ", ".join(f"{lk.split('.')[-1][:16]} {c1[lk] - c0[lk]:+.2f}" for lk in c1))
            summary[f"{ds}_delta_chi2"] = dict(lcdm=lcdm, proxy_profile=prof,
                                               by_likelihood={lk: c1[lk] - c0[lk] for lk in c1})
    out = KIT / "logs" / f"lcdm_minimize_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    out.write_text(json.dumps(dict(results=results, summary=summary, classy=classy.__file__), indent=1))
    print(f"\n   written {out.relative_to(KIT)}", flush=True)


if __name__ == "__main__":
    main()
