"""What do the non-linear module's settings do to the likelihoods themselves? The run's Cobaya model at
training points of the oscillating corner (dark acoustic oscillations: f_idm > 0.2, log10 z_stop > 4.3,
delta_Neff < 0.05) and at control points, near the best fit, for each setting; the log-likelihood of every
likelihood, and its difference from the last setting given (the reference):

    python smoke/kmax_loglike.py RUN COBAYA LABEL --setting halofit:5 --setting halofit:30 \
        [--corner 10] [--control 4] [--within 10] [--only planck_2018_lowl.TT ...]

A setting is MODULE:K_MAX (P_k_max_1/Mpc; with a non-linear module CLASS++ computes to at least 5 anyway).
Points: --corner random corner points and --control random points with f_idm < 0.05, both inside the second
set's box and within --within of the run's best log-likelihood (seed 2), and the best point. --only keeps
the named likelihoods (the laptop has no PR4 lensing); the default is every likelihood of COBAYA.
Writes logs/kmax_loglike_<LABEL>.json. Made 21 October 2026.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from classy_build_points import latest_data

def build(info, module, kmax):
    from cobaya.model import get_model

    info = json.loads(json.dumps(info))
    extra = info["theory"]["classy"].setdefault("extra_args", {})
    extra["non_linear"] = module
    extra["P_k_max_1/Mpc"] = kmax
    return get_model(info)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", type=Path)
    p.add_argument("cobaya", type=Path)
    p.add_argument("label")
    p.add_argument("--setting", action="append", required=True, help="MODULE:K_MAX, the last is the reference")
    p.add_argument("--corner", type=int, default=10)
    p.add_argument("--control", type=int, default=4)
    p.add_argument("--within", type=float, default=10.0)
    p.add_argument("--only", nargs="+")
    args = p.parse_args()

    info = yaml.safe_load(args.cobaya.read_text())
    info.pop("sampler", None); info.pop("output", None)
    if args.only:
        missing = set(args.only) - set(info["likelihood"])
        if missing:
            raise SystemExit(f"not likelihoods of {args.cobaya}: {sorted(missing)}")
        info["likelihood"] = {k: v for k, v in info["likelihood"].items() if k in args.only}
    refused_value = json.loads((args.run / "refused_value.json").read_text())["value"]
    d = pd.read_csv(latest_data(args.run))
    ok = np.isfinite(d.loglkl) & ~np.isclose(d.loglkl, refused_value)
    box = d.f_idm_drmd.le(0.4) & d.m_int.between(0.06, 0.3) & d.logGeff.le(-0.8)
    near = ok & box & (d.loglkl > d.loglkl[ok].max() - args.within)
    corner = near & d.f_idm_drmd.gt(0.2) & d.log10_zstop.gt(4.3) & d.delta_Neff_drmd.lt(0.05)
    control = near & d.f_idm_drmd.lt(0.05)
    rng = np.random.default_rng(2)
    pick = lambda mask, n: list(rng.choice(np.flatnonzero(mask), size=min(n, int(mask.sum())), replace=False))
    points = [("best", int(np.argmax(np.where(ok, d.loglkl, -np.inf))))]
    points += [("corner", int(i)) for i in pick(corner, args.corner)] + [("control", int(i)) for i in pick(control, args.control)]

    settings = [(s.split(":")[0], float(s.split(":")[1])) for s in args.setting]
    results = {}
    for module, kmax in settings:
        model = build(info, module, kmax)
        names = list(model.parameterization.sampled_params())
        missing = [n for n in names if n not in d.columns]
        if missing:
            raise SystemExit(f"the training set has no column for {missing}")
        for kind, i in points:
            values = {n: float(d.iloc[i][n]) for n in names}
            logpost = model.logposterior(values, make_finite=False)
            if logpost.loglikes is None or len(logpost.loglikes) == 0:   # outside COBAYA's prior: not evaluated
                print(f"[{args.label}] row {i} lies outside the prior of {args.cobaya}: left out", flush=True)
                continue
            results[(kind, i, module, kmax)] = dict(zip(model.likelihood.keys(), map(float, logpost.loglikes)))
        likelihoods = list(model.likelihood.keys())
        print(f"[{args.label}] {module} k_max {kmax:g}: {len(points)} points done", flush=True)

    Path("logs").mkdir(exist_ok=True)
    Path(f"logs/kmax_loglike_{args.label}.json").write_text(json.dumps(
        [dict(kind=k, row=i, module=m, kmax=km, loglikes=v) for (k, i, m, km), v in results.items()], indent=1))
    points = [(kind, i) for kind, i in points if all((kind, i, m, k) in results for m, k in settings)]
    reference = settings[-1]
    print(f"[{args.label}] log-likelihood minus that with {reference[0]} k_max {reference[1]:g}; "
          f"training points within {args.within:g} of the run's best")
    head = "".join(f"{name.split('.')[-1][:16]:>18s}" for name in likelihoods)
    for module, kmax in settings[:-1]:
        print(f"[{args.label}] {module} k_max {kmax:g}")
        print(f"   {'point':>14s} {'dNeff':>7s} {'f_idm':>6s} {'lg zstop':>8s}{head}{'total':>10s}")
        for kind, i in points:
            r = d.iloc[i]
            diff = [results[(kind, i, module, kmax)][n] - results[(kind, i) + reference][n] for n in likelihoods]
            print(f"   {kind:>7s} {i:6d} {r.delta_Neff_drmd:7.4f} {r.f_idm_drmd:6.3f} {r.log10_zstop:8.2f}"
                  + "".join(f"{x:18.3f}" for x in diff) + f"{sum(diff):10.3f}", flush=True)


if __name__ == "__main__":
    main()
