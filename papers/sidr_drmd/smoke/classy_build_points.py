"""CLASS++ alone (no Cobaya) at a run's refused points: are the CMB spectra finite with the classy found
on sys.path? One build per call; smoke/submit_classy_build_test.sh calls it once per build.

    python smoke/classy_build_points.py results/<timestamp>_<RUN> LABEL [--max N] [--set KEY=VALUE ...] [--drop KEY ...]

The CLASS inputs are exactly those Cobaya passes for the run (read back from classy's `pars` on the
laptop, 21 October 2026): its Cobaya file's extra_args plus output "pCl mPk tCl lCl" (mPk because sigma8
and S8 are derived), P_k_max_1/Mpc = 1, l_max_scalars = 2500, lensing. A first version without mPk and
with l_max 2600 gave finite spectra even with the build the runs use: the inputs matter.
Prints, per point, whether TT, EE and the lensing potential are finite, the largest D_EE(l = 2..29), and D_TT at l = 2, 220, 1500; then a summary line, and writes
logs/classy_build_<LABEL>.json for comparing the builds point by point.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

T0 = 2.7255e6


def latest_data(run):
    found = {int(m.group(1)): p for p in (run / "training_data").glob("data_it_*.csv")
             if (m := re.fullmatch(r"data_it_(\d+)\.csv", p.name))}
    return found[max(found)]


BASE, DATA = None, None


def evaluate(i):
    """One point: its spectra with the classy on sys.path (a fresh Class each time)."""
    from classy import Class

    r = DATA.iloc[i]
    pars = dict(BASE, omega_b=r.omega_b, omega_cdm=r.omega_cdm, H0=r.H0, A_s=1e-10 * np.exp(r.logA), n_s=r.n_s,
                tau_reio=r.tau_reio, delta_Neff_drmd=r.delta_Neff_drmd, f_idm_drmd=r.f_idm_drmd,
                z_stop=10 ** r.log10_zstop, G_over_aH_drmd_ini=10 ** r.log10_G_over_aH,
                **{"nui1.m": r.m_int / 3, "nui1.log10G_eff": r.logGeff})
    c = Class(); c.set(pars); t = time.time()
    entry = dict(row=int(i), dNeff=float(r.delta_Neff_drmd), f_idm=float(r.f_idm_drmd))
    try:
        c.compute()
        cl = c.lensed_cl(2500); ell = np.arange(len(cl["tt"]))
        D = {k: ell * (ell + 1) * cl[k] * T0 ** 2 / (2 * np.pi) for k in ("tt", "ee")}
        entry.update(finite=dict(tt=bool(np.isfinite(cl["tt"][2:]).all()), ee=bool(np.isfinite(cl["ee"][2:]).all()),
                                 pp=bool(np.isfinite(cl["pp"][2:]).all())),
                     dee_max=float(np.nanmax(D["ee"][2:30])) if np.isfinite(D["ee"][2:30]).any() else None,
                     dtt=[float(D["tt"][l]) for l in (2, 220, 1500)])
    except Exception as error:
        entry.update(error=str(error).strip().splitlines()[-1][:200])
    finally:
        c.struct_cleanup(); c.empty()
    entry["seconds"] = time.time() - t
    return entry


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", type=Path)
    p.add_argument("label")
    p.add_argument("--max", type=int)
    p.add_argument("--sample", type=int, help="N random finite rows inside the second set's box with delta_Neff < 0.13, "
                   "in place of the refused ones (seed 1: the same rows for every call)")
    p.add_argument("--processes", type=int, default=1, help="points evaluated in parallel")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override one CLASS input (repeatable)")
    p.add_argument("--drop", action="append", default=[], metavar="KEY", help="leave one CLASS input out (repeatable)")
    args = p.parse_args()
    import classy
    from classy import Class

    info = yaml.safe_load(next((args.run / "likelihood_input").glob("*.yaml")).read_text())
    base = dict(info["theory"]["classy"].get("extra_args", {}))
    base.update({"output": "pCl mPk tCl lCl", "lensing": "yes", "l_max_scalars": 2500, "P_k_max_1/Mpc": 1})
    for item in args.set:
        key, value = item.split("=", 1)
        base[key] = value
    for key in args.drop:
        base.pop(key, None)
    refused_value = json.loads((args.run / "refused_value.json").read_text())["value"]
    data = pd.read_csv(latest_data(args.run))
    if args.sample:
        inside = (np.isfinite(data["loglkl"]) & ~np.isclose(data["loglkl"], refused_value) & data["f_idm_drmd"].le(0.4)
                  & data["m_int"].between(0.06, 0.3) & data["logGeff"].le(-0.8) & data["delta_Neff_drmd"].lt(0.13))
        rows = np.sort(np.random.default_rng(1).choice(np.flatnonzero(inside), size=min(args.sample, int(inside.sum())), replace=False))
    else:
        rows = np.flatnonzero(np.isclose(data["loglkl"], refused_value))[:args.max]
    print(f"[{args.label}] classy from {classy.__file__}; {len(rows)} {'sampled' if args.sample else 'refused'} points; changed inputs: set {args.set or '-'}, "
          f"dropped {args.drop or '-'}", flush=True)
    global BASE, DATA
    BASE, DATA = base, data
    if args.processes > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.processes) as pool:
            out = pool.map(evaluate, [int(i) for i in rows], chunksize=1)
    else:
        out = [evaluate(int(i)) for i in rows]
    for entry in out:
        ok = "error" not in entry and all(entry["finite"].values())
        print(f"[{args.label}] row {entry['row']:6d} dNeff {entry['dNeff']:.4f} f_idm {entry['f_idm']:.3f}: "
              + ("FINITE" if ok else "NOT FINITE " + json.dumps(entry.get("finite", entry.get("error"))))
              + (f"  max D_EE {entry['dee_max']:.4f}  D_TT(2, 220, 1500) {np.round(entry['dtt'], 1).tolist()}" if ok else ""), flush=True)
    good = sum("error" not in e and all(e["finite"].values()) for e in out)
    print(f"[{args.label}] SUMMARY: {good} of {len(out)} {'sampled' if args.sample else 'refused'} points give finite spectra", flush=True)
    Path("logs").mkdir(exist_ok=True)
    Path(f"logs/classy_build_{args.label}.json").write_text(json.dumps(dict(classy=classy.__file__, points=out), indent=1))


if __name__ == "__main__":
    main()
