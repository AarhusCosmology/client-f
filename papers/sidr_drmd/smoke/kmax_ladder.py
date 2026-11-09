"""How high must k_max go for a non-linear module to give finite spectra at a run's bad points? CLASS++
alone (no Cobaya), with exactly the inputs Cobaya passes for a second-set run, one module, and a ladder
of P_k_max_1/Mpc values:

    python smoke/kmax_ladder.py RUN COBAYA LABEL --module halofit|hmcode --kmax 5 7 10 ... \
        [--points refused|inbox|sample:N] [--processes P]

RUN is a run directory (its latest training set and refused_value.json give the points), COBAYA the
Cobaya file whose extra_args are used (runs/sidr_drmd2_pp_low/cobaya.yaml: the second set's). With
halofit or HMcode CLASS++ computes the perturbations to max(P_k_max_1/Mpc, nonlinear_min_k_max = 5)
(input_module.cpp), so the runs' P_k_max_1/Mpc = 1 is k_max = 5 and a ladder starts there.

Points: `refused`, the run's refused points (41 in the first Pantheon+ run: NaN from halofit on Grendel);
`inbox`, those of them inside the second set's box (f_idm <= 0.4, m_int in [0.06, 0.3], logGeff <= -0.8);
`sample:N`, N random finite points inside that box with delta_Neff < 0.13 (seed 1); `best`, the run's
best point (for timing: run it with one process).

Prints, per k_max, how many points give finite TT, EE and lensing-potential spectra, the median time of
an evaluation, and how far the lensed spectra move from those at the highest k_max of the ladder: the
largest |C_l / C_l(top) - 1| at points finite at both, over the multipoles the likelihoods use (TT l <= 2500,
EE l <= 2000, lensing potential 8 <= L <= 400) and for the lensing potential over all L <= 2500. Writes
logs/kmax_ladder_<LABEL>.json and the spectra, logs/kmax_ladder_<LABEL>.npz.
Made 21 October 2026 for the SIDR + DRMD second set.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE, DATA = None, None


def latest_data(run):
    """The run's latest training set, training_data/data_it_N.csv with the largest N."""
    found = {int(m.group(1)): p for p in (run / "training_data").glob("data_it_*.csv")
             if (m := re.fullmatch(r"data_it_(\d+)\.csv", p.name))}
    return found[max(found)]
RANGES = {"tt": ("tt", 2, 2500), "ee": ("ee", 2, 2000), "pp400": ("pp", 8, 400), "pp": ("pp", 2, 2500)}


def evaluate(task):
    """One point at one k_max: (row, k_max, finite?, error, seconds, lensed TT/EE/pp)."""
    from classy import Class

    i, kmax = task
    r = DATA.iloc[i]
    pars = dict(BASE, omega_b=r.omega_b, omega_cdm=r.omega_cdm, H0=r.H0, A_s=1e-10 * np.exp(r.logA), n_s=r.n_s,
                tau_reio=r.tau_reio, delta_Neff_drmd=r.delta_Neff_drmd, f_idm_drmd=r.f_idm_drmd,
                z_stop=10 ** r.log10_zstop, G_over_aH_drmd_ini=10 ** r.log10_G_over_aH,
                **{"nui1.m": r.m_int / 3, "nui1.log10G_eff": r.logGeff, "P_k_max_1/Mpc": kmax})
    c = Class(); c.set(pars); t = time.time()
    entry = dict(row=int(i), kmax=kmax, dNeff=float(r.delta_Neff_drmd), f_idm=float(r.f_idm_drmd))
    try:
        c.compute()
        cl = c.lensed_cl(2500)
        spectra = {k: cl[k][2:].tolist() for k in ("tt", "ee", "pp")}
        entry.update(finite=all(np.isfinite(v).all() for v in spectra.values()), spectra=spectra)
    except Exception as error:
        entry.update(finite=False, error=str(error).strip().splitlines()[-1][:200])
    finally:
        c.struct_cleanup(); c.empty()
    entry["seconds"] = time.time() - t
    return entry


def choose_rows(data, refused_value, which):
    refused = np.isclose(data["loglkl"], refused_value)
    box = data["f_idm_drmd"].le(0.4) & data["m_int"].between(0.06, 0.3) & data["logGeff"].le(-0.8)
    if which == "refused":
        return np.flatnonzero(refused)
    if which == "best":
        return np.array([int(np.argmax(np.where(refused, -np.inf, data["loglkl"].to_numpy())))])
    if which == "inbox":
        return np.flatnonzero(refused & box)
    if which.startswith("sample:"):
        inside = np.isfinite(data["loglkl"]) & ~refused & box & data["delta_Neff_drmd"].lt(0.13)
        n = min(int(which.split(":")[1]), int(inside.sum()))
        return np.sort(np.random.default_rng(1).choice(np.flatnonzero(inside), size=n, replace=False))
    raise SystemExit(f"--points must be refused, inbox, best or sample:N, not {which!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", type=Path)
    p.add_argument("cobaya", type=Path)
    p.add_argument("label")
    p.add_argument("--module", choices=("halofit", "hmcode"), required=True)
    p.add_argument("--kmax", type=float, nargs="+", required=True)
    p.add_argument("--points", default="refused")
    p.add_argument("--processes", type=int, default=1)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="one more CLASS input (repeatable), e.g. extrapolation_method=4 (extrap_hmcode)")
    args = p.parse_args()
    import classy

    global BASE, DATA
    BASE = dict(yaml.safe_load(args.cobaya.read_text())["theory"]["classy"]["extra_args"])
    BASE.update({"output": "pCl mPk tCl lCl", "lensing": "yes", "l_max_scalars": 2500, "non_linear": args.module})
    BASE.update(item.split("=", 1) for item in args.set)
    refused_value = json.loads((args.run / "refused_value.json").read_text())["value"]
    DATA = pd.read_csv(latest_data(args.run))
    rows = choose_rows(DATA, refused_value, args.points)
    ladder = sorted(set(args.kmax))
    print(f"[{args.label}] classy from {classy.__file__}; {args.module}; {len(rows)} points ({args.points}); "
          f"k_max ladder {ladder}; set {args.set or '-'}", flush=True)
    tasks = [(int(i), k) for i in rows for k in ladder]
    if args.processes > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.processes) as pool:
            out = pool.map(evaluate, tasks, chunksize=1)
    else:
        out = [evaluate(t) for t in tasks]
    by = {(e["row"], e["kmax"]): e for e in out}
    top = ladder[-1]
    print(f"[{args.label}] {'k_max':>6} {'finite':>9} {'median s':>9}   largest |C_l/C_l(k_max {top:g}) - 1|: "
          "TT l<=2500, EE l<=2000, pp 8<=L<=400, pp L<=2500")
    summary = []
    for k in ladder:
        here = [by[(int(i), k)] for i in rows]
        good = sum(e["finite"] for e in here)
        moved = {s: 0.0 for s in RANGES}
        for i in rows:
            a, b = by[(int(i), k)], by[(int(i), top)]
            if a["finite"] and b["finite"]:
                for s, (spectrum, lo, hi) in RANGES.items():
                    x, y = (np.asarray(e["spectra"][spectrum])[lo - 2:hi - 1] for e in (a, b))
                    moved[s] = max(moved[s], float(np.max(np.abs(x / y - 1))))
        seconds = float(np.median([e["seconds"] for e in here]))
        summary.append(dict(kmax=k, finite=good, points=len(here), median_seconds=seconds, moved=moved))
        print(f"[{args.label}] {k:6g} {good:4d}/{len(here):<4d} {seconds:9.2f}   "
              + ", ".join(f"{moved[s]:.1e}" for s in RANGES), flush=True)
    failing = sorted({(e["row"], e.get("error", "NaN spectra")) for e in out if not e["finite"] and e["kmax"] == top})
    for row, why in failing:
        print(f"[{args.label}] still not finite at k_max {top:g}: row {row}: {why}", flush=True)
    print(f"[{args.label}] SUMMARY {args.module}: " + ", ".join(f"k_max {s['kmax']:g} {s['finite']}/{s['points']}"
                                                         for s in summary), flush=True)
    Path("logs").mkdir(exist_ok=True)
    nan = np.full((3, 2499), np.nan)
    np.savez_compressed(f"logs/kmax_ladder_{args.label}.npz", row=[e["row"] for e in out], kmax=[e["kmax"] for e in out],
                        finite=[e["finite"] for e in out], seconds=[e["seconds"] for e in out],
                        spectra=np.array([[e["spectra"][s] for s in ("tt", "ee", "pp")] if "spectra" in e else nan
                                          for e in out]))   # l = 2..2500
    for e in out:
        e.pop("spectra", None)
    Path(f"logs/kmax_ladder_{args.label}.json").write_text(
        json.dumps(dict(classy=classy.__file__, module=args.module, points=args.points, summary=summary, evaluations=out),
                   indent=1))


if __name__ == "__main__":
    main()
