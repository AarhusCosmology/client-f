"""The cases of the preset study, each timed by reproduce.py's computation in a process of its own, with a time limit.

    OMP_NUM_THREADS=6 python ladder.py kmax --pre CLASSPP/pre/act_dr6.pre [--timeout 300]
        # the SIDR + DRMD best fit with the preset, nonlinear_min_k_max on a ladder
    OMP_NUM_THREADS=6 python ladder.py ingredients --pre CLASSPP/pre/act_dr6.pre [--kmax 70]
        # the preset at that k_max, the model's ingredients switched on one at a time (written, not run)

Appends one JSON line per case to ladder_<name>.jsonl beside this script and prints it.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

from reproduce import compute, read_ini

HERE = Path(__file__).resolve().parent

# The ingredients, as changes to sidr_drmd_bestfit.ini.
STANDARD_NU = {"N_ncdm": "1", "deg_ncdm": "3", "m_ncdm": "0.02", "T_ncdm": "0.71611"}
SIDR_KEYS = ("nui1.type", "nui1.deg", "nui1.T", "nui1.m", "nui1.log10G_eff")
INGREDIENTS = {
    "LCDM, three standard ncdm (0.06 eV), no DRMD": dict(drop=SIDR_KEYS, set={**STANDARD_NU, "delta_Neff_drmd": "0",
                                                                          "f_idm_drmd": "0"}),
    "standard ncdm + DRMD (best fit)": dict(drop=SIDR_KEYS, set=STANDARD_NU),
    "SIDR weak (log10 G_eff -5.5), no DRMD": dict(set={"nui1.log10G_eff": "-5.5", "delta_Neff_drmd": "0",
                                                       "f_idm_drmd": "0"}),
    "SIDR strong (best fit), no DRMD": dict(set={"delta_Neff_drmd": "0", "f_idm_drmd": "0"}),
    "SIDR weak (log10 G_eff -5.5) + DRMD (best fit)": dict(set={"nui1.log10G_eff": "-5.5"}),
    "SIDR strong + DRMD (the best fit)": dict(),
}


def timed(pars, timeout):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=compute, args=(pars, queue))
    t = time.time()
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return dict(seconds=None, finished=False, wall=time.time() - t)
    return {**queue.get(), "finished": True, "wall": time.time() - t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ladder", choices=("kmax", "ingredients"))
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--kmax", nargs="*", default=None)
    ap.add_argument("--pre", required=True, help="pre/act_dr6.pre of the CLASS++ tree")
    a = ap.parse_args()
    point = read_ini(HERE / "sidr_drmd_bestfit.ini")
    pre = read_ini(a.pre)
    cases = []
    if a.ladder == "kmax":
        for k in a.kmax or ["2 (default)", "10", "20", "30", "40", "50", "70"]:
            p = {**point, **pre}
            if k.startswith("2 "):
                p.pop("nonlinear_min_k_max")
            else:
                p["nonlinear_min_k_max"] = k
            cases.append((f"nonlinear_min_k_max = {k}", p))
    else:
        k = (a.kmax or ["70"])[0]
        for name, change in INGREDIENTS.items():
            p = {key: v for key, v in point.items() if key not in change.get("drop", ())}
            p.update(change.get("set", {}))
            p.update(pre)
            p["nonlinear_min_k_max"] = k
            cases.append((f"{name}, nonlinear_min_k_max = {k}", p))
    out = HERE / f"ladder_{a.ladder}.jsonl"
    for name, pars in cases:
        r = timed(pars, a.timeout)
        r.update(case=name, timeout=a.timeout, threads=os.environ.get("OMP_NUM_THREADS"))
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")
        state = (f"{r['seconds']:.1f} s" if r["finished"] and "error" not in r
                 else f"ERROR {r['error']}" if "error" in r else f"> {a.timeout:.0f} s")
        print(f"{name:70s} {state}", flush=True)


if __name__ == "__main__":
    main()
