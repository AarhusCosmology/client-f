"""Time one CLASS++ computation of an .ini file, optionally with a precision preset and overrides, with a time limit.

    OMP_NUM_THREADS=6 python reproduce.py sidr_drmd_bestfit.ini --pre ~/CLASSpp/pre/act_dr6.pre [--timeout 600] \
        [--drop nonlinear_min_k_max] [--set nonlinear_min_k_max=10 nui1.log10G_eff=-5.5]

The classy on sys.path computes the same inputs that `./class FILE.ini PRE.pre` reads: the keys of the .ini, then
those of the preset (minus --drop), then --set. Each computation runs in a process of its own, which is killed at
the time limit. Prints the wall time and, if it finished, the lensed D_l^TT at l = 1000, 3000 and 6000.
"""
import argparse
import multiprocessing as mp
import time


def read_ini(path):
    pars = {}
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, value = (s.strip() for s in line.split("=", 1))
            if key != "root":
                pars[key] = value
    return pars


def compute(pars, queue):
    import numpy as np
    from classy import Class

    c = Class()
    c.set(pars)
    t = time.time()
    try:
        c.compute()
        cl = c.lensed_cl(6000)
        ell = np.array([1000, 3000, 6000])
        dl = ell * (ell + 1) * cl["tt"][ell] * 2.7255e6 ** 2 / (2 * np.pi)
        queue.put(dict(seconds=time.time() - t, dl=dl.tolist()))
    except Exception as error:                      # noqa: BLE001 - reported
        queue.put(dict(seconds=time.time() - t, error=str(error).strip().splitlines()[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ini")
    ap.add_argument("--pre")
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()
    pars = read_ini(a.ini)
    if a.pre:
        pre = read_ini(a.pre)
        clash = sorted(set(pre) & set(pars))
        if clash:
            raise SystemExit(f"keys in both files (CLASS refuses them): {clash}")
        pars.update({k: v for k, v in pre.items() if k not in a.drop})
    for item in a.set:
        key, value = item.split("=", 1)
        pars[key] = value
    import classy
    print("classy:", classy.__file__)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=compute, args=(pars, queue))
    proc.start()
    proc.join(a.timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        print(f"not finished in {a.timeout:.0f} s")
        return
    r = queue.get()
    if "error" in r:
        print(f"CLASS error after {r['seconds']:.1f} s: {r['error']}")
    else:
        print(f"{r['seconds']:.1f} s; lensed D_l^TT(1000, 3000, 6000) = "
              + ", ".join(f"{x:.4f}" for x in r["dl"]) + " muK^2")


if __name__ == "__main__":
    main()
