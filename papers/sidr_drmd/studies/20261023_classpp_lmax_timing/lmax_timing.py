"""What l_max costs in CLASS++, module by module (23 October 2026).

    OMP_NUM_THREADS=1 PYTHONPATH=CLASSPP/_install python lmax_timing.py --pre CLASSPP/pre/act_dr6.pre

classy builds its modules lazily, so compute(level=[name]) for the modules in order times each one alone. For two
points (a LambdaCDM point and the SIDR + DRMD best fit of ../20261023_classpp_act_preset_timing/sidr_drmd_bestfit.ini),
two settings sets and four l_max, with the output Cobaya asks for in our runs (tCl pCl lCl mPk, P_k_max_1/Mpc = 1,
lensing). Each case is computed twice in a fresh instance and the faster is kept. Writes lmax_timing.json beside this
script.

  old  the earlier runs: halofit, evolver_perturbations 1, tol_perturb_integration 1e-7, defaults otherwise
  new  the new runs: old + pre/act_dr6.pre without its nonlinear_min_k_max = 70, and nonlinear_min_k_max = 10
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "20261023_classpp_act_preset_timing"))
from reproduce import read_ini  # noqa: E402

MODULES = ["background", "thermodynamics", "perturb", "primordial", "nonlinear", "transfer", "spectra", "lensing"]
LMAX = [2500, 4000, 6500, 9001]


def points():
    sidr = {k: v for k, v in read_ini(HERE.parent / "20261023_classpp_act_preset_timing" / "sidr_drmd_bestfit.ini").items()
            if k not in ("output", "lensing", "l_max_scalars", "non_linear", "evolver_perturbations",
                         "tol_perturb_integration")}
    lcdm = dict(H0=67.42, A_s=2.1178785569948e-09, n_s=0.9672, omega_b=0.02228, omega_cdm=0.1196, tau_reio=0.0611,
                N_ur=2.0328, N_ncdm=1, m_ncdm=0.06, T_ncdm=0.71611)
    return {"LCDM": lcdm, "SIDR + DRMD best fit": sidr}


def time_modules(pars):
    from classy import Class

    c = Class()
    c.set(pars)
    out = {}
    for m in MODULES:
        t = time.perf_counter()
        c.compute(level=[m])
        out[m] = time.perf_counter() - t
    c.struct_cleanup()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    a = ap.parse_args()
    import classy
    pre = {k: v for k, v in read_ini(a.pre).items() if k != "nonlinear_min_k_max"}
    base = {"output": "tCl pCl lCl mPk", "P_k_max_1/Mpc": 1, "lensing": "yes", "non_linear": "halofit",
            "evolver_perturbations": 1, "tol_perturb_integration": 1e-7}
    sets = {"old": {}, "new": {**pre, "nonlinear_min_k_max": 10}}
    results = []
    print(f"classy {classy.__version__} from {classy.__file__}")
    print(f"{'point':22s} {'set':4s} {'l_max':>5s} {'total':>7s}  " + " ".join(f"{m[:8]:>8s}" for m in MODULES), flush=True)
    for pname, p in points().items():
        for sname, s in sets.items():
            for lmax in LMAX:
                runs = [time_modules({**p, **base, **s, "l_max_scalars": lmax}) for _ in range(2)]
                best = min(runs, key=lambda r: sum(r.values()))
                results.append(dict(point=pname, set=sname, l_max=lmax, seconds=best, total=sum(best.values())))
                print(f"{pname:22s} {sname:4s} {lmax:5d} {sum(best.values()):7.2f}  "
                      + " ".join(f"{best[m]:8.2f}" for m in MODULES), flush=True)
    (HERE / "lmax_timing.json").write_text(json.dumps(dict(classy=classy.__file__, version=classy.__version__,
                                                           threads=1, results=results), indent=1))


if __name__ == "__main__":
    main()
