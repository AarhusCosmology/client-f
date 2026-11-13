"""PDP+ ladder: CLASS++'s fast lensing (the PDP+ runs: the correlation function integrated on theta < pi/16) against
accurate_lensing = 1 (the H0WC runs'), in the norm of compare3.py: REF against REFL (the lensing alone, converged
sampling) and P, PL against REFL (the runs, and the runs with accurate lensing).

    python compare_lensing.py
"""
from pathlib import Path

import numpy as np

here = Path(__file__).parent
d = {n: np.load(here / f"pp_{n}.npz") for n in ("P", "PL", "REF", "REFL", "L8Q")}
T0 = 2.7255e6
l = np.arange(len(d["REFL"]["0_tt"]))
beam = np.exp(l * (l + 1) * (7.0 / 60 * np.pi / 180) ** 2 / (8 * np.log(2)))
NT = (30.0 / 60 * np.pi / 180) ** 2 * beam / T0 ** 2
NP = (60.0 / 60 * np.pi / 180) ** 2 * beam / T0 ** 2
tt, te, ee = d["REFL"]["0_tt"], d["REFL"]["0_te"], d["REFL"]["0_ee"]
with np.errstate(divide="ignore", invalid="ignore"):
    var = dict(tt=2 / ((2 * l + 1) * 0.6) * (tt + NT) ** 2, ee=2 / ((2 * l + 1) * 0.6) * (ee + NP) ** 2,
               te=(te ** 2 + (tt + NT) * (ee + NP)) / ((2 * l + 1) * 0.6))
lmax = dict(tt=2500, te=2000, ee=2000)
norm = lambda x: float(sum(np.sum(x[s][2:lmax[s] + 1] ** 2 / var[s][2:lmax[s] + 1]) for s in ("tt", "te", "ee")))
signal = lambda n, k: {s: d[n][f"{k}_{s}"] - d[n][f"0_{s}"] for s in ("tt", "te", "ee")}
print("seconds per call: " + ", ".join(f"{n} {np.median(v['times']):.1f}" for n, v in d.items()))
print("LambdaCDM spectra against REFL: " + ", ".join(
    f"{n} {norm({s: d[n][f'0_{s}'] - d['REFL'][f'0_{s}'] for s in ('tt', 'te', 'ee')}):.3f}" for n in ("P", "PL", "REF")))
for k, (A, w, p) in enumerate(d["REFL"]["cases"]):
    if k == 0:
        continue
    ref = signal("REFL", k)
    print(f"omega {w:5.0f} phi {p:.2f}: signal {norm(ref):6.2f}  error against REFL: "
          + "  ".join(f"{n} {norm({s: signal(n, k)[s] - ref[s] for s in ref}):.3f}" for n in ("REF", "P", "PL", "L8Q")))
