"""Compare the high-omega ladder (ladder3.py): the oscillation signal (A_osc 0.02 minus A_osc 0) of each setting against
the densest one, as a noise-weighted chi^2 (the norm of sampling_test/compare2.py and compare_wc.py):
  pp: Planck-like noise (30 and 60 uK-arcmin, 7' beam, f_sky 0.6), TT to l = 2500, TE and EE to 2000;
  wc: ACT-like noise (15 and 21 uK-arcmin, 1.4' beam, f_sky 0.3), TT, TE and EE at l = 30-6327.
The noise is that of a single unbinned experiment, so the numbers are upper bounds on what binned likelihoods see.

    python compare3.py pp      (reads pp_*.npz)
    python compare3.py wc      (reads wc_*.npz)
"""
import glob
import sys
from pathlib import Path

import numpy as np

DATA = sys.argv[1]
here = Path(__file__).parent
files = {Path(f).stem.split("_", 1)[1]: np.load(f) for f in sorted(glob.glob(str(here / f"{DATA}_*.npz")))}
order = ["P", "PK", "PQ", "L4K", "L8Q200", "L8Q", "L4Q200", "L4Q", "REF", "REFP"] if DATA == "pp" else ["W", "WK", "WQ", "WL200", "WL", "WLQ", "WREF"]
names = [n for n in order if n in files]
top = names[-1]
T0 = 2.7255e6
if DATA == "pp":
    lmax, fsky, beam_arcmin, nt, npol = dict(tt=2500, te=2000, ee=2000), 0.6, 7.0, 30.0, 60.0
    lmin = 2
else:
    lmax, fsky, beam_arcmin, nt, npol = dict(tt=6327, te=6327, ee=6327), 0.3, 1.4, 15.0, 21.0
    lmin = 30
first = files[top]
L = len(first["0_tt"])
l = np.arange(L)
beam = np.exp(l * (l + 1) * (beam_arcmin / 60 * np.pi / 180) ** 2 / (8 * np.log(2)))
NT = (nt / 60 * np.pi / 180) ** 2 * beam / T0 ** 2
NP = (npol / 60 * np.pi / 180) ** 2 * beam / T0 ** 2
tt, te, ee = first["0_tt"], first["0_te"], first["0_ee"]
with np.errstate(divide="ignore", invalid="ignore"):
    var = dict(tt=2 / ((2 * l + 1) * fsky) * (tt + NT) ** 2, ee=2 / ((2 * l + 1) * fsky) * (ee + NP) ** 2,
               te=(te ** 2 + (tt + NT) * (ee + NP)) / ((2 * l + 1) * fsky))


def norm(diff, parts=("tt", "te", "ee")):
    return float(sum(np.sum(diff[s][lmin:lmax[s] + 1] ** 2 / var[s][lmin:lmax[s] + 1]) for s in parts))


def signal(d, k):
    return {s: d[f"{k}_{s}"] - d[f"0_{s}"] for s in ("tt", "te", "ee")}


cases = first["cases"]
print(f"== {DATA}: seconds per call (one thread) and peak memory: "
      + ", ".join(f"{n} {np.median(files[n]['times']):.1f} s {float(files[n]['rss_gb']):.1f} GB" for n in names))
lcdm = {n: norm({s: files[n][f"0_{s}"] - first[f"0_{s}"] for s in ("tt", "te", "ee")}) for n in names[:-1]}
print(f"   LambdaCDM spectra (A_osc 0) against {top}: " + ", ".join(f"{n} {v:.3f}" for n, v in lcdm.items()))
print(f"   oscillation signal (A_osc 0.02) and the error of each setting against {top} (same norm):")
for k in range(1, len(cases)):
    A, w, p = cases[k]
    ref = signal(first, k)
    errs = []
    for n in names[:-1]:
        e = norm({s: signal(files[n], k)[s] - ref[s] for s in ref})
        errs.append(f"{n} {e:7.3f}")
    print(f"   omega {w:5.0f} phi {p:.2f}: signal {norm(ref):7.2f} (TT {norm(ref, ('tt',)):6.2f})  " + "  ".join(errs))
