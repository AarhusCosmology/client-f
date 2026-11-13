"""How the binning of the high-l CMB likelihoods reduces their sensitivity to a fast oscillation (9 November 2026).

    python analysis/binning200.py        # from paper/; seconds; writes plots/data/binning200.json

The imprint of the oscillation is the difference of the lensed spectra with A_osc = 0.02 and 0 at omega = 60-200 from
CLASS++ with the converged sampling (the CLASS++ ladder of results/20261102_high_omega_ladder: pp_REFP, phases 0 and
pi/2, for PDP+; wc_WREF for H0WC), as in Table tab:classpp_cl. Its chi^2 is computed for each likelihood twice: with the
data as they are compared with the theory, and as if the same data were given multipole by multipole. The ratio is the
fraction of the oscillation's chi^2 that survives the binning.
  - CamSpec PR4 (Cobaya's planck_NPIPE_highl_CamSpec): multipole by multipole (its data vector and covariance hold every
    multipole of 30 <= l <= 2500 in TT, 30-2000 in TE and EE), so the ratio is 1. Noise: the Planck-like model of
    build200.py's ladder_norms (f_sky 0.6, 7 arcmin, 30 and 60 muK arcmin).
  - SPT-3G D1, the foreground-marginalised version (SPTlite in candl): its own window functions (TT 52, TE and EE 72 bins
    of Delta l = 50 from l = 400) and covariance. The variance of a single multipole is taken as N_b times the variance
    of its bin, and both chi^2 use the diagonal of the covariance.
  - ACT-lite: its 45 top-hat bins, l = 576-6325 (Delta l = 50 up to 2025, then 100, 200 and 400), as printed on Grendel
    by grendel/act_lite_bins.py from ACTDR6CMBonly v1.0 (dr6_data_cmbonly.fits, 9 November 2026); they are bins of the
    multi-frequency DR6 data on the laptop (ACTDR6MFLike v1.0), which this script checks. Noise: the ACT-like model of
    ladder_norms (f_sky 0.3, 1.4 arcmin, 15 and 21 muK arcmin).
H0WC uses CamSpec only at l < 1000 in TT and l < 600 in TE and EE; its budget adds the three parts as independent.
"""
import json
from pathlib import Path

import os

import numpy as np
from astropy.io import fits

HERE = Path(__file__).resolve().parent.parent
LADDER = Path(os.environ.get("CLIENT_PAPER_DATA", HERE.parent / "data")) / "results/20261102_high_omega_ladder"
PACKAGES = Path(os.environ["COBAYA_PACKAGES_PATH"]) / "data"           # the likelihood data, as Cobaya installs it
SPT = PACKAGES / "spt_candl_data/spt_candl_data/SPT3G_D1_TnE_v0/lite"
ACT = PACKAGES / "ACTDR6MFLike/v1.0/dr6_data.fits"
T0 = 2.7255e6


def noise_var(ref, fsky, beam, nt, npol):
    ell = np.arange(len(ref["0_tt"]))
    b = np.exp(ell * (ell + 1) * (beam / 60 * np.pi / 180) ** 2 / (8 * np.log(2)))
    NT, NP = ((x / 60 * np.pi / 180) ** 2 * b / T0 ** 2 for x in (nt, npol))
    tt, te, ee = ref["0_tt"], ref["0_te"], ref["0_ee"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(tt=2 / ((2 * ell + 1) * fsky) * (tt + NT) ** 2, ee=2 / ((2 * ell + 1) * fsky) * (ee + NP) ** 2,
                    te=(te ** 2 + (tt + NT) * (ee + NP)) / ((2 * ell + 1) * fsky))


def unbinned(diff, var, ranges):
    return sum(float(np.sum(diff[s][lo:hi + 1] ** 2 / var[s][lo:hi + 1])) for s, (lo, hi) in ranges.items())


def binned_tophat(diff, var, bins):
    """bins: spectrum -> list of (lo, hi), top hat in C_l (the same as top hat in D_l up to the slope within a bin)."""
    out = 0.0
    for s, bl in bins.items():
        for lo, hi in bl:
            n = hi - lo + 1
            out += float(np.mean(diff[s][lo:hi + 1]) ** 2 / (np.sum(var[s][lo:hi + 1]) / n ** 2))
    return out


# ACT-lite's bins (ACTDR6CMBonly v1.0, the same for TT, TE and EE), from grendel/act_lite_bins.py on 9 November 2026
ACT_LITE = ([(576 + 50 * i, 625 + 50 * i) for i in range(29)] + [(2026 + 100 * i, 2125 + 100 * i) for i in range(5)]
            + [(2526 + 200 * i, 2725 + 200 * i) for i in range(3)] + [(3126 + 400 * i, 3525 + 400 * i) for i in range(8)])


def act_bins():
    """ACT-lite's bins, checked to be bins of the multi-frequency DR6 data on the laptop."""
    assert len(ACT_LITE) == 45 and ACT_LITE[28] == (1976, 2025) and ACT_LITE[-1] == (5926, 6325)
    h = fits.open(ACT)
    d = h[3].data
    ell, W = np.array(d["values"]), np.array(d["weight"])
    mf = {(int(ell[W[:, b] > 0].min()), int(ell[W[:, b] > 0].max())) for b in range(W.shape[1])}
    assert set(ACT_LITE) <= mf, "ACT-lite's bins are not bins of the multi-frequency data"
    return {s: ACT_LITE for s in ("tt", "te", "ee")}


def spt_lite():
    """The windows (on l = 2 ... ) and the diagonal of the covariance of SPTlite, D_l in muK^2, in the order TT, TE, EE."""
    cov = np.loadtxt(SPT / "SPT3G_D1_TnE_lite_cov.txt")
    out, k = {}, 0
    for s, n in (("tt", 52), ("te", 72), ("ee", 72)):
        w = np.loadtxt(SPT / f"windows/{s.upper()}_lxl_window_functions.txt")
        out[s] = dict(ell=w[:, 0].astype(int), W=w[:, 1:], var=np.diag(cov)[k:k + n])
        assert w.shape[1] - 1 == n
        k += n
    return out


def spt_chi2(diff_D, spt):
    """diff_D: spectrum -> Delta D_l on l = 0 ... ; binned and unbinned (N_b times the bin variance per multipole)."""
    b_chi2 = u_chi2 = 0.0
    for s, d in spt.items():
        ell, W, var = d["ell"], d["W"], d["var"]
        x = diff_D[s][ell]
        for j in range(W.shape[1]):
            w = W[:, j]
            core = w > 0.2 * w.max()           # the bin's Delta l = 50 (the files carry tiny tails over all l)
            nb = int(np.sum(core))
            b_chi2 += float((w @ x / w.sum()) ** 2 / var[j])
            u_chi2 += float(np.sum(x[core] ** 2) / (nb * var[j]))
    return b_chi2, u_chi2


out = {}
for data, ref_name in (("pp", "pp_REFP"), ("wc", "wc_WREF")):
    ref = np.load(LADDER / f"{ref_name}.npz")
    ell = np.arange(len(ref["0_tt"]))
    var_planck = noise_var(ref, 0.6, 7.0, 30.0, 60.0)
    var_act = noise_var(ref, 0.3, 1.4, 15.0, 21.0)
    planck_ranges = (dict(tt=(30, 2500), te=(30, 2000), ee=(30, 2000)) if data == "pp" else
                     dict(tt=(30, 999), te=(30, 599), ee=(30, 599)))
    abins = act_bins()
    act_ranges = {s: (abins[s][0][0], abins[s][-1][1]) for s in abins}
    spt = spt_lite() if data == "wc" else None
    rows = {}
    for k, (A, w, p) in enumerate(ref["cases"]):
        if k == 0:
            continue
        diff = {s: ref[f"{k}_{s}"] - ref[f"0_{s}"] for s in ("tt", "te", "ee")}
        r = rows.setdefault(float(w), dict(planck=0.0, act_binned=0.0, act_unbinned=0.0, spt_binned=0.0, spt_unbinned=0.0, phases=0))
        r["planck"] += unbinned(diff, var_planck, planck_ranges)
        if data == "wc":
            r["act_binned"] += binned_tophat(diff, var_act, abins)
            r["act_unbinned"] += unbinned(diff, var_act, act_ranges)
            diff_D = {s: ell * (ell + 1) / (2 * np.pi) * diff[s] * T0 ** 2 for s in diff}
            b, u = spt_chi2(diff_D, spt)
            r["spt_binned"] += b
            r["spt_unbinned"] += u
        r["phases"] += 1
    for r in rows.values():                      # the mean over the phases computed
        n = r.pop("phases")
        for key in r:
            r[key] /= n
        if data == "wc":
            r["act_ratio"] = r["act_binned"] / r["act_unbinned"]
            r["spt_ratio"] = r["spt_binned"] / r["spt_unbinned"]
    out[data] = rows
    print(data)
    for w, r in sorted(rows.items()):
        print(f"  omega {w:5.0f}: " + ", ".join(f"{k} {v:.3g}" for k, v in r.items()))
(HERE / "plots/data/binning200.json").write_text(json.dumps(out, indent=1) + "\n")
print("written", HERE / "plots/data/binning200.json")
