"""Residuals of the mimicry fits in units of the per-bin error (Planck+ACT-like weights), and the lensing ratio."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mimic

LMAX, DL = 5000, 60
fig, ax = plt.subplots(4, 2, figsize=(7.2, 8.4), sharex="row", constrained_layout=True)
for j, (case, label) in enumerate([("best", r"best fit, $\log_{10}G_{\rm eff} = -2.81$"),
                                   ("strong", r"strong mode, $\log_{10}G_{\rm eff} = -1.7$")]):
    runs = {r"no SI": ("As_ns", "off"), r"$A_s, n_s$": ("As_ns", "fit"), r"$+\alpha_s$": ("As_ns_as", "fit"),
            r"$+\alpha_s, \beta_s, \theta_s$": ("As_ns_as_bs_theta", "fit")}
    colours = ["0.6", "C0", "C1", "C3"]
    d0 = np.load(f"spectra_{case}_PA_As_ns.npz")
    on = dict(tt=d0["on_tt"], te=d0["on_te"], ee=d0["on_ee"])
    W, use = mimic.weights(on, "P+A", LMAX)
    cov = np.zeros_like(W)
    cov[use] = np.linalg.inv(W[use])
    edges = np.arange(30, 5001, DL)
    for (lab, (tag, which)), c in zip(runs.items(), colours):
        d = np.load(f"spectra_{case}_PA_{tag}.npz")
        for i, x in enumerate(("tt", "te", "ee")):
            r = d[f"{which}_{x}"] - d[f"on_{x}"]
            s2 = cov[:, i, i]
            mids, z = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = use[lo:hi]
                if m.sum() < 5:
                    continue
                w = 1 / s2[lo:hi][m]
                mids.append((lo + hi) / 2)
                z.append(np.sum(w * r[lo:hi][m]) / np.sqrt(np.sum(w)))
            ax[i, j].plot(mids, z, color=c, lw=1.0 if which == "fit" else 0.8, label=lab)
        L = np.arange(30, 2001)
        ax[3, j].plot(L, d[f"{which}_pp"][L] / d["on_pp"][L] - 1, color=c, lw=1.0, label=lab)
    for i, x in enumerate(("TT", "TE", "EE")):
        ax[i, j].axhline(0, color="k", lw=0.5)
        ax[i, j].set_ylabel(rf"$\Delta C_\ell^{{\rm {x}}}/\sigma_{{\rm bin}}$" if j == 0 else "")
        ax[i, j].set_ylim(-6, 6)
    ax[3, j].axhline(0, color="k", lw=0.5)
    ax[3, j].axvspan(40, 763, color="0.9", zorder=0)
    ax[3, j].set_ylabel(r"$C_L^{\phi\phi}/C_L^{\phi\phi,\,\rm SI} - 1$" if j == 0 else "")
    ax[3, j].set_xlabel(r"$L$")
    ax[2, j].set_xlabel(r"$\ell$")
    ax[0, j].set_title(label, fontsize=9)
ax[0, 0].legend(fontsize=7, loc="lower left", title="SI off, refitted:", title_fontsize=7)
fig.suptitle("Self-interaction mimicked by the primordial spectrum (H0WC+SH0ES best fit; bins of 60, P+A-like errors)",
             fontsize=8)
fig.savefig("mimic_residuals.png", dpi=170)
print("written")
