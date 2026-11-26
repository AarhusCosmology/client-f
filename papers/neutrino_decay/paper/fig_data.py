"""Figures and numbers of the data section (9-10 November 2026), in the style of pubstyle.py (PublicationPlots).

    python -I fig_data.py [KIT_DIR]     # KIT_DIR: CLIENT_PAPER_KIT, or else this branch's papers/neutrino_decay/

Reads, from the folder the CLiENT jobs ran in (KIT_DIR):
  results/<run>/analysis/posterior_it_14.npz                           network 14's T = 1 chain, prior flat in log10 Gamma
  results/<run>/analysis/linear_prior/posterior_log10Gamma_it_14.npz   the same network sampled with a prior flat in Gamma
  logs/node_breakdown_7063596.json     every node of the 1D profiles re-evaluated with Cobaya, chi^2 by likelihood
  logs/lcdm_minimize_7057592.json      LambdaCDM with nu3 at 0.05 eV and massless
  results_profiles/dncdm_proxy_<ds>_peak_ns/true_surface/   the 2D profile of log10 Gamma and n_s
and writes fig/fig_posterior_gamma.pdf, fig_corner.pdf, fig_profiles.pdf, fig_profile2d.pdf and numbers_data.txt.
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pubstyle  # noqa: E402

pubstyle.use()
import matplotlib.pyplot as plt  # noqa: E402

KIT = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CLIENT_PAPER_KIT") or Path(__file__).resolve().parents[1])
OUT = pubstyle.FIG
W = pubstyle.TEXTWIDTH
RUN = {"pp": "20261106_121906_dncdm_proxy_pp", "wc": "20261106_122443_dncdm_proxy_wc"}
NAME = {"pp": "Planck+BAO+SN", "wc": "SPA+BAO+SN"}
COL = {"pp": "C0", "wc": "C3"}
GLAB = r"$\log_{10}(\Gamma\,/\,{\rm km\,s^{-1}\,Mpc^{-1}})$"
lines = []


def wquant(x, w, q):
    i = np.argsort(x)
    c = np.cumsum(w[i])
    return np.interp(q, c / c[-1], x[i])


def chain(ds, flat):
    f = "analysis/linear_prior/posterior_log10Gamma_it_14.npz" if flat else "analysis/posterior_it_14.npz"
    z = np.load(KIT / "results" / RUN[ds] / f)
    return z["x"].astype(float), [str(n) for n in z["names"]]


# ---------------------------------------------------------------- 1D posterior of log10 Gamma, both priors
fig, ax = plt.subplots(1, 1, figsize=(W, 0.36 * W))
bins = np.linspace(-2, 14, 161)
mid = 0.5 * (bins[1:] + bins[:-1])
for ds in ("pp", "wc"):
    for flat, ls in ((False, "--"), (True, "-")):
        x, names = chain(ds, flat)
        g = x[:, names.index("log10Gamma")]
        h = gaussian_filter1d(np.histogram(g, bins=bins)[0].astype(float), 1.5)
        prior = r"flat in $\Gamma$" if flat else r"flat in $\log_{10}\Gamma$"
        ax.plot(mid, h / h.max(), color=COL[ds], ls=ls, label=f"{NAME[ds]}, {prior}")
ax.set_xlabel(GLAB)
ax.set_ylabel(r"$P/P_{\rm max}$")
ax.set_xlim(-2, 14)
ax.set_ylim(0, 1.05)
ax.legend(loc="upper left")
pubstyle.cropsave(fig, OUT / "fig_posterior_gamma.pdf")
plt.close(fig)

# ---------------------------------------------------------------- numbers of the posteriors
for ds in ("pp", "wc"):
    x, names = chain(ds, False)
    g = x[:, names.index("log10Gamma")]
    xf, nf = chain(ds, True)
    gf = xf[:, nf.index("log10Gamma")]
    w = np.ones_like(gf)
    lines.append(f"{NAME[ds]}: {len(g)} samples flat in log10Gamma, {len(gf)} flat in Gamma")
    lines.append(f"   log10Gamma 95% upper limit, prior flat in log10Gamma: {np.quantile(g, 0.95):.2f}; share above 12: "
                 f"{np.mean(g > 12):.3f} (flat in Gamma {np.mean(gf > 12):.3f})")
    lines.append("   prior flat in Gamma, quantiles of log10Gamma: " + ", ".join(
        f"{q}: {wquant(gf, w, q):.2f} (Gamma {10 ** wquant(gf, w, q):.3g})" for q in (0.025, 0.05, 0.16, 0.5, 0.84, 0.95, 0.975)))
    q = [np.quantile(g, v) for v in (0.16, 0.5, 0.84, 0.95)]
    lines.append(f"   log10Gamma, prior flat in log10Gamma: 16/50/84% {q[0]:.2f} {q[1]:.2f} {q[2]:.2f}, 95% upper {q[3]:.2f}")
    for p in ("omega_b", "omega_cdm", "H0", "logA", "n_s", "tau_reio"):
        lines.append(f"   {p:10s} flat in Gamma {xf[:, nf.index(p)].mean():.5f} +- {xf[:, nf.index(p)].std():.5f}   "
                     f"(flat in log10Gamma {x[:, names.index(p)].mean():.5f} +- {x[:, names.index(p)].std():.5f})")
    lines.append("")

# ---------------------------------------------------------------- corner plot (GetDist), prior flat in Gamma
from getdist import MCSamples, plots  # noqa: E402

samples = []
for ds in ("pp", "wc"):
    xf, nf = chain(ds, True)
    pars = np.column_stack([10 ** (xf[:, nf.index("log10Gamma")] - 12), xf[:, nf.index("n_s")],
                            xf[:, nf.index("H0")], xf[:, nf.index("tau_reio")]])
    samples.append(MCSamples(samples=pars, names=["Gamma12", "n_s", "H0", "tau_reio"],
                             labels=[r"\Gamma\,[10^{12}\,{\rm km\,s^{-1}\,Mpc^{-1}}]", r"n_s",
                                     r"H_0\,[{\rm km\,s^{-1}\,Mpc^{-1}}]", r"\tau_{\rm reio}"],
                             ranges={"Gamma12": [0, None]}, label=NAME[ds],
                             settings={"smooth_scale_2D": 0.4, "smooth_scale_1D": 0.4}))
# GetDist computes the smoothed 1D densities and the 2D densities with their 68% and 95% levels; the plot itself is
# drawn here, in the style of the other figures.
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

PARS = ["Gamma12", "n_s", "H0", "tau_reio"]
LABS = [r"$\Gamma\ [10^{12}\,{\rm km\,s^{-1}\,Mpc^{-1}}]$", r"$n_s$", r"$H_0\ [{\rm km\,s^{-1}\,Mpc^{-1}}]$",
        r"$\tau_{\rm reio}$"]
n = len(PARS)
lims = []
for k, par in enumerate(PARS):
    lo, hi = [], []
    for smp in samples:
        d = smp.get1DDensity(par)
        keep = d.x[d.P > 0.01 * d.P.max()]
        lo.append(keep.min())
        hi.append(keep.max())
    lims.append((0.0 if par == "Gamma12" else min(lo), max(hi)))
fig, axes = plt.subplots(n, n, figsize=(W, W))
for i in range(n):
    for j in range(n):
        ax = axes[i, j]
        if j > i:
            ax.axis("off")
            continue
        for smp, ds in zip(samples, ("pp", "wc")):
            if i == j:
                d = smp.get1DDensity(PARS[i])
                ax.plot(d.x, d.P / d.P.max(), color=COL[ds])
            else:
                d2 = smp.get2DDensity(PARS[j], PARS[i])
                l68, l95 = d2.getContourLevels([0.68, 0.95])
                top = d2.P.max() * 1.01
                ax.contourf(d2.x, d2.y, d2.P, levels=[l95, l68, top], colors=[COL[ds], COL[ds]], alpha=0.25)
                ax.contourf(d2.x, d2.y, d2.P, levels=[l68, top], colors=[COL[ds]], alpha=0.45)
                ax.contour(d2.x, d2.y, d2.P, levels=[l95, l68], colors=[COL[ds]], linewidths=0.8)
        ax.set_xlim(*lims[j])
        ax.xaxis.set_major_locator(MaxNLocator(3, prune="both"))
        if i == j:
            ax.set_ylim(0, 1.08)
            ax.set_yticks([])
        else:
            ax.set_ylim(*lims[i])
            ax.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
        ax.tick_params(labelsize=pubstyle.FONTSIZE * 0.8)
        if i == n - 1:
            ax.set_xlabel(LABS[j])
        else:
            ax.set_xticklabels([])
        if j == 0 and i > 0:
            ax.set_ylabel(LABS[i])
        elif i != j:
            ax.set_yticklabels([])
axes[0, n - 1].legend(handles=[Patch(facecolor=COL[ds], alpha=0.45, edgecolor=COL[ds], label=NAME[ds])
                               for ds in ("pp", "wc")], loc="upper right")
fig.subplots_adjust(left=0.11, right=0.985, bottom=0.1, top=0.985, wspace=0.06, hspace=0.06)
fig.savefig(OUT / "fig_corner.pdf")
plt.close(fig)

# ---------------------------------------------------------------- 1D profiles, the totals of both combinations
# One panel with the total of each combination and its massless limit (27 November 2026, after the ERC CoG figure
# 1b): the contribution of each likelihood is not plotted, only written to numbers_data.txt, which is where the text's
# statements of which data pulls in which direction come from.
BD = json.loads((KIT / "logs/node_breakdown_7063596.json").read_text())["rows"]
LC = json.loads((KIT / "logs/lcdm_minimize_7057592.json").read_text())["summary"]
GROUP = {"planck_2018_lowl.TT": r"Planck low-l TT", "planck_2018_lowl.EE": r"Planck low-l EE",
         "planck_2018_lowl.EE_sroll2": r"Planck low-l EE", "planck_NPIPE_highl_CamSpec.TTTEEE": r"Planck high-l",
         "planckpr4lensing.PlanckPR4Lensing": "CMB lensing", "act_dr6_lenslike.ACTDR6LensLike": "CMB lensing",
         "muse3glike.cobaya.spt3g_2yr_delensed_ee_optimal_pp_muse": "CMB lensing",
         "act_lite_trimmed.ACTDR6CMBonlyTrimmed": "ACT DR6", "candl.interface.CandlCobayaLikelihood": "SPT-3G D1",
         "bao.desi_dr2.desi_bao_all": "DESI BAO", "sn.pantheonplus": "Pantheon+"}
MPC_KM = 3.0856775814913673e19     # 1/(km/s/Mpc) in seconds
prof = {}
fig, ax = plt.subplots(1, 1, figsize=(W, 0.5 * W))
ax.axhline(0, color="0.5", lw=0.6, zorder=1)
massless = {}
for ds in ("pp", "wc"):
    rows = sorted(BD[ds], key=lambda r: r["log10Gamma"])
    gg = np.array([r["log10Gamma"] for r in rows])
    i0 = int(np.argmin(abs(gg + 2)))
    tot = -2 * np.array([r["logpost"] for r in rows])
    tot = tot - tot[i0]
    prof[ds] = (gg, tot, np.array([r["params"]["n_s"] for r in rows]), rows[i0]["logpost"])
    m0 = -2 * (LC[f"{ds}_m0"]["logpost_minus_ln16"] - rows[i0]["logpost"])
    massless[ds] = m0
    ax.axhline(m0, color=COL[ds], lw=1.0, ls=":", zorder=2)
    ax.plot(gg, tot, color=COL[ds], lw=1.8, label=NAME[ds], zorder=3)
    imin = int(np.argmin(tot))
    imin_lo = int(np.argmin(np.where(gg < 10, tot, np.inf)))
    lev = tot[imin] + 4
    cross = [gg[j] + (gg[j + 1] - gg[j]) * (lev - tot[j]) / (tot[j + 1] - tot[j])
             for j in range(imin, len(gg) - 1) if tot[j] < lev <= tot[j + 1]]
    lines.append(f"{NAME[ds]} profile against the stable node: low-Gamma minimum {tot[imin_lo]:.2f} at {gg[imin_lo]:.3f}; "
                 f"global minimum {tot[imin]:.2f} at {gg[imin]:.4f}; massless LambdaCDM {m0:.2f}; Delta chi^2 = 4 above "
                 f"the global minimum at log10Gamma {cross[0] if cross else float('nan'):.2f}")
    # the contribution of each likelihood, between two nodes of the profile
    groups = {}
    for lk in rows[0]["chi2"]:
        v = np.array([r["chi2"][lk] for r in rows])
        groups[GROUP[lk]] = groups.get(GROUP[lk], 0) + v
    def breakdown(i, j, what):
        parts = ", ".join(f"{n} {groups[n][j] - groups[n][i]:+.2f}" for n in groups)
        lines.append(f"  {NAME[ds]} {what} (log10Gamma {gg[i]:.3f} -> {gg[j]:.3f}): total {tot[j] - tot[i]:+.2f}; {parts}")
    breakdown(i0, imin_lo, "stable -> low-Gamma minimum")
    breakdown(i0, int(np.argmin(abs(gg - 11.8))), "stable -> 11.8")
    breakdown(imin_lo, int(np.argmin(abs(gg - 12.8125))), "low-Gamma minimum -> the SPA+BAO+SN peak")
ax.text(-1.8, min(massless.values()) - 0.2, r"$\Lambda$CDM, massless $\nu_3$", ha="left", va="top",
        fontsize=pubstyle.FONTSIZE * 0.8)
ax.text(-1.8, 0.2, r"stable $\nu_3$", ha="left", va="bottom", fontsize=pubstyle.FONTSIZE * 0.8)
ax.legend(loc="upper left", handlelength=1.8)
ax.set_xlabel(GLAB)
ax.set_ylabel(r"$\Delta\chi^2$")
ax.set_xlim(-2.1, 13.85)
ax.set_ylim(-6.6, 5.0)
top = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
top.set_xticks([np.log10(MPC_KM) - k for k in (18, 15, 12, 9, 6)],
               labels=[rf"$10^{{{k}}}$" for k in (18, 15, 12, 9, 6)])
top.set_xticks([np.log10(MPC_KM) - k for k in range(6, 22) if -2.1 < np.log10(MPC_KM) - k < 13.85], minor=True)
top.set_xlabel(r"$\tau_0\,[{\rm s}]$")
pubstyle.cropsave(fig, OUT / "fig_profiles.pdf")
plt.close(fig)

# ---------------------------------------------------------------- 2D profile of log10 Gamma and n_s
fig, axes = plt.subplots(1, 2, figsize=(W, 0.45 * W), sharey=True)
for ax, ds in zip(axes, ("pp", "wc")):
    s = KIT / "results_profiles" / f"dncdm_proxy_{ds}_peak_ns" / "true_surface"
    method = json.load(open(s / "method.json"))
    xa, ya = [np.array(a, float) for a in method["axes"]]
    logl = np.full((len(xa), len(ya)), np.nan)
    for f in glob.glob(str(s / "node_*.json")):
        i, j = map(int, Path(f).stem.split("_")[1:3])
        logl[i, j] = json.load(open(f))["logl"]
    gg, tot, ns1, lp0 = prof[ds]
    ref = max(np.nanmax(logl), lp0 - 0.5 * tot.min())        # the best true value of either study
    d2 = -2 * (logl - ref)
    done = np.isfinite(d2)
    from matplotlib.colors import BoundaryNorm
    levels = [0, 1, 2, 4, 6, 9, 16, 25, 100]
    cs = ax.pcolormesh(xa, ya, np.ma.masked_invalid(d2).T, shading="nearest", cmap="Blues_r",
                       norm=BoundaryNorm(levels, plt.get_cmap("Blues_r").N, extend="max"))
    if done.all():
        ax.contour(xa, ya, d2.T, levels=[1, 4, 9], colors="k", linewidths=0.7)
    m = (gg >= xa[0]) & (gg <= xa[-1])
    ax.plot(gg[m], ns1[m], color="C3", lw=1.2, ls="--", label=r"$n_s$ along the 1D profile")
    ax.set_xlim(xa[0], xa[-1])
    ax.set_ylim(ya[0], ya[-1])
    ax.set_xlabel(GLAB)
    ax.text(0.04, 0.96, NAME[ds], transform=ax.transAxes, ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="0.6", boxstyle="round,pad=0.3"))
    if ds == "wc":                                        # lower right is empty in this panel
        ax.legend(loc="lower right", fontsize=pubstyle.FONTSIZE * 0.8)
    lines.append(f"{NAME[ds]} 2D profile log10Gamma x n_s: {done.sum()} of {done.size} nodes; best {ref:.3f}")
axes[0].set_ylabel(r"$n_s$")
fig.subplots_adjust(left=0.09, right=0.88, bottom=0.17, top=0.97, wspace=0.06)
cax = fig.add_axes([0.9, 0.17, 0.02, 0.8])
fig.colorbar(cs, cax=cax, label=r"$\Delta\chi^2$", ticks=[0, 1, 2, 4, 6, 9, 16, 25])
fig.savefig(OUT / "fig_profile2d.pdf")
plt.close(fig)

(OUT / "numbers_data.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
