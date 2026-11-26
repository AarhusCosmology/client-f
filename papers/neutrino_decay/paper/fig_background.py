#!/usr/bin/env python3
"""Fig. 1-2 -- how the sector fills, equilibrates and finally decays."""
import numpy as np
import matplotlib.pyplot as plt

import plib
import pstyle as ps

ps.use()
FIG = plib.Path(__file__).resolve().parent / "fig"
FIG.mkdir(exist_ok=True)

MASS = 0.3
T_NCDM = 0.71611
T_CMB_EV = 2.7255 * 8.617333e-5   # K -> eV
# <q>/T for a relativistic FD species with zero chemical potential.
A_NR = 3.1515 * T_NCDM * T_CMB_EV / MASS


def load(G, m=MASS):
    bg = plib.bg_exact(G, f"{m:g}")
    if bg is None:
        return None
    a = 1.0 / (1.0 + bg["z"])
    o = np.argsort(a)
    a = a[o]
    d = dict(a=a,
             rH=bg["(.)rho_dncdm1"][o],
             rl=bg["(.)rho_dr_dncdm1_l"][o],
             rp=bg["(.)rho_dr_dncdm1_phi"][o],
             nH=bg["(.)number_dncdm1"][o],
             bo=bg["bose_occupancy_dncdm1"][o])
    d["rs"] = d["rH"] + d["rl"] + d["rp"]
    return d


def equilibrium_window(a, fphi, Ncom, f_on=0.01):
    """The quasi-equilibrium epoch: from where the boson carries 1% of the
    sector's energy (the reverse reaction now has something to work with) to
    where the comoving parent number starts falling faster than one e-fold per
    e-fold of a (the inverse decays have shut off and the parent is draining).

    Both ends are read off the solution, not imposed.  Freezing of N_H alone is
    not a usable criterion -- it is also frozen before the collision ever
    switches on, and inside the epoch it is not frozen but merely slow
    (dln N/dln a ~ -0.3 against a decay rate many Hubble times fast)."""
    on = np.where(fphi > f_on)[0]
    if not len(on):
        return None
    s = np.gradient(np.log(np.maximum(Ncom, 1e-300)), np.log(a))
    lo_i = int(on[0])
    end = np.where((np.arange(len(a)) > lo_i) & (s < -1.0))[0]
    hi_i = int(end[0]) if len(end) else len(a) - 1
    return a[lo_i], a[hi_i]


# ═══ figure 1: species fractions, one panel per Gamma ════════════════════════
GAMMAS = ["1e5", "1e7", "1e9", "1e11"]
fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.6), sharey=True)
A_LO, A_HI = 1e-7, 3e-1

wins = {}
for ax, G in zip(axes, GAMMAS):
    d = load(G)
    a = d["a"]
    m = (a > A_LO / 3) & (a < 1.0)
    Ncom = d["nH"] * a ** 3
    win = equilibrium_window(a[m], (d["rp"] / d["rs"])[m], Ncom[m] / Ncom[m][0])
    wins[G] = win
    if win:
        ax.axvspan(*win, color=ps.GRID, lw=0, zorder=0)
    ax.axvline(A_NR, color=ps.INK3, lw=0.6, ls=(0, (3, 3)), zorder=1)
    for key, col, lab in (("rH", ps.C_H, r"$\nu_H$"),
                          ("rl", ps.C_L, r"$\nu_\ell$"),
                          ("rp", ps.C_PHI, r"$\varphi$")):
        ax.plot(a[m], d[key][m] / d["rs"][m], color=col, lw=1.4, label=lab,
                solid_capstyle="round", zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(A_LO, A_HI)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"scale factor $a$")
    ax.set_xticks([1e-6, 1e-4, 1e-2])
    ax.text(0.5, 1.02, rf"$\Gamma = 10^{{{G.split('e')[1]}}}\,{{\rm km\,s^{{-1}}\,Mpc^{{-1}}}}$", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7.5)

axes[0].set_ylabel(r"$\bar\rho_i\,/\,\bar\rho_{\rm sec}$")
axes[0].annotate(r"$a_{\rm nr}$", xy=(A_NR, 0.02), xytext=(-2, 0),
                 textcoords="offset points", ha="right", va="bottom",
                 fontsize=6.8, color=ps.INK3)

axes[0].legend(loc="upper left", handlelength=1.5)

axes[-1].annotate("quasi-equilibrium", xy=(np.sqrt(wins["1e11"][0] * wins["1e11"][1]), 0.945),
                  ha="center", va="center", fontsize=6.0, color=ps.INK2)

fig.subplots_adjust(wspace=0.10)
ps.finish(fig, FIG / "fig_bg_fractions.pdf")


# ═══ figure 2: the Gamma ladder ═════════════════════════════════════════════
LADDER = ["1e5", "1e6", "1e7", "1e8", "1e9", "1e10", "1e11"]
cols = ps.seq(len(LADDER), lo=0, hi=9)

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3))
axN, axE, axB = axes

plateau = []
for G, c in zip(LADDER, cols):
    d = load(G)
    if d is None:
        continue
    a = d["a"]
    m = (a > 1e-7) & (a < 1.0)
    aa = a[m]
    Ncom = (d["nH"][m] * aa ** 3)
    Ncom = Ncom / Ncom[0]
    Ecom = (d["rs"][m] * aa ** 4)
    Ecom = Ecom / Ecom[0]
    axN.plot(aa, Ncom, color=c, lw=1.3)
    axE.plot(aa, Ecom, color=c, lw=1.3)
    axB.plot(aa, np.maximum(d["bo"][m], 1e-4), color=c, lw=1.3)
    j = np.argmin(np.abs(aa - 1e-3))
    plateau.append((G, Ncom[j], Ecom[-1]))

for ax in axes:
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1.0)
    ax.set_xlabel(r"scale factor $a$")
    ax.axvline(A_NR, color=ps.INK3, lw=0.6, ls=(0, (3, 3)), zorder=0)

axN.set_ylabel(r"$N_{\nu_H}\,a^3$  (norm.)")
axN.set_ylim(0, 1.08)
axE.set_ylabel(r"$a^4\bar\rho_{\rm sec}$  (norm.)")
axE.set_ylim(0.9, 2.9)
axB.set_ylabel(r"$\max_q \bar f_\varphi$")
axB.set_yscale("log")
axB.set_ylim(1e-3, 1e3)

# The ladder is a magnitude, so it is a sequential ramp with the two ends
# labelled directly rather than an eight-entry legend.
for ax in (axN,):
    ax.annotate(r"$\Gamma_0=10^{5}$", xy=(2.5e-2, 0.55), color=cols[0],
                fontsize=6.8, ha="left")
    ax.annotate(r"$10^{11}$", xy=(1.3e-3, 0.55), color=cols[-1],
                fontsize=6.8, ha="right")
axE.annotate(r"$\Gamma_0=10^{5}$", xy=(2e-2, 2.63), color=cols[0], fontsize=6.8)
axE.annotate(r"$\geq 10^{7}$", xy=(2e-2, 1.32), color=cols[-1], fontsize=6.8)

for ax, letter in zip(axes, "abc"):
    ps.panel(ax, f"({letter})", loc=(-0.03, 1.02))

fig.subplots_adjust(wspace=0.32)
ps.finish(fig, FIG / "fig_bg_ladder.pdf")

print("a_nr =", A_NR)
for g, n, e in plateau:
    print(f"  Gamma={g:5s}  N_H a^3 at a=1e-3: {n:.4f}   final a^4 rho_sec: {e:.4f}")

print("equilibrium windows (a_start, a_end, decades):")
for G, w in wins.items():
    if w:
        print(f"  Gamma={G:5s}  {w[0]:.3e} .. {w[1]:.3e}   {np.log10(w[1]/w[0]):.2f} decades")
