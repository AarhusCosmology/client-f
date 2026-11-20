#!/usr/bin/env python3
"""Fig: how each model's error runs with Gamma, in C_l^TT and in P(k).

One number per cell -- the band rms of the residual against the best available
reference -- so the whole campaign fits on one axis. The shaded band is the
exact cell's own error, and the vertical rule marks the exact ceiling: to its
right the reference is the Galerkin m6 cell and the m6 curve is absent by
construction, because a model cannot be scored against itself.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pstyle as P
import rta
from common import FIG, FULL, MASS_LABEL, STYLE, above_ceiling, exact_rungs

P.use()
MODELS = ["m6", "ours", "copw"]


def series(mass, mdl, kind="cl", band=FULL, which="exact"):
    """Band-rms residual per rung, for ONE reference.

    `which="exact"` uses only rungs with an exact cell; `which="above"` uses only
    rungs above the exact ceiling, where the reference is m6. They are never
    joined into one line: the reference alternates rung by rung below the
    ceiling (see common.ref_for), so a single line through every rung traces the
    reference and not the model.
    """
    gls = exact_rungs(mass, mdl) if which == "exact" else above_ceiling(mass, mdl)
    ref = "x" if which == "exact" else "m6"
    g, v = [], []
    for gl in gls:
        if ref == mdl:
            continue
        if kind == "cl":
            x, y = rta.resid(mdl, gl, mass, ref=ref)
            if x is None:
                continue
            r = rta.rms(x, y, *band)
        else:
            x, y = rta.resid_pk(mdl, gl, mass, ref=ref)
            if x is None:
                continue
            m = np.isfinite(y)
            r = float(np.sqrt(np.mean(y[m] ** 2))) if m.any() else np.nan
        if np.isfinite(r):
            g.append(float(gl))
            v.append(100 * r)
    return np.array(g), np.array(v)


def exact_ceiling(mass):
    r = rta.rungs(mass, "x")
    return float(r[-1]) if r else np.nan


def own_error(mass, band=FULL):
    g, v = [], []
    for gl in rta.rungs(mass, "x"):
        e, s = rta.D.cl_error(mass, gl, order=2.0)
        if e is None:
            continue
        m = (e >= band[0]) & (e <= band[1])
        g.append(float(gl))
        v.append(100 * float(np.sqrt(np.mean(s[m] ** 2))))
    return np.array(g), np.array(v)


def draw():
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.2), sharex=True)
    for j, kind in enumerate(("cl", "pk")):
        for i, mass in enumerate(("0.3", "0.06")):
            ax = axes[i][j]
            for mdl in MODELS:
                st = STYLE[mdl]
                g, v = series(mass, mdl, kind, which="exact")
                if len(g):
                    ax.plot(g, v, color=st["c"], ls=st["ls"], lw=st["lw"], marker="o", ms=3.0, label=st["label"])
                # above the ceiling: open markers, no line joining them to the series referenced to the full
                # solution, because they are a different measurement
                ga, va = series(mass, mdl, kind, which="above")
                if len(ga):
                    ax.plot(ga, va, color=st["c"], ls="none", marker="o", ms=3.4, mfc="none", mew=1.0)
            if kind == "cl":
                ge, ve = own_error(mass)
                if len(ge):
                    ax.plot(ge, ve, color=P.INK3, lw=0.9, ls=(0, (1, 1.6)), label="full solution, own error")
            xc = exact_ceiling(mass)
            if np.isfinite(xc):
                ax.axvline(xc, color=P.INK3, lw=0.7, ls="-.")
            ax.set_yscale("log")
            ax.text(0.96, 0.05, MASS_LABEL[mass], transform=ax.transAxes, ha="right", va="bottom")
            if j == 0:
                ax.set_ylabel(r"rms $\Delta C_\ell^{TT}/C_\ell^{TT}$ [\%]")
            else:
                ax.set_ylabel(r"rms $\Delta P/P$ [\%]")
            if i == 1:
                ax.set_xlabel(r"$\log_{10}(\Gamma\,/\,{\rm km\,s^{-1}Mpc^{-1}})$")
    # One legend for the four panels, above them; open symbols are explained in the caption.
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995), handlelength=2.4)
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.085, top=0.87, wspace=0.42, hspace=0.08)
    P.finish(fig, FIG / "fig_ladder.pdf", crop=False)


if __name__ == "__main__":
    draw()
