#!/usr/bin/env python3
"""Fig: the size of the effect the four models are competing to reproduce.

Without this, a reader has no scale against which to judge a 0.1% closure error.
The signal is taken relative to the lowest rung on the ladder rather than to
LCDM: at Gamma = 1e5 the parent is still effectively stable over the epochs that
matter, so this isolates the decay while holding the cosmology fixed.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pstyle as P
import rta
from common import FIG, MASS_LABEL, smooth

P.use()


def draw():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2))
    for ax, mass in zip(axes, ("0.3", "0.06")):
        base = rta.rungs(mass, "x")[0]
        # exact where it exists, m6 above -- drawn in the same ramp but dashed,
        # so the reader sees the ladder continue and sees where it changed hands.
        gl_x = rta.rungs(mass, "x")[1:]
        gl_m = [g for g in rta.rungs(mass, "m6")
                if float(g) > float(rta.rungs(mass, "x")[-1])]
        cols = P.seq(len(gl_x) + len(gl_m))
        for i, gl in enumerate(gl_x + gl_m):
            mdl = "x" if gl in gl_x else "m6"
            e, y = rta.resid(mdl, gl, mass, ref="x" if mdl == "x" else "m6")
            # ratio to the BASE rung, not to the same-rung reference
            a, b = rta.cell(mdl, gl, mass), rta.cell("x", base, mass)
            if a is None or b is None:
                continue
            ell, r = rta.D.cl_ratio(b, a, key="TT")
            ax.plot(ell, 100 * smooth(r, 9), color=cols[i], lw=1.2,
                    ls="-" if mdl == "x" else "--", label=rf"$10^{{{float(gl):g}}}$")
        ax.axhline(0, color=P.INK3, lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\ell$")
        # label and legend in the corner the curves leave free: upper left at 0.3 eV, upper right at 0.06 eV,
        # where the largest Gamma peaks at the top of the panel at ell ~ 20
        left = mass == "0.3"
        ax.text(0.04 if left else 0.96, 0.95, MASS_LABEL[mass], transform=ax.transAxes,
                ha="left" if left else "right", va="top")
        ax.legend(title=r"$\Gamma\ [{\rm km\,s^{-1}\,Mpc^{-1}}]$", ncol=2,
                  loc="upper left" if left else "upper right",
                  bbox_to_anchor=(0.0, 0.88) if left else (1.0, 0.88), fontsize=6.5, title_fontsize=6.5,
                  handlelength=1.6, columnspacing=0.8)
    for ax in axes:
        ax.set_ylabel(r"$C_\ell^{TT}(\Gamma)/C_\ell^{TT}(\Gamma=10^{5})-1$ [\%]")
    fig.subplots_adjust(wspace=0.3)
    P.finish(fig, FIG / "fig_signal.pdf")


if __name__ == "__main__":
    draw()
