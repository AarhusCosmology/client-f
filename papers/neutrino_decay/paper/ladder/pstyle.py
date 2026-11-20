"""The figure helpers of the notes' scripts (pstyle: use, finish, panel, label_end, CAT, INK, seq, ...), redone in
the paper's classical Matplotlib style (pubstyle.py, after AarhusCosmology/PublicationPlots). The same file is copied
into tools/ladder/ and tools/rate/, so that the scripts taken from rtanote and proxynote draw in this style. The
palette versions they came with are in tools/old_style/.
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

_here = Path(__file__).resolve().parent
for _p in (_here, _here.parent):
    if (_p / "pubstyle.py").exists():
        sys.path.insert(0, str(_p))
import pubstyle  # noqa: E402

CAT = [c["color"] for c in matplotlib.rcParamsDefault["axes.prop_cycle"]]     # Matplotlib's own cycle, C0 ... C9
C_H, C_L, C_PHI = CAT[0], CAT[1], CAT[2]
C_EXACT, C_MOM, C_RTA, C_OFF = "k", CAT[0], CAT[1], CAT[6]
INK, INK2, INK3, GRID, SURF = "k", "0.3", "0.5", "0.85", "white"


def seq(n, lo=0, hi=None):
    """n colours of a sequential map, light to dark (for a ladder in Gamma)."""
    return [plt.cm.viridis_r(v) for v in np.linspace(0.12, 0.95, n)]


def use():
    pubstyle.use()


def panel(ax, letter, title=None, loc=(0.0, 1.02)):
    ax.text(loc[0], loc[1], f"({letter.strip('()')})" if letter else "", transform=ax.transAxes, ha="left",
            va="bottom")


def label_end(ax, x, y, text, color, dx=2.0, dy=0.0, ha="left", va="center", fs=None):
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points", color=color,
                fontsize=fs or pubstyle.FONTSIZE * 0.85, ha=ha, va=va)


def finish(fig, path, w=None, h=None, crop=True):
    """At the paper's text width, keeping the aspect ratio the script asked for; margins by pdfcrop, or, with
    crop=False, the script's own subplots_adjust (a figure legend outside the axes)."""
    w0, h0 = (w, h) if w is not None else fig.get_size_inches()
    fig.set_size_inches(pubstyle.TEXTWIDTH, pubstyle.TEXTWIDTH * h0 / w0)
    if crop:
        pubstyle.cropsave(fig, path)
    else:
        fig.savefig(path)
    plt.close(fig)
    print("wrote", path)
