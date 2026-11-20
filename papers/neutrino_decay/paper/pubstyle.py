"""The figures' style, following AarhusCosmology/PublicationPlots (10 November 2026).

- A figure is made at the width it has in the paper: TEXTWIDTH for \\includegraphics[width=\\textwidth] (JCAP, 440 pt).
- All text is set by LaTeX in Computer Modern, at the journal's 11 pt divided by Matplotlib's ratio 12/10 between axis
  labels and other text, so that the figures read at the size of the text around them.
- Classical Matplotlib otherwise: its default colour cycle, full frames, its default line widths.
- No tight_layout and no bbox_inches='tight', which change the width: the margins are measured with pdfcrop and set
  with subplots_adjust (cropsave, from PublicationPlots' ClassExamples/NcdmPlot.py).
"""
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEXTWIDTH = 440 / 72.27          # inches
FONTSIZE = 11 / 1.2
PREAMBLE = r"""
    \usepackage{lmodern}
    \usepackage{amsmath}
    \usepackage{amsfonts}
    \usepackage{mathtools}
"""
FIG = Path(__file__).resolve().parent / "fig"
FIG.mkdir(exist_ok=True)


def use():
    matplotlib.rcdefaults()
    matplotlib.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "cmr10",
        "font.size": FONTSIZE,
        "mathtext.fontset": "cm",
        "text.latex.preamble": PREAMBLE,
        "legend.fontsize": FONTSIZE * 0.9,
        "legend.frameon": False,
        "axes.formatter.use_mathtext": True,
    })


def cropsave(fig, fname, pad=0.01):
    """Save with the margins measured by pdfcrop (PublicationPlots' cropsave), at the figure's own width."""
    frac = 0.3
    fname = Path(fname)
    tmp, tmp_cropped = fname.with_suffix(".tmp.pdf"), fname.with_suffix(".tmp_cropped.pdf")
    plt.figure(fig.number)
    plt.draw()
    state = [(ax.get_xticks(), ax.get_yticks(), ax.get_xlim(), ax.get_ylim(), ax.get_xscale(), ax.get_yscale())
             for ax in fig.axes]

    def restore():
        for ax, (xt, yt, xl, yl, xs, ys) in zip(fig.axes, state):
            if xs == "linear":
                ax.set_xticks(xt)
            if ys == "linear":
                ax.set_yticks(yt)
            ax.set_xlim(xl)
            ax.set_ylim(yl)

    def measure():
        restore()
        fig.savefig(tmp)
        out = subprocess.run(f"pdfcrop --hires {tmp} {tmp_cropped} >/dev/null && pdfinfo {tmp} && pdfinfo {tmp_cropped}",
                             shell=True, capture_output=True, text=True).stdout
        size, cropped = [tuple(map(float, line.split()[2::2])) for line in out.splitlines() if line.startswith("Page size")]
        return cropped[0] / size[0], cropped[1] / size[1]

    fig.subplots_adjust(left=frac, right=1 + frac, bottom=frac, top=1 - frac)
    fac_left, _ = measure()
    fig.subplots_adjust(left=frac, right=1 - frac, bottom=frac, top=1 + frac)
    _, fac_bottom = measure()
    fig.subplots_adjust(left=-frac, right=1 - frac, bottom=frac, top=1 - frac)
    fac_right, _ = measure()
    fig.subplots_adjust(left=frac, right=1 - frac, bottom=-frac, top=1 - frac)
    _, fac_top = measure()
    for f in (tmp, tmp_cropped):
        f.unlink(missing_ok=True)
    left, right = frac + fac_left - 1, 2 - frac - fac_right
    bottom, top = frac + fac_bottom - 1, 2 - frac - fac_top
    width, height = right - left, top - bottom
    fig.subplots_adjust(left=left + pad / 2 * width, right=right - pad / 2 * width,
                        bottom=bottom + pad / 2 * height, top=top - pad / 2 * height)
    restore()
    fig.savefig(fname)
