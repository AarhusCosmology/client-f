"""Shared settings of the plotting scripts: data sets, colours, parameter labels, and the data files in data/.

Every figure of the paper is made by one script in this folder (plot_*.py) from the files in data/ alone, with numpy,
matplotlib and GetDist; `python make_all.py` makes them all. The figures are written to ../Figures/, where the LaTeX
source reads them. The data files are written by ../build.py from the runs (see README.md).

PAPER_SERIES=wc in the environment (make_all.py --series wc) switches to the runs on the H0 World Cup data of Sec. 5:
data from data/wc/ (written by ../build_wc.py), data sets wc and wcmb, and every figure name ending in _wc (or in the
data set's own name). Without it everything is as before.
"""
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SERIES = os.environ.get("PAPER_SERIES", "drmd2")
DATA_DIR = HERE / "data" / ("wc" if SERIES == "wc" else "")
FIGURES = HERE.parent / "Figures"

if SERIES == "wc":
    DATA = {"wc": r"$H_0{\rm WC}$", "wcmb": r"$H_0{\rm WC}$+SH0ES"}
    COLOURS = {"wc": "#2b6cb0", "wcmb": "#c0392b"}
else:
    DATA = {"pp": "PDP+", "ppshoes": "PDP+SH0ES"}
    COLOURS = {"pp": "#2b6cb0", "ppshoes": "#c0392b"}
COSMO = ["omega_b", "omega_cdm", "H0", "logA", "n_s", "tau_reio", "m_int", "logGeff", "delta_Neff_drmd",
         "f_idm_drmd", "log10_zstop", "log10_G_over_aH"]
LABELS = {"omega_b": r"\omega_\mathrm{b}", "omega_cdm": r"\omega_\mathrm{cdm}", "H0": r"H_0",
          "logA": r"\ln(10^{10}A_\mathrm{s})", "n_s": r"n_\mathrm{s}", "tau_reio": r"\tau_\mathrm{reio}",
          "m_int": r"m_\mathrm{int}", "logGeff": r"\log_{10}(G_\mathrm{eff}\,\mathrm{MeV}^2)",
          "delta_Neff_drmd": r"\Delta N_\mathrm{eff}", "f_idm_drmd": r"f_\mathrm{idm}",
          "log10_zstop": r"\log_{10}z_\mathrm{stop}", "log10_G_over_aH": r"\log_{10}(\mathcal{G}/\mathcal{H})_\mathrm{ini}",
          "sigma8": r"\sigma_8", "S8": r"S_8", "Omega_m": r"\Omega_\mathrm{m}", "rs_drag": r"r_\mathrm{drag}",
          "f_idr_drmd": r"f_\mathrm{idr}", "log10_zdec": r"\log_{10}z_\mathrm{dec}"}
UNITS = {"H0": r"\;[\mathrm{km\,s^{-1}\,Mpc^{-1}}]", "m_int": r"\;[\mathrm{eV}]", "rs_drag": r"\;[\mathrm{Mpc}]"}
# Axis ranges of the posterior figures where the prior's would leave most of a panel empty (27 October 2026); every
# other axis is GetDist's choice from the samples. f_idm (prior up to 0.4): above 0.15 lie 1.7% of the PDP+
# posterior, a thin tail to the prior's edge, and at most 0.14% of the others (../build.py's macro <tag>fidmtail, quoted
# in the captions). m_int in the World Cup series (prior up to 0.3 eV): 0.01% above 0.2 eV; with PDP+ and PDP+SH0ES 2.6-3.7%,
# so that axis keeps the prior there.
PLOT_LIMITS = dict({"f_idm_drmd": (0.0, 0.15)}, **({"m_int": (0.06, 0.2)} if SERIES == "wc" else {}))


def load_json(name):
    return json.loads((DATA_DIR / name).read_text())


def posterior(d):
    """The marginalised posterior of data set d as GetDist samples, with the settings the paper uses."""
    from getdist import MCSamples

    z = np.load(DATA_DIR / f"posterior_{d}.npz")
    names = [str(n) for n in z["names"]]
    ranges = json.loads(str(z["ranges"]))
    return MCSamples(samples=z["samples"].astype(np.float64), weights=z["weights"], names=names,
                     labels=[LABELS.get(n, n) for n in names], ranges={n: tuple(r) for n, r in ranges.items()},
                     label=DATA[d], settings=dict(smooth_scale_1D=0.3, smooth_scale_2D=0.4))


def prior_box(d):
    """The prior range of every sampled cosmological parameter of data set d."""
    return {n: tuple(r) for n, r in json.loads(str(np.load(DATA_DIR / f"posterior_{d}.npz")["ranges"])).items()}


# All text in the figures is typeset by LaTeX with the paper's fonts: Latin Modern in T1 encoding, as main-client.tex
# loads them (\usepackage[T1]{fontenc}, \usepackage{lmodern}), for text and mathematics alike. This needs a LaTeX
# installation with the lmodern package (TeX Live has it).
LATEX_PREAMBLE = r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}\usepackage{amssymb}"


def figure_path(name):
    """../Figures/<name>.pdf; in the World Cup series a name that does not end in a data set's name gets _wc."""
    if SERIES == "wc" and not name.endswith(tuple(f"_{d}" for d in DATA)):
        name += "_wc"
    return FIGURES / f"{name}.pdf"


def save(fig, name):
    """Save the figure as figure_path(name), cropped to its content with no margin: the PDF's left and right edges are
    the leftmost and rightmost drawn elements, so a figure included at \\textwidth spans exactly the width of the text."""
    fig.savefig(figure_path(name), bbox_inches="tight", pad_inches=0)
    print("written:", figure_path(name))


# The figures are drawn at the width of the text (jcappub's \textwidth, 440 pt), so that the paper prints them at scale 1
# and their fonts at the sizes the scripts give (28 October 2026: drawn 10-13 inches wide and shrunk to 0.39-0.63, the
# labels printed at 3-6 pt).
TEXTWIDTH = 440 / 72.27          # inches
# single points (the global best fits of Figs. 1 and 4, the points of SInu in Figs. 3 and 5) are diamonds of one size,
# printed alike since every figure is printed at scale 1 (28 October 2026)
DIAMOND_SIZE, DIAMOND_EDGE = 5.0, 0.8          # marker size and edge width in pt


def ink_box(fig, dpi=600):
    """The extent of everything drawn on fig, in inches, from a rendering at dpi. A figure cropped to the bounding boxes
    of its texts (bbox_inches="tight") can end half a point short of its outermost ink, since the boxes include the
    glyphs' side bearings."""
    from matplotlib.transforms import Bbox
    old = fig.get_dpi()
    fig.set_dpi(dpi)
    fig.canvas.draw()
    ink = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].min(axis=2) < 250
    fig.set_dpi(old)
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    height = ink.shape[0]
    return Bbox.from_extents(cols[0] / dpi, (height - 1 - rows[-1]) / dpi, (cols[-1] + 1) / dpi, (height - rows[0]) / dpi)


def save_text_width(fig, name, layout=lambda: None):
    """Save fig as figure_path(name), exactly the width of the text and cropped to its ink, so that at width=\\textwidth
    its outermost ink lies on the left and right margins and it is printed at scale 1. The drawing is widened or
    narrowed until that holds; layout() lays the figure out again after every change of its size."""
    for _ in range(8):
        layout()
        box = ink_box(fig)
        if abs(box.width - TEXTWIDTH) < 0.05 / 72.27:
            break
        width, height = fig.get_size_inches()
        fig.set_size_inches(width * TEXTWIDTH / box.width, height)
    else:
        raise SystemExit(f"{name}: the width did not converge ({box.width * 72.27:.2f} pt)")
    fig.savefig(figure_path(name), bbox_inches=box, pad_inches=0)
    print("written:", figure_path(name), f"({box.width * 72.27:.2f} x {box.height * 72.27:.2f} pt)")


def setup():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
                                "text.latex.preamble": LATEX_PREAMBLE})
    FIGURES.mkdir(exist_ok=True)
