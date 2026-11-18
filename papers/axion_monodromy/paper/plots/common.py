"""Shared settings of the plotting scripts (as in the SIDR + DRMD draft's plots/common.py): data sets, colours, labels,
the data files in data/, and the figure layout.

Every figure of the paper is made by one script in this folder (plot_*.py) from the files in data/ alone, with numpy,
matplotlib and GetDist; `python make_all.py` makes them all. The figures are written to ../Figures/, where the LaTeX
source reads them. The data files are written by ../build.py from the runs (see ../README.md).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIGURES = HERE.parent / "Figures"

DATA = {"pp": "PDP+", "wc": r"$H_0{\rm WC}$"}
COLOURS = {"pp": "#2b6cb0", "wc": "#c0392b"}
LABELS = {"omega_b": r"\omega_\mathrm{b}", "omega_cdm": r"\omega_\mathrm{cdm}", "theta_s_100": r"100\,\theta_\mathrm{s}",
          "logA": r"\ln(10^{10}A_\mathrm{s})", "n_s": r"n_\mathrm{s}", "nrun": r"\mathrm{d}n_\mathrm{s}/\mathrm{d}\ln k",
          "tau_reio": r"\tau_\mathrm{reio}", "A_osc": r"A_\mathrm{osc}", "omega": r"\omega", "phi": r"\phi",
          "H0": r"H_0", "sigma8": r"\sigma_8", "S8": r"S_8", "Omega_m": r"\Omega_\mathrm{m}",
          "rs_drag": r"r_\mathrm{drag}"}


def load_json(name):
    return json.loads((DATA_DIR / name).read_text())


def posterior(d, file=None, contours=None):
    """The marginalised posterior of data set d as GetDist samples (file: another file in data/ than posterior_<d>.npz;
    contours: the shares of GetDist's contours, if not its default 0.68, 0.95, 0.99)."""
    from getdist import MCSamples

    z = np.load(DATA_DIR / (file or f"posterior_{d}.npz"))
    names = [str(n) for n in z["names"]]
    ranges = json.loads(str(z["ranges"]))
    settings = dict(smooth_scale_1D=-1, smooth_scale_2D=0.2)
    if contours is not None:
        settings["contours"] = " ".join(str(c) for c in contours)
    return MCSamples(samples=z["samples"].astype(np.float64), weights=z["weights"], names=names,
                     labels=[LABELS.get(n, n) for n in names], ranges={n: tuple(r) for n, r in ranges.items()},
                     label=DATA[d], settings=settings)
# 1D: GetDist's automatic smoothing (29 October 2026): a fixed 0.3 standard deviations widened the peak at omega ~ 51
# from 2.2 to 8.4 (full width at half maximum), since the standard deviation of the multimodal omega posterior is about
# 13. 2D: 0.2 standard deviations; the automatic choice made the (omega, phi) panel patchy.


# All text in the figures is typeset by LaTeX in the paper's fonts (T1, Latin Modern), as main-monodromy.tex loads them.
LATEX_PREAMBLE = r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}\usepackage{amssymb}"
# The figures are drawn at the width of the text (jcappub's \textwidth, 440 pt) and printed at scale 1.
TEXTWIDTH = 440 / 72.27
DIAMOND_SIZE, DIAMOND_EDGE = 5.0, 0.8


def ink_box(fig, dpi=600):
    """The extent of everything drawn on fig, in inches, from a rendering at dpi."""
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
    """Save fig as ../Figures/<name>.pdf, exactly the width of the text and cropped to its ink."""
    for _ in range(8):
        layout()
        box = ink_box(fig)
        if abs(box.width - TEXTWIDTH) < 0.05 / 72.27:
            break
        width, height = fig.get_size_inches()
        fig.set_size_inches(width * TEXTWIDTH / box.width, height)
    else:
        raise SystemExit(f"{name}: the width did not converge ({box.width * 72.27:.2f} pt)")
    # 2 pt of white above and below: the ink box of the rendering can end inside a glyph at the bottom (the axis label);
    # the width stays that of the text
    from matplotlib.transforms import Bbox
    box = Bbox.from_extents(box.x0, box.y0 - 2 / 72.27, box.x1, box.y1 + 2 / 72.27)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches=box, pad_inches=0)
    print("written:", FIGURES / f"{name}.pdf", f"({box.width * 72.27:.2f} x {box.height * 72.27:.2f} pt)")


def setup():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
                                "text.latex.preamble": LATEX_PREAMBLE})
    FIGURES.mkdir(exist_ok=True)


# what is still the emulator's in a true profile, until the true stage of a study has ended
EMULATOR_SHADE = dict(color="#e8743b", alpha=0.22, lw=0)


def shade_cells(ax, x, flags):
    """Shade the grid cells of the uniform axis x where flags is true, one span per run of cells."""
    half = 0.5 * (x[1] - x[0])
    k = 0
    while k < len(x):
        if flags[k]:
            m = k
            while m + 1 < len(x) and flags[m + 1]:
                m += 1
            ax.axvspan(x[k] - half, x[m] + half, **EMULATOR_SHADE)
            k = m
        k += 1
