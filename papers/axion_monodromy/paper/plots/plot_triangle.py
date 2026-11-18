"""The posterior of the key parameters (68%, 95% and 99.7% contours) with the best fit of the true profiles as a diamond.

    python plot_triangle.py

omega <= 200 (7 November 2026): data/posterior200_<d>.npz, the posterior of the newest combined emulator of each data
set (written by ../build200.py; PDP+: the seven parts, H0WC: the eight), one figure per data set (8 November):
../Figures/triangle_key.pdf for PDP+ and ../Figures/triangle_key_wc.pdf for H0WC. The diamond is the best fit of the
true profiles, from data/profile2d_<d>_omega_phi.json once that file holds the profiles up to omega = 200 (the H0WC
file still holds those of the omega <= 60 study, and its figure has no diamond until it does).

The posterior of omega over [1, 200] is many narrow peaks (the main one about 2 wide) with a standard deviation of about
55, which no kernel width of GetDist's resolves: its automatic width fails and falls back to a wide one, and the 2D
setting of 0.2 standard deviations is 11 in omega. The panels with omega are therefore drawn here from histograms in
bins of 0.5 in omega (40 bins of the other parameter), smoothed by 1.5 bins (SMOOTH), with the same 68%, 95% and 99.7% levels (the
regions of highest density holding that share of the samples) and GetDist's colours; the other panels are GetDist's.
The outer level is 99.7% (until 18 November 2026 GetDist's 99%, Luca Janken's comment), as the 11.83 of the profiles.
"""
import numpy as np

from common import COLOURS, DATA, DATA_DIR, DIAMOND_EDGE, DIAMOND_SIZE, TEXTWIDTH, load_json, posterior, save_text_width, setup

PARAMS = ["n_s", "nrun", "A_osc", "omega", "phi"]
LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 9, 6.5, 8.5
OMEGA_BIN = 0.5
OTHER_BINS = 40
SHARES = (0.997, 0.95, 0.68)  # the contours, outermost first; GetDist's panels get the same (posterior(contours=...))
A_TICKS = [0.02, 0.05, 0.08]
SMOOTH = 1.5               # bins: 0.75 in omega, a third of the width of the main peak


def smooth(h, sigma):
    """h smoothed by a Gaussian of sigma bins along each axis (sigma per axis), edges by reflection."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(h, sigma, mode="reflect")


def levels(h):
    """The density levels whose regions of highest density hold SHARES of the total, increasing."""
    flat = np.sort(h.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    return [flat[min(np.searchsorted(cum, s), len(flat) - 1)] for s in SHARES]


def omega_panels(g, samples, weights, names, limits, colours, line_colour):
    """Redraw every panel with omega from fine histograms (see the docstring)."""
    w = samples[:, names.index("omega")]
    edges_w = np.arange(1.0, 200.0 + OMEGA_BIN / 2, OMEGA_BIN)
    centres_w = 0.5 * (edges_w[1:] + edges_w[:-1])
    for i, py in enumerate(PARAMS):
        for j, px in enumerate(PARAMS[:i + 1]):
            if "omega" not in (px, py):
                continue
            ax = g.subplots[i, j]
            for artist in list(ax.collections) + list(ax.lines):
                artist.remove()
            if i == j:
                h, _ = np.histogram(w, bins=edges_w, weights=weights)
                ax.plot(centres_w, h / h.max(), "-", color=line_colour, lw=0.8)
                ax.set_xlim(1, 200)
                continue
            other = px if py == "omega" else py
            v = samples[:, names.index(other)]
            edges_v = np.linspace(*limits[other], OTHER_BINS + 1)
            centres_v = 0.5 * (edges_v[1:] + edges_v[:-1])
            h, _, _ = np.histogram2d(w, v, bins=[edges_w, edges_v], weights=weights)
            h = smooth(h, (SMOOTH, SMOOTH))
            lv = levels(h)
            # the grid out to the outer bin edges (each end bin's value), so that the regions reach the axes where
            # they meet a prior bound (A_osc = 0, omega = 1 and 200) instead of stopping half a bin inside
            h = np.pad(h, 1, mode="edge")
            grid_w = np.concatenate([edges_w[:1], centres_w, edges_w[-1:]])
            grid_v = np.concatenate([edges_v[:1], centres_v, edges_v[-1:]])
            if py == "omega":            # omega on the vertical axis
                X, Y, H = grid_v, grid_w, h
            else:
                X, Y, H = grid_w, grid_v, h.T
            ax.contourf(X, Y, H, levels=lv + [H.max() * 1.01], colors=colours)
            ax.contour(X, Y, H, levels=lv[:1], colors=[line_colour], linewidths=0.6, alpha=0.5)
            ax.set_xlim(*(limits[px] if px != "omega" else (1, 200)))
            ax.set_ylim(*(limits[py] if py != "omega" else (1, 200)))


def figure(d, out):
    from getdist import plots

    post = posterior(d, f"posterior200_{d}.npz", contours=sorted(SHARES))
    g = plots.get_subplot_plotter(width_inch=TEXTWIDTH * 0.75)
    g.settings.scaling = False
    g.settings.num_plot_contours = len(SHARES)
    g.settings.legend_fontsize = LEGEND_SIZE
    g.settings.axes_labelsize = LABEL_SIZE
    g.settings.axes_fontsize = TICK_SIZE
    g.triangle_plot([post], PARAMS, filled=True, contour_colors=[COLOURS[d]], legend_labels=[DATA[d]], legend_loc="upper right")
    # the axis limits GetDist chose, and its filled colours (outermost first), read off a panel it drew
    limits = {p: g.subplots[len(PARAMS) - 1, k].get_xlim() for k, p in enumerate(PARAMS)}
    fill = next(c for c in g.subplots[1, 0].collections if len(getattr(c, "levels", [])) == len(SHARES) + 1)
    colours = [tuple(f) for f in fill.get_facecolor()[:len(SHARES)]]
    z = np.load(DATA_DIR / f"posterior200_{d}.npz")
    omega_panels(g, z["samples"].astype(float), z["weights"], [str(n) for n in z["names"]], limits, colours, COLOURS[d])
    file = DATA_DIR / f"profile2d_{d}_omega_phi.json"
    if file.exists() and max(load_json(file.name)["axes"][0]) > 60:          # the profiles up to omega = 200
        best = load_json(file.name)["best_x"]
        for i, py in enumerate(PARAMS):
            for j, px in enumerate(PARAMS[:i]):
                g.subplots[i, j].plot(best[px], best[py], "D", ms=DIAMOND_SIZE, mfc=COLOURS[d], mec="k", mew=DIAMOND_EDGE, zorder=5)
    # the same ticks of A_osc in both figures (GetDist chooses them from each posterior's range)
    k = PARAMS.index("A_osc")
    for i in range(len(PARAMS)):
        if i >= k:
            g.subplots[i, k].set_xticks(A_TICKS)
    for j in range(k):
        g.subplots[k, j].set_yticks(A_TICKS)
    g.fig.subplots_adjust(wspace=0.05, hspace=0.05)
    save_text_width(g.fig, out)


def main():
    setup()
    for d, out in (("pp", "triangle_key"), ("wc", "triangle_key_wc")):
        if (DATA_DIR / f"posterior200_{d}.npz").exists():
            figure(d, out)


if __name__ == "__main__":
    main()
