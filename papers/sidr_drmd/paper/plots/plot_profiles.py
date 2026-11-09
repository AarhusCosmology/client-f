"""The true profile likelihoods of the twelve cosmological parameters with the marginalised posteriors, both data sets of
a section in one figure (26 October 2026; before, one figure per data set).

    python plot_profiles.py                    # Sec. 4: PDP+ and PDP+SH0ES -> ../Figures/profiles.pdf
    PAPER_SERIES=wc python plot_profiles.py    # Sec. 5: H0WC and H0WC+SH0ES -> ../Figures/profiles_wc.pdf

data/profiles.json, per data set and parameter: the profile nodes (x; t, the true Delta chi^2 to the global best fit).
data/posterior_<data>.npz: the posteriors. In every panel, each data set's posterior (shaded, no outline) and its profile
likelihood exp(-Delta chi^2 / 2) (a line, in a strong colour of the data set's hue); the posterior normalised to a maximum
of 1, the profile relative to the global best fit, so that its maximum is below 1 where no node reaches the best fit (PDP+
f_idm: 0.87); no node markers.
"""
import numpy as np

from common import COLOURS, COSMO, DATA, LABELS, TEXTWIDTH, UNITS, load_json, posterior, prior_box, save_text_width, setup

# the profile lines: strong colours of the data sets' hues, so that each line is told apart and matched to its posterior
LINE = {"#2b6cb0": "#0b3fa8", "#c0392b": "#d01c0b"}
# Drawn at the width of the text (common.save_text_width; until 28 October 2026 13 inches wide and shrunk to 0.48, with
# labels of 4-5 pt). The panels share the probability axis, so that only the first column carries its labels.
HEIGHT = 0.68                    # the height of the figure relative to its width
NCOL = 3                         # columns of the legend, which spans the panels


def main():
    setup()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    profs = load_json("profiles.json")
    sets = list(DATA)
    post = {d: posterior(d) for d in sets}
    boxes = {d: prior_box(d) for d in sets}
    fig, axes = plt.subplots(3, 4, figsize=(TEXTWIDTH, HEIGHT * TEXTWIDTH), sharey=True)
    for ax, n in zip(axes.ravel(), COSMO):
        seen, lo_box, hi_box, curves = [], -np.inf, np.inf, {}
        for d in sets:
            dens = post[d].get1DDensity(n)
            px, pd = np.asarray(dens.x), np.asarray(dens.P) / np.max(dens.P)
            rows = profs.get(d, {}).get("P", {}).get(n, [])
            xs = np.array([r["x"] for r in rows])
            like = np.exp(-0.5 * np.array([r["t"] for r in rows])) if rows else np.array([])
            curves[d] = (px, pd, xs, like)
            seen.append(px[pd > 1e-3])
            if len(xs):
                seen.append(xs[like > 1e-3])
            lo_box, hi_box = max(lo_box, boxes[d][n][0]), min(hi_box, boxes[d][n][1])
        seen = np.concatenate(seen)
        lo, hi = seen.min(), seen.max()
        pad = 0.03 * (hi - lo)
        lo, hi = max(lo - pad, lo_box), min(hi + pad, hi_box)
        for d in sets:                       # the posteriors underneath
            px, pd, _, _ = curves[d]
            ax.fill_between(px, pd, color=COLOURS[d], alpha=0.22, lw=0)
        for d in sets:                       # the profiles on top
            _, _, xs, like = curves[d]
            if len(xs):
                ax.plot(xs, like, "-", color=LINE.get(COLOURS[d], COLOURS[d]), lw=1.1)
        ax.axhline(np.exp(-0.5), color="0.45", lw=0.5, ls="--", zorder=0)    # Delta chi^2 = 1
        ax.axhline(np.exp(-2.0), color="0.45", lw=0.5, ls=":", zorder=0)     # Delta chi^2 = 4
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.08)
        ax.set_xlabel(f"${LABELS[n]}{UNITS.get(n, '')}$", fontsize=8.5, labelpad=1.5)
        ax.tick_params(labelsize=7, pad=1.5, length=2.5, width=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(3))          # at print size four labels of omega_b overlap
        ax.set_yticks([0, 0.5, 1])
        for side in ax.spines.values():
            side.set_linewidth(0.6)
    for ax in axes[:, 0]:
        ax.set_ylabel("relative probability", fontsize=8.5, labelpad=1.5)
    handles = []
    for d in sets:
        handles += [Line2D([], [], color=LINE.get(COLOURS[d], COLOURS[d]), lw=1.1, label=f"{DATA[d]}: profile likelihood"),
                    Patch(facecolor=COLOURS[d], alpha=0.3, edgecolor="none", label=f"{DATA[d]}: posterior")]
    handles += [Line2D([], [], color="0.45", lw=0.7, ls="--", label=r"$\Delta\chi^2 = 1$"),
                Line2D([], [], color="0.45", lw=0.7, ls=":", label=r"$\Delta\chi^2 = 4$")]
    # the legend above the panels and no wider than they are
    rows = -(-len(handles) // NCOL)
    legend = fig.legend(handles=handles, loc="lower left", ncol=NCOL, mode="expand", fontsize=7.5, frameon=False,
                        handlelength=1.8, columnspacing=1.2, borderaxespad=0, borderpad=0, labelspacing=0.3)

    def layout():
        top = 1 - (rows * 10.5 + 3) / 72.27 / fig.get_size_inches()[1]     # room for the legend's rows of 7.5 pt
        fig.tight_layout(pad=0.05, h_pad=0.5, w_pad=0.4, rect=(0, 0, 1, top))
        left, right = axes[0, 0].get_position().x0, axes[0, -1].get_position().x1
        legend.set_bbox_to_anchor((left, top + 0.004, right - left, 1 - top - 0.004), transform=fig.transFigure)
    save_text_width(fig, "profiles", layout)
    plt.close(fig)


if __name__ == "__main__":
    main()
