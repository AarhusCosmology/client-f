"""The posteriors of both data sets (68% and 95% contours) with the global best fits as diamonds.

    python plot_triangle.py

data/posterior_<data>.npz: the combined emulators' posteriors (BEST, four seeds pooled; weighted samples of the twelve
cosmological and the derived parameters). data/best_fits.json: the global best fit of each data set, the point of
highest true likelihood found (a profile-likelihood node), with its derived parameters from the derived-parameter
emulators. Writes ../Figures/triangle_key.pdf (the full triangle of all twelve parameters, triangle_full.pdf, was left out
of the paper on 27 October 2026). The axes of common.PLOT_LIMITS (f_idm; m_int in the
World Cup series) are narrower than the prior; the others are GetDist's.
"""
from common import (COLOURS, DATA, DIAMOND_EDGE, DIAMOND_SIZE, PLOT_LIMITS, TEXTWIDTH, load_json, posterior,
                    save_text_width, setup)

FIGURES_MADE = (("triangle_key", ["H0", "S8", "n_s", "logGeff", "delta_Neff_drmd", "f_idm_drmd", "m_int"]),)
# Drawn at the width of the text (common.save_text_width; until 28 October 2026 10 inches wide and shrunk to 0.63, with
# GetDist's font scaling), with the sizes below as printed.
LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 9, 6.5, 8.5
# the global best fits as diamonds of common.DIAMOND_SIZE, as the single points of Figs. 3 and 5 (stars until 28 October
# 2026), filled in the data set's colour


def main():
    setup()
    import matplotlib.pyplot as plt
    from getdist import plots
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    post = {d: posterior(d) for d in DATA}
    stars = load_json("best_fits.json")
    for name, params in FIGURES_MADE:
        g = plots.get_subplot_plotter(width_inch=TEXTWIDTH)
        g.settings.scaling = False               # the font sizes as given, not scaled with the size of the panels
        g.settings.num_plot_contours = 2
        g.settings.legend_fontsize = LEGEND_SIZE
        g.settings.axes_labelsize = LABEL_SIZE
        g.settings.axes_fontsize = TICK_SIZE
        g.triangle_plot([post[d] for d in DATA], params, filled=True,
                        contour_colors=[COLOURS[d] for d in DATA], legend_labels=list(DATA.values()),
                        param_limits={n: r for n, r in PLOT_LIMITS.items() if n in params})
        for i in range(len(params)):
            for j in range(i):
                ax = g.subplots[i, j]
                for d in DATA:
                    v = stars[d]["values"]
                    ax.plot(v[params[j]], v[params[i]], "D", color=COLOURS[d], ms=DIAMOND_SIZE, mec="black",
                            mew=DIAMOND_EDGE, zorder=20)
        # one legend: the two posteriors and their global best fits
        if g.legend is not None:
            g.legend.remove()
        handles = [Patch(facecolor=COLOURS[d], alpha=0.7, edgecolor=COLOURS[d], label=DATA[d]) for d in DATA] + [
            Line2D([], [], ls="", marker="D", color=COLOURS[d], mec="black", mew=DIAMOND_EDGE, ms=DIAMOND_SIZE,
                   label=f"{DATA[d]}: global best fit")
            for d in DATA]
        g.fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=g.settings.legend_fontsize,
                     frameon=False)
        save_text_width(g.fig, name)
        plt.close("all")


if __name__ == "__main__":
    main()
