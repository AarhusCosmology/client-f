"""Figs. profile_breakdown_pp and _ppshoes: the true profile likelihoods split likelihood by likelihood.

    python plot_profile_breakdown.py

data/profile_breakdown.json, per data set: the likelihoods, and for every cosmological parameter the profile nodes with
x, t (the true Delta chi^2 to the global best fit, as in the profile figures), dchi2 (each likelihood's chi^2 minus its
value at the global best fit) and dprior (the same for -2 ln prior: the Gaussian priors of the CamSpec calibrations).
t is the sum of the dchi2 and dprior. data/posterior_<data>.npz: the posteriors, for the same parameter ranges as the
profile figures. Writes ../Figures/profile_breakdown_pp.pdf and profile_breakdown_ppshoes.pdf.
"""
import numpy as np

from common import COSMO, DATA, LABELS, TEXTWIDTH, UNITS, load_json, posterior, prior_box, save_text_width, setup

# the likelihoods' names in the figures, and colours (Okabe-Ito, distinguishable in colour-blind vision)
NAMES = {"planck_2018_lowl.TT": r"low-$\ell$ TT", "planck_2018_lowl.EE": r"low-$\ell$ EE",
         "planck_NPIPE_highl_CamSpec.TTTEEE": "CamSpec TTTEEE", "planckpr4lensing.PlanckPR4Lensing": "PR4 lensing",
         "bao.desi_dr2.desi_bao_all": "DESI DR2", "sn.pantheonplus": "Pantheon+", "sn.pantheonplusshoes": "Pantheon+SH0ES"}
COLOURS = {"planck_2018_lowl.TT": "#E69F00", "planck_2018_lowl.EE": "#56B4E9",
           "planck_NPIPE_highl_CamSpec.TTTEEE": "#009E73", "planckpr4lensing.PlanckPR4Lensing": "#CC79A7",
           "bao.desi_dr2.desi_bao_all": "#0072B2", "sn.pantheonplus": "#D55E00", "sn.pantheonplusshoes": "#D55E00"}
# the World Cup data of Sec. 5 (PAPER_SERIES=wc); "sh0es_mb" is the SH0ES prior on M_B of wcmb, a prior in Cobaya, shown
# as data (build_wc.py takes it out of the priors)
NAMES.update({"planck_2018_lowl.EE_sroll2": r"low-$\ell$ EE (SRoll2)", "act_dr6_cmbonly": "ACT-lite",
              "candl.interface.CandlCobayaLikelihood": "SPT-lite", "act_dr6_lenslike.ACTDR6LensLike": "ACT+PR4 lensing",
              "muse3glike.cobaya.spt3g_2yr_delensed_ee_optimal_pp_muse": "SPT lensing (MUSE)",
              "sh0es_mb": r"SH0ES ($M_B$ prior)"})
COLOURS.update({"planck_2018_lowl.EE_sroll2": "#56B4E9", "act_dr6_cmbonly": "#B8860B",
                "candl.interface.CandlCobayaLikelihood": "#882255", "act_dr6_lenslike.ACTDR6LensLike": "#CC79A7",
                "muse3glike.cobaya.spt3g_2yr_delensed_ee_optimal_pp_muse": "#44AA99", "sh0es_mb": "#661100"})
TMAX = 14.0          # the profile figures show exp(-t/2) > 1e-3, i.e. t < 13.8
# Drawn at the width of the text (common.save_text_width; until 28 October 2026 13 inches wide and shrunk to 0.39-0.48,
# with labels of 3-4 pt, and the one-row legend of the World Cup figures stuck out beyond the panels).
HEIGHT = 0.80                    # the height of the figure relative to its width
NCOL = 4                         # columns of the legend, which spans the panels


def broken(xs, ys, nodes):
    """xs, ys with a NaN between neighbours that have a node of the profile (nodes: its coordinates) between them, so that a
    line is not drawn across nodes left out (a node whose best value came from a study without a breakdown). Until 28
    October 2026 any gap wider than 1.6 times the usual spacing broke the line, which also broke it between the wider-spaced
    nodes of the profiles over the whole prior (f_idm, log10 G_eff)."""
    x_out, y_out = [xs[0]], [ys[0]]
    for x0, x1, y1 in zip(xs[:-1], xs[1:], ys[1:]):
        if any(x0 + 1e-9 < v < x1 - 1e-9 for v in nodes):
            x_out.append(np.nan)
            y_out.append(np.nan)
        x_out.append(x1)
        y_out.append(y1)
    return np.array(x_out), np.array(y_out)


def main():
    setup()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    data = load_json("profile_breakdown.json")
    profiles = load_json("profiles.json")
    for d in DATA:
        if d not in data:                       # no breakdown of this data set yet (a draft of the World Cup section)
            continue
        b = data[d]
        samples, box = posterior(d), prior_box(d)
        fig, axes = plt.subplots(3, 4, figsize=(TEXTWIDTH, HEIGHT * TEXTWIDTH))
        for ax, n in zip(axes.ravel(), COSMO):
            rows = sorted(b["P"].get(n, []), key=lambda r: r["x"])
            if not rows:
                ax.set_xlabel(f"${LABELS[n]}{UNITS.get(n, '')}$ (pending)", fontsize=8.5, labelpad=1.5)
                continue
            xs = np.array([r["x"] for r in rows])
            t = np.array([r["t"] for r in rows])
            nodes = [r["x"] for r in profiles.get(d, {}).get("P", {}).get(n, [])]
            # the x range of the profile figure: where the profile or the posterior is above 1e-3 of its maximum
            dens = samples.get1DDensity(n)
            px, pd = np.asarray(dens.x), np.asarray(dens.P) / np.max(dens.P)
            seen = np.concatenate([xs[t < TMAX], px[pd > 1e-3]])
            lo, hi = seen.min(), seen.max()
            pad = 0.03 * (hi - lo)
            lo, hi = max(lo - pad, box[n][0]), min(hi + pad, box[n][1])
            inside = (xs >= lo) & (xs <= hi)
            shown = [t[inside]]
            for k in b["likelihoods"]:
                y = np.array([r["dchi2"][k] for r in rows])
                ax.plot(*broken(xs, y, nodes), "-", color=COLOURS[k], lw=0.8)
                shown.append(y[inside])
            prior = np.array([r["dprior"] for r in rows])
            if np.max(np.abs(prior[inside])) > 0.05:
                ax.plot(*broken(xs, prior, nodes), ":", color="0.4", lw=0.8)
                shown.append(prior[inside])
            ax.plot(*broken(xs, t, nodes), "-", color="k", lw=1.2)
            ax.plot(xs, t, "o", color="k", ms=1.4)
            ax.axhline(0, color="0.6", lw=0.5, zorder=0)
            values = np.concatenate(shown)
            ymin, ymax = min(values.min(), -1.0), min(max(values.max(), 4.0), TMAX)
            ax.set_ylim(ymin - 0.06 * (ymax - ymin), ymax + 0.06 * (ymax - ymin))
            ax.set_xlim(lo, hi)
            ax.set_xlabel(f"${LABELS[n]}{UNITS.get(n, '')}$", fontsize=8.5, labelpad=1.5)
            ax.tick_params(labelsize=7, pad=1.5, length=2.5, width=0.6)
            ax.xaxis.set_major_locator(MaxNLocator(3))          # at print size four labels of omega_b overlap
            ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
            for side in ax.spines.values():
                side.set_linewidth(0.6)
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$\Delta\chi^2$", fontsize=8.5, labelpad=1.5)
        handles = [Line2D([], [], color="k", lw=1.2, marker="o", ms=2.0, label="total (the profile)")]
        handles += [Line2D([], [], color=COLOURS[k], lw=1.0, label=NAMES[k]) for k in b["likelihoods"]]
        handles += [Line2D([], [], color="0.4", lw=0.9, ls=":", label="priors")]
        # the legend above the panels and no wider than they are; the dataset is named in the caption
        rows = -(-len(handles) // NCOL)
        legend = fig.legend(handles=handles, loc="lower left", ncol=NCOL, mode="expand", fontsize=7.5, frameon=False,
                            handlelength=1.8, columnspacing=1.2, borderaxespad=0, borderpad=0, labelspacing=0.3)

        def layout(fig=fig, axes=axes, legend=legend, rows=rows):
            top = 1 - (rows * 10.5 + 3) / 72.27 / fig.get_size_inches()[1]     # room for the legend's rows of 7.5 pt
            fig.tight_layout(pad=0.05, h_pad=0.5, w_pad=0.3, rect=(0, 0, 1, top))
            left, right = axes[0, 0].get_position().x0, axes[0, -1].get_position().x1
            legend.set_bbox_to_anchor((left, top + 0.004, right - left, 1 - top - 0.004), transform=fig.transFigure)
        save_text_width(fig, f"profile_breakdown_{d}", layout)
        plt.close(fig)


if __name__ == "__main__":
    main()
