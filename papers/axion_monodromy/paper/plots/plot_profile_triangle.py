"""The true profile likelihood of a dataset in the oscillation parameters, drawn as a corner plot.

    python plot_profile_triangle.py

data/profile2d_<d>_<pair>.json (written by ../build.py): the three surfaces (A_osc, omega), (A_osc, phi) and (omega, phi) on
their 50 x 50 grids, each node maximised over all other parameters with the true likelihood, finished or not (a node not
done yet is null), all relative to the same best fit. data/posterior_<d>.npz: the combined emulator's posterior.
Writes ../Figures/profile_triangle.pdf for PDP+ and ../Figures/profile_triangle_wc.pdf for H0WC, the latter only when its
surfaces exist:
  - below the diagonal, the 2D profiles, with Delta chi^2 < 2.30, 6.18 and 11.83 (the Gaussian 68%, 95% and 99.7% levels
    for two parameters) shaded, and the best fit as a diamond; nodes not done yet in pale red;
  - on the diagonal, the 1D profile likelihood ratio exp(-Delta chi^2 / 2) of each parameter relative to the best fit
    (black; below 1 in omega, whose grid does not contain the best fit), the better of the two surfaces which contain it at
    each grid value, the best fit as a diamond at 1, and its marginalised posterior (blue, red for H0WC) with a maximum of 1;
    dashed lines at Delta chi^2 = 1 and 4 for the profile.
Until the true stage of a study has ended, what is still the emulator's value is shaded: the nodes below the diagonal, and on
the diagonal the grid values whose best node is one of these.
"""
import numpy as np

from common import (COLOURS, DATA_DIR, DIAMOND_EDGE, DIAMOND_SIZE, EMULATOR_SHADE, LABELS, TEXTWIDTH, load_json, posterior, save_text_width,
                    setup, shade_cells)

PARAMS = ["A_osc", "omega", "phi"]
PAIRS = {("A_osc", "omega"): "aosc_omega", ("A_osc", "phi"): "aosc_phi", ("omega", "phi"): "omega_phi"}
LEVELS = [2.30, 6.18, 11.83]
LEVELS_1D = [1, 4]
# 1, 2 and 3 sigma: the profile in the purples of Steen's parameter-estimation papers (PRL 95, 221301, 2005, Fig. 3:
# #741474 and #bc7cf8, read off the figure) and a pale tint of the second for the third; the posterior in blue
FILLS = ["#741474", "#bc7cf8", "#e1c4fc"]
NOT_DONE = "#f4d9d9"
LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 9, 6.5, 8


def figure(d, out):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    surf = {}
    for (p0, p1), name in PAIRS.items():
        e = load_json(f"profile2d_{d}_{name}.json")
        assert e["parameters"] == [p0, p1]
        Z = np.array([[np.nan if v is None else v for v in row] for row in e["logl"]], float)
        # the best fit is the best true node; an emulator node still above it is drawn at Delta chi^2 = 0
        # closed (build200.py, WC_CLOSED): the studies have stopped and the nodes not refined keep the emulator's value,
        # which the caption says; they are no longer shaded
        emulator = np.zeros(Z.shape, bool) if e.get("closed") else np.array(e["emulator"], bool)
        surf[(p0, p1)] = dict(x=np.array(e["axes"][0]), y=np.array(e["axes"][1]), dchi=np.maximum(2 * (e["best"] - Z), 0),
                              best=e["best_x"], done=e["done"], true_nodes=e["true_nodes"], emulator=emulator,
                              outside=np.array(e.get("outside", np.zeros(Z.shape, bool)), bool))
    axis = {p: next(s["x"] if k[0] == p else s["y"] for k, s in surf.items() if p in k) for p in PARAMS}
    # omega <= 200 (build200.py): the chain of the newest combined emulator of each data set (PDP+: the seven parts, H0WC:
    # the eight; the PDP+ profiles' studies
    # are those of the six-part emulator, but their values are the true likelihood's)
    post = posterior(d, f"posterior200_{d}.npz") if (DATA_DIR / f"posterior200_{d}.npz").exists() else posterior(d)

    n = len(PARAMS)
    fig, axes = plt.subplots(n, n, figsize=(TEXTWIDTH, TEXTWIDTH * 0.92))
    for r in range(n):
        for c in range(n):
            ax = axes[r, c]
            px, py = PARAMS[c], PARAMS[r]
            if c > r:
                ax.axis("off")
                continue
            if c == r:
                # the 1D profile: the better of the two surfaces which contain the parameter, at each grid value
                prof = np.full(len(axis[px]), np.inf)
                from_emulator = np.zeros(len(axis[px]), bool)
                for (p0, p1), s in surf.items():
                    if px not in (p0, p1):
                        continue
                    k = 1 if px == p0 else 0                 # the axis to maximise over
                    dchi = np.where(np.isfinite(s["dchi"]), s["dchi"], np.inf)
                    at = np.argmin(dchi, axis=k)
                    here = np.take_along_axis(dchi, np.expand_dims(at, k), k).squeeze(k)
                    flag = np.take_along_axis(s["emulator"], np.expand_dims(at, k), k).squeeze(k)
                    better = here < prof
                    prof, from_emulator = np.where(better, here, prof), np.where(better, flag, from_emulator)
                prof = np.where(np.isfinite(prof), prof, np.nan)
                shade_cells(ax, axis[px], from_emulator)
                ok = np.isfinite(prof)
                for lv in LEVELS_1D:
                    ax.axhline(np.exp(-0.5 * lv), color="0.45", ls="--", lw=0.6)
                ax.plot(axis[px][ok], np.exp(-0.5 * prof[ok]), "-", color="k", lw=1.0)
                if px == "omega" and axis[px][-1] - axis[px][0] > 100:
                    # omega up to 200: GetDist's kernel cannot resolve peaks of width ~1 over the whole range (its bandwidth
                    # estimate fails and falls back to a wide one), so the posterior is a histogram in bins of 0.5
                    edges = np.arange(axis[px][0], axis[px][-1] + 0.25, 0.5)
                    pd_, _ = np.histogram(post.samples[:, post.index[px]], bins=edges, weights=post.weights)
                    xs = 0.5 * (edges[1:] + edges[:-1])
                else:
                    density = post.get1DDensity(px)
                    xs = np.linspace(axis[px][0], axis[px][-1], 400)
                    pd_ = density.Prob(xs)
                ax.plot(xs, pd_ / np.nanmax(pd_), "-", color=COLOURS[d], lw=1.0)
                # the best fit (10 November, Steen): in omega it lies between two nodes of the grid, above the profile
                best = next(iter(surf.values()))["best"]
                ax.plot(best[px], 1.0, "D", ms=DIAMOND_SIZE, mfc=COLOURS[d], mec="k", mew=DIAMOND_EDGE, zorder=4, clip_on=False)
                ax.set_ylim(0, 1.08)
                ax.set_yticks([])
            else:
                s = surf[(px, py)]
                X, Y = np.meshgrid(s["x"], s["y"], indexing="ij")
                missing = ~np.isfinite(s["dchi"]) & ~s["outside"]      # outside a window's range: left white
                if missing.any():
                    ax.pcolormesh(s["x"], s["y"], np.where(missing, 1.0, np.nan).T, cmap=ListedColormap([NOT_DONE]),
                                  shading="nearest", rasterized=True)
                dchi = np.ma.masked_invalid(s["dchi"])
                ax.contourf(X, Y, dchi, levels=[-1e-9] + LEVELS, colors=FILLS)
                if s["emulator"].any():
                    shade = {k: v for k, v in EMULATOR_SHADE.items() if k == "alpha"}
                    ax.pcolormesh(s["x"], s["y"], np.where(s["emulator"], 1.0, np.nan).T, cmap=ListedColormap([EMULATOR_SHADE["color"]]),
                                  shading="nearest", rasterized=True, zorder=1.5, linewidth=0, **shade)
                ax.contour(X, Y, dchi, levels=LEVELS, colors="k", linewidths=0.4, zorder=2)
                ax.plot(s["best"][px], s["best"][py], "D", ms=DIAMOND_SIZE, mfc=COLOURS[d], mec="k", mew=DIAMOND_EDGE)
                ax.set_ylim(axis[py][0], axis[py][-1])
                if c == 0:
                    ax.set_ylabel(f"${LABELS[py]}$", fontsize=LABEL_SIZE)
                else:
                    ax.set_yticklabels([])
            ax.set_xlim(axis[px][0], axis[px][-1])
            ax.tick_params(labelsize=TICK_SIZE)
            if r == n - 1:
                ax.set_xlabel(f"${LABELS[px]}$", fontsize=LABEL_SIZE)
            else:
                ax.set_xticklabels([])
    handles = [Line2D([], [], color="k", lw=1.0, label="profile likelihood"),
               Line2D([], [], color="0.45", ls="--", lw=0.6, label=r"$\Delta\chi^2 = 1$ and 4"),
               Line2D([], [], color=COLOURS[d], lw=1.0, label="marginalised posterior"),
               Line2D([], [], ls="", marker="D", ms=DIAMOND_SIZE, mfc=COLOURS[d], mec="k", mew=DIAMOND_EDGE, label="best fit")]
    handles += [Patch(facecolor=f, edgecolor="k", lw=0.4, label=rf"$\Delta\chi^2 < {lv:.2f}$") for f, lv in zip(FILLS, LEVELS)]
    if any((~np.isfinite(s["dchi"]) & ~s["outside"]).any() for s in surf.values()):
        handles.append(Patch(facecolor=NOT_DONE, label="nodes not done yet"))
    if any(s["emulator"].any() for s in surf.values()):
        handles.append(Patch(label="not yet final", **{k: EMULATOR_SHADE[k] for k in ("color", "alpha")}))
    axes[0, n - 1].legend(handles=handles, loc="upper right", fontsize=LEGEND_SIZE, frameon=False)
    fig.subplots_adjust(wspace=0.06, hspace=0.06)
    save_text_width(fig, out)


def main():
    setup()
    figure("pp", "profile_triangle")
    if all((DATA_DIR / f"profile2d_wc_{name}.json").exists() for name in PAIRS.values()):
        figure("wc", "profile_triangle_wc")


if __name__ == "__main__":
    main()
