"""Render the three surfaces on common coordinates and retain raw node values."""

import json, csv
from pathlib import Path
import numpy as np
from profiles.io import save


def write_surface_outputs(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import chi2, norm
    from scipy.interpolate import RectBivariateSpline
    from matplotlib.lines import Line2D

    joint_levels = chi2.ppf(2 * norm.cdf([1.0, 2.0]) - 1, df=2).tolist()

    out = Path(out)
    s = json.loads((out / "surrogate_surface.json").read_text())
    t = json.loads((out / "true_surface.json").read_text())
    axes = [np.array(a) for a in s["axes"]]
    names = s["names"]
    values = [
        2 * (s["best_logl"] - np.array(s["logl"])),
        # float: a node the likelihood refused has no value (None), NaN here.
        2 * (t["best_logl"] - np.array(t["direct_logl"], float)),
        2 * (t["best_logl"] - np.array(t["logl"], float)),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    ceiling = max(6.18, min(25.0, max(float(np.nanmax(v)) for v in values)))
    for ax, z, title in zip(
        axs, values, ["Surrogate", "True at surrogate", "True optimized"]
    ):
        im = ax.pcolormesh(
            *axes, z.T, shading="auto", vmin=0, vmax=ceiling, cmap="viridis_r"
        )
        levels = [v for v in joint_levels if np.nanmin(z) < v < np.nanmax(z)]
        if levels:
            ax.contour(*axes, z.T, levels=levels, colors="black", linewidths=0.8)
        ax.set(xlabel=names[0], ylabel=names[1], title=title)
        ax.ticklabel_format(axis="both", useOffset=False)
    fig.colorbar(im, ax=axs, label=r"$2(\ell_{\max}-\ell)$ (display clipped)", extend="max")
    fig.suptitle("Two fixed coordinates; all remaining parameters profiled")
    fig.savefig(out / "surface.png", dpi=160)
    fig.savefig(out / "surface.pdf")
    plt.close(fig)
    # Separate isocontours use a common smooth interpolator, never extra backend
    # calls. Keep the raw-cell, linear-contour figure above as a resolution check.
    # The spline sees twice the native-objective difference clipped at 25, far above both levels: one
    # very deep node (a grid corner, or a derived-coordinate target out of reach)
    # otherwise rings through the whole interpolant. A spline needs every node, and a node the
    # likelihood refused has no value: it is not excluded, and is never filled in. A surface
    # with one is drawn as linear contours on the nodes that have values, with the gap showing.
    fine = [np.linspace(a[0], a[-1], 401) for a in axes]
    smooth = [
        RectBivariateSpline(
            *axes, np.minimum(z, 25.0), kx=min(3, len(axes[0]) - 1), ky=min(3, len(axes[1]) - 1)
        )(*fine)
        if np.isfinite(z).all()
        else None
        for z in values
    ]
    colors = ["#31688e", "#e69f00", "#cc4778"]
    styles = ["--", ":", "-"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    for i, (ax, level) in enumerate(zip(axs, joint_levels)):
        for z, raw, color, style in zip(smooth, values, colors, styles):
            if z is None:
                if np.nanmin(raw) < level < np.nanmax(raw):
                    ax.contour(
                        *axes,
                        np.ma.masked_invalid(raw).T,
                        levels=[level],
                        colors=[color],
                        linestyles=[style],
                        linewidths=1.5,
                    )
            elif z.min() < level < z.max():
                ax.contour(
                    *fine,
                    z.T,
                    levels=[level],
                    colors=[color],
                    linestyles=[style],
                    linewidths=1.5,
                )
        ax.set(
            xlabel=names[0],
            ylabel=names[1],
            title=f"{i+1}σ joint reference: 2(ℓmax − ℓ) = {level:.4f}",
        )
        ax.ticklabel_format(axis="both", useOffset=False)
        ax.grid(alpha=0.15)
    fig.legend(
        handles=[
            Line2D([0], [0], color=co, ls=st, label=label)
            for co, st, label in zip(
                colors, styles, ["Surrogate", "True at surrogate", "True optimized"]
            )
        ],
        loc="outside lower center",
        ncol=3,
    )
    fig.suptitle(
        f"Joint 2D Gaussian levels • spline interpolation of {len(axes[0])} × {len(axes[1])} nodes"
        + (" (linear, with gaps, where nodes have no value)" if any(z is None for z in smooth) else "")
    )
    fig.savefig(out / "isocontours.png", dpi=170)
    fig.savefig(out / "isocontours.pdf")
    plt.close(fig)
    spline_minima = [float(z.min()) for z in smooth if z is not None]
    flags = [d.get("flag", 0) for row in t["diagnostics"] for d in row]
    ranges = json.loads((out / "ranges.json").read_text())
    warning = []
    if min(spline_minima) < -0.01:
        warning.append(
            "Spline interpolation dips below the anchor by more than 0.01 in 2*(max native log objective - native log objective); inspect resolution before interpreting contours."
        )
    if not t["anchor_consistent"]:
        warning.append(
            "A node beats the unrestricted true anchor; normalization unsettled."
        )
    if any(flags):
        warning.append(
            f"{sum(f!=0 for f in flags)} node solvers returned nonzero flags."
        )
    from profiles.reporting import anchor_warning, refused_warning

    without = int(np.isnan(values[2]).sum())
    if without:
        warning.append(
            f"{without} node(s) have no true value (the likelihood refused their start). Those nodes are not excluded, only "
            "unknown: the true curves are linear contours on the nodes that have values, not splines, and stop at the gap."
        )
    warning += anchor_warning(t) + refused_warning(out)
    for n, r in ranges.items():
        if r["clipped"]:
            warning.append(f"{n}: requested range clipped to saved bounds.")
    save(
        out / "surface_diagnostics.json",
        dict(
            warnings=warning,
            nonzero_solver_flags=sum(f != 0 for f in flags),
            surrogate_quality=s["quality"],
            anchor_consistent=t["anchor_consistent"],
            accuracy_certified=False,
            contour_levels=joint_levels,
            spline_min_delta_chi2=spline_minima,
            isocontour_interpolation="tensor spline, degree min(3,nodes-1), on twice the native log-objective difference clipped at 25, no additional backend calls"
            + ("; linear on the nodes with values for a curve with nodes that have none" if any(z is None for z in smooth) else ""),
            contour_note="Gaussian reference levels applied to twice the native log-objective difference (including backend priors); coverage not certified. Grid interpolation only.",
        ),
    )
    with (out / "surface.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                *names,
                "surrogate_logl",
                "direct_logl",
                "true_logl",
                "surrogate_dchi2",
                "direct_dchi2",
                "true_dchi2",
                "solver_flag",
            ]
        )
        for i, a in enumerate(axes[0]):
            for j, b in enumerate(axes[1]):
                w.writerow(
                    [
                        a,
                        b,
                        s["logl"][i][j],
                        t["direct_logl"][i][j],
                        t["logl"][i][j],
                        *[v[i, j] for v in values],
                        t["diagnostics"][i][j].get("flag", 0),
                    ]
                )
    counts = json.loads((out / "counts.json").read_text())
    (out / "README.md").write_text(
        "# Two-dimensional profile\n\n![Surface](surface.png)\n\n"
        "![Joint 1σ and 2σ isocontours](isocontours.png)\n\n"
        f'Pair: {names}. Grid: {len(axes[0])} × {len(axes[1])}. New backend attempts: {counts["backend_attempts"]}.\n\n'
        "The ordinate is 2*(maximum native log objective - native log objective), including backend priors; it is not necessarily likelihood-only chi-square. Historical dchi2 CSV keys retain this definition. The surrogate uses its own unrestricted anchor. Both true curves share the optimized true anchor. All nuisance coordinates remain free. Contours are interpolated on this grid; successful local optimization is not an accuracy or global-optimum certificate. Levels 2.30 and 6.18 are Gaussian reference levels, not a coverage claim for nonregular ridges or a posterior credible-region calculation.\n\n"
        + "\n".join("- " + v for v in warning)
        + "\n"
    )
