"""Portable figures and common-grid tables; no optimization/evaluation."""

import csv, json
from pathlib import Path
import numpy as np
from profiles.io import save
from profiles.metric import crossings


def anchor_warning(true):
    """The anchor is the best point of the study, the zero of every true curve. Its solver's
    verdict was in true_profiles.json only (anchor_diagnostics); an anchor that did not
    converge can still rise, and every curve with it."""
    diagnostics = true.get("anchor_diagnostics") or {}
    flag = diagnostics.get("flag") if isinstance(diagnostics, dict) else None
    if flag in (0, None):
        return []
    reason = diagnostics.get("reason")
    return ["The anchor, the best point of the study and the zero of every true curve, did not converge "
            f"(solver flag {flag}" + (f", {reason}" if reason else "") + "): the normalization may still rise."]


def refused_warning(out):
    """Design and validation points the likelihood refused, which the study left out."""
    path = Path(out) / "refused_points.json"
    if not path.exists():
        return []
    listed = json.loads(path.read_text())
    return [f"The likelihood refused {sum(map(len, listed.values()))} design or validation point(s), left out of the "
            "profile training: refused_points.json lists them, so the network has fewer new points than added_points."]


def crossings_with_values(grid, dchi2, level):
    """crossings over the nodes that have a value: a node the likelihood refused has none."""
    keep = np.isfinite(dchi2)
    return crossings(grid[keep], dchi2[keep], level) if keep.sum() >= 2 else []


def write_outputs(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    out = Path(out)
    read = lambda name: json.loads((out / name).read_text())
    s, t, d = [read(n + "_profiles.json") for n in ["surrogate", "true", "direct"]]
    rows = []
    intervals = {}
    warnings = []
    ranges = read("ranges.json")
    n = len(s["names"])
    fig, axes = plt.subplots(
        (n + 1) // 2,
        2,
        figsize=(11, 3.5 * ((n + 1) // 2)),
        squeeze=False,
        layout="constrained",
    )
    for ax, name in zip(axes.flat, s["names"]):
        g = np.array(t["profiles"][name]["grid"])
        ys = 2 * (s["best_logl"] - np.array(s["profiles"][name]["logl"]))
        # float: a node the likelihood refused has no value (None), NaN here.
        yt = 2 * (t["best_logl"] - np.array(t["profiles"][name]["logl"], float))
        yd = 2 * (t["best_logl"] - np.array(d["profiles"][name]["logl"], float))
        for y, label, style in [
            (ys, "Surrogate", "o-"),
            (yd, "True at surrogate", ":"),
            (yt, "True optimized", "s-"),
        ]:
            ax.plot(g, y, style, markersize=3, label=label)
        ax.axhline(1, color="gray", alpha=0.3)
        ax.axhline(4, color="gray", alpha=0.3)
        ax.set(xlabel=name, ylabel=r"$2(\ell_{\max}-\ell)$")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.ticklabel_format(axis="x", useOffset=False)
        ax.grid(alpha=0.2)
        for j, x in enumerate(g):
            rows.append(
                dict(
                    parameter=name,
                    coordinate=x,
                    surrogate_logl=s["profiles"][name]["logl"][j],
                    direct_logl=d["profiles"][name]["logl"][j],
                    true_logl=t["profiles"][name]["logl"][j],
                    surrogate_dchi2=ys[j],
                    direct_dchi2=yd[j],
                    true_dchi2=yt[j],
                    solver_flag=t["profiles"][name]["optimization"][j].get("flag"),
                )
            )
        intervals[name] = {
            str(level): dict(
                surrogate=crossings_with_values(g, ys, level),
                direct=crossings_with_values(g, yd, level),
                true=crossings_with_values(g, yt, level),
            )
            for level in [1.0, 4.0]
        }
        without = int(np.isnan(yt).sum())
        if without:
            warnings.append(
                f"{name}: {without} node(s) without a true value (the likelihood refused their start); "
                "intervals interpolate across them."
            )
        if ranges[name]["clipped"]:
            warnings.append(
                f"{name}: requested sigma range clipped to saved model bounds."
            )
        if "no two-sided" in ranges[name]["kind"]:
            warnings.append(
                f"{name}: no two-sided surrogate 1-sigma interval; full domain used."
            )
        for level, r in intervals[name].items():
            if len(r["true"]) != 2:
                warnings.append(
                    f'{name}: true level {level} has {len(r["true"])} grid-interpolated crossings; do not infer a two-sided interval.'
                )
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.legend(
        *axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=3
    )
    fig.suptitle("Surrogate and true profiles (native training objective)")
    fig.savefig(out / "profiles.png", dpi=160)
    fig.savefig(out / "profiles.pdf")
    plt.close(fig)
    with (out / "profiles.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    quality = read("true/complete.json")
    if not quality["anchor_consistent"]:
        warnings.append(
            "A profile node exceeds the true anchor; normalization is not settled."
        )
    if quality["budget_exhausted_nodes"]:
        warnings.append(
            f'{quality["budget_exhausted_nodes"]} nodes have nonzero solver flags; inspect optimizer diagnostics.'
        )
    warnings += anchor_warning(t) + refused_warning(out)
    save(
        out / "intervals_grid.json",
        dict(
            method="linear interpolation on the requested grid; NOT refined crossings",
            parameters=intervals,
        ),
    )
    save(
        out / "diagnostics.json",
        dict(
            warnings=warnings,
            accuracy_certified=False,
            normalization="surrogate uses its own peak; both true curves share optimized true anchor",
        ),
    )
    counts = read("counts.json")
    training_path = out / "training/training.json"
    if training_path.exists():
        train = read("training/training.json")
        checkpoint = f'Primary checkpoint: contrast, epoch {train["selected"]["contrast"]["epoch"]}. '
    else:
        checkpoint = "Primary checkpoint: unchanged saved source network; no augmentation, validation acquisition or retraining. "
    (out / "README.md").write_text(
        "# CLiENT profiles\n\n![Profiles](profiles.png)\n\n"
        + checkpoint
        + f'New backend attempts across all resumed sessions: {counts["backend_attempts"]}.\n\n'
        + "The ordinate is 2*(maximum native log objective - native log objective). All curves use the native objective matching the original training labels, including backend priors; this is not necessarily a likelihood-only chi-square. Historical CSV keys ending in dchi2 retain this same numerical definition for compatibility. Exact imported values and cache hits are not new calls. Historical baseline attempts are unknown; row count is not a call counter.\n\n"
        + "The surrogate has its own peak; direct and optimized true curves share the corrected true anchor. Lower discrepancies are better. Solver success is not a global-optimum or objective-accuracy certificate.\n\n"
        + "Intervals in intervals_grid.json are grid-interpolated, not refined. All starts, nuisance optima and flags are retained in JSON.\n\n"
        + (
            "\n".join("- " + w for w in warnings)
            if warnings
            else "No boundary/solver warnings were raised."
        )
        + "\n"
    )
