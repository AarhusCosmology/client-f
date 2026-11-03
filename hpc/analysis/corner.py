"""Corner plot of a network's T = 1 samples, with the contour levels asked for.

    python analysis/corner.py RUN_DIR [--network N ...] [--parameters 12 | --names a b c]
                              [--contours 0.68 0.95 0.99] [--out FILE]

Reads RUN_DIR/analysis/posterior_it_N.npz, which analysis/report.py wrote (the last network
by default; several networks are overlaid). Nothing is sampled and no network is loaded, so
it runs in seconds wherever the run's analysis/ folder is. The training box
(runs/<RUN>/domain_bounds.json) gives the hard edges, and the published values of
runs/<RUN>/reference_values.json, when there are any, are drawn as dashed lines.

A chain of a region with several ellipsoids that does not record how it was sampled is not
plotted: the shares of its modes may be those of its start (chain_sampling.py). One whose
cell weights are marked as not reliable is plotted, with a warning.

The outer contours are the least certain part of such a plot twice over: few samples lie
out there, and the true check tests the network where most of its posterior is.
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from chain_sampling import chain_sampling

HERE = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("CLIENT_HPC_DIR") or HERE) / "runs"     # README.md: CLIENT_HPC_DIR
LABELS = {"logA": r"\ln(10^{10} A_\mathrm{s})", "ns": r"n_\mathrm{s}", "theta_MC_100": r"100\theta_\mathrm{MC}",
          "ombh2": r"\Omega_\mathrm{b} h^2", "omch2": r"\Omega_\mathrm{c} h^2", "mnu": r"\Sigma m_\nu\,[\mathrm{eV}]",
          "nnu": r"N_\mathrm{eff}", "omk": r"\Omega_K", "w": r"w_0", "wa": r"w_a", "Alens": r"A_\mathrm{lens}", "tau": r"\tau",
          "omega_b": r"\omega_\mathrm{b}", "omega_cdm": r"\omega_\mathrm{cdm}", "H0": r"H_0", "n_s": r"n_\mathrm{s}",
          "tau_reio": r"\tau"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--network", type=int, nargs="*", help="networks to show (default: the last one with samples)")
    parser.add_argument("--parameters", type=int, default=12, help="the first N parameters")
    parser.add_argument("--names", nargs="*", help="these parameters instead")
    parser.add_argument("--contours", type=float, nargs="*", default=[0.68, 0.95, 0.99])
    parser.add_argument("--smooth", type=float, help="getdist's smoothing scale for 1D and 2D, in standard deviations "
                                                     "(default: 1D automatic, 2D 0.2, as the axion paper's figures)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    from getdist import MCSamples, plots

    analysis = args.run / "analysis"
    # posterior_it_N.npz only: a replaced chain is kept beside it as posterior_it_N.previous-<id>.npz
    files = {int(m.group(1)): p for p in analysis.glob("posterior_it_*.npz")
             if (m := re.fullmatch(r"posterior_it_(\d+)\.npz", p.name))}
    if not files:
        raise SystemExit(f"no posterior_it_N.npz in {analysis}: run analysis/report.py (without --no-posteriors) first")
    networks = args.network or [max(files)]
    missing = [n for n in networks if n not in files]
    if missing:
        raise SystemExit(f"no samples of network {missing} in {analysis}; there are {sorted(files)}")
    for n in networks:
        stop, warning = chain_sampling(files[n], args.run, n)
        if stop:
            raise SystemExit(f"{files[n]} {stop}. Run analysis/report.py again: it samples the network anew.")
        if warning:
            print(f"WARNING: {files[n].name} {warning}.")

    run_name = re.sub(r"^\d{8}_\d{6}_", "", args.run.resolve().name)
    box_file, reference_file = (RUNS / run_name / name for name in ("domain_bounds.json", "reference_values.json"))
    box = json.loads(box_file.read_text()) if box_file.exists() else {}
    reference = json.loads(reference_file.read_text())["parameters"] if reference_file.exists() else {}

    contours = sorted(args.contours)
    samples = []
    for n in networks:
        stored = np.load(files[n], allow_pickle=True)
        names = [str(name) for name in stored["names"]]
        shown = args.names or names[:args.parameters]
        columns = [names.index(name) for name in shown]
        x = stored["x"].astype(np.float64)[:, columns]
        # getdist finds the density level that holds each of `contours`; a level at a hard edge needs the edge
        samples.append(MCSamples(samples=x, names=shown, labels=[LABELS.get(name, name.replace("_", r"\_")) for name in shown],
                                 label=f"network {n}", ranges={name: tuple(box[name]) for name in shown if name in box},
                                 settings=dict(contours=contours, smooth_scale_2D=0.2 if args.smooth is None else args.smooth,
                                               smooth_scale_1D=-1 if args.smooth is None else args.smooth)))
        print(f"network {n}: {len(x)} samples of {len(shown)} parameters; outside the outermost contour about "
              f"{(1 - contours[-1]) * len(x):.0f} of them")

    g = plots.get_subplot_plotter(width_inch=max(9, 1.5 * len(shown)))
    g.settings.num_plot_contours = len(contours)
    g.settings.legend_fontsize = 14
    g.triangle_plot(samples, filled=True, markers={name: reference[name]["centre"] for name in shown if name in reference},
                    legend_labels=[f"network {n}: " + ", ".join(f"{100 * c:g}%" for c in contours) + " contours" for n in networks])
    out = args.out or analysis / ("corner_" + "_".join(f"{100 * c:g}" for c in contours) + ".png")
    g.export(str(out))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
