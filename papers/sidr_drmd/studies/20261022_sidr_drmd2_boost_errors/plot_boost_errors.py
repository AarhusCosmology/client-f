"""Boost errors and true checks of the second SIDR + DRMD set, one data set per figure (22 October 2026).

    python results/20261022_sidr_drmd2_boost_errors/plot_boost_errors.py [ppshoes|pp]

At its boost points: each part's network N at the 10 000 posterior points of boost N (RUN/boost/boost_it_N.csv,
truth `loglkl`, network `predicted`), points it had not seen, against N - S boosts trained on (S: the
hand-over, boost/start.json). Points outside the region the boost was sampled in (rho > edge,
boost/region_it_N.npz; the first boost has no region) get the region's wall, not the network: they are
counted and left out of the solid curve, and the dashed curve keeps them. At its posterior samples: the
true check (500 points) of the combined network at S+1, S+5 and S+10 (analysis/true_check.json of the
combined runs). Nothing is evaluated. Writes boost_errors_<data>.png and .json beside this file.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
KIT = REPO / "results" / "20261007_grendel_kit" / "hpc_client"


def part_errors(run):
    from sampling.region import Region

    start = json.loads((run / "boost" / "start.json").read_text())["start_iteration"]
    rows = []
    for csv in sorted(run.glob("boost/boost_it_*.csv"), key=lambda p: int(re.search(r"_it_(\d+)", p.name).group(1))):
        n = int(re.search(r"_it_(\d+)", csv.name).group(1))
        names = csv.read_text().split("\n", 1)[0].split(",")
        table = np.genfromtxt(csv, delimiter=",", skip_header=1, usecols=range(len(names) - 1))
        x = table[:, :names.index("loglkl")]
        d = table[:, names.index("loglkl")] - table[:, names.index("predicted")]
        ok = np.isfinite(d)
        region_path = run / "boost" / f"region_it_{n}.npz"
        inside = Region.load(region_path).rho(x) <= Region.load(region_path).inflate if region_path.exists() else np.ones(len(x), bool)
        keep = ok & inside
        rows.append(dict(iteration=n, boosts=n - start, points=int(ok.sum()), outside=int((ok & ~inside).sum()),
                         rms_all=float(np.sqrt(np.mean(d[ok] ** 2))), rms=float(np.sqrt(np.mean(d[keep] ** 2))),
                         mean=float(np.mean(d[keep])), q99=float(np.quantile(np.abs(d[keep]), 0.99)),
                         largest=float(np.max(np.abs(d[keep])))))
    return start, rows


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else "ppshoes"
    title = {"ppshoes": "Planck + PR4 lensing + DESI DR2 + Pantheon+SH0ES", "pp": "Planck + PR4 lensing + DESI DR2 + Pantheon+"}[data]
    parts = {side: sorted(KIT.glob(f"results/*_sidr_drmd2_{data}_{side}"))[-1] for side in ("low", "high")}
    out = dict(parts={}, combined=[])
    for side, run in parts.items():
        start, rows = part_errors(run)
        out["parts"][side] = dict(run=run.name, start=start, boosts=rows)
    for label, k in (("boost1", 1), ("boost5", 5), ("boost10", 10)):
        found = sorted(KIT.glob(f"results/*_sidr_drmd2_{data}_combined_{label}/analysis/true_check.json"))
        if found:
            e = json.loads(found[-1].read_text())["networks"][0]
            t = e["truth_minus_network"]
            out["combined"].append(dict(boosts=k, run=found[-1].parent.parent.name, rms=t["rms"], mean=t["mean"],
                                        largest=t["largest"], points=e["points"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colours = {"low": "#1f5fa8", "high": "#c0392b"}
    labels = {"low": r"low part ($\Delta N_\mathrm{eff} \leq 0.13$)", "high": r"high part ($\Delta N_\mathrm{eff} \geq 0.07$)"}
    low_bias = -1.0                                          # the bias panel's floor; lower values are written in
    for side, p in out["parts"].items():
        b = [r["boosts"] for r in p["boosts"]]
        ax[0].plot(b, [r["rms"] for r in p["boosts"]], "o-", color=colours[side], mfc="white", ms=5,
                   label=f"{labels[side]}: at its boost points")
        for r in p["boosts"]:
            if r["outside"]:
                ax[0].annotate(f"{r['outside']} pt{'s' * (r['outside'] > 1)} outside the region\n(rms {r['rms_all']:.0f} with them)",
                               (r["boosts"], r["rms"]), xytext=(0, 26), textcoords="offset points", ha="center", fontsize=6.5,
                               color=colours[side], arrowprops=dict(arrowstyle="-", color=colours[side], lw=0.6))
        ax[1].plot(b, [max(r["mean"], low_bias) for r in p["boosts"]], "o-", color=colours[side], mfc="white", ms=5, label=labels[side])
        for r in p["boosts"]:
            if r["mean"] < low_bias:
                ax[1].annotate(f"{r['mean']:.1f}", (r["boosts"], low_bias), xytext=(6, 4 if side == "low" else 14),
                               textcoords="offset points", fontsize=7, color=colours[side])
    if out["combined"]:
        c = out["combined"]
        ax[0].plot([e["boosts"] for e in c], [e["rms"] for e in c], "s-", color="k", ms=7, lw=2,
                   label="combined network: true check at its posterior samples (500)")
        ax[1].plot([e["boosts"] for e in c], [e["mean"] for e in c], "s-", color="k", ms=7, lw=2, label="combined network: true check")
    ax[0].set(yscale="log", ylim=(0.05, 60), xlabel="boosts the network was trained on",
              ylabel="rms of truth minus network (log-units)", title="Error of the network")
    ax[1].axhline(0, color="0.5", lw=0.8)
    ax[1].set(ylim=(low_bias - 0.05, 0.25), xlabel="boosts the network was trained on",
              ylabel="mean of truth minus network (log-units)", title="Bias of the network (values below -1 written in)")
    for a in ax:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.grid(alpha=0.25)
    ax[0].legend(fontsize=7, loc="upper right")
    ax[1].legend(fontsize=7, loc="lower right")
    starts = sorted({p["start"] for p in out["parts"].values()})
    fig.suptitle(f"SIDR + DRMD, {title}: the boost phase (hand-over at network {' and '.join(map(str, starts))})", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / f"boost_errors_{data}.png", dpi=140)
    (HERE / f"boost_errors_{data}.json").write_text(json.dumps(out, indent=1))
    for side, p in out["parts"].items():
        print(f"{side} (S = {p['start']}): " + ", ".join(f"{r['boosts']}: {r['rms']:.2f}" + (f" [{r['outside']} outside, all {r['rms_all']:.1f}]" if r["outside"] else "")
                                                       for r in p["boosts"]))
    print("combined true checks: " + ", ".join(f"S+{e['boosts']}: rms {e['rms']:.2f}, mean {e['mean']:+.3f}, largest {e['largest']:.2f}" for e in out["combined"]))
    print("written:", HERE / f"boost_errors_{data}.png")


if __name__ == "__main__":
    main()
