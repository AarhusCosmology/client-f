"""The World Cup profiles without SH0ES: the first ones (whole prior) against those with the cut (27 October 2026).

    python wc_profiles_compare.py [OUT.pdf]       # default: wc_profiles_compare.pdf here (a working figure, not the paper's)

The first profiles (studies sidr_drmd_wc_combined_boost10_{cosmo_hops60,fidm_wide,logGeff_wide} and the polish passes:
at every node the best true value of the node and of every point moved to it) took their optima wherever they were,
and 221 of the 324 nodes ended at 0 < Delta N_eff < 0.01, the region the paper leaves out. The new ones are the studies
on the run's copy with the cut (..._dncut_{cosmo,fidm_wide,logGeff_wide}; kit analysis/make_cut_run.py), read from
their node checkpoints in the kit's logs/ (analysis/snapshot_profiles.sh copies those of a running study), with the
log-posterior put back on the run's objective (build_wc.CUT_PRIOR_SHIFT), and improved by the polish passes
(wc_polish_setup.py, logs/chi2_points_wc_dncut_polish*_*.json) and the chains of true-likelihood optimisations
(wc_chain_setup.py, logs/profile_chain_wc_*_*.json) where those found better values. Both are drawn as Delta chi^2 to the same
reference, the best true value found outside the cut region (the paper's star); an old node whose best point lies in
the cut region is marked, and lies below zero where that region fits better.
"""
import json
import sys
from pathlib import Path

import numpy as np

import build_wc as W

D = "wc"
KIT = W.KIT
HERE = Path(__file__).resolve().parent
OLD = ("cosmo", "fidm_wide", "logGeff_wide")
NEW = [f"sidr_drmd_{D}_combined_boost10_dncut_{k}" for k in ("cosmo", "fidm_wide", "logGeff_wide")]
ORDER = ["omega_b", "omega_cdm", "H0", "logA", "n_s", "tau_reio", "m_int", "logGeff", "delta_Neff_drmd", "f_idm_drmd",
         "log10_zstop", "log10_G_over_aH"]
LABELS = {"omega_b": r"$\omega_\mathrm{b}$", "omega_cdm": r"$\omega_\mathrm{cdm}$", "H0": r"$H_0$",
          "logA": r"$\ln(10^{10}A_\mathrm{s})$", "n_s": r"$n_\mathrm{s}$", "tau_reio": r"$\tau_\mathrm{reio}$",
          "m_int": r"$m_\mathrm{int}$ [eV]", "logGeff": r"$\log_{10}(G_\mathrm{eff}\,\mathrm{MeV}^2)$",
          "delta_Neff_drmd": r"$\Delta N_\mathrm{eff}$", "f_idm_drmd": r"$f_\mathrm{idm}$", "log10_zstop": r"$\log_{10}z_\mathrm{stop}$",
          "log10_G_over_aH": r"$\log_{10}(\mathcal{G}/\mathcal{H})_\mathrm{ini}$"}


def names():
    run = W.latest(f"results/*_sidr_drmd_{D}_combined_boost10")
    return [q["name"] if isinstance(q, dict) else q for q in json.loads((run / "metadata.json").read_text())["parameters"]]


def cut_region(q):
    return 0 < q["delta_Neff_drmd"] < W.B.CUT


def old_profiles(names_):
    """{parameter: {coordinate: (log-objective, parameters)}}: the best of the old nodes and the points moved to them."""
    best = {}

    def offer(name, c, logl, q):
        key = round(c, 6)
        cur = best.setdefault(name, {}).get(key)
        if cur is None or logl > cur[0]:
            best[name][key] = (logl, q)
    for name, c, q, logl in W.node_points(D, names_):
        offer(name, c, logl, q)
    for e, q, f in W.moved_results(D):
        if q is not None and "logpost" in e and "profile" in e and np.isfinite(e["logpost"]):
            offer("f_idm_drmd" if e["profile"] == "f_idm_drmd_wide" else e["profile"], e["coordinate"], e["logpost"], q)
    return best


def new_profiles(names_):
    """The same from the cut studies' node checkpoints (on the run's objective), and the number of nodes per study."""
    best, counts = {}, {}
    for study in NEW:
        folder = KIT / "logs" / f"profiles_{study}_nodes"
        if not (folder / "surrogate_profiles.json").exists():
            continue
        profiled = json.loads((folder / "surrogate_profiles.json").read_text())["names"]
        files = sorted(folder.glob("node_*.json"))
        counts[study] = len(files)
        for f in files:
            node = json.loads(f.read_text())
            if len(node["x"]) != len(names_) or not np.isfinite(node["logl"]):
                continue
            name, c = profiled[int(f.stem.split("_")[1])], float(node["coordinate"])
            logl = float(node["logl"]) - W.CUT_PRIOR_SHIFT
            cur = best.setdefault(name, {}).get(round(c, 6))
            if cur is None or logl > cur[0]:
                best[name][round(c, 6)] = (logl, dict(zip(names_, map(float, node["x"]))))
    # the polish passes of the cut studies (wc_polish_setup.py), evaluated with the run's own Cobaya file: points moved to
    # a node's value bound the profile there, so each node keeps the best of its own value and theirs
    import wc_polish_setup
    polished = wc_polish_setup.polish_results(names_)
    for prof, c, q, logl in polished:
        key = round(c, 6)
        if key in best.get(prof, {}) and logl > best[prof][key][0]:
            best[prof][key] = (logl, q)
    if polished:
        counts[f"polish {len(polished)} points"] = 1
    # the chains of true-likelihood optimisations node by node (wc_chain_setup.py, the kit's smoke/profile_chain.py)
    import wc_chain_setup
    chained = wc_chain_setup.chain_results(names_)
    for prof, c, q, logl in chained:
        key = round(c, 6)
        if key in best.get(prof, {}) and logl > best[prof][key][0]:
            best[prof][key] = (logl, q)
    if chained:
        counts[f"chains {len(chained)} nodes"] = 1
    return best, counts


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "wc_profiles_compare.pdf"
    names_ = names()
    old = old_profiles(names_)
    new, counts = new_profiles(names_)
    star = json.loads((HERE / "plots" / "data" / "wc" / "best_fits.json").read_text())[D]
    ref = max([star["logl"]] + [v[0] for p in new.values() for v in p.values()]
              + [v[0] for p in old.values() for v in p.values() if not cut_region(v[1])])
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    for ax, name in zip(axes.flat, ORDER):
        for prof, colour, lab in ((old, "0.55", "first profiles"), (new, "#2b6cb0", "with the cut")):
            pts = sorted(prof.get(name, {}).items())
            if not pts:
                continue
            x = np.array([c for c, _ in pts])
            t = np.array([2 * (ref - v[0]) for _, v in pts])
            inside = np.array([cut_region(v[1]) for _, v in pts])
            ax.plot(x, t, "-", color=colour, lw=1.4, label=lab, zorder=2 if prof is old else 3)
            ax.plot(x[~inside], t[~inside], "o", color=colour, ms=3.5, zorder=4)
            if inside.any():
                ax.plot(x[inside], t[inside], "o", mfc="none", mec="#c0392b", ms=5, mew=1.1, zorder=4,
                        label="first: best point at $0<\\Delta N_\\mathrm{eff}<0.01$")
        ax.axhline(0, color="k", lw=0.6)
        ax.axhline(1, color="0.3", lw=0.7, ls="--")
        ax.axhline(4, color="0.3", lw=0.7, ls=":")
        ax.set_ylim(-2, 12)
        ax.set_xlabel(LABELS[name])
        n_new = len(new.get(name, {}))
        ax.set_title(f"{n_new} new node{'s' if n_new != 1 else ''}", fontsize=9)
    axes[0, 0].set_ylabel(r"$\Delta\chi^2$")
    axes[1, 0].set_ylabel(r"$\Delta\chi^2$")
    axes[2, 0].set_ylabel(r"$\Delta\chi^2$")
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=False)
    done = ", ".join(k if k.startswith(("polish", "chains")) else f"{k.split('_dncut_')[-1]} {v}" for k, v in counts.items()) or "none yet"
    fig.suptitle(f"World Cup data without SH0ES: profiles relative to the best true value outside the cut "
                 f"(ln L = {ref:.2f}); nodes of the cut studies: {done}", y=0.995, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out)
    print(f"written: {out}; reference ln L {ref:.3f}; new nodes per study {counts}")


if __name__ == "__main__":
    main()
