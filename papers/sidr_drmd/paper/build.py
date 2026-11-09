"""Figures, tables and in-text numbers of the CLiENT version of the SIDR + DRMD paper (22 October 2026).

    python build.py [--label boost5]

Everything the LaTeX source (main-client.tex) shows that comes from the runs is made here, from the Grendel
kit's synced results (../20261007_grendel_kit/hpc_client), so that a later set of networks only needs this
script run again with another --label (boost10 when the S+10 package is complete):

  posteriors   the combined emulator of each data set after LABEL (S + offset) boosts, sampled with BEST and
               pooled over its seeds (analysis/best_posterior_it_0.npz of results/*_sidr_drmd2_<data>_combined_
               <LABEL>, written by the kit's analysis/best_posterior.py); derived parameters from each part's
               latest derived-parameter emulator (the low part for delta_Neff < 0.10, the high part above)
  accuracy     the true checks of the combined emulators (analysis/true_check.json, every label there is) and
               the error of each part's network at the points of its next boost (boost/boost_it_N.csv)
  best fits    the network best fits of ../20261022_s5_aic_preview (aic_preview.json, profile_logGeff.json:
               the S+5 networks) evaluated with CLASS++ on Grendel per likelihood (kit logs/
               chi2_breakdown_<data>_<job>.json, smoke/submit_chi2_breakdown.sh)

Writes Figures/*.pdf, tables/*.tex and numbers.tex (macros used in the text) beside this file.
"""
import os
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PLOT_DATA = HERE / "plots" / "data"          # the data the plotting scripts in plots/ read (see plots/README.md)
REPO = Path(os.environ.get("CLIENT_PAPER_DATA", HERE.parent / "data"))   # the results the builds read (README.md)
KIT = Path(os.environ.get("CLIENT_PAPER_KIT", REPO / "kit"))              # the cluster folder the jobs ran in
# the emulator best fits per label (aic_preview.py, profile_logGeff.py --label L --out DIR)
BESTFITS = {"boost5": REPO / "results" / "20261022_s5_aic_preview", "boost10": REPO / "results" / "20261022_s10_best_fits"}
sys.path.insert(0, os.environ.get("CLIENT_REPO") or str(HERE.parents[2]))   # CLiENT: this branch of client-f
sys.path.insert(0, str(HERE.parent / "studies" / "20261022_sidr_drmd2_boost_errors"))

DATA = {"pp": r"\PDP{}", "ppshoes": r"\PDPS{}"}   # the paper's macros: Planck+DESI+Pantheon+, without and with SH0ES
OFFSET = {"boost1": 1, "boost5": 5, "boost10": 10}
COSMO = ["omega_b", "omega_cdm", "H0", "logA", "n_s", "tau_reio", "m_int", "logGeff", "delta_Neff_drmd",
         "f_idm_drmd", "log10_zstop", "log10_G_over_aH"]
LABELS = {"omega_b": r"\omega_\mathrm{b}", "omega_cdm": r"\omega_\mathrm{cdm}", "H0": r"H_0",
          "logA": r"\ln(10^{10}A_\mathrm{s})", "n_s": r"n_\mathrm{s}", "tau_reio": r"\tau_\mathrm{reio}",
          "m_int": r"m_\mathrm{int}", "logGeff": r"\log_{10}(G_\mathrm{eff}\,\mathrm{MeV}^2)",
          "delta_Neff_drmd": r"\Delta N_\mathrm{eff}", "f_idm_drmd": r"f_\mathrm{idm}",
          "log10_zstop": r"\log_{10}z_\mathrm{stop}", "log10_G_over_aH": r"\log_{10}(\mathcal{G}/\mathcal{H})_\mathrm{ini}",
          "sigma8": r"\sigma_8", "S8": r"S_8", "Omega_m": r"\Omega_\mathrm{m}", "rs_drag": r"r_\mathrm{drag}",
          "f_idr_drmd": r"f_\mathrm{idr}", "log10_zdec": r"\log_{10}z_\mathrm{dec}"}
UNITS = {"H0": r"\;[\mathrm{km\,s^{-1}\,Mpc^{-1}}]", "m_int": r"\;[\mathrm{eV}]", "rs_drag": r"\;[\mathrm{Mpc}]"}
COLOURS = {"pp": "#2b6cb0", "ppshoes": "#c0392b"}


def chi2_file(data, network, labels=("LCDM", "full", "DRMD", "SInu_strong", "SInu_weak"), required=True):
    """The newest per-likelihood job output (kit logs/chi2_breakdown_<data>_<job>.json) of this network that has the
    points `labels` (the model-comparison best fits by default; the posterior mean is a job of its own)."""
    def has(f):
        r = json.loads(f.read_text())
        return r.get("network") == network and set(labels) <= {e["label"] for e in r["results"]}
    found = [f for f in sorted(KIT.glob(f"logs/chi2_breakdown_{data}_*.json"), key=lambda f: int(f.stem.rsplit("_", 1)[1]))
             if has(f)]
    if not found and required:
        raise SystemExit(f"no logs/chi2_breakdown_{data}_*.json of {network} with {labels}: run smoke/submit_chi2_breakdown.sh")
    return found[-1] if found else None
MODELS = [("full", r"SI$\nu$+DRMD", 12), ("DRMD", "DRMD", 11), ("SInu_strong", r"SI$\nu$", 8), ("LCDM", r"$\Lambda$CDM", 7)]
STRONG = -2.5                           # the earlier boundary of the strong mode (until 28 October 2026; see mode_minimum)


def mode_minimum(g, w, box=(-5.5, -0.8)):
    """The boundary of the strongly interacting mode in log10 Geff (28 October 2026, Steen: "use the minimum between the
    modes as the threshold"): the minimum of the marginalised posterior between the strong mode's peak (the highest
    local maximum above -2.2) and the lower mode, from GetDist's 1D density with smooth_scale_1D 0.15. With the paper's
    0.3 the H0WC posterior has no minimum: its strong mode is a shoulder. Shares above it at 0.10-0.30 smoothing:
    PDP+ 65-67.5% (a flat valley, -2.3 to -2.6), PDP+SH0ES 30.2-30.4% (-2.18), H0WC+SH0ES 0.61% (-1.98)."""
    from getdist import MCSamples
    s = MCSamples(samples=np.asarray(g, float)[:, None], weights=np.asarray(w, float), names=["g"], ranges={"g": tuple(box)},
                  settings=dict(smooth_scale_1D=0.15))
    xs = np.linspace(-3.0, -1.2, 361)
    P = np.asarray(s.get1DDensity("g").Prob(xs)).ravel()
    peaks = [j for j in range(1, len(xs) - 1) if P[j] > P[j - 1] and P[j] >= P[j + 1] and xs[j] > -2.2]
    if not peaks:
        raise SystemExit("mode_minimum: no strongly interacting mode in the posterior of log10 Geff")
    k = max(peaks, key=lambda j: P[j])
    return float(xs[int(np.argmin(P[:k + 1]))])
# The analysis leaves out 0 < Delta N_eff < CUT (26 October 2026, Steen): there, at large f_idm, the interacting dark
# matter coupled to very little dark radiation has a small sound speed, and the likelihood rests on the non-linear power
# spectrum where halofit is not calibrated (it raises the lensing power while the linear power falls). The posteriors are
# those with Delta N_eff >= CUT; Delta N_eff = 0 (LCDM, SInu) is not affected.
CUT = 0.01
# A study on the run's copy with the cut (kit analysis/make_cut_run.py) evaluates the Cobaya prior of Delta N_eff uniform
# on [CUT, 2] instead of [0, 2]: its log-posterior is higher by this for the same likelihood, and is put back on the
# paper's objective (26 October 2026; 0.01 in chi^2, below the noise, but exact).
CUT_PRIOR_SHIFT = float(np.log(2.0 / (2.0 - CUT)))
# the lower bound of the prior on log10 Geff, the proxy for no self-interaction (Sec. 3.5)
GEFF_OFF = -5.5


def nested_models(q):
    """The nested models whose parameter space holds the point q of the combined model (a parameter dict): DRMD at the
    lower bound of log10 Geff, SInu at Delta N_eff = f_idm = 0, LCDM on both (Sec. 3.5)."""
    off = abs(float(q["logGeff"]) - GEFF_OFF) < 1e-9
    sinu = float(q["delta_Neff_drmd"]) == 0 and float(q["f_idm_drmd"]) == 0
    return [m for m, inside in (("DRMD", off), ("SInu", sinu), ("LCDM", off and sinu)) if inside]


# Profile studies (copied from results_profiles/ on Grendel into the kit's logs/, which is synced). A node computed
# by more than one study keeps the best true value; the emulator's profile comes from the first study listing it.
#   ("csv", file, study): profiles.csv of a complete study, and the study's name for its node statuses in its job log
#   ("nodes", directory): node_*.json checkpoints and surrogate_profiles.json of a study whose last step failed
PROFILE_SOURCES = {
    "pp": [("csv", "logs/profiles_pp_boost10.csv", "sidr_drmd2_pp_combined_boost10_cosmo_gtol3"),
           ("csv", "logs/profiles_pp_ocdm_maxfun.csv", "sidr_drmd2_pp_combined_boost10_ocdm_maxfun")],
    "ppshoes": [("csv", "logs/profiles_ppshoes_boost10.csv", "sidr_drmd2_ppshoes_combined_boost10_cosmo"),
                ("nodes", "logs/profiles_ppshoes_fidm_wide")],
}


# The cut at Delta N_eff = CUT (26 October 2026): the nodes of the studies above whose optimum lies at 0 < Delta N_eff <
# CUT are left out, and recomputed on a copy of the run with the box cut there (kit analysis/make_cut_run.py
# --suffix dncut; one study per affected parameter, on the same grid). Their tables join PROFILE_SOURCES once they exist.
CUT_STUDIES = {"pp": ["sidr_drmd2_pp_combined_boost10_dncut_" + p for p in ("omega_b3", "omega_cdm", "H0", "m_int", "fidm", "zstop3")]}
# (omega_b and z_stop with three nodes each: the profile code needs at least three; the first two-node studies stopped)
CUT_NODE_DIRS = {"pp": [("logs/profiles_pp_boost10_nodes", "logs/profiles_pp_boost10.csv"),
                        ("logs/profiles_pp_ocdm_maxfun_nodes", "logs/profiles_pp_ocdm_maxfun.csv")]}


def cut_nodes(data, names):
    """(parameter, rounded coordinate) of every node of the original studies of `data` whose optimum lies below the cut."""
    import csv
    out = set()
    for folder, table in CUT_NODE_DIRS.get(data, []):
        order = []
        for r in csv.DictReader(open(KIT / table)):
            if r["parameter"] not in order:
                order.append(r["parameter"])
        for f in sorted((KIT / folder).glob("node_*.json")):
            node = json.loads(f.read_text())
            x = dict(zip(names, node["x"]))
            if len(node["x"]) == len(names) and 0 < x["delta_Neff_drmd"] < CUT:
                out.add((order[int(f.stem.split("_")[1])], round(float(node["coordinate"]), 6)))
    return out


def cut_sources(data):
    """The recomputed studies of `data` whose table is in logs/ (the others are still running)."""
    return [("csv", f"logs/profiles_{s}.csv", s) for s in CUT_STUDIES.get(data, []) if (KIT / "logs" / f"profiles_{s}.csv").exists()]


# Node checkpoints (the optimum x of every node) of the S+10 profile studies, copied on Grendel from
# results_profiles/<study>/true/node_*.json into the kit's logs/ (synced): the global best fit is the best of them.
NODE_DIRS = {"pp": ["logs/profiles_pp_boost10_nodes"],
             "ppshoes": ["logs/profiles_ppshoes_boost10_nodes", "logs/profiles_ppshoes_fidm_wide"]}


def global_best(data, names, label, fits_chi2):
    """The best true point found: (x over the run's sampled parameters, true log-objective, source). The best stored
    profile node, or the CLASS++ value at the emulator's full-model best fit where that is higher or no node is stored."""
    best = None
    for folder in NODE_DIRS[data]:
        for f in sorted((KIT / folder).glob("node_*.json")):
            node = json.loads(f.read_text())
            if len(node["x"]) == len(names) and (best is None or node["logl"] > best[1]):
                best = (np.array(node["x"], float), float(node["logl"]), f"profile node {folder}/{f.name}")
    full = fits_chi2["full"]
    if best is None or full["logpost"] > best[1]:
        best = (np.array([full["params"][n] for n in names], float), float(full["logpost"]),
                "the emulator's full-model best fit, evaluated with CLASS++")
    return best


def profile_evaluations(data, names):
    """The combined model at the points of the profiles of data whose value was calculated with the true likelihood and
    split by likelihood (logs/chi2_points_<data>_profiles*_*.json), outside 0 < Delta N_eff < CUT, as entries of the model
    comparison (label full, with params and source). The parameters come from the job or else from the studies' node
    files (matched by coordinate and value); values computed with the cut copy of the Cobaya file are put back on the
    run's objective (CUT_PRIOR_SHIFT)."""
    nodes = []
    for folder in NODE_DIRS[data] + [f"logs/profiles_{data}_ocdm_maxfun_nodes"]:
        for f in sorted((KIT / folder).glob("node_*.json")):
            node = json.loads(f.read_text())
            if len(node["x"]) == len(names):
                nodes.append((round(float(node["coordinate"]), 6), float(node["logl"]), dict(zip(names, map(float, node["x"])))))
    out = []
    for f in sorted(KIT.glob(f"logs/chi2_points_{data}_profiles*_*.json")):
        r = json.loads(f.read_text())
        shift = CUT_PRIOR_SHIFT if "dncut" in r["cobaya"] else 0.0
        for e in r["results"]:
            if "chi2" not in e or not np.isfinite(e.get("logpost", -np.inf)):
                continue
            q = e.get("params") or next((p for c, logl, p in nodes if c == round(float(e.get("coordinate", np.nan)), 6)
                                         and abs(logl - e["logpost"]) < 1e-6), None)
            if q is None or 0 < float(q.get("delta_Neff_drmd", 1.0)) < CUT:
                continue
            out.append(dict(e, label="full", params=q, logpost=e["logpost"] - shift,
                            minus2_logprior=e["minus2_logprior"] + 2 * shift, source=f"{f.name} ({e.get('label')})"))
    return out


def node_statuses(study):
    """{(parameter, node): reason} from the job log of a profile study."""
    for f in sorted(KIT.glob("logs/profiles_*.out")):
        text = f.read_text(errors="replace")
        if f"results_profiles/{study}.yaml" in text:
            return {(m.group(1), int(m.group(2))): m.group(3)
                    for m in re.finditer(r"(\w+) node (\d+) \([^)]*\): \d+ calls, (\w+)", text.replace("\r", "\n"))}
    return {}


def read_profile_source(source, smax=None):
    """(parameter, coordinate, true log-objective, emulator Delta chi^2 to its own best, reason) per node. smax: the
    emulator's best to measure from instead (a study of a few nodes, recomputed with the cut, is measured from the best
    of the data set's main study, so that every node of a profile has the same reference)."""
    import csv
    if source[0] == "csv":
        status = node_statuses(source[2])
        rows = list(csv.DictReader(open(KIT / source[1])))
        if smax is None:
            smax = max(float(r["surrogate_logl"]) for r in rows)
        count = {}
        for r in rows:
            j = count.get(r["parameter"], 0)
            count[r["parameter"]] = j + 1
            yield (r["parameter"], float(r["coordinate"]), float(r["true_logl"]), 2 * (smax - float(r["surrogate_logl"])),
                   status.get((r["parameter"], j), "?"), float(r["direct_logl"]))
    else:
        folder = KIT / source[1]
        scan = json.loads((folder / "surrogate_profiles.json").read_text())
        names = scan["names"]
        for f in sorted(folder.glob("node_*.json")):
            k, j = (int(v) for v in re.findall(r"\d+", f.stem))
            node = json.loads(f.read_text())
            name = names[k]
            emu = scan["profiles"][name]["logl"][j]
            yield (name, float(node["coordinate"]), float(node["logl"]), 2 * (scan["best_logl"] - emu),
                   node["diagnostics"]["reason"], float(node["direct_logl"]))


def latest(pattern):
    found = sorted(KIT.glob(pattern))
    if not found:
        raise SystemExit(f"nothing matches {KIT / pattern}")
    return found[-1]


def number_of(path):
    return int(re.search(r"_it_(\d+)", path.name).group(1))


def derived_values(x, names, data):
    """The derived parameters at x from each part's latest derived emulator, by delta_Neff."""
    import tensorflow as tf
    import derived.emulator  # noqa: F401  (registers the emulator's layers)

    out, dnames = None, None
    low = x[:, names.index("delta_Neff_drmd")] < 0.10
    for side, mask in (("low", low), ("high", ~low)):
        part = latest(f"results/*_sidr_drmd2_{data}_{side}")
        model_file = max(part.glob("derived_data/derived_model_it_*.keras"), key=number_of)
        meta = json.loads((part / "derived_data" / "derived_parameters.json").read_text())
        these = [p["name"] for p in meta["parameters"]]
        if dnames is None:
            dnames, out = these, np.full((len(x), len(these)), np.nan)
        elif these != dnames:
            raise SystemExit(f"{part}: derived parameters {these}, not {dnames}")
        model = tf.keras.models.load_model(model_file, compile=False)
        if mask.any():
            out[mask] = model.predict(x[mask].astype(np.float32), batch_size=65536, verbose=0)
    return out, dnames


def posterior(data, label):
    from getdist import MCSamples

    run = latest(f"results/*_sidr_drmd2_{data}_combined_{label}")
    npz = run / "analysis" / "best_posterior_it_0.npz"
    if not npz.exists():
        raise SystemExit(f"{npz} is missing: the BEST seeds and analysis/best_posterior.py have not run for {label}")
    z = np.load(npz, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    x, w = z["x"].astype(np.float64), z["weights"].astype(np.float64)
    keep = w > w.max() * 1e-10
    x, w, runs = x[keep], w[keep], z["run"][keep]
    box = json.loads((KIT / "runs" / f"sidr_drmd2_{data}_combined_{label}" / "domain_bounds.json").read_text())
    box = box.get("bounds", box)
    # the cut at Delta N_eff = CUT: the samples below are left out, and their share, run by run, kept for the text
    out = x[:, names.index("delta_Neff_drmd")] < CUT
    ids = np.unique(runs)
    cut = dict(share=float(w[out].sum() / w.sum()), per_run=[float(w[out & (runs == r)].sum() / w[runs == r].sum()) for r in ids])
    x, w, runs = x[~out], w[~out], runs[~out]
    box = dict(box, delta_Neff_drmd=[CUT, box["delta_Neff_drmd"][1]])
    d, dnames = derived_values(x, names, data)
    allnames = names + dnames
    samples = MCSamples(samples=np.column_stack([x, d]), weights=w, names=allnames,
                        labels=[LABELS.get(n, n) for n in allnames],
                        ranges={n: tuple(box[n]) for n in names if n in box}, label=DATA[data],
                        settings=dict(smooth_scale_1D=0.3, smooth_scale_2D=0.4))
    summary = json.loads((run / "analysis" / "best_posterior_it_0.json").read_text())
    strong_min = mode_minimum(x[:, names.index("logGeff")], w)
    for key, column, edge in (("strong", "logGeff", strong_min), ("delta_Neff_drmd>0.1", "delta_Neff_drmd", 0.1)):
        above = x[:, names.index(column)] > edge
        per_run = [w[above & (runs == r)].sum() / w[runs == r].sum() for r in ids]
        sd = float(np.std(per_run, ddof=1))
        summary["splits"][key] = dict(best=float(w[above].sum() / w.sum()), seed_sd=sd, error=sd / np.sqrt(len(ids)),
                                      edge=edge)
    return dict(run=run, samples=samples, x=x, w=w / w.sum(), names=names, summary=summary,
                seeds=len(summary["runs"]), box=box, runs=runs, cut=cut)


def latex_value(samples, name):
    """68% two-sided, or a 95% one-sided limit where the posterior reaches a prior edge."""
    p = samples.getMargeStats().parWithName(name)
    if p.limits[1].limitType() == "none":
        return "unconstrained"                  # the 95% region reaches both ends of the prior
    if p.limits[0].twotail and p.limits[1].twotail:
        return samples.getInlineLatex(name, limit=1).split("=", 1)[1].strip()
    text = samples.getInlineLatex(name, limit=2)
    for sign in ("<", ">"):
        if sign in text:
            return sign + " " + text.split(sign, 1)[1].strip()
    return text.split("=", 1)[1].strip()


def write_plot_data(name, value):
    """A JSON file of plots/data/, numpy numbers made plain."""
    def plain(v):
        if isinstance(v, dict):
            return {str(k): plain(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        if isinstance(v, np.ndarray):
            return [plain(x) for x in v.tolist()]
        if isinstance(v, (np.floating, np.integer, np.bool_)):
            return v.item()
        return v
    PLOT_DATA.mkdir(parents=True, exist_ok=True)
    (PLOT_DATA / name).write_text(json.dumps(plain(value), indent=1))


def accuracy_table():
    """tables/accuracy.tex: the true checks of every combined emulator, Sec. 4 and Sec. 5 in one table (26 October 2026),
    from plots/data/accuracy.json and plots/data/wc/accuracy.json (build_wc.py); one row per data set, root mean square
    and largest |true - emulated ln L| at 500 posterior samples after 1, 5 and 10 boosts. Both builds call it."""
    rows = [("accuracy.json", "pp", r"\PDP{}"), ("accuracy.json", "ppshoes", r"\PDPS{}"),
            ("wc/accuracy.json", "wc", r"\HWC{}"), ("wc/accuracy.json", "wcmb", r"\HWCS{}")]
    ks = sorted(OFFSET.values())
    lines = [r"\begin{tabular}{l" + "cc" * len(ks) + "}", r"\toprule",
             r" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{\textbf{{after {k} boost{'s' * (k > 1)}}}}}" for k in ks) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(ks))),
             r"\textbf{Data} & " + " & ".join("rms & largest" for _ in ks) + r" \\", r"\midrule"]
    for i, (name, d, title) in enumerate(rows):
        f = PLOT_DATA / name
        checks = {e["k"]: e for e in json.loads(f.read_text())[d]["checks"]} if f.exists() else {}
        cells = []
        for k in ks:
            e = checks.get(k)
            cells += ["--", "--"] if e is None else [f"{e['rms']:.2f}", f"{e['largest']:.2f}"]
        if i == 2:
            lines.append(r"\midrule")
        lines.append(f"{title} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "tables" / "accuracy.tex").write_text("\n".join(lines) + "\n")


def fmt_share(value, error):
    """Two decimals, or as many as the error needs for one significant digit where it would print as 0.00 (28 October
    2026, a review of the draft: the World Cup shares printed as 0.10 +- 0.00 and 0.04 +- 0.00)."""
    k = 2 if error is None or error <= 0 or round(error, 2) > 0 else max(2, -int(np.floor(np.log10(error))))
    return f"{value:.{k}f}" + ("" if error is None else rf" \pm {error:.{k}f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--label", default="boost5", choices=sorted(OFFSET))
    args = parser.parse_args()
    label = args.label
    (HERE / "Figures").mkdir(exist_ok=True)
    (HERE / "tables").mkdir(exist_ok=True)

    import subprocess

    macros = {}

    def macro(name, value):
        assert re.fullmatch(r"[A-Za-z]+", name), name
        macros[name] = value

    # ---------------------------------------------------------------- posteriors
    post = {d: posterior(d, label) for d in DATA}
    # for the plotting scripts: the weighted samples of the twelve cosmological and the derived parameters (single
    # precision, as the BEST chains and the derived emulators give them) and the prior ranges of the former
    PLOT_DATA.mkdir(parents=True, exist_ok=True)
    for d in DATA:
        allnames = post[d]["samples"].getParamNames().list()
        keep = COSMO + [n for n in allnames if n not in post[d]["names"]]
        cols = [allnames.index(n) for n in keep]
        np.savez_compressed(PLOT_DATA / f"posterior_{d}.npz", samples=post[d]["samples"].samples[:, cols].astype(np.float32),
                            weights=post[d]["samples"].weights, names=np.array(keep),
                            ranges=json.dumps({n: list(post[d]["box"][n]) for n in COSMO}))
    rows = [("$\\Lambda$CDM", ["omega_b", "omega_cdm", "H0", "logA", "n_s", "tau_reio"]),
            ("SI$\\nu$", ["m_int", "logGeff"]),
            ("DRMD", ["delta_Neff_drmd", "f_idm_drmd", "log10_zstop", "log10_G_over_aH"]),
            ("Derived", ["sigma8", "S8", "Omega_m", "rs_drag", "f_idr_drmd"])]
    # Thomas Tram's layout (28 October 2026): the group in a column of its own on the left, the rules over the other three
    lines = [r"\begin{tabular}{rlcc}", r"\cmidrule[\heavyrulewidth](lr){2-4}",
             r"& \textbf{Parameter} & \textbf{\PDP{}} & \textbf{\PDPS{}} \\", r"\cmidrule(lr){2-4}"]
    for title, params in rows:
        group = []
        for n in params:
            values = [latex_value(post[d]["samples"], n) for d in DATA]
            cells = [v if v == "unconstrained" else f"${v}$" for v in values]
            group.append(f"${LABELS[n]}{UNITS.get(n, '')}$ & " + " & ".join(cells) + r" \\")
        if title == "SI$\\nu$":
            cells = []
            for d in DATA:
                s = post[d]["summary"]["splits"]["strong"]
                cells.append(f"${fmt_share(s['best'], s['error'])}$")
            # the share of the strong mode, above the minimum between the modes (mode_minimum; the caption says so)
            group.append(r"$P_\mathrm{strong}$ & " + " & ".join(cells) + r" \\")
        lines += [f"{title} & " + group[0]] + ["& " + row for row in group[1:]]
        lines.append(r"\cmidrule(lr){2-4}" if title != "Derived" else r"\cmidrule[\heavyrulewidth](lr){2-4}")
    lines.append(r"\end{tabular}")
    (HERE / "tables" / "params.tex").write_text("\n".join(lines) + "\n")

    for d, tag in (("pp", "PP"), ("ppshoes", "SH")):
        s, summ = post[d]["samples"], post[d]["summary"]
        for n, key in (("H0", "Hzero"), ("delta_Neff_drmd", "DNeff"), ("f_idm_drmd", "fidm"), ("S8", "Seight"),
                       ("n_s", "ns"), ("m_int", "mint"), ("logGeff", "Geff"), ("sigma8", "sigmaeight")):
            macro(f"{tag}{key}", latex_value(s, n))
        sp = summ["splits"]["strong"]
        macro(f"{tag}strong", f"{100 * sp['best']:.0f}")
        macro(f"{tag}strongerr", f"{100 * (sp['error'] or 0):.0f}")
        macro(f"{tag}strongmin", f"{sp['edge']:.2f}")
        dn = summ["splits"]["delta_Neff_drmd>0.1"]
        macro(f"{tag}lowDN", f"{100 * (1 - dn['best']):.0f}")
        # How the evidence depends on the lower edge of the log10 Geff prior. Below about -3 the likelihood is flat (no
        # interaction on observable scales), so the posterior density per unit log10 Geff at the lower edge, p0, fixes
        # it: with W the prior's width, ln B(self-interaction : none, DRMD free in both) = -ln(W p0); with the prior
        # restricted to the strong mode (log10 Geff > edge, the minimum between the modes) it would be ln(s / ((upper - edge) p0)), s the strong
        # mode's share; and it tends to 0 as the lower edge is moved to ever weaker coupling. The error is the scatter
        # of the BEST seeds (standard deviation over the seeds / sqrt of their number).
        lo, hi = post[d]["box"]["logGeff"]
        width = hi - lo
        # from the posterior with the cut at Delta N_eff = CUT
        g_all = post[d]["x"][:, post[d]["names"].index("logGeff")]
        w_all, r_all = post[d]["w"], post[d]["runs"]
        p0 = w_all[g_all < lo + 1.0].sum()
        seeds = []
        for r in np.unique(r_all):
            m = r_all == r
            seeds.append(-np.log(width * w_all[m][g_all[m] < lo + 1.0].sum() / w_all[m].sum()))
        macro(f"{tag}plateau", f"{p0:.2f}")
        macro(f"{tag}lnB", f"{-np.log(width * p0):.2f}")
        macro(f"{tag}lnBerr", f"{np.std(seeds, ddof=1) / np.sqrt(len(seeds)):.2f}")
        edge = summ["splits"]["strong"]["edge"]
        macro(f"{tag}lnBstrong", f"{np.log(w_all[g_all > edge].sum() / ((hi - edge) * p0)):.2f}")
        macro("Geffwidth", f"{width:g}")
        z = [r["lnZ"] for r in summ["runs"]]
        macro(f"{tag}seeds", str(len(z)))
        macro(f"{tag}lnZscatter", f"{np.std(z, ddof=1):.2f}" if len(z) > 1 else "--")
        x, w, names = post[d]["x"], post[d]["w"], post[d]["names"]
        fi, zs = x[:, names.index("f_idm_drmd")], x[:, names.index("log10_zstop")]
        corner = (fi > 0.15) & (zs > 4)
        share = 100 * w[corner].sum()
        macro(f"{tag}corner", f"{share:.1f}" if share >= 0.05 else "$<0.1$")
        # the posterior beyond the f_idm axis of the posterior figures (plots/common.py PLOT_LIMITS: 0.15), for the captions
        tail = 100 * w[fi > 0.15].sum() / w.sum()
        macro(f"{tag}fidmtail", f"{tail:.1f}" if tail >= 0.05 else "$<0.1$")
        # where the corner's samples lie in Delta N_eff (just above the cut, for Pantheon+): their 84% quantile
        if share >= 0.05:
            dn_c = x[corner, names.index("delta_Neff_drmd")]
            o = np.argsort(dn_c)
            cw = np.cumsum(w[corner][o]) / w[corner].sum()
            macro(f"{tag}cornerDNhigh", f"{dn_c[o][np.searchsorted(cw, 0.84)]:.2f}")
        else:
            macro(f"{tag}cornerDNhigh", "--")
        macro(f"{tag}cutshare", f"{100 * post[d]['cut']['share']:.0f}")
        macro(f"{tag}cutsharemin", f"{100 * min(post[d]['cut']['per_run']):.0f}")
        macro(f"{tag}cutsharemax", f"{100 * max(post[d]['cut']['per_run']):.0f}")

    # The cost of an evidence: network evaluations counted over the first 120 iterations (15 360 dead points) of a BEST
    # run of the Pantheon+ emulator with the paper's settings, on the laptop (../20261023_evidence_calls/count_calls.py:
    # client.py evidence with the function BEST is given wrapped in a counter; 2159 per dead point over the first 40
    # iterations, 2123 over the next 80), times the dead points of the Grendel runs (evidence/evidence_it_0_nlive2048
    # *.json, which also give their times on 16 cores). An estimate to within much better than a factor of 2.
    counts = sorted((REPO / "results" / "20261023_evidence_calls").glob("partial_it*_pp_seed42.json"),
                    key=lambda f: int(re.search(r"_it(\d+)_", f.name).group(1)))
    if counts:
        c = json.loads(counts[-1].read_text())
        runs = [json.loads(f.read_text()) for d in DATA
                for f in sorted(latest(f"results/*_sidr_drmd2_{d}_combined_{label}").glob("evidence/evidence_it_0_nlive2048*.json"))]
        dead = [r["n_dead"] for r in runs]
        minutes = [r["seconds"] / 60 for r in runs]
        n = c["evaluations_per_dead_point"] * np.mean(dead)
        e = int(np.floor(np.log10(n)))
        macro("evidenceevals", f"{n / 10 ** e:.1f}\\times10^{{{e}}}")
        macro("evidenceperdead", f"{round(c['evaluations_per_dead_point'], -1):.0f}")
        macro("evidencefraction", f"{100 * c['n_dead'] / np.mean(dead):.0f}")
        macro("evidencesteps", str(5 * c["ndim"]))
        macro("evidencelive", str(c["n_live"]))
        macro("evidenceupdates", str(c["n_live_updates"]))
        # the same number of evaluations of the true likelihood, at the core time per evaluation quoted in Sec. 3.4
        core_seconds, year = 9.0, 3.156e7
        macro("evidencetrueyears", f"{n * core_seconds / year:.0f}")
        macro("evidencetrueyearsfifth", f"{n * core_seconds / year / 5:.0f}")
        macro("evidencedeadmin", f"{min(dead) / 1e3:.0f}\\,000")
        macro("evidencedeadmax", f"{max(dead) / 1e3:.0f}\\,000")
        macro("evidenceminmin", f"{min(minutes):.0f}")
        macro("evidenceminmax", f"{max(minutes):.0f}")
    macro("label", f"S+{OFFSET[label]}")
    macro("draftnote", "they will be replaced by those of the final emulators after ten boosts and by the exact profile "
          "likelihoods when these are complete" if label != "boost10" else
          "the best fits of the model comparison are the emulator's, evaluated with CLASS++")
    # the noise of the true likelihood: see the accuracy section below (macros PPnoise, SHnoise and the rest)
    macro("nboosts", {1: "one", 5: "five", 10: "ten"}[OFFSET[label]])

    # the global best fits (stars in the triangle plots) and the posterior means
    runs_fit = json.loads((BESTFITS[label] / "aic_preview.json").read_text()).get("runs") or {
        "pp": "20261022_081623_sidr_drmd2_pp_combined_boost5", "ppshoes": "20261022_081623_sidr_drmd2_ppshoes_combined_boost5"}
    stars = {}
    for d in DATA:
        fits_chi2 = {e["label"]: e for e in json.loads(chi2_file(d, runs_fit[d]).read_text())["results"]}
        x, logl, source = global_best(d, post[d]["names"], label, fits_chi2)
        dv, dn = derived_values(x[None], post[d]["names"], d)
        stars[d] = dict(x=x, logl=logl, source=source, values=dict(zip(post[d]["names"] + dn, list(x) + list(dv[0]))))
        print(f"[stars] {d}: {source}, true log-objective {logl:.3f}")

    write_plot_data("best_fits.json", {d: dict(logl=stars[d]["logl"], source=stars[d]["source"],
                                               values={k: float(v) for k, v in stars[d]["values"].items()}) for d in DATA})

    # ---------------------------------------------------------------- accuracy
    from plot_boost_errors import part_errors
    acc = {}
    for d in DATA:
        parts = {side: latest(f"results/*_sidr_drmd2_{d}_{side}") for side in ("low", "high")}
        acc[d] = dict(parts={side: dict(zip(("start", "boosts"), part_errors(run))) for side, run in parts.items()},
                      checks=[])
        for lab, k in sorted(OFFSET.items(), key=lambda t: t[1]):
            found = sorted(KIT.glob(f"results/*_sidr_drmd2_{d}_combined_{lab}/analysis/true_check.json"))
            if found:
                e = json.loads(found[-1].read_text())["networks"][0]
                t = e["truth_minus_network"]
                acc[d]["checks"].append(dict(k=k, rms=t["rms"], mean=t["mean"], largest=t["largest"], n=e["points"]))
    write_plot_data("accuracy.json", acc)
    accuracy_table()

    # ---------------------------------------------------------------- the noise floor of the true likelihood (appendix)
    # kit true_profiles/noise_probe.py on the S+10 combined runs (jobs 6942764/65): 21 points along lines of +-0.25
    # posterior sigma through the posterior mean, one per sampled parameter and four random directions in which every
    # parameter moves. The probe itself fits a quartic; here the noise of a line is the rms about a polynomial of
    # degree NOISE_DEGREE (with the degrees of freedom of the fit removed): along log10 z_stop, log10 Geff,
    # log10 (G/aH)_ini and (Pantheon+) f_idm the true value is smooth but too curved for a quartic over the line
    # (residual 0.02-0.15 with a quartic, at most 0.004 with degree 8), while along the noisy lines the rms is the same
    # for degrees 2 to 8 and the residuals of neighbouring points are not positively correlated.
    NOISE_DEGREE = 8
    probe = {}
    for d in DATA:
        found = sorted(KIT.glob(f"results/*_sidr_drmd2_{d}_combined_boost10/analysis/noise_probe.json"))
        if found:
            probe[d] = json.loads(found[-1].read_text())
    if len(probe) == len(DATA):
        s_line = np.array(probe["pp"]["s"])
        background = ["omega_b", "omega_cdm", "H0", "tau_reio", "m_int"]
        smooth = ["logGeff", "log10_zstop", "log10_G_over_aH"]
        foreground = ["amp_143", "amp_217", "amp_143x217", "n_143", "n_217", "n_143x217", "calTE", "calEE"]
        rand = [k for k in probe["pp"]["lines"] if k.startswith("random")]
        nparams = sum(1 for k in probe["pp"]["lines"] if not k.startswith("random"))

        def residual(d, line, degree=NOISE_DEGREE):
            v = np.array(probe[d]["lines"][line]["values"], float)
            return v - np.polyval(np.polyfit(s_line, v, degree), s_line)

        def noise(d, line, degree=NOISE_DEGREE):
            return float(np.sqrt(np.sum(residual(d, line, degree) ** 2) / (len(s_line) - degree - 1)))

        def sci(v):
            e = int(np.floor(np.log10(v)))
            m = round(v / 10 ** e)
            if m == 10:
                m, e = 1, e + 1
            return f"{m}\\times10^{{{e}}}" if m != 1 else f"10^{{{e}}}"

        bg = [noise(d, k) for d in DATA for k in background]
        macro("noisebgmin", f"{min(bg):.2f}")
        macro("noisebgmax", f"{max(bg):.2f}")
        macro("noiseprimmax", f"{max(noise(d, k) for d in DATA for k in ('logA', 'n_s')):.3f}")
        macro("noiseAplanck", f"{max(noise(d, 'A_planck') for d in DATA):.3f}")
        macro("noisefgmax", sci(max(noise(d, k) for d in DATA for k in foreground)))
        macro("noisesmoothquartic", f"{max(noise(d, k, 4) for d in DATA for k in smooth):.2f}")
        macro("noisesmoothmax", f"{max(noise(d, k) for d in DATA for k in smooth):.3f}")
        macro("noiserepro", sci(max(abs(l["difference"]) for d in DATA for l in probe[d]["labels"])))
        noisy = [(d, k) for d in DATA for k in background + rand if noise(d, k) > 0.02]
        lag = [np.corrcoef(residual(d, k)[:-1], residual(d, k)[1:])[0, 1] for d, k in noisy]
        macro("noiselagmin", f"{min(lag):+.2f}".replace("-", "$-$"))
        macro("noiselagmax", f"{max(lag):+.2f}".replace("-", "$-$"))
        for d, tag in (("pp", "PP"), ("ppshoes", "SH")):
            j = probe[d]
            centre = j["lines"]["omega_b"]["values"][len(s_line) // 2]
            r = [noise(d, k) for k in rand]
            eps = float(np.median(r))
            macro(f"{tag}noise", f"{eps:.2f}")
            macro(f"{tag}noisemax", f"{max(r):.2f}")
            macro(f"{tag}noisemin", f"{min(r):.3f}")
            macro(f"{tag}noiseNeff", f"{noise(d, 'delta_Neff_drmd'):.3f}")
            macro(f"{tag}noisecentre", f"{max(l['now'] for l in j['labels']) - centre:.0f}")
            macro(f"{tag}noisefloor", f"{(nparams - 1) * max(r) ** 2 / j['summary']['step'] ** 2:.1f}")
            macro(f"{tag}noisechi", f"{2 * eps:.2f}")
            final = next((e for e in acc[d]["checks"] if e["k"] == OFFSET[label]), None)
            macro(f"{tag}emuonly", f"{np.sqrt(max(final['rms'] ** 2 - eps ** 2, 0)):.2f}" if final else "--")
        macro("noisespan", f"{probe['pp']['span']:g}")
        macro("noisepoints", str(probe["pp"]["points"]))
        macro("noisespacing", f"{s_line[1] - s_line[0]:.3f}")
        macro("noisenparams", str(nparams))
        macro("noisenfree", str(nparams - 1))
        macro("noisenewtonstep", f"{probe['pp']['summary']['step']:g}")
        macro("noisedegree", str(NOISE_DEGREE))


    # ---------------------------------------------------------------- the noise at the best fits
    # kit smoke/noise_probe_cobaya.py, job 6949742 (results/20261023_noise_variants/make_variants.py): the probe above
    # (21 points along +-0.25 posterior sigma, four random directions), with the Pantheon+ likelihoods and precision
    # settings, centred on best fits instead of the posterior mean, with the model switched on step by step. The first
    # three share the emulator best fit on the LambdaCDM face; massive neutrinos sum to 0.08 eV. Same statistic as above.
    # Only numbers now (Sec. 3.4 quotes \numnoisebffull); their table left the paper with the noise appendix on 27 October
    # 2026 and is no longer written (28 October; the last one is in retired/tables_20261028/).
    VARIANTS = ["lcdm_massless", "lcdm_mnu", "sinu_off", "sinu_strong", "full"]
    BACKGROUND = ["omega_b", "omega_cdm", "H0", "tau_reio", "m_int"]
    variant = {}
    for name in VARIANTS:
        found = sorted(KIT.glob(f"logs/noise_probe_{name}_*.json"))
        if found:
            variant[name] = json.loads(found[-1].read_text())
    if len(variant) == len(VARIANTS):
        def vnoise(j, line, degree=NOISE_DEGREE):
            s = np.array(j["s"], float)
            v = np.array(j["lines"][line]["values"], float)
            r = v - np.polyval(np.polyfit(s, v, degree), s)
            return float(np.sqrt(np.sum(r ** 2) / (len(s) - degree - 1)))

        eps_bf = {}
        for name in VARIANTS:
            j = variant[name]
            eps_bf[name] = float(np.median([vnoise(j, k) for k in j["lines"] if k.startswith("random")]))
            if j["summary"]["centre_spread"] != 0:
                raise SystemExit(f"noise variant {name}: the centre did not reproduce")
        macro("noisebfmin", f"{min(eps_bf.values()):.3f}")
        macro("noisebfmax", f"{max(eps_bf.values()):.3f}")
        macro("noisebflcdm", f"{eps_bf['lcdm_massless']:.3f}")
        macro("noisebffull", f"{eps_bf['full']:.3f}")
        macro("noisebfstrong", f"{eps_bf['sinu_strong']:.3f}")
        macro("noisebfsingle", f"{max(vnoise(variant[n], k) for n in VARIANTS for k in BACKGROUND if k in variant[n]['lines']):.2f}")
        macro("noisebfchi", f"{2 * max(eps_bf.values()):.2f}")
        rf = [vnoise(variant["full"], k) for k in variant["full"]["lines"] if k.startswith("random")]
        macro("noisebffloor", f"{(nparams - 1) * max(rf) ** 2 / probe['pp']['summary']['step'] ** 2:.2f}")
        for d, tag in (("pp", "PP"), ("ppshoes", "SH")):        # where the posterior means lie: between the modes
            z = np.load(PLOT_DATA / f"posterior_{d}.npz", allow_pickle=True)
            names = [str(n) for n in z["names"]]
            macro(f"{tag}noisecentreGeff", f"{np.average(z['samples'][:, names.index('logGeff')], weights=z['weights']):.1f}"
                  .replace("-", "$-$"))

    # the true check in the corner of dark acoustic oscillations (f_idm > 0.15, z_stop > 10^4)
    for d, tag in (("pp", "PP"), ("ppshoes", "SH")):
        found = sorted(KIT.glob(f"results/*_sidr_drmd2_{d}_combined_{label}/analysis/true_check_it_0.npz"))
        if found:
            tc = np.load(found[-1], allow_pickle=True)
            nm = [str(v) for v in tc["names"]]
            xc = tc["x"].astype(float)
            dd = tc["truth"] - tc["network"]
            m = (xc[:, nm.index("f_idm_drmd")] > 0.15) & (xc[:, nm.index("log10_zstop")] > 4) & np.isfinite(dd)
            macro(f"{tag}cornerchecks", str(int(m.sum())))
            macro(f"{tag}cornercheckmean", f"{dd[m].mean():+.2f}" if m.any() else "--")
    for d, tag in (("pp", "PP"), ("ppshoes", "SH")):
        c = {e["k"]: e for e in acc[d]["checks"]}
        e = c.get(OFFSET[label]) or c[max(c)]
        macro(f"{tag}checkrms", f"{e['rms']:.2f}")
        macro(f"{tag}checklargest", f"{e['largest']:.1f}")
        macro(f"{tag}checkmean", f"{e['mean']:+.3f}")
        for side in ("low", "high"):
            last = acc[d]["parts"][side]["boosts"][-1]
            macro(f"{tag}{side}lastboost", f"{last['rms']:.2f}")
        starts = sorted({acc[d]["parts"][s]["start"] for s in ("low", "high")})
        macro(f"{tag}handover", " and ".join(map(str, [acc[d]["parts"]["low"]["start"], acc[d]["parts"]["high"]["start"]])))
        n = 0
        for side in ("low", "high"):
            part = latest(f"results/*_sidr_drmd2_{d}_{side}")
            data_file = max(part.glob("training_data/data_it_*.csv"), key=number_of)
            n += sum(1 for _ in open(data_file)) - 1
        macro(f"{tag}trainingpoints", f"{n / 1e3:.0f}\\,000")

    # ---------------------------------------------------------------- best fits and model comparison
    chi2 = {}
    for d in DATA:
        aic_dir = BESTFITS[label]
        runs_fit = json.loads((aic_dir / "aic_preview.json").read_text()).get("runs") or {
            "pp": "20261022_081623_sidr_drmd2_pp_combined_boost5", "ppshoes": "20261022_081623_sidr_drmd2_ppshoes_combined_boost5"}
        r = json.loads(chi2_file(d, runs_fit[d]).read_text())
        chi2[d] = {e["label"]: e for e in r["results"]}
    # the combined model at the best point found (28 October 2026, Steen): a point of its profiles, calculated with the
    # true likelihood, where one is higher than the emulators' fit (they were 0.6 and 0.8 lower in chi^2 than the stars).
    # The emulators' own fits stay in emulated, for the agreement of the true and emulated likelihoods there.
    # The nested models likewise at the points of the profiles on their boundaries (nested_models; 28 October 2026, Steen):
    # DRMD 0.40 (PDP+) and 0.24 (PDP+SH0ES) lower in chi^2, SInu 0.55 lower with PDP+.
    emulated = {d: dict(chi2[d]) for d in DATA}
    for d in DATA:
        for e in profile_evaluations(d, post[d]["names"]):
            for key in ["full"] + [{"SInu": "SInu_strong"}.get(m, m) for m in nested_models(e["params"])]:
                if e["logpost"] > chi2[d][key]["logpost"]:
                    chi2[d][key] = dict(e, label=key)
        print(f"[fits] {d}: " + ", ".join(f"{k} from {chi2[d][k].get('source', 'the emulators')}" for k in chi2[d]))
    fits = json.loads((BESTFITS[label] / "aic_preview.json").read_text())
    lines = [r"\begin{tabular}{lc|ccc|ccc}", r"\toprule",
             r" & & \multicolumn{3}{c|}{\textbf{\PDP{}}} & \multicolumn{3}{c}{\textbf{\PDPS{}}} \\",
             r"\textbf{Model} & $N$ & $\chi^2$ & $\Delta\chi^2$ & $\Delta$AIC & $\chi^2$ & $\Delta\chi^2$ & $\Delta$AIC \\",
             r"\midrule"]
    for key, name, n in MODELS:
        cells = []
        for d in DATA:
            ref = chi2[d]["LCDM"]["chi2_total"]
            v = chi2[d][key]["chi2_total"]
            dc = v - ref
            cells += [f"{v:.2f}", "$0$" if key == "LCDM" else f"${dc:+.2f}$",
                      "$0$" if key == "LCDM" else f"${dc + 2 * (n - 7):+.2f}$"]
        lines.append(f"{name} & {n} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "tables" / "model_comparison.tex").write_text("\n".join(lines) + "\n")

    # the best fits inside the region the paper leaves out, 0 < Delta N_eff < CUT, for the discussion (27 October 2026):
    # minimisations with the true likelihood alone on the kit's smoke/cobaya_pp_dnvalley.yaml (pp_region_minimise_setup.py),
    # against the fits of the table, which all lie outside it. "(pending)" until the jobs are in logs/.
    below, running = {}, []
    for prefix, model in (("full", "full"), ("drmd", "DRMD")):
        for f in sorted(KIT.glob(f"logs/minimise_{prefix}_pp_valley_*.json")):
            r = json.loads(f.read_text())
            if r.get("cobaya") != "smoke/cobaya_pp_dnvalley.yaml":
                continue
            # the job rewrites its JSON after every start: it is final once its log ends with "done"
            out = KIT / "logs" / f"minimise_{f.stem.rsplit('_', 1)[1]}.out"
            if not (out.exists() and out.read_text().rstrip().endswith("done")):
                running.append(f.stem.rsplit("_", 1)[1])
            for entry in r["starts"].values():
                b = entry.get("best")
                if b and np.isfinite(b.get("logpost", -np.inf)) and (model not in below or b["logpost"] > below[model]["logpost"]):
                    below[model] = b
    words = ["overabove", "overfull"] + [f"{k}{w}" for w in ("full", "drmd") for k in ("dchi", "gain", "aic")] + [
        f"{side}{k}" for side in ("below", "above") for k in ("DN", "fidm", "zstop", "GH", "Geff", "Hzero", "ocdm", "ns")] + [
        "TT", "EE", "cam", "lens", "desi", "sn"]
    if len(below) == 2:
        ref_l, full_a = chi2["pp"]["LCDM"]["chi2_total"], chi2["pp"]["full"]
        lik = lambda b: sum(b["chi2"].values())  # noqa: E731
        macro("PPbelowoverabove", f"{full_a['chi2_total'] - lik(below['full']):+.2f}")
        macro("PPbelowoverfull", f"{lik(below['full']) - full_a['chi2_total']:.2f}")   # how much worse the fit inside is
        for model, word, n in (("full", "full", 12), ("DRMD", "drmd", 11)):
            dc = lik(below[model]) - ref_l
            macro(f"PPbelowdchi{word}", f"{dc:+.1f}")
            macro(f"PPbelowgain{word}", f"{-dc:.1f}")
            macro(f"PPbelowaic{word}", f"{dc + 2 * (n - 7):+.1f}")
        for side, q in (("below", below["full"]["params"]), ("above", chi2["pp"]["full"]["params"])):
            dn = q["delta_Neff_drmd"]
            macro(f"PP{side}DN", f"{dn:.2f}" if dn >= 0.05 else f"{dn:.3f}" if dn >= 0.001 else
                  f"{dn / 10 ** np.floor(np.log10(dn)):.0f}\\times10^{{{int(np.floor(np.log10(dn)))}}}")
            for k, key, fmt in (("fidm", "f_idm_drmd", ".3f"), ("zstop", "log10_zstop", ".2f"), ("GH", "log10_G_over_aH", ".1f"),
                                ("Geff", "logGeff", ".2f"), ("Hzero", "H0", ".2f"), ("ocdm", "omega_cdm", ".4f"), ("ns", "n_s", ".3f")):
                macro(f"PP{side}{k}", f"{q[key]:{fmt}}")
        for word, name in (("TT", "planck_2018_lowl.TT"), ("EE", "planck_2018_lowl.EE"), ("cam", "planck_NPIPE_highl_CamSpec"),
                           ("lens", "planckpr4lensing"), ("desi", "bao.desi_dr2"), ("sn", "sn.")):
            kb = next(k for k in below["full"]["chi2"] if k.startswith(name))
            ka = next(k for k in full_a["chi2"] if k.startswith(name))
            macro(f"PPbelow{word}", f"{below['full']['chi2'][kb] - full_a['chi2'][ka]:+.2f}")
    else:
        for w in words:
            macro(f"PPbelow{w}" if not w.startswith(("below", "above")) else f"PP{w}", "(pending)")
    macro("PPbelownote", (r" {\color{red}\small [Preliminary: minimisation job " + ", ".join(running)
                          + r" was still running at the last sync.]}" if running else "") if len(below) == 2 else
          r" {\color{red}\small [Pending: the minimisations of DRMD and the SI$\nu$+DRMD model in this region for \PDP{} "
          r"(kit smoke/minimise\_\{drmd,full\}\_pp\_valley.json). The best profile node there is 2.5 above the global "
          r"best fit in $\chi^2$, an upper bound.]}")

    def like_key(d, name):
        return next(k for k in chi2[d]["LCDM"]["chi2"] if k.startswith(name)) if name == "sn" else name

    calibshift = {}
    # the chi^2 of each likelihood at these best fits: numbers in the text only. Their table (tab:chi2_breakdown) left the
    # paper on 27 October 2026 and is no longer written (28 October; the last one is in retired/tables_20261028/).
    for d, tag in (("pp", "PP"), ("ppshoes", "SH")):
        ref = chi2[d]["LCDM"]["chi2_total"]
        for key, word in (("full", "full"), ("DRMD", "drmd"), ("SInu_strong", "sinu"), ("SInu_weak", "weak")):
            dc = chi2[d][key]["chi2_total"] - ref
            macro(f"{tag}dchi{word}", f"{dc:+.1f}")
            macro(f"{tag}gain{word}", f"{-dc:.1f}")
            n = dict(full=12, DRMD=11, SInu_strong=8, SInu_weak=8)[key]
            macro(f"{tag}aic{word}", f"{dc + 2 * (n - 7):+.1f}")
        # the combination against each of its parts: the chi^2 gained by the combined model over DRMD and over SInu alone
        macro(f"{tag}fullgainoverdrmd", f"{chi2[d]['DRMD']['chi2_total'] - chi2[d]['full']['chi2_total']:.1f}")
        macro(f"{tag}fullgainoversinu", f"{chi2[d]['SInu_strong']['chi2_total'] - chi2[d]['full']['chi2_total']:.1f}")
        # how much lower the combined model's best fit (a point of its profiles) is than the emulators' fit (Sec. 3)
        macro(f"{tag}fullprofilegain", f"{emulated[d]['full']['chi2_total'] - chi2[d]['full']['chi2_total']:.1f}")
        for key, word in (("DRMD", "drmd"), ("SInu_strong", "sinu")):
            macro(f"{tag}{word}profilegain", f"{emulated[d][key]['chi2_total'] - chi2[d][key]['chi2_total']:.1f}")
        # the calibration priors counted as part of the likelihood instead (Sec. 3.5): how much each Delta chi^2 of the
        # table to LCDM would change, the objective (-2 ln posterior) minus the chi^2 of the likelihoods, model to LCDM
        prior = {k: -2 * chi2[d][k]["logpost"] - chi2[d][k]["chi2_total"] for k, _, _ in MODELS}
        calibshift[d] = max(abs(prior[k] - prior["LCDM"]) for k in prior)
        worst = max(abs(emulated[d][k]["logpost"] - emulated[d][k]["network_logpost"]) for k in emulated[d])
        macro(f"{tag}bestfiterr", f"{worst:.2f}")
        cam = "planck_NPIPE_highl_CamSpec.TTTEEE"
        macro(f"{tag}sinucamspec", f"{chi2[d]['SInu_strong']['chi2'][cam] - chi2[d]['LCDM']['chi2'][cam]:+.1f}")
        macro(f"{tag}sinudesi", f"{chi2[d]['SInu_strong']['chi2']['bao.desi_dr2.desi_bao_all'] - chi2[d]['LCDM']['chi2']['bao.desi_dr2.desi_bao_all']:+.1f}")
        macro(f"{tag}sinulowTT", f"{chi2[d]['SInu_strong']['chi2']['planck_2018_lowl.TT'] - chi2[d]['LCDM']['chi2']['planck_2018_lowl.TT']:+.1f}")
        sn = like_key(d, "sn")
        macro(f"{tag}sinusn", f"{chi2[d]['SInu_strong']['chi2'][sn] - chi2[d]['LCDM']['chi2'][sn]:+.1f}")
        macro(f"{tag}drmdsn", f"{chi2[d]['DRMD']['chi2'][sn] - chi2[d]['LCDM']['chi2'][sn]:+.1f}")
        macro(f"{tag}drmdcamspec", f"{chi2[d]['DRMD']['chi2'][cam] - chi2[d]['LCDM']['chi2'][cam]:+.1f}")
        # the coordinates of the fits of the table (a point of the profiles where one is better than the emulator's fit)
        macro(f"{tag}sinuns", f"{chi2[d]['SInu_strong']['params']['n_s']:.3f}")
        macro(f"{tag}lcdmns", f"{chi2[d]['LCDM']['params']['n_s']:.3f}")
        macro(f"{tag}sinuHzero", f"{chi2[d]['SInu_strong']['params']['H0']:.1f}")
        macro(f"{tag}lcdmHzero", f"{chi2[d]['LCDM']['params']['H0']:.1f}")
        macro(f"{tag}sinuGeff", f"{chi2[d]['SInu_strong']['params']['logGeff']:.2f}")
    macro("calibshift", f"{max(calibshift.values()):.1f}")
    for key, word in (("LCDM", "lcdm"), ("SInu_strong", "sinu"), ("DRMD", "drmd"), ("full", "full")):
        cost = chi2["ppshoes"][key]["chi2_total"] - chi2["pp"][key]["chi2_total"]
        macro(f"shoescost{word}", f"{cost:.1f}")
        # how much of LCDM's increase the model avoids (Sec. 4.3; the text had "roughly 29" and "roughly 7" typed in
        # until 28 October 2026, when they were 30.2 and 8.0)
        lcdm = chi2["ppshoes"]["LCDM"]["chi2_total"] - chi2["pp"]["LCDM"]["chi2_total"]
        macro(f"shoesrelief{word}", f"{lcdm - cost:.0f}")


    # ---------------------------------------------------------------- profile likelihoods
    # kit logs/profiles_<data>_<label>.csv: copied from results_profiles/<study>/profiles.csv on Grendel (the study's
    # columns: the emulator's profile, the true value at its optimum, the true value optimised by the residual
    # Newton stage). The faces Delta Neff = 0 and log10 Geff = -5.5 are also reached by the CLASS++ values at the
    # emulator best fits of SInu and DRMD (above): a profile node there is at most that, and is replaced by it
    # where it is higher (the node's refinement started from a false maximum of the emulator and stayed there).
    prof_files = {d: KIT / "logs" / f"profiles_{d}_{label}.csv" for d in DATA}
    if all(f.exists() for f in prof_files.values()):
        import csv
        profs = {}
        star_gaps, mean_kinds, rest_max = [], [], []
        for d, tag in (("pp", "PP"), ("ppshoes", "SH")):
            nodes = {}                      # (parameter, coordinate) -> the best true value found for that node
            def offer(name, x, true_logl, emu_dchi2, reason, direct_logl):
                # the emulator's value and the true value at its optimum come from the first study of the node
                key = (name, round(x, 6))
                first = nodes.get(key)
                if first is None or true_logl > first["true"]:
                    nodes[key] = dict(x=x, true=true_logl, reason=reason,
                                      s=emu_dchi2 if first is None else first["s"],
                                      direct=direct_logl if first is None else first["direct"])
            below = cut_nodes(d, post[d]["names"])
            recomputed = cut_sources(d)
            main_smax = max(float(r["surrogate_logl"]) for r in csv.DictReader(open(KIT / PROFILE_SOURCES[d][0][1])))
            for source in PROFILE_SOURCES[d] + recomputed:
                for item in read_profile_source(source, main_smax if source in recomputed else None):
                    if source in recomputed:
                        offer(item[0], item[1], item[2] - CUT_PRIOR_SHIFT, item[3], item[4], item[5] - CUT_PRIOR_SHIFT)
                    elif (item[0], round(item[1], 6)) not in below:
                        offer(*item)
            # moved points (pp_polish_setup.py, both passes, evaluated with the cut copy's Cobaya file): any true value at a node's
            # coordinate bounds the profile there; it can only improve a node that exists
            for f in sorted(KIT.glob(f"logs/chi2_points_{d}_profiles_dncut_polish*_*.json")):
                for e in json.loads(f.read_text())["results"]:
                    if "logpost" not in e or not np.isfinite(e["logpost"]) or "profile" not in e:
                        continue
                    if (e["profile"], round(e["coordinate"], 6)) in nodes:
                        offer(e["profile"], e["coordinate"], e["logpost"] - CUT_PRIOR_SHIFT, None, "moved", None)
            missing = sorted(k for k in below if k not in nodes)
            macro(f"{tag}cutnodes", str(len(below)))
            macro(f"{tag}cutnodesmissing", str(len(missing)))
            tmax = max([n["true"] for n in nodes.values()] + [e["logpost"] for e in chi2[d].values()])
            # the global best fit (the stars) against this reference, and the posterior mean: its true value from the kit's
            # job on smoke/chi2_points_posterior_mean.json (results/20261023_posterior_mean/make_points.py), else the
            # emulator's value there (in the points file)
            macro(f"{tag}stardchi", f"{2 * (tmax - stars[d]['logl']):.2f}")
            star_gaps.append(2 * (tmax - stars[d]["logl"]))
            # red notes in the text while a star is not the best true point or the mean's value is the emulator's
            macro("starnote", "" if max(star_gaps) < 0.01 else
                  r" {\color{red}\small [Stars: not yet the best profile nodes, which are on Grendel (results\_profiles/"
                  r"\ldots/true/node\_*.json); "
                  + ", ".join(f"{DATA[k]} {g:.2f}" for k, g in zip(DATA, star_gaps)) + r" below them in $\chi^2$.]}")
            pm_file = chi2_file(d, post[d]["run"].name, labels=("posterior_mean",), required=False)
            # with the cut at Delta N_eff = CUT the Pantheon+ mean moved: its own job (posterior_mean_cut_points.py)
            pm_cut = sorted(KIT.glob(f"logs/chi2_points_{d}_posterior_mean_cut_*.json"))
            if pm_cut:
                pm = next(e for e in json.loads(pm_cut[-1].read_text())["results"] if e["label"] == "posterior_mean")
                pm_logl, pm_kind = pm["logpost"], "true"
            elif post[d]["cut"]["share"] > 1e-3:      # Pantheon+SH0ES loses 1.6e-5 of its weight: its mean is unchanged
                pm_logl, pm_kind = None, "pending"
            elif pm_file is not None:
                pm = next(e for e in json.loads(pm_file.read_text())["results"] if e["label"] == "posterior_mean")
                pm_logl, pm_kind = pm["logpost"], "true"
            else:
                pts = json.loads((KIT / "smoke" / "chi2_points_posterior_mean.json").read_text())["data"][d]
                if pts["network"] != post[d]["run"].name:
                    raise SystemExit(f"smoke/chi2_points_posterior_mean.json is not of {post[d]['run'].name}")
                pm_logl, pm_kind = next(q for q in pts["points"] if q["label"] == "posterior_mean")["network_logpost"], "emulated"
            macro(f"{tag}meandchi", f"{2 * (tmax - pm_logl):.0f}" if pm_logl is not None else "(pending)")
            # where the mean's chi^2 comes from: likelihood by likelihood against the full model's best fit (the model
            # comparison's point, within a few tenths of the global best; the profile nodes keep no per-likelihood values)
            if pm_kind == "true":
                full = chi2[d]["full"]["chi2"]
                diff = {k: pm["chi2"][k] - full[k] for k in full}
                cam, lens = "planck_NPIPE_highl_CamSpec.TTTEEE", "planckpr4lensing.PlanckPR4Lensing"
                macro(f"{tag}meancamspec", f"{diff[cam]:.0f}")
                macro(f"{tag}meanlensing", f"{diff[lens]:.0f}")
                macro(f"{tag}meanfulldchi", f"{sum(diff.values()):.0f}")
                rest_max.append(max(abs(v) for k, v in diff.items() if k not in (cam, lens)))
                macro("meanrest", f"{max(rest_max):.1f}")
            elif pm_kind == "pending":
                for key in ("meancamspec", "meanlensing", "meanfulldchi"):
                    macro(f"{tag}{key}", "(pending)")
            macro("meankind", pm_kind)
            mean_kinds.append(pm_kind)
            macro("meannote", "" if all(k == "true" for k in mean_kinds) else
                  r" {\color{red}\small [The values calculated with the true likelihood at the \PDP{} mean with the cut at $\Delta N_\mathrm{eff} = 0.01$ "
                  r"follow from the job on smoke/chi2\_points\_posterior\_mean\_cut.json.]}" if "pending" in mean_kinds else
                  r" {\color{red}\small [The emulator's value at the mean; the value calculated with the true likelihood follows from the job on "
                  r"smoke/chi2\_points\_posterior\_mean.json.]}")
            mean = (post[d]["x"] * post[d]["w"][:, None]).sum(axis=0)
            for n, short_name, fmt in (("logGeff", "Geff", ".1f"), ("delta_Neff_drmd", "DNeff", ".2f"), ("f_idm_drmd", "fidm", ".3f")):
                macro(f"{tag}mean{short_name}", format(mean[post[d]["names"].index(n)], fmt))
                macro(f"{tag}star{short_name}", format(stars[d]["values"][n], fmt))
            P = {}
            for (name, _), n in nodes.items():
                P.setdefault(name, []).append(dict(x=n["x"], t=2 * (tmax - n["true"]), s=n["s"], reason=n["reason"],
                                                   d=2 * (tmax - n["direct"]), replaced=False))
            for name in P:
                P[name].sort(key=lambda r: r["x"])
            face = {"delta_Neff_drmd": (0.0, 2 * (tmax - chi2[d]["SInu_strong"]["logpost"])),
                    "logGeff": (-5.5, 2 * (tmax - chi2[d]["DRMD"]["logpost"]))}
            extra = {}
            for name, (edge, bound) in face.items():
                node = P[name][0]
                if abs(node["x"] - edge) < 1e-6:
                    macro(f"{tag}profnode{'DN' if name == 'delta_Neff_drmd' else 'Geff'}", f"{node['t']:.1f}")
                    if node["t"] > bound:
                        node["t"], node["replaced"] = bound, True
                else:
                    extra[name] = (edge, bound)          # outside the profile's range: shown as a point of its own
                macro(f"{tag}prof{'DNzero' if name == 'delta_Neff_drmd' else 'weak'}",
                      f"{min(bound, node['t']) if abs(node['x'] - edge) < 1e-6 else bound:.1f}")
            fz = P["f_idm_drmd"][0]
            macro(f"{tag}proffidmzero", f"{fz['t']:.1f}")
            far = [r for r in P["f_idm_drmd"] if r["x"] >= 0.15]
            if far:
                macro(f"{tag}proffidmfar", f"{min(r['t'] for r in far):.0f}--{max(r['t'] for r in far):.0f}")
                macro(f"{tag}proffidmtenth", f"{min(P['f_idm_drmd'], key=lambda r: abs(r['x'] - 0.1))['t']:.0f}")
            macro(f"{tag}profnodes", str(sum(len(v) for v in P.values())))
            macro(f"{tag}profcertified", str(sum(r["reason"] == "converged" for v in P.values() for r in v)))
            # emulator against truth near the minima: the offset of its curve and the scatter about it
            diffs = np.array([r["s"] - r["t"] for name in P for r in P[name] if r["t"] < 4 and not r["replaced"]])
            off = float(np.median(diffs))
            macro(f"{tag}profoffset", f"{off:.1f}")
            macro(f"{tag}profagree", f"{np.sqrt(np.mean((diffs - off) ** 2)):.2f}")
            # the study's own summary, from its job log
            log = next((f for f in sorted(KIT.glob("logs/profiles_*.out"))
                        if f"sidr_drmd2_{d}_combined_{label}_cosmo" in f.read_text(errors="replace")
                        and "COMPLETE" in f.read_text(errors="replace")), None)
            if log is not None:
                text = log.read_text(errors="replace").replace("\r", "\n")
                calls = re.search(r"true calls: (\d+) of the budget", text)
                ended = re.search(r"ended as (.*)", text)
                gain = re.search(r"\(log-objective\): median ([0-9.e+-]+), 90% under ([0-9.e+-]+), most ([0-9.e+-]+)", text)
                macro(f"{tag}profcalls", f"{int(calls.group(1)):,}".replace(",", r"\,"))
                status = {k: int(v) for k, v in re.findall(r"(converged|stalled|budget) (\d+)", ended.group(1))}
                macro(f"{tag}profconverged", str(status.get("converged", 0)))
                macro(f"{tag}profstalled", str(status.get("stalled", 0)))
                macro(f"{tag}profbudget", str(status.get("budget", 0)))
                macro(f"{tag}profgainmed", f"{float(gain.group(1)):.2f}")
                macro(f"{tag}profgainninety", f"{float(gain.group(2)):.2f}")
                macro(f"{tag}profgainmax", f"{float(gain.group(3)):.1f}")
            profs[d] = dict(P=P, extra=extra, tmax=tmax)

        def interval(rows):
            x = np.array([r["x"] for r in rows]); y = np.array([r["t"] for r in rows])
            i0 = int(np.argmin(y)); lo = hi = None
            for i in range(i0, 0, -1):
                if y[i - 1] >= 1:
                    lo = float(np.interp(1, [y[i], y[i - 1]], [x[i], x[i - 1]])); break
            for i in range(i0, len(x) - 1):
                if y[i + 1] >= 1:
                    hi = float(np.interp(1, [y[i], y[i + 1]], [x[i], x[i + 1]])); break
            return float(x[i0]), lo, hi

        def digits(width):
            return max(0, int(-np.floor(np.log10(abs(width)))) + 1) if width > 0 else 3

        def cell(rows):
            best, lo, hi = interval(rows)
            if lo is None and hi is None:
                return "unconstrained"
            if lo is None:
                k = digits(hi - best)
                return f"$< {hi:.{k}f}$"
            if hi is None:
                k = digits(best - lo)
                return f"$> {lo:.{k}f}$"
            k = digits(min(hi - best, best - lo))
            return f"${best:.{k}f}\\,[{lo:.{k}f},\\,{hi:.{k}f}]$"

        lines = [r"\begin{tabular}{lcc}", r"\toprule",
                 r"\textbf{Parameter} & \textbf{\PDP{}} & \textbf{\PDPS{}} \\", r"\midrule"]
        for n in COSMO:
            lines.append(f"${LABELS[n]}{UNITS.get(n, '')}$ & " + " & ".join(cell(profs[d]["P"][n]) for d in DATA) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (HERE / "tables" / "profile_intervals.tex").write_text("\n".join(lines) + "\n")

        # for the plotting scripts: the nodes of every profile, the face points outside a profile's range, and the true
        # chi^2 at the emulator's best fits on the face Delta Neff = f_idm = 0 (for the log10 Geff figure)
        write_plot_data("profiles.json", {d: dict(
            tmax=profs[d]["tmax"],
            P={n: [dict(x=r["x"], t=r["t"], s=r["s"], d=r["d"], reason=r["reason"], replaced=r["replaced"]) for r in rows]
               for n, rows in profs[d]["P"].items()},
            extra={n: list(v) for n, v in profs[d]["extra"].items()},
            face=[dict(label=k, logGeff=chi2[d][k]["params"]["logGeff"], dchi2=2 * (profs[d]["tmax"] - chi2[d][k]["logpost"]))
                  for k in ("LCDM", "SInu_weak", "SInu_strong")]) for d in DATA})

        # chi^2 of every likelihood along the profiles (Figs. profile_breakdown_*): the kit's job on
        # smoke/chi2_profile_points.json (results/20261024_chi2_compare/make_points.py: every node of these studies with the
        # runs' own set-up; its log-posterior reproduces each node's value), relative to the global best fit. A node replaced
        # by the best fit on its face takes that fit's values (the model-comparison job above). A node whose value came from
        # a study without a breakdown (the omega_cdm node of the call-limit rerun) is left out and counted.
        face_fit = {"delta_Neff_drmd": "SInu_strong", "logGeff": "DRMD"}
        breakdown = {}
        for d in DATA:
            # every job on profile nodes of this data set (the profile file's set, and the follow-up's ocdm rerun nodes)
            found = sorted(KIT.glob(f"logs/chi2_points_{d}_profiles_*.json"))
            if not found:
                continue
            by = {}
            below = cut_nodes(d, post[d]["names"])
            for f, r in ((f, r) for f in found for r in json.loads(f.read_text())["results"]):
                if "chi2" not in r:
                    continue
                if "dncut" not in f.name and ("f_idm_drmd" if r["profile"] == "f_idm_drmd_wide" else r["profile"],
                                              round(r["coordinate"], 6)) in below:
                    continue
                if "dncut" in f.name:             # evaluated with the cut copy's Cobaya prior (CUT_PRIOR_SHIFT)
                    r = dict(r, logpost=r["logpost"] - CUT_PRIOR_SHIFT, minus2_logprior=r["minus2_logprior"] + 2 * CUT_PRIOR_SHIFT)
                name = "f_idm_drmd" if r["profile"] == "f_idm_drmd_wide" else r["profile"]
                key = (name, round(r["coordinate"], 6))
                if key not in by or r["logpost"] > by[key]["logpost"]:
                    by[key] = r
            tmax = profs[d]["tmax"]
            ref = max(list(by.values()) + list(chi2[d].values()), key=lambda r: r["logpost"])
            if abs(ref["logpost"] - tmax) > 1e-3:
                raise SystemExit(f"profile breakdown {d}: no breakdown at the global best fit ({ref['logpost']} against {tmax})")
            out, skipped = {}, 0
            for n, rows in profs[d]["P"].items():
                pts = []
                for row in rows:
                    src = chi2[d][face_fit[n]] if row["replaced"] else by.get((n, round(row["x"], 6)))
                    if src is None or abs(2 * (tmax - src["logpost"]) - row["t"]) > 1e-3:
                        skipped += 1
                        continue
                    pts.append(dict(x=row["x"], t=row["t"], dprior=src["minus2_logprior"] - ref["minus2_logprior"],
                                    dchi2={k: src["chi2"][k] - ref["chi2"][k] for k in ref["chi2"]}))
                out[n] = pts
            breakdown[d] = dict(likelihoods=list(ref["chi2"]), jobs=[f.name for f in found], P=out, skipped=skipped)
        if len(breakdown) == len(DATA):
            write_plot_data("profile_breakdown.json", breakdown)
            macro("profbreakdownskipped", str(sum(b["skipped"] for b in breakdown.values())))
            # the numbers the text quotes: one likelihood's Delta chi^2 at the low or high end of a profile
            cam, tt, desi = "planck_NPIPE_highl_CamSpec.TTTEEE", "planck_2018_lowl.TT", "bao.desi_dr2.desi_bao_all"

            def at(d, n, end, key):
                rows = sorted(breakdown[d]["P"][n], key=lambda r: r["x"])
                r = rows[0] if end == "low" else rows[-1]
                return r["x"], r["dchi2"][next(k for k in r["dchi2"] if k.startswith(key))]
            for d, tag in (("pp", "PP"), ("ppshoes", "SH")):
                macro(f"{tag}bdGeffcam", f"{at(d, 'logGeff', 'low', cam)[1]:+.1f}")
                macro(f"{tag}bdGeffTT", f"{at(d, 'logGeff', 'low', tt)[1]:+.1f}".replace("-", "$-$"))
                macro(f"{tag}bdDNhighcam", f"{at(d, 'delta_Neff_drmd', 'high', cam)[1]:+.1f}")
                macro(f"{tag}bdDNhighx", f"{at(d, 'delta_Neff_drmd', 'high', cam)[0]:.1f}")
            macro("PPbdHlowDESI", f"{at('pp', 'H0', 'low', desi)[1]:+.1f}")
            macro("PPbdHlowx", f"{at('pp', 'H0', 'low', desi)[0]:.1f}")
            macro("SHbdHlowSN", f"{at('ppshoes', 'H0', 'low', 'sn.')[1]:+.1f}")
            macro("SHbdHlowx", f"{at('ppshoes', 'H0', 'low', 'sn.')[0]:.1f}")
            macro("SHbdDNlowSN", f"{at('ppshoes', 'delta_Neff_drmd', 'low', 'sn.')[1]:+.1f}")
            macro("SHbdDNlowx", f"{at('ppshoes', 'delta_Neff_drmd', 'low', 'sn.')[0]:.2f}")
            macro("SHbdfidmcam", f"{at('ppshoes', 'f_idm_drmd', 'low', cam)[1]:+.1f}")
            macro("SHbdfidmSN", f"{at('ppshoes', 'f_idm_drmd', 'low', 'sn.')[1]:+.1f}")
        for d, tag in (("pp", "PP"), ("ppshoes", "SH")):
            far = [r for r in profs[d]["P"]["f_idm_drmd"] if r["x"] >= 0.15]
            macro(f"{tag}profcornermin", f"{min(r['t'] for r in far):.1f}" if far else "--")

    # ---------------------------------------------------------------- dark acoustic oscillations (appendix)
    # The points where the unpatched CLASS++ gave NaN spectra (the first set's Pantheon+ run, 41 refused points; those
    # inside the second set's prior are counted). The numbers are those of the report of 21 October 2026
    # (../20261021_report_dao_nonlinear: numbers.tex, by its build.py). The figure of these points over the posteriors and
    # the noise figure were left out of the paper on 27 October 2026, and their data are no longer written (28 October).
    dao_dir = REPO / "results" / "20261021_report_dao_nonlinear"
    tp = np.load(dao_dir / "data" / "training_points.npz", allow_pickle=True)
    bad = tp["refused"] & tp["inbox"]
    macro("daobadall", str(int(tp["refused"].sum())))
    macro("daobadinbox", str(int(bad.sum())))
    report_numbers = dict(re.findall(r"\\newcommand\{\\N(\w+)\}\{([^}]*)\}", (dao_dir / "numbers.tex").read_text()))
    for key in ("TrainingPoints", "CornerPPMin", "CornerPPMax", "BestPPLin", "CornerLinMin", "CornerLinMax", "BestLin",
                "PatchedPPMin", "PatchedPPMax", "BestPP", "LnAkSeven", "LnAkHundred"):
        macro("dao" + key, report_numbers[key])

    # ---------------------------------------------------------------- priors
    import yaml
    cob = yaml.safe_load((KIT / "runs" / f"sidr_drmd2_pp_combined_{label}" / "cobaya.yaml").read_text())["params"]
    lines = [r"\begin{tabular}{lc}", r"\toprule", r"\textbf{Parameter} & \textbf{Prior range} \\", r"\midrule"]
    for n in COSMO:
        p = cob[n]["prior"]
        lines.append(f"${LABELS[n]}{UNITS.get(n, '')}$ & $[{p['min']:g},\\ {p['max']:g}]$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "tables" / "priors.tex").write_text("\n".join(lines) + "\n")

    (HERE / "numbers.tex").write_text("% made by build.py --label " + label + "; do not edit\n"
                                      + "".join(f"\\newcommand{{\\num{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))
    (HERE / "build_summary.json").write_text(json.dumps(dict(label=label, macros=macros,
                                                             runs={d: str(post[d]["run"].name) for d in DATA}), indent=1))
    # the figures: every one is made by a script in plots/ from plots/data alone (python plots/make_all.py)
    subprocess.run([sys.executable, "make_all.py"], cwd=HERE / "plots", check=True)
    print(f"label {label}: {len(macros)} numbers; tables, plot data and figures written")
    for k, v in sorted(macros.items()):
        print(f"  \\num{k} = {v}")


if __name__ == "__main__":
    main()
