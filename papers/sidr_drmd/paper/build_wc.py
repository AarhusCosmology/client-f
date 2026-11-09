"""Figures, tables and in-text numbers of Sec. 5 of the CLiENT version of the SIDR + DRMD paper: the runs on the data of
the H0 World Cup (25 October 2026).

    python build_wc.py

The counterpart of build.py (Sec. 4, the second set of runs) for the kit's runs sidr_drmd_<data>_{low,high}, data sets
wc (the World Cup baseline) and wcmb (with SH0ES as the Gaussian prior on M_B), from the kit's synced results:

  posteriors   the combined emulator after the most boosts there is a pooled BEST posterior for (boost10, else boost5,
               else boost1; analysis/best_posterior_it_0.npz), with derived parameters from each part's derived emulator
  accuracy     the true checks of the combined emulators and each part's error at the points of its next boost
  best fits    the AIC points of the kit's analysis/submit_aic_points.sh (logs/chi2_points_<data>_bestfits_<job>.json,
               of the newest combined network that has them), true chi^2 per likelihood; in wcmb the SH0ES prior on
               M_B is counted as data: its chi^2, ((M_B + 19.253) / 0.027)^2, is added to the likelihoods'
  profiles     the boost10 profile studies (_cosmo, _fidm_wide, _logGeff_wide), from the kit's logs/ (copied there by
               analysis/submit_profile_breakdown.sh, or by hand from a study still running); what is there is used
  breakdown    logs/chi2_points_<data>_profiles_<job>.json, every profile node split by likelihood

Writes tables/*_wc.tex, numbers_wc.tex (macros \\numWC..., \\numMB...), plots/data/wc/ and, with plots/make_all.py
--series wc, Figures/*_wc.pdf, *_wcmb.pdf. Whatever is not there yet is left out and listed in the macro \\numWCpending,
which the text shows in red. build.py and the Sec. 4 files are not touched.
"""
import os
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import build as B

HERE, KIT = B.HERE, B.KIT
PLOT_DATA = HERE / "plots" / "data" / "wc"
DATA = {"wc": r"\HWC{}", "wcmb": r"\HWCS{}"}   # the paper's macros: the H0 World Cup data, without and with SH0ES
TAGS = {"wc": "WC", "wcmb": "MB"}
OFFSET = B.OFFSET
COSMO, LABELS, UNITS, STRONG = B.COSMO, B.LABELS, B.UNITS, B.STRONG
MB = (-19.253, 0.027)                   # the SH0ES prior on M_B (data in wcmb)
MB_NORM = np.log(2 * np.pi * MB[1] ** 2)
# the likelihoods in the tables, in this order; "sh0es_mb" is the M_B prior of wcmb
LIKELIHOODS = [("planck_2018_lowl.TT", r"low-$\ell$ TT"), ("planck_2018_lowl.EE_sroll2", r"low-$\ell$ EE"),
               ("planck_NPIPE_highl_CamSpec.TTTEEE", "CamSpec"), ("act_dr6_cmbonly", "ACT"),
               ("candl.interface.CandlCobayaLikelihood", "SPT"), ("act_dr6_lenslike.ACTDR6LensLike", r"ACT+PR4 $\phi\phi$"),
               ("muse3glike.cobaya.spt3g_2yr_delensed_ee_optimal_pp_muse", r"SPT $\phi\phi$"),
               ("bao.desi_dr2.desi_bao_all", "DESI"), ("sn.pantheonplus", "Pantheon+"), ("sh0es_mb", "SH0ES")]
# the four models of the AIC table (the AIC points of make_points.py --series wc: SInu_best is the best fit on the SInu face)
MODELS = [("full", r"SI$\nu$+DRMD", 12), ("DRMD", "DRMD", 11), ("SInu_best", r"SI$\nu$", 8), ("LCDM", r"$\Lambda$CDM", 7)]
# the boundary of the strong mode of each data set: the minimum of its posterior of log10 Geff between the modes
# (build.mode_minimum; 28 October 2026, Steen), set by main() from the posteriors before the best fits are read. The
# AIC point "strong" (the kit's analysis/aic/make_points.py) was chosen above -2.0, the boundary of before; a candidate
# for the strong mode's best fit (Table 9) must lie above the data set's boundary. A caller of chi2_points outside main
# (wc_polish_setup.py) keeps the AIC job's -2.0.
STRONG_MIN = {}
# best fits of the final emulators at maxima the true likelihood does not have, which the paper names (Sec. 5.4: the DRMD
# fit without SH0ES, 7.6 too high in ln L); left out of the agreement quoted for the other fits (finalfiterr)
FALSE_MAXIMA = {"wc": ("DRMD",)}
STUDIES = ("cosmo", "fidm_wide", "logGeff_wide")
# The paper leaves out 0 < Delta N_eff < 0.01 (26 October 2026, Steen; build.py CUT): there the interacting dark matter,
# tightly coupled to a negligible amount of dark radiation, has a small sound speed and suppresses the small-scale power,
# and the likelihood rests on non-linear corrections that are numerically unreliable. Without SH0ES the highest true
# values found lie there (Delta N_eff ~ 3e-4). Posteriors, stars and best fits leave the region out; the macros
# <tag>valley* (valley_macros) still describe its best point for the appendix (App. DAO quotes where it is), and
# <tag>valleyshare* the share of the posterior removed. The profiles in the region (study _dnvalley_cosmo) are not used.
# The cut applies to both data sets (with SH0ES it removes 8e-6 of the posterior; Sec. 3.2 of the paper).
VALLEY = {"wc": 0.01, "wcmb": 0.01}
LENSING = ("act_dr6_lenslike.ACTDR6LensLike", "muse3glike.cobaya.spt3g_2yr_delensed_ee_optimal_pp_muse")
SMALL_SCALES = ("act_dr6_cmbonly", "candl.interface.CandlCobayaLikelihood")
# profiles not resolved near their minimum: none since the profiles with the cut (28 October 2026). Before, without SH0ES
# the nodes of tau_reio and ln(10^10 A_s) next to the minimum lay several units too high on both branches, and the table
# gave no interval for them.
UNRESOLVED = {}
# Without SH0ES the paper's profiles are those with the cut (28 October 2026): the studies on the run's copy with the prior
# Delta N_eff >= 0.01 (sidr_drmd_wc_combined_boost10_dncut_{cosmo,fidm_wide,logGeff_wide}), whose refinement mostly ended
# at its limit of calls, improved by the points moved to their nodes (wc_polish_setup.py, four passes) and by the chains
# of true-likelihood optimisations node by node from the best fit and from the second minimum at the cut (wc_chain_setup.py,
# the kit's smoke/profile_chain.py); at every node the best true value. The first profiles (before the cut) remain the
# source of the valley's numbers (valley_macros).
CUT_PROFILES = {"wc"}


def latest(pattern):
    return B.latest(pattern)


def write_plot_data(name, value):
    PLOT_DATA.mkdir(parents=True, exist_ok=True)
    B.PLOT_DATA, saved = PLOT_DATA, B.PLOT_DATA
    try:
        B.write_plot_data(name, value)
    finally:
        B.PLOT_DATA = saved


def newest_label(d, path):
    """The label with the most boosts whose combined run has `path` (relative to the run)."""
    for label in sorted(OFFSET, key=OFFSET.get, reverse=True):
        found = sorted(KIT.glob(f"results/*_sidr_drmd_{d}_combined_{label}"))
        if found and (found[-1] / path).exists():
            return label
    raise SystemExit(f"no combined run of {d} with {path}")


def derived_values(x, names, d):
    """The derived parameters at x from each part's latest derived emulator, by delta_Neff (as build.py)."""
    import tensorflow as tf
    import derived.emulator  # noqa: F401

    out, dnames = None, None
    low = x[:, names.index("delta_Neff_drmd")] < 0.10
    for side, mask in (("low", low), ("high", ~low)):
        part = latest(f"results/*_sidr_drmd_{d}_{side}")
        model_file = max(part.glob("derived_data/derived_model_it_*.keras"), key=B.number_of)
        these = [p["name"] for p in json.loads((part / "derived_data" / "derived_parameters.json").read_text())["parameters"]]
        if dnames is None:
            dnames, out = these, np.full((len(x), len(these)), np.nan)
        elif these != dnames:
            raise SystemExit(f"{part}: derived parameters {these}, not {dnames}")
        if mask.any():
            model = tf.keras.models.load_model(model_file, compile=False)
            out[mask] = model.predict(x[mask].astype(np.float32), batch_size=65536, verbose=0)
    return out, dnames


def posterior(d, label):
    from getdist import MCSamples

    run = latest(f"results/*_sidr_drmd_{d}_combined_{label}")
    z = np.load(run / "analysis" / "best_posterior_it_0.npz", allow_pickle=True)
    names = [str(n) for n in z["names"]]
    x, w = z["x"].astype(np.float64), z["weights"].astype(np.float64)
    keep = w > w.max() * 1e-10
    x, w, runs = x[keep], w[keep], z["run"][keep]
    box = json.loads((KIT / "runs" / f"sidr_drmd_{d}_combined_{label}" / "domain_bounds.json").read_text())
    box = box.get("bounds", box)
    valley = None
    if d in VALLEY:
        # the paper's posterior of d is that with Delta N_eff >= VALLEY[d] (26 October 2026): the samples in the valley
        # are set aside (their share, run by run, and their own constraints are reported) and the box starts at the cut
        dn = x[:, names.index("delta_Neff_drmd")]
        out = dn < VALLEY[d]
        ids = np.unique(runs)
        valley = dict(share=w[out].sum() / w.sum(), per_run=[w[out & (runs == r)].sum() / w[runs == r].sum() for r in ids],
                      x=x[out], w=w[out] / w[out].sum(), runs=runs[out])
        x, w, runs = x[~out], w[~out], runs[~out]
        box = dict(box, delta_Neff_drmd=[VALLEY[d], box["delta_Neff_drmd"][1]])
    dv, dnames = derived_values(x, names, d)
    allnames = names + dnames
    samples = MCSamples(samples=np.column_stack([x, dv]), weights=w, names=allnames,
                        labels=[LABELS.get(n, n) for n in allnames],
                        ranges={n: tuple(box[n]) for n in names if n in box}, label=DATA[d],
                        settings=dict(smooth_scale_1D=0.3, smooth_scale_2D=0.4))
    summary = json.loads((run / "analysis" / "best_posterior_it_0.json").read_text())
    # the shares, recomputed from the (cut) posterior: pooled, and the standard error from the scatter of the runs. The
    # strong mode lies above the minimum between the modes (build.mode_minimum)
    ids = np.unique(runs)
    splits = [("strong", "logGeff", B.mode_minimum(x[:, names.index("logGeff")], w))]
    if valley is not None:
        splits.append(("delta_Neff_drmd>0.1", "delta_Neff_drmd", 0.1))
    for key, column, edge in splits:
        above = x[:, names.index(column)] > edge
        per_run = [w[above & (runs == r)].sum() / w[runs == r].sum() for r in ids]
        sd = float(np.std(per_run, ddof=1))
        summary["splits"][key] = dict(best=float(w[above].sum() / w.sum()), seed_sd=sd, error=sd / np.sqrt(len(ids)),
                                      edge=edge)
    return dict(run=run, label=label, samples=samples, x=x, w=w / w.sum(), names=names, summary=summary, box=box,
                runs=runs, valley=valley)


def chi2_points(d, kind, merge=False, profiles=True):
    """The AIC job outputs of data set d (kind: bestfits or posterior_mean), each result with chi2["sh0es_mb"] (wcmb)
    and chi2_data, the likelihoods' sum plus SH0ES, and the label of the network it was fitted on (fit_label).
    Without merge: the newest network's job. With merge: every job, and for each point the one of highest true value,
    the best fit found of that model (25 October: the S+10 network's DRMD optimum of the World Cup data was a false
    maximum, 7.6 too high, whose true chi^2 was 15 worse than the S+5 network's DRMD fit). profiles=False leaves out
    the points of the profiles (profile_points), for the emulators' own fits."""
    jobs = []
    for f in sorted(KIT.glob(f"logs/chi2_points_{d}_{kind}_*.json"), key=lambda f: int(f.stem.rsplit("_", 1)[1])):
        r = json.loads(f.read_text())
        label = re.search(r"_combined_(boost\d+)$", r.get("network", ""))
        if label:
            jobs.append((label.group(1), r, f.name))
    if not jobs:
        return None
    jobs.sort(key=lambda j: OFFSET[j[0]])
    out = {}
    for label, r, name in (jobs if merge else jobs[-1:]):
        # the points themselves (their parameters), as the AIC job kept them beside its fits
        spec = json.loads((KIT / "logs" / f"aic_wc_{label}" / f"chi2_points_wc_{label}_{kind}.json").read_text())
        params = {q["label"]: q["params"] for q in spec["sets"][f"{d}_{kind}"]["points"]}
        for e in r["results"]:
            if "chi2" not in e:
                continue
            if merge and in_valley(d, params[e["label"]]):
                continue                          # below the cut at Delta N_eff = 0.01 (26 October 2026)
            if merge and e["label"] == "strong" and params[e["label"]]["logGeff"] <= STRONG_MIN.get(d, -2.0):
                continue                          # not in the strong mode (STRONG_MIN)
            e = dict(e, chi2=dict(e["chi2"]), params=params[e["label"]], fit_label=label, file=name)
            mb = e["minus2_logprior_gaussian"].get("Mb")
            if mb is not None:
                e["chi2"]["sh0es_mb"] = mb - MB_NORM
            e["chi2_data"] = sum(e["chi2"].values())
            if e["label"] not in out or e["logpost"] > out[e["label"]]["logpost"]:
                out[e["label"]] = e
    if merge and kind == "bestfits":
        # LCDM, DRMD and the combined model minimised with the true likelihood alone (table_minimisations): every
        # start's best point outside the cut is a candidate (26 October 2026)
        for prefix, model in (("lcdm", "LCDM"), ("drmd", "DRMD"), ("full", "full")):
            for f, start, b in table_minimisations(d, prefix):
                e = dict(label=model, chi2=dict(b["chi2"]), params=dict(b["params"]), logpost=b["logpost"],
                         minus2_logprior=b["minus2_logprior"], minus2_logprior_gaussian=b["minus2_logprior_gaussian"],
                         fit_label="direct", file=f"{f.name} ({start})")
                mb = e["minus2_logprior_gaussian"].get("Mb")
                if mb is not None:
                    e["chi2"]["sh0es_mb"] = mb - MB_NORM
                e["chi2_data"] = sum(e["chi2"].values())
                if model not in out or e["logpost"] > out[model]["logpost"]:
                    out[model] = e
        # the combined model also at every point of its profiles calculated with the true likelihood (profile_points;
        # 28 October 2026: with SH0ES the best of them, on the profile of log10 (G/H)_ini, is 0.49 above the fit of the
        # S+1 emulator in ln L, and the emulators' fits had been the table's)
        points = profile_points(d) if profiles else []
        for e in points:
            if "full" not in out or e["logpost"] > out["full"]["logpost"]:
                out["full"] = e
        # and its fit in the strongly interacting mode (strong: log10 Geff > STRONG_MIN) at every such point of its profiles
        # (28 October 2026, a review of the draft: they were 2.5 and 1.5 lower in chi^2 than the emulators' fits in that mode)
        for e in points:
            if e["params"]["logGeff"] > STRONG_MIN.get(d, -2.0) and ("strong" not in out or e["logpost"] > out["strong"]["logpost"]):
                out["strong"] = dict(e, label="strong")
        # and the nested models at those on their boundaries (build.py nested_models; 28 October 2026, Steen: SInu 0.51
        # lower in chi^2 without SH0ES, nothing else changed)
        for e in points:
            for key in ({"SInu": "SInu_best"}.get(m, m) for m in B.nested_models(e["params"])):
                if key not in out or e["logpost"] > out[key]["logpost"]:
                    out[key] = dict(e, label=key)
        # the models are nested (LCDM in DRMD and in SInu, all of them in the combined model): a nested model's better
        # fit is a fit of the larger one (26 October 2026: with the cut, the emulators' DRMD fits on the World Cup data
        # without SH0ES lie at Delta N_eff = 0, worse than LCDM's)
        for big, smaller in (("DRMD", ("LCDM",)), ("SInu_best", ("LCDM",)), ("full", ("DRMD", "SInu_best", "LCDM"))):
            for k in smaller:
                if big in out and k in out and out[k]["logpost"] > out[big]["logpost"]:
                    out[big] = dict(out[k], label=big, fit_label=f"{k} (nested)")
    label, r, name = jobs[-1]
    return dict(label=label, results=out, file=name, network=r["network"])


def fit_errors(d):
    """{network: {label: true minus emulated log-posterior}} at the emulators' own best fits of d (the AIC jobs,
    logs/chi2_points_<d>_bestfits_<job>.json; the newest job of each network)."""
    out = {}
    for f in sorted(KIT.glob(f"logs/chi2_points_{d}_bestfits_*.json"), key=lambda f: int(f.stem.rsplit("_", 1)[1])):
        r = json.loads(f.read_text())
        label = re.search(r"_combined_(boost\d+)$", r.get("network", ""))
        if label:
            out[label.group(1)] = {e["label"]: e["logpost_minus_network"] for e in r["results"]
                                   if "chi2" in e and e.get("logpost_minus_network") is not None}
    return out


def in_valley(d, params):
    """Whether a point of data set d (parameter dict, or None when unknown) lies in the valley of VALLEY."""
    return d in VALLEY and params is not None and 0 < float(params.get("delta_Neff_drmd", 1.0)) < VALLEY[d]


# A Cobaya file with the prior of Delta N_eff cut to [CUT, 2] (the kit's analysis/make_cut_run.py, smoke/cobaya_<d>_dncut.yaml)
# normalises the uniform prior over 2 - CUT instead of 2: its log-posterior is higher by this for the same likelihood. Values
# computed with it are put back on the paper's objective (26 October 2026; 0.01 in chi^2). The chi^2 are not affected.
CUT_PRIOR_SHIFT = float(np.log(2.0 / (2.0 - B.CUT)))


def table_minimisations(d, model):
    """(file, start, best point) of every minimisation with the true likelihood alone (the kit's smoke/minimise_true.py,
    logs/minimise_<model>_<d>_*_<job>.json) made with the paper's objective: the run's own Cobaya file or its copy with
    the cut, whose log-posterior is put back on the run's (CUT_PRIOR_SHIFT). Others are tests with another prior (the
    neutrino-mass floor of smoke/cobaya_wcmb_mnulow.yaml, the valley's smoke/cobaya_wc_dnvalley.yaml) and are left out."""
    own, cut = f"runs/sidr_drmd_{d}_combined_boost10/cobaya.yaml", f"smoke/cobaya_{d}_dncut.yaml"
    for f in sorted(KIT.glob(f"logs/minimise_{model}_{d}_*.json")):
        r = json.loads(f.read_text())
        if r.get("cobaya") not in (own, cut):
            continue
        for start, entry in r["starts"].items():
            b = entry.get("best")
            if not b or not np.isfinite(b.get("logpost", -np.inf)) or in_valley(d, b["params"]):
                continue
            if r["cobaya"] == cut:
                b = dict(b, logpost=b["logpost"] - CUT_PRIOR_SHIFT, minus2_logprior=b["minus2_logprior"] + 2 * CUT_PRIOR_SHIFT)
            yield f, start, b


def moved_results(d):
    """(result, parameters, file) of every plateau and polish job of d: the parameters from the job's output, else from
    the points file the job evaluated (matched by label and coordinate), else None."""
    out = []
    for f in sorted(KIT.glob(f"logs/chi2_points_{d}_plateau*_*.json")) + sorted(KIT.glob(f"logs/chi2_points_{d}_polish*_*.json")):
        r = json.loads(f.read_text())
        spec = json.loads((KIT / "smoke" / ("chi2_points_wc_polish.json" if "_polish" in r["set"]
                                            else "chi2_points_wc_plateau.json")).read_text())["sets"].get(r["set"], {})
        pts = {q["label"]: q for q in spec.get("points", [])}
        for e in r["results"]:
            q = e.get("params") or (pts[e["label"]]["params"] if e.get("label") in pts and "coordinate" in e
                                    and abs(pts[e["label"]]["coordinate"] - e["coordinate"]) < 1e-9 else None)
            out.append((e, q, f))
    return out


def node_points(d, names):
    """(profiled parameter, coordinate, parameters, true value) of every profile node of d."""
    out = []
    for study in (n for kind in STUDIES for n in study_names(d, kind)):
        folder = KIT / "logs" / f"profiles_{study}_nodes"
        if not (folder / "surrogate_profiles.json").exists():
            continue
        profiled = json.loads((folder / "surrogate_profiles.json").read_text())["names"]
        for f in sorted(folder.glob("node_*.json")):
            node = json.loads(f.read_text())
            if len(node["x"]) == len(names) and np.isfinite(node["logl"]):
                out.append((profiled[int(f.stem.split("_")[1])], float(node["coordinate"]),
                            dict(zip(names, map(float, node["x"]))), float(node["logl"])))
    return out


def profile_points(d):
    """Every point of the profiles of d outside the cut whose value was calculated with the true likelihood and split by
    likelihood, as an entry of chi2_points (label full, fit_label profile): the breakdown jobs of the profile studies
    (logs/chi2_points_<d>_profiles_*.json, parameters from the studies' nodes, matched by parameter, coordinate and
    value; logs/chi2_points_<d>_dncut_profiles_*.json), the points moved to profile nodes (moved_results and
    logs/chi2_points_<d>_dncut_polish*_*.json) and the chains (logs/profile_chain_<d>_*_<job>.json). Values computed with
    a cut Cobaya file are put back on the run's objective (CUT_PRIOR_SHIFT)."""
    names = [q["name"] if isinstance(q, dict) else q for q in json.loads(
        (latest(f"results/*_sidr_drmd_{d}_combined_boost10") / "metadata.json").read_text())["parameters"]]
    nodes = {(n, round(c, 6)): (q, l) for n, c, q, l in node_points(d, names)}
    found = []                                   # (evaluation, parameters, cut Cobaya file, where)
    for f in sorted(KIT.glob(f"logs/chi2_points_{d}_profiles_*.json")) + sorted(KIT.glob(f"logs/chi2_points_{d}_dncut_profiles_*.json")):
        r = json.loads(f.read_text())
        for e in r["results"]:
            q, l = nodes.get((e.get("profile"), round(float(e.get("coordinate", np.nan)), 6)), (None, None))
            q = e.get("params") or (q if l is not None and abs(l - e.get("logpost", np.nan)) < 1e-6 else None)
            found.append((e, q, "dncut" in r["cobaya"], f"{f.name} ({e.get('label')})"))
    found += [(e, q, False, f"{f.name} ({e.get('label')})") for e, q, f in moved_results(d)]
    for f in sorted(KIT.glob(f"logs/chi2_points_{d}_dncut_polish*_*.json")):
        r = json.loads(f.read_text())
        found += [(e, e.get("params"), "dncut" in r["cobaya"], f"{f.name} ({e.get('label')})") for e in r["results"]]
    for f in sorted(KIT.glob(f"logs/profile_chain_{d}_*_[0-9]*.json")):
        r = json.loads(f.read_text())
        found += [(node["best"], node["best"].get("params"), "dncut" in r["cobaya"], f"{f.name} ({ch['profile']} {node['coordinate']:g})")
                  for ch in r["chains"].values() for node in ch.get("nodes", []) if node.get("best") is not None]
    out = []
    for e, q, cut, where in found:
        if "chi2" not in e or q is None or not np.isfinite(e.get("logpost", -np.inf)) or in_valley(d, q):
            continue
        shift = CUT_PRIOR_SHIFT if cut else 0.0
        v = dict(label="full", chi2=dict(e["chi2"]), params=dict(q), logpost=e["logpost"] - shift,
                 minus2_logprior=e["minus2_logprior"] + 2 * shift, minus2_logprior_gaussian=e["minus2_logprior_gaussian"],
                 fit_label="profile", file=where)
        mb = v["minus2_logprior_gaussian"].get("Mb")
        if mb is not None:
            v["chi2"]["sh0es_mb"] = mb - MB_NORM
        v["chi2_data"] = sum(v["chi2"].values())
        out.append(v)
    return out


def cut_profile_nodes(d, names):
    """The profiles of d with the cut: {(parameter, rounded coordinate): dict(x, true, s, direct, reason, src, origin)} for
    every node of the cut studies, with the best true value found there on the run's objective by the study, its
    breakdown job (logs/chi2_points_<d>_dncut_profiles_*.json), the polish passes (logs/chi2_points_<d>_dncut_polish*_*.json)
    and the chains (logs/profile_chain_<d>_*_<job>.json). src is the evaluation behind the value (parameters, log-posterior,
    chi^2 per likelihood, priors), for the split by likelihood; values computed with a cut Cobaya file are put back on the
    run's objective (CUT_PRIOR_SHIFT). Also the chains' number of evaluations of the true likelihood."""
    def own(e, cut):
        e = dict(e)
        if cut:
            e["logpost"] -= CUT_PRIOR_SHIFT
            e["minus2_logprior"] = e["minus2_logprior"] + 2 * CUT_PRIOR_SHIFT
        return e
    nodes = {}
    for k in STUDIES:
        name = f"sidr_drmd_{d}_combined_boost10_dncut_{k}"
        if not (KIT / "logs" / f"profiles_{name}.csv").exists():
            continue
        for n, x, true_logl, emu, reason, direct in B.read_profile_source(("csv", f"logs/profiles_{name}.csv", name)):
            key = (n, round(x, 6))
            if key not in nodes or true_logl - CUT_PRIOR_SHIFT > nodes[key]["true"]:
                nodes[key] = dict(x=x, true=true_logl - CUT_PRIOR_SHIFT, s=emu, direct=direct - CUT_PRIOR_SHIFT, reason=reason,
                                  src=None, origin="study")
    candidates = []
    for f in sorted(KIT.glob(f"logs/chi2_points_{d}_dncut_profiles_*.json")):
        r = json.loads(f.read_text())
        candidates += [(e, "study", own(e, "dncut" in r["cobaya"])) for e in r["results"] if "chi2" in e]
    for f in sorted(KIT.glob(f"logs/chi2_points_{d}_dncut_polish*_*.json")):
        r = json.loads(f.read_text())
        candidates += [(e, "polish", own(e, "dncut" in r["cobaya"])) for e in r["results"] if "chi2" in e and "params" in e]
    evaluations = 0
    for f in sorted(KIT.glob(f"logs/profile_chain_{d}_*_[0-9]*.json")):
        r = json.loads(f.read_text())
        for ch in r["chains"].values():
            for node in ch.get("nodes", []):
                evaluations += node.get("evaluations", 0) if node.get("ended") else 0
                if node.get("best") is not None:
                    e = dict(node["best"], profile=ch["profile"], coordinate=node["coordinate"])
                    candidates.append((e, "chain", own(e, "dncut" in r["cobaya"])))
    for e, origin, v in candidates:
        key = (e["profile"], round(float(e["coordinate"]), 6))
        if key not in nodes or not np.isfinite(v["logpost"]) or in_valley(d, v.get("params")):
            continue
        n = nodes[key]
        # the study's own value is replaced by its breakdown job's, the same point recomputed, so that value and split agree
        # (a node of two studies, the wide ones share coordinates: the better of the two)
        recomputed = origin == "study" and n["origin"] == "study" and (n["src"] is None or v["logpost"] > n["src"]["logpost"])
        if v["logpost"] > n["true"] or recomputed:
            n.update(true=v["logpost"], src=v, origin=origin)
    return nodes, evaluations


def valley_macros(d, tag, star, names, moved_all, fits, P, macro):
    """The valley of d (VALLEY) at its best point, the global best fit: where it is, how it compares with the broad
    minimum outside it, and what switching the interaction off costs there, split by likelihood (profile_polish.py moved
    the point to every node, so the same point with f_idm = 0 and with the first Delta N_eff node was evaluated); and the
    broad minimum's own Delta chi^2 < 1 ranges in Delta N_eff, omega_cdm and H0 (its profile, measured from its best)."""
    x, logl, _ = star
    ps = dict(zip(names, map(float, x)))
    if not in_valley(d, ps):
        return
    nodes = node_points(d, names)
    with_params = [(e, q) for e, q, _ in moved_all if q is not None and "logpost" in e]
    broad = max([l for _, _, q, l in nodes if not in_valley(d, q)] + [e["logpost"] for e, q in with_params if not in_valley(d, q)]
                + [e["logpost"] for e in fits.values() if not in_valley(d, e.get("params"))])
    same = lambda q, skip: all(abs(q[n] - ps[n]) <= 1e-9 * max(1.0, abs(ps[n])) for n in ps if n != skip)  # noqa: E731
    own = next((e for e, q in with_params if "chi2" in e and same(q, None)), None)
    off = next((e for e, q in with_params if "chi2" in e and abs(q["f_idm_drmd"]) < 1e-12 and same(q, "f_idm_drmd")), None)
    dn = min((r["x"] for r in P.get("delta_Neff_drmd", []) if r["x"] > 0), default=None)
    step = next((e for e, q in with_params if dn is not None and abs(q["delta_Neff_drmd"] - dn) < 1e-6
                 and same(q, "delta_Neff_drmd")), None)
    e10 = int(np.floor(np.log10(ps["delta_Neff_drmd"])))
    macro(f"{tag}valleyDN", f"{ps['delta_Neff_drmd'] / 10 ** e10:.0f}\\times10^{{{e10}}}")
    macro(f"{tag}valleyfidm", f"{ps['f_idm_drmd']:.2f}")
    macro(f"{tag}valleyzstop", f"{ps['log10_zstop']:.1f}")
    macro(f"{tag}valleyGH", f"{ps['log10_G_over_aH']:.0f}")
    macro(f"{tag}valleyGeff", f"{ps['logGeff']:.1f}")
    macro(f"{tag}valleyHzero", f"{ps['H0']:.1f}")
    macro(f"{tag}valleygain", f"{2 * (logl - broad):.1f}")
    distinct = {}                        # a node of two studies (the wide ones share some coordinates): its better one
    for n, c, q, l in nodes:
        key = (n, round(c, 6))
        if key not in distinct or l > distinct[key][0]:
            distinct[key] = (l, in_valley(d, q))
    macro(f"{tag}valleynodes", str(sum(v for _, v in distinct.values())))
    # decoupling, G = aH with the model's G(z) (Sec. 2.3), and R and the sound speed of the coupled fluid there
    omega_g = 2.4728e-5
    omega_r = omega_g * (1 + 7 / 8 * (4 / 11) ** (4 / 3) * (3.044 + ps["delta_Neff_drmd"]))
    zz = np.logspace(1, 6, 20001)
    g = (10 ** ps["log10_G_over_aH"] / np.sqrt(1 + (ps["omega_b"] + ps["omega_cdm"]) / omega_r / (1 + zz))
         * np.exp(-(10 ** ps["log10_zstop"] + 1) / (1 + zz)))
    zdec = float(zz[np.argmin(np.abs(np.log(g)))])
    R = 0.75 * ps["f_idm_drmd"] * ps["omega_cdm"] / (ps["delta_Neff_drmd"] * 7 / 8 * (4 / 11) ** (4 / 3) * omega_g * (1 + zdec))
    macro(f"{tag}valleyzdec", f"{round(zdec, -2):.0f}")
    macro(f"{tag}valleyR", f"{round(R, -3):.0f}")
    macro(f"{tag}valleycs", f"{1 / np.sqrt(3 * (1 + R)):.3f}")
    if own is not None and off is not None:
        dchi = {k: off["chi2"][k] - own["chi2"][k] for k in own["chi2"]}
        macro(f"{tag}valleyoff", f"{2 * (own['logpost'] - off['logpost']):.1f}")
        macro(f"{tag}valleylens", f"{sum(dchi.get(k, 0.0) for k in LENSING):.1f}")
        macro(f"{tag}valleysmall", f"{sum(dchi.get(k, 0.0) for k in SMALL_SCALES):.1f}")
    if own is not None and step is not None:
        macro(f"{tag}valleystep", f"{2 * (own['logpost'] - step['logpost']):.0f}")
        macro(f"{tag}valleystepDN", f"{dn:.3f}")
    # the broad minimum: its best point, and its profile (best value outside the valley at each node) in three parameters
    bpts = [(n, c, l) for n, c, q, l in nodes if not in_valley(d, q)] + [
        (e.get("profile", "logGeff"), e["coordinate"], e["logpost"]) for e, q in with_params if "coordinate" in e and not in_valley(d, q)]
    for n, short, fmt in (("delta_Neff_drmd", "DN", ".2f"), ("omega_cdm", "ocdm", ".4f"), ("H0", "Hzero", ".1f")):
        by = {}
        for m, c, l in bpts:
            if m == n:
                by[round(c, 6)] = max(by.get(round(c, 6), -np.inf), l)
        xs = np.array(sorted(by)); ys = np.array([2 * (broad - by[v]) for v in xs])
        if len(xs) < 3:
            continue
        inside = np.flatnonzero(ys < 1)
        if not len(inside):
            continue
        a, b, i0 = inside[0], inside[-1], int(np.argmin(ys))
        lo = float(np.interp(1, [ys[a], ys[a - 1]], [xs[a], xs[a - 1]])) if a > 0 else None
        hi = float(np.interp(1, [ys[b], ys[b + 1]], [xs[b], xs[b + 1]])) if b < len(xs) - 1 else None
        macro(f"{tag}broad{short}", format(xs[i0], fmt))
        if lo is not None and hi is not None:
            macro(f"{tag}broad{short}range", f"[{lo:{fmt}},\\,{hi:{fmt}}]")


def study_names(d, study):
    """The studies of d of one kind in the kit's logs/: the queued name and any resubmission under a longer one (the wc
    _cosmo study was resubmitted as _cosmo_hops60 after its scan hit the hop limit, 25 October 2026)."""
    base = f"sidr_drmd_{d}_combined_boost10_{study}"
    return sorted({f.name[len("profiles_"):].removesuffix(".csv").removesuffix("_nodes")
                   for f in KIT.glob(f"logs/profiles_{base}*")})


def profile_sources(d):
    """The boost10 profile studies of d in the kit's logs/: a complete study's profiles.csv, else its node checkpoints."""
    sources = []
    for name in (n for study in STUDIES for n in study_names(d, study)):
        if (KIT / "logs" / f"profiles_{name}.csv").exists():
            sources.append(("csv", f"logs/profiles_{name}.csv", name))
        elif (KIT / "logs" / f"profiles_{name}_nodes" / "surrogate_profiles.json").exists():
            sources.append(("nodes", f"logs/profiles_{name}_nodes"))
    return sources


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    (HERE / "tables").mkdir(exist_ok=True)
    macros, pending = {}, []

    def macro(name, value):
        assert re.fullmatch(r"[A-Za-z]+", name), name
        macros[name] = value

    # ---------------------------------------------------------------- posteriors
    post = {d: posterior(d, newest_label(d, "analysis/best_posterior_it_0.npz")) for d in DATA}
    for d in DATA:
        macro(f"{TAGS[d]}label", f"S+{OFFSET[post[d]['label']]}")
        if post[d]["label"] != "boost10":
            pending.append(f"{DATA[d]}: posterior of the emulator after {OFFSET[post[d]['label']]} boosts")
    PLOT_DATA.mkdir(parents=True, exist_ok=True)
    for d in DATA:
        allnames = post[d]["samples"].getParamNames().list()
        keep = COSMO + [n for n in allnames if n not in post[d]["names"]]
        cols = [allnames.index(n) for n in keep]
        np.savez_compressed(PLOT_DATA / f"posterior_{d}.npz", samples=post[d]["samples"].samples[:, cols].astype(np.float32),
                            weights=post[d]["samples"].weights, names=np.array(keep),
                            ranges=json.dumps({n: list(post[d]["box"][n]) for n in COSMO}))
    rows = [(r"$\Lambda$CDM", ["omega_b", "omega_cdm", "H0", "logA", "n_s", "tau_reio"]),
            (r"SI$\nu$", ["m_int", "logGeff"]),
            ("DRMD", ["delta_Neff_drmd", "f_idm_drmd", "log10_zstop", "log10_G_over_aH"]),
            ("Derived", ["sigma8", "S8", "Omega_m", "rs_drag", "f_idr_drmd"])]
    # Thomas Tram's layout (28 October 2026), as tables/params.tex (build.py)
    lines = [r"\begin{tabular}{rlcc}", r"\cmidrule[\heavyrulewidth](lr){2-4}",
             r"& \textbf{Parameter} & \textbf{\HWC{}} & \textbf{\HWCS{}} \\", r"\cmidrule(lr){2-4}"]
    for title, params in rows:
        group = []
        for n in params:
            values = [B.latex_value(post[d]["samples"], n) for d in DATA]
            group.append(f"${LABELS[n]}{UNITS.get(n, '')}$ & " + " & ".join(v if v == "unconstrained" else f"${v}$" for v in values)
                         + r" \\")
        if title == r"SI$\nu$":
            cells = []
            for d in DATA:
                s = post[d]["summary"]["splits"]["strong"]
                cells.append(f"${B.fmt_share(s['best'], s['error'])}$")
            group.append(r"$P_\mathrm{strong}$ & " + " & ".join(cells) + r" \\")
        lines += [f"{title} & " + group[0]] + ["& " + row for row in group[1:]]
        lines.append(r"\cmidrule(lr){2-4}" if title != "Derived" else r"\cmidrule[\heavyrulewidth](lr){2-4}")
    lines.append(r"\end{tabular}")
    (HERE / "tables" / "params_wc.tex").write_text("\n".join(lines) + "\n")

    for d in DATA:
        tag, s, summ = TAGS[d], post[d]["samples"], post[d]["summary"]
        for n, key in (("H0", "Hzero"), ("delta_Neff_drmd", "DNeff"), ("f_idm_drmd", "fidm"), ("S8", "Seight"),
                       ("n_s", "ns"), ("m_int", "mint"), ("logGeff", "Geff"), ("sigma8", "sigmaeight"),
                       ("omega_cdm", "ocdm"), ("tau_reio", "tau")):
            macro(f"{tag}{key}", B.latex_value(s, n))
        sp = summ["splits"]["strong"]
        k = 1 if 100 * (sp["error"] or 0) < 1 else 0
        macro(f"{tag}strong", f"{100 * sp['best']:.{k}f}")
        macro(f"{tag}strongerr", f"{100 * (sp['error'] or 0):.{k}f}")
        macro(f"{tag}strongmin", f"{sp['edge']:.2f}")
        STRONG_MIN[d] = sp["edge"]
        macro(f"{tag}lowDN", f"{100 * (1 - summ['splits']['delta_Neff_drmd>0.1']['best']):.0f}")
        z = [r["lnZ"] for r in summ["runs"]]
        macro(f"{tag}seeds", str(len(z)))
        macro(f"{tag}lnZ", f"{np.mean(z):.2f}")
        macro(f"{tag}lnZscatter", f"{np.std(z, ddof=1):.2f}" if len(z) > 1 else "--")
        x, w, names = post[d]["x"], post[d]["w"], post[d]["names"]
        fi, zs = x[:, names.index("f_idm_drmd")], x[:, names.index("log10_zstop")]
        share = 100 * w[(fi > 0.15) & (zs > 4)].sum()
        macro(f"{tag}corner", f"{share:.1f}" if share >= 0.05 else "$<0.1$")
        # the posterior beyond the f_idm axis of the posterior figures (plots/common.py PLOT_LIMITS: 0.15), for the captions
        tail = 100 * w[fi > 0.15].sum() / w.sum()
        macro(f"{tag}fidmtail", f"{tail:.1f}" if tail >= 0.05 else "$<0.1$")
        # the flat ridge at Delta N_eff ~ 0 (f_idm > 0.1): the nested sampler weights it very differently from run to
        # run, so its share is given with the scatter of the runs (the error of their mean)
        zr, zw = post[d]["runs"], w
        runs = np.unique(zr)
        ridge = [zw[(fi > 0.1) & (zr == r)].sum() / zw[zr == r].sum() for r in runs]
        corner = [zw[(fi > 0.15) & (zs > 4) & (zr == r)].sum() / zw[zr == r].sum() for r in runs]
        macro(f"{tag}ridge", f"{np.mean(ridge):.2f}")
        macro(f"{tag}ridgeerr", f"{np.std(ridge, ddof=1) / np.sqrt(len(runs)):.2f}")
        macro(f"{tag}ridgemin", f"{min(ridge):.2f}")
        macro(f"{tag}ridgemax", f"{max(ridge):.2f}")
        macro(f"{tag}cornererr", f"{100 * np.std(corner, ddof=1) / np.sqrt(len(runs)):.1f}")
        v = post[d]["valley"]
        if v is not None and v["share"] > 1e-3:
            # the valley, set aside: its share of the whole posterior (pooled and run by run) and its own constraints
            macro(f"{tag}valleyshare", f"{100 * v['share']:.0f}")
            macro(f"{tag}valleysharemin", f"{100 * min(v['per_run']):.0f}")
            macro(f"{tag}valleysharemax", f"{100 * max(v['per_run']):.0f}")
            macro(f"{tag}valleyruns", str(len(v["per_run"])))
            vx, vw = v["x"], v["w"]
            for n, key, fmt in (("H0", "Hzero", ".1f"), ("omega_cdm", "ocdm", ".4f"), ("f_idm_drmd", "fidm", ".2f"),
                                ("logGeff", "Geff", ".1f")):
                col = vx[:, names.index(n)]
                mean = float((col * vw).sum())
                sd = float(np.sqrt(((col - mean) ** 2 * vw).sum()))
                macro(f"{tag}valleypost{key}", f"{mean:{fmt}} \\pm {sd:{fmt}}")

    # ---------------------------------------------------------------- best fits and model comparison
    fits = {d: chi2_points(d, "bestfits", merge=True) for d in DATA}
    for d in DATA:
        if fits[d] is None:
            raise SystemExit(f"no AIC points of {d} (logs/chi2_points_{d}_bestfits_*.json): run analysis/submit_aic_points.sh")
        macro(f"{TAGS[d]}fitlabel", f"S+{OFFSET[fits[d]['label']]}")
        if fits[d]["label"] != "boost10":
            pending.append(f"{DATA[d]}: best fits of the emulator after {OFFSET[fits[d]['label']]} boosts")
        def source(lab):
            if lab.endswith("(nested)"):
                return f"the best fit of {lab.split()[0].replace('_', ' ')}"
            if lab == "profile":
                return "a point of its profiles"
            return "direct minimisation" if lab == "direct" else f"S+{OFFSET[lab]}"
        older = sorted((k, e["fit_label"]) for k, e in fits[d]["results"].items() if e["fit_label"] != fits[d]["label"])
        macro(f"{TAGS[d]}fitolder", ", ".join(f"{k.replace('_', ' ')} ({source(lab)})" for k, lab in older) or "none")
        print(f"[fits] {d}: " + ", ".join(f"{k} from {source(e['fit_label'])} ({e['file']})" for k, e in sorted(fits[d]["results"].items())))
    chi2 = {d: fits[d]["results"] for d in DATA}
    # the calibration priors of the CMB likelihoods counted as part of the likelihood instead (Sec. 3.5; the SH0ES prior
    # on M_B is in chi2_data already): the largest change of a Delta chi^2 to LCDM in Table 6
    shift = []
    for d in DATA:
        prior = {k: -2 * chi2[d][k]["logpost"] - chi2[d][k]["chi2_data"] for k, _, _ in MODELS}
        shift += [abs(prior[k] - prior["LCDM"]) for k in prior]
    macro("WCcalibshift", f"{max(shift):.1f}")
    lines = [r"\begin{tabular}{lc|ccc|ccc}", r"\toprule",
             r" & & \multicolumn{3}{c|}{\textbf{\HWC{}}} & \multicolumn{3}{c}{\textbf{\HWCS{}}} \\",
             r"\textbf{Model} & $N$ & $\chi^2$ & $\Delta\chi^2$ & $\Delta$AIC & $\chi^2$ & $\Delta\chi^2$ & $\Delta$AIC \\",
             r"\midrule"]
    for key, name, n in MODELS:
        cells = []
        for d in DATA:
            ref, v = chi2[d]["LCDM"]["chi2_data"], chi2[d][key]["chi2_data"]
            cells += [f"{v:.2f}", "$0$" if key == "LCDM" else f"${v - ref:+.2f}$",
                      "$0$" if key == "LCDM" else f"${v - ref + 2 * (n - 7):+.2f}$"]
        lines.append(f"{name} & {n} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "tables" / "model_comparison_wc.tex").write_text("\n".join(lines) + "\n")

    # one block per dataset, the likelihoods as rows and the models as columns, in Thomas Tram's layout (the dataset in a
    # column of its own, the rules over the other columns). 28 October 2026: with the likelihoods as columns the table was
    # 14 cm wider than the text once its \resizebox was gone.
    cols = [m for m in MODELS if m[0] != "LCDM"] + [("SInu_weak", r"SI$\nu$", 8), ("strong", r"SI$\nu$+DRMD", 12)]
    kinds = ["", "", "", "(moderate)", "(strong)"]
    span = f"2-{len(cols) + 2}"
    lines = [r"\begin{tabular}{rl" + "r" * len(cols) + "}", rf"\cmidrule[\heavyrulewidth](lr){{{span}}}",
             r"& \textbf{Likelihood} & " + " & ".join(rf"\textbf{{{name}}}" for _, name, _ in cols) + r" \\",
             r"& & " + " & ".join(kinds) + r" \\", rf"\cmidrule(lr){{{span}}}"]
    for d in DATA:
        used = [(k, title) for k, title in LIKELIHOODS if k in chi2[d]["LCDM"]["chi2"]]
        diff = {key: [chi2[d][key]["chi2"][k] - chi2[d]["LCDM"]["chi2"][k] for k, _ in used] for key, _, _ in cols}
        block = [f"{title} & " + " & ".join(f"${diff[key][i]:+.2f}$" for key, _, _ in cols) + r" \\"
                 for i, (_, title) in enumerate(used)]
        # the dataset's name turned upright across its block (\usepackage{multirow}): written across, the longer name made
        # the table 49 pt wider than the text
        group = rf"\multirow{{{len(block) + 1}}}{{*}}{{\rotatebox[origin=c]{{90}}{{{DATA[d]}}}}}"
        lines += [f"{group} & " + block[0]] + ["& " + row for row in block[1:]]
        lines += [rf"\cmidrule(lr){{{span}}}", "& total & " + " & ".join(f"${sum(diff[key]):+.2f}$" for key, _, _ in cols) + r" \\"]
        lines.append(rf"\cmidrule(lr){{{span}}}" if d != list(DATA)[-1] else rf"\cmidrule[\heavyrulewidth](lr){{{span}}}")
    lines.append(r"\end{tabular}")
    (HERE / "tables" / "chi2_breakdown_wc.tex").write_text("\n".join(lines) + "\n")

    for d in DATA:
        tag, c = TAGS[d], chi2[d]
        ref = c["LCDM"]["chi2_data"]
        for key, word in (("full", "full"), ("DRMD", "drmd"), ("SInu_best", "sinu"), ("SInu_weak", "weak"), ("strong", "strong")):
            dc = c[key]["chi2_data"] - ref
            macro(f"{tag}dchi{word}", f"{dc:+.1f}")
            macro(f"{tag}gain{word}", f"{-dc + 0.0:.1f}".replace("-0.0", "0.0"))
            macro(f"{tag}aic{word}", f"{dc + 2 * (dict(full=12, DRMD=11, SInu_best=8, SInu_weak=8, strong=12)[key] - 7):+.1f}")
            macro(f"{tag}{word}Geff", f"{c[key]['params']['logGeff']:.2f}" if "params" in c[key] else "--")
        macro(f"{tag}strongoverfull", f"{c['strong']['chi2_data'] - c['full']['chi2_data']:.1f}")
        # DRMD with three parameters, as the H0 World Cup counts it ((G/H)_ini fixed there; 26 October 2026)
        macro(f"{tag}aicdrmdthree", f"{c['DRMD']['chi2_data'] - ref + 2 * 3:+.1f}")
        if d == "wc":
            # the best fits inside the region the paper leaves out, 0 < Delta N_eff < CUT, for the discussion (27 October
            # 2026): minimisations with the true likelihood alone on smoke/cobaya_wc_dnvalley.yaml (Delta N_eff in
            # [0, CUT]; its log-posterior is ln(2 / CUT) higher than the run's), against the table's fits outside it
            below = {}
            for prefix, model in (("full", "full"), ("drmd", "DRMD")):
                for f in sorted(KIT.glob(f"logs/minimise_{prefix}_{d}_valley_*.json")):
                    r = json.loads(f.read_text())
                    if r.get("cobaya") != f"smoke/cobaya_{d}_dnvalley.yaml":
                        continue
                    for entry in r["starts"].values():
                        b = entry.get("best")
                        if b and np.isfinite(b.get("logpost", -np.inf)) and (model not in below or b["logpost"] > below[model]["logpost"]):
                            below[model] = b
            if len(below) == 2:
                lik = lambda b: sum(b["chi2"].values())  # noqa: E731
                fb, fa = below["full"], c["full"]
                macro(f"{tag}belowoverabove", f"{fa['chi2_data'] - lik(fb):.2f}")
                for model, word, n in (("full", "full", 12), ("DRMD", "drmd", 11)):
                    dc = lik(below[model]) - ref
                    macro(f"{tag}belowdchi{word}", f"{dc:+.1f}")
                    macro(f"{tag}belowgain{word}", f"{-dc:.1f}")
                    macro(f"{tag}belowaic{word}", f"{dc + 2 * (n - 7):+.1f}")
                for side, q in (("below", fb["params"]), ("above", fa["params"])):
                    macro(f"{tag}{side}DN", f"{q['delta_Neff_drmd']:.2f}" if q["delta_Neff_drmd"] >= 0.005 else
                          f"{q['delta_Neff_drmd'] / 10 ** np.floor(np.log10(q['delta_Neff_drmd'])):.0f}"
                          f"\\times10^{{{int(np.floor(np.log10(q['delta_Neff_drmd'])))}}}")
                    macro(f"{tag}{side}fidm", f"{q['f_idm_drmd']:.3f}")
                    macro(f"{tag}{side}zstop", f"{q['log10_zstop']:.2f}")
                    macro(f"{tag}{side}GH", f"{q['log10_G_over_aH']:.1f}")
                    macro(f"{tag}{side}Geff", f"{q['logGeff']:.2f}")
                    macro(f"{tag}{side}Hzero", f"{q['H0']:.2f}")
                    macro(f"{tag}{side}ocdm", f"{q['omega_cdm']:.4f}")
                    macro(f"{tag}{side}ns", f"{q['n_s']:.3f}")
                for k, word in (("planck_2018_lowl.TT", "TT"), ("planck_2018_lowl.EE_sroll2", "EE"),
                                ("planck_NPIPE_highl_CamSpec.TTTEEE", "cam"), ("act_dr6_cmbonly", "act"),
                                ("candl.interface.CandlCobayaLikelihood", "spt"), ("act_dr6_lenslike.ACTDR6LensLike", "actlens"),
                                ("muse3glike.cobaya.spt3g_2yr_delensed_ee_optimal_pp_muse", "sptlens"),
                                ("bao.desi_dr2.desi_bao_all", "desi"), ("sn.pantheonplus", "sn")):
                    macro(f"{tag}below{word}", f"{fb['chi2'][k] - fa['chi2'][k]:+.2f}")
        if d == "wcmb":
            # why our DRMD fits these data better than the H0 World Cup's (Sec. 5.5; the paper's drmd_gap_setup.py):
            # the neutrino-mass floor (LCDM and DRMD minimised with the true likelihood with m_int >= 0.003 eV instead
            # of 0.06, the World Cup's sum >= 0) and the non-linear correction (HMcode 2020 against halofit at the best
            # fits of the AIC table, nuisance parameters re-optimised; 26 October 2026)
            def with_mb(b):
                return sum(b["chi2"].values()) + b["minus2_logprior_gaussian"]["Mb"] - MB_NORM
            low = {}
            for prefix, model in (("lcdm", "LCDM"), ("drmd", "DRMD")):
                for f in sorted(KIT.glob(f"logs/minimise_{prefix}_{d}_mnulow_*.json")):
                    for entry in json.loads(f.read_text())["starts"].values():
                        b = entry.get("best")
                        if b and np.isfinite(b.get("logpost", -np.inf)) and (model not in low or b["logpost"] > low[model]["logpost"]):
                            low[model] = b
            if len(low) == 2:
                macro(f"{tag}mnulowlcdm", f"{c['LCDM']['chi2_data'] - with_mb(low['LCDM']):.1f}")
                macro(f"{tag}mnulowdrmd", f"{c['DRMD']['chi2_data'] - with_mb(low['DRMD']):.1f}")
                macro(f"{tag}mnulowdchidrmd", f"{with_mb(low['DRMD']) - with_mb(low['LCDM']):+.1f}")
                macro(f"{tag}mnulowfloor", f"{min(q['params']['m_int'] for q in low.values()):.3f}")
            nl = {}
            for kind in ("halofit", "hmcode2020"):
                for f in sorted(KIT.glob(f"logs/chi2_points_{d}_{kind}_*.json")):
                    res = {e["label"]: e for e in json.loads(f.read_text())["results"] if "chi2" in e}
                    if "LCDM" in res and "DRMD" in res:
                        nl[kind] = with_mb(res["DRMD"]) - with_mb(res["LCDM"])
            if len(nl) == 2:
                macro(f"{tag}hmcodeshift", f"{abs(nl['hmcode2020'] - nl['halofit']):.1f}")
        # the combination against each of its parts (Sec. 5.5 and the conclusions)
        gd, gs = c["DRMD"]["chi2_data"] - c["full"]["chi2_data"], c["SInu_best"]["chi2_data"] - c["full"]["chi2_data"]
        macro(f"{tag}fullgainoverdrmd", f"{gd:.1f}")
        macro(f"{tag}fullgainoversinu", f"{gs:.1f}")
        macro(f"{tag}aicfulloverdrmd", f"{-gd + 2 * (12 - 11):+.1f}")
        macro(f"{tag}aicfulloversinu", f"{-gs + 2 * (12 - 8):+.1f}")
        macro(f"{tag}sumparts", f"{2 * ref - c['DRMD']['chi2_data'] - c['SInu_best']['chi2_data']:.1f}")
        for key, word in (("sh0es_mb", "SH"), ("planck_NPIPE_highl_CamSpec.TTTEEE", "cam"), ("act_dr6_cmbonly", "ACT"),
                          ("muse3glike.cobaya.spt3g_2yr_delensed_ee_optimal_pp_muse", "SPTlens"),
                          ("planck_2018_lowl.EE_sroll2", "EE"), ("planck_2018_lowl.TT", "TT"),
                          ("candl.interface.CandlCobayaLikelihood", "SPT")):
            if key in c["full"]["chi2"]:
                macro(f"{tag}fvd{word}", f"{c['full']['chi2'][key] - c['DRMD']['chi2'][key]:+.1f}")
        for key, word in (("full", "full"), ("DRMD", "drmd"), ("LCDM", "lcdm")):
            q = c[key]["params"]
            macro(f"{tag}{word}fitHzero", f"{q['H0']:.1f}")
            macro(f"{tag}{word}fitocdm", f"{q['omega_cdm']:.4f}")
            macro(f"{tag}{word}fitDNeff", f"{q['delta_Neff_drmd']:.2f}")
            macro(f"{tag}{word}fitfidm", f"{q['f_idm_drmd']:.2f}")
            macro(f"{tag}{word}fitns", f"{q['n_s']:.3f}")
        # how precisely the best fits are known: the spread over the AIC jobs of the emulators after 1, 5 and 10 boosts
        jobs = []
        for f in sorted(KIT.glob(f"logs/chi2_points_{d}_bestfits_*.json")):
            res = {e["label"]: e for e in json.loads(f.read_text())["results"] if "chi2" in e}
            jobs.append({k: e["chi2_total"] + (e["minus2_logprior_gaussian"]["Mb"] - MB_NORM
                                               if "Mb" in e["minus2_logprior_gaussian"] else 0) for k, e in res.items()})
        for key, word in (("full", "full"), ("DRMD", "drmd"), ("SInu_best", "sinu")):
            v = [j[key] for j in jobs if key in j]
            macro(f"{tag}{word}spread", f"{max(v) - min(v):.1f}")
        macro(f"{tag}fitjobs", str(len(jobs)))
        # the direct minimisation of DRMD: its starts' minima and how far below the emulators' DRMD fits they lie
        direct = []
        for f, start, b in table_minimisations(d, "drmd"):
            mb = b["minus2_logprior_gaussian"].get("Mb")
            direct.append(sum(b["chi2"].values()) + (mb - MB_NORM if mb is not None else 0))
        if direct:
            emulated = min(j["DRMD"] for j in jobs if "DRMD" in j)
            words = "zero one two three four five six seven eight nine ten".split()
            macro(f"{tag}directstarts", words[len(direct)] if len(direct) < len(words) else str(len(direct)))
            macro(f"{tag}directspread", f"{max(direct) - min(direct):.2f}")
            macro(f"{tag}directgain", f"{emulated - min(direct):.1f}")
        # LCDM minimised the same way (with SH0ES, kit job 6962213; without, smoke/minimise_lcdm_wc.json): how far below
        # the emulators' LCDM fit, the reference of every Delta chi^2 of the table
        lcdm = [sum(b["chi2"].values()) + (b["minus2_logprior_gaussian"]["Mb"] - MB_NORM if "Mb" in b["minus2_logprior_gaussian"] else 0)
                for f, start, b in table_minimisations(d, "lcdm")]
        macro(f"{tag}lcdmdirect", "yes" if lcdm else "no")
        if lcdm:
            macro(f"{tag}lcdmdirectgain", f"{min(j['LCDM'] for j in jobs if 'LCDM' in j) - min(lcdm):.1f}")
        own = chi2_points(d, "bestfits", merge=True, profiles=False)["results"]
        # how much lower the combined model's best fit (a point of its profiles) is than the best of the other fits (Sec. 5.4)
        macro(f"{tag}fullprofilegain", f"{own['full']['chi2_data'] - c['full']['chi2_data']:.1f}")
        # the true against the emulated likelihood at the emulators' best fits (28 October 2026, a review of the draft: the
        # bestfiterr of before, 0.13 and 0.05, was the largest at the fits chosen for the table other than SInu's, which
        # without SH0ES was the strong mode's alone): at every best fit of the final emulator except the false maxima the
        # text names, and at the fits of the combined model and DRMD of all three emulators (Sec. 5.4, with SH0ES)
        err = fit_errors(d)
        macro(f"{tag}finalfiterr", f"{max(abs(v) for k, v in err['boost10'].items() if k not in FALSE_MAXIMA.get(d, ())):.1f}")
        if d == "wcmb":
            macro(f"{tag}fitagree", f"{max(abs(e[k]) for e in err.values() for k in ('full', 'DRMD')):.2f}")
        if d == "wcmb":
            macro(f"{tag}shoesfull", f"{c['full']['chi2']['sh0es_mb']:.1f}")
            macro(f"{tag}shoeslcdm", f"{c['LCDM']['chi2']['sh0es_mb']:.1f}")
        # where the strong mode loses against the intermediate coupling: likelihood by likelihood
        diff = {k: c["strong"]["chi2"][k] - c["full"]["chi2"][k] for k in c["full"]["chi2"]}
        for k, word in (("act_dr6_cmbonly", "act"), ("candl.interface.CandlCobayaLikelihood", "spt"),
                        ("planck_NPIPE_highl_CamSpec.TTTEEE", "cam")):
            macro(f"{tag}strong{word}", f"{diff[k]:+.1f}".replace("-0.0", "0.0").replace("+0.0", "0.0"))

    # the posterior means (the AIC job's second points file)
    for d in DATA:
        pm = chi2_points(d, "posterior_mean")
        if pm is not None and "posterior_mean" in pm["results"]:
            e = pm["results"]["posterior_mean"]
            ref = chi2[d]["full"]["chi2_data"]          # the best fit found of the full model
            macro(f"{TAGS[d]}meandchi", f"{e['chi2_data'] - ref:.0f}")
            macro(f"{TAGS[d]}meanlabel", f"S+{OFFSET[pm['label']]}")

    # ---------------------------------------------------------------- accuracy
    from plot_boost_errors import part_errors
    acc = {}
    for d in DATA:
        parts = {side: latest(f"results/*_sidr_drmd_{d}_{side}") for side in ("low", "high")}
        acc[d] = dict(parts={side: dict(zip(("start", "boosts"), part_errors(run))) for side, run in parts.items()}, checks=[])
        for lab, k in sorted(OFFSET.items(), key=lambda t: t[1]):
            found = sorted(KIT.glob(f"results/*_sidr_drmd_{d}_combined_{lab}/analysis/true_check.json"))
            if found:
                e = json.loads(found[-1].read_text())["networks"][0]
                t = e["truth_minus_network"]
                acc[d]["checks"].append(dict(k=k, rms=t["rms"], mean=t["mean"], largest=t["largest"], n=e["points"]))
    write_plot_data("accuracy.json", acc)
    B.accuracy_table()                       # one table of the true checks for Sec. 4 and Sec. 5 (build.py)
    for d in DATA:
        tag, c = TAGS[d], {e["k"]: e for e in acc[d]["checks"]}
        e = c[max(c)]
        macro(f"{tag}checklabel", f"S+{max(c)}")
        macro(f"{tag}checkrms", f"{e['rms']:.2f}")
        macro(f"{tag}checklargest", f"{e['largest']:.1f}")
        if max(c) != 10:
            pending.append(f"{DATA[d]}: true check after {max(c)} boosts")
        for side in ("low", "high"):
            b = acc[d]["parts"][side]["boosts"]
            macro(f"{tag}{side}lastboost", f"{b[-1]['rms']:.2f}")
            macro(f"{tag}{side}boosts", str(len(b)))
            if len(b) < 10:
                pending.append(f"{DATA[d]}, {side} part: {len(b)} of 10 boosts")
        macro(f"{tag}handover", " and ".join(str(acc[d]["parts"][s]["start"]) for s in ("low", "high")))
        n = sum(sum(1 for _ in open(max(latest(f"results/*_sidr_drmd_{d}_{s}").glob("training_data/data_it_*.csv"),
                                        key=B.number_of))) - 1 for s in ("low", "high"))
        macro(f"{tag}trainingpoints", f"{n / 1e3:.0f}\\,000")

    # ---------------------------------------------------------------- global best fits (stars) and profiles
    profs, stars, replaced_by, valley_best = {}, {}, {}, {}  # replaced_by[d][(name, x)]: the point whose true value replaced a node
    cut_src = {}                         # cut_src[d][(name, x)]: the evaluation behind each node of the profiles with the cut
    for d in DATA:
        tag, c = TAGS[d], chi2[d]
        nodes, finished = {}, set()          # finished: the parameters of complete studies (their profiles.csv)
        sources = profile_sources(d)
        for source in sources:
            for name, x, true_logl, emu, reason, direct in B.read_profile_source(source):
                if source[0] == "csv":
                    finished.add(name)
                key = (name, round(x, 6))
                first = nodes.get(key)
                if first is None or true_logl > first["true"]:
                    nodes[key] = dict(x=x, true=true_logl, reason=reason, s=emu if first is None else first["s"],
                                      direct=direct if first is None else first["direct"])
        # the global best fit: the best stored profile node, or the full-model AIC point where that is higher
        best = (np.array([c["full"]["params"][n] for n in post[d]["names"]]) if "params" in c["full"] else None,
                c["full"]["logpost"], "the emulator's full-model best fit, evaluated with CLASS++")
        for name in (n for study in STUDIES for n in study_names(d, study)):
            folder = KIT / "logs" / f"profiles_{name}_nodes"
            for f in sorted(folder.glob("node_*.json")):
                node = json.loads(f.read_text())
                if len(node["x"]) == len(post[d]["names"]) and node["logl"] > best[1]:
                    best = (np.array(node["x"], float), float(node["logl"]), f"profile node {folder.name}/{f.name}")
        # a point moved to a profile node (plateau_points.py, profile_polish.py) can be the best true value found; its
        # parameters are in the job's output (params) or else in the points file it came from (checked by coordinate)
        for f in sorted(KIT.glob(f"logs/chi2_points_{d}_plateau*_*.json")) + sorted(KIT.glob(f"logs/chi2_points_{d}_polish*_*.json")):
            r = json.loads(f.read_text())
            spec = json.loads((KIT / "smoke" / ("chi2_points_wc_polish.json" if "_polish" in r["set"]
                                                else "chi2_points_wc_plateau.json")).read_text())["sets"].get(r["set"], {})
            pts = {q["label"]: q for q in spec.get("points", [])}
            for e in r["results"]:
                if "logpost" not in e or e["logpost"] <= best[1]:
                    continue
                q = e.get("params") or (pts[e["label"]]["params"] if e["label"] in pts
                                        and abs(pts[e["label"]]["coordinate"] - e["coordinate"]) < 1e-9 else None)
                if q is not None:
                    best = (np.array([q[n] for n in post[d]["names"]], float), float(e["logpost"]),
                            f"{r['set']} {e['label']} ({f.name})")
        stars[d] = best
        if d in VALLEY:
            # the posterior of d is cut at Delta N_eff >= VALLEY[d]: its star is the best true value there, and the
            # global best fit (in the valley) goes to valley_macros
            valley_best[d] = best
            names_d = post[d]["names"]
            cand = [(l, np.array([q[n] for n in names_d], float), f"profile node ({n_}, {c_:.4g})")
                    for n_, c_, q, l in node_points(d, names_d) if not in_valley(d, q)]
            cand += [(e["logpost"], np.array([q[n] for n in names_d], float), f"{f.name} {e.get('label')}")
                     for e, q, f in moved_results(d) if q is not None and "logpost" in e and not in_valley(d, q)]
            cand += [(e["logpost"], np.array([e["params"][n] for n in names_d], float), f"AIC {k} ({e['fit_label']})")
                     for k, e in c.items() if "params" in e and not in_valley(d, e["params"])]
            l, xb, src = max(cand, key=lambda t: t[0])
            stars[d] = (xb, float(l), src + f", the best outside 0 < Delta N_eff < {VALLEY[d]}")
        moved = [r["logpost"] for f in sorted(KIT.glob(f"logs/chi2_points_{d}_plateau*_*.json"))
                 + sorted(KIT.glob(f"logs/chi2_points_{d}_polish*_*.json"))
                 for r in json.loads(f.read_text())["results"] if "logpost" in r]
        tmax = max([n["true"] for n in nodes.values()] + [e["logpost"] for e in c.values()] + moved)
        if not nodes:
            pending.append(f"{DATA[d]}: profile likelihoods")
            continue
        done = {s[1].split("_boost10_")[-1].replace(".csv", "") for s in sources if s[0] == "csv"}
        if not any(k.startswith("cosmo") for k in done) or any(s[0] == "nodes" for s in sources):
            pending.append(f"{DATA[d]}: profile likelihoods incomplete")
        P = {}
        for (name, _), n in nodes.items():
            P.setdefault(name, []).append(dict(x=n["x"], t=2 * (tmax - n["true"]), s=n["s"], reason=n["reason"],
                                               d=2 * (tmax - n["direct"]), replaced=False))
        for name in P:
            P[name].sort(key=lambda r: r["x"])
        face = {"delta_Neff_drmd": (0.0, 2 * (tmax - c["SInu_best"]["logpost"])),
                "logGeff": (-5.5, 2 * (tmax - c["DRMD"]["logpost"]))}
        extra, replaced_by[d] = {}, {}
        for name, (edge, bound) in face.items():
            if name in P and abs(P[name][0]["x"] - edge) < 1e-6:
                if P[name][0]["t"] > bound:
                    P[name][0]["t"], P[name][0]["replaced"] = bound, True
                    replaced_by[d][(name, round(edge, 6))] = c["SInu_best" if name == "delta_Neff_drmd" else "DRMD"]
            else:
                extra[name] = (edge, bound)
        # true values at other points with the profiled parameter at a node's value bound that node from above: the
        # DRMD best fit moved along the plateau of weak interaction (plateau_points.py) and good points of other nodes
        # moved to every node (profile_polish.py); logs/chi2_points_<d>_{plateau,polish}*_<job>.json. The best is used.
        bounds = {}                          # the best value per (parameter, node) over every such job and set
        for f in sorted(KIT.glob(f"logs/chi2_points_{d}_plateau*_*.json")) + sorted(KIT.glob(f"logs/chi2_points_{d}_polish*_*.json")):
            for r in json.loads(f.read_text())["results"]:
                if "logpost" not in r or "coordinate" not in r:
                    continue
                key = (r.get("profile", "logGeff"), round(r["coordinate"], 6))
                if key not in bounds or r["logpost"] > bounds[key]["logpost"]:
                    bounds[key] = r
        improved = 0
        for (name, x), b in bounds.items():
            for row in P.get(name, []):
                if round(row["x"], 6) == x and 2 * (tmax - b["logpost"]) < row["t"]:
                    row["t"], row["replaced"] = 2 * (tmax - b["logpost"]), True
                    replaced_by[d][(name, x)] = b
                    improved += 1
        macro(f"{tag}profimproved", str(improved))
        if d in VALLEY:
            valley_macros(d, tag, valley_best[d], post[d]["names"], moved_results(d), c, P, macro)
        if d in CUT_PROFILES:
            # the profiles with the cut replace the first ones (the valley's numbers above are the first profiles'): every
            # node of the cut studies at the best true value found there, measured from the best of these and the fits
            cut_nodes, chain_evaluations = cut_profile_nodes(d, post[d]["names"])
            tmax = max([n["true"] for n in cut_nodes.values()]
                       + [e["logpost"] for e in c.values() if not in_valley(d, e.get("params"))])
            P, extra, replaced_by[d], cut_src[d] = {}, {}, {}, {}
            for (name, key), n in cut_nodes.items():
                P.setdefault(name, []).append(dict(x=n["x"], t=2 * (tmax - n["true"]), s=n["s"], reason=n["reason"],
                                                   d=2 * (tmax - n["direct"]), replaced=False, origin=n["origin"]))
                cut_src[d][(name, key)] = n["src"]
            for name in P:
                P[name].sort(key=lambda r: r["x"])
            face = {"delta_Neff_drmd": (0.0, 2 * (tmax - c["SInu_best"]["logpost"])),
                    "logGeff": (-5.5, 2 * (tmax - c["DRMD"]["logpost"]))}
            for name, (edge, bound) in face.items():
                if name in P and abs(P[name][0]["x"] - edge) < 1e-6:
                    if P[name][0]["t"] > bound:
                        P[name][0]["t"], P[name][0]["replaced"] = bound, True
                        replaced_by[d][(name, round(edge, 6))] = c["SInu_best" if name == "delta_Neff_drmd" else "DRMD"]
                else:
                    extra[name] = (edge, bound)
            finished, done = {name for name, _ in cut_nodes}, {"cosmo"}
            macro(f"{tag}profimproved", str(sum(n["origin"] != "study" for n in cut_nodes.values())))
            macro(f"{tag}chainevals", f"{round(chain_evaluations, -3):,.0f}".replace(",", r"\,"))
            # the second minimum, at the cut in the region of dark acoustic oscillations: its best node
            branch = max((n for n in cut_nodes.values() if n["src"] is not None
                          and n["src"]["params"]["delta_Neff_drmd"] < 0.05 and n["src"]["params"]["f_idm_drmd"] > 0.15),
                         key=lambda n: n["true"])
            macro(f"{tag}branchgap", f"{2 * (tmax - branch['true']):.1f}")
            macro(f"{tag}branchfidm", f"{branch['src']['params']['f_idm_drmd']:.1f}")
            macro(f"{tag}branchHzero", f"{branch['src']['params']['H0']:.1f}")
            macro(f"{tag}branchGeff", f"{branch['src']['params']['logGeff']:.1f}")
        if not any(k.startswith("cosmo") for k in done):
            finished = set()                   # the profiles of a data set are read once all its studies are complete
        profs[d] = dict(P=P, extra=extra, tmax=tmax, finished=sorted(finished),
                        face=[dict(label=k, logGeff=c[k]["params"]["logGeff"], dchi2=2 * (tmax - c[k]["logpost"]))
                              for k in ("LCDM", "SInu_weak", "SInu_best") if "params" in c[k]])
        if "f_idm_drmd" in P:
            macro(f"{tag}proffidmzero", f"{P['f_idm_drmd'][0]['t']:.1f}")
            far = [r["t"] for r in P["f_idm_drmd"] if r["x"] >= 0.15]
            if far:
                macro(f"{tag}proffidmfar", f"{min(far):.0f}--{max(far):.0f}")
        macro(f"{tag}profDNzero", f"{face['delta_Neff_drmd'][1]:.1f}")
        if "logGeff" in P:
            plateau = [r["t"] for r in P["logGeff"] if r["x"] < -4.0]
            strong = [r["t"] for r in P["logGeff"] if STRONG_MIN[d] < r["x"] < -1.3]
            if plateau:
                macro(f"{tag}profweak", f"{np.median(plateau):.0f}")
            if strong:
                macro(f"{tag}profstrong", f"{min(strong):.1f}")
        macro(f"{tag}profnodes", str(sum(len(v) for v in P.values())))
        macro(f"{tag}profcertified", str(sum(r["reason"] == "converged" for v in P.values() for r in v)))
    if profs:
        write_plot_data("profiles.json", profs)

    def cell(rows):
        # the interval around the best fit in which the profile lies below Delta chi^2 = 1, as in Sec. 4 (build.py). From 26
        # to 28 October 2026 it was the whole range below 1, because single nodes too high cut the first profiles short;
        # with the cut profiles the regions below 1 away from the best fit are the second minimum at the cut, which the
        # caption names instead of merging it into the interval
        if not rows:
            return "--"
        x = np.array([r["x"] for r in rows]); y = np.array([r["t"] for r in rows])
        i0 = int(np.argmin(y)); lo = hi = None
        for i in range(i0, 0, -1):
            if y[i - 1] >= 1:
                lo = float(np.interp(1, [y[i], y[i - 1]], [x[i], x[i - 1]])); break
        for i in range(i0, len(x) - 1):
            if y[i + 1] >= 1:
                hi = float(np.interp(1, [y[i], y[i + 1]], [x[i], x[i + 1]])); break
        digits = lambda w: max(0, int(-np.floor(np.log10(abs(w)))) + 1) if w > 0 else 3   # noqa: E731
        best = float(x[i0])
        if lo is None and hi is None:
            return "unconstrained"
        if lo is None:
            return f"$< {hi:.{digits(hi - best)}f}$"
        if hi is None:
            return f"$> {lo:.{digits(best - lo)}f}$"
        k = digits(min(hi - best, best - lo))
        return f"${best:.{k}f}\\,[{lo:.{k}f},\\,{hi:.{k}f}]$"

    if profs:
        lines = [r"\begin{tabular}{lcc}", r"\toprule",
                 r"\textbf{Parameter} & \textbf{\HWC{}} & \textbf{\HWCS{}} \\", r"\midrule"]
        for n in COSMO:
            lines.append(f"${LABELS[n]}{UNITS.get(n, '')}$ & "
                         + " & ".join(("not resolved" if n in UNRESOLVED.get(d, ()) else cell(profs[d]["P"].get(n, [])))
                                      if d in profs and n in profs[d]["finished"] else "(in progress)" for d in DATA) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (HERE / "tables" / "profile_intervals_wc.tex").write_text("\n".join(lines) + "\n")

    # stars: the derived parameters at each global best fit
    out = {}
    for d in DATA:
        x, logl, source = stars[d]
        dv, dn = derived_values(x[None], post[d]["names"], d)
        out[d] = dict(logl=logl, source=source, values={k: float(v) for k, v in zip(post[d]["names"] + dn, list(x) + list(dv[0]))})
        print(f"[stars] {d}: {source}, true log-objective {logl:.3f}")
    write_plot_data("best_fits.json", out)

    # ---------------------------------------------------------------- the profiles split by likelihood
    breakdown = {}
    for d in profs:
        if d in CUT_PROFILES:
            # the evaluation behind every node (study, breakdown job, polish pass or chain; cut_profile_nodes)
            found = [Path(f"logs/chi2_points_{d}_dncut_profiles, _dncut_polish*, profile_chain_{d}_* (cut_profile_nodes)")]
            by = {k: v for k, v in cut_src[d].items() if v is not None}
        else:
            found = sorted(KIT.glob(f"logs/chi2_points_{d}_profiles_*.json"))
            if not found:
                pending.append(f"{DATA[d]}: profiles split by likelihood")
                continue
            by = {}
            for r in (r for f in found for r in json.loads(f.read_text())["results"]):
                if "chi2" in r:
                    key = (r["profile"], round(r["coordinate"], 6))
                    if key not in by or r["logpost"] > by[key]["logpost"]:
                        by[key] = r

        def split(r):
            chi = dict(r["chi2"])
            mb = r.get("minus2_logprior_gaussian", {}).get("Mb")
            prior = r["minus2_logprior"]
            if mb is not None:
                chi["sh0es_mb"], prior = mb - MB_NORM, prior - mb
            return chi, prior
        tmax = profs[d]["tmax"]
        ref = max(list(by.values()) + list(replaced_by[d].values()), key=lambda r: r["logpost"])
        ref_chi, ref_prior = split(ref)
        out, skipped = {}, 0
        for n, rows in profs[d]["P"].items():
            pts = []
            for row in rows:
                src = replaced_by[d].get((n, round(row["x"], 6))) if row["replaced"] else by.get((n, round(row["x"], 6)))
                if src is None or abs(2 * (tmax - src["logpost"]) - row["t"]) > 1e-3:
                    skipped += 1
                    continue
                chi, prior = split(src)
                pts.append(dict(x=row["x"], t=2 * (ref["logpost"] - src["logpost"]), dprior=prior - ref_prior,
                                dchi2={k: chi[k] - ref_chi[k] for k in ref_chi}))
            out[n] = pts
        breakdown[d] = dict(likelihoods=list(ref_chi), jobs=[f.name for f in found], P=out, skipped=skipped)
    if breakdown:
        write_plot_data("profile_breakdown.json", breakdown)
    # the numbers the text quotes: one likelihood's Delta chi^2 at the low or high end of a profile
    keys = {"SH": "sh0es_mb", "ACT": "act_dr6_cmbonly", "cam": "planck_NPIPE_highl_CamSpec.TTTEEE",
            "SPT": "candl.interface.CandlCobayaLikelihood", "EE": "planck_2018_lowl.EE_sroll2",
            "DESI": "bao.desi_dr2.desi_bao_all"}

    def at(d, n, end, key):
        rows = sorted(breakdown[d]["P"][n], key=lambda r: r["x"])
        r = rows[0] if end == "low" else rows[-1]
        return r["x"], r["dchi2"].get(keys[key], float("nan"))
    for d in breakdown:
        tag, P = TAGS[d], breakdown[d]["P"]
        if not all(n in P and P[n] for n in ("H0", "delta_Neff_drmd", "f_idm_drmd", "logGeff")):
            continue
        for n, short_name, fmt in (("H0", "H", ".1f"), ("delta_Neff_drmd", "DN", ".2f")):
            for end, key in (("low", "SH"), ("high", "ACT")):
                x, v = at(d, n, end, key)
                macro(f"{tag}bd{short_name}{end}{key}", f"{v:+.1f}")
                macro(f"{tag}bd{short_name}{end}x", format(x, fmt))
        for key in ("cam", "SH", "ACT"):
            macro(f"{tag}bdfidm{key}", f"{at(d, 'f_idm_drmd', 'low', key)[1]:+.1f}")
        for key in ("SH", "cam"):
            macro(f"{tag}bdGeff{key}", f"{at(d, 'logGeff', 'low', key)[1]:+.1f}")
        for key in ("DESI", "cam"):    # without SH0ES, BAO and CamSpec set the low end of H0 (26 October)
            macro(f"{tag}bdHlow{key}", f"{at(d, 'H0', 'low', key)[1]:+.1f}")

    macro("WCpending", "" if not pending else
          r"{\color{red}\small [Draft, " + "25 October 2026" + r": preliminary, " + "; ".join(pending) + r".]}")
    (HERE / "numbers_wc.tex").write_text("% made by build_wc.py; do not edit\n"
                                         + "".join(f"\\newcommand{{\\num{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))
    (HERE / "build_wc_summary.json").write_text(json.dumps(dict(macros=macros, pending=pending,
                                                                runs={d: post[d]["run"].name for d in DATA}), indent=1))
    subprocess.run([sys.executable, "make_all.py", "--series", "wc"], cwd=HERE / "plots", check=True)
    print(f"{len(macros)} numbers; pending: {pending}")
    for k, v in sorted(macros.items()):
        print(f"  \\num{k} = {v}")


if __name__ == "__main__":
    main()
