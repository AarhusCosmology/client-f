"""Numbers, tables and plot data of the paper (the draft main-monodromy200.tex, for omega <= 200).

    python build200.py              # a text edit: the profile nodes from the cache (seconds)
    python build200.py --profiles   # after a sync: reread the profile nodes (minutes)

Writes numbers200.tex,
tables/priors.tex, tables/zones.tex, tables/client_settings.tex, tables/client_settings_b.tex,
tables/classpp_likelihood.tex, tables/classpp_cl.tex, tables/training_{pp,wc}.tex, tables/sampler_cost.tex and
plots/data/scan_omega.json. Reads:
  - the run definitions in the kit's runs/ (client.yaml, cobaya.yaml, domain_bounds.json): the low parts
    axion_{pp,wc}_low200 and the zones axion_pp_zone{1,2a,2b,3,4} and axion_wc_zone{1,2a,2b,3,4a,4b};
  - their results as far as they have been synced (results/<time>_<part>: metrics.json, boost/, analysis/true_check.json),
    for the error of each network; a part still running fills its column as far as it has got, and the caption says so;
  - the settings of the profiles and of the evidence of the omega <= 60 study (runs/axion_pp_combined/profiles_*.yaml,
    the evidences of results/20261029_074412_axion_pp_combined), which this study is to use as well;
  - the probes of the likelihood (logs/axion_probe_<job>.json and .out) and the CLASS++ ladder of the high-omega study
    (LADDER), for Appendix B and Fig. scan_omega;
  - CLiENT's configuration schema (the laptop's mirror), for the defaults that the run files leave out;
  - the standard samplers on the combined emulators (SAMPLER, results/20261107_sampler_cost/sampler_cost.json of the
    work tree), for Appendix C.
"""
import glob
import json
import math
import os
import pickle
import re
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
# Where the results are (README.md, "Numbers, tables and figures"): CLIENT_PAPER_KIT, the cluster folder the jobs ran
# in (its runs/, logs/ and results/; this paper's folder by default), and CLIENT_PAPER_DATA, which holds the results of
# the smaller studies (results/<date>_*). CLiENT is this branch of client-f, or CLIENT_REPO.
DATA_ROOT = Path(os.environ.get("CLIENT_PAPER_DATA", HERE.parent / "data"))
KIT = Path(os.environ.get("CLIENT_PAPER_KIT", HERE.parent))
sys.path.insert(0, os.environ.get("CLIENT_REPO") or str(HERE.parents[2]))
RES, LOGS, RUNS = KIT / "results", KIT / "logs", KIT / "runs"
TABLES, DATA = HERE / "tables", HERE / "plots" / "data"
TABLES.mkdir(exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

# The parts of each dataset in increasing omega, and the bands in which neighbouring parts are blended
# (make_runs_zones.py, make_runs_zone2_split.py, make_runs_wc_zones.py and make_runs_zone1_split.py of
# results/20261102_high_omega_ladder; zone 1 of both split at omega = 37.5 on 6 November)
PARTS = {"pp": ["axion_pp_low200", "axion_pp_zone1a", "axion_pp_zone1b", "axion_pp_zone2a", "axion_pp_zone2b", "axion_pp_zone3",
                "axion_pp_zone4"],
         "wc": ["axion_wc_low200", "axion_wc_zone1a", "axion_wc_zone1b", "axion_wc_zone2a", "axion_wc_zone2b", "axion_wc_zone3",
                "axion_wc_zone4a", "axion_wc_zone4b"]}
BANDS = {"pp": [(2.6, 3.0), (36.75, 38.25), (60.5, 62.0), (87.25, 88.75), (110.5, 112.0), (153.5, 155.0)],
         "wc": [(6.0, 6.1), (36.75, 38.25), (60.5, 62.0), (87.25, 88.75), (110.5, 112.0), (153.5, 155.0), (173.75, 175.25)]}
PP_OLD_COMB = RES / "20261029_074412_axion_pp_combined"       # the omega <= 60 study: its evidence settings
PROFILES_OLD = {"omega_phi": "omega_phi", "aosc_omega": "aosc_omega", "aosc_phi": "aosc_phi_legacy"}

LABEL = {"omega_b": r"\omega_\mathrm{b}", "omega_cdm": r"\omega_\mathrm{cdm}", "theta_s_100": r"100\,\theta_\mathrm{s}",
         "logA": r"\ln(10^{10}A_\mathrm{s})", "n_s": r"n_\mathrm{s}", "nrun": r"\mathrm{d}n_\mathrm{s}/\mathrm{d}\ln k",
         "tau_reio": r"\tau_\mathrm{reio}", "A_osc": r"A_\mathrm{osc}", "omega": r"\omega", "phi": r"\phi"}
COSMO = ["omega_b", "omega_cdm", "theta_s_100", "logA", "n_s", "nrun", "tau_reio"]
OSC = ["A_osc", "omega", "phi"]
numbers = {}


def num(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    numbers[name] = value


# The true nodes of the profile studies, cached (Steen, 10 November 2026: a build for a text edit must not take a quarter of
# an hour). They are 50 000 and more files of ~15 kB in the kit's logs/ (most of it the refinement's state), and a pass over
# them took ~80 s, several passes a build. The few fields this script uses are cached in NODE_CACHE (CLIENT_PAPER_CACHE), per
# study:
#   build200.py              reads the cache, as of the last build with --profiles, and no node file;
#   build200.py --profiles   rereads the node files (after a sync from Grendel) and rewrites the cache.
# Without a cache a build reads the files as with --profiles. Everything else is computed from these records as before, so
# a build without --profiles gives the profile numbers, tables and figure data of the last --profiles build.
PROFILES = "--profiles" in sys.argv[1:]
NODE_CACHE = Path(os.environ.get("CLIENT_PAPER_CACHE", HERE / ".cache")) / "profile_nodes.pkl"
NODE_CACHE.parent.mkdir(parents=True, exist_ok=True)
_FROM_CACHE = not PROFILES and NODE_CACHE.exists()
_CACHE = pickle.loads(NODE_CACHE.read_bytes()) if _FROM_CACHE else dict(written=None, nodes={})
_NODES = _CACHE["nodes"]


def study_nodes(study):
    """The true nodes of a profile study, {(i, j): record} with the fields this script reads (logl, x, direct_logl, reason,
    backend_calls as node_records takes it, attempts as node_calls takes it, mtime), from the cache or the node files."""
    if study in _NODES or _FROM_CACHE:
        return _NODES.get(study, {})          # in the cache: a study without node files at the last --profiles build
    out = {}
    for fn in glob.glob(str(LOGS / f"profiles_{study}_surface/true_surface/node_*.json")):
        i, j = map(int, Path(fn).stem.split("_")[1:])
        e = json.loads(Path(fn).read_text())
        dg = e.get("diagnostics") or {}
        out[(i, j)] = dict(logl=e["logl"], x=e["x"], direct_logl=e.get("direct_logl"), reason=dg.get("reason"),
                           backend_calls=dg.get("backend_calls", 0),
                           attempts=dg.get("lifetime_node_attempts", dg.get("backend_calls")),
                           mtime=Path(fn).stat().st_mtime)
    _NODES[study] = out
    return out


print("profile nodes:", f"from the cache of {time.strftime('%-d %B %H:%M', time.localtime(_CACHE['written']))}"
      if _FROM_CACHE else "from the node files (the cache is rewritten at the end)")


def probe_cost(job):
    text = (LOGS / f"axion_probe_{job}.out").read_text(errors="replace")
    return float(re.search(r"phase probe: \d+ s; \d+ of \d+ finite; median ([0-9.]+) s", text).group(1))


def big(v):
    return f"{v:,}".replace(",", "\\,")


def power_of_ten(v):
    e = round(math.log10(v))
    assert math.isclose(v, 10.0 ** e)
    return f"$10^{{{e}}}$"


# ---------------------------------------------------------------- the parts: one likelihood, boxes differing in omega
cobaya, boxes, conf = {}, {}, {}
for data, parts in PARTS.items():
    files = {p: (RUNS / p / "cobaya.yaml").read_bytes() for p in parts}
    assert len(set(files.values())) == 1, f"{data}: the parts' cobaya.yaml differ"      # client.py combine needs this
    cobaya[data] = yaml.safe_load(files[parts[0]])
    boxes[data] = [json.loads((RUNS / p / "domain_bounds.json").read_text()) for p in parts]
    for b in boxes[data][1:]:
        assert {k: v for k, v in b.items() if k != "omega"} == {k: v for k, v in boxes[data][0].items() if k != "omega"}
    pr = cobaya[data]["params"]["omega"]["prior"]
    om = [b["omega"] for b in boxes[data]]
    assert om[0][0] == pr["min"] and om[-1][1] == pr["max"], (data, om, pr)
    for (lo_box, hi_box), (lo, hi) in zip(zip(om, om[1:]), BANDS[data]):
        assert lo_box[0] < lo < hi <= lo_box[1] and hi_box[0] <= lo < hi < hi_box[1], (data, lo, hi)
    cs = []
    for p in parts:
        c = yaml.safe_load((RUNS / p / "client.yaml").read_text())
        assert c.pop("likelihood")["input"] == f"runs/{p}/cobaya.yaml"
        cs.append(c)
    assert all(c == cs[0] for c in cs), f"{data}: the parts' settings differ"
    conf[data] = cs[0]
# H0WC: the settings of PDP+, with a mode lighter than 10% given fewer points of a boost
wc = json.loads(json.dumps(conf["wc"]))
num("WCbalancefloor", f"{wc['acquisition']['boost'].pop('balance_floor'):g}")
assert wc == conf["pp"], "the settings of H0WC differ from those of PDP+ by more than the balance floor"
for data, name in (("pp", "PP"), ("wc", "WC")):
    num(f"{name}parts", len(PARTS[data]))
    num(f"{name}nsampled", sum(1 for v in cobaya[data]["params"].values() if isinstance(v, dict) and "prior" in v))

# The priors and the training boxes (Thomas Tram's layout); the boxes in omega are those of Table tab:zones
with open(TABLES / "priors.tex", "w") as f:
    f.write("\\begin{tabular}{rlccc}\n\\cmidrule[\\heavyrulewidth](lr){2-5}\n"
            " & \\textbf{Parameter} & \\textbf{Prior} & \\multicolumn{2}{c}{\\textbf{Training boxes}} \\\\\n"
            " & & & \\PDP{} & \\HWC{} \\\\\n\\cmidrule(lr){2-5}\n")
    for group, keys in (("$\\Lambda$CDM", COSMO), ("Oscillation", OSC)):
        for k, key in enumerate(keys):
            pr = cobaya["pp"]["params"][key]["prior"]
            assert key == "phi" or cobaya["wc"]["params"][key]["prior"] == pr, key     # phi: each its own window
            if key == "A_osc":
                # the prior of the analysis (Steen, 13 November 2026): the runs' Cobaya prior is [0, 0.5], but the
                # emulators are trained on [0, 0.1], the evidences are shifted to that prior (`shift` in the evidence
                # section) and the Savage-Dickey ratio uses its density, as Sec. 3.2 and Table 3 say; the table shows it
                assert all(boxes[d][0]["A_osc"] == [0.0, 0.1] for d in ("pp", "wc"))
                pr = dict(min=0.0, max=0.1)
            cells = []
            for data in ("pp", "wc"):
                b = boxes[data][0]
                if key == "omega":
                    cells.append("Table~\\ref{tab:zones}")
                elif key == "phi":
                    cells.append(f"$[{b['phi'][0]:.3f}, {b['phi'][1]:.3f}]$")
                else:
                    cells.append("prior" if b[key] == [pr["min"], pr["max"]] else f"$[{b[key][0]:g}, {b[key][1]:g}]$")
            prior = "$[\\phi_0 - \\pi, \\phi_0 + \\pi]$" if key == "phi" else f"$[{pr['min']:g}, {pr['max']:g}]$"
            f.write(f"{group if k == 0 else ''} & ${LABEL[key]}$ & {prior} & {cells[0]} & {cells[1]} \\\\\n")
        if group != "Oscillation":
            f.write("\\cmidrule(lr){2-5}\n")
    f.write("\\cmidrule[\\heavyrulewidth](lr){2-5}\n\\end{tabular}\n")
for data, name in (("pp", "PP"), ("wc", "WC")):
    num(f"{name}phizero", f"{0.5 * sum(boxes[data][0]['phi']):+.2f}".replace("+", ""))

# ---------------------------------------------------------------- the precision of CLASS++ (Appendix B)
# The likelihood itself: the checks of the probes (smoke/axion_probe.py, its sampling ladder) give the gain in ln L of
# the oscillation at the maxima of each scan, with the other parameters at the reference point, for several samplings
# of CLASS++ and a very dense one. CLASS++ alone: the laptop ladder of the high-omega study (LADDER, ladder3.py and
# compare3.py there), the noise-weighted chi^2 between the lensed spectra with A_osc = 0.02 and 0 ("the imprint") and
# the same norm of the error of each sampling.
DEFAULT_L = "l_logstep=1.12,l_linstep=50"          # CLASS++'s own sampling of the multipoles


def probe_maxima(job):
    """The checks of a probe at the maxima of its scan (their amplitudes fitted; the grid at A_osc = 0.02 and the
    point at A_osc = 0 are left out), as improvements in chi^2 with each sampling: CLASS++'s default multipoles, the
    runs' settings for omega <= 60, those for omega <= 200 and the dense reference (None where not computed)."""
    d = json.loads((LOGS / f"axion_probe_{job}.json").read_text())
    high = d["run"].endswith("omega200")             # the probe's own settings are those for omega <= 200
    old = next((s for s in d["ladder"] if s.startswith("l_logstep=1.026")), None)
    rows = {}
    for c in d["check"]:
        if c["A_osc"] in (0.0, 0.02):
            continue
        g = c["gains"]
        rows[(round(c["omega"], 2), round(c["A_osc"], 4))] = dict(
            omega=c["omega"], A=c["A_osc"],
            default=2 * g[DEFAULT_L] if DEFAULT_L in g else None,
            runs60=2 * (g[old] if high else c["gain_run"]),
            runs200=2 * c["gain_run"] if high else None,
            ref=2 * g[d["dense"]])
    return list(rows.values()), d


def cell(v):
    return "--" if v is None else f"${v:.2f}$"


cl_rows = {"pp": [], "wc": []}
for data, jobs in (("pp", (6973447, 6973710, 7010679)), ("wc", (7010680,))):
    for job in jobs:
        cl_rows[data] += probe_maxima(job)[0]
    cl_rows[data].sort(key=lambda r: (r["omega"], r["A"]))
with open(TABLES / "classpp_likelihood.tex", "w") as f:
    f.write("\\begin{tabular}{rcccccc}\n\\cmidrule[\\heavyrulewidth](lr){2-7}\n"
            " & $\\omega$ & $\\Aosc$ & \\multicolumn{4}{c}{Improvement in $\\chi^2$ with the sampling} \\\\\n"
            "\\cmidrule(lr){4-7}\n"
            " & & & default & $\\omega \\lesssim 90$ & $\\omega \\leq 200$ & reference \\\\\n\\cmidrule(lr){2-7}\n")
    for data, name in (("pp", "\\PDP{}"), ("wc", "\\HWC{}")):
        for k, r in enumerate(cl_rows[data]):
            f.write(f"{name if k == 0 else ''} & ${r['omega']:g}$ & ${r['A']:.3f}$ & {cell(r['default'])} & "
                    f"{cell(r['runs60'])} & {cell(r['runs200'])} & {cell(r['ref'])} \\\\\n")
        if data == "pp":
            f.write("\\cmidrule(lr){2-7}\n")
    f.write("\\cmidrule[\\heavyrulewidth](lr){2-7}\n\\end{tabular}\n")
pp_rows, wc_rows = cl_rows["pp"], cl_rows["wc"]
at = lambda rows, w: next(r for r in rows if abs(r["omega"] - w) < 0.01)
num("CLdefaultFifty", f"{at(pp_rows, 50)['default']:.1f}")
num("CLrefFifty", f"{at(pp_rows, 50)['ref']:.1f}")
num("CLrunsNinety", f"{max(abs(r['runs60'] - r['ref']) for r in pp_rows if r['omega'] <= 90):.2f}")
num("CLrunsHundred", f"{abs(at(pp_rows, 100)['runs60'] - at(pp_rows, 100)['ref']):.1f}")
num("CLnewPP", f"{max(abs(r['runs200'] - r['ref']) for r in pp_rows if r['runs200'] is not None and r['omega'] < 199):.2f}")
num("CLnewPPedge", f"{abs(at(pp_rows, 200)['runs200'] - at(pp_rows, 200)['ref']):.1f}")
num("CLnewWC", f"{max(abs(r['runs200'] - r['ref']) for r in wc_rows if r['omega'] < 199):.2f}")
num("CLoldworst", f"{max(abs(r['runs60'] - r['ref']) for r in pp_rows + wc_rows):.0f}")
_, d_pp = probe_maxima(7010679)
_, d_wc = probe_maxima(7010680)
_, d_mid = probe_maxima(6973710)
# the cost of an evaluation: the settings of the runs (those for omega <= 200), and those sufficient up to omega ~ 90
num("PPcost", f"{probe_cost(7010679):.0f}")
num("WCcost", f"{probe_cost(7010680):.0f}")
num("PPcostold", f"{probe_cost(6973710):.1f}")
# the reference against the runs' settings, both timed in the same check of a probe (core seconds per evaluation)
RUN_CHECK = "run's settings, check"
num("CLrefRatioPP", f"{d_mid['cost']['(3) ' + d_mid['dense']][0] / d_mid['cost'][RUN_CHECK][0]:.0f}")
old_wc = next(s for s in d_wc["ladder"] if s.startswith("l_logstep=1.026"))
num("CLrefRatioWC", f"{d_wc['cost']['(2) ' + d_wc['dense']][0] / d_wc['cost']['(1) ' + old_wc][0]:.0f}")

# CLASS++ alone (LADDER): the imprint of A_osc = 0.02 and the error of each sampling, against REFP (PDP+, twice as
# dense as REF in all three samplings) and WREF (H0WC); for PDP+ the larger of the two phases 0 and pi/2.
LADDER = DATA_ROOT / "results/20261102_high_omega_ladder"
T0_UK = 2.7255e6
LADDER_COLUMNS = {"pp": ["P", "PK", "L4K", "PQ", "L8Q"], "wc": ["W", "WK", "WL", "WQ", "WLQ"]}


def ladder_norms(data, top, cols):
    files = {n: np.load(LADDER / f"{data}_{n}.npz") for n in cols + [top]}
    if data == "pp":       # Planck-like noise, TT to 2500, TE and EE to 2000
        lmax, fsky, beam, nt, npol, lmin = dict(tt=2500, te=2000, ee=2000), 0.6, 7.0, 30.0, 60.0, 2
    else:                  # ACT-like noise, l = 30-6327
        lmax, fsky, beam, nt, npol, lmin = dict(tt=6327, te=6327, ee=6327), 0.3, 1.4, 15.0, 21.0, 30
    first = files[top]
    ell = np.arange(len(first["0_tt"]))
    b = np.exp(ell * (ell + 1) * (beam / 60 * np.pi / 180) ** 2 / (8 * np.log(2)))
    NT, NP = ((x / 60 * np.pi / 180) ** 2 * b / T0_UK ** 2 for x in (nt, npol))
    tt, te, ee = first["0_tt"], first["0_te"], first["0_ee"]
    with np.errstate(divide="ignore", invalid="ignore"):
        var = dict(tt=2 / ((2 * ell + 1) * fsky) * (tt + NT) ** 2, ee=2 / ((2 * ell + 1) * fsky) * (ee + NP) ** 2,
                   te=(te ** 2 + (tt + NT) * (ee + NP)) / ((2 * ell + 1) * fsky))

    def norm(diff):
        return float(sum(np.sum(diff[s][lmin:lmax[s] + 1] ** 2 / var[s][lmin:lmax[s] + 1]) for s in ("tt", "te", "ee")))

    def imprint(fl, k):
        return {s: fl[f"{k}_{s}"] - fl[f"0_{s}"] for s in ("tt", "te", "ee")}

    out = {}
    for k, (A, w, p) in enumerate(first["cases"]):
        if k == 0:
            continue
        ref = imprint(first, k)
        row = out.setdefault(float(w), {"imprint": 0.0, **{n: 0.0 for n in cols}})
        row["imprint"] = max(row["imprint"], norm(ref))
        for n in cols:
            row[n] = max(row[n], norm({s: imprint(files[n], k)[s] - ref[s] for s in ref}))
    seconds = {n: float(np.median(files[n]["times"])) for n in cols}
    return out, seconds


ladder = {data: ladder_norms(data, top, LADDER_COLUMNS[data]) for data, top in (("pp", "REFP"), ("wc", "WREF"))}
with open(TABLES / "classpp_cl.tex", "w") as f:
    f.write("\\begin{tabular}{rccccccc}\n\\cmidrule[\\heavyrulewidth](lr){2-8}\n"
            " & $\\omega$ & Imprint & \\multicolumn{5}{c}{Error with the sampling} \\\\\n\\cmidrule(lr){4-8}\n"
            " & & & $\\omega \\lesssim 90$ & table & $\\ell$ & $k$ & $\\omega \\leq 200$ \\\\\n\\cmidrule(lr){2-8}\n")
    for data, name in (("pp", "\\PDP{}"), ("wc", "\\HWC{}")):
        rows, seconds = ladder[data]
        for k, (w, r) in enumerate(rows.items()):
            f.write(f"{name if k == 0 else ''} & ${w:g}$ & ${r['imprint']:.1f}$ & "
                    + " & ".join(f"${r[n]:.3f}$" for n in LADDER_COLUMNS[data]) + " \\\\\n")
        f.write(" & \\multicolumn{2}{l}{time per spectrum [s]} & "
                + " & ".join(f"${seconds[n]:.1f}$" for n in LADDER_COLUMNS[data]) + " \\\\\n")
        if data == "pp":
            f.write("\\cmidrule(lr){2-8}\n")
    f.write("\\cmidrule[\\heavyrulewidth](lr){2-8}\n\\end{tabular}\n")
# fast against accurate lensing (PDP+ at the reference's sampling, REF against REFL)
lens, _ = ladder_norms("pp", "REFL", ["REF"])
num("CLlensLow", f"{max(r['REF'] for w, r in lens.items() if w <= 130):.3f}")
num("CLlensHigh", f"{max(r['REF'] for w, r in lens.items() if w >= 160):.3f}")
# the primordial table of 200 points per decade (the runs') against 1000, once l and k are dense, at omega = 200
table200, _ = ladder_norms("pp", "REFP", ["L8Q200"])
num("CLtableTwoHundred", f"{table200[200.0]['L8Q200']:.2f}")
num("CLnewTwoHundred", f"{ladder['pp'][0][200.0]['L8Q']:.2f}")

# The scans of the likelihood over omega (Fig. of Appendix B): the improvement in chi^2 over A_osc = 0 at the reference
# point, maximised over A_osc and phi, as the probes computed it. PDP+ is joined from three probes, each used where it is
# the newest: 6973447 (omega 1-40) and 6973710 (40-55), with the settings of the runs, and 7010679 (55-200), with the
# settings for omega <= 200; H0WC is 7010680 (55-200) alone.
SCAN_PIECES = {"pp": [(6973447, 1.0, 40.0), (6973710, 40.0, 55.0), (7010679, 55.0, 200.01)], "wc": [(7010680, 55.0, 200.01)]}
scan = {}
for data, pieces in SCAN_PIECES.items():
    om, dchi = [], []
    for job, lo, hi in pieces:
        prof = json.loads((LOGS / f"axion_probe_{job}.json").read_text())["profile"]
        for p_ in prof:
            if lo <= p_["omega"] < hi:
                om.append(p_["omega"])
                dchi.append(2 * p_["gain"])
    order = np.argsort(om)
    scan[data] = dict(omega=[float(om[k]) for k in order], dchi2=[float(dchi[k]) for k in order],
                      pieces=[dict(job=j, omega=[lo, min(hi, 200.0)]) for j, lo, hi in pieces])
(DATA / "scan_omega.json").write_text(json.dumps(scan) + "\n")

# The zones (Table tab:zones): the training box of each part in omega, the band to the next part, and the largest
# maxima of the scan of the likelihood (Fig. scan_omega) in the range in which the part is used
def scan_maxima(data, lo, hi, at_least=3.0, count=3):
    if data not in scan:
        return []
    om, dchi = np.array(scan[data]["omega"]), np.array(scan[data]["dchi2"])
    peaks = [k for k in range(1, len(om) - 1) if dchi[k] >= dchi[k - 1] and dchi[k] >= dchi[k + 1]
             and lo <= om[k] < hi and dchi[k] >= at_least]
    return sorted(peaks, key=lambda k: -dchi[k])[:count]


with open(TABLES / "zones.tex", "w") as f:
    f.write("\\begin{tabular}{rcccl}\n\\cmidrule[\\heavyrulewidth](lr){2-5}\n"
            " & Part & Training box in $\\omega$ & Band to the next & Largest maxima of the scan \\\\\n\\cmidrule(lr){2-5}\n")
    for data, name in (("pp", "\\PDP{}"), ("wc", "\\HWC{}")):
        om = [b["omega"] for b in boxes[data]]
        scanned = min(scan[data]["omega"])
        for k, part in enumerate(PARTS[data]):
            used = (om[0][0] if k == 0 else BANDS[data][k - 1][1], om[-1][1] if k == len(om) - 1 else BANDS[data][k][0])
            label = "low" if part.endswith("low200") else part.rsplit("zone", 1)[1]
            band = "--" if k == len(om) - 1 else f"${BANDS[data][k][0]:g}$--${BANDS[data][k][1]:g}$"
            if used[1] <= scanned:
                peaks = "not scanned"
            else:
                o, dc = np.array(scan[data]["omega"]), np.array(scan[data]["dchi2"])
                ks = scan_maxima(data, *used)
                peaks = ", ".join(f"${o[j]:g}$ (${dc[j]:.1f}$)" for j in ks) or "none above 3"
                if used[0] < scanned:
                    peaks += f" (scanned from ${scanned:g}$)"
            f.write(f"{name if k == 0 else ''} & {label} & $[{om[k][0]:g}, {om[k][1]:g}]$ & {band} & {peaks} \\\\\n")
        if data == "pp":
            f.write("\\cmidrule(lr){2-5}\n")
    f.write("\\cmidrule[\\heavyrulewidth](lr){2-5}\n\\end{tabular}\n")

# The scan in the text (Sec. 2.2): the highest maximum below omega = 60 and the highest above, and the largest
# improvement of the scan inside the bands (where the parts are blended)
def scan_top(data, lo, hi, count):
    o, dc = np.array(scan[data]["omega"]), np.array(scan[data]["dchi2"])
    return [(o[k], dc[k]) for k in scan_maxima(data, lo, hi, 0.0, count)]


w, v = scan_top("pp", 1.0, 60.0, 1)[0]
num("ScanPPbestomega", f"{w:g}")
num("ScanPPbest", f"{v:.1f}")
for tag, (w, v) in zip("ABC", scan_top("pp", 60.0, 200.01, 3)):
    num(f"ScanPPhigh{tag}omega", f"{w:g}")
    num(f"ScanPPhigh{tag}", f"{v:.1f}")
w, v = scan_top("wc", 60.0, 200.01, 1)[0]
num("ScanWChighomega", f"{w:g}")
num("ScanWChigh", f"{v:.1f}")
for data, name in (("pp", "PP"), ("wc", "WC")):
    o, dc = np.array(scan[data]["omega"]), np.maximum(np.array(scan[data]["dchi2"]), 0.0)
    inside = [dc[(o >= lo) & (o <= hi)].max() for lo, hi in BANDS[data] if lo >= min(o)]
    num(f"BandMax{name}", f"{max(inside):.1f}")

# ---------------------------------------------------------------- the settings of CLiENT (Tables tab:client, tab:client_b)
# Read from the run files (identical for all parts of a dataset, and for the two datasets but for the balance floor);
# the defaults the files leave out (the margin of the members) from CLiENT's schema; the profiles and the evidence with
# the settings of the omega <= 60 study, which this study is to use as well.
from config.config import Config  # noqa: E402

c = conf["pp"]
# client-f refuses two keys of the runs' files (README.md, "What a rerun reproduces"); its schema gets them without
CLIENT_F_REFUSES = {"acquisition": ("batch_size",), "training": ("warm_start_preserve_function",)}
c_f = {k: ({kk: vv for kk, vv in v.items() if kk not in CLIENT_F_REFUSES.get(k, ())} if isinstance(v, dict) else v)
       for k, v in c.items()}
full = Config.from_dict(json.loads(json.dumps(dict(c_f, likelihood={"wrapper": "cobaya", "input": "cobaya.yaml"}))))
acq, boost, audit, train = c["acquisition"], c["acquisition"]["boost"], c["acquisition"]["audit"], c["training"]
pconf = {pair: yaml.safe_load((RUNS / f"axion_pp_combined/profiles_{name}.yaml").read_text()) for pair, name in PROFILES_OLD.items()}
evs = [json.loads(Path(e).read_text()) for e in sorted(glob.glob(str(PP_OLD_COMB / "evidence/evidence_it_0_nlive*.json")))
       if re.search(r"evidence_it_0_nlive\d+(_seed\d+)?\.json$", e)]
yes = {True: "yes", False: "no"}
assert boost["mode_depth"] == "auto" and train["lr_schedule"] == "plateau" and pconf["omega_phi"].get("true_optimizer", "corrected") == "corrected"
assert {pconf[p]["surrogate_optimizer"] for p in pconf} == {"hybrid", "legacy"} and pconf["aosc_phi"]["surrogate_optimizer"] == "legacy"
assert train["members"] > 1 and train["member_pilot"] > 0
num("Members", train["members"])
num("Pilot", train["member_pilot"])
num("PilotPool", big(train["members"] * train["member_pilot"]))
num("Margin", f"{100 * full.training.member_margin:.0f}")
num("Boosts", boost["boosts"])
num("Boostpoints", big(boost["n"]))
settings = [
    ("prior", "n_samples", "points of the Latin hypercube design", big(c["prior"]["n_samples"])),
    ("", "n_inject", "points from a Gaussian fit to it", c["prior"]["n_inject"]),
    ("acquisition", "n_append", "new points per iteration", big(acq["n_append"])),
    ("", "target_temperature", "temperature $T$ of the chain", f"{acq['target_temperature']:g}"),
    ("", "pool_factor", "candidates per selected point", acq["pool_factor"]),
    ("", "n_neighbors", "neighbours in the density estimates", acq["n_neighbors"]),
    ("", "n_inject", "Gaussian points per iteration", acq["n_inject"]),
    ("", "inject_window", "fitted points, in units of $Tn/2$", acq.get("inject_window", 16)),
    ("acquisition.audit", "tau", "check maxima this far above the best", audit["tau"]),
    ("", "chain", "boosts: also the cooled chain", yes[audit["chain"]]),
    ("", "posterior", "boosts: posterior samples checked", audit["posterior"]),
    ("convergence", "threshold", r"largest scatter, in units of $T$", c["convergence"]["threshold"]),
    ("", "consecutive", "for this many networks in a row", c["convergence"]["consecutive"]),
    ("acquisition.boost", "boosts", "number of boosts", boost["boosts"]),
    ("", "n", "posterior points per boost", big(boost["n"])),
    ("", "density_filter", "select where the training set is sparse", yes[boost["density_filter"]]),
    ("", "mixing", "merge basins below this memory", boost["mixing"]),
    ("", "min_share", "smallest share of walkers in a mode", boost["min_share"]),
    ("", "balance_floor", "modes below this get fewer points", numbers["WCbalancefloor"] + r" (\HWC{})"),
    ("", "mode_depth", "drop modes this far below the highest", "$Tn/2$"),
    ("", "background", "function fitted near the maximum", boost["background"]),
    ("", "train_domain", "training domain, in units of its radius", boost["train_domain"]),
    ("", "region", "region edge, in units of the reach", boost["region"]),
    ("", "region_quantile", "quantile that sets the reach", boost["region_quantile"]),
    ("model", "n_layers", "hidden layers", c["model"]["n_layers"]),
    ("", "n_neurons", "neurons per hidden layer", c["model"]["n_neurons"]),
    ("", "activation", r"activation function of ref.~\cite{SPECULATOR:2020}", c["model"]["activation"]),
    ("training", "loss", "error relative to the depth", train["loss"]),
    ("", "sigma_level", r"level (in $\sigma$) that sets the margin $c$", train["sigma_level"]),
    ("", "msre_ess_floor", "smallest effective fraction of points", train["msre_ess_floor"]),
    ("", "learning_rate", "initial learning rate (Adam)", power_of_ten(train["learning_rate"])),
    ("", "lr_schedule", rf"$\times {train['lr_factor']:g}$ if no improvement in {train['lr_patience']} epochs",
     train["lr_schedule"]),
    ("", "batch_size", "points per batch", train["batch_size"]),
    ("", "warm_start", "the first network starts from the previous one",
     yes[train["warm_start"] and train.get("warm_start_preserve_function", True)]),
    ("", "members", "trainings per network in the boosts", train["members"]),
    ("", "member_pilot", "posterior points of each network checked", train["member_pilot"]),
    ("", "member_margin", "lower error needed to replace the first", f"{full.training.member_margin:g}"),
    ("sampling", "n_walkers", "walkers of the ensemble sampler", c["sampling"]["n_walkers"]),
    ("", "n_steps", rf"steps, after a burn-in of {big(c['sampling']['burn_in'])}", big(c["sampling"]["n_steps"])),
    ("profiles", "profile_points", "nodes per parameter", pconf["omega_phi"]["profile_points"]),
    ("", "surrogate_optimizer", r"scan; legacy for $(\Aosc, \phi)$", "hybrid"),
    ("", "hybrid_hops", "most rounds of the hybrid scan", pconf["omega_phi"]["hybrid_hops"]),
    ("", "corrected_method", "refinement with the true likelihood", pconf["omega_phi"].get("corrected_method", "newton")),
    ("", "maxfun", "most true evaluations per node", pconf["omega_phi"]["maxfun"]),
    ("", "region", "the emulator inside its region", pconf["omega_phi"]["region"]),
    ("evidence", "-{}-n-live", "live points of BEST", evs[0]["n_live"]),
    ("", "-{}-n-live-updates", "live points replaced per iteration", evs[0]["n_live_updates"]),
    ("", "-{}-tolerance", "evidence left in the live points", power_of_ten(evs[0]["tolerance"])),
]
split = next(k for k, row in enumerate(settings) if row[0] == "model")       # the points (a), the networks and tools (b)
for name, rows in (("client_settings", settings[:split]), ("client_settings_b", settings[split:])):
    with open(TABLES / f"{name}.tex", "w") as f:
        # Gaps of 5 pt instead of 6: with 6 the table of the points was 5 pt wider than the text (18 November 2026)
        f.write("\\setlength{\\tabcolsep}{5pt}\n\\begin{tabular}{llp{5.6cm}l}\n\\cmidrule[\\heavyrulewidth](lr){2-4}\n"
                " & Setting & Meaning & Value \\\\\n\\cmidrule(lr){2-4}\n")
        for k, (group, key, meaning, value) in enumerate(rows):
            if group and k:
                f.write("\\cmidrule(lr){2-4}\n")
            g = f"\\texttt{{{group}}}" if group else ""
            key = key.replace("_", "\\_")
            f.write(f"{g} & \\texttt{{{key}}} & {meaning} & {value} \\\\\n")
        f.write("\\cmidrule[\\heavyrulewidth](lr){2-4}\n\\end{tabular}\n")

# ---------------------------------------------------------------- the error of each network (Tables tab:training_pp, _wc)
# For each part, the scatter of the network minus the true likelihood at points on which the network had not been
# trained: at the tempered batch that each network of the iterative stage selected (metrics.json, held_out_sd, the
# number the job log prints and Table training of the omega <= 60 draft gave), and at the posterior points of each boost
# (boost_it_N.csv, drawn by network N; the sd of its boost_it_N.json). Then the rms (bias included) of the network minus
# the true likelihood at the 500 posterior samples of the kit's true check (analysis/true_check.json), for the network
# that drew each boost and for the final one, trained on all boosts. Refused points and values that are not finite are
# left out. A part still running shows what its files hold, and the caption names it with its progress.
def result_dir(part):
    """The part's results, or None while it has not been synced (a part just submitted)."""
    found = sorted(RES.glob(f"2026*_{part}"))
    assert len(found) <= 1, (part, found)
    return found[0] if found else None


def part_label(part):
    return "low" if part.endswith("low200") else part.rsplit("zone", 1)[1]


progress, check_points = {}, set()
for data in PARTS:
    n_boosts = conf[data]["acquisition"]["boost"]["boosts"]
    for part in PARTS[data]:
        p = result_dir(part)
        if p is None or not (p / "metrics.json").exists():
            progress[part] = dict(start=None, tempered=[], boosts=[], checks=[], done=False, started=p is not None)
            continue
        start = json.loads((p / "boost/start.json").read_text())["start_iteration"] if (p / "boost/start.json").exists() else None
        acq = sorted(json.loads((p / "metrics.json").read_text())["acquisition"], key=lambda a: a["iteration"])
        tempered = [a["held_out_sd"] for a in acq if start is None or a["iteration"] < start]
        assert [a["iteration"] for a in acq][:len(tempered)] == list(range(len(tempered))), part
        refused = json.loads((p / "refused_value.json").read_text()).get("value")
        boosts = []
        for it in range(start, start + n_boosts) if start is not None else ():
            if not (p / f"boost/boost_it_{it}.csv").exists():
                break
            b = pd.read_csv(p / f"boost/boost_it_{it}.csv")
            ok = np.isfinite(b.loglkl) & np.isfinite(b.predicted) & (b.loglkl != refused)
            boosts.append(float(np.std(b.predicted[ok] - b.loglkl[ok])))
            sd = json.loads((p / f"boost/boost_it_{it}.json").read_text())["error"]["sd"]
            assert math.isclose(boosts[-1], sd, rel_tol=1e-3), (part, it, boosts[-1], sd)
        checked = {}
        if (p / "analysis/true_check.json").exists():
            for n in json.loads((p / "analysis/true_check.json").read_text())["networks"]:
                assert start is not None and n["iteration"] >= start, (part, n["iteration"], start)
                check_points.add(n["points"])
                checked[n["iteration"]] = n["truth_minus_network"]["rms"]
        # the network that drew boost k (k = 1 .. n_boosts) is start + k - 1; the final one is start + n_boosts
        checks = [checked.get(start + k) for k in range(n_boosts + 1)] if start is not None else []
        # finished once the network trained on the last boost is there
        done = start is not None and (p / f"trained_models/model_it_{start + n_boosts}.keras").exists()
        progress[part] = dict(start=start, tempered=tempered, boosts=boosts, checks=checks, done=done)
assert len(check_points) <= 1, check_points                          # the same number of samples in all checks
num("TrainingCheckn", f"{min(check_points)}" if check_points else "\\draft{?}")


def cell(values, k, fmt):
    return format(values[k], fmt) if k < len(values) and values[k] is not None else ""


def plural(items):
    return f"part{'s' if len(items) > 1 else ''} {', '.join(items)}"


for data, name in (("pp", "PP"), ("wc", "WC")):
    parts, n_boosts = PARTS[data], conf[data]["acquisition"]["boost"]["boosts"]
    cols, last = [progress[p] for p in parts], len(parts) + 2
    with open(TABLES / f"training_{data}.tex", "w") as f:
        f.write(f"\\begin{{tabular}}{{rl{'c' * len(parts)}}}\n\\cmidrule[\\heavyrulewidth](lr){{2-{last}}}\n"
                f" & & \\multicolumn{{{len(parts)}}}{{c}}{{\\textbf{{Part}} (Table~\\ref{{tab:zones}})}} \\\\\n"
                f" & & {' & '.join(part_label(p) for p in parts)} \\\\\n\\cmidrule(lr){{2-{last}}}\n")
        for k in range(max(len(c["tempered"]) for c in cols)):        # as many networks as the longest iterative stage
            f.write(f"{'Iterations' if k == 0 else ''} & network {k} & "
                    f"{' & '.join(cell(c['tempered'], k, '.3g') for c in cols)} \\\\\n")
        f.write(f"\\cmidrule(lr){{2-{last}}}\n")
        for k in range(n_boosts):                                     # all boosts, filled as the runs proceed
            f.write(f"{'Boosts' if k == 0 else ''} & {k + 1} & {' & '.join(cell(c['boosts'], k, '.2g') for c in cols)} \\\\\n")
        f.write(f"\\cmidrule(lr){{2-{last}}}\n")
        for k in range(n_boosts + 1):                                 # the network of each boost, then the final one
            f.write(f"{'Posterior' if k == 0 else ''} & {k + 1 if k < n_boosts else 'final'} & "
                    f"{' & '.join(cell(c['checks'], k, '.2g') for c in cols)} \\\\\n")
        f.write(f"\\cmidrule[\\heavyrulewidth](lr){{2-{last}}}\n\\end{{tabular}}\n")
    # the parts still running, and finished parts whose networks have not all been compared with the true likelihood,
    # for the caption (red); empty once all are finished and compared
    running, unchecked = [], []
    for p in parts:
        if not progress[p]["done"]:
            k = len(progress[p]["boosts"])
            stage = ("not started" if not progress[p].get("started", True) else
                     "iterative stage" if progress[p]["start"] is None else
                     f"{k} of {n_boosts} boosts" if k < n_boosts else "training the final network")
            running.append(f"{part_label(p)} ({stage})")
        elif None in progress[p]["checks"]:
            unchecked.append(part_label(p))
    note = f"Still running: {plural(running)}." if running else ""
    if unchecked:
        note += f" Not yet compared with the true likelihood at all networks: {plural(unchecked)}."
    num(f"Training{name}note", f"\\draft{{{note.strip()}}}" if note else "")
    # the network after which the iterative stage ended (Appendix A.1), over the parts that have reached the boosts
    after = sorted({progress[p]["start"] - 1 for p in parts if progress[p]["start"] is not None})
    if not after:
        words = "\\draft{?}"
    elif len(after) == 1:
        words = f"{after[0]}"
    elif len(after) == 2 and after[1] == after[0] + 1:
        words = f"{after[0]} or {after[1]}"
    else:
        words = f"{after[0]} to {after[-1]}"
    waiting = [part_label(p) for p in parts if progress[p]["start"] is None]
    if waiting:
        words += f" \\draft{{(so far; {plural(waiting)} still in the iterative stage)}}"
    num(f"{name}handoverafter", words)

# ---------------------------------------------------------------- the posterior after four boosts (Sec. numerics:client)
# The boosts went on after the posterior had stopped changing, to show how low the pointwise error can be driven (Steen,
# 7 November 2026: the paper says so). For each part of PDP+, network start + EARLY (trained on EARLY boosts) against the
# final one: the largest shift of the posterior mean of the ten cosmological and oscillation parameters, in standard
# deviations of the final posterior, and the largest relative change of a standard deviation (the kit's chains,
# analysis/posterior_it_N.npz, equal weights; two halves of a final chain differ by 0.04 to 0.09 standard deviations);
# the evidence over its region, which weights the part in the combined emulator (bridge sampling of the network's modes,
# boost/boost_it_N.json; the final network draws no boost, so the one before it); the training points with a value
# (training_data/data_it_N.csv, refused points and values that are not finite left out, as the combine does); and the rms
# error at its posterior samples (the true check, the last block of Table tab:training_pp).
EARLY = 4
EARLY_WC = 5          # H0WC (Steen, 10 November 2026: a robust posterior and evidence after five boosts; checked below)
KEY = ["omega_b", "omega_cdm", "theta_s_100", "logA", "n_s", "nrun", "tau_reio", "A_osc", "omega", "phi"]


def early_metrics(d, early):
    """Network start + early of each part of dataset d against its final one: the largest shift of a posterior mean (in
    final standard deviations), the largest relative change of a standard deviation, the largest change of ln Z over the
    part's region, the training points with a value then and at the end, and the rms errors at posterior samples then and
    at the end (the true checks)."""
    n_d = conf[d]["acquisition"]["boost"]["boosts"]
    shift = width = dlnz = 0.0
    points = {"early": 0, "final": 0}
    for part in PARTS[d]:
        p, s = result_dir(part), progress[part]["start"]
        chains = []
        for n in (s + early, s + n_d):
            z = np.load(p / f"analysis/posterior_it_{n}.npz", allow_pickle=True)
            names = [str(x) for x in z["names"]]
            chains.append(z["x"][:, [names.index(k) for k in KEY]])
        a, b = chains
        shift = max(shift, float(np.max(np.abs(a.mean(0) - b.mean(0)) / b.std(0))))
        width = max(width, float(np.max(np.abs(a.std(0) / b.std(0) - 1))))

        def lnz(n):
            modes = [m["log_evidence"] for m in json.loads((p / f"boost/boost_it_{n}.json").read_text())["modes"]]
            top = max(modes)
            return top + math.log(sum(math.exp(v - top) for v in modes))
        dlnz = max(dlnz, abs(lnz(s + early) - lnz(s + n_d - 1)))
        refused = json.loads((p / "refused_value.json").read_text()).get("value")
        for key, n in (("early", s + early), ("final", s + n_d)):
            y = pd.read_csv(p / f"training_data/data_it_{n}.csv", usecols=["loglkl"]).loglkl
            points[key] += int((np.isfinite(y) & (y != refused)).sum())
    early_rms = [progress[p]["checks"][early] for p in PARTS[d]]
    final_rms = [progress[p]["checks"][n_d] for p in PARTS[d]]
    return dict(shift=shift, width=width, dlnz=dlnz, points=points, early_rms=early_rms, final_rms=final_rms)


for d, tag, early in (("pp", "PP", EARLY), ("wc", "WC", EARLY_WC)):
    if all(progress[p]["done"] and None not in progress[p]["checks"] for p in PARTS[d]):
        m = early_metrics(d, early)
        num(f"{tag}earlyBoosts", ["zero", "one", "two", "three", "four", "five", "six"][early])
        num(f"{tag}earlyShift", f"{m['shift']:.2f}")
        num(f"{tag}earlyWidth", f"{round(100 * m['width'])}")
        num(f"{tag}earlylnZ", f"{m['dlnz']:.2f}")
        num(f"{tag}earlyPoints", f"{m['points']['early']:,}".replace(",", "\\,"))
        num(f"{tag}finalPoints", f"{m['points']['final']:,}".replace(",", "\\,"))
        num(f"{tag}earlyRmsLow", f"{min(m['early_rms']):.2f}")
        num(f"{tag}earlyRmsHigh", f"{max(m['early_rms']):.2f}")
        num(f"{tag}finalRmsLow", f"{min(m['final_rms']):.2f}")
        num(f"{tag}finalRmsHigh", f"{max(m['final_rms']):.2f}")
        for k in ((3, 4, 5, 6, 7, 8) if d == "wc" else (early,)):
            m = early_metrics(d, k)
            print(f"{tag} after {k} boosts: mean shift <= {m['shift']:.3f} sd, sd change <= {m['width']:.3f}, "
                  f"ln Z change <= {m['dlnz']:.3f}, training points {m['points']['early']} of {m['points']['final']}, "
                  f"rms {min(m['early_rms']):.2f}-{max(m['early_rms']):.2f} -> {min(m['final_rms']):.2f}-{max(m['final_rms']):.2f}")
    else:
        for k in ("earlyBoosts", "earlyShift", "earlyWidth", "earlylnZ", "earlyPoints", "finalPoints", "earlyRmsLow",
                  "earlyRmsHigh", "finalRmsLow", "finalRmsHigh"):
            num(tag + k, "\\draft{?}")

# ---------------------------------------------------------------- how well the members' pilot predicts (App. A.2)
# For every network trained with several members, in all emulators of both datasets (also those split later: the single
# emulators for 2.5 <= omega <= 200 and the first zones), the error of the kept network at the pilot points (pilot_error
# in trained_models/members/members_it_N.json) against the rms of the same network minus the true likelihood at the
# points of its own boost (boost_it_N.json, drawn by network N). 11 November 2026: within 30% for 87% of 170 networks,
# more than 30% higher for 4% (the comparison with the true check at posterior samples gives 80% and 4%).
pilot_ratio = []
for part in PARTS["pp"] + PARTS["wc"] + ["axion_pp_high200", "axion_pp_zone1", "axion_pp_zone2", "axion_wc_high200",
                                         "axion_wc_zone1"]:
    p = result_dir(part)
    if p is None:
        continue
    for m in sorted((p / "trained_models/members").glob("members_it_*.json")):
        rec = json.loads(m.read_text())
        kept = next(r for r in rec["members"] if r["member"] == rec["kept"])
        b = p / f"boost/boost_it_{int(m.stem.rsplit('_', 1)[1])}.json"
        if b.exists() and kept.get("pilot_error") is not None and np.isfinite(kept["pilot_error"]):
            pilot_ratio.append(json.loads(b.read_text())["error"]["rms"] / kept["pilot_error"])
pilot_ratio = np.array(pilot_ratio)
num("PilotNets", f"{len(pilot_ratio)}")
num("PilotWithin", f"{100 * np.mean(abs(pilot_ratio - 1) <= 0.3):.0f}")
num("PilotUnder", f"{100 * np.mean(pilot_ratio > 1.3):.0f}")
print(f"pilot: {len(pilot_ratio)} networks, boost error within 30% of the pilot error for "
      f"{100 * np.mean(abs(pilot_ratio - 1) <= 0.3):.0f}%, more than 30% higher for {100 * np.mean(pilot_ratio > 1.3):.0f}%")

# ---------------------------------------------------------------- the 2D profiles of PDP+ (Fig. profile_triangle)
# The three surfaces of the combined emulator in the oscillation parameters, from the snapshots of the profile studies in
# the kit's logs/ (analysis/snapshot_profiles.sh), finished or not. (A_osc, omega) and (omega, phi) are each four studies
# of 50 x 50 nodes, one per window in omega (runs/axion_pp_combined200, make_profiles200.py); the windows share their edge
# columns and are joined into one surface along omega, with the better of the two values at a shared column. A node
# without a true value yet shows the emulator's value from the study's scan (shaded in the figure). Windows 1 and 2 of
# (A_osc, omega) are the studies with A_osc narrowed to 0.06 and 0.09 (make_profiles200_narrow.py, 6 November evening:
# with 0.08 and 0.1 the scans put the optimum of 57 and 31 nodes at the region's edge, and the studies would have stopped
# after their scans); until their scans are in, the scans of the first studies stand in (the fallbacks with the older
# scan, node by node the better of these and the hybrid scan, as build.py did for H0WC). A window that ends below
# A_osc = 0.1 is interpolated linearly in A_osc onto the grid of the others, and above its end it lies outside the study.
# (A_osc, phi) is the study with omega free over the prior joined node by node, by the largest value, with four studies
# that search omega only around the maxima it missed (make_profiles200_aosc_phi_windows.py, 7 November: compared with
# the other two planes' true optima it was up to 23 too low in chi^2 at omega 179 and 196); join_ranges.
# Writes plots/data/profile2d_pp_<pair>.json and, for the marginalised posterior on the diagonal,
# plots/data/posterior200_pp.npz (the sampled parameters of the newest combined emulator's chain, POST_COMB).
PROF_COMB = sorted(RES.glob("2026*_axion_pp_combined200"))[0]     # the profiles' emulator (six parts, 6 November)
# the posterior in the figures: the newest combined emulator with a chain (7 November: the seven parts, zone 1 split);
# the profiles are values of the true likelihood and stay those of the six-part emulator's studies
POST_COMB = sorted(d for d in RES.glob("2026*_axion_pp_combined200") if (d / "analysis/posterior_it_0.npz").exists())[-1]
POST_PARTS = ["", "one", "two", "three", "four", "five", "six", "seven", "eight"][
    len(json.loads((POST_COMB / "combine.json").read_text())["parts"])]
PROF_RUNS = RUNS / "axion_pp_combined200"
# pair: per window (study, the study whose scan stands in while the study's own has not settled, its YAML), or a list of
# these in order of preference, of which the first with a scan or true nodes in its snapshot is used
PROF_WINDOWS = {
    "aosc_omega": [[("axion_pp200_aosc_omega_w1_a060_legacy", "axion_pp200_aosc_omega_w1_a060", "fallback/profiles_aosc_omega_w1_a060_legacy.yaml"),
                    ("axion_pp200_aosc_omega_w1_a060", None, "profiles_aosc_omega_w1_a060.yaml"),
                    ("axion_pp200_aosc_omega_w1_legacy", "axion_pp200_aosc_omega_w1", "fallback/profiles_aosc_omega_w1_legacy.yaml")],
                   [("axion_pp200_aosc_omega_w2_a090_legacy", "axion_pp200_aosc_omega_w2_a090", "fallback/profiles_aosc_omega_w2_a090_legacy.yaml"),
                    ("axion_pp200_aosc_omega_w2_a090", None, "profiles_aosc_omega_w2_a090.yaml"),
                    ("axion_pp200_aosc_omega_w2_legacy", "axion_pp200_aosc_omega_w2", "fallback/profiles_aosc_omega_w2_legacy.yaml")],
                   ("axion_pp200_aosc_omega_w3", None, "profiles_aosc_omega_w3.yaml"),
                   ("axion_pp200_aosc_omega_w4", None, "profiles_aosc_omega_w4.yaml")],
    "omega_phi": [(f"axion_pp200_omega_phi_w{k}", None, f"profiles_omega_phi_w{k}.yaml") for k in range(1, 5)],
    # (A_osc, phi): the study with omega free over the prior, then the four with omega in a narrow range
    # (make_profiles200_aosc_phi_windows.py, 7 November; CLiENT's search_ranges), each before its fallback with the older scan
    "aosc_phi": [("axion_pp200_aosc_phi", None, "profiles_aosc_phi.yaml")]
                + [[(f"axion_pp200_aosc_phi_{t}_legacy", None, f"fallback/profiles_aosc_phi_{t}_legacy.yaml"),
                    (f"axion_pp200_aosc_phi_{t}", None, f"profiles_aosc_phi_{t}.yaml")]
                   for t in ("om18", "om30", "om180", "om196")],
}
PROF_NAMES = [p["name"] for p in json.loads((PROF_COMB / "metadata.json").read_text())["parameters"]]


def in_snapshot(study):
    """Whether the kit's logs/ hold a scan or true nodes of the study."""
    d = LOGS / f"profiles_{study}_surface"
    return any((d / n).exists() for n in ("surrogate_surface.json", "failed_surrogate_diagnostics.json")) \
        or bool(study_nodes(study))


def chosen(window):
    """The study of a window: the first of its alternatives in the snapshots, else the last."""
    if isinstance(window, tuple):
        return window
    return next((w for w in window if in_snapshot(w[0])), window[-1])


PROF_WINDOWS = {pair: [chosen(w) for w in ws] for pair, ws in PROF_WINDOWS.items()}


def emulator_scan(study):
    """A study's scan of the emulator, as far as it has come (the last pass of a scan still running is the same file), or
    the scan of a study that stopped because it did not settle; None before its first pass or without a snapshot."""
    for name in ("surrogate_surface.json", "failed_surrogate_diagnostics.json"):
        file = LOGS / f"profiles_{study}_surface" / name
        if study and file.exists():
            return json.loads(file.read_text())
    return None


def profile_window(study, standin, yaml_name, runs=None, names=None):
    """One study on its 50 x 50 grid: the true nodes done so far, the emulator's elsewhere (runs, names: the data set's
    YAML folder and parameters, PDP+'s by default)."""
    runs, names = runs or PROF_RUNS, names or PROF_NAMES
    conf = yaml.safe_load((runs / yaml_name).read_text())
    pars = list(conf["parameters"])
    ax = [np.linspace(*conf["ranges"][p], conf["profile_points"]) for p in pars]
    shape = (len(ax[0]), len(ax[1]))
    Z, X = np.full(shape, np.nan), np.full(shape + (len(names),), np.nan)
    true = np.zeros(shape, bool)
    for (i, j), e in study_nodes(study).items():
        Z[i, j], X[i, j], true[i, j] = e["logl"], e["x"], True
    own, other = emulator_scan(study), emulator_scan(standin)
    scan = own
    for s in (own, other):
        if s is not None:
            # the checkpoint of a scan still running has no names yet (the workflow adds them after its check), and a
            # failed hybrid scan the scan's own p0, p1, ...
            assert s.get("full_parameter_names", names) in (names, [f"p{i}" for i in range(len(names))]), study
            assert np.shape(s["xopt"])[-1] == len(names), study
            assert all(np.allclose(a, b) for a, b in zip(s["axes"], ax)), (study, "grid")
    if other is not None and not (own is not None and own["quality"].get("converged")):
        if own is None:
            scan = other
        else:      # node by node the better of the two scans
            a, b = np.array(own["logl"], float), np.array(other["logl"], float)
            pick = np.fmax(a, b) == a
            scan = dict(logl=np.where(pick, a, b), xopt=np.where(pick[..., None], np.array(own["xopt"], float),
                                                                   np.array(other["xopt"], float)))
    emu = np.zeros(shape, bool)
    S = np.full(shape, np.nan)        # the network's own profile at every node: the scan's optimum (Figs. omega, omega_wc)
    if scan is not None:
        S, SX = np.array(scan["logl"], float), np.array(scan["xopt"], float)
        fill = ~true & np.isfinite(S)
        Z[fill], X[fill], emu[fill] = S[fill], SX[fill], True
    return dict(pars=pars, ax=ax, Z=Z, X=X, true=true, emu=emu, S=S)


def join_windows(ws):
    """Windows following each other in omega, as one surface; the other parameter on the grid of the window that reaches
    furthest in it, a window that ends earlier interpolated onto it (linearly; its true and emulator flags from the nearest
    node) and outside beyond its end."""
    pars = ws[0]["pars"]
    if len(ws) == 1:                                           # a single study, (A_osc, phi)
        return dict(ws[0], outside=np.zeros(ws[0]["Z"].shape, bool))
    k = pars.index("omega")
    o = 1 - k
    ref = max((w["ax"][o] for w in ws), key=lambda a: a[-1])
    parts = []
    for w in ws:
        assert w["pars"] == pars
        Z, X, T, E, S = (np.moveaxis(w[q], o, 0) for q in ("Z", "X", "true", "emu", "S"))  # the other parameter first
        src = w["ax"][o]
        if not np.allclose(src, ref):
            inside = ref <= src[-1] + 1e-12
            near = np.clip(np.rint(np.interp(ref, src, np.arange(len(src)))).astype(int), 0, len(src) - 1)
            Zi = np.full((len(ref), Z.shape[1]), np.nan)
            for c in range(Z.shape[1]):
                if np.isfinite(Z[:, c]).all():
                    Zi[inside, c] = np.interp(ref[inside], src, Z[:, c])
                else:
                    Zi[inside, c] = Z[near[inside], c]
            S = np.where(inside[:, None], S[near], np.nan)
            Z, X, T, E = Zi, X[near], T[near] & inside[:, None], E[near] & inside[:, None]
            out = np.repeat(~inside[:, None], Z.shape[1], axis=1)
        else:
            out = np.zeros(Z.shape, bool)
        parts.append(dict(om=w["ax"][k], Z=Z, X=X, T=T, E=E, O=out, S=S))
    om, Z, X, T, E, O, S = (parts[0][q] for q in ("om", "Z", "X", "T", "E", "O", "S"))
    for p in parts[1:]:
        assert math.isclose(om[-1], p["om"][0]), "the windows must share their edge column"
        a, b = Z[:, -1], p["Z"][:, 0]
        new = np.where(np.isnan(a), True, np.where(np.isnan(b), False, b > a))      # the better value at the shared column
        for arr, q in ((Z, "Z"), (X, "X"), (T, "T"), (E, "E"), (O, "O")):
            arr[:, -1] = np.where(new[(...,) + (None,) * (arr.ndim - 2)], p[q][:, 0], arr[:, -1])
        S[:, -1] = np.fmax(S[:, -1], p["S"][:, 0])
        om = np.concatenate([om, p["om"][1:]])
        Z, X = np.concatenate([Z, p["Z"][:, 1:]], 1), np.concatenate([X, p["X"][:, 1:]], 1)
        T, E, O, S = (np.concatenate([u, p[q][:, 1:]], 1) for u, q in ((T, "T"), (E, "E"), (O, "O"), (S, "S")))
    back = (lambda a: np.moveaxis(a, 0, o))
    axes = [None, None]
    axes[k], axes[o] = om, ref
    return dict(pars=pars, ax=axes, Z=back(Z), X=back(X), true=back(T), emu=back(E), outside=back(O), S=back(S))


def join_ranges(ws):
    """(A_osc, phi) from studies on the same phi grid with omega in different ranges, the first over the whole prior:
    node by node the largest value. A study whose grid ends below the first's in A_osc is interpolated linearly in A_osc
    onto it (where both neighbouring nodes have a value) and covers the nodes up to its end. A node is done ("true") once
    every study that covers it has its true value there; until then it shows the largest value so far and is shaded, also
    before a study has a snapshot at all. "valtrue": the value shown is a true value on a node of its study (for the best
    fit)."""
    ref = ws[0]
    pars, (A, P) = ref["pars"], ref["ax"]
    assert pars == ["A_osc", "phi"]
    Z, X = ref["Z"].copy(), ref["X"].copy()
    final, valtrue = ref["true"].copy(), ref["true"].copy()
    for w in ws[1:]:
        assert w["pars"] == pars and np.allclose(w["ax"][1], P) and math.isclose(w["ax"][0][0], A[0])
        a = w["ax"][0]
        if len(a) == len(A) and np.allclose(a, A):
            Zw, Xw, Tw, Vw, cover = w["Z"], w["X"], w["true"], w["true"], np.ones(Z.shape, bool)
        else:
            inside = A <= a[-1] + 1e-12
            idx = np.interp(A[inside], a, np.arange(len(a)))
            lo = np.floor(idx).astype(int)
            hi, f = np.minimum(lo + 1, len(a) - 1), (idx - np.floor(idx))[:, None]
            Zw, Xw = np.full(Z.shape, np.nan), np.full(X.shape, np.nan)
            Tw, Vw = np.zeros(Z.shape, bool), np.zeros(Z.shape, bool)
            Zw[inside] = (1 - f) * w["Z"][lo] + f * w["Z"][hi]             # NaN where either neighbour has none
            Xw[inside] = w["X"][np.where(f[:, 0] < 0.5, lo, hi)]
            Tw[inside] = w["true"][lo] & w["true"][hi]
            Vw[inside] = Tw[inside] & np.isclose(f, 0)                     # a true value only on a node of the study
            cover = np.repeat(inside[:, None], Z.shape[1], axis=1)
        better = np.isfinite(Zw) & ~(Zw <= Z)                              # also where Z has no value yet
        Z, X = np.where(better, Zw, Z), np.where(better[..., None], Xw, X)
        valtrue = np.where(better, Vw, valtrue)
        final &= ~cover | Tw
    return dict(pars=pars, ax=[A, P], Z=Z, X=X, true=final & np.isfinite(Z), emu=np.isfinite(Z) & ~final,
                valtrue=valtrue & np.isfinite(Z), outside=np.zeros(Z.shape, bool))


def profile_data(d, prof, windows, runs, names, bands, closed=False):
    """Write plots/data/profile2d_<d>_<pair>.json and omega200_<d>.json from the joined surfaces; return what the
    notes need."""
    # the best fit: the best true node as soon as there is one (the emulator's stand-in values can lie above it, by up to
    # 0.23 in ln L near omega = 51 in the stand-ins of 6 November; the figure draws such nodes at Delta chi^2 = 0)
    # (a true value of (A_osc, phi) counts while studies covering its node still run: "valtrue", join_ranges)
    # Until half the nodes are true (H0WC on 8 November: 4%), the best of all values, the emulator's included: the best
    # true node of a sparse set lay 1.4 below the emulator's maximum, and most of the surfaces were drawn at Delta chi^2 = 0.
    vt = {pair: s.get("valtrue", s["true"]) for pair, s in prof.items()}
    mostly_true = sum(int(v.sum()) for v in vt.values()) >= 0.5 * sum(s["Z"].size for s in prof.values())
    flat = [(s["Z"][idx], s["X"][idx]) for pair, s in prof.items()
            for idx in zip(*np.nonzero(np.isfinite(s["Z"]) & (vt[pair] if mostly_true else True)))]
    best_l, best_xv = max(flat, key=lambda t: t[0])
    best_x = {n: float(v) for n, v in zip(names, best_xv)}
    ao = prof["aosc_omega"]
    # the best fit without oscillation: the row A_osc = 0, its true values as soon as there are any, the emulator's until
    # then (the value does not depend on omega there)
    row0, true0 = ao["Z"][0], ao["true"][0] & np.isfinite(ao["Z"][0])
    lcdm = float(np.nanmax(row0[true0])) if true0.any() else float(np.nanmax(row0))
    for pair, s in prof.items():
        (DATA / f"profile2d_{d}_{pair}.json").write_text(json.dumps(dict(
            parameters=s["pars"], axes=[a.tolist() for a in s["ax"]], logl=np.where(np.isfinite(s["Z"]), s["Z"], None).tolist(),
            done=int(np.isfinite(s["Z"]).sum()), true_nodes=int(s["true"].sum()), emulator=(s["emu"] & ~s["true"]).tolist(),
            outside=s["outside"].tolist(), best=float(best_l), best_x=best_x, lcdm=lcdm, closed=bool(closed))))
    # Fig. omega: the improvement in chi^2 over the best fit without oscillation of the best node at each omega of the
    # (omega, phi) surface, whether that node is still the emulator's, and the bands of the combined emulator
    op = prof["omega_phi"]
    assert op["pars"] == ["omega", "phi"]
    Zc = np.where(np.isfinite(op["Z"]), op["Z"], -np.inf)
    at = Zc.argmax(axis=1)
    colmax = Zc[np.arange(len(at)), at]
    (DATA / f"omega200_{d}.json").write_text(json.dumps(dict(
        omega=op["ax"][0].tolist(), gain=[2 * (v - lcdm) if np.isfinite(v) else None for v in colmax],
        emulator=(op["emu"] & ~op["true"])[np.arange(len(at)), at].tolist(), lnp=float(best_l), lnp_lcdm=lcdm,
        lcdm_true=bool(true0.any()), bands=bands, closed=bool(closed),
        # the network's own profile (the scans' optima, best over phi at each omega), against the same best fit without
        # oscillation as the true one, so that the difference of the two curves is the network's error
        network_gain=[2 * (v - lcdm) if np.isfinite(v) else None for v in np.nanmax(np.where(np.isfinite(op["S"]), op["S"], np.nan), axis=1)])))
    n_op = int(op["true"].sum())
    n_true = sum(int(s["true"].sum()) for s in prof.values())
    # the joined grids (a shared edge column once), without the nodes outside the windows' ranges (white in the figures)
    n_all = sum(int((~s["outside"]).sum()) for s in prof.values())
    snap = max((e["mtime"] for ws in windows.values() for w in ws for e in study_nodes(w[0]).values()), default=None)
    when = __import__("time").strftime("%-d %B %H:%M", __import__("time").localtime(snap)) if snap else "?"
    counts = f"{n_true:,} of {n_all:,}".replace(",", "\\,")
    ends = [yaml.safe_load((runs / w[2]).read_text())["ranges"]["A_osc"][1] for w in windows["aosc_omega"]]
    cut = [f"{a:g}" for a in ends if a < max(ends)]
    assert all(a < max(ends) for a in ends[:len(cut)]), "only the lowest windows end early"
    return dict(best_l=best_l, best_x=best_x, n_op=n_op, n_true=n_true, n_all=n_all, when=when, cut=cut, op_size=op["Z"].size)


def cut_sentence(cut):
    """The caption's sentence on the windows of (A_osc, omega) whose grid ends lower (white above)."""
    return ("" if not cut else
            f"In the lowest window of $(\\Aosc, \\omega)$ the grid ends at $\\Aosc = {cut[0]}$ (white above)." if len(cut) == 1 else
            f"In the {['', '', 'two', 'three'][len(cut)]} lowest windows of $(\\Aosc, \\omega)$ the grid ends at "
            f"$\\Aosc = {', '.join(cut[:-1])}$ and {cut[-1]}, respectively (white above).")


prof = {pair: (join_ranges if pair == "aosc_phi" else join_windows)([profile_window(*w) for w in ws])
        for pair, ws in PROF_WINDOWS.items()}
if any(np.isfinite(s["Z"]).any() for s in prof.values()):     # a surface without a snapshot yet is all "not done"
    info = profile_data("pp", prof, PROF_WINDOWS, PROF_RUNS, PROF_NAMES,
                        json.loads((POST_COMB / "combine.json").read_text())["bands"])
    best_l, best_x, n_op, n_true, n_all, when = (info[k] for k in ("best_l", "best_x", "n_op", "n_true", "n_all", "when"))
    op = prof["omega_phi"]
    counts = f"{n_true:,} of {n_all:,}".replace(",", "\\,")
    state = (f"all {n_all:,} nodes calculated with the true likelihood".replace(",", "\\,") if n_true == n_all else
             f"as of {when}: {counts} nodes calculated with the true likelihood and final; the others (shaded) show the "
             "emulator's value or, in $(\\Aosc, \\phi)$, a true value that the studies still running may raise")
    num("PPprofnote", f"\\draft{{Profiles of the combined emulator before zone 1 was split, {state}. The posterior on "
                      f"the diagonal is that of the combined emulator of the {POST_PARTS} parts.}}")
    num("PPprofwindows", "four")
    num("PPprofcut", cut_sentence(info["cut"]))
    counts = f"{n_op:,} of {op['Z'].size:,}".replace(",", "\\,")
    # once every node of (omega, phi) is done there is no emulator value left to mention
    n_grid = f"{op['Z'].size:,}".replace(",", "\\,")
    done = (f"all {n_grid} nodes of $(\\omega, \\phi)$ calculated with the true likelihood" if n_op == op["Z"].size else
            f"{counts} nodes of $(\\omega, \\phi)$ calculated with the true likelihood, the others show the emulator's "
            "value (shaded)")
    num("PPomeganote", f"\\draft{{The profile as of {when}: {done}; the posterior of the combined emulator of the "
                       f"{POST_PARTS} parts.}}")
    z = np.load(POST_COMB / "analysis/posterior_it_0.npz", allow_pickle=True)
    assert [str(n) for n in z["names"]] == PROF_NAMES
    pr = {n: [cobaya["pp"]["params"][n]["prior"]["min"], cobaya["pp"]["params"][n]["prior"]["max"]] for n in PROF_NAMES
          if isinstance(cobaya["pp"]["params"].get(n), dict) and "min" in (cobaya["pp"]["params"][n].get("prior") or {})}
    np.savez(DATA / "posterior200_pp.npz", samples=z["x"].astype(float), weights=np.ones(len(z["x"])),
             names=np.array(PROF_NAMES), ranges=json.dumps(pr))
    print("profiles of PDP+:", {p: (int(s["true"].sum()), int(np.isfinite(s["Z"]).sum()), s["Z"].shape) for p, s in prof.items()},
          "best", round(float(best_l), 2), {k: round(best_x[k], 3) for k in ("A_osc", "omega", "phi")})

# ---------------------------------------------------------------- the posterior of H0WC (Fig. triangle_wc)
# The chain of the newest H0WC combined emulator (8 November: the eight parts, results/<time>_axion_wc_combined200), as
# plots/data/posterior200_wc.npz for its corner plot; its own parameters (the H0WC likelihoods' nuisances) and priors.
# The weights of its region cells, which share the posterior between the parts, are all marked unreliable by CLiENT
# (bridge efficiencies 0.01-0.10); WCpostnote says so until they have been checked against the nested sampling.
WC_COMBS = sorted(d for d in RES.glob("2026*_axion_wc_combined200") if (d / "analysis/posterior_it_0.npz").exists())
if WC_COMBS:
    z = np.load(WC_COMBS[-1] / "analysis/posterior_it_0.npz", allow_pickle=True)
    names = [str(n) for n in z["names"]]
    pr = {n: [cobaya["wc"]["params"][n]["prior"]["min"], cobaya["wc"]["params"][n]["prior"]["max"]] for n in names
          if isinstance(cobaya["wc"]["params"].get(n), dict) and "min" in (cobaya["wc"]["params"][n].get("prior") or {})}
    np.savez(DATA / "posterior200_wc.npz", samples=z["x"].astype(float), weights=np.ones(len(z["x"])),
             names=np.array(names), ranges=json.dumps(pr))
    cells = json.loads(str(z["sampling_json"])).get("cells", [])
    combine = json.loads((WC_COMBS[-1] / "combine.json").read_text())
    parts = len(combine["parts"])
    unreliable = [c for c in cells if c.get("efficiency", 1) < 0.1]
    cells_text = (f"all {len(cells)} of its" if len(unreliable) == len(cells) else f"{len(unreliable)} of its {len(cells)}")
    # the check of the cell weights: the share of the posterior in each part's range in the emulator's chain against that
    # in the weighted chain of its nested sampling (BEST, evidence/), which weights the whole emulator at once
    best_chain = WC_COMBS[-1] / "evidence/evidence_it_0_nlive2048.txt"
    check = ""
    if unreliable and best_chain.exists():
        bnames = [l.split()[0].rstrip("*") for l in open(best_chain.with_suffix(".paramnames"))]
        c = np.loadtxt(best_chain)
        w_b, om_b = c[:, 0], c[:, 2 + bnames.index("omega")]
        om_m = z["x"][:, names.index("omega")]
        edges = [1.0] + [0.5 * (a + b) for a, b in combine["bands"]] + [200.0]
        shares = [(w_b[(om_b >= lo) & (om_b < hi)].sum() / w_b.sum(), np.mean((om_m >= lo) & (om_m < hi)))
                  for lo, hi in zip(edges[:-1], edges[1:])]
        worst = max(abs(a - b) for a, b in shares[1:])
        check = (f" Checked against the nested sampling of the emulator, which weights it as a whole: the share of the posterior "
                 f"in the range of each part agrees within {worst:.3f}, except below $\\omega = {edges[1]:.3g}$ "
                 f"({100 * shares[0][1]:.1f}\\% against {100 * shares[0][0]:.1f}\\%).")
    num("WCpostnote", f"\\draft{{The combined emulator of the {['', '', '', '', '', '', '', 'seven', 'eight'][parts]} parts. "
        + (f"The weights of {cells_text} region cells, which share the posterior between the parts, are marked unreliable "
           "(Appendix~\\ref{app:split})." + (check or " To be checked against the nested sampling.") if unreliable else "") + "}")
    print(f"posterior of H0WC: {len(z['x'])} samples from {WC_COMBS[-1].name}, {len(unreliable)} of {len(cells)} cells unreliable")
else:
    num("WCpostnote", "\\draft{The combined emulator of \\HWC{} is not yet available.}")

# ---------------------------------------------------------------- the 2D profiles of H0WC (Figs. omega_wc, profile2d_wc)
# As for PDP+ (above), from the studies of runs/axion_wc_combined200 (make_profiles200_wc.py, 7 November), on the H0WC
# combined emulator of the eight parts (results/20261107_084828_axion_wc_combined200). Windows 1 and 2 of (A_osc, omega) are
# the studies with A_osc narrowed to 0.04 and 0.07 (make_profiles200_wc_narrow.py, 8 November: with 0.06 and 0.09 the scans
# put node optima beyond the region's edge); (A_osc, phi) is the study with omega free joined with the five that search omega
# only around the highest maxima of the H0WC scan (windows/). Writes plots/data/profile2d_wc_<pair>.json and
# omega200_wc.json, replacing the omega <= 60 stand-ins, once a study of each plane has a snapshot.
WC_RUNS = RUNS / "axion_wc_combined200"
WC_WINDOWS = {
    "aosc_omega": [[(f"axion_wc200_aosc_omega_w{k}_{tag}_legacy", f"axion_wc200_aosc_omega_w{k}_{tag}",
                     f"fallback/profiles_aosc_omega_w{k}_{tag}_legacy.yaml"),
                    (f"axion_wc200_aosc_omega_w{k}_{tag}", None, f"profiles_aosc_omega_w{k}_{tag}.yaml")]
                   for k, tag in ((1, "a040"), (2, "a070"))]
                  + [[(f"axion_wc200_aosc_omega_w{k}_legacy", f"axion_wc200_aosc_omega_w{k}", f"fallback/profiles_aosc_omega_w{k}_legacy.yaml"),
                      (f"axion_wc200_aosc_omega_w{k}", None, f"profiles_aosc_omega_w{k}.yaml")] for k in (3, 4)],
    "omega_phi": [(f"axion_wc200_omega_phi_w{k}", None, f"profiles_omega_phi_w{k}.yaml") for k in range(1, 5)],
    "aosc_phi": [("axion_wc200_aosc_phi", None, "profiles_aosc_phi.yaml")]
                + [[(f"axion_wc200_aosc_phi_{t}_legacy", None, f"fallback/profiles_aosc_phi_{t}_legacy.yaml"),
                    (f"axion_wc200_aosc_phi_{t}", None, f"windows/profiles_aosc_phi_{t}.yaml")]
                   for t in ("om82", "om144", "om158", "om169", "om195")],
}
WC_WINDOWS = {pair: [chosen(w) for w in ws] for pair, ws in WC_WINDOWS.items()}


def jobs_stopped(studies):
    """Whether no study of the list still has a job running at the last sync: each is complete (its status.txt), or the
    newest job log that writes to it (logs/profiles_<job>.out) ends, with "done", a time limit, a cancellation or an error.
    Returns that and the studies still running."""
    newest = {}
    for log in LOGS.glob("profiles_[0-9]*.out"):
        with log.open(errors="replace") as f:
            m = re.search(r"output results_profiles/(\S+)", f.read(3000))
        job = int(log.stem.split("_")[1])
        if m and job > newest.get(m.group(1), (0, None))[0]:
            newest[m.group(1)] = (job, log)
    running = []
    for study in studies:
        status = LOGS / f"profiles_{study}_surface" / "status.txt"
        if status.exists() and status.read_text().strip() == "complete":
            continue
        job, log = newest.get(study, (None, None))
        if log is None:
            running.append(study)
            continue
        with log.open("rb") as f:
            f.seek(max(0, log.stat().st_size - 4000))
            tail = f.read().decode(errors="replace")
        if not re.search(r"\sdone\s*$|DUE TO TIME LIMIT|CANCELLED|error:", tail):
            running.append(study)
    return not running, running


# Closed (Steen, 12 November 2026): once the jobs of the H0WC studies have stopped at their time limit, the nodes whose
# refinement has not finished keep the value of the emulator, and no further jobs are run. They all lie at large amplitude,
# far below the best fit, where the exact profile would not change the figures visibly. Then every profile number of
# H0WC is final, the profile of A_osc uses the emulator's value at those nodes as the figure does (the best fit stays the
# best true node), the captions say how many nodes show the emulator's value, and the figures are not shaded.
# BUILD200_CLOSE_WC=1 in the environment closes them regardless (for a preview).
WC_CLOSED, WC_RUNNING = jobs_stopped(sorted({w[0] for ws in WC_WINDOWS.values() for w in ws}))
if os.environ.get("BUILD200_CLOSE_WC") == "1":
    WC_CLOSED = True
# On 13 November 2026 the profiles were closed by hand, from the snapshot of 15:07 (34 697 of 35 000 nodes). The jobs then
# finished all 35 000 nodes (synced 14 November): the studies are final without closing, and the closing note is empty.
print(f"H0WC profiles closed: {WC_CLOSED}" + ("" if WC_CLOSED else f" (jobs running: {', '.join(WC_RUNNING)})"))
if WC_COMBS and all(any(in_snapshot(w[0]) or (w[1] and in_snapshot(w[1])) for w in ws) for ws in WC_WINDOWS.values()):
    WC_NAMES = [p["name"] for p in json.loads((WC_COMBS[-1] / "metadata.json").read_text())["parameters"]]
    prof_wc = {pair: (join_ranges if pair == "aosc_phi" else join_windows)(
                   [profile_window(*w, runs=WC_RUNS, names=WC_NAMES) for w in ws]) for pair, ws in WC_WINDOWS.items()}
    info = profile_data("wc", prof_wc, WC_WINDOWS, WC_RUNS, WC_NAMES, closed=WC_CLOSED, bands=
                        json.loads((WC_COMBS[-1] / "combine.json").read_text())["bands"])
    counts = f"{info['n_true']:,} of {info['n_all']:,}".replace(",", "\\,")
    # the same test as profile_results: all nodes inside the region calculated with the true likelihood. The two notes
    # are then empty: unlike PDP+, the profiles and the posterior of H0WC are of the same emulator (11 November, Steen)
    wc_profiles_final = all(bool(s["true"][~s["outside"]].all()) for s in prof_wc.values())
    # closed: how many nodes keep the emulator's value, from which amplitude, and how far below the best fit
    open_nodes = [(float(s["ax"][0][i]), float(2 * (info["best_l"] - s["Z"][i, j])))
                  for s in (prof_wc["aosc_omega"], prof_wc["aosc_phi"]) for i, j in zip(*np.nonzero(~s["true"] & ~s["outside"]))]
    closed_note = ("" if not open_nodes else
                   f"{big(info['n_all'] - info['n_true'])} of the {big(info['n_all'])} nodes, all in the planes with $\\Aosc$ at "
                   f"$\\Aosc \\geq {min(a for a, _ in open_nodes):.2f}$ and more than "
                   f"{math.floor(10 * min(d for _, d in open_nodes)) / 10:.1f} below the best fit in $\\chi^2$, show the value of "
                   "the emulator, since their refinement with the true likelihood was not completed.")
    num("WCprofnote", "" if wc_profiles_final else closed_note if WC_CLOSED else
                      f"\\draft{{Profiles of the combined emulator of the eight parts, as of {info['when']}: {counts} nodes "
                      "calculated with the true likelihood and final; the others (shaded) show the emulator's value or, in "
                      "$(\\Aosc, \\phi)$, a true value that the studies still running may raise.}")
    num("WCprofcut", cut_sentence(info["cut"]))
    n_grid = f"{info['op_size']:,}".replace(",", "\\,")
    done = (f"all {n_grid} nodes of $(\\omega, \\phi)$ calculated with the true likelihood" if info["n_op"] == info["op_size"] else
            f"{info['n_op']:,}".replace(",", "\\,") + f" of {n_grid} nodes of $(\\omega, \\phi)$ calculated with the true "
            "likelihood, the others show the emulator's value (shaded)")
    num("WComeganote", "" if wc_profiles_final or WC_CLOSED else
                       f"\\draft{{The profile as of {info['when']}: {done}; the posterior of the combined emulator of the eight parts.}}")
    print("profiles of H0WC:", {p: (int(s["true"].sum()), int(np.isfinite(s["Z"]).sum()), s["Z"].shape) for p, s in prof_wc.items()},
          "best", round(float(info["best_l"]), 2), {k: round(info["best_x"][k], 3) for k in ("A_osc", "omega", "phi")})
else:
    for k in ("WCprofnote", "WComeganote"):
        num(k, "\\draft{Stand-in from the analysis up to $\\omega = 60$.}")
    num("WCprofcut", "")

# ---------------------------------------------------------------- the results for omega <= 200 (Secs. 4-6, Appendix A)
# The numbers of the results sections, from the omega <= 200 runs (9 November 2026), replacing the omega <= 60 stand-ins:
#   - the posteriors: the chains of the combined emulators (PDP+: POST_COMB, the seven parts; H0WC: the eight parts), the
#     derived parameters from each part's own derived emulator in the range in which its network is used (between the
#     middles of the bands), Table tab:params;
#   - the comparison with the true likelihood: the kit's true check of each combined emulator (analysis/true_check_it_0.npz,
#     500 posterior samples);
#   - the evidence: BEST of each combined emulator (evidence/), in Cobaya's convention for A_osc in [0, 0.5], shifted by
#     ln 5 to the paper's prior [0, 0.1]; the four seeds and the featureless model (A_osc in [0, 1e-4], also with omega
#     inside one part at a time) come from grendel/queue_axion200_evidence.sh. Until those are synced, ln B is the
#     Savage-Dickey estimate from the chain, marked as preliminary (\prelim);
#   - the profiles: the joined surfaces of the figures (prof, prof_wc above) and the node files of their studies. The
#     numbers of H0WC are marked as preliminary until all its nodes are true.
# The look-elsewhere effect (Gross and Vitells, arXiv:1005.1891): at fixed omega the improvement 2 (ln L - ln L_0) over the
# featureless best fit follows chi^2 with two degrees of freedom (amplitude and phase) without an oscillation; with N(u0)
# upcrossings of the level u0 by the profile in omega, the expected number of maxima above the best fit's improvement u is
# N(u0) (u / u0)^{1/2} exp(-(u - u0) / 2), and the global p-value is at most the local one plus that number.
from scipy.special import logsumexp            # noqa: E402
from scipy.stats import chi2 as chi2_dist       # noqa: E402

DERIVED = ["H0", "sigma8", "S8", "Omega_m", "rs_drag"]
DLABEL = {"H0": r"H_0\;[\mathrm{km\,s^{-1}\,Mpc^{-1}}]", "sigma8": r"\sigma_8", "S8": r"S_8", "Omega_m": r"\Omega_\mathrm{m}",
          "rs_drag": r"r_\mathrm{drag}\;[\mathrm{Mpc}]"}
PEAK = (49.0, 52.5)                             # the peak of the PDP+ best fit in the posterior
LEE_LEVELS = (0.5, 1.0, 2.0, 3.0)               # u0 of the upcrossing count; the text quotes u0 = 1 and the range
COMB = {"PP": POST_COMB, "WC": WC_COMBS[-1] if WC_COMBS else None}


def prelim(v, final):
    """A number that is not final yet: in red in the PDF (\\prelim, main-monodromy200.tex)."""
    return v if final else f"\\prelim{{{v}}}"


def chain_with_derived(tag, comb):
    """The combined emulator's chain, with its derived parameters from each part's derived emulator."""
    from derived.emulator import load_derived_model
    z = np.load(comb / "analysis/posterior_it_0.npz", allow_pickle=True)
    names = [str(n) for n in z["names"]]
    x = z["x"].astype(float)
    combine = json.loads((comb / "combine.json").read_text())
    edges = [-np.inf] + [0.5 * (a + b) for a, b in combine["bands"]] + [np.inf]
    om = x[:, names.index("omega")]
    der = None
    for k, q in enumerate(combine["parts"]):
        run = RES / Path(q["run"]).name
        it = int(re.search(r"region_it_(\d+)\.npz", next(a for a in q["artifacts"] if "region_it_" in a)).group(1))
        dnames = [p["name"] for p in json.loads((run / "derived_data/derived_parameters.json").read_text())["parameters"]]
        if der is None:
            der, dnames0 = np.full((len(x), len(dnames)), np.nan), dnames
        assert dnames == dnames0
        sel = np.nonzero((om >= edges[k]) & (om < edges[k + 1]))[0]
        if len(sel):
            m = load_derived_model(run / f"derived_data/derived_model_it_{it}.keras")
            der[sel] = np.concatenate([np.asarray(m(x[sel[i:i + 8192]].astype(np.float32), training=False))
                                       for i in range(0, len(sel), 8192)])
    assert np.isfinite(der).all(), tag
    return pd.DataFrame(np.column_stack([x, der]), columns=names + dnames0), z


def interval(v):
    """Mean and the 68% interval of the samples, as Table tab:params gives it (symmetric when the two sides agree)."""
    mean = float(np.mean(v))
    lo, hi = np.quantile(v, [0.16, 0.84])
    sd = float(np.std(v))
    digits = max(0, 2 - int(math.floor(math.log10(sd))) - 1) if sd > 0 else 4
    up, dn = hi - mean, mean - lo
    if abs(up - dn) < 0.1 * sd:
        return f"${mean:.{digits}f}\\pm {sd:.{digits}f}$"
    return f"${mean:.{digits}f}^{{+{up:.{digits}f}}}_{{-{dn:.{digits}f}}}$"


chains, raw = {}, {}
for tag, comb in COMB.items():
    if comb is not None:
        chains[tag], raw[tag] = chain_with_derived(tag, comb)

# -- posterior numbers
for tag, ch in chains.items():
    om, A = ch.omega.to_numpy(), ch.A_osc.to_numpy()
    low_edge = 0.5 * sum(BANDS["pp" if tag == "PP" else "wc"][0])
    num(f"{tag}nchain", big(len(ch)))
    num(f"{tag}peakshare", f"{100 * np.mean((om >= PEAK[0]) & (om < PEAK[1])):.0f}")
    num(f"{tag}highshare", f"{100 * np.mean(om >= 60):.0f}")
    num(f"{tag}lowshare", f"{100 * np.mean(om < low_edge):.1f}")
    num(f"{tag}Aup", f"{np.quantile(A, 0.95):.3f}")
    num(f"{tag}Aupsixty", f"{np.quantile(A[om <= 60], 0.95):.3f}")
    num(f"{tag}Ahighshare", f"{100 * np.mean(A > 0.03):.0f}")
    num(f"{tag}Ahighhigh", f"{100 * np.mean(om[A > 0.03] > 150):.0f}")
    num(f"{tag}Ahighlow", f"{100 * np.mean(om[A > 0.03] < 6):.1f}")
    # the volume effect in omega (Sec. results:profiles): the 95% upper limit on A_osc at fixed omega, i.e. the range of
    # amplitudes over which the posterior of omega integrates, below omega = 2, at 3 <= omega < 10 and above omega = 180
    # (the ranges are written out in the text)
    for name, sel in (("AupOmLow", om < 2), ("AupOmMid", (om >= 3) & (om < 10)), ("AupOmHigh", om >= 180)):
        num(f"{tag}{name}", f"{np.quantile(A[sel], 0.95):.1g}")
num("WCppshare", f"{100 * np.mean((chains['WC'].omega >= PEAK[0]) & (chains['WC'].omega < PEAK[1])):.1f}")
num("evenshare", f"{100 * (PEAK[1] - PEAK[0]) / (200 - 1):.1f}")

# the region cells of the combined emulators: how many CLiENT marks unreliable, and the check of the shares of the parts
# against the weighted chain of BEST, which integrates the emulator as a whole
for tag, comb in COMB.items():
    z = raw[tag]
    cells = json.loads(str(z["sampling_json"])).get("cells", [])
    num(f"{tag}cells", len(cells))
    num(f"{tag}cellsbad", sum(1 for c in cells if c.get("efficiency", 1) < 0.1))
    names = [str(n) for n in z["names"]]
    combine = json.loads((comb / "combine.json").read_text())
    best_chain = comb / "evidence/evidence_it_0_nlive2048.txt"
    bnames = [l.split()[0].rstrip("*") for l in open(best_chain.with_suffix(".paramnames"))]
    c = np.loadtxt(best_chain)
    w_b, om_b = c[:, 0], c[:, 2 + bnames.index("omega")]
    om_m = z["x"][:, names.index("omega")]
    edges = [1.0] + [0.5 * (a + b) for a, b in combine["bands"]] + [200.0]
    diff = [abs(w_b[(om_b >= lo) & (om_b < hi)].sum() / w_b.sum() - np.mean((om_m >= lo) & (om_m < hi)))
            for lo, hi in zip(edges[:-1], edges[1:])]
    num(f"{tag}cellcheck", f"{100 * max(diff):.1f}")
    print(f"{tag}: region cells {len(cells)}, unreliable {numbers[tag + 'cellsbad']}, shares of the parts against BEST: "
          + ", ".join(f"{100 * d:.1f}" for d in diff))

# -- Table tab:params
def entry(tag, key):
    if key in ("omega", "phi"):
        return "multimodal"
    if key == "A_osc":
        return f"$< {np.quantile(chains[tag].A_osc, 0.95):.3f}$"
    return interval(chains[tag][key].to_numpy())


with open(TABLES / "params.tex", "w") as f:
    f.write("\\begin{tabular}{rlcc}\n\\cmidrule[\\heavyrulewidth](lr){2-4}\n"
            "& \\textbf{Parameter} & \\textbf{\\PDP{}} & \\textbf{\\HWC{}} \\\\\n\\cmidrule(lr){2-4}\n")
    for group, keys in (("$\\Lambda$CDM", COSMO), ("Oscillation", OSC), ("Derived", DERIVED)):
        for k, key in enumerate(keys):
            lab = LABEL.get(key, DLABEL.get(key))
            f.write(f"{group if k == 0 else ''} & ${lab}$ & {entry('PP', key)} & {entry('WC', key)} \\\\\n")
        f.write("\\cmidrule(lr){2-4}\n" if group != "Derived" else "")
    f.write("\\cmidrule[\\heavyrulewidth](lr){2-4}\n\\end{tabular}\n")

# -- the comparison with the true likelihood (Appendix app:accuracy)
for tag, comb in COMB.items():
    tc = np.load(comb / "analysis/true_check_it_0.npz", allow_pickle=True)
    names = [str(n) for n in tc["names"]]
    tx, dt = tc["x"].astype(float), tc["truth"] - tc["network"]
    assert np.isfinite(dt).all(), tag
    w = np.exp(dt - dt.max())
    w /= w.sum()
    num(f"{tag}checkn", len(dt))
    num(f"{tag}checkrms", f"{np.sqrt(np.mean(dt ** 2)):.2f}")
    num(f"{tag}checkmax", f"{np.abs(dt).max():.1f}")
    num(f"{tag}checkcorr", f"{logsumexp(dt) - np.log(len(dt)):.3f}")
    num(f"{tag}checkess", f"{1 / np.sum(w ** 2):.0f}")
    tom = tx[:, names.index("omega")]
    inside = (tom >= PEAK[0]) & (tom < PEAK[1])
    num(f"{tag}checkpeak", f"{np.sqrt(np.mean(dt[inside] ** 2)):.2f}")
    num(f"{tag}checkhigh", f"{np.sqrt(np.mean(dt[tom >= 60] ** 2)):.2f}")
    idx = [names.index(k) for k in COSMO + OSC]
    shift = max(abs((w @ tx[:, k] - tx[:, k].mean()) / tx[:, k].std()) for k in idx)
    width = max(abs(np.sqrt(w @ (tx[:, k] - w @ tx[:, k]) ** 2) / tx[:, k].std() - 1) for k in idx)
    num(f"{tag}checkshift", f"{shift:.2f}")
    num(f"{tag}checkwidth", f"{100 * width:.0f}")
    depth = float(json.loads((comb / "analysis/true_check.json").read_text())["networks"][0]["median_depth_below_best"])
    num(f"{tag}checkdepth", f"{depth:.1f}")

# -- the evidence (Sec. results:comparison, Appendix app:evidence)
evidence_final = True
lnb = {}
for tag, comb in COMB.items():
    load = lambda t: json.loads((comb / f"evidence/evidence_it_0_nlive2048{t}.json").read_text())
    a_prior = next(q for q in load("")["priors"] if q["name"] == "A_osc")
    shift = math.log((a_prior["upper"] - a_prior["lower"]) / 0.1)
    osc = [load("")] + [load(f"_seed{s}") for s in (43, 44, 45)
                        if (comb / f"evidence/evidence_it_0_nlive2048_seed{s}.json").exists()]
    lnz1 = np.array([e["lnZ"] for e in osc]) + shift
    if len(osc) > 1:
        err1 = lnz1.std(ddof=1) / math.sqrt(len(lnz1))
        num(f"{tag}lnZscatter", f"{lnz1.std(ddof=1):.2f}")
    else:
        err1 = osc[0]["sigma_lnZ"]
        num(f"{tag}lnZscatter", "\\prelim{?}")
    num(f"{tag}nseeds", len(osc))
    num(f"{tag}lnZ", prelim(f"{lnz1.mean():.2f}", len(osc) == 4))
    num(f"{tag}lnZerr", prelim(f"{err1:.2f}", len(osc) == 4))
    num(f"{tag}lnZsingle", f"{osc[0]['sigma_lnZ']:.2f}")
    # the Savage-Dickey density ratio at A_osc = 0 for the prior [0, 0.1]; the density near 0 is flat to 2% for eps from
    # 5e-4 to 3e-3
    A = chains[tag].A_osc.to_numpy()
    eps = 1e-3
    sd = -math.log(np.mean(A < eps) / eps * 0.1)
    num(f"{tag}lnBsd", f"{sd:.1f}")
    num(f"{tag}lnBhalf", f"{math.log(2 * np.mean(A < 0.05)):.1f}")          # ln B higher by this for A_osc in [0, 0.05]
    lcdm_tags = ["lcdm", "lcdm_seed43", "lcdm_seed44", "lcdm_seed45"]
    have = [t for t in lcdm_tags if (comb / f"evidence/evidence_it_0_nlive2048_{t}.json").exists()]
    if have:
        runs = [load(f"_{t}") for t in have]
        for e in runs:
            q = next(p for p in e["priors"] if p["name"] == "A_osc")
            assert (q["lower"], q["upper"]) == (0.0, 1e-4) and e["n_live"] == 2048
        lnz0 = np.array([e["lnZ"] for e in runs])
        err0 = lnz0.std(ddof=1) / math.sqrt(len(lnz0)) if len(lnz0) > 1 else runs[0]["sigma_lnZ"]
        final = len(have) == 4 and len(osc) == 4
        num(f"{tag}lnZlcdm", prelim(f"{lnz0.mean():.2f}", final))
        num(f"{tag}lnZlcdmerr", prelim(f"{err0:.2f}", final))
        lnb[tag] = (lnz1.mean() - lnz0.mean(), math.hypot(err0, err1), final)
        parts = sorted((comb / "evidence").glob("evidence_it_0_nlive2048_lcdm_part*.json"))
        if len(parts) >= 2:
            v = [json.loads(p.read_text())["lnZ"] for p in parts]
            num(f"{tag}lcdmbands", f"{max(v) - min(v):.1f}")
        else:
            num(f"{tag}lcdmbands", "\\prelim{?}")
    else:
        lnb[tag] = (sd, float("nan"), False)
        num(f"{tag}lnZlcdm", "\\prelim{?}")
        num(f"{tag}lnZlcdmerr", "\\prelim{?}")
        num(f"{tag}lcdmbands", "\\prelim{?}")
    b, e, final = lnb[tag]
    evidence_final &= final
    num(f"{tag}lnB", prelim(f"{b:.1f}", final))
    num(f"{tag}lnBerr", prelim(f"{e:.1f}" if np.isfinite(e) else "?", final))
num("lnBnote", "" if evidence_final else
    " \\draft{Until the four seeds and the evidences of the featureless model are synced (\\texttt{grendel/queue\\_axion200\\_evidence.sh}), "
    "$\\ln Z$ is that of a single run and $\\ln B$ the Savage--Dickey estimate (red).}")


# -- the profiles (Secs. results:profiles, actspt:profiles, Appendix app:profiles)
def node_records(studies, names):
    """Every node of the studies: (logl, direct_logl, reason, calls, x as a dict, study)."""
    out = []
    for st in studies:
        for (i, j), e in sorted(study_nodes(st).items()):
            if e["logl"] is None or not np.isfinite(e["logl"]):
                continue
            out.append(dict(i=i, j=j, logl=e["logl"], direct=e["direct_logl"], reason=e["reason"],
                            calls=e["backend_calls"], x=dict(zip(names, e["x"])), study=st))
    return out


def lee(gain, u):
    """The look-elsewhere estimate from a profile in omega (gain: improvement over the featureless best fit)."""
    g = np.where(np.isfinite(gain), gain, -np.inf)
    res = {}
    for u0 in LEE_LEVELS:
        n_up = int(np.sum((g[:-1] < u0) & (g[1:] >= u0)))
        res[u0] = (n_up, n_up * math.sqrt(u / u0) * math.exp(-(u - u0) / 2))
    return res


SETTLED_MARGIN = 2.0      # chi^2 below the best fit that the nodes not final yet must keep (profile_results, "settled")


def profile_results(tag, prof_d, windows, names, closed=False):
    """The numbers of a dataset's profiles; returns the best fit and the featureless one. closed: the studies have stopped
    and the nodes not refined keep the emulator's value (WC_CLOSED)."""
    final = closed or all(bool(s["true"][~s["outside"]].all()) for s in prof_d.values())
    P = lambda v: prelim(v, final)
    vt = {pair: s.get("valtrue", s["true"]) for pair, s in prof_d.items()}
    flat = [(s["Z"][idx], s["X"][idx]) for pair, s in prof_d.items()
            for idx in zip(*np.nonzero(np.isfinite(s["Z"]) & vt[pair]))]
    best_l, best_xv = max(flat, key=lambda t: t[0])
    bx = dict(zip(names, best_xv))
    ao, op, ap = prof_d["aosc_omega"], prof_d["omega_phi"], prof_d["aosc_phi"]
    # Settled (12 November 2026, Steen): while the studies of a dataset run, a number is shown final once the nodes that
    # are still being refined can no longer change it. Those nodes lie in the two planes with A_osc. Settled requires the
    # (omega, phi) plane to be complete and every node that is not final yet to lie more than SETTLED_MARGIN below the
    # best fit in chi^2 by its present value (a node without a value keeps everything red). Of the 32 264 finished nodes
    # of H0WC on 12 November, none whose emulator start lay more than 2 below the best fit ended within 1.86 of it, and
    # no node that ended within 6 of it gained more than 1.78 in the refinement. A number of the profile of A_osc is
    # settled only where it lies below the lowest amplitude of those nodes (pend_A), and the statistics of the nodes and
    # what the nodes at large amplitude decide stay with `final`.
    pend = [(float(s["ax"][0][np.nonzero(m)[0]].min()), float(np.min(2 * (best_l - s["Z"][m]))))
            for s in (ao, ap) for m in [~s["true"] & ~s["outside"]] if m.any()]
    # and where in omega they lie, for the maxima of the profile in omega (each made of the nodes within 1.5 of it):
    # in (A_osc, omega) the node's column, in (A_osc, phi) the frequency of its present optimum
    i_om = names.index("omega")
    pend_om = [(float(s["ax"][1][j]) if s is ao else float(s["X"][i, j][i_om]), float(2 * (best_l - s["Z"][i, j])))
               for s in (ao, ap) for i, j in zip(*np.nonzero(~s["true"] & ~s["outside"]))]
    pend_A = min((a for a, _ in pend), default=np.inf)
    pend_d = min((d for _, d in pend), default=np.inf)
    settled = final or (bool(op["true"][~op["outside"]].all()) and pend_d > SETTLED_MARGIN)
    Ps = lambda v: prelim(v, settled)
    print(f"{tag} profiles: final {final}, settled {settled} (nodes not final from A_osc = {pend_A:.4g}, "
          f"at least {pend_d:.2f} below the best fit)")
    row0 = ao["true"][0] & np.isfinite(ao["Z"][0])
    lcdm = float(np.nanmax(ao["Z"][0][row0]))
    dchi = 2 * (best_l - lcdm)
    num(f"{tag}lcdmspread", Ps(f"{2 * np.ptp(ao['Z'][0][row0]):.2f}"))
    num(f"{tag}dchi", Ps(f"{dchi:.1f}"))
    num(f"{tag}aic", Ps(f"{-dchi + 6:.1f}"))
    num(f"{tag}bestomega", Ps(f"{bx['omega']:.1f}"))
    num(f"{tag}bestomegaround", Ps(f"{bx['omega']:.0f}"))           # "omega ~ 51" in the abstract and the conclusions
    num(f"{tag}bestA", Ps(f"{bx['A_osc']:.3f}"))
    num(f"{tag}bestphi", Ps(f"{bx['phi']:.2f}".replace("-0.00", "0.00")))
    # the profile in omega: the best node over phi at each omega of (omega, phi), true values only
    Zop = np.where(op["true"] & np.isfinite(op["Z"]), op["Z"], np.nan)
    om_ax = op["ax"][0]
    gain = 2 * (np.nanmax(np.where(np.isfinite(Zop), Zop, -np.inf), axis=1) - lcdm)
    gain[~np.isfinite(gain)] = np.nan
    # its maxima away from the best fit, each as high as any true node of the three planes within 1.5 of it in omega
    nodes = node_records([w[0] for ws in windows.values() for w in ws], names)
    peaks = [k for k in range(1, len(gain) - 1) if np.isfinite(gain[k]) and gain[k] >= np.nanmax(gain[max(k - 2, 0):k + 3])]
    peaks = [k for k in sorted(peaks, key=lambda k: -gain[k]) if abs(om_ax[k] - bx["omega"]) > 3]
    second = []
    for k in peaks:
        if all(abs(om_ax[k] - s) > 3 for s, _ in second):
            near = [n["logl"] for n in nodes if abs(n["x"]["omega"] - om_ax[k]) < 1.5]
            second.append((om_ax[k], 2 * (max(near + [Zop[k][np.isfinite(Zop[k])].max()]) - lcdm)))
        if len(second) == 4:
            break
    for label, (o, g) in zip("ABCD", second):
        # settled unless a node not final yet within 1.5 of it in omega could rise above it by the margin
        ok = final or (settled and all(d > (dchi - g) + SETTLED_MARGIN for w, d in pend_om if abs(w - o) < 1.5))
        for name, v in (("omega", f"{o:.0f}"), ("dchi", f"{g:.1f}"), ("gap", f"{dchi - g:.1f}")):
            num(f"{tag}second{label}{name}", prelim(v, ok))
    low_cut = 0.5 * sum(BANDS["pp" if tag == "PP" else "wc"][0])
    num(f"{tag}lowdchi", Ps(f"{np.nanmax(gain[om_ax < low_cut]):.1f}"))
    # the number of separate ranges of omega in which the improvement exceeds 4, and the look-elsewhere estimate
    above = np.where(np.isfinite(gain), gain, -np.inf) >= 4
    num(f"{tag}nabovefour", Ps(f"{int(np.sum(above[1:] & ~above[:-1]) + above[0])}"))
    res = lee(gain, dchi)
    num(f"{tag}localp", Ps(f"{chi2_dist.sf(dchi, 2):.3f}"))
    num(f"{tag}upcross", Ps(f"{res[1.0][0]}"))
    num(f"{tag}lee", Ps(f"{res[1.0][1]:.1f}"))
    num(f"{tag}leelo", Ps(f"{min(v[1] for v in res.values()):.1f}"))
    num(f"{tag}leehi", Ps(f"{max(v[1] for v in res.values()):.1f}"))
    print(f"{tag} look-elsewhere: Delta chi2 {dchi:.2f}, local p {chi2_dist.sf(dchi, 2):.4f}, "
          + ", ".join(f"u0 {u0}: {n} upcrossings, N {v:.2f}" for u0, (n, v) in res.items()))
    # the contours of the (omega, phi) plane: the separate ranges of omega with a node inside 2.30 and inside 6.18
    colmin = 2 * (best_l - np.nanmax(np.where(np.isfinite(Zop), Zop, -np.inf), axis=1))
    for lv, name in ((2.30, "one"), (6.18, "two")):
        ins = colmin < lv
        num(f"{tag}islands{name}", Ps(f"{int(np.sum(ins[1:] & ~ins[:-1]) + ins[0])}"))
    # the ridge of the main peak in (omega, phi): d phi / d omega and the wavenumber where the oscillation keeps its phase
    ph_ax = op["ax"][1]
    d2 = 2 * (best_l - Zop)
    ridge = [k for k in range(len(om_ax)) if np.isfinite(d2[k]).any() and np.nanmin(d2[k]) < 6.18 and abs(om_ax[k] - bx["omega"]) <= 3]
    if len(ridge) >= 3:
        slope = np.polyfit(om_ax[ridge], np.unwrap([ph_ax[int(np.nanargmin(d2[k]))] for k in ridge]), 1)[0]
        kfix = 0.05 * math.exp(-slope)
        num(f"{tag}tilt", Ps(f"{slope:.2f}"))
        num(f"{tag}kfix", Ps(f"{kfix:.3f}"))
        num(f"{tag}ellfix", Ps(f"{round(kfix * 14000, -2):.0f}"))
    # the profile of A_osc: at each grid value the better of the two planes that contain it (true values), its intervals
    # (interpolated in sqrt(Delta chi^2)), its value at the top of the grid, the frequency of the best node at large
    # amplitude, and the lowest Delta chi^2 at omega = 1 (the oscillation compensated by n_s and the running)
    A_ax = ao["ax"][0]
    rows = []
    for s in (ao, ap):
        Zs = np.where((True if closed else s.get("valtrue", s["true"])) & np.isfinite(s["Z"]), s["Z"], -np.inf)
        assert np.allclose(s["ax"][0], A_ax)
        rows.append(2 * (best_l - Zs.max(axis=1)))
    pA = np.fmin(*rows)
    sA, k0 = np.sqrt(np.maximum(pA, 0)), int(np.argmin(pA))

    def crossing(level, step):
        k = k0
        while 0 <= k + step < len(pA) and pA[k + step] < level:
            k += step
        if not 0 <= k + step < len(pA):
            return float("nan")
        t = (math.sqrt(level) - sA[k]) / (sA[k + step] - sA[k])
        return A_ax[k] + t * (A_ax[k + step] - A_ax[k])

    step = float(A_ax[1] - A_ax[0])
    PA = lambda a: prelim(f"{a:.3f}", final or (settled and a < pend_A - step))
    for lv, name in ((1, "one"), (4, "four")):
        num(f"{tag}A{name}lo", PA(crossing(lv, -1)))
        num(f"{tag}A{name}hi", PA(crossing(lv, 1)))
    num(f"{tag}Atopdchi", P(f"{pA[-1]:.1f}" if np.isfinite(pA[-1]) else "?"))
    Zao = np.where((True if closed else ao["true"]) & np.isfinite(ao["Z"]), ao["Z"], -np.inf)
    at = Zao.argmax(axis=1)
    big_rows = [i for i in range(len(A_ax)) if A_ax[i] >= 0.05 and np.isfinite(Zao[i]).any()]
    num(f"{tag}Ahighomega", P(f"{np.median(ao['ax'][1][at[big_rows]]):.0f}"))
    # the amplitude above which the best node is no longer at the best fit's frequency
    switch = [i for i in range(k0 + 1, len(A_ax)) if np.isfinite(Zao[i]).any() and abs(ao["ax"][1][at[i]] - bx["omega"]) > 3]
    if switch:
        num(f"{tag}Aswitch", P(f"{A_ax[switch[0]]:.3f}"))
    col1 = 2 * (best_l - Zao[:, 0])
    col1_done = bool(ao["true"][:, 0][~ao["outside"][:, 0]].all())
    num(f"{tag}Alowomega", prelim(f"{np.min(col1[np.isfinite(col1)]):.1f}", final or (settled and col1_done)))
    # the refinement with the true likelihood (Appendix app:profiles): the nodes of all studies inside Delta chi^2 < 11.83
    inside = [n for n in nodes if 2 * (best_l - n["logl"]) < 11.83]
    refine = np.array([2 * (n["logl"] - n["direct"]) for n in inside if n["direct"] is not None])
    num(f"{tag}profinside", P(big(len(inside))))
    num(f"{tag}profgainhigh", P(f"{np.percentile(refine, 90):.2f}"))
    num(f"{tag}profcalls", P(f"{np.mean([n['calls'] for n in inside]):.0f}"))
    num(f"{tag}profconv", P(f"{100 * np.mean([n['reason'] == 'converged' for n in inside]):.0f}"))
    budget = [n for n in inside if n["reason"] == "budget"]
    num(f"{tag}profbudget", P(big(len(budget))))
    num(f"{tag}profbudgetmin", P(f"{min(2 * (best_l - n['logl']) for n in budget):.1f}" if budget else "--"))
    num(f"{tag}proftotal", P(latex_float(sum(n["calls"] for n in nodes))))
    # the two planes that contain omega give the 1D profile of omega twice: their largest difference where both are true
    Zc = np.where(ao["true"] & np.isfinite(ao["Z"]), ao["Z"], np.nan)
    assert np.allclose(ao["ax"][1], om_ax)
    both = np.isfinite(Zc).all(axis=0) & np.isfinite(Zop).all(axis=1)
    if both.any():
        num(f"{tag}profcross", P(f"{2 * np.max(np.abs(np.nanmax(Zc, axis=0) - np.nanmax(Zop, axis=1))[both]):.2f}"))
    return dict(best_l=best_l, bx=bx, lcdm=lcdm, dchi=dchi, final=final, settled=settled, pend_A=pend_A, prof=prof_d)


def latex_float(v, digits=2):
    e = int(math.floor(math.log10(abs(v))))
    m = round(v / 10 ** e, digits - 1)
    if abs(m) >= 10:                                 # 9.97e5 rounds to 10.0e5: write 1.0e6
        m, e = m / 10, e + 1
    return f"\\ensuremath{{{m:.{digits - 1}f}\\times10^{{{e}}}}}"   # 1.0e6, not 1e6 (the same digits as 1.4e6)


res_pp = profile_results("PP", prof, PROF_WINDOWS, PROF_NAMES)
res_wc = profile_results("WC", prof_wc, WC_WINDOWS, WC_NAMES, closed=WC_CLOSED) if "prof_wc" in globals() else None

# the best fit of PDP+ in the data of H0WC: in its (omega, phi) plane the nearest node to the frequency and phase of the
# PDP+ best fit (its amplitude and improvement), in its (A_osc, omega) plane the nearest node to its frequency and
# amplitude (chi^2 there, for any phase, against the featureless best fit), and the largest improvement in 49 <= omega < 52.5
if res_wc:
    P = lambda v: prelim(v, res_wc["final"])
    px, pw = res_pp["bx"], res_wc["prof"]
    op, ao = pw["omega_phi"], pw["aosc_omega"]
    om_ax, ph_ax = op["ax"]
    i = int(np.argmin(np.abs(om_ax - px["omega"])))
    j = int(np.argmin(np.abs(np.angle(np.exp(1j * (ph_ax - px["phi"]))))))
    Ps = lambda v: prelim(v, res_wc["settled"])
    num("WCatPPA", Ps(f"{op['X'][i, j][WC_NAMES.index('A_osc')]:.3f}"))
    num("WCatPPgain", Ps(f"{2 * (op['Z'][i, j] - res_wc['lcdm']):.1f}"))
    ia = int(np.argmin(np.abs(ao["ax"][0] - px["A_osc"])))
    ja = int(np.argmin(np.abs(ao["ax"][1] - px["omega"])))
    num("WCatPPamp", prelim(f"{-2 * (ao['Z'][ia, ja] - res_wc['lcdm']):.1f}",
                            res_wc["final"] or (res_wc["settled"] and ao["ax"][0][ia] < res_wc["pend_A"])))
    sel = (om_ax >= PEAK[0]) & (om_ax < PEAK[1])
    num("WCppgain", Ps(f"{2 * (np.nanmax(op['Z'][sel]) - res_wc['lcdm']):.1f}"))
    num("WCprofstate", "" if res_wc["final"] else
        " \\draft{The profiles of \\HWC{} are still running, and the numbers from them (red) are those of the nodes "
        "calculated so far.}")

# the oscillation of the best fit in terms of the potential (Sec. theory): f = 1/(omega phi_*), b from A_osc = delta n_s
PHI_STAR = 10.5
bx = res_pp["bx"]
num("fbest", latex_float(1 / (bx["omega"] * PHI_STAR), 2))
num("bbest", f"{bx['A_osc'] / (3 * math.sqrt(2 * math.pi / bx['omega'])):.3f}")
num("fnlbest", f"{bx['A_osc'] * bx['omega'] ** 2 / 8:.0f}")

# -- Table tab:model_comparison
with open(TABLES / "model_comparison.tex", "w") as f:
    f.write("\\begin{tabular}{rlcccc}\n\\cmidrule[\\heavyrulewidth](lr){2-6}\n"
            " & \\textbf{Model} & $N$ & $\\Delta\\chi^2$ & $\\Delta$AIC & $\\ln B$ \\\\\n\\cmidrule(lr){2-6}\n")
    for tag, name, r in (("PP", "\\PDP{}", res_pp), ("WC", "\\HWC{}", res_wc)):
        b, e, fin = lnb[tag]
        bcell = f"${b:.1f} \\pm {e:.1f}$" if np.isfinite(e) else f"${b:.1f}$"
        cells = [f"${-r['dchi']:.1f}$", f"${-r['dchi'] + 6:+.1f}$", bcell]
        cells = [c if ok else f"\\prelim{{{c}}}" for c, ok in zip(cells, (r["settled"], r["settled"], fin))]
        f.write(f"{name} & $\\Lambda$CDM + running & 7 & 0 & 0 & 0 \\\\\n & + oscillation & 10 & {' & '.join(cells)} \\\\\n")
        f.write("\\cmidrule(lr){2-6}\n" if tag == "PP" else "")
    f.write("\\cmidrule[\\heavyrulewidth](lr){2-6}\n\\end{tabular}\n")


# ---------------------------------------------------------------- the bands and the modes of the boosts (Appendices A.2, A.6)
# From plots/data/bands200.json (analysis/bands200.py, which evaluates the parts' networks; run it again only if a part or a
# combined emulator changes): each part's network at the true training points of its neighbour in the overlap of their
# boxes (within 30 of the best point in ln P), the combined network inside each band, and the modes of each part's boosts.
# For the lowest band of PDP+ also the profile nodes just above it against the true training points of the low part in the
# same cells of the (A_osc, omega) grid: those nodes are refined from the optimum of the emulator above the band.
BANDS_FILE = DATA / "bands200.json"
if BANDS_FILE.exists():
    bd = json.loads(BANDS_FILE.read_text())
    for tag, data in (("PP", "pp"), ("WC", "wc")):
        parts_b, bands_b = bd[data]["parts"], bd[data]["bands"]
        zones = parts_b if data == "pp" else parts_b[1:]
        assert all(set(p["modes"]) == {1} and p["cells"] == 1 for p in zones), f"{tag}: a part with several modes"
        others = bands_b[1:]
        num(f"{tag}bandothers", f"{max(max(e[k]['all']['std'] for k in ('lower_net_at_upper_points', 'upper_net_at_lower_points')) for e in others):.2f}")
        num(f"{tag}bandcomb", f"{max(e['combined_in_band']['all']['rms'] for e in others):.2f}")
        low = bands_b[0]
        num(f"{tag}bandlowstd", f"{low['lower_net_at_upper_points']['all']['std']:.2f}")
        up = low["upper_net_at_lower_points"]
        num(f"{tag}bandupstd", f"{up['all']['std']:.2f}")
        num(f"{tag}bandupmean", f"{-up['all']['mean']:.2f}")
        num(f"{tag}bandcomblow", f"{low['combined_in_band']['all']['rms']:.2f}")
        num(f"{tag}bandoverlap", f"${low['overlap'][0]:g} \\leq \\omega \\leq {low['overlap'][1]:g}$")
    wl = bd["wc"]["parts"][0]
    num("WClowmodesmin", min(wl["modes"]))
    num("WClowmodesmax", max(wl["modes"]))
    num("WClowcells", ["", "one", "two", "three", "four"][wl["cells"]])
    byA = bd["pp"]["bands"][0]["upper_net_at_lower_points"]["by_A"]
    num("PPbandupA", f"{-byA[0]['mean']:.1f}")
    num("PPbandupB", f"{-byA[1]['mean']:.0f}")
    num("PPbandupC", f"{-byA[2]['mean']:.0f}")
    # the share of the posterior in the lowest band: the combined emulator's chain, and the low part's own chain scaled to the
    # combined chain's share below the band
    lo_b, hi_b = bd["pp"]["bands"][0]["band"]
    low_run = RES / bd["pp"]["parts"][0]["name"]
    zl = np.load(low_run / f"analysis/posterior_it_{bd['pp']['parts'][0]['network']}.npz", allow_pickle=True)
    oml = zl["x"][:, [str(n) for n in zl["names"]].index("omega")].astype(float)
    omc = chains["PP"].omega.to_numpy()
    scale = np.mean(omc < lo_b) / np.mean(oml < lo_b)
    num("PPbandshare", f"{100 * np.mean((omc >= lo_b) & (omc < hi_b)):.2f}")
    num("PPbandsharelow", f"{100 * scale * np.mean((oml >= lo_b) & (oml < hi_b)):.2f}")
    # the profile nodes next to the band against the low part's training points in their cells
    ao = prof["aosc_omega"]
    A_ax, om_ax = ao["ax"]
    tr = pd.read_csv(low_run / f"training_data/data_it_{bd['pp']['parts'][0]['network']}.csv", usecols=["A_osc", "omega", "loglkl"])
    tr = tr[np.isfinite(tr.loglkl) & (tr.loglkl > -1e4)]
    dA, dO = A_ax[1] - A_ax[0], om_ax[1] - om_ax[0]
    excess = []
    for j in np.nonzero(om_ax <= bd["pp"]["parts"][0]["box"][1] + dO / 2)[0]:
        for i in range(len(A_ax)):
            m = (np.abs(tr.omega - om_ax[j]) < dO / 2) & (np.abs(tr.A_osc - A_ax[i]) < dA / 2)
            if m.any() and np.isfinite(ao["Z"][i, j]):
                t = tr.loglkl[m].max()
                excess.append((t - ao["Z"][i, j], 2 * (res_pp["best_l"] - t)))
    excess = np.array(excess)
    low_nodes = excess[excess[:, 0] > 0.2]
    num("PPbandnodemax", f"{low_nodes[:, 0].max():.0f}")
    num("PPbandnodedchi", f"{low_nodes[:, 1].min():.0f}")
    print(f"bands: PP others std <= {numbers['PPbandothers']}, combined <= {numbers['PPbandcomb']}; WC {numbers['WCbandothers']}, "
          f"{numbers['WCbandcomb']}; lowest PP band: upper net too low by {numbers['PPbandupA']}, {numbers['PPbandupB']}, "
          f"{numbers['PPbandupC']}; posterior in band {numbers['PPbandshare']}% against {numbers['PPbandsharelow']}%; "
          f"{len(low_nodes)} nodes below training points by up to {numbers['PPbandnodemax']}, all at Delta chi2 > {numbers['PPbandnodedchi']}")
else:
    print("no plots/data/bands200.json: run analysis/bands200.py")

# ---------------------------------------------------------------- the binning of ACT and SPT (Sec. actspt:setup, Appendix B)
# From plots/data/binning200.json (analysis/binning200.py): the chi^2 of the imprint of A_osc = 0.02 in the bandpowers of
# ACT DR6 and SPT-3G D1 against the same data multipole by multipole, and the budgets of PDP+ (CamSpec, not binned) and
# H0WC (CamSpec at l < 1000 in TT and l < 600 in TE and EE, ACT and SPT binned), with the noise models of ladder_norms.
BIN_FILE = DATA / "binning200.json"
if BIN_FILE.exists():
    bn = json.loads(BIN_FILE.read_text())
    wc_b = {float(k): v for k, v in bn["wc"].items()}
    pp_b = {float(k): v for k, v in bn["pp"].items()}
    for w in (60.0, 100.0, 150.0, 200.0):
        tag = {60.0: "Sixty", 100.0: "Hundred", 150.0: "OneFifty", 200.0: "TwoHundred"}[w]
        num(f"BinACT{tag}", f"{round(100 * wc_b[w]['act_ratio']):.0f}")
        num(f"BinSPT{tag}", f"{round(100 * wc_b[w]['spt_ratio']):.0f}")
    for w, tag in ((60.0, "Sixty"), (200.0, "TwoHundred")):
        total_wc = wc_b[w]["planck"] + wc_b[w]["act_binned"] + wc_b[w]["spt_binned"]
        num(f"BinWC{tag}", f"{total_wc:.1f}")
        num(f"BinPP{tag}", f"{pp_b[w]['planck']:.1f}")
        num(f"BinWCunbinned{tag}", f"{wc_b[w]['planck'] + wc_b[w]['act_unbinned'] + wc_b[w]['spt_unbinned']:.1f}")
    num("BinWCPlanckShareTwoHundred", f"{round(100 * wc_b[200.0]['planck'] / (wc_b[200.0]['planck'] + wc_b[200.0]['act_binned'] + wc_b[200.0]['spt_binned'])):.0f}")
    print("binning:", {k: numbers[k] for k in numbers if k.startswith("Bin")})
else:
    print("no plots/data/binning200.json: run analysis/binning200.py")

# ---------------------------------------------------------------- the emulators' own profiles (Appendix app:profiles)
# Steen, 8 November 2026: the rms difference between the true profile and the emulator's, in the 1 sigma region and in
# the whole region (Delta chi^2 < 2.30 and < 11.83, the levels of the figures), no table. At each node of each study in
# the figures (PROF_WINDOWS, WC_WINDOWS): 2 (true value - the emulator's maximum at that node), the latter from the
# study's scan (surrogate_surface.json), which the refinement starts from. Each study is compared with its own scan, on
# its own nodes: joining the windows interpolates in A_osc, and the joined (A_osc, phi) surface keeps no emulator values.
# Delta chi^2 below the best true node of the dataset.
LEVELS_2D = (2.30, 11.83)


def profile_errors(prefix, studies):
    """The rms of 2 (true - emulator) over the nodes inside each level of LEVELS_2D at which the refinement converged (at the
    others the true value only bounds the profile from below); Delta chi^2 from the best true node of all nodes."""
    T, S, conv = [], [], []
    for st in studies:
        d = LOGS / f"profiles_{st}_surface"
        if not (d / "surrogate_surface.json").exists():
            continue
        scan = np.array(json.loads((d / "surrogate_surface.json").read_text())["logl"], float)
        for (i, j), node in sorted(study_nodes(st).items()):
            v = node["logl"]
            if v is not None and np.isfinite(v) and np.isfinite(scan[i, j]):
                T.append(v)
                S.append(scan[i, j])
                conv.append(node["reason"] == "converged")
    T, S, conv = np.array(T), np.array(S), np.array(conv, bool)
    dchi, diff = 2 * (T.max() - T), 2 * (T - S)
    return [(int((conv & (dchi < lv)).sum()), int((dchi < lv).sum()), float(np.sqrt(np.mean(diff[conv & (dchi < lv)] ** 2))))
            for lv in LEVELS_2D], len(T)


for name, windows in (("PP", PROF_WINDOWS), ("WC", WC_WINDOWS)):
    studies = [w[0] for ws in windows.values() for w in ws]
    (one, whole), n_nodes = profile_errors(name, studies)
    num(f"{name}profrmsOne", f"{one[2]:.2f}")
    num(f"{name}profrmsAll", f"{whole[2]:.2f}")
    print(f"emulator profile against the true one, {name}, converged nodes: rms {one[2]:.3f} at {one[0]} of {one[1]} nodes "
          f"(Delta chi^2 < 2.30), {whole[2]:.3f} at {whole[0]} of {whole[1]} nodes (< 11.83), of {n_nodes} true nodes")
    if name == "WC":
        total = 2500 * len(studies)
        num("WCprofrmsNote", "" if n_nodes >= total or WC_CLOSED else
            f" \\draft{{For \\HWC{{}} from the {big(n_nodes)} of {big(total)} nodes calculated so far.}}")

# ---------------------------------------------------------------- the cost of standard methods (Appendix app:cost)
# Steen, 7-8 November 2026: how many evaluations of the true likelihood, and how much wall time, a stable posterior, the
# evidence and the profile likelihoods would take with standard samplers calling CLASS++ directly, compared with CLiENT.
# The standard samplers were run on the combined emulators in place of CLASS++, counting the calls of the likelihood
# (SAMPLER: the work tree's results/20261107_sampler_cost; its README.md describes the runs, and its summarise.py writes
# sampler_cost.json: Metropolis-Hastings calls corrected for proposals outside the prior box, which cost no call; runs
# that did not reach R-1 = 0.01 extrapolated with R-1 proportional to 1 / calls, which overestimates the PDP+ zone run,
# the one that did, by about 1.4; nested sampling from the calls per new point of dynesty's slice sampling and BEST's
# n_dead / n_live and H on the same emulator; PROSPECT on the PDP+ emulator in zone 1b, A_osc = 0.01 to 0.04).
# CLiENT's evaluations are the attempts in the call journals of its parts (objective_calls/summary.json: every true
# evaluation of the loop, the audits and the member pilots included; not the comparisons with the true likelihood of
# Appendix app:accuracy), up to the member pilot of network start + EARLY for the posterior after EARLY boosts. Its wall
# time is the sum of the times of the iterations (metrics.json; up to the training of network start + EARLY), of the
# longest part, since the parts ran at the same time on nodes of their own. A profile node costs the evaluations of its
# refinement (lifetime_node_attempts of true_surface/node_*.json), over all studies of a dataset's three planes.
# Wall times of the standard methods, which no run measured: Metropolis-Hastings with MH_CHAINS chains (more chains do
# not shorten the burn-in of each), each evaluation on EVAL_CORES cores at a speed-up of EVAL_SPEEDUP (CLASS++ does not
# parallelise well beyond a few cores); nested sampling parallel over the live points, one evaluation per core on
# NS_CORES cores; PROSPECT one annealing chain per node and repetition, its evaluations as for Metropolis-Hastings.
SAMPLER = DATA_ROOT / "results/20261107_sampler_cost/sampler_cost.json"
MH_CHAINS, EVAL_CORES, EVAL_SPEEDUP, NS_CORES = 8, 4, 3, 96
COST = {"pp": probe_cost(7010679), "wc": probe_cost(7010680)}        # core seconds per evaluation (Appendix app:classpp)


def sci(v, digits=2, approx=False):
    """v in LaTeX, as a power of ten from 10^4 on (digits significant digits), with a thin space below."""
    if v < 1e4:
        s = big(int(round(v)))
    else:
        e = int(math.floor(math.log10(v)))
        m = round(v / 10 ** e, digits - 1)
        if m >= 10:
            m, e = m / 10, e + 1
        s = f"{m:.{digits - 1}f}\\times10^{{{e}}}" if digits > 1 else f"{m:.0f}\\times10^{{{e}}}"
    return ("\\sim " if approx else "") + s


def human(sec):
    h = sec / 3600
    if h < 36:
        return f"{h:.0f} h" if h >= 3 else f"{h:.1f} h"
    d = h / 24
    if d < 14:
        return f"{d:.0f} days"
    if d < 63:
        return f"{d / 7:.0f} weeks"
    if d < 548:
        return f"{d / 30.44:.0f} months"
    return f"{d / 365.25:.1f} years"


def journal_attempts(part, upto=None):
    """The true evaluations of a part's loop: all of them, or those of the iterations before upto and the member pilot
    that chose network upto."""
    stages = json.loads((result_dir(part) / "objective_calls/summary.json").read_text())["stages"]
    total = 0
    for name, v in stages.items():
        m = re.search(r"/it(\d+)", name)
        it = int(m.group(1)) if m else -1
        if upto is None or it < upto or (it == upto and name.startswith("members/")):
            total += v["attempts"]
    return total


def part_wall(part, upto=None):
    m = json.loads((result_dir(part) / "metrics.json").read_text())
    its = {r["iteration"]: r["total_iteration_time"] for r in m["iteration"]}
    if upto is None:
        return sum(its.values())
    train = {r["iteration"]: r["training_time"] for r in m["training"]}
    return sum(v for k, v in its.items() if k < upto) + train.get(upto, 0.0)


def node_calls(studies):
    calls = []
    for st in studies:
        for _, e in sorted(study_nodes(st).items()):
            if e["attempts"] is not None:
                calls.append(e["attempts"])
    return np.array(calls, float)


if SAMPLER.exists() and all(progress[p]["done"] for p in PARTS["pp"] + PARTS["wc"]):
    sc = json.loads(SAMPLER.read_text())
    rows = {"pp": [], "wc": []}            # (goal, method, evaluations, wall time) of Table tab:cost
    for d, name in (("pp", "PP"), ("wc", "WC")):
        s, cost = sc[d], COST[d]
        z, f, ns = s["mh_zone1b"], s["mh_full"], s["nested"]
        zone_calls = z["calls_at"]["0.01"] or z["calls_001_extrapolated"]
        zone_approx = z["calls_at"]["0.01"] is None
        # over the whole prior at least the calls of the zone (its two chains' R-1 extrapolated alone can come out lower)
        full_calls = max(f["calls_001_extrapolated"], zone_calls)
        mh_wall = lambda c: c / MH_CHAINS * cost / EVAL_SPEEDUP
        num(f"SC{name}mhZone", f"\\ensuremath{{{sci(zone_calls, 1 if zone_approx else 2, zone_approx)}}}")
        num(f"SC{name}mhZoneWall", human(mh_wall(zone_calls)))
        num(f"SC{name}mhFull", f"\\ensuremath{{{sci(full_calls, 1, True)}}}")
        num(f"SC{name}mhFullWall", human(mh_wall(full_calls)))
        num(f"SC{name}mhFullRun", f"\\ensuremath{{{sci(f['calls'])}}}")
        num(f"SC{name}mhFullR", f"{f['R_last']:.2f}")
        num(f"SC{name}mhPerProposal", f"{z['calls_per_proposal']:.2f}")
        if zone_approx:
            num(f"SC{name}mhZoneRun", f"\\ensuremath{{{sci(z['calls'])}}}")
            num(f"SC{name}mhZoneStuck", f"{z['R_after_002'][0]:.3f} and {z['R_after_002'][1]:.3f}")
        num(f"SC{name}nsPerPoint", f"{round(ns['calls_per_point'], -1):.0f}")
        num(f"SC{name}nsSlices", f"{ns['slices']}")
        num(f"SC{name}nsDeadPerLive", f"{ns['best']['dead_per_live']:.0f}")
        num(f"SC{name}nsH", f"{ns['best']['H']:.0f}")
        for n, tag in (("500", "Low"), ("2048", "High")):
            c = ns["cases"][n]
            num(f"SC{name}ns{tag}", f"\\ensuremath{{{sci(c['calls'], 2, True)}}}")
            num(f"SC{name}ns{tag}Sigma", f"{c['sigma_lnZ']:.2f}")
            num(f"SC{name}ns{tag}Wall", human(c["calls"] * cost / NS_CORES))
        rows[d] += [("Posterior, $38 \\leq \\omega \\leq 60$", "Metropolis--Hastings",
                     f"${sci(zone_calls, 1 if zone_approx else 2, zone_approx)}$", human(mh_wall(zone_calls))),
                    ("Posterior, $1 \\leq \\omega \\leq 200$", "Metropolis--Hastings",
                     f"${sci(full_calls, 1, True)}$", human(mh_wall(full_calls)))]
        for n in ("500", "2048"):
            c = ns["cases"][n]
            rows[d].append((f"Evidence, $\\sigma(\\ln Z) = {c['sigma_lnZ']:.2f}$", f"Nested sampling, {n} live points",
                            f"${sci(c['calls'], 2, True)}$", human(c["calls"] * cost / NS_CORES)))
        if "prospect" in s:
            pr = s["prospect"]["settings"]
            lo, hi = pr[min(pr, key=int)], pr[max(pr, key=int)]
            seq = lambda st: (st["steps"] + 1) * st["iterations"] * cost / EVAL_SPEEDUP     # one chain, in sequence
            num(f"SC{name}prLow", big(lo["calls_per_point"]))
            num(f"SC{name}prHigh", big(hi["calls_per_point"]))
            num(f"SC{name}prStepsLow", f"{lo['steps']}")
            num(f"SC{name}prStepsHigh", f"{hi['steps']}")
            num(f"SC{name}prGapLow", f"{max(lo['gap_at_20']):.2f}")
            num(f"SC{name}prGapHigh", f"{max(hi['gap_at_20']):.2f}")
            num(f"SC{name}prToOne", f"{lo['calls_to_01'][0]}--{lo['calls_to_01'][1]}")
            num(f"SC{name}prStartGap", f"{min(lo['start_gap']):.1f}--{max(lo['start_gap']):.1f}")
            nm = s["prospect"]["nomcmc"]
            wrong = {k: v for k, v in nm.items() if max(p["gap_at_20"] for p in v) > 1}
            num(f"SC{name}prStarts", ["", "one", "two", "three", "four", "five", "six"][len(nm)])
            num(f"SC{name}prWrong", ["", "one", "two", "three", "four", "five", "six"][len(wrong)])
            gaps = sorted({round(p["gap_at_20"], 1) for v in wrong.values() for p in v})
            num(f"SC{name}prWrongGaps", f"{gaps[0]:.1f} and {gaps[-1]:.1f}" if len(gaps) > 1 else f"{gaps[0]:.1f}")
            rows[d].append(("Profile likelihood, one node", "Simulated annealing (PROSPECT) after Metropolis--Hastings",
                            f"${big(lo['calls_per_point'])}$--${sci(hi['calls_per_point'])}$", f"{human(seq(lo))}--{human(seq(hi))}"))
        # CLiENT
        early = {"pp": EARLY, "wc": EARLY_WC}[d]
        final_calls = sum(journal_attempts(p) for p in PARTS[d])
        final_wall = max(part_wall(p) for p in PARTS[d])
        cores = max(len({fn.name.split("-")[0] for fn in (result_dir(p) / "objective_calls").glob("rank*")}) for p in PARTS[d])
        num(f"SC{name}clientFinal", f"\\ensuremath{{{sci(final_calls)}}}")
        num(f"{name}evals", numbers[f"SC{name}clientFinal"])      # Secs. numerics:client and actspt:setup
        num(f"SC{name}clientFinalWall", human(final_wall))
        num(f"SC{name}clientCores", f"{cores}")
        num(f"SC{name}clientAlloc", f"\\ensuremath{{{sci(sum(part_wall(p) for p in PARTS[d]) * cores / 3600)}}}")
        num(f"SC{name}clientFinalCoreHours", f"\\ensuremath{{{sci(final_calls * cost / 3600)}}}")
        if early is not None:
            early_calls = sum(journal_attempts(p, progress[p]["start"] + early) for p in PARTS[d])
            early_wall = max(part_wall(p, progress[p]["start"] + early) for p in PARTS[d])
            num(f"SC{name}clientEarly", f"\\ensuremath{{{sci(early_calls)}}}")
            num(f"SC{name}clientEarlyWall", human(early_wall))
            num(f"SC{name}clientEarlyCoreHours", f"\\ensuremath{{{sci(early_calls * cost / 3600)}}}")
            num(f"SC{name}clientEarlyAlloc",
                f"\\ensuremath{{{sci(sum(part_wall(p, progress[p]['start'] + early) for p in PARTS[d]) * cores / 3600)}}}")
            rows[d].append(("Posterior", f"\\client{{}}, {['', 'one', 'two', 'three', 'four', 'five', 'six'][early]} boosts",
                            f"${sci(early_calls)}$", human(early_wall)))
        rows[d].append(("Posterior and evidence", "\\client{}, ten boosts", f"${sci(final_calls)}$", human(final_wall)))
        studies = [w[0] for ws in (PROF_WINDOWS if d == "pp" else WC_WINDOWS).values() for w in ws]
        calls = node_calls(studies)
        if calls.size:
            # a study's grid has 2500 nodes: fewer node files over the studies means that some are still running
            so_far = calls.size < 2500 * len(studies) and not (d == "wc" and WC_CLOSED)
            mark = (lambda t: f"\\draft{{{t}}}") if so_far else (lambda t: t)
            num(f"SC{name}profSoFar", "\\draft{(studies still running)}" if so_far else "")
            text_mark = (lambda t: f"\\prelim{{{t}}}") if so_far else (lambda t: t)      # in the text (app:cost)
            num(f"SC{name}profMean", text_mark(f"{calls.mean():.0f}"))
            num(f"SC{name}profMedian", text_mark(f"{np.median(calls):.0f}"))
            num(f"SC{name}profNodes", big(int(calls.size)))
            rows[d].append(("Profile likelihood, one node", "\\client{}, refinement (mean)",
                            mark(f"{calls.mean():.0f}"), ""))
            rows[d].append(("2D profile likelihood, 2500 nodes", "\\client{}, refinement", mark(f"${sci(2500 * calls.mean())}$"), ""))
    # the maxima of the PDP+ profile in omega (plots/data/omega200_pp.json, written above)
    om = json.loads((DATA / "omega200_pp.json").read_text())
    g = np.array([np.nan if v is None else v for v in om["gain"]], float)
    g = g[np.isfinite(g)]
    peaks = [i for i in range(1, len(g) - 1) if g[i] > g[i - 1] and g[i] >= g[i + 1]]
    num("SCPPomegaMaxima", f"{len(peaks)}")
    num("SCPPomegaMaximaFive", f"{sum(g.max() - g[i] < 5 for i in peaks)}")
    num("SCmhExtrapCheck", f"{max(sc['check_inverse_rule'].values()):.1f}")      # the 1 / calls rule on the PDP+ zone run
    num("SCPPnsRatio", f"{sc['pp']['nested']['cases']['500']['calls'] / sum(journal_attempts(p) for p in PARTS['pp']):.0f}")
    pr = sc["pp"]["prospect"]["settings"]
    surface = 2500 * node_calls([w[0] for ws in PROF_WINDOWS.values() for w in ws]).mean()
    num("SCPPprPointsLow", f"{surface / pr[max(pr, key=int)]['calls_per_point']:.0f}")      # points of PROSPECT costing as
    num("SCPPprPointsHigh", f"{round(surface / pr[min(pr, key=int)]['calls_per_point'], -1):.0f}")  # much as one surface
    # Appendix app:cost in H0WC's numbers (Steen, 10 November 2026): nested sampling with the 2048 live points of the evidence
    # runs against the evaluations of the final emulators, and the PROSPECT points (run with PDP+) costing as much as one
    # surface of H0WC; red while the H0WC studies run
    num("SCWCnsRatio", f"{round(sc['wc']['nested']['cases']['2048']['calls'] / sum(journal_attempts(p) for p in PARTS['wc']), -1):.0f}")
    wc_studies = [w[0] for ws in WC_WINDOWS.values() for w in ws]
    wc_calls = node_calls(wc_studies)
    wp = (lambda t: f"\\prelim{{{t}}}") if wc_calls.size < 2500 * len(wc_studies) and not WC_CLOSED else (lambda t: t)
    surface_wc = 2500 * wc_calls.mean()
    num("SCWCprPointsLow", wp(f"{surface_wc / pr[max(pr, key=int)]['calls_per_point']:.0f}"))
    num("SCWCprPointsHigh", wp(f"{round(surface_wc / pr[min(pr, key=int)]['calls_per_point'], -1):.0f}"))
    num("SCmhChains", f"{MH_CHAINS}")
    num("SCevalCores", f"{EVAL_CORES}")
    num("SCevalSpeedup", f"{EVAL_SPEEDUP}")
    num("SCnsCores", f"{NS_CORES}")
    # Table tab:cost (Thomas Tram's layout), H0WC only (Steen, 10 November 2026): no posterior of the zone 38 <= omega <= 60
    # (a Markov chain cannot be run part by part, since the relative weights of the parts are then unknown; the zone's chains
    # remain the lower bound of the posterior over the whole prior, full_calls), and PROSPECT, run with PDP+ only, with its
    # evaluations (set by the cooling schedule) and the wall time of an evaluation of H0WC (the caption says so).
    # Nested sampling (Steen, 10 November): on the cores that CLiENT's runs of H0WC used at the same time (one per part,
    # SCWCclientCores each), with EVAL_CORES cores per evaluation (speed-up EVAL_SPEEDUP, as for Metropolis-Hastings), so
    # n_proc processes, and PolyChord's parallel speed-up n_live ln(1 + n_proc / n_live) (Handley et al. 2015, eq. 20: nearly
    # linear up to n_proc ~ n_live, diminishing returns beyond). n_live sets the core hours and sigma(ln Z) = sqrt(H / n_live).
    # Two rows: sigma(ln Z) ~ 1 (n_live = H; too few live points for the maxima in omega, which the text says) and the 2048
    # live points of the evidence runs. Core hours: evaluations times the core seconds of an evaluation, for all rows
    # (CLiENT's runs held SCWCclientAlloc in all, training included). CLiENT: also after EARLY_WC boosts, when the posterior
    # and the evidence had stopped changing (the posterior after EARLY_WC boosts above).
    s, cost = sc["wc"], COST["wc"]
    ns = s["nested"]
    per_live = ns["best"]["dead_per_live"] * ns["calls_per_point"]
    n_one = int(round(ns["best"]["H"]))                       # sigma(ln Z) = sqrt(H / n_live) = 1
    calls_one, calls_high = n_one * per_live, ns["cases"]["2048"]["calls"]
    wc_cores = max(len({fn.name.split("-")[0] for fn in (result_dir(p) / "objective_calls").glob("rank*")}) for p in PARTS["wc"])
    ns_total_cores = len(PARTS["wc"]) * wc_cores
    n_proc = ns_total_cores // EVAL_CORES
    ns_wall = lambda n_live, calls: calls / (n_live * math.log(1 + n_proc / n_live)) * cost / EVAL_SPEEDUP
    ch = lambda calls: calls * cost / 3600
    z, f = s["mh_zone1b"], s["mh_full"]
    zone_calls = z["calls_at"]["0.01"] or z["calls_001_extrapolated"]
    full_calls = max(f["calls_001_extrapolated"], zone_calls)
    pr = sc["pp"]["prospect"]["settings"]
    lo, hi = pr[min(pr, key=int)], pr[max(pr, key=int)]
    seq_wc = lambda st: (st["steps"] + 1) * st["iterations"] * cost / EVAL_SPEEDUP     # one chain, in sequence
    final_calls = sum(journal_attempts(p) for p in PARTS["wc"])
    final_wall = max(part_wall(p) for p in PARTS["wc"])
    node_mean = wc_calls.mean()
    mark = (lambda t: f"\\draft{{{t}}}") if wc_calls.size < 2500 * len(wc_studies) and not WC_CLOSED else (lambda t: t)
    num("SCWCnsOneLive", f"{n_one}")
    num("SCWCnsOne", f"\\ensuremath{{{sci(calls_one, 2, True)}}}")
    num("SCWCnsOneCoreHours", f"\\ensuremath{{{sci(ch(calls_one), 2, True)}}}")
    num("SCWCnsHighCoreHours", f"\\ensuremath{{{sci(ch(calls_high), 2, True)}}}")
    num("SCWCnsSequence", f"\\ensuremath{{{sci(per_live, 1, True)}}}")
    num("SCWCnsTotalCores", f"{ns_total_cores}")
    num("SCWCnsProcs", f"{n_proc}")
    num("SCWCnsOneWall", human(ns_wall(n_one, calls_one)))
    num("SCWCnsHighWall", human(ns_wall(2048, calls_high)))
    early_calls_wc = sum(journal_attempts(p, progress[p]["start"] + EARLY_WC) for p in PARTS["wc"])
    early_wall_wc = max(part_wall(p, progress[p]["start"] + EARLY_WC) for p in PARTS["wc"])
    r = int(round(calls_one / early_calls_wc))                                     # sigma ~ 1 against CLiENT after EARLY_WC
    num("SCWCnsOneRatio", {1: "the same number", 2: "twice", 3: "three times", 4: "four times"}.get(r, f"{r} times"))
    num("SCWCnsHighRatio", f"{round(calls_high / early_calls_wc, -1):.0f}")          # sigma = 0.13 against the same
    # What a looser stop would save (Steen, 10 November): from the dead points of the BEST run of the H0WC emulator, the prior
    # volume (in iterations per live point) between the contour where the live points still hold half the evidence and the
    # one where they hold the run's tolerance (0.001, PolyChord's default precision_criterion). It does not change sigma(ln Z).
    dead = np.loadtxt(WC_COMBS[-1] / "evidence/evidence_it_0_nlive2048.txt", usecols=(0, 1))      # weight, -ln L
    tol = json.loads((WC_COMBS[-1] / "evidence/evidence_it_0_nlive2048.json").read_text())["tolerance"]
    wd, lnLd = dead[:, 0] / dead[:, 0].sum(), -dead[:, 1]
    o = np.argsort(-lnLd)
    inner = np.cumsum(wd[o])                                  # the evidence above a contour: what the live points hold there
    lnXd = np.log(np.cumsum(wd[o] * np.exp(-(lnLd[o] - lnLd.max()))))                 # the prior volume above it, + const
    saved = (lnXd[np.searchsorted(inner, 0.5)] - lnXd[np.searchsorted(inner, tol)]) / ns["best"]["dead_per_live"]
    num("SCWCnsStopSaving", f"{100 * saved:.0f}\\%")
    num("SCnsTolerance", f"{100 * tol:g}\\%")
    table = [
        ("Posterior", "Metropolis--Hastings", f"${sci(full_calls, 1, True)}$", f"${sci(ch(full_calls), 1, True)}$",
         human(full_calls / MH_CHAINS * cost / EVAL_SPEEDUP)),
        ("Evidence, $\\sigma(\\ln Z) \\approx 1$", f"Nested sampling, {n_one} live points", f"${sci(calls_one, 2, True)}$",
         f"${sci(ch(calls_one), 2, True)}$", human(ns_wall(n_one, calls_one))),
        (f"Evidence, $\\sigma(\\ln Z) = {ns['cases']['2048']['sigma_lnZ']:.2f}$", "Nested sampling, 2048 live points",
         f"${sci(calls_high, 2, True)}$", f"${sci(ch(calls_high), 2, True)}$", human(ns_wall(2048, calls_high))),
        ("Profile likelihood, one node", "Simulated annealing (PROSPECT) after Metropolis--Hastings",
         f"${big(lo['calls_per_point'])}$--${sci(hi['calls_per_point'])}$", f"{ch(lo['calls_per_point']):.0f}--{ch(hi['calls_per_point']):.0f}",
         f"{human(seq_wc(lo))}--{human(seq_wc(hi))}"),
        ("Posterior and evidence", f"\\client{{}}, {['', 'one', 'two', 'three', 'four', 'five', 'six'][EARLY_WC]} boosts",
         f"${sci(early_calls_wc)}$", f"${sci(ch(early_calls_wc))}$", human(early_wall_wc)),
        ("Posterior and evidence", "\\client{}, ten boosts (as run)", f"${sci(final_calls)}$", f"${sci(ch(final_calls))}$", human(final_wall)),
        ("Profile likelihood, one node", "\\client{}, refinement (mean)", mark(f"{node_mean:.0f}"), mark(f"{ch(node_mean):.1f}"), ""),
        ("2D profile likelihood, 2500 nodes", "\\client{}, refinement", mark(f"${sci(2500 * node_mean)}$"),
         mark(f"${sci(ch(2500 * node_mean))}$"), ""),
    ]
    with open(TABLES / "sampler_cost.tex", "w") as tf:
        # one dataset, so no group column (Steen, 10 November); the caption names H0WC
        tf.write("\\begin{tabular}{>{\\raggedright\\arraybackslash}p{3.2cm}>{\\raggedright\\arraybackslash}p{3.4cm}rrr}\n\\cmidrule[\\heavyrulewidth](lr){1-5}\n"
                 "\\textbf{Goal} & \\textbf{Method} & \\textbf{Evaluations} & \\textbf{Core hours} & \\textbf{Wall time} \\\\\n\\cmidrule(lr){1-5}\n")
        for j, (goal, method, ev, core, wall) in enumerate(table):
            if j and method.startswith("\\client") and not table[j - 1][1].startswith("\\client"):
                tf.write("\\cmidrule(lr){1-5}\n")
            tf.write(f"{goal} & {method} & {ev} & {core} & {wall} \\\\\n")
        tf.write("\\cmidrule[\\heavyrulewidth](lr){1-5}\n\\end{tabular}\n")
    print("cost of standard methods:", {d: [(r[0], r[2], r[3]) for r in rows[d]] for d in rows})
else:
    print("cost of standard methods: not yet (sampler_cost.json or parts missing)")

# ---------------------------------------------------------------- the cache of the profile nodes (above)
if not _FROM_CACHE:
    NODE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = NODE_CACHE.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(dict(written=time.time(), nodes=_NODES)))
    tmp.replace(NODE_CACHE)
    print(f"profile nodes: cache written, {sum(len(v) for v in _NODES.values())} nodes of {len(_NODES)} studies")

# ---------------------------------------------------------------- the numbers
with open(HERE / "numbers200.tex", "w") as f:
    f.write("% written by build200.py: do not edit\n")
    for k, v in numbers.items():
        f.write(f"\\newcommand{{\\num{k}}}{{{v}}}\n")
print(json.dumps(numbers, indent=1))
