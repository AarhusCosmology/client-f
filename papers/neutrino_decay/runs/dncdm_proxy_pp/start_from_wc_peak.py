"""Make results_profiles/dncdm_proxy_pp_peak_gamma_wcstart: the PDP+ peak study (dncdm_proxy_pp_peak_gamma) again, with
every node's true stage started from the H0WC true optimum at that node (dncdm_proxy_wc_peak_gamma, the same grid).
9 November 2026; profiles_peak_log10Gamma_wcstart.yaml says why and how it is run.

    python runs/dncdm_proxy_pp/start_from_wc_peak.py [KIT_DIR]        # KIT_DIR defaults to the current directory

The study folder gets the PDP+ study's manifest.json, audit.json and ranges.json, and its surrogate_profiles.json with
  - profiles.log10Gamma.xopt[j] = the PDP+ true optimum at node j, its six cosmological parameters replaced by the
    H0WC true optimum's at node j (the Planck nuisances stay PDP+'s: H0WC's CamSpec is cut at l < 1000 in TT);
  - best_x the same for the H0WC best fit (its node 12.8125 seeds the chains).
The network's values (logl) are kept: the study's "surrogate" curve is still the network's profile, only the starts
moved. start_override.json records both starts of every node. Nothing is written if the folder exists.
"""
import json
import shutil
import sys
from pathlib import Path

KIT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
P = KIT / "results_profiles"
PP, WC = P / "dncdm_proxy_pp_peak_gamma", P / "dncdm_proxy_wc_peak_gamma"
OUT = P / "dncdm_proxy_pp_peak_gamma_wcstart"
COSMO = ["omega_b", "omega_cdm", "H0", "logA", "n_s", "tau_reio"]


def read(path):
    return json.loads(path.read_text())


for s in (PP, WC):
    if not (s / "complete.json").exists():
        raise SystemExit("FATAL: {} is not a finished study".format(s))
if OUT.exists() or OUT.with_name(OUT.name + ".newton_settings.json").exists():
    raise SystemExit("FATAL: {} exists; this script only makes a new study".format(OUT))

surrogate = read(PP / "surrogate_profiles.json")
manifest = read(PP / "manifest.json")
pp_true, wc_true = read(PP / "true_profiles.json"), read(WC / "true_profiles.json")
names = surrogate["full_parameter_names"]
if pp_true["full_parameter_names"] != names or wc_true["full_parameter_names"][:len(names)] != names:
    raise SystemExit("FATAL: the parameters differ: {} / {}".format(names, wc_true["full_parameter_names"]))
grid = surrogate["profiles"]["log10Gamma"]["grid"]
for other in (pp_true, wc_true):
    g = other["profiles"]["log10Gamma"]["grid"]
    if len(g) != len(grid) or max(abs(a - b) for a, b in zip(g, grid)) > 1e-9:
        raise SystemExit("FATAL: the grids differ")
bounds = [manifest["bounds"][n] for n in names]
cosmo = [names.index(n) for n in COSMO]


def start(pp_x, wc_x, take=cosmo):
    x = list(pp_x[:len(names)])
    for i in take:
        x[i] = wc_x[i]
    for i, (v, (lo, hi)) in enumerate(zip(x, bounds)):
        if not lo <= v <= hi:
            raise SystemExit("FATAL: {} = {} is outside the box [{}, {}]".format(names[i], v, lo, hi))
    return x


pp_xopt, wc_xopt = pp_true["profiles"]["log10Gamma"]["xopt"], wc_true["profiles"]["log10Gamma"]["xopt"]
new = [start(p, w) for p, w in zip(pp_xopt, wc_xopt)]
record = dict(
    made_by="runs/dncdm_proxy_pp/start_from_wc_peak.py",
    from_study=str(PP.relative_to(KIT)), starts_from=str(WC.relative_to(KIT)),
    replaced=COSMO, kept="the PDP+ true optimum's other parameters (Planck nuisances)",
    names=names, grid=grid,
    network_start=surrogate["profiles"]["log10Gamma"]["xopt"], new_start=new,
    network_best_x=surrogate["best_x"],
)
surrogate["profiles"]["log10Gamma"]["xopt"] = new
# the best fit's log10Gamma too: its node (12.8125, the bottom of the H0WC peak) is where the chains begin
surrogate["best_x"] = start(pp_true["best_x"], wc_true["best_x"], take=[names.index("log10Gamma")] + cosmo)
record["new_best_x"] = surrogate["best_x"]

OUT.mkdir()
for name in ("manifest.json", "audit.json", "ranges.json"):
    shutil.copy2(PP / name, OUT / name)
(OUT / "surrogate_profiles.json").write_text(json.dumps(surrogate, indent=1))
(OUT / "start_override.json").write_text(json.dumps(record, indent=1))
settings = PP.with_name(PP.name + ".newton_settings.json")
if settings.exists():
    shutil.copy2(settings, OUT.with_name(OUT.name + ".newton_settings.json"))
print("made {}: {} nodes, chains seeded at log10Gamma {:.4f}".format(OUT.relative_to(KIT), len(new),
                                                                     surrogate["best_x"][names.index("log10Gamma")]))
for j in (0, len(grid) // 2, len(grid) - 1):
    print("  node {:2d} ({:.4f}): ".format(j, grid[j]) + "  ".join(
        "{} {:.5g} -> {:.5g}".format(n, record["network_start"][j][i], new[j][i]) for n, i in zip(COSMO, cosmo)))
