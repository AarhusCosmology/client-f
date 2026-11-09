"""True-likelihood best fits of DRMD and the combined model on the World Cup data without SH0ES, in the two regions of
Delta N_eff (26 October 2026):

    python wc_region_minimise_setup.py

Without SH0ES the best fit lies in a narrow valley at Delta N_eff ~ 3e-4 (Sec. 5.5). The paper treats Delta N_eff >= 0.01
(the cut) and the valley, Delta N_eff <= 0.01, separately; the best fits of the AIC table for DRMD and the combined model
are emulator fits that lie in the valley. This writes into the kit:
  smoke/cobaya_wc_dncut.yaml, smoke/cobaya_wc_dnvalley.yaml   the run's Cobaya file with the prior of Delta N_eff cut
                                                              (analysis/make_cut_run.py's cut_prior)
  smoke/minimise_{drmd,full}_wc_{cut,valley}.json             specs for smoke/minimise_true.py (smoke/submit_minimise.sh)
DRMD has log10 G_eff fixed at -5.5 (11 slow parameters), the combined model frees it (12). The starts are the best true
values found in the region (profile nodes, polish and plateau points, the AIC fits), two per spec from different
sources; the search coordinates are whitened by the covariance of the pooled posterior's samples in the region
(conditional on log10 G_eff for DRMD).
"""
import importlib.util
import json

import numpy as np

import build_wc as W

KIT = W.KIT
D = "wc"
RUN = f"runs/sidr_drmd_{D}_combined_boost10"
NUISANCE = ["A_planck", "amp_143", "amp_217", "amp_143x217", "n_143", "n_217", "n_143x217", "calTE", "calEE",
            "P_act", "Tcal", "Ecal"]
SLOW = ["omega_b", "omega_cdm", "H0", "logA", "n_s", "tau_reio", "m_int", "delta_Neff_drmd", "f_idm_drmd",
        "log10_zstop", "log10_G_over_aH"]
CUT = 0.01
REGIONS = {"cut": (CUT, None), "valley": (None, CUT)}

spec_ = importlib.util.spec_from_file_location("make_cut_run", KIT / "analysis" / "make_cut_run.py")
make_cut_run = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(make_cut_run)


def inside(region, p):
    dn = p["delta_Neff_drmd"]
    return dn >= CUT if region == "cut" else 0 < dn <= CUT


def candidates(names):
    """(log-posterior, parameters, origin) of every true value of D with known parameters."""
    out = [(l, q, "profile node") for _, _, q, l in W.node_points(D, names)]
    out += [(e["logpost"], {n: float(q[n]) for n in names}, f"{f.name} {e.get('label')}")
            for e, q, f in W.moved_results(D) if q is not None and "logpost" in e]
    for label in ("boost1", "boost5", "boost10"):
        path = KIT / "logs" / f"aic_wc_{label}" / f"chi2_points_wc_{label}_bestfits.json"
        if not path.exists():
            continue
        pts = json.loads(path.read_text())["sets"][f"{D}_bestfits"]["points"]
        res = {}
        for f in sorted(KIT.glob(f"logs/chi2_points_{D}_bestfits_*.json")):
            r = json.loads(f.read_text())
            if str(r.get("network", "")).endswith(f"_{label}"):
                res = {e["label"]: e for e in r["results"] if "logpost" in e}
        for q in pts:
            if q["label"] in res:
                out.append((res[q["label"]]["logpost"], {n: float(q["params"][n]) for n in names}, f"AIC {q['label']} ({label})"))
    return out


def covariance(region, slow, given):
    run = W.latest(f"results/*_sidr_drmd_{D}_combined_boost10")
    z = np.load(run / "analysis" / "best_posterior_it_0.npz", allow_pickle=True)
    zn = [str(n) for n in z["names"]]
    x, w = z["x"].astype(float), z["weights"].astype(float)
    keep = np.array([inside(region, {"delta_Neff_drmd": v}) for v in x[:, zn.index("delta_Neff_drmd")]])
    x, w = x[keep][:, [zn.index(n) for n in slow + given]], w[keep] / w[keep].sum()
    m = (x * w[:, None]).sum(axis=0)
    C = ((x - m).T * w) @ (x - m)
    k = len(slow)
    if given:
        C = C[:k, :k] - C[:k, k:] @ np.linalg.solve(C[k:, k:], C[k:, :k])
    return C[:k, :k]


def main():
    names = [q["name"] if isinstance(q, dict) else q for q in json.loads(
        (W.latest(f"results/*_sidr_drmd_{D}_combined_boost10") / "metadata.json").read_text())["parameters"]]
    base = (KIT / RUN / "cobaya.yaml").read_text()
    yamls = {}
    for region, (lo, hi) in REGIONS.items():
        path = KIT / "smoke" / f"cobaya_{D}_dn{region}.yaml"
        path.write_text(f"# {path.name}: {RUN}/cobaya.yaml with Delta N_eff in "
                        f"[{lo if lo is not None else 0}, {hi if hi is not None else 2}] (wc_region_minimise_setup.py, 26 October 2026)\n"
                        + make_cut_run.cut_prior(base, "delta_Neff_drmd", lo, hi))
        yamls[region] = f"smoke/{path.name}"
    cands = candidates(names)

    def best(region, predicate=lambda q: True, skip=()):
        pool = sorted((t for t in cands if inside(region, t[1]) and predicate(t[1]) and t[2] not in skip), key=lambda t: -t[0])
        return pool[0]

    def aic(label, key):
        return next(t for t in cands if t[2] == f"AIC {key} ({label})")

    def start(label, t, **moved):
        l, q, origin = t
        return dict(label=label, params=dict({n: float(q[n]) for n in names}, **moved),
                    note=f"{origin} (true ln L {l:.2f})" + (f", moved to {moved}" if moved else ""))

    cut_best, valley_best = best("cut"), best("valley")
    lcdm = aic("boost10", "LCDM")
    starts = {
        ("full", "cut"): [start("broad_best", cut_best), start("emulator_full_s5", aic("boost5", "full"))],
        ("full", "valley"): [start("valley_best", valley_best), start("emulator_full_s1", aic("boost1", "full"))],
        ("drmd", "cut"): [start("broad_best_weak", cut_best, logGeff=-5.5),
                          start("lcdm_plus_dr", lcdm, delta_Neff_drmd=0.05, f_idm_drmd=0.01, log10_zstop=4.4, log10_G_over_aH=8.0)],
        ("drmd", "valley"): [start("emulator_drmd_s5", aic("boost5", "DRMD")),
                             start("valley_best_weak", best("valley", lambda q: q["logGeff"] < -4.5), logGeff=-5.5)],
    }
    for (model, region), st in starts.items():
        slow = SLOW + (["logGeff"] if model == "full" else [])
        fixed = {"logGeff": -5.5} if model == "drmd" else {}
        for x in st:
            assert inside(region, x["params"]), (model, region, x["label"])
        C = covariance(region, slow, list(fixed))
        spec = dict(made="wc_region_minimise_setup.py", cobaya=yamls[region], job_env=f"{RUN}/job.env", fixed=fixed,
                    nuisance=NUISANCE, scales={n: float(np.sqrt(C[i, i])) for i, n in enumerate(slow)},
                    covariance=dict(names=slow, matrix=C.tolist()), starts=st)
        out = KIT / "smoke" / f"minimise_{model}_{D}_{region}.json"
        out.write_text(json.dumps(spec, indent=1))
        print(f"{out.relative_to(KIT)}:")
        for x in st:
            q = x["params"]
            print(f"   {x['label']:18s} dNeff {q['delta_Neff_drmd']:.4f} f_idm {q['f_idm_drmd']:.3f} logGeff {q['logGeff']:.2f} H0 {q['H0']:.2f} <- {x['note']}")


if __name__ == "__main__":
    main()
