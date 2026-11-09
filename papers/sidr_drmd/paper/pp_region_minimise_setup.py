"""True-likelihood best fits of DRMD and the combined model on the Pantheon+ data (Sec. 4) inside the region the paper
leaves out, 0 < Delta N_eff < 0.01 (27 October 2026):

    python pp_region_minimise_setup.py

The paper's discussion compares the best fit there with the one it uses (Delta N_eff >= 0.01). On the World Cup data
without SH0ES both come from minimisations with the true likelihood alone (wc_region_minimise_setup.py). For Pantheon+
the only points with a true value in the region were 29 profile nodes, each with one parameter held fixed; the best of
them (the f_idm node at 0.05) is 2.5 above the global best fit in chi^2, an upper bound only. This writes into the kit:
  smoke/cobaya_pp_dnvalley.yaml                 the run's Cobaya file with the prior of Delta N_eff cut to [0, 0.01]
                                                (analysis/make_cut_run.py's cut_prior; its log-posterior is ln 200
                                                higher than the run's for the same likelihood, build.py puts it back)
  smoke/minimise_{drmd,full}_pp_valley.json     specs for smoke/minimise_true.py (smoke/submit_minimise.sh SPEC NAME)
DRMD has log10 G_eff fixed at -5.5 (11 slow parameters), the combined model frees it (12). Two starts each:
  full: the best and the second-best point in the region, from profiles of two different parameters (both in the
        strongly interacting mode);
  DRMD: the best point in the region with log10 G_eff set to -5.5, and the emulators' DRMD best fit of Table 4
        (Delta N_eff 0.52, weak coupling) moved into the region (Delta N_eff 0.005, f_idm 0.1).
The search coordinates are whitened by the covariance of the run's posterior samples in the region (the posterior
before the cut; for DRMD conditional on log10 G_eff).
"""
import csv
import importlib.util
import json

import numpy as np

import build as B

KIT = B.KIT
D = "pp"
RUN = "runs/sidr_drmd2_pp_combined_boost10"
NUISANCE = ["A_planck", "amp_143", "amp_217", "amp_143x217", "n_143", "n_217", "n_143x217", "calTE", "calEE"]
SLOW = ["omega_b", "omega_cdm", "H0", "logA", "n_s", "tau_reio", "m_int", "delta_Neff_drmd", "f_idm_drmd",
        "log10_zstop", "log10_G_over_aH"]

spec_ = importlib.util.spec_from_file_location("make_cut_run", KIT / "analysis" / "make_cut_run.py")
make_cut_run = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(make_cut_run)


def inside(q):
    return 0 < q["delta_Neff_drmd"] < B.CUT


def run_dir():
    return B.latest(f"results/*_sidr_drmd2_{D}_combined_boost10")


def names():
    return [q["name"] if isinstance(q, dict) else q for q in json.loads((run_dir() / "metadata.json").read_text())["parameters"]]


def region_points(names_):
    """(log-posterior, parameters, origin) of the profile nodes of the first studies in the region, best first."""
    out = []
    for folder, table in B.CUT_NODE_DIRS[D]:
        order = []
        for r in csv.DictReader(open(KIT / table)):
            if r["parameter"] not in order:
                order.append(r["parameter"])
        for f in sorted((KIT / folder).glob("node_*.json")):
            node = json.loads(f.read_text())
            if len(node["x"]) != len(names_) or not np.isfinite(node["logl"]):
                continue
            q = dict(zip(names_, map(float, node["x"])))
            if inside(q):
                prof = order[int(f.stem.split("_")[1])]
                out.append((float(node["logl"]), q, prof, f"{folder.split('/')[-1]}/{f.name} ({prof} = {node['coordinate']:.6g})"))
    return sorted(out, key=lambda t: -t[0])


def covariance(slow, given):
    z = np.load(run_dir() / "analysis" / "best_posterior_it_0.npz", allow_pickle=True)
    zn = [str(n) for n in z["names"]]
    x, w = z["x"].astype(float), z["weights"].astype(float)
    dn = x[:, zn.index("delta_Neff_drmd")]
    keep = (dn > 0) & (dn < B.CUT)
    x, w = x[keep][:, [zn.index(n) for n in slow + given]], w[keep] / w[keep].sum()
    m = (x * w[:, None]).sum(axis=0)
    C = ((x - m).T * w) @ (x - m)
    k = len(slow)
    if given:
        C = C[:k, :k] - C[:k, k:] @ np.linalg.solve(C[k:, k:], C[k:, :k])
    print(f"   covariance from {int(keep.sum())} posterior samples in the region ({100 * z['weights'][keep].sum() / z['weights'].sum():.1f}% of the weight)")
    return C[:k, :k]


def main():
    names_ = names()
    base = (KIT / RUN / "cobaya.yaml").read_text()
    path = KIT / "smoke" / f"cobaya_{D}_dnvalley.yaml"
    path.write_text(f"# {path.name}: {RUN}/cobaya.yaml with Delta N_eff in [0, {B.CUT}] (the paper's pp_region_minimise_setup.py, "
                    f"27 October 2026)\n" + make_cut_run.cut_prior(base, "delta_Neff_drmd", None, B.CUT))
    pts = region_points(names_)
    best = pts[0]
    second = next(t for t in pts[1:] if t[2] != best[2])
    drmd_fit = json.loads((B.BESTFITS["boost10"] / "aic_preview.json").read_text())[D]["DRMD"]["x"]

    def start(label, q, note, **moved):
        return dict(label=label, params=dict({n: float(q[n]) for n in names_}, **moved),
                    note=note + (f", moved to {moved}" if moved else ""))
    starts = {
        "full": [start("region_best", best[1], f"{best[3]} (true ln L {best[0]:.2f})"),
                 start("region_second", second[1], f"{second[3]} (true ln L {second[0]:.2f})")],
        "drmd": [start("region_best_weak", best[1], f"{best[3]} (true ln L {best[0]:.2f})", logGeff=-5.5),
                 start("emulator_drmd_moved", drmd_fit, "the emulators' DRMD best fit of Table 4 (boost10)",
                       delta_Neff_drmd=0.005, f_idm_drmd=0.1)],
    }
    for model, st in starts.items():
        slow = SLOW + (["logGeff"] if model == "full" else [])
        fixed = {"logGeff": -5.5} if model == "drmd" else {}
        for s in st:
            assert inside(s["params"]), (model, s["label"])
        C = covariance(slow, list(fixed))
        spec = dict(made="pp_region_minimise_setup.py", cobaya=f"smoke/{path.name}", job_env=f"{RUN}/job.env", fixed=fixed,
                    nuisance=NUISANCE, scales={n: float(np.sqrt(C[i, i])) for i, n in enumerate(slow)},
                    covariance=dict(names=slow, matrix=C.tolist()), starts=st)
        out = KIT / "smoke" / f"minimise_{model}_{D}_valley.json"
        out.write_text(json.dumps(spec, indent=1))
        print(f"{out.relative_to(KIT)}:")
        for s in st:
            q = s["params"]
            print(f"   {s['label']:20s} dNeff {q['delta_Neff_drmd']:.4f} f_idm {q['f_idm_drmd']:.3f} logGeff {q['logGeff']:.2f} "
                  f"H0 {q['H0']:.2f} n_s {q['n_s']:.4f} <- {s['note']}")
    print(f"smoke/{path.name}: prior of Delta N_eff [0, {B.CUT}]")


if __name__ == "__main__":
    main()
