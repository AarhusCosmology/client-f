"""The starts of the true-likelihood minimisation of DRMD on the World Cup data with SH0ES (26 October 2026).

    python drmd_minimise_points.py

The combined model's best fit is 17 units of chi^2 below DRMD's (Sec. 5.5); DRMD's best fit is the best of the
emulators' fits, evaluated with CLASS++. This writes the kit's smoke/minimise_drmd_wcmb.json for smoke/minimise_true.py:
log10 Geff fixed at -5.5 (DRMD), the 13 nuisance parameters optimised at every point, the other 11 parameters searched
in coordinates whitened by the covariance of the S+10 posterior conditional on log10 Geff, from four starts:
    drmd_fit         the best DRMD fit of the AIC jobs (merged)
    drmd_fit_s1      the DRMD fit of the S+1 emulator (another network)
    plateau_best     the best true point found with log10 Geff < -4 (plateau and polish jobs), moved to -5.5
    full_best_moved  the combined model's global best fit with log10 Geff moved to -5.5
"""
import json

import numpy as np

import build_wc as W
import profile_polish as PP

KIT = W.KIT
D = "wcmb"
NUISANCE = ["A_planck", "amp_143", "amp_217", "amp_143x217", "n_143", "n_217", "n_143x217", "calTE", "calEE",
            "P_act", "Tcal", "Ecal", "Mb"]
SLOW = ["omega_b", "omega_cdm", "H0", "logA", "n_s", "tau_reio", "m_int", "delta_Neff_drmd", "f_idm_drmd",
        "log10_zstop", "log10_G_over_aH"]


def main():
    run = W.latest(f"results/*_sidr_drmd_{D}_combined_boost10")
    names = [q["name"] if isinstance(q, dict) else q for q in json.loads((run / "metadata.json").read_text())["parameters"]]
    z = np.load(run / "analysis" / "best_posterior_it_0.npz", allow_pickle=True)
    zn = [str(n) for n in z["names"]]
    cols = [zn.index(n) for n in SLOW + ["logGeff"]]
    x, w = z["x"][:, cols].astype(float), z["weights"] / z["weights"].sum()
    mean = (x * w[:, None]).sum(axis=0)
    C = ((x - mean).T * w) @ (x - mean)
    k = len(SLOW)
    cond = C[:k, :k] - np.outer(C[:k, k], C[k, :k]) / C[k, k]          # given log10 Geff
    scales = {n: float(np.sqrt(cond[i, i])) for i, n in enumerate(SLOW)}

    fits = W.chi2_points(D, "bestfits", merge=True)["results"]
    s1 = json.loads((KIT / "logs" / "aic_wc_boost1" / "chi2_points_wc_boost1_bestfits.json").read_text())["sets"][f"{D}_bestfits"]
    s1_drmd = next(q["params"] for q in s1["points"] if q["label"] == "DRMD")
    moved = PP.moved_points(D, names) + PP.sources(D)
    plateau = max((m for m in moved if m["params"]["logGeff"] < -4.0), key=lambda m: m["logl"])
    star = json.loads((W.PLOT_DATA / "best_fits.json").read_text())[D]
    full_best = {n: float(star["values"][n]) for n in names}

    def start(label, params, note):
        return dict(label=label, params={n: float(params[n]) for n in names}, note=note)
    starts = [start("drmd_fit", fits["DRMD"]["params"], f"best DRMD fit of the AIC jobs ({fits['DRMD']['fit_label']})"),
              start("drmd_fit_s1", s1_drmd, "the DRMD fit of the S+1 emulator"),
              start("plateau_best", dict(plateau["params"], logGeff=-5.5),
                    f"{plateau['origin']} (true ln L {plateau['logl']:.2f} at log10 Geff {plateau['params']['logGeff']:.2f})"),
              start("full_best_moved", dict(full_best, logGeff=-5.5), f"the global best fit ({star['source']})")]
    spec = dict(made="drmd_minimise_points.py", cobaya=f"runs/sidr_drmd_{D}_combined_boost10/cobaya.yaml",
                job_env=f"runs/sidr_drmd_{D}_combined_boost10/job.env", fixed={"logGeff": -5.5}, nuisance=NUISANCE,
                scales=scales, covariance=dict(names=SLOW, matrix=cond.tolist()), starts=starts,
                reference=dict(drmd_chi2=fits["DRMD"]["chi2_data"], full_chi2=fits["full"]["chi2_data"],
                               global_best_logpost=star["logl"]))
    path = KIT / "smoke" / "minimise_drmd_wcmb.json"
    path.write_text(json.dumps(spec, indent=1))
    for s in starts:
        print(f"{s['label']:16s} H0 {s['params']['H0']:.2f} dNeff {s['params']['delta_Neff_drmd']:.3f} n_s {s['params']['n_s']:.4f} : {s['note']}")
    print("scales:", {n: round(v, 4) for n, v in scales.items()})
    print("written:", path)


if __name__ == "__main__":
    main()
