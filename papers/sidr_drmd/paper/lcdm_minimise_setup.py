"""LCDM on the World Cup data without SH0ES, minimised with the true likelihood alone (26 October 2026):

    python lcdm_minimise_setup.py

With SH0ES the minimisation of LCDM with the true likelihood (drmd_gap_setup.py, kit job 6962213) came out 2.2 lower
in chi^2 than the emulators' LCDM fit of the AIC table, and every Delta chi^2 of that column is measured from it. This
makes the same minimisation without SH0ES, so that both columns of Table 5 have the true LCDM minimum as reference.
Writes the kit's smoke/minimise_lcdm_wc.json (smoke/submit_minimise.sh SPEC NAME): LCDM (Delta N_eff = f_idm = 0,
log10 G_eff = -5.5) with the run's own Cobaya file, starting from the LCDM fits of the three emulators (S+1, S+5,
S+10), the seven slow parameters whitened by the S+10 posterior's covariance conditional on the fixed ones, as
drmd_gap_setup.py does with SH0ES.
"""
import json

import numpy as np

import build_wc as W

KIT = W.KIT
D = "wc"
RUN = f"runs/sidr_drmd_{D}_combined_boost10"
NUISANCE = ["A_planck", "amp_143", "amp_217", "amp_143x217", "n_143", "n_217", "n_143x217", "calTE", "calEE",
            "P_act", "Tcal", "Ecal"]
SLOW = ["omega_b", "omega_cdm", "H0", "logA", "n_s", "tau_reio", "m_int"]
FIXED = {"logGeff": -5.5, "delta_Neff_drmd": 0.0, "f_idm_drmd": 0.0, "log10_zstop": 4.0, "log10_G_over_aH": 8.0}


def conditional_covariance(slow, fixed):
    """Covariance of `slow` in the S+10 posterior (all of it: the cut does not concern LCDM), conditional on `fixed`."""
    run = W.latest(f"results/*_sidr_drmd_{D}_combined_boost10")
    z = np.load(run / "analysis" / "best_posterior_it_0.npz", allow_pickle=True)
    zn = [str(n) for n in z["names"]]
    x, w = z["x"][:, [zn.index(n) for n in slow + fixed]].astype(float), z["weights"] / z["weights"].sum()
    m = (x * w[:, None]).sum(axis=0)
    C = ((x - m).T * w) @ (x - m)
    k = len(slow)
    return C[:k, :k] - C[:k, k:] @ np.linalg.solve(C[k:, k:], C[k:, :k])


def main():
    names = [q["name"] if isinstance(q, dict) else q for q in json.loads(
        (W.latest(f"results/*_sidr_drmd_{D}_combined_boost10") / "metadata.json").read_text())["parameters"]]
    starts = []
    for label in ("boost1", "boost5", "boost10"):
        s = json.loads((KIT / "logs" / f"aic_wc_{label}" / f"chi2_points_wc_{label}_bestfits.json").read_text())["sets"][f"{D}_bestfits"]
        p = next(q["params"] for q in s["points"] if q["label"] == "LCDM")
        starts.append(dict(label=f"lcdm_{label}", params=dict({n: float(p[n]) for n in names}, **FIXED),
                           note=f"the LCDM fit of the {label} emulator"))
    C = conditional_covariance(SLOW, list(FIXED))
    spec = dict(made="lcdm_minimise_setup.py", purpose="LCDM with the true likelihood, m_int >= 0.06 (the reference of Table 5)",
                cobaya=f"{RUN}/cobaya.yaml", job_env=f"{RUN}/job.env", fixed=FIXED, nuisance=NUISANCE,
                scales={n: float(np.sqrt(C[i, i])) for i, n in enumerate(SLOW)},
                covariance=dict(names=SLOW, matrix=C.tolist()), starts=starts)
    out = KIT / "smoke" / f"minimise_lcdm_{D}.json"
    out.write_text(json.dumps(spec, indent=1))
    print(f"{out.relative_to(KIT)}: {len(starts)} starts, {len(SLOW)} slow parameters, fixed {FIXED}")


if __name__ == "__main__":
    main()
