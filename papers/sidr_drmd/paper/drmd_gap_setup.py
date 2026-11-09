"""Why our DRMD fits the World Cup data with SH0ES better than the H0 World Cup's: the set-ups (26 October 2026).

    python drmd_gap_setup.py

Sec. 5.5: our DRMD lowers chi^2 by 12.0 relative to LCDM on the World Cup data with SH0ES, the World Cup's by 6.86
(paper II, Table 5). Two differences we can test with our own likelihood:
  - the neutrino mass: our m_int (the sum of three degenerate masses) has the prior [0.06, 0.3] eV and every best fit
    sits at 0.06; the World Cup lets the sum go to 0 (paper II, Table 1), and DESI DR2 pulls it down. If the floor costs
    LCDM more than DRMD, it inflates our Delta chi^2.
  - the non-linear power spectrum: we use halofit, the World Cup HMcode 2020 (paper II, Sec. V.5 marginalises over
    HMcode 2020's T_AGN, "neglected in the baseline").
It also checks our LCDM best fit, which is the emulators' and was never minimised with the true likelihood as DRMD's
was. Writes into the kit:
  smoke/cobaya_wcmb_hmcode2020.yaml   the run's Cobaya file with non_linear hmcode, hmcode_version 2020
  smoke/cobaya_wcmb_mnulow.yaml       the run's Cobaya file with m_int down to 0.003 eV (1 meV each)
  smoke/chi2_points_drmd_gap.json     sets wcmb_halofit and wcmb_hmcode2020: the four best fits of the AIC table, the
                                      nuisance parameters re-optimised at each point (smoke/submit_chi2_points.sh)
  smoke/minimise_lcdm_wcmb.json       LCDM (Delta N_eff = f_idm = 0, log10 G_eff = -5.5) with the true likelihood
  smoke/minimise_lcdm_wcmb_mnulow.json   the same with the lower mass floor
  smoke/minimise_drmd_wcmb_mnulow.json   DRMD (log10 G_eff = -5.5) with the lower mass floor
(smoke/submit_minimise.sh SPEC NAME), as drmd_minimise_points.py does for DRMD with the floor at 0.06.
"""
import json
import re

import numpy as np

import build_wc as W

KIT = W.KIT
D = "wcmb"
RUN = f"runs/sidr_drmd_{D}_combined_boost10"
NUISANCE = ["A_planck", "amp_143", "amp_217", "amp_143x217", "n_143", "n_217", "n_143x217", "calTE", "calEE",
            "P_act", "Tcal", "Ecal", "Mb"]
LCDM_SLOW = ["omega_b", "omega_cdm", "H0", "logA", "n_s", "tau_reio", "m_int"]
DRMD_SLOW = LCDM_SLOW + ["delta_Neff_drmd", "f_idm_drmd", "log10_zstop", "log10_G_over_aH"]
LCDM_FIXED = {"logGeff": -5.5, "delta_Neff_drmd": 0.0, "f_idm_drmd": 0.0, "log10_zstop": 4.0, "log10_G_over_aH": 8.0}
MNU_LOW = 0.003


def variant(name, edit):
    text = (KIT / RUN / "cobaya.yaml").read_text()
    new = edit(text)
    assert new != text, name
    path = KIT / "smoke" / name
    path.write_text(f"# {name}: {RUN}/cobaya.yaml changed by the paper's drmd_gap_setup.py (26 October 2026)\n" + new)
    return f"smoke/{name}"


def hmcode(text):
    out, n = re.subn(r"(?m)^(\s+)non_linear: halofit.*$", r"\1non_linear: hmcode\n\1hmcode_version: '2020'", text)
    assert n == 1
    return out


def mnulow(text):
    out, n = re.subn(r"(?m)^(  m_int:\n    prior:\n      min: )0\.06$", rf"\g<1>{MNU_LOW}", text)
    assert n == 1
    return out


def conditional_covariance(slow, fixed):
    """Covariance of `slow` in the S+10 posterior, conditional on the parameters `fixed` (a Gaussian approximation)."""
    run = W.latest(f"results/*_sidr_drmd_{D}_combined_boost10")
    z = np.load(run / "analysis" / "best_posterior_it_0.npz", allow_pickle=True)
    zn = [str(n) for n in z["names"]]
    cols = [zn.index(n) for n in slow + fixed]
    x, w = z["x"][:, cols].astype(float), z["weights"] / z["weights"].sum()
    m = (x * w[:, None]).sum(axis=0)
    C = ((x - m).T * w) @ (x - m)
    k = len(slow)
    cond = C[:k, :k] - C[:k, k:] @ np.linalg.solve(C[k:, k:], C[k:, :k])
    return cond


def spec(cobaya, fixed, slow, starts, made_for):
    fixed_names = [n for n in fixed if n not in slow]
    cond = conditional_covariance(slow, fixed_names)
    return dict(made="drmd_gap_setup.py", purpose=made_for, cobaya=cobaya, job_env=f"{RUN}/job.env", fixed=fixed,
                nuisance=NUISANCE, scales={n: float(np.sqrt(cond[i, i])) for i, n in enumerate(slow)},
                covariance=dict(names=slow, matrix=cond.tolist()), starts=starts)


def main():
    hm = variant("cobaya_wcmb_hmcode2020.yaml", hmcode)
    low = variant("cobaya_wcmb_mnulow.yaml", mnulow)
    fits = W.chi2_points(D, "bestfits", merge=True)["results"]
    names = [q["name"] if isinstance(q, dict) else q for q in json.loads(
        (W.latest(f"results/*_sidr_drmd_{D}_combined_boost10") / "metadata.json").read_text())["parameters"]]
    points = [dict(label=k, params={n: float(fits[k]["params"][n]) for n in names}, logl=float(fits[k]["logpost"]),
                   note=f"best fit of {k} in the AIC table ({fits[k]['fit_label']})")
              for k in ("LCDM", "DRMD", "full", "SInu_best")]
    sets = {"wcmb_halofit": dict(cobaya=f"{RUN}/cobaya.yaml", job_env=f"{RUN}/job.env", optimise=NUISANCE, points=points),
            "wcmb_hmcode2020": dict(cobaya=hm, job_env=f"{RUN}/job.env", optimise=NUISANCE, points=points)}
    (KIT / "smoke" / "chi2_points_drmd_gap.json").write_text(json.dumps(dict(made="drmd_gap_setup.py", sets=sets), indent=1))

    # LCDM starts: the LCDM fits of the three emulators (S+1, S+5, S+10), with the DRMD parameters at their fixed values
    lcdm = []
    for label in ("boost1", "boost5", "boost10"):
        s = json.loads((KIT / "logs" / f"aic_wc_{label}" / f"chi2_points_wc_{label}_bestfits.json").read_text())["sets"][f"{D}_bestfits"]
        p = next(q["params"] for q in s["points"] if q["label"] == "LCDM")
        lcdm.append(dict(label=f"lcdm_{label}", params=dict({n: float(p[n]) for n in names}, **LCDM_FIXED),
                         note=f"the LCDM fit of the {label} emulator"))
    drmd = dict(fits["DRMD"]["params"])
    drmd_starts = [dict(label="drmd_direct", params={n: float(drmd[n]) for n in names},
                        note=f"DRMD's best fit with m_int >= 0.06 ({fits['DRMD']['fit_label']})"),
                   dict(label="drmd_direct_light", params=dict({n: float(drmd[n]) for n in names}, m_int=0.02),
                        note="the same with m_int = 0.02")]
    light = [dict(q, label=q["label"] + "_light", params=dict(q["params"], m_int=0.02)) for q in lcdm[2:]]
    out = {"minimise_lcdm_wcmb.json": spec(f"{RUN}/cobaya.yaml", LCDM_FIXED, LCDM_SLOW, lcdm,
                                           "LCDM with the true likelihood, m_int >= 0.06 (is the emulators' LCDM fit high?)"),
           "minimise_lcdm_wcmb_mnulow.json": spec(low, LCDM_FIXED, LCDM_SLOW, lcdm[2:] + light,
                                                  f"LCDM with the true likelihood, m_int >= {MNU_LOW}"),
           "minimise_drmd_wcmb_mnulow.json": spec(low, {"logGeff": -5.5}, DRMD_SLOW, drmd_starts,
                                                  f"DRMD with the true likelihood, m_int >= {MNU_LOW}")}
    for name, s in out.items():
        (KIT / "smoke" / name).write_text(json.dumps(s, indent=1))
        print(f"smoke/{name}: {len(s['starts'])} starts, {len(s['scales'])} slow parameters, fixed {s['fixed']}")
    print(f"smoke/chi2_points_drmd_gap.json: {', '.join(sets)} ({len(points)} points each); {hm}; {low}")


if __name__ == "__main__":
    main()
