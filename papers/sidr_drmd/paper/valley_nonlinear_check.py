"""Why the paper leaves out 0 < Delta N_eff < 0.01 (App. DAO, 26 October 2026): the lensing at the best point of that
region on the World Cup data without SH0ES, in linear theory, with halofit and with HMcode 2020 (a local build of CLASS++
bfe6729e, whose _install is $CLASSPP_INSTALL), each relative to the same point with f_idm = 0. Run with the clienv python; prints the numbers
quoted in the appendix. Kept as the record; the paper does not explore the region further.
"""
import os, sys, json, glob
sys.path.insert(0, os.environ["CLASSPP_INSTALL"])
import classy, numpy as np
# the cluster folder whose logs/ hold the polish of the H0WC profiles (by default this folder)
K = os.environ.get("CLIENT_PAPER_KIT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
best = None
for f in glob.glob(K + "logs/chi2_points_wc_polish3_*.json"):
    for e in json.load(open(f))["results"]:
        if e.get("params") and "logpost" in e and (best is None or e["logpost"] > best["logpost"]):
            best = e
p0 = best["params"]
LS = np.array([100, 300, 500, 1000, 1500, 2000]); KS = np.array([0.1, 0.3, 1.0, 3.0])
def run(p, nl):
    c = classy.Class()
    c.set(dict({"omega_b": p["omega_b"], "omega_cdm": p["omega_cdm"], "H0": p["H0"], "A_s": 1e-10 * np.exp(p["logA"]), "n_s": p["n_s"],
                "tau_reio": p["tau_reio"], "N_ur": 0.00441, "nui1.type": "ncdm_self_interacting", "nui1.deg": 3, "nui1.T": 0.71611,
                "nui1.m": p["m_int"] / 3, "nui1.log10G_eff": p["logGeff"], "delta_Neff_drmd": p["delta_Neff_drmd"], "f_idm_drmd": p["f_idm_drmd"],
                "z_stop": 10 ** p["log10_zstop"], "G_over_aH_drmd_ini": 10 ** p["log10_G_over_aH"], "output": "tCl,pCl,lCl,mPk", "lensing": "yes",
                "l_max_scalars": 3000, "P_k_max_1/Mpc": 10.0, "evolver_perturbations": 1, "nonlinear_min_k_max": 10, "z_max_pk": 1.0}, **nl))
    c.compute()
    pp = c.raw_cl(2500)["pp"][LS]
    nlratio = np.array([c.pk(k, 0.5) / c.pk_lin(k, 0.5) for k in KS]) if nl else None
    c.struct_cleanup(); c.empty()
    return pp, nlratio
for label, nl in [("linear (no non-linear correction)", {}), ("halofit", {"non_linear": "halofit"}),
                  ("HMcode 2020", {"non_linear": "hmcode", "hmcode_version": "2020"})]:
    v, vb = run(p0, nl); r, rb = run(dict(p0, f_idm_drmd=0.0), nl)
    line = f"{label:34s} C_pp(valley)/C_pp(f_idm=0) - 1 at L={list(LS)}: {np.round(v / r - 1, 4)}"
    if vb is not None:
        line += f"\n{'':34s} non-linear boost P/P_lin at z=0.5, k={list(KS)}: valley {np.round(vb, 3)}, f_idm=0 {np.round(rb, 3)}"
    print(line)
