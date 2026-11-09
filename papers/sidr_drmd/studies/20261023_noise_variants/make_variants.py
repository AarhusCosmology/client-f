"""The variants of the noise probe on Grendel (kit smoke/noise_probe_cobaya.py, smoke/submit_noise_variants.sh).

    python results/20261023_noise_variants/make_variants.py

Question (Steen, 23 October 2026): is the noise of the SIDR + DRMD true likelihood, 0.10 in ln L for Pantheon+ (paper,
Appendix "The noise floor of the true likelihood"), particular to the massive self-interacting neutrinos and the DRMD
sector? All variants use the Pantheon+ run's likelihoods, its CLASS++ precision settings (evolver_perturbations 1,
tol_perturb_integration 1e-7, halofit) and its patched CLASS++. Each one switches off one ingredient:

  lcdm_massless   LCDM with massless neutrinos (N_ur = 3.044)
  lcdm_mnu        LCDM with three standard (non-interacting) massive neutrinos, CLASS's ncdm, sum 0.08 eV
  sinu_off        the SIDR + DRMD code on its LCDM face: log10 Geff = -5.5 (no interaction on observable scales),
                  Delta Neff = f_idm = 0, sum of the masses 0.08 eV
  sinu_strong     the SInu face in the strongly interacting mode (the emulator best fit of SInu, CLASS++ checked)
  full            the full model at its global best fit (the best profile node)

The first three share a centre: the emulator best fit on the LCDM face (CLASS++ checked; kit smoke/chi2_points_boost10
.json). The neutrino mass is 0.08 eV in every variant that has massive neutrinos. The best fits sit at the lower edge of
its prior, 0.06, where a line through them would leave the prior. Lines are +-0.25 posterior sigma, the sigma of the BEST
posterior of the Pantheon+ combined emulator. For log10 Geff it is the sigma of the strong mode alone; the
whole posterior's 1.2 spans both modes. Writes runs/noise_lcdm_massless/cobaya.yaml, runs/noise_lcdm_mnu/cobaya.yaml
and smoke/noise_variants.json in the kit.
"""
import copy
import json
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
KIT = REPO / "results" / "20261007_grendel_kit" / "hpc_client"
PAPER = REPO / "results" / "20261022_preprint_client"
SIDR = "runs/sidr_drmd2_pp_combined_boost10/cobaya.yaml"
LCDM_PARAMS = ["omega_b", "omega_cdm", "H0", "logA", "n_s", "tau_reio"]
NUISANCE = ["A_planck", "amp_143", "amp_217", "amp_143x217", "n_143", "n_217", "n_143x217", "calTE", "calEE"]
EXTENSION = ["m_int", "nui1.m", "logGeff", "nui1.log10G_eff", "delta_Neff_drmd", "f_idm_drmd", "log10_zstop", "z_stop",
             "log10_G_over_aH", "G_over_aH_drmd_ini", "f_idr_drmd", "log10_zdec"]
MASS = 0.08


def lcdm_yaml(massive):
    """The SIDR + DRMD Cobaya file with the self-interacting neutrinos and DRMD taken out: the same likelihoods,
    nuisance priors, precision settings and derived parameters; massless or standard massive neutrinos."""
    info = yaml.safe_load((KIT / SIDR).read_text())
    extra = info["theory"]["classy"]["extra_args"]
    for k in [k for k in extra if k.startswith("nui1.")]:
        del extra[k]
    params = info["params"]
    for k in EXTENSION:
        params.pop(k, None)
    if massive:
        extra.update({"N_ur": 0.00441, "N_ncdm": 1, "deg_ncdm": 3, "T_ncdm": 0.71611})
        sidr = yaml.safe_load((KIT / SIDR).read_text())["params"]
        params["m_int"] = copy.deepcopy(sidr["m_int"])
        params["m_ncdm"] = {"value": "lambda m_int: m_int/3", "derived": False}
    else:
        extra["N_ur"] = 3.044
    return info


def main():
    header = ("# Made by results/20261023_noise_variants/make_variants.py (23 October 2026) from {sidr}: the same likelihoods,\n"
              "# nuisance priors, CLASS++ precision settings and derived parameters, with the self-interacting neutrinos and\n"
              "# DRMD taken out; {nu}. For the noise probe (smoke/submit_noise_variants.sh), not for a CLiENT run.\n")
    for label, massive, nu in (("noise_lcdm_massless", False, "massless neutrinos (N_ur = 3.044)"),
                               ("noise_lcdm_mnu", True, "three standard massive neutrinos (CLASS ncdm, m_ncdm = m_int / 3)")):
        folder = KIT / "runs" / label
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "cobaya.yaml").write_text(header.format(sidr=SIDR, nu=nu) + yaml.safe_dump(lcdm_yaml(massive), sort_keys=False))

    points = {p["label"]: p["params"] for p in json.loads((KIT / "smoke" / "chi2_points_boost10.json").read_text())["data"]["pp"]["points"]}
    best = json.loads((PAPER / "plots" / "data" / "best_fits.json").read_text())["pp"]
    run = sorted(KIT.glob("results/*_sidr_drmd2_pp_combined_boost10"))[-1]
    summary = {p["name"]: p for p in json.loads((run / "analysis" / "best_posterior_it_0.json").read_text())["parameters"]}
    sigma = {k: v["sd"] for k, v in summary.items()}
    z = np.load(run / "analysis" / "best_posterior_it_0.npz", allow_pickle=True)
    names = [str(n) for n in z["names"]]
    g, w = z["x"][:, names.index("logGeff")], z["weights"]
    strong = g > -2.5
    mu = np.average(g[strong], weights=w[strong])
    sigma["logGeff"] = float(np.sqrt(np.average((g[strong] - mu) ** 2, weights=w[strong])))

    face = dict(points["LCDM"], m_int=MASS)
    full = {k: best["values"][k] for k in names}
    full["m_int"] = MASS
    variants = {
        "lcdm_massless": dict(cobaya="runs/noise_lcdm_massless/cobaya.yaml", description="LCDM, massless neutrinos",
                              centre={k: face[k] for k in LCDM_PARAMS + NUISANCE}, vary=LCDM_PARAMS + NUISANCE),
        "lcdm_mnu": dict(cobaya="runs/noise_lcdm_mnu/cobaya.yaml",
                         description=f"LCDM, three standard massive neutrinos, sum {MASS} eV",
                         centre={k: face[k] for k in LCDM_PARAMS + ["m_int"] + NUISANCE}, vary=LCDM_PARAMS + ["m_int"] + NUISANCE),
        "sinu_off": dict(cobaya=SIDR, description=f"SIDR + DRMD on the LCDM face (log10 Geff = -5.5, Delta Neff = f_idm = 0), sum {MASS} eV",
                         centre=face, vary=LCDM_PARAMS + ["m_int"] + NUISANCE),
        "sinu_strong": dict(cobaya=SIDR, description="SInu face, strongly interacting mode (Delta Neff = f_idm = 0)",
                            centre=points["SInu_strong"], vary=LCDM_PARAMS + ["m_int", "logGeff"] + NUISANCE),
        "full": dict(cobaya=SIDR, description=f"full model at its global best fit (m_int moved to {MASS} eV)", centre=full,
                     vary=names),
    }
    for v in variants.values():
        v["sigma"] = {k: float(sigma[k]) for k in v["vary"]}
    out = dict(made="23 October 2026, results/20261023_noise_variants/make_variants.py", data="pp", variants=variants)
    (KIT / "smoke" / "noise_variants.json").write_text(json.dumps(out, indent=1))
    for label, v in variants.items():
        print(f"{label:14s} {len(v['vary']):2d} varied  {v['description']}")
    print("written:", KIT / "smoke" / "noise_variants.json", "and runs/noise_lcdm_{massless,mnu}/cobaya.yaml")


if __name__ == "__main__":
    main()
