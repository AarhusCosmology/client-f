"""Linear P(k) at z = 0 for the report's figure of the spectra: the run's best fit, two finite points of the
second set's box (a small f_idm and an oscillating one), and five points that gave NaN spectra. Needs CLASS++'s
classy on PYTHONPATH (an unpatched build). Writes data/pk_lin.npz.

    PYTHONPATH=... python compute_pk.py KIT
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from classy import Class

kit = Path(sys.argv[1])
here = Path(__file__).resolve().parent
base = dict(yaml.safe_load((kit / "runs/sidr_drmd2_pp_low/cobaya.yaml").read_text())["theory"]["classy"]["extra_args"])
base.update({"output": "mPk", "non_linear": "none", "P_k_max_1/Mpc": 40, "z_max_pk": 0})
d = pd.read_csv(kit / "results/20261020_122013_sidr_drmd_pp/training_data/data_it_18.csv")
rows = {"best fit": int(d["loglkl"].idxmax()), "f_idm 0.003": 60282, "f_idm 0.39 (finite)": 125969,
        "NaN: row 7111": 7111, "NaN: row 101819": 101819, "NaN: row 67958": 67958, "NaN: row 56202": 56202,
        "NaN: row 73809": 73809}
k = np.logspace(-2, np.log10(40), 700)
out = {"k": k}
for label, row in rows.items():
    r = d.iloc[row]
    pars = dict(base, omega_b=r.omega_b, omega_cdm=r.omega_cdm, H0=r.H0, A_s=1e-10 * np.exp(r.logA), n_s=r.n_s,
                tau_reio=r.tau_reio, delta_Neff_drmd=r.delta_Neff_drmd, f_idm_drmd=r.f_idm_drmd,
                z_stop=10 ** r.log10_zstop, G_over_aH_drmd_ini=10 ** r.log10_G_over_aH,
                **{"nui1.m": r.m_int / 3, "nui1.log10G_eff": r.logGeff})
    c = Class(); c.set(pars); c.compute()
    out[f"pk {label}"] = np.array([c.pk_lin(x, 0) for x in k])
    out[f"par {label}"] = np.array([row, r.delta_Neff_drmd, r.f_idm_drmd, r.log10_zstop, r.log10_G_over_aH])
    c.struct_cleanup(); c.empty()
    print(label, "done", flush=True)
np.savez_compressed(here / "data" / "pk_lin.npz", labels=np.array(list(rows)), **out)
