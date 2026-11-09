"""External primordial spectrum (alpha_s = beta_s = 0) against CLASS++'s analytic one at the best fit; timing."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mimic
from classy import Class

b = json.loads(mimic.BEST.read_text())["wcmb"]["values"]
p = {n: b[n] for n in ("omega_b", "omega_cdm", "H0", "n_s", "tau_reio", "m_int", "logGeff", "delta_Neff_drmd",
                       "f_idm_drmd", "log10_zstop", "log10_G_over_aH")}
p.update(ln10A=b["logA"], alpha_s=0.0, beta_s=0.0)
ext = mimic.spectra(p, 5000)
d = dict(mimic.EXTRA)
d.update({"omega_b": p["omega_b"], "omega_cdm": p["omega_cdm"], "tau_reio": p["tau_reio"], "H0": p["H0"],
          "A_s": 1e-10 * np.exp(p["ln10A"]), "n_s": p["n_s"], "nui1.m": p["m_int"] / 3, "nui1.log10G_eff": p["logGeff"],
          "delta_Neff_drmd": p["delta_Neff_drmd"], "f_idm_drmd": p["f_idm_drmd"], "z_stop": 10 ** p["log10_zstop"],
          "G_over_aH_drmd_ini": 10 ** p["log10_G_over_aH"], "output": "tCl,pCl,lCl,mPk", "lensing": "yes",
          "l_max_scalars": 5000, "P_k_max_1/Mpc": 10.0, "z_pk": 0.0})
M = Class(); M.set(d); t = time.time(); M.compute(); dt = time.time() - t
cl = M.lensed_cl(5000); T0 = M.T_cmb() * 1e6
for x in ("tt", "ee"):
    r = ext[x][30:4001] / (cl[x][30:4001] * T0**2) - 1
    print(f"{x}: max |external/analytic - 1| for 30 <= l <= 4000: {np.max(np.abs(r)):.2e}")
print(f"te: max |diff| / sqrt(TT EE): {np.max(np.abs(ext['te'][30:4001] - cl['te'][30:4001] * T0**2) / np.sqrt(ext['tt'][30:4001] * ext['ee'][30:4001])):.2e}")
print(f"seconds: external {ext['seconds']:.1f}, analytic {dt:.1f}; 100 theta_s {ext['theta']:.6f}")
