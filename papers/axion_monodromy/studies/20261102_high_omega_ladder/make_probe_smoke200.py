"""The probe's smoke test for the omega <= 200 inputs (2 November 2026), as 20261028_axion_runs/smoke_test/
make_probe_smoke.py: runs/axion_probe200_smoke in a scratch kit with the real run's theory block (CLASS++, external_Pk
with custom7, the converged sampling), its 10 cosmological parameters and the CMB stand-in likelihood about a fiducial
with an oscillation A_osc 0.02, omega 150, phi 1.0 at the reference cosmology.

    python make_probe_smoke200.py KIT SMOKE_KIT CLASSY_LIB [RUN]      (RUN: runs/axion_pp_omega200 by default)
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

kit, smoke, lib = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
src = sys.argv[4] if len(sys.argv) > 4 else "axion_pp_omega200"
sys.path.insert(0, lib)
from classy import Class  # noqa: E402

real = yaml.safe_load((kit / "runs" / src / "cobaya.yaml").read_text())
box = json.loads((kit / "runs" / src / "domain_bounds.json").read_text())
cosmo = ["omega_b", "omega_cdm", "theta_s_100", "logA", "n_s", "nrun", "tau_reio", "A_osc", "omega", "phi"]
values = ["100*theta_s"] + [f"custom{i}" for i in range(1, 7)]
for f in ("smoke/axion_probe.py", "domain.py", "loading.py", "refused.py", "boltzmann_build.py"):
    (smoke / f).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(kit / f, smoke / f)
(smoke / "runs" / src).mkdir(parents=True, exist_ok=True)
(smoke / "logs").mkdir(exist_ok=True)
shutil.copy(kit / "runs" / src / "generate_Pk.py", smoke / "runs" / src / "generate_Pk.py")
shutil.copy(Path(__file__).parent.parent / "20261028_axion_runs/smoke_test/cl_standin.py", smoke / "cl_standin.py")
run = smoke / "runs" / "axion_probe200_smoke"
run.mkdir(parents=True, exist_ok=True)

ref = {k: real["params"][k]["ref"] for k in cosmo}
extra = dict(real["theory"]["classy"]["extra_args"])
extra["command"] = f"{sys.executable} {smoke / 'runs' / src / 'generate_Pk.py'}"
c = Class()
c.set(dict(extra, output="tCl pCl lCl", lensing="yes", l_max_scalars=2500, omega_b=ref["omega_b"],
           omega_cdm=ref["omega_cdm"], tau_reio=ref["tau_reio"], **{"100*theta_s": ref["theta_s_100"]},
           custom1=1e-10 * np.exp(ref["logA"]), custom2=ref["n_s"], custom3=ref["nrun"],
           custom4=0.02, custom5=150.0, custom6=1.0))
c.compute()
cl = c.lensed_cl(2500)
np.savez(run / "fiducial.npz", tt=cl["tt"], te=cl["te"], ee=cl["ee"])

info = dict(likelihood={"cl_standin.ClStandin": {"python_path": str(smoke), "fiducial": str(run / "fiducial.npz")}},
            theory={"classy": real["theory"]["classy"]},
            params={k: real["params"][k] for k in cosmo + values})
(run / "cobaya.yaml").write_text(yaml.safe_dump(info, sort_keys=False, width=10000))
(run / "domain_bounds.json").write_text(json.dumps({k: box[k] for k in cosmo}, indent=1) + "\n")
(run / "client.yaml").write_text(yaml.safe_dump(dict(likelihood=dict(wrapper="cobaya", input="runs/axion_probe200_smoke/cobaya.yaml")),
                                                sort_keys=False))
print(run, "extra_args:", real["theory"]["classy"]["extra_args"])
