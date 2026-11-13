"""(The omega <= 200 copy of 20261028_axion_runs/smoke_test/make_standin.py.) An analytic stand-in of a run of the kit, for a smoke test of its set-up through launch.py: the run's
cobaya.yaml with its theory, likelihoods, derived and fixed parameters replaced by one external likelihood of
the same sampled parameters (same priors and refs), and its client.yaml cut to smoke size with every key of
the real one kept.

    python make_standin.py KIT_RUN_DIR SMOKE_KIT RUN_NAME

The stand-in: a Gaussian about the refs (widths a twelfth of the box, Planck-like for the cosmological
parameters) plus an oscillation gain 300 A cos(phi - 0.5 - 0.3 (omega - 150)) exp(-(omega - 150)^2 / 8) - 5000 A^2,
a peak of 4.5 at omega 150, A 0.03 (make_standin200.py: the omega 32 of make_standin.py moved into [55, 200]).
"""
import json
import shutil
import sys
from pathlib import Path

import yaml

src, kit, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
run = kit / "runs" / name
run.mkdir(parents=True, exist_ok=True)
info = yaml.safe_load((src / "cobaya.yaml").read_text())
box = json.loads((src / "domain_bounds.json").read_text())
sampled = {k: v for k, v in info["params"].items() if isinstance(v, dict) and "prior" in v}
assert set(sampled) == set(box), (sorted(set(sampled) ^ set(box)))
sigma = dict(omega_b=0.00014, omega_cdm=0.0009, theta_s_100=0.00029, logA=0.014, n_s=0.0040, nrun=0.0067,
             tau_reio=0.0071)
centre = {}
for k, v in sampled.items():
    ref = v.get("ref")
    centre[k] = float(ref if isinstance(ref, (int, float)) else ref.get("loc", 0.5 * sum(box[k])))
args = ", ".join(sampled)
gauss = " + ".join(f"((({k}) - {centre[k]!r}) / {sigma.get(k, (box[k][1] - box[k][0]) / 12)!r})**2"
                   for k in sampled if k not in ("A_osc", "omega", "phi"))
gain = ("300*A_osc*np.cos(phi - 0.5 - 0.3*(omega - 150))*np.exp(-(omega - 150)**2/8) - 5000*A_osc**2")
params = {}
for k, v in sampled.items():
    params[k] = {key: v[key] for key in ("prior", "ref", "proposal", "latex") if key in v}
standin = dict(likelihood={"standin": {"external": f"lambda {args}: -0.5*({gauss}) + {gain}"}}, params=params)
(run / "cobaya.yaml").write_text("# analytic stand-in of the axion PDP+ runs (make_standin.py)\n"
                                 + yaml.safe_dump(standin, sort_keys=False, width=10000))
shutil.copy(src / "domain_bounds.json", run / "domain_bounds.json")
cfg_text = (src / "client.yaml").read_text().replace(f"runs/{src.name}/cobaya.yaml", f"runs/{name}/cobaya.yaml")
cfg = yaml.safe_load(cfg_text)
cfg["likelihood"]["input"] = f"runs/{name}/cobaya.yaml"
for p in cfg["likelihood"].get("identity_files", []):          # the real path: both parts name axion_pp's script
    (kit / p).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src.parent.parent / p, kit / p)
cfg["prior"].update(n_samples=300, n_inject=20)
cfg["acquisition"].update(n_append=200, n_inject=20)
cfg["acquisition"]["audit"].update(posterior=20)
cfg["acquisition"]["boost"].update(n=300, boosts=2, start_iteration=2)
cfg["model"].update(n_layers=2, n_neurons=32)
cfg["training"].update(n_epochs=40, patience=10, lr_patience=5, lr_grace=5)
cfg["sampling"].update(n_walkers=64, n_steps=1500, burn_in=300, thin=5, chunk_size=500)
cfg["convergence"].update(max_iterations=5)
(run / "client.yaml").write_text(f"# smoke-size copy of runs/{src.name}/client.yaml (make_standin.py)\n"
                                 + yaml.safe_dump(cfg, sort_keys=False))
print(f"{run}: {len(sampled)} sampled parameters")
