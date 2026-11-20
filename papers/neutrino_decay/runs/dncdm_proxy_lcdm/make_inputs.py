"""The LambdaCDM pair beside the DNCDM proxy profiles (9 November 2026, Thomas): how much of the improvement from the
Gamma -> 0 plateau to the minimum at log10Gamma ~ 6 is the background losing nu3's mass?

    python -I runs/dncdm_proxy_lcdm/make_inputs.py [KIT_DIR]        # laptop, seconds; KIT_DIR defaults to .

For each data set (pp = PDP+, wc = H0WC) two Cobaya inputs from the proxy run's own cobaya.yaml, which differ from it
only in the decaying species and from each other only in one number, nu3's mass:
  m005  nu3 = ncdm_standard, m 0.05 eV: LambdaCDM with masses 0.05 + 0.0087 (+ 0 for nu1), the proxy's Gamma -> 0;
  m0    nu3 = ncdm_standard, m 0 (CLASS++ makes it 1e-5 eV: relativistic until today): nu3 has lost its mass.
nu3 keeps the proxy parent's temperature, degeneracy and momentum grid (quadrature_strategy 3, momenta_bins 16); nu2 and
everything else (N_ur 0.0044 + 1.0132 for the massless nu1, precision, likelihoods, priors) are the run's. Measured at
the PDP+ Gamma -> 0 optimum with CLASS++ cb2255f4 on the laptop (fixed parameters, Gaussian Planck-like / ACT-like
noise): massless nu3 as NCDM or as UR differ by Delta chi2 0.002 / 0.011, the two momentum grids by 0.002 / 0.007,
the proxy at log10Gamma -2 and m005 by 0.010 / 0.043; removing the mass is 26 / 115. The proxy at log10Gamma 6 is
m0 up to 0.13 / 0.85, mostly C_l^phiphi (-0.4%: about a third of the mass's lensing suppression remains, from the decay
products, also at log10Gamma 9), which is the decay's own physics, not a set-up difference.

Also written: covmat_<ds>.txt (the CLiENT run's network-14 posterior covariance without log10Gamma: Cobaya's minimizer
scales by it) and starts_<ds>_<variant>.json (start 0 = the true profile optimum: m005 at the log10Gamma = -2 node,
m0 at the best node in log10Gamma [5, 7.5]; starts 1-5 = start 0 plus a 1-sigma Gaussian draw of that covariance,
seed 20261109, inside the box). minimize_lcdm.py runs them.
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

KIT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
HERE = KIT / "runs" / "dncdm_proxy_lcdm"
RUNS = {"pp": "results/20261106_121906_dncdm_proxy_pp", "wc": "results/20261106_122443_dncdm_proxy_wc"}
STARTS = 6
NU3 = {"nu3.type": "ncdm_standard", "nu3.T": 0.71611, "nu3.deg": 1, "nu3.quadrature_strategy": 3, "nu3.momenta_bins": 16}
MASS = {"m005": 0.05, "m0": 0}

for ds, run in RUNS.items():
    source = KIT / "runs" / f"dncdm_proxy_{ds}" / "cobaya.yaml"
    info = yaml.safe_load(source.read_text())
    extra = info["theory"]["classy"]["extra_args"]
    if extra.get("N_ur") != 0.0044 or "dncdm1.type" not in extra:
        raise SystemExit(f"FATAL: {source} is not the proxy set-up this script knows")
    params = info["params"]
    names = [n for n, p in params.items() if isinstance(p, dict) and "prior" in p and n != "log10Gamma"]

    # the run's posterior (network 14), without log10Gamma
    chain = np.load(KIT / run / "analysis" / "posterior_it_14.npz", allow_pickle=True)
    cnames = [str(n) for n in chain["names"]]
    X = np.asarray(chain["x"], float)[:, [cnames.index(n) for n in names]]
    cov = np.cov(X, rowvar=False)
    (HERE / f"covmat_{ds}.txt").write_text("# " + " ".join(names) + "\n"
                                           + "\n".join(" ".join(f"{v:.8e}" for v in row) for row in cov) + "\n")
    # the box the profiles were run in (a Gaussian prior has no min/max in the yaml)
    study = KIT / "results_profiles" / f"dncdm_proxy_{ds}_gamma"
    bounds = json.loads((study / "manifest.json").read_text())["bounds"]
    box = np.array([bounds[n] for n in names], float)

    # the true profile optima the starts begin at
    prof = json.loads((study / "true_profiles.json").read_text())
    pn = prof["full_parameter_names"]
    grid = np.array(prof["profiles"]["log10Gamma"]["grid"])
    logl = np.array(prof["profiles"]["log10Gamma"]["logl"], float)
    xopt = np.array(prof["profiles"]["log10Gamma"]["xopt"], float)
    window = np.where((grid >= 5) & (grid <= 7.5))[0]
    node = {"m005": int(np.argmin(np.abs(grid + 2))), "m0": int(window[np.argmax(logl[window])])}

    for variant, m in MASS.items():
        out = dict(info)
        out["theory"] = {"classy": dict(info["theory"]["classy"])}
        ex = {k: v for k, v in extra.items() if not k.startswith("dncdm1.")}
        ex["N_ur"] = 1.0176                     # 0.0044 + 1.0132: the massless nu1, as in the proxy (there inside dncdm1)
        ex.update(NU3)
        ex["nu3.m"] = m
        out["theory"]["classy"]["extra_args"] = ex
        out["params"] = {k: v for k, v in params.items() if k not in ("log10Gamma", "dncdm1.log10Gamma")}
        out.pop("sampler", None)
        out.pop("output", None)
        header = (f"# LambdaCDM beside the DNCDM proxy, {'PDP+' if ds == 'pp' else 'H0WC'}: nu3 = ncdm_standard m {m} eV "
                  f"({'the proxy at Gamma -> 0' if m else 'nu3 massless: CLASS++ uses 1e-5 eV'}).\n"
                  f"# Made by runs/dncdm_proxy_lcdm/make_inputs.py from runs/dncdm_proxy_{ds}/cobaya.yaml (9 November 2026),"
                  f" which says what every other line is; only\n# dncdm1.* (now nu3.*), N_ur (+1.0132, the massless nu1)"
                  f" and log10Gamma differ. minimize_lcdm.py adds the sampler and the output.\n")
        (HERE / f"cobaya_{ds}_{variant}.yaml").write_text(header + yaml.safe_dump(out, sort_keys=False, width=120))

        x0 = xopt[node[variant]][[pn.index(n) for n in names]]
        rng = np.random.default_rng(20261109 + (0 if ds == "pp" else 1) * 10 + (0 if variant == "m005" else 1))
        draws = rng.multivariate_normal(np.zeros(len(names)), cov, size=STARTS - 1)
        starts = [x0] + [np.clip(x0 + d, box[:, 0] + 1e-9 * (box[:, 1] - box[:, 0]),
                                 box[:, 1] - 1e-9 * (box[:, 1] - box[:, 0])) for d in draws]
        (HERE / f"starts_{ds}_{variant}.json").write_text(json.dumps(dict(
            names=names, profile_node=dict(log10Gamma=float(grid[node[variant]]), logl=float(logl[node[variant]])),
            starts=[[float(v) for v in s] for s in starts]), indent=1))
        print(f"{ds} {variant}: {len(names)} parameters; start 0 = the profile node log10Gamma {grid[node[variant]]:+.2f} "
              f"(true logpost {logl[node[variant]]:.3f})")
