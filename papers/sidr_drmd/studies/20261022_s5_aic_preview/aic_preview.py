"""Best-fit chi^2 of the four nested models on the S+5 combined networks (preview of the AIC table).

    python results/20261022_s5_aic_preview/aic_preview.py

For each data set (pp, ppshoes) the combined network 0 of the S+5 package is maximised inside its
region (model.network.restrict_model, float64) with L-BFGS-B from many starts, with parameters fixed
on the box faces that define each limit:
    full        nothing fixed                                   N = 12
    DRMD        logGeff = -5.5 (the box's lower edge: G_eff -> 0) N = 11
    SInu        delta_Neff = 0, f_idm = 0                       N = 8
    LCDM        delta_Neff = 0, f_idm = 0, logGeff = -5.5       N = 7
(N as in the preprint's AIC table: free cosmological parameters; m_int stays free in every model.)
With delta_Neff = f_idm = 0, log10 z_stop and log10 G/aH do not enter the true likelihood; they are
left free here and their effect on the network is reported as noise.
The network is Cobaya's log-posterior (likelihoods + normalised priors), so
chi^2 = -2 (network - log prior), with the log prior from cobaya.yaml (uniform widths, Gaussian
A_planck, calTE, calEE; for --series wc also Tcal, and the SH0ES M_B prior of wcmb is data: DATA_PRIORS). Starts: the best training points (true values) and the heaviest BEST samples,
each put onto the model's face. The best true training point near each face is listed as a check.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

import os
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import tensorflow as tf  # noqa: E402
import yaml  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import norm  # noqa: E402

from model.network import restrict_model  # noqa: E402
from sampling.targets import load_snapshot  # noqa: E402

KIT = REPO / "results" / "20261007_grendel_kit" / "hpc_client"
RUNS = {"pp": "20261022_081623_sidr_drmd2_pp_combined_boost5",
        "ppshoes": "20261022_081623_sidr_drmd2_ppshoes_combined_boost5"}


# the runs on the H0 World Cup data (24 October 2026): data sets wc and wcmb, runs results/*_sidr_drmd_<data>_combined_<label>
SERIES = {"drmd2": ("sidr_drmd2", ("pp", "ppshoes")), "wc": ("sidr_drmd", ("wc", "wcmb"))}
TITLES = {"pp": "Planck + PR4 lensing + DESI DR2 + Pantheon+", "ppshoes": "... + Pantheon+SH0ES",
          "wc": "H0 World Cup data: Planck cut + ACT + SPT + lensing + DESI DR2 + Pantheon+", "wcmb": "... + the SH0ES M_B prior"}


def setup():
    """--label boost5 (default: the runs above, outputs here) or boost10 etc. with --out DIR; --series wc for the runs on the
    H0 World Cup data (then --label and --out are needed: the default runs and outputs are the second set's S+5)."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="boost5")
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--series", choices=sorted(SERIES), default="drmd2")
    args, _ = parser.parse_known_args()
    if args.series != "drmd2" and args.out == HERE:
        raise SystemExit("--series wc needs --out (this folder holds the second set's S+5 results)")
    prefix, data_sets = SERIES[args.series]
    if args.series != "drmd2":
        RUNS.clear()
        RUNS.update({d: None for d in data_sets})
    if args.label != "boost5" or args.series != "drmd2":
        for data in RUNS:
            RUNS[data] = sorted(KIT.glob(f"results/*_{prefix}_{data}_combined_{args.label}"))[-1].name
    args.out.mkdir(parents=True, exist_ok=True)
    return args
MODELS = {"full": ({}, 12), "DRMD": ({"logGeff": -5.5}, 11),
          "SInu": ({"delta_Neff_drmd": 0.0, "f_idm_drmd": 0.0}, 8),
          "LCDM": ({"delta_Neff_drmd": 0.0, "f_idm_drmd": 0.0, "logGeff": -5.5}, 7)}
# "near the face" for the true-value check from the training points
NEAR = {"logGeff": lambda v: v < -5.0, "delta_Neff_drmd": lambda v: v < 0.01, "f_idm_drmd": lambda v: v < 0.01}


# Gaussian priors that are data, counted in chi^2 as ((x - loc) / scale)^2: the SH0ES M_B prior of the World Cup runs'
# wcmb (the second set's ppshoes had SH0ES in its Pantheon+ likelihood). Only their normalisation is subtracted.
DATA_PRIORS = {"Mb"}


def log_prior(cobaya_file, names):
    params = yaml.safe_load(Path(cobaya_file).read_text())["params"]
    terms = []
    for name in names:
        prior = params[name]["prior"]
        if "dist" in prior:
            assert prior["dist"] == "norm", prior
            terms.append(("data" if name in DATA_PRIORS else "norm", float(prior["loc"]), float(prior["scale"])))
        else:
            terms.append(("uniform", float(prior["min"]), float(prior["max"])))

    def f(x):
        x = np.atleast_2d(x)
        out = np.zeros(len(x))
        for j, (kind, a, b) in enumerate(terms):
            if kind == "norm":
                out += norm.logpdf(x[:, j], a, b)
            elif kind == "data":
                out += -0.5 * np.log(2 * np.pi * b * b)
            else:
                out += -np.log(b - a)
        return out
    return f


def main():
    args = setup()
    rng = np.random.default_rng(7)
    results = {}
    for data, name in RUNS.items():
        run = KIT / "results" / name
        target, identity = load_snapshot(run, 0, "run")
        names = list(target.metadata.param_names)
        lower, upper = np.array([target.metadata.bounds[n] for n in names], float).T
        model = restrict_model(target.model, target.region, "float64")
        region = target.region
        prior = log_prior(KIT / "runs" / name.split("_", 2)[2] / "cobaya.yaml", names)
        train = np.loadtxt(run / "training_data" / "data_it_0.csv", delimiter=",", skiprows=1)
        tx, ty = train[:, :-1], train[:, -1]
        best = np.vstack([np.loadtxt(f) for f in sorted(run.glob("evidence/evidence_it_0_nlive2048*.txt"))])   # every BEST run there
        bx = best[np.argsort(best[:, 0])[::-1][:3000], 2:]

        @tf.function(input_signature=[tf.TensorSpec([1, len(names)], tf.float64)])
        def value_and_gradient(t):
            with tf.GradientTape() as tape:
                tape.watch(t)
                y = model(t, training=False)[0, 0]
            return y, tape.gradient(y, t)

        def value(x):
            return model(tf.constant(np.atleast_2d(x), tf.float64), training=False).numpy().ravel()

        results[data] = {}
        for label, (fixed, n_free) in MODELS.items():
            fix_idx = {names.index(k): v for k, v in fixed.items()}
            free = [j for j in range(len(names)) if j not in fix_idx]
            # starts: best 150 training points and 150 heavy BEST samples, put on the face; plus the
            # best 50 training points near the face
            near = np.ones(len(tx), bool)
            for k in fixed:
                near &= NEAR[k](tx[:, names.index(k)])
            order = np.argsort(ty)[::-1]
            starts = np.vstack([tx[order[:150]], bx[rng.choice(len(bx), 150, replace=False)],
                                tx[near][np.argsort(ty[near])[::-1][:50]]])
            for j, v in fix_idx.items():
                starts[:, j] = v
            starts = np.clip(starts, lower, upper)
            vals = value(starts)
            starts = starts[np.argsort(vals)[::-1][:60]]          # climb from the 60 best

            def negative(z):
                x = np.empty(len(names))
                for j, v in fix_idx.items():
                    x[j] = v
                x[free] = z
                y, g = value_and_gradient(tf.constant(x[None], tf.float64))
                return -float(y.numpy()), -g.numpy()[0][free]

            found = []
            for s in starts:
                r = minimize(negative, s[free], jac=True, method="L-BFGS-B",
                             bounds=list(zip(lower[free], upper[free])), options=dict(maxiter=3000, gtol=1e-7))
                x = s.copy()
                x[free] = r.x
                found.append((-r.fun, x))
            found.sort(key=lambda t: -t[0])
            top, xbest = found[0]
            rho = float(region.rho(xbest[None])[0])
            spread = float(top - found[min(4, len(found) - 1)][0])      # best minus 5th best optimum
            entry = dict(n=n_free, net=top, chi2=float(-2 * (top - prior(xbest)[0])), rho=rho,
                         inside=rho <= region.inflate, x=dict(zip(names, xbest.tolist())), fifth_gap=spread)
            # noise along directions the true likelihood does not see at delta_Neff = f_idm = 0
            if "f_idm_drmd" in fixed:
                iz, ig = names.index("log10_zstop"), names.index("log10_G_over_aH")
                grid = np.repeat(xbest[None], 25, axis=0)
                zz, gg = np.meshgrid(np.linspace(lower[iz], upper[iz], 5), np.linspace(lower[ig], upper[ig], 5))
                grid[:, iz], grid[:, ig] = zz.ravel(), gg.ravel()
                gv = value(grid)
                entry["zstop_GaH_range"] = [float(gv.min() - top), float(gv.max() - top)]
            # true check from the training points near the face
            if near.any():
                k = np.argmax(np.where(near, ty, -np.inf))
                entry["true_near"] = dict(points=int(near.sum()), chi2=float(-2 * (ty[k] - prior(tx[k])[0])),
                                          net_there=float(value(tx[k])[0]), true_there=float(ty[k]),
                                          x={n_: float(v) for n_, v in zip(names, tx[k])})
            else:
                entry["true_near"] = dict(points=0)
            results[data][label] = entry
            t = entry["true_near"]
            print(f"{data:8s} {label:5s} N={n_free:2d}  chi2_net={entry['chi2']:.2f}  rho={rho:.2f}  "
                  f"gap to 5th optimum={spread:.2f}  "
                  + (f"zstop/GaH range={entry['zstop_GaH_range'][0]:+.2f}..{entry['zstop_GaH_range'][1]:+.2f}  " if "zstop_GaH_range" in entry else "")
                  + (f"best true near face: chi2={t['chi2']:.2f} (net-true there {t['net_there'] - t['true_there']:+.2f}, {t['points']} pts)" if t["points"] else "no training point near the face"),
                  flush=True)
    (args.out / "aic_preview.json").write_text(json.dumps(dict(results, runs=dict(RUNS)), indent=1))


if __name__ == "__main__":
    main()
