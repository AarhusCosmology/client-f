"""Network profile of chi^2 along log10 Geff on the combined networks, with and without DRMD.

    python analysis/aic/profile_logGeff.py --series wc --label boost10 --out logs/aic_wc_boost10

The kit's copy (24 October 2026) of results/20261022_s5_aic_preview/profile_logGeff.py (see aic_preview.py here).

At each of 20 values of log10 Geff in [-5.5, -0.8] the network is maximised (as aic_preview.py) over
everything else, either on the SInu face (delta_Neff = f_idm = 0) or in the full model. Starts: the
best training points within 0.25 of the node (on the face for SInu) and the best overall, put on the
node. The best true training value per bin is overlaid. Writes profile_logGeff.{json,png}.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aic_preview import KIT, RUNS, REPO, TITLES, log_prior, setup  # noqa: E402

import tensorflow as tf  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from model.network import restrict_model  # noqa: E402
from sampling.targets import load_snapshot  # noqa: E402

NODES = np.linspace(-5.5, -0.8, 20)
FACES = {"SInu face (no DRMD)": {"delta_Neff_drmd": 0.0, "f_idm_drmd": 0.0}, "full model": {}}


def main():
    args = setup()
    out = {}
    for data, name in RUNS.items():
        run = KIT / "results" / name
        target, _ = load_snapshot(run, 0, "run")
        names = list(target.metadata.param_names)
        lower, upper = np.array([target.metadata.bounds[n] for n in names], float).T
        model = restrict_model(target.model, target.region, "float64")
        prior = log_prior(KIT / "runs" / name.split("_", 2)[2] / "cobaya.yaml", names)
        t = np.loadtxt(run / "training_data" / "data_it_0.csv", delimiter=",", skiprows=1)
        tx, ty = t[:, :-1], t[:, -1]
        iG = names.index("logGeff")

        @tf.function(input_signature=[tf.TensorSpec([1, len(names)], tf.float64)])
        def value_and_gradient(v):
            with tf.GradientTape() as tape:
                tape.watch(v)
                y = model(v, training=False)[0, 0]
            return y, tape.gradient(y, v)

        def value(x):
            return model(tf.constant(np.atleast_2d(x), tf.float64), training=False).numpy().ravel()

        out[data] = {}
        for face, fixed in FACES.items():
            on_face = np.ones(len(tx), bool)
            for k in fixed:
                on_face &= tx[:, names.index(k)] < 0.01
            rows = []
            for g in NODES:
                fix = {names.index(k): v for k, v in fixed.items()}
                fix[iG] = g
                free = [j for j in range(len(names)) if j not in fix]
                near = on_face & (np.abs(tx[:, iG] - g) < 0.25)
                pick = np.concatenate([np.flatnonzero(near)[np.argsort(ty[near])[::-1][:8]],
                                       np.flatnonzero(on_face)[np.argsort(ty[on_face])[::-1][:8]]])
                starts = tx[pick].copy()
                for j, v in fix.items():
                    starts[:, j] = v
                starts = starts[np.argsort(value(starts))[::-1][:6]]

                def negative(z):
                    x = np.empty(len(names))
                    for j, v in fix.items():
                        x[j] = v
                    x[free] = z
                    y, gr = value_and_gradient(tf.constant(x[None], tf.float64))
                    return -float(y.numpy()), -gr.numpy()[0][free]

                best = None
                for s in starts:
                    r = minimize(negative, s[free], jac=True, method="L-BFGS-B",
                                 bounds=list(zip(lower[free], upper[free])), options=dict(maxiter=3000, gtol=1e-7))
                    if best is None or -r.fun > best[0]:
                        x = s.copy()
                        x[free] = r.x
                        best = (-r.fun, x)
                chi2 = float(-2 * (best[0] - prior(best[1])[0]))
                bin_ = on_face & (np.abs(tx[:, iG] - g) < 0.125)
                true = float(np.min(-2 * (ty[bin_] - prior(tx[bin_])))) if bin_.any() else None
                rows.append(dict(logGeff=float(g), chi2=chi2, true_bin=true, true_points=int(bin_.sum()),
                                 x=dict(zip(names, best[1].tolist()))))
                print(f"{data:8s} {face:20s} logGeff={g:+.2f} chi2={chi2:.2f} best true in bin="
                      f"{'-' if true is None else f'{true:.2f}'} ({int(bin_.sum())})", flush=True)
            out[data][face] = rows
    (args.out / "profile_logGeff.json").write_text(json.dumps(dict(out, runs=dict(RUNS)), indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    for ax, (data, label) in zip(axes, ((d, TITLES[d]) for d in RUNS)):
        ref = min(r["chi2"] for rows in out[data].values() for r in rows)
        for (face, rows), colour in zip(out[data].items(), ("#c0392b", "#1f5fa8")):
            g = [r["logGeff"] for r in rows]
            ax.plot(g, [r["chi2"] - ref for r in rows], "-o", ms=3, color=colour, label=f"network, {face}")
            tg = [r["logGeff"] for r in rows if r["true_bin"] is not None]
            ax.plot(tg, [r["true_bin"] - ref for r in rows if r["true_bin"] is not None], "x", color=colour,
                    alpha=0.7, label=f"best true training point in bin, {face}")
        ax.set_xlabel(r"$\log_{10}(G_\mathrm{eff}\,\mathrm{MeV}^2)$")
        ax.set_ylabel(r"$\chi^2 - \chi^2_\mathrm{min}$ (network, full model)")
        ax.set_title(label, fontsize=10)
        ax.set_ylim(-1, 30)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{args.label} combined networks: profile in log10 Geff (everything else maximised)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out / "profile_logGeff.png", dpi=130)
    print("written:", args.out / "profile_logGeff.png")


if __name__ == "__main__":
    main()
