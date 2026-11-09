"""The posterior-mean points for the per-likelihood chi^2 job (smoke/chi2_points_mp.py).

    python analysis/aic/posterior_mean_points.py --series wc --label boost10

The kit's copy (24 October 2026) of results/20261023_posterior_mean/make_points.py, with --label (the network
package; boost10 is the paper's) and the paths of aic_preview.py here. Writes
smoke/chi2_points_<series>_<label>_posterior_mean.json (wc), as the laptop's copy smoke/chi2_points_posterior_mean.json
(drmd2).

The paper's posteriors are the S+10 combined emulators sampled with BEST, four seeds pooled (kit
results/*_sidr_drmd2_<data>_combined_boost10/analysis/best_posterior_it_0.npz, weighted samples). Their mean over all
21 sampled parameters is the point "posterior_mean"; the job evaluates the true likelihood there, so that the paper can
quote its Delta chi^2 against the global best fit. Each point carries the emulator's value (inside its region), as the
job prints truth minus network. Writes the kit's smoke/chi2_points_posterior_mean.json.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[2]
REPO = Path(os.environ.get("CLIENT_REPO") or KIT.parents[1])      # CLiENT: this branch of client-f
sys.path.insert(0, str(REPO))

import tensorflow as tf  # noqa: E402

from model.network import restrict_model  # noqa: E402
from sampling.targets import load_snapshot  # noqa: E402


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", choices=("drmd2", "wc"), default="drmd2",
                        help="wc: the runs on the H0 World Cup data (24 October 2026), written in the format of the kit's "
                             "smoke/chi2_points_mp.py to smoke/chi2_points_wc_posterior_mean.json")
    parser.add_argument("--label", default="boost10")
    args = parser.parse_args()
    prefix, data_sets = {"drmd2": ("sidr_drmd2", ("pp", "ppshoes")), "wc": ("sidr_drmd", ("wc", "wcmb"))}[args.series]
    out = dict(made="23 October 2026, results/20261023_posterior_mean/make_points.py", data={})
    for d in data_sets:
        run = sorted(KIT.glob(f"results/*_{prefix}_{d}_combined_{args.label}"))[-1]
        post = np.load(run / "analysis" / "best_posterior_it_0.npz", allow_pickle=True)
        names = [str(n) for n in post["names"]]
        w = post["weights"] / post["weights"].sum()
        mean = (post["x"] * w[:, None]).sum(axis=0)
        target, identity = load_snapshot(run, 0, "run")
        if list(target.metadata.param_names) != names:
            raise SystemExit(f"{run.name}: the posterior's parameters are not the network's")
        model = restrict_model(target.model, target.region, "float64")
        net = float(model(tf.constant(mean[None], tf.float64), training=False).numpy().ravel()[0])
        out["data"][d] = dict(network=run.name, model_sha256=identity.get("model_sha256"),
                              cobaya=f"runs/{prefix}_{d}_combined_{args.label}/cobaya.yaml",
                              points=[dict(label="posterior_mean", params=dict(zip(names, mean.tolist())), network_logpost=net)])
        print(d, run.name, f"emulator at the posterior mean {net:.2f}")
    if args.series != "drmd2":
        sets = {f"{d}_posterior_mean": dict(cobaya=spec["cobaya"], job_env=spec["cobaya"].replace("cobaya.yaml", "job.env"),
                                            network=spec["network"], points=spec["points"])
                for d, spec in out["data"].items()}
        path = KIT / "smoke" / f"chi2_points_{args.series}_{args.label}_posterior_mean.json"
        path.write_text(json.dumps(dict(made=out["made"], sets=sets), indent=1))
        print("written:", path)
        return
    path = KIT / "smoke" / "chi2_points_posterior_mean.json"
    path.write_text(json.dumps(out, indent=1))
    print("written:", path)


if __name__ == "__main__":
    main()
