"""The points for the per-likelihood chi^2 job (smoke/chi2_points_mp.py), from the network fits of aic_preview.py and
profile_logGeff.py.

    python analysis/aic/make_points.py --series wc --label boost10 --out logs/aic_wc_boost10

The kit's copy (24 October 2026) of results/20261022_s5_aic_preview/make_points.py (see aic_preview.py here).

Per data set: the network best fits of aic_preview.json (full, DRMD, SInu = the strong mode, LCDM) and the
SInu-face profile node at log10 Geff ~ -4 of profile_logGeff.json (the weak plateau away from the box edge). With
--series wc the SInu fit is SInu_best (intermediate coupling there), and the point strong is added (below).
Each point carries the network's own value (Cobaya's log-posterior) so the job can print truth - network.
Writes the kit's smoke/chi2_points_s5.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aic_preview import KIT, RUNS, setup  # noqa: E402

import tensorflow as tf  # noqa: E402

from model.network import restrict_model  # noqa: E402
from sampling.targets import load_snapshot  # noqa: E402


def main():
    args = setup()
    fits = json.loads((args.out / "aic_preview.json").read_text())
    profile = json.loads((args.out / "profile_logGeff.json").read_text())
    out = dict(made="22 October 2026, results/20261022_s5_aic_preview/make_points.py", data={})
    for data, name in RUNS.items():
        run = KIT / "results" / name
        target, identity = load_snapshot(run, 0, "run")
        names = list(target.metadata.param_names)
        model = restrict_model(target.model, target.region, "float64")
        rows = profile[data]["SInu face (no DRMD)"]
        weak = min(rows, key=lambda r: abs(r["logGeff"] + 4.0))
        chosen = [("full", fits[data]["full"]["x"]), ("DRMD", fits[data]["DRMD"]["x"]),
                  ("SInu_strong", fits[data]["SInu"]["x"]), (f"SInu_weak", weak["x"]), ("LCDM", fits[data]["LCDM"]["x"])]
        if args.series != "drmd2":
            # on the World Cup data the SInu face's best fit is at intermediate coupling (log10 Geff ~ -3), not in the
            # strong mode; the strong mode's point is the full model's best profile node at log10 Geff > -2
            chosen[2] = ("SInu_best", chosen[2][1])
            strong = min((r for r in profile[data]["full model"] if r["logGeff"] > -2.0), key=lambda r: r["chi2"])
            chosen.append(("strong", strong["x"]))
        points = []
        for label, x in chosen:
            v = np.array([x[n] for n in names])
            net = float(model(tf.constant(v[None], tf.float64), training=False).numpy().ravel()[0])
            points.append(dict(label=label, params=dict(zip(names, v.tolist())), network_logpost=net))
        out["data"][data] = dict(network=name, model_sha256=identity.get("model_sha256"),
                                 cobaya=f"runs/{name.split('_', 2)[2]}/cobaya.yaml", points=points)
        print(data, [(p["label"], round(p["network_logpost"], 2), round(p["params"]["logGeff"], 2)) for p in points])
    if args.series != "drmd2":
        # the runs on the H0 World Cup data: the format of smoke/chi2_points_mp.py (each set names its Cobaya file and
        # its job.env, the CLASS++ build of the runs), one set per data set: <data>_bestfits
        sets = {f"{data}_bestfits": dict(cobaya=spec["cobaya"], job_env=spec["cobaya"].replace("cobaya.yaml", "job.env"),
                                         network=spec["network"], model_sha256=spec["model_sha256"],
                                         points=[dict(q, logl=None) for q in spec["points"]])
                for data, spec in out["data"].items()}
        path = KIT / "smoke" / f"chi2_points_{args.series}_{args.label}_bestfits.json"
        path.write_text(json.dumps(dict(made=out["made"], sets=sets), indent=1))
        print("written:", path, "(sets:", ", ".join(sets), ")")
        return
    path = KIT / "smoke" / ("chi2_points_s5.json" if args.label == "boost5" else f"chi2_points_{args.label}.json")
    path.write_text(json.dumps(out, indent=1))
    print("written:", path)


if __name__ == "__main__":
    main()
