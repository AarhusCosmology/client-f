"""The bands of the combined emulators and the modes of the boosts, for Appendices A.2 and A.6 (9 November 2026).

    python analysis/bands200.py          # from paper/; about ten minutes, writes plots/data/bands200.json

For each band between two neighbouring parts of a combined emulator (combine.json: the parts at their final networks and
the bands), at the true training points of the two parts inside the overlap of their training boxes in omega, with a
value within DEPTH of the best training point of the two in ln P (refused points and values that are not finite left out):
  - each part's network, restricted to its own region as in the combined emulator (model.network.restrict_model), at the
    points of the OTHER part, on which it was not trained: the mean and the standard deviation of network - truth, and the
    share of the points outside its region (where it gives the wall);
  - the combined network at all these points inside the band itself;
  - for the lowest band the same split by A_osc (the omega <= 60 study found the upper part poor at moderate amplitudes
    just above the band).
For each part, the number of modes in each boost (boost/boost_it_N.json, the boosts S to S + 9 of its own hand-over S) and
the number of cells of its final region.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os  # noqa: E402
HERE = Path(__file__).resolve().parent.parent
KIT = Path(os.environ.get("CLIENT_PAPER_KIT", HERE.parent))           # the cluster folder (README.md)
sys.path.insert(0, os.environ.get("CLIENT_REPO") or str(HERE.parents[2]))   # CLiENT: this branch of client-f
import tensorflow as tf  # noqa: E402

tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)
from model.network import load_model, restrict_model  # noqa: E402
from sampling.region import Region  # noqa: E402

RES = KIT / "results"
OUT = Path(__file__).resolve().parent.parent / "plots" / "data" / "bands200.json"
COMBS = {"pp": "20261107_084147_axion_pp_combined200", "wc": "20261107_084828_axion_wc_combined200"}
DEPTH = 30.0
A_BINS = [(0.0, 0.005), (0.005, 0.015), (0.015, 0.1)]


def evaluate(model, x):
    return np.concatenate([model(tf.constant(x[i:i + 8192], tf.float64), training=False).numpy()[:, 0]
                           for i in range(0, len(x), 8192)]) if len(x) else np.zeros(0)


def stats(d):
    d = np.asarray(d, float)
    return dict(n=int(len(d)), mean=float(d.mean()) if len(d) else None, std=float(d.std()) if len(d) else None,
                rms=float(np.sqrt(np.mean(d ** 2))) if len(d) else None)


out = {}
for data, comb in COMBS.items():
    cdir = RES / comb
    combine = json.loads((cdir / "combine.json").read_text())
    names = [p["name"] for p in json.loads((cdir / "metadata.json").read_text())["parameters"]]
    parts = []
    for q in combine["parts"]:
        run = RES / Path(q["run"]).name
        it = int(re.search(r"region_it_(\d+)\.npz", next(a for a in q["artifacts"] if "region_it_" in a)).group(1))
        box = json.loads((KIT / "runs" / re.sub(r"^\d{8}_\d{6}_", "", run.name) / "domain_bounds.json").read_text())["omega"]
        df = pd.read_csv(run / f"training_data/data_it_{it}.csv")
        assert [c for c in df.columns if c != "loglkl"] == names, run
        refused = json.loads((run / "refused_value.json").read_text()).get("value") if (run / "refused_value.json").exists() else None
        ok = np.isfinite(df.loglkl) & (df.loglkl != refused if refused is not None else True)
        region = Region.load(run / f"boost/region_it_{it}.npz")
        start = json.loads((run / "boost/start.json").read_text())["start_iteration"]
        modes = []
        for n in range(start, it):
            f = run / f"boost/boost_it_{n}.json"
            if f.exists():
                modes.append(len(json.loads(f.read_text())["modes"]))
        parts.append(dict(name=run.name, it=it, box=box, x=df.loc[ok, names].to_numpy(float), y=df.loc[ok, "loglkl"].to_numpy(float),
                          region=region, modes=modes, cells=int(len(region.means))))
        print(data, run.name, "network", it, "box", box, "points", int(ok.sum()), "modes per boost", modes, "cells", len(region.means))
    best = max(p["y"].max() for p in parts)
    comb_model = restrict_model(load_model(cdir / "trained_models/model_it_0.keras"), Region.load(cdir / "boost/region_it_0.npz"))
    k_om, k_a = names.index("omega"), names.index("A_osc")
    nets = {}
    bands = []
    for b, (lo, hi) in enumerate(combine["bands"]):
        L, U = parts[b], parts[b + 1]
        olo, ohi = U["box"][0], L["box"][1]
        entry = dict(band=[lo, hi], overlap=[olo, ohi], lower=L["name"], upper=U["name"])
        for who, other, net_part in (("lower", U, L), ("upper", L, U)):
            if net_part["name"] not in nets:
                nets[net_part["name"]] = restrict_model(load_model(RES / net_part["name"] / f"trained_models/model_it_{net_part['it']}.keras"),
                                                        net_part["region"])
            sel = (other["x"][:, k_om] >= olo) & (other["x"][:, k_om] <= ohi) & (best - other["y"] < DEPTH)
            x, y = other["x"][sel], other["y"][sel]
            d = evaluate(nets[net_part["name"]], x) - y
            walled = d < -500                       # the region's wall: 1000 below the best point and lower
            e = dict(all=stats(d[~walled]), walled=int(walled.sum()), points=int(len(d)))
            e["by_A"] = [dict(A=list(a), **stats(d[~walled & (x[:, k_a] >= a[0]) & (x[:, k_a] < a[1])]),
                              walled=int((walled & (x[:, k_a] >= a[0]) & (x[:, k_a] < a[1])).sum())) for a in A_BINS]
            entry[f"{who}_net_at_{'upper' if who == 'lower' else 'lower'}_points"] = e
        xs = np.concatenate([L["x"], U["x"]])
        ys = np.concatenate([L["y"], U["y"]])
        sel = (xs[:, k_om] >= lo) & (xs[:, k_om] <= hi) & (best - ys < DEPTH)
        dc = evaluate(comb_model, xs[sel]) - ys[sel]
        entry["combined_in_band"] = dict(all=stats(dc[dc > -500]), walled=int((dc < -500).sum()))
        bands.append(entry)
        a, c = entry["lower_net_at_upper_points"]["all"], entry["upper_net_at_lower_points"]["all"]
        print(f"  band {lo}-{hi} (overlap {olo}-{ohi}): lower net at upper points n={a['n']} mean {a['mean']} std {a['std']}; "
              f"upper net at lower points n={c['n']} mean {c['mean']} std {c['std']}; combined in band {entry['combined_in_band']}")
    out[data] = dict(depth=DEPTH, bands=bands, parts=[dict(name=p["name"], network=p["it"], box=p["box"], modes=p["modes"],
                                                           cells=p["cells"]) for p in parts])
OUT.write_text(json.dumps(out, indent=1, default=float) + "\n")
print("written", OUT)
