"""The nodes of finished profile studies as a points file for smoke/chi2_points_mp.py (24 October 2026):

    python true_profiles/node_points.py POINTS --set NAME --cobaya runs/<RUN>/cobaya.yaml --job-env runs/<RUN>/job.env \
        --run results/<time>_<RUN> STUDY [STUDY ...]

Each STUDY is results_profiles/<STUDY>.yaml (its `parameters`, whose order numbers the node files) and
results_profiles/<STUDY>/true/node_PPP_NNNNN.json (its nodes: x over the run's parameters, in the order of the run's
metadata.json, and the node's true log-objective logl). Every node becomes a point with label <parameter>:<NN> (with the
study's name in front when there are several studies), profile = the parameter, coordinate and logl; chi2_points_mp.py
evaluates every likelihood there and checks that the total reproduces logl. Adds the set to POINTS (made if missing).
"""
import argparse
import json
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("points", type=Path)
    p.add_argument("--set", required=True)
    p.add_argument("--cobaya", required=True)
    p.add_argument("--job-env", required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("studies", nargs="+")
    a = p.parse_args()

    params = json.loads((a.run / "metadata.json").read_text())["parameters"]
    names = [q["name"] if isinstance(q, dict) else q for q in params]
    points = []
    for study in a.studies:
        config = yaml.safe_load((HERE / "results_profiles" / f"{study}.yaml").read_text())
        profiled = list(config["parameters"])
        files = sorted((HERE / "results_profiles" / study / "true").glob("node_*.json"))
        if not files:
            raise SystemExit(f"no node checkpoints in results_profiles/{study}/true/")
        for f in files:
            i, k = map(int, re.findall(r"node_(\d+)_(\d+)", f.name)[0])
            node = json.loads(f.read_text())
            if len(node["x"]) != len(names):
                raise SystemExit(f"{f}: {len(node['x'])} coordinates, the run has {len(names)} parameters")
            label = f"{profiled[i]}:{k:02d}" if len(a.studies) == 1 else f"{study}/{profiled[i]}:{k:02d}"
            points.append(dict(label=label, profile=profiled[i], coordinate=float(node["coordinate"]),
                               params=dict(zip(names, map(float, node["x"]))), logl=float(node["logl"]), study=study))
        print(f"{study}: {len(files)} nodes of {', '.join(profiled)}")
    spec = json.loads(a.points.read_text()) if a.points.exists() else dict(made="true_profiles/node_points.py", sets={})
    spec["sets"][a.set] = dict(cobaya=a.cobaya, job_env=a.job_env, run=str(a.run), studies=a.studies, points=points)
    a.points.write_text(json.dumps(spec, indent=1))
    print(f"written: {a.points} (set {a.set}, {len(points)} points)")


if __name__ == "__main__":
    main()
