"""Upper bounds on the log10 Geff profile on the plateau of weak interaction (25 October 2026).

    python plateau_points.py wcmb [wc]

On the World Cup data the final emulators have false maxima at weak coupling (log10 Geff < -3.4), outside the region
their boosts trained, and the profile nodes there ended far above the plateau (27-45 in chi^2 for wcmb, where the DRMD
best fit at log10 Geff = -5.5 is about 17 above the best fit). Below about -4 the interaction does not act on observable
scales, so the true chi^2 of the DRMD best fit moved to another log10 Geff bounds the profile there from above. This
writes those points: the DRMD point of the newest AIC job (logs/aic_wc_<label>/chi2_points_wc_<label>_bestfits.json)
with log10 Geff set to each node of the _logGeff_wide study below -3.4, into the kit's smoke/chi2_points_wc_plateau.json
(set <data>_plateau; profile logGeff, coordinate the node's value), and set <data>_plateau_neighbour (below). On Grendel:

    sbatch smoke/submit_chi2_points.sh smoke/chi2_points_wc_plateau.json

build_wc.py takes the smaller of a node's value and its bound (logs/chi2_points_<data>_plateau_<job>.json).
"""
import csv
import json
import sys

import build_wc as W

KIT = W.KIT
EDGE = -3.4          # nodes below this are on the plateau (the converged nodes above it agree with the emulator)
NEIGHBOUR = 0.6      # how far below the innermost converged node its point is moved


def main():
    data_sets = sys.argv[1:] or ["wcmb"]
    path = KIT / "smoke" / "chi2_points_wc_plateau.json"
    spec = json.loads(path.read_text()) if path.exists() else dict(made="plateau_points.py", sets={})
    for d in data_sets:
        study = f"sidr_drmd_{d}_combined_boost10_logGeff_wide"
        source = KIT / "logs" / f"profiles_{study}.csv"
        if not source.exists():
            raise SystemExit(f"no {source} (snapshot or breakdown job first)")
        nodes = sorted({round(float(r["coordinate"]), 6) for r in csv.DictReader(open(source))})
        fits = W.chi2_points(d, "bestfits")
        drmd = fits["results"]["DRMD"]["params"]
        points = [dict(label=f"logGeff:{j:02d}", profile="logGeff", coordinate=g, params=dict(drmd, logGeff=g),
                       note=f"DRMD best fit of the {fits['label']} AIC job moved to log10 Geff = {g}")
                  for j, g in enumerate(nodes) if g < EDGE]
        spec["sets"][f"{d}_plateau"] = dict(cobaya=f"runs/sidr_drmd_{d}_combined_boost10/cobaya.yaml",
                                            job_env=f"runs/sidr_drmd_{d}_combined_boost10/job.env",
                                            network=fits["network"], points=points)
        print(f"{d}: {len(points)} points, log10 Geff {points[0]['coordinate']:.2f} to {points[-1]['coordinate']:.2f}, "
              f"from the DRMD fit of {fits['network']}")
        # Between the plateau and the minimum the interaction acts and the moved DRMD fit is a poor bound (wcmb, -3.54:
        # 26.6, where the converged node at -3.35 is 6.6): the innermost converged node above EDGE, moved down to the
        # nodes within NEIGHBOUR of it, bounds those as well. Set <data>_plateau_neighbour (build_wc.py reads every
        # logs/chi2_points_<data>_plateau*_<job>.json and keeps the best value per node).
        rows = list(csv.DictReader(open(source)))
        log = next((f for f in sorted(KIT.glob("logs/profiles_*.out")) if f"results_profiles/{study}.yaml"
                    in f.read_text(errors="replace")), None)
        status = {}
        if log is not None:
            import re
            status = {int(m.group(1)): m.group(2) for m in re.finditer(r"logGeff node (\d+) \([^)]*\): \d+ calls, (\w+)",
                                                                       log.read_text(errors="replace"))}
        inner = min((j for j, r in enumerate(rows) if float(r["coordinate"]) > EDGE and status.get(j) == "converged"),
                    key=lambda j: float(rows[j]["coordinate"]), default=None)
        if inner is not None:
            node = json.loads((KIT / "logs" / f"profiles_{study}_nodes" / f"node_000_{inner:05d}.json").read_text())
            names = [q["name"] if isinstance(q, dict) else q
                     for q in json.loads((W.latest(f"results/*_sidr_drmd_{d}_combined_boost10") / "metadata.json")
                                         .read_text())["parameters"]]          # the network the study profiled
            x = dict(zip(names, map(float, node["x"])))
            g0 = float(rows[inner]["coordinate"])
            near = [g for g in nodes if g < EDGE and g0 - g <= NEIGHBOUR]
            spec["sets"][f"{d}_plateau_neighbour"] = dict(
                cobaya=f"runs/sidr_drmd_{d}_combined_boost10/cobaya.yaml", job_env=f"runs/sidr_drmd_{d}_combined_boost10/job.env",
                network=fits["network"],
                points=[dict(label=f"logGeff:{j:02d}", profile="logGeff", coordinate=g, params=dict(x, logGeff=g),
                             note=f"the converged node at log10 Geff = {g0:.2f} moved to {g}") for j, g in enumerate(near)])
            print(f"{d}: {len(near)} neighbour points ({', '.join(f'{g:.2f}' for g in near)}) from the node at {g0:.2f}")
    path.write_text(json.dumps(spec, indent=1))
    print("written:", path)


if __name__ == "__main__":
    main()
