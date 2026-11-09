"""Better values for the nodes of a profile likelihood from points already evaluated (25 October 2026).

    python profile_polish.py wc:logGeff wc:f_idm_drmd wcmb:f_idm_drmd wcmb:logGeff
    python profile_polish.py --unconverged wc wcmb          # second pass: every unconverged node of every profile

Any true log-likelihood at a point with the profiled parameter at a node's value bounds the profile there from below (in
ln L; from above in chi^2), so the profile is the best such value found. On the World Cup data the final emulators have
false maxima along the whole-prior profiles of log10 Geff and f_idm, and many nodes there ended at their budget far
above their neighbours (25 October: wc's log10 Geff nodes jump between 0.6 and 15 on the plateau, wcmb's f_idm nodes
between 21 and 96). This moves good points of other nodes to every node: for each node of the profile, the N_BEST points
of highest true value and the N_NEAR points nearest in the profiled parameter, among every profile node of the data set
(logs/profiles_sidr_drmd_<data>_combined_boost10_*_nodes/node_*.json, any parameter's) and the AIC best fits, each with
the profiled parameter set to the node's value. Writes the kit's smoke/chi2_points_wc_polish.json, set
<data>_polish_<parameter> (profile, coordinate of every point). On Grendel:

    sbatch smoke/submit_chi2_points.sh smoke/chi2_points_wc_polish.json

build_wc.py keeps, for every node, the best value of the node itself and of every point evaluated at its coordinate.
"""
import csv
import json
import sys

import numpy as np

import build_wc as W

KIT = W.KIT
N_BEST, N_NEAR = 10, 10
N_BEST2, N_NEAR2 = 6, 6          # the second pass (--unconverged): more nodes, fewer points each
N_NEIGHBOURS = 2                 # and the best point at each of the 2 nearest nodes on either side (26 October)


def sources(d):
    """Every point of d with a known true value: profile nodes (in the boost10 network's parameter order) and AIC fits."""
    names = [q["name"] if isinstance(q, dict) else q
             for q in json.loads((W.latest(f"results/*_sidr_drmd_{d}_combined_boost10") / "metadata.json").read_text())["parameters"]]
    out = []
    for f in sorted(KIT.glob(f"logs/profiles_sidr_drmd_{d}_combined_boost10_*_nodes/node_*.json")):
        node = json.loads(f.read_text())
        if len(node["x"]) == len(names) and np.isfinite(node["logl"]):
            out.append(dict(params=dict(zip(names, map(float, node["x"]))), logl=float(node["logl"]),
                            origin=f"{f.parent.name}/{f.name}"))
    fits = W.chi2_points(d, "bestfits", merge=True)
    for k, e in fits["results"].items():
        out.append(dict(params=dict(e["params"]), logl=float(e["logpost"]), origin=f"AIC {k} ({e['fit_label']})"))
    return out


def moved_points(d, names):
    """The points of earlier plateau and polish jobs of d, with their true values (from the job output's params, or
    the points file they came from, checked by coordinate)."""
    out = []
    for f in sorted(KIT.glob(f"logs/chi2_points_{d}_plateau*_*.json")) + sorted(KIT.glob(f"logs/chi2_points_{d}_polish*_*.json")):
        r = json.loads(f.read_text())
        spec = json.loads((KIT / "smoke" / ("chi2_points_wc_polish.json" if "_polish" in r["set"]
                                            else "chi2_points_wc_plateau.json")).read_text())["sets"].get(r["set"], {})
        pts = {q["label"]: q for q in spec.get("points", [])}
        for e in r["results"]:
            q = e.get("params") or (pts[e["label"]]["params"] if e["label"] in pts
                                    and abs(pts[e["label"]]["coordinate"] - e.get("coordinate", 1e9)) < 1e-9 else None)
            if "logpost" in e and q is not None:
                out.append(dict(params={n: float(q[n]) for n in names}, logl=float(e["logpost"]), origin=f"{f.name} {e['label']}"))
    return out


def all_nodes(d):
    """[(parameter, coordinate)] of every profile node of d, over all its studies in logs/."""
    out = set()
    for folder in sorted(KIT.glob(f"logs/profiles_sidr_drmd_{d}_combined_boost10_*_nodes")):
        names = json.loads((folder / "surrogate_profiles.json").read_text())["names"]
        for f in sorted(folder.glob("node_*.json")):
            out.add((names[int(f.stem.split("_")[1])], float(json.loads(f.read_text())["coordinate"])))
    return sorted(out)


def unconverged(d):
    """[(parameter, coordinate)] of every profile node of d that did not converge, over all its studies in logs/."""
    out = set()
    for folder in sorted(KIT.glob(f"logs/profiles_sidr_drmd_{d}_combined_boost10_*_nodes")):
        names = json.loads((folder / "surrogate_profiles.json").read_text())["names"]
        for f in sorted(folder.glob("node_*.json")):
            k = int(f.stem.split("_")[1])
            node = json.loads(f.read_text())
            if node["diagnostics"]["reason"] != "converged":
                out.add((names[k], float(node["coordinate"])))
    return sorted(out)


def best_near(src, p, c, n, window):
    """n points near c in p: the best point at each value of p within `window` of c (other than c itself, the node's
    own tries), highest true value first, filled up with the nearest values. Nearest by value alone picked copies of one
    point at the node's own coordinate, and the nearest distinct values poor points (26 October, wcmb log10 Geff)."""
    best = {}
    for s in src:
        v = round(s["params"][p], 6)
        if v != round(c, 6) and (v not in best or s["logl"] > best[v]["logl"]):
            best[v] = s
    inside = sorted((v for v in best if abs(v - c) <= window), key=lambda v: -best[v]["logl"])[:n]
    rest = sorted((v for v in best if v not in inside), key=lambda v: abs(v - c))[:n - len(inside)]
    return [best[v] for v in inside + rest]


def neighbour_best(src, p, c, coords, k=N_NEIGHBOURS):
    """The best point found at each of the k nearest nodes of p on either side of c (any study), as evaluated there:
    a continuation from the neighbours. best_near ranks points by their own value, so points near the best fit always
    win and the neighbours' optima were never moved (26 October, wcmb f_idm = 0.067 next to the plateau beyond 0.08)."""
    others = [x for x in coords if abs(x - c) > 1e-9]
    below = sorted((x for x in others if x < c), reverse=True)[:k]
    above = sorted(x for x in others if x > c)[:k]
    out = []
    for x in below + above:
        at = [s for s in src if abs(s["params"][p] - x) < 1e-6]
        if at:
            out.append(max(at, key=lambda s: s["logl"]))
    return out


def main():
    if sys.argv[1:2] == ["--unconverged"]:
        # the second pass (26 October): every unconverged node of every profile, from every point evaluated so far;
        # --only P ... restricts it to those profiles, --name N names the set <data>_<N> (default polish2)
        import argparse
        a = argparse.ArgumentParser()
        a.add_argument("data", nargs="+")
        a.add_argument("--only", nargs="+")
        a.add_argument("--name", default="polish2")
        args = a.parse_args(sys.argv[2:])
        path = KIT / "smoke" / "chi2_points_wc_polish.json"
        spec = json.loads(path.read_text()) if path.exists() else dict(made="profile_polish.py", sets={})
        for d in args.data:
            names = [q["name"] if isinstance(q, dict) else q for q in json.loads(
                (W.latest(f"results/*_sidr_drmd_{d}_combined_boost10") / "metadata.json").read_text())["parameters"]]
            src = sources(d) + moved_points(d, names)
            unconverged_all = all_nodes(d)
            by_value = sorted(src, key=lambda s: -s["logl"])[:N_BEST2]
            points = []
            targets = [(p, c) for p, c in unconverged(d) if not args.only or p in args.only]
            # the window: three node spacings of the parameter's profiles (the median gap between its nodes)
            window = {p: 3 * float(np.median(np.diff(sorted({c for q, c in unconverged_all if q == p}))))
                      for p in {p for p, _ in targets}}
            # points an earlier set of d already evaluated (same profile, node and point) are not asked for again
            # (26 October: a quarter of wc_polish3 repeated wc_polish2)
            def key(q):
                return q["profile"], round(q["coordinate"], 6), tuple(round(v, 7) for _, v in sorted(q["params"].items()))
            done = {key(q) for name, st in spec["sets"].items() if name.startswith(f"{d}_") and name != f"{d}_{args.name}"
                    for q in st["points"] if "profile" in q}
            repeated = 0
            coords = {p: sorted({c for q, c in unconverged_all if q == p}) for p, _ in targets}
            for j, (p, c) in enumerate(targets):
                near = best_near(src, p, c, N_NEAR2, window[p])
                chosen = {s["origin"]: s for s in by_value + near + neighbour_best(src, p, c, coords[p])}
                for k, s in enumerate(chosen.values()):
                    q = dict(label=f"{p}:{j:03d}:{k:02d}", profile=p, coordinate=c, params=dict(s["params"], **{p: c}),
                             note=f"{s['origin']} (true ln L {s['logl']:.2f}) moved to {p} = {c}")
                    if key(q) in done:
                        repeated += 1
                    else:
                        points.append(q)
            print(f"{d}: {repeated} points left out, evaluated in an earlier set")
            spec["sets"][f"{d}_{args.name}"] = dict(cobaya=f"runs/sidr_drmd_{d}_combined_boost10/cobaya.yaml",
                                                   job_env=f"runs/sidr_drmd_{d}_combined_boost10/job.env", points=points)
            print(f"{d}: {len(targets)} unconverged nodes, {len(src)} source points, {len(points)} points (set {d}_{args.name})")
        path.write_text(json.dumps(spec, indent=1))
        print("written:", path)
        return
    targets = [a.split(":", 1) for a in sys.argv[1:]] or [["wcmb", "f_idm_drmd"]]
    path = KIT / "smoke" / "chi2_points_wc_polish.json"
    spec = json.loads(path.read_text()) if path.exists() else dict(made="profile_polish.py", sets={})
    for d, p in targets:
        study = "logGeff_wide" if p == "logGeff" else "fidm_wide"
        csv_file = KIT / "logs" / f"profiles_sidr_drmd_{d}_combined_boost10_{study}.csv"
        coords = [float(r["coordinate"]) for r in csv.DictReader(open(csv_file))]
        src = sources(d)
        by_value = sorted(src, key=lambda s: -s["logl"])[:N_BEST]
        points = []
        for j, c in enumerate(coords):
            near = sorted(src, key=lambda s: abs(s["params"][p] - c))[:N_NEAR]
            chosen = {s["origin"]: s for s in by_value + near}
            for k, s in enumerate(chosen.values()):
                points.append(dict(label=f"{p}:{j:02d}:{k:02d}", profile=p, coordinate=c, params=dict(s["params"], **{p: c}),
                                   note=f"{s['origin']} (true ln L {s['logl']:.2f}) moved to {p} = {c}"))
        spec["sets"][f"{d}_polish_{p}"] = dict(cobaya=f"runs/sidr_drmd_{d}_combined_boost10/cobaya.yaml",
                                              job_env=f"runs/sidr_drmd_{d}_combined_boost10/job.env", points=points)
        print(f"{d} {p}: {len(coords)} nodes, {len(src)} source points, {len(points)} points")
    path.write_text(json.dumps(spec, indent=1))
    print("written:", path)


if __name__ == "__main__":
    main()
