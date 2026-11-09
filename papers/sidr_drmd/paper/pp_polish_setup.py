"""Better values for the Pantheon+ profile nodes recomputed with the cut at Delta N_eff = 0.01 (27 October 2026):

    python pp_polish_setup.py             # first pass: smoke/chi2_points_pp_polish.json
    python pp_polish_setup.py --second    # second pass: smoke/chi2_points_pp_polish2.json

The 25 nodes of the Sec. 4 profiles whose optimum lay at 0 < Delta N_eff < 0.01 were recomputed on the run's copy with
the cut (kit analysis/make_cut_run.py, studies sidr_drmd2_pp_combined_boost10_dncut_*). Some ended at their budget off
the curve (27 October: m_int's last two nodes at Delta chi^2 10.8 and 10.9 after 5.5 at the node before). Any true value
at a point with the profiled parameter at a node's value bounds the profile there, so this moves, for every recomputed
node, the best points of the N_NEIGHBOURS nearest nodes on either side of the same profile (old nodes outside the cut
region and recomputed ones) to the node's value, as profile_polish.py does for the World Cup profiles. Writes the kit's
smoke/chi2_points_pp_polish.json, set pp_profiles_dncut_polish, evaluated with the cut copy's Cobaya file (the cut
studies' own objective; build.py puts it back with CUT_PRIOR_SHIFT, and its profile breakdown reads the set). On Grendel:

    sbatch smoke/submit_chi2_points.sh smoke/chi2_points_pp_polish.json

The second pass (--second) also takes the first pass's evaluated points (logs/chi2_points_pp_profiles_dncut_polish_*.json)
as sources, so that a node improved by it passes its new best on (27 October: m_int's node at 0.278 came down from
10.8 to 7.7, the last node at 0.288 stayed at 10.9), and leaves out every move the first pass already evaluated. It also follows the profile's
ridge: moving a point in the profiled parameter alone does not, where the other parameters' optimum moves with it (m_int
with H0 and omega_cdm: the first pass's moves onto m_int's last nodes cost 6-7 in chi^2). For every recomputed node it
adds the node optima (the profile searches' own end points, not moved points) interpolated between the nearest node on
either side, and extrapolated linearly from the two nearest nodes on each side and from the next two, clipped onto the
box and out of the cut region (Delta N_eff to the cut where it falls inside). Set
pp_profiles_dncut_polish2, file smoke/chi2_points_pp_polish2.json.
"""
import csv
import json
import sys

import numpy as np

import build as B

KIT = B.KIT
D = "pp"
RUN = "runs/sidr_drmd2_pp_combined_boost10_dncut"
N_NEIGHBOURS = 2


def names():
    run = B.latest(f"results/*_sidr_drmd2_{D}_combined_boost10")
    return [q["name"] if isinstance(q, dict) else q for q in json.loads((run / "metadata.json").read_text())["parameters"]]


def all_nodes(names_):
    """(parameter, coordinate, parameters, log-objective on the run's objective) of every Pantheon+ profile node with a
    finite value outside 0 < Delta N_eff < CUT: the original studies' and the recomputed ones'."""
    out = []
    for folder, table in B.CUT_NODE_DIRS[D]:
        order = []
        for r in csv.DictReader(open(KIT / table)):
            if r["parameter"] not in order:
                order.append(r["parameter"])
        for f in sorted((KIT / folder).glob("node_*.json")):
            node = json.loads(f.read_text())
            out.append((order[int(f.stem.split("_")[1])], float(node["coordinate"]), node, 0.0))
    for study in B.CUT_STUDIES[D]:
        folder = KIT / "logs" / f"profiles_{study}_nodes"
        if not folder.exists():
            continue
        order = json.loads((folder / "surrogate_profiles.json").read_text())["names"]
        for f in sorted(folder.glob("node_*.json")):
            node = json.loads(f.read_text())
            out.append((order[int(f.stem.split("_")[1])], float(node["coordinate"]), node, B.CUT_PRIOR_SHIFT))
    kept = []
    for name, c, node, shift in out:
        if len(node["x"]) != len(names_) or not np.isfinite(node["logl"]):
            continue
        q = dict(zip(names_, map(float, node["x"])))
        if 0 < q["delta_Neff_drmd"] < B.CUT:
            continue
        kept.append((name, c, q, float(node["logl"]) - shift))
    return kept


def first_pass(names_):
    """The first pass's evaluated points as sources ((profile, coordinate, parameters, log-objective on the run's
    objective), and the moves it made ((profile, coordinate, rounded source parameters)."""
    out, done = [], set()
    for f in sorted(KIT.glob(f"logs/chi2_points_{D}_profiles_dncut_polish_*.json")):
        for e in json.loads(f.read_text())["results"]:
            if "profile" not in e or "params" not in e:
                continue
            q = {n: float(e["params"][n]) for n in names_}
            done.add(move_key(e["profile"], e["coordinate"], q))
            if "logpost" in e and np.isfinite(e["logpost"]):
                out.append((e["profile"], float(e["coordinate"]), q, float(e["logpost"]) - B.CUT_PRIOR_SHIFT))
    return out, done


def move_key(name, c, q):
    """A move: the target node and the point moved there (its parameters, the profiled one at the target's value)."""
    return (name, round(c, 6), tuple(round(q[n], 7) for n in sorted(q) if n != name))


def bounds():
    run = B.latest(f"results/*_sidr_drmd2_{D}_combined_boost10")
    return {q["name"]: (q["lower"], q["upper"]) for q in json.loads((run / "metadata.json").read_text())["parameters"]}


def ridge_points(name, c, optima, box):
    """(kind, coordinates used, parameters) of the node optima of profile `name` interpolated to c between the nearest
    node on either side, and extrapolated to c from the two nearest on each side."""
    best = {}
    for t in optima:
        if t[0] == name and abs(t[1] - c) > 1e-9 and (round(t[1], 6) not in best or t[3] > best[round(t[1], 6)][3]):
            best[round(t[1], 6)] = t
    below = sorted((t for k, t in best.items() if k < c), key=lambda t: c - t[1])
    above = sorted((t for k, t in best.items() if k > c), key=lambda t: t[1] - c)
    pairs = []
    if below and above:
        pairs.append(("interpolated", below[0], above[0]))
    for side in (below, above):
        for k in (0, 1):                  # from the two nearest nodes, and from the next two (the nearest may be poor)
            if len(side) >= k + 2:
                pairs.append(("extrapolated", side[k + 1], side[k]))
    for kind, a, b in pairs:
        f = (c - a[1]) / (b[1] - a[1])
        p = {n: a[2][n] + f * (b[2][n] - a[2][n]) for n in a[2]}
        p[name] = c
        # onto the box and out of the cut region (its boundary): any point there bounds the profile
        p = {n: min(max(v, box[n][0]), box[n][1]) if n in box else v for n, v in p.items()}
        if 0 < p["delta_Neff_drmd"] < B.CUT:
            p["delta_Neff_drmd"] = B.CUT
        yield kind, (a[1], b[1]), p


def main():
    second = "--second" in sys.argv[1:]
    names_ = names()
    nodes = all_nodes(names_)
    optima = list(nodes)
    done = set()
    if second:
        moved, done = first_pass(names_)
        if not moved:
            raise SystemExit("no first-pass results in logs/ (chi2_points_pp_profiles_dncut_polish_*.json): sync first")
        nodes += moved
    targets = sorted(B.cut_nodes(D, names_))
    sets_points, seen = [], set()
    for name, c in targets:
        same = [t for t in nodes if t[0] == name and abs(t[1] - c) > 1e-9]
        # the best point at each coordinate of this profile, then the nearest coordinates on either side
        best = {}
        for t in same:
            key = round(t[1], 6)
            if key not in best or t[3] > best[key][3]:
                best[key] = t
        below = sorted((t for k, t in best.items() if k < c), key=lambda t: c - t[1])[:N_NEIGHBOURS]
        above = sorted((t for k, t in best.items() if k > c), key=lambda t: t[1] - c)[:N_NEIGHBOURS]
        for j, (_, c0, q, logl) in enumerate(below + above):
            p = dict(q, **{name: c})
            key = move_key(name, c, p)
            if key in seen or key in done:
                continue
            seen.add(key)
            sets_points.append(dict(label=f"{name}:{round(c, 6)}:{j:02d}", profile=name, coordinate=c, params=p,
                                    note=f"the best point of the node at {name} = {c0:.6g} (true ln L {logl:.2f}) moved to {c:.6g}"))
        if second:
            for j, (kind, (ca, cb), p) in enumerate(ridge_points(name, c, optima, bounds()), start=10):
                key = move_key(name, c, p)
                if key in seen or key in done:
                    continue
                seen.add(key)
                sets_points.append(dict(label=f"{name}:{round(c, 6)}:{j:02d}", profile=name, coordinate=c, params=p,
                                        note=f"the node optima at {name} = {ca:.6g} and {cb:.6g} {kind} to {c:.6g}"))
    set_name = "pp_profiles_dncut_polish2" if second else "pp_profiles_dncut_polish"
    spec = dict(made="pp_polish_setup.py" + (" --second" if second else ""), sets={set_name: dict(
        cobaya=f"{RUN}/cobaya.yaml", job_env=f"{RUN}/job.env", points=sets_points)})
    out = KIT / "smoke" / ("chi2_points_pp_polish2.json" if second else "chi2_points_pp_polish.json")
    out.write_text(json.dumps(spec, indent=1))
    by = {}
    for q in sets_points:
        by[q["profile"]] = by.get(q["profile"], 0) + 1
    print(f"{out.relative_to(KIT)}: {len(sets_points)} points for {len(targets)} nodes ({by})")


if __name__ == "__main__":
    main()
