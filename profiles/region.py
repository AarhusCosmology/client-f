"""The profile setting `region`: a network trained on boosts is searched inside their region.

A run with `acquisition.boost.region` writes RUN/boost/region_it_N.npz (sampling/region.py):
the part of the box the posterior points of its boosts reach. From then on the run's own
chains, audit and evidence see the network only there, and the network is not kept honest
anywhere else: it has maxima outside that the truth does not have. A profile is a search for
the network's maximum on a slice of the box, so it finds them. On a real 49-parameter run the
optima of the bare network were 40 posterior sigma from the data, 90 to 160 above the true peak
in the network and 600 to 700 below it in truth, and no node of the true stage converged; in
the region every node of the same study converged in one gradient round (docs/create_profiles.md,
"Networks trained on boosts").

`region: run` (the default) uses the region as the run saved it, `region: E` puts its edge at E
times the reach of the boost points, `region: null` is the bare network over the whole box,
which is what every study made before the key used. A network of the ordinary loop has no
region and is used over the whole box whatever the setting.

A node whose optimum is not inside the region cannot be computed from this network: its slice
of the box misses the region, or the network still rises where the region ends. Such nodes are
cut from the ends of the parameter's range, the scan is repeated on the shorter range with the
same number of nodes, and the cut is recorded (`region` in ranges.json, `region_cuts` in the
scan's file). The study ends only when fewer than two nodes of a parameter are inside.
"""
import json
from pathlib import Path

import numpy as np

# A node counts as inside when its optimum lies below this share of the region's edge. A search
# pressed against the edge ends just under it, and an optimum that close is no stationary point
# of the network.
INSIDE = 0.98
# Scans of one set of grids, the first included, before the study gives up cutting.
PASSES = 4


from sampling import targets
from sampling.targets import region_file


def view_region(run):
    """A derived-coordinate view's record of the region of its source run (view.json `region`,
    derived/coordinates.py), or None. Raises ValueError for a view made before views carried
    the region, from a network that has one: that view is the bare network."""
    path = Path(run) / "view.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    if summary.get("source_region") and "region" not in summary:
        raise ValueError(
            f"this view was made before views carried the region of their run's boosts ({summary['source_region']}): "
            "its network is the bare one. Make the view again to search the network inside the region, or set "
            "`region: null` in the profile settings to search the bare network over the whole box")
    return summary.get("region")


def resolve(run, iteration, setting):
    """What a study's results depend on (sampling/targets.py resolve): None for the bare network,
    or the region's file, hash and edge. For a view, the region its source run saved, copied
    into the view, and the pivot map that takes the view's points to the run's coordinates."""
    if setting is None:
        return None
    copied = view_region(run)
    if copied is None:
        return targets.resolve(run, iteration, setting)
    from run.storage import digest
    from sampling.region import Region

    path, pivots = Path(run) / copied["file"], Path(run) / "likelihood_input" / "pivot_map.keras"
    if digest(path) != copied["sha256"]:
        raise ValueError(f"Surrogate region changed: {path} is not the copy the view was made with")
    edge = Region.load(path).inflate if setting == "run" else float(setting)
    return dict(file=copied["file"], sha256=digest(path), edge=float(edge),
                coordinates=dict(file="likelihood_input/pivot_map.keras", sha256=digest(pivots)))


def load(run, entry):
    """The region of a manifest's entry, with the entry's edge: a Region, or for a view a
    ViewRegion, which measures the view's points in the run's coordinates."""
    region = targets.load(run, entry)
    if "coordinates" not in entry:
        return region
    from derived.coordinates import ViewRegion
    from model.network import load_model
    from run.storage import digest

    path = Path(run) / entry["coordinates"]["file"]
    if digest(path) != entry["coordinates"]["sha256"]:
        raise ValueError(f"Surrogate region changed: {path}")
    return ViewRegion(region, load_model(path))


def restrict(network, region):
    """What the searches see of a network: the network inside `region`, its wall outside
    (model.network.restrict_model; for a view, derived.coordinates.restrict_view)."""
    from derived.coordinates import ViewRegion, restrict_view
    from model.network import restrict_model

    if isinstance(region, ViewRegion):
        return restrict_view(network, region.region)
    return restrict_model(network, region)


def describe(region, entry):
    seen = (", in the coordinates of the view's source run (a point z of the view is inside when x(z) is)"
            if "coordinates" in entry else "")
    return (f"the network is searched inside {Path(entry['file']).name} only{seen}: {len(region)} mode(s), "
            f"{' + '.join(map(str, region.rows))} boost points, edge at {region.inflate:g} times their reach of "
            f"{', '.join(f'{r:.1f}' for r in region.reach)} sigma; outside, the region's wall")


def outside(region, result):
    """{name: one flag per node}: the node's optimum is at the region's edge or beyond it."""
    return {name: region.rho(np.asarray(record["xopt"], float)) > INSIDE * region.inflate
            for name, record in result["profiles"].items()}


def at_edge(region, result):
    """What to add to the error of a scan that did not converge: whether the network's best
    point inside the region lies at the region's edge."""
    rho = float(region.rho(np.asarray(result["best_x"], float)[None])[0])
    if rho <= INSIDE * region.inflate:
        return ""
    return (f" (the network's maximum inside the region of its boosts lies at the region's edge, rho = {rho:.3g} "
            f"of {region.inflate:g}: it rises above its value at the data there. A smaller `region` keeps the "
            "search to where the boost points are)")


def shorten(result, flags, grids, names):
    """The grids with the nodes outside the region cut from their ends.

    Of a parameter's nodes inside, the consecutive ones around the node nearest the best fit
    are kept, and the grid becomes as many nodes as before between the first and the last of
    them. Returns (grids, cuts) with cuts[name] = dict(asked, kept, outside). Raises ValueError
    for a parameter with fewer than two such nodes.
    """
    full = list(result["full_parameter_names"])
    best = np.asarray(result["best_x"], float)
    new, cuts = [], {}
    for name, grid in zip(names, grids):
        grid = np.asarray(grid, float)
        out = np.asarray(flags[name], bool)
        if not out.any():
            new.append(grid)
            continue
        inside = np.flatnonzero(~out)
        if len(inside):
            centre = inside[np.argmin(abs(grid[inside] - best[full.index(name)]))]
            first = last = int(centre)
            while first > 0 and not out[first - 1]:
                first -= 1
            while last < len(grid) - 1 and not out[last + 1]:
                last += 1
        if not len(inside) or first == last:
            raise ValueError(
                f"{name}: fewer than two neighbouring nodes between {grid[0]:g} and {grid[-1]:g} have "
                "their optimum inside the region of the network's boosts; give a range inside it, or "
                "set `region` in the profile settings (a larger edge, or null for the bare network)")
        new.append(np.linspace(grid[first], grid[last], len(grid)))
        cuts[name] = dict(asked=[float(grid[0]), float(grid[-1])],
                          kept=[float(grid[first]), float(grid[last])],
                          outside=[float(v) for v in grid[out]])
    return new, cuts


def scan(scan_once, region, grids, names, log=print):
    """`scan_once(grids)` until no node is outside the region, the grids cut in between.

    Returns the last result; when anything was cut it carries `region_cuts`, per parameter the
    range first asked for, the one kept and every node value found outside. After PASSES scans
    with nodes still outside it raises RuntimeError.
    """
    cuts = {}
    for attempt in range(PASSES):
        result = scan_once(grids)
        flags = outside(region, result)
        if not any(flag.any() for flag in flags.values()):
            if cuts:
                result["region_cuts"] = cuts
            return result
        if attempt == PASSES - 1:
            break
        grids, new = shorten(result, flags, grids, names)
        for name, cut in new.items():
            earlier = cuts.get(name)
            cuts[name] = dict(asked=earlier["asked"] if earlier else cut["asked"], kept=cut["kept"],
                              outside=(earlier["outside"] if earlier else []) + cut["outside"])
            log(f"region: {len(cut['outside'])} node(s) of {name} have their optimum outside the region of the "
                f"network's boosts; its range is cut from [{cut['asked'][0]:g}, {cut['asked'][1]:g}] to "
                f"[{cut['kept'][0]:g}, {cut['kept'][1]:g}] and scanned again", flush=True)
    still = ", ".join(f"{name} ({int(flag.sum())})" for name, flag in flags.items() if flag.any())
    raise RuntimeError(
        f"after {PASSES} scans nodes still have their optimum at the edge of the region of the network's "
        f"boosts: {still}. The network rises towards the edge there; give ranges further inside, or set "
        "`region` in the profile settings")
