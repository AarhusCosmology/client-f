"""Better values for the World Cup profiles with the cut (wc, without SH0ES) from points already evaluated, the first
profiles' points included (27 October 2026, Steen: "include the old points in the polish"):

    python wc_polish_setup.py             # first pass: smoke/chi2_points_wc_dncut_polish.json, set wc_dncut_polish
    python wc_polish_setup.py --second    # second pass: smoke/chi2_points_wc_dncut_polish2.json, set wc_dncut_polish2
    python wc_polish_setup.py --third     # third pass: smoke/chi2_points_wc_dncut_polish3.json, set wc_dncut_polish3
    python wc_polish_setup.py --fourth    # fourth pass, along the degeneracies: ..._polish4.json, set wc_dncut_polish4

The studies on the run's copy with the cut (sidr_drmd_wc_combined_boost10_dncut_{cosmo,fidm_wide,logGeff_wide}) ended
mostly at their evaluation limit, and at 13:02 on 27 October n_s, ln(10^10 A_s), tau and m_int never came below
Delta chi^2 ~ 2 although the best fit lies inside their ranges (wc_profiles_compare.py). Any true value at a point with the
profiled parameter at a node's value bounds the profile there, so for every node of those studies that did not converge
this moves, with the profiled parameter set to the node's value:
  - the best point at each of the N_NEIGHBOURS nearest coordinates of the same profile on either side (a continuation);
  - the N_NEAR best points within WINDOW node spacings in the profiled parameter, from any profile or fit;
  - the N_BEST best points overall;
  - the node optima of the same profile interpolated between the nearest coordinate on either side and extrapolated from
    the two nearest on each side and from the next two (the profile's ridge, as pp_polish_setup.py --second), clipped
    onto the box and out of the cut region.
The third pass improved no node by more than 1 in chi^2, while ln(10^10 A_s), tau_reio, n_s, m_int and H0 stayed jagged
by 2-3 near the minimum: a point moved in the profiled parameter alone leaves the degeneracies (A_s e^{-2 tau}, the
densities with n_s and H0). The fourth pass therefore moves the N_BEST best points overall and the best points at the
N_NEIGHBOURS nearest coordinates on either side along the posterior's degeneracy instead: every other parameter is
shifted by its regression on the profiled one, Cov(x_n, x_p) / Var(x_p) (x_p from the point's value to the node's), from
the posterior samples in the mode of the best fit (Delta N_eff > 0.1, -3.6 < log10 G_eff < -2), clipped onto the box
and out of the cut region.
Sources are every point with a true value outside 0 < Delta N_eff < CUT: the cut studies' nodes (log-posterior put back
on the run's objective, build_wc.CUT_PRIOR_SHIFT), the first profiles' nodes and the points earlier polish and plateau
jobs moved to them (build_wc.node_points, moved_results: the old points), and the AIC best fits; a later pass adds
the results of the earlier ones. Moves already evaluated by an earlier pass are left out. Points are evaluated with the run's
own Cobaya file: outside the cut its objective is the cut copy's up to the constant CUT_PRIOR_SHIFT. On Grendel:

    sbatch smoke/submit_chi2_points.sh smoke/chi2_points_wc_dncut_polish.json
"""
import json
import sys

import numpy as np

import build_wc as W

KIT = W.KIT
D = "wc"
CUT = W.B.CUT
RUN = f"runs/sidr_drmd_{D}_combined_boost10"
NEW = [f"sidr_drmd_{D}_combined_boost10_dncut_{k}" for k in ("cosmo", "fidm_wide", "logGeff_wide")]
N_NEIGHBOURS, N_NEAR, N_BEST, WINDOW = 2, 6, 4, 3


def meta():
    run = W.latest(f"results/*_sidr_drmd_{D}_combined_boost10")
    params = json.loads((run / "metadata.json").read_text())["parameters"]
    names = [q["name"] if isinstance(q, dict) else q for q in params]
    box = {q["name"]: (q["lower"], q["upper"]) for q in params if isinstance(q, dict) and "lower" in q}
    return names, box


def outside(q):
    return not 0 < q["delta_Neff_drmd"] < CUT


def new_nodes(names):
    """(profile, coordinate, parameters, ln L on the run's objective, converged) of the cut studies' nodes."""
    out = []
    for study in NEW:
        folder = KIT / "logs" / f"profiles_{study}_nodes"
        if not (folder / "surrogate_profiles.json").exists():
            continue
        profiled = json.loads((folder / "surrogate_profiles.json").read_text())["names"]
        for f in sorted(folder.glob("node_*.json")):
            node = json.loads(f.read_text())
            if len(node["x"]) != len(names) or not np.isfinite(node["logl"]):
                continue
            out.append((profiled[int(f.stem.split("_")[1])], float(node["coordinate"]), dict(zip(names, map(float, node["x"]))),
                        float(node["logl"]) - W.CUT_PRIOR_SHIFT, node["diagnostics"]["reason"] == "converged"))
    return out


PASSES = ("wc_dncut_polish", "wc_dncut_polish2", "wc_dncut_polish3", "wc_dncut_polish4")


def polish_results(names, sets=PASSES):
    """Every evaluated point of the polish passes: (profile, coordinate, parameters, ln L on the run's objective). The
    job's output holds index, label and values only; the point comes from the points file, matched by index and label."""
    out = []
    for name in sets:
        spec_file = KIT / "smoke" / f"chi2_points_{name}.json"
        if not spec_file.exists():
            continue
        pts = json.loads(spec_file.read_text())["sets"][name]["points"]
        for f in sorted(KIT.glob(f"logs/chi2_points_{name}_[0-9]*.json")):
            for e in json.loads(f.read_text())["results"]:
                q = pts[e["index"]]
                if q["label"] != e["label"]:
                    raise SystemExit(f"{f.name}: point {e['index']} is {e['label']} in the output and {q['label']} in {spec_file.name}")
                if np.isfinite(e.get("logpost", np.nan)):
                    out.append((q["profile"], float(q["coordinate"]), {n: float(q["params"][n]) for n in names}, float(e["logpost"])))
    return out


def sources(names, npass):
    """Every point with a true value outside the cut region, as (profile or None, coordinate or None, parameters, ln L, origin)."""
    out = [(p, c, q, l, "cut study node") for p, c, q, l, _ in new_nodes(names)]
    out += [(p, c, q, l, "first profiles' node") for p, c, q, l in W.node_points(D, names)]
    for e, q, f in W.moved_results(D):
        if q is not None and "logpost" in e and np.isfinite(e["logpost"]):
            p = e.get("profile")
            out.append(("f_idm_drmd" if p == "f_idm_drmd_wide" else p, e.get("coordinate"), {n: float(q[n]) for n in names},
                        float(e["logpost"]), f"{f.name} {e.get('label', '')}"))
    fits = W.chi2_points(D, "bestfits", merge=True)
    for k, e in fits["results"].items():
        out.append((None, None, {n: float(e["params"][n]) for n in names}, float(e["logpost"]), f"AIC {k}"))
    if npass > 1:
        out += [(p, c, q, l, "earlier pass") for p, c, q, l in polish_results(names, sets=PASSES[:npass - 1])]
    return [s for s in out if outside(s[2])]


def move_key(name, c, q):
    return (name, round(c, 6), tuple(round(q[n], 7) for n in sorted(q) if n != name))


def ridge(name, c, at, box):
    """(kind, coordinates used, parameters): the best points of the profile's coordinates interpolated and extrapolated to c."""
    below = sorted((k for k in at if k < c - 1e-9), reverse=True)
    above = sorted(k for k in at if k > c + 1e-9)
    pairs = [("interpolated", below[0], above[0])] if below and above else []
    for side in (below, above):
        for k in (0, 1):
            if len(side) >= k + 2:
                pairs.append(("extrapolated", side[k + 1], side[k]))
    for kind, a, b in pairs:
        qa, qb = at[a][0], at[b][0]
        f = (c - a) / (b - a)
        p = {n: qa[n] + f * (qb[n] - qa[n]) for n in qa}
        p[name] = c
        p = {n: min(max(v, box[n][0]), box[n][1]) if n in box else v for n, v in p.items()}
        if 0 < p["delta_Neff_drmd"] < CUT:
            p["delta_Neff_drmd"] = CUT
        yield kind, (a, b), p


def mode_covariance(names):
    """The weighted covariance of `names` over the posterior samples in the mode of the best fit (Delta N_eff > 0.1,
    -3.6 < log10 G_eff < -2); also wc_chain_setup.py's."""
    run = W.latest(f"results/*_sidr_drmd_{D}_combined_boost10")
    z = np.load(run / "analysis" / "best_posterior_it_0.npz", allow_pickle=True)
    zn = [str(n) for n in z["names"]]
    x, w = z["x"].astype(float)[:, [zn.index(n) for n in names]], z["weights"].astype(float)
    dn, lg = x[:, names.index("delta_Neff_drmd")], x[:, names.index("logGeff")]
    keep = (dn > 0.1) & (lg > -3.6) & (lg < -2.0)
    x, w = x[keep], w[keep] / w[keep].sum()
    m = w @ x
    return ((x - m).T * w) @ (x - m)


def regression(names):
    """{p: {n: Cov(x_n, x_p) / Var(x_p)}} from the posterior samples in the mode of the best fit (the fourth pass)."""
    C = mode_covariance(names)
    return {p: dict(zip(names, C[:, i] / C[i, i])) for i, p in enumerate(names)}


def along(q, name, c, beta, box):
    """q moved to name = c along the degeneracy (beta[name]), clipped onto the box and out of the cut region."""
    p = {n: v + beta[name][n] * (c - q[name]) for n, v in q.items()}
    p[name] = c
    p = {n: min(max(v, box[n][0]), box[n][1]) if n in box else v for n, v in p.items()}
    if 0 < p["delta_Neff_drmd"] < CUT:
        p["delta_Neff_drmd"] = CUT
    return p


def main():
    flags = sys.argv[1:]
    npass = 4 if "--fourth" in flags else 3 if "--third" in flags else 2 if "--second" in flags else 1
    names, box = meta()
    src = sources(names, npass)
    targets = sorted({(p, round(c, 6)) for p, c, _, _, conv in new_nodes(names) if not conv})
    for p, c in targets:
        assert p != "delta_Neff_drmd" or not 0 < c < CUT, (p, c)
    done = set()
    for name in PASSES[:npass - 1]:
        prev = json.loads((KIT / "smoke" / f"chi2_points_{name}.json").read_text())
        done |= {move_key(q["profile"], q["coordinate"], q["params"]) for st in prev["sets"].values() for q in st["points"]}
    by_value = sorted(src, key=lambda s: -s[3])[:N_BEST]
    beta = regression(names) if npass == 4 else None
    points, seen = [], set()
    for p, c in targets:
        # the best point at each coordinate of this profile (nodes of the cut studies and of the first profiles, moved points)
        at = {}
        for s in src:
            if s[0] == p and s[1] is not None and (round(s[1], 6) not in at or s[3] > at[round(s[1], 6)][1]):
                at[round(s[1], 6)] = (s[2], s[3], s[4])
        new_coords = sorted({round(cc, 6) for pp, cc, _, _, _ in new_nodes(names) if pp == p})
        spacing = float(np.median(np.diff(new_coords))) if len(new_coords) > 1 else 0.0
        below = sorted((k for k in at if k < c - 1e-9), reverse=True)[:N_NEIGHBOURS]
        above = sorted(k for k in at if k > c + 1e-9)[:N_NEIGHBOURS]
        chosen = [(at[k][0], f"best point at {p} = {k:.6g} ({at[k][2]}, ln L {at[k][1]:.2f})") for k in below + above]
        if npass == 4:
            chosen += [(s[2], f"{s[4]}, one of the {N_BEST} best points (ln L {s[3]:.2f})") for s in by_value]
            chosen = [(along(q, p, c, beta, box), f"{note}, along the degeneracy") for q, note in chosen]
        near = {}
        for s in src:
            v = round(s[2][p], 6)
            if abs(v - c) > 1e-9 and abs(v - c) <= WINDOW * spacing and (v not in near or s[3] > near[v][3]):
                near[v] = s
        if npass < 4:
            chosen += [(s[2], f"{s[4]} at {p} = {s[2][p]:.6g} (ln L {s[3]:.2f})")
                       for s in sorted(near.values(), key=lambda s: -s[3])[:N_NEAR]]
            chosen += [(s[2], f"{s[4]}, one of the {N_BEST} best points (ln L {s[3]:.2f})") for s in by_value]
            chosen += [(q, f"node optima at {p} = {a:.6g} and {b:.6g} {kind}") for kind, (a, b), q in ridge(p, c, at, box)]
        for j, (q, note) in enumerate(chosen):
            q = dict(q, **{p: c})
            key = move_key(p, c, q)
            if key in seen or key in done:
                continue
            seen.add(key)
            points.append(dict(label=f"{p}:{c}:{j:02d}", profile=p, coordinate=c, params=q, note=f"{note} moved to {p} = {c:.6g}"))
    name = PASSES[npass - 1]
    out = KIT / "smoke" / f"chi2_points_{name}.json"
    out.write_text(json.dumps(dict(made="wc_polish_setup.py" + {1: "", 2: " --second", 3: " --third", 4: " --fourth"}[npass],
                                   sets={name: dict(cobaya=f"{RUN}/cobaya.yaml", job_env=f"{RUN}/job.env", points=points)}),
                              indent=1))
    by = {}
    for q in points:
        by[q["profile"]] = by.get(q["profile"], 0) + 1
    print(f"{out.relative_to(KIT)}: {len(points)} points for {len(targets)} unconverged nodes from {len(src)} source points "
          f"outside the cut ({by}); about {len(points) / 0.575 / 3600:.1f} h on one q48 node")


if __name__ == "__main__":
    main()
