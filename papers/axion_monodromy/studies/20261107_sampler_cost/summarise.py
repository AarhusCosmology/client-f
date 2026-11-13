"""The numbers of the comparison with standard samplers, for the paper (build200.py reads sampler_cost.json).

    python summarise.py

Reads the outputs of this folder (README.md describes the runs) and writes sampler_cost.json. Calls are likelihood
calls, counted on the combined emulators in place of CLASS++.
  - Metropolis-Hastings (chains/<tag>zone1b, chains/<tag>full): Cobaya's progress files give the accepted samples N of
    all chains and the acceptance rate at each convergence test; N / acceptance is the number of proposals, and the
    calls are that times the calls per proposal of calls_per_proposal_<case>.json (a proposal outside the prior box
    costs no call). For each R-1 threshold the calls at the first test below it. A run that did not reach R-1 = 0.01 is
    extrapolated from its last test with R-1 proportional to 1 / calls (calls_001_extrapolated). Applied to the PDP+
    zone run, the one run that got there, the rule overestimates its calls by the factor in check_inverse_rule (from
    its first tests below 0.05 and 0.02); a borrowed ratio of that run predicted fewer calls for the H0WC zone than
    the H0WC run had already made without reaching 0.01.
  - Nested sampling (nested_<tag>500.log, dynesty, 500 live points): the calls per new live point in the steady state,
    the slope of ncall over the last 300 iterations. The number of iterations is n_dead = (n_dead / n_live) n_live with
    the ratio and the information H of BEST's nested sampling of the same combined emulator (its evidence JSON, live
    fraction 1e-3), and sigma(ln Z) = sqrt(H / n_live).
  - PROSPECT (prospect/output_pp_zone1b_*, prospect/nomcmc/): the convergence.json of analyse_prospect.py.
"""
import json
import re
from pathlib import Path

import os

import numpy as np

here = Path(__file__).resolve().parent
KIT_RES = Path(os.environ.get("CLIENT_PAPER_KIT", here.parents[1])) / "results"     # the cluster folder (README.md)
EMULATOR = {"pp": "20261107_084147_axion_pp_combined200", "wc": "20261107_084828_axion_wc_combined200"}
TAG = {"pp": "", "wc": "wc_"}
THRESHOLDS = (0.05, 0.02, 0.01)


def progress(case):
    """(calls, R-1) at each convergence test of a Metropolis-Hastings run."""
    rows = [l.split() for l in (here / "chains" / case / "mcmc.progress").read_text().splitlines() if not l.startswith("#")]
    r = json.loads((here / f"calls_per_proposal_{case}.json").read_text())["calls_per_proposal"]
    N, acc, R = (np.array([float(x[k]) for x in rows]) for k in (0, 2, 3))
    return r * N / acc, R, r


def first_below(calls, R, x):
    k = np.nonzero(R <= x)[0]
    return float(calls[k[0]]) if k.size else None


out = {}
mh = {}
for d in ("pp", "wc"):
    for case in ("zone1b", "full"):
        calls, R, r = progress(TAG[d] + case)
        chains = len(list((here / "chains" / (TAG[d] + case)).glob("mcmc.[0-9].txt")))
        mh[(d, case)] = calls, R
        out.setdefault(d, {})[f"mh_{case}"] = dict(
            chains=chains, calls_per_proposal=r, calls=float(calls[-1]), R_last=float(R[-1]),
            calls_at={str(x): first_below(calls, R, x) for x in THRESHOLDS})
z = out["pp"]["mh_zone1b"]
out["check_inverse_rule"] = {x: z["calls_at"][x] * float(x) / 0.01 / z["calls_at"]["0.01"] for x in ("0.05", "0.02")}
for d in ("pp", "wc"):
    z, f = out[d]["mh_zone1b"], out[d]["mh_full"]
    if z["calls_at"]["0.01"] is None:
        z["calls_001_extrapolated"] = z["calls"] * z["R_last"] / 0.01
        c, R = mh[(d, "zone1b")]
        late = c >= z["calls_at"]["0.02"]
        z["R_after_002"] = [float(R[late].min()), float(R[late].max())]
    f["calls_001_extrapolated"] = f["calls"] * f["R_last"] / 0.01
    # nested sampling
    text = (here / f"nested_{TAG[d]}500.log").read_text(errors="replace").replace("\r", "\n")
    pairs = sorted({(int(a), int(b)) for a, b in re.findall(r"\|\s*(\d+)/\d+ \[[^\]]*\| nc: *\d+ \| ncall: (\d+)", text)})
    it, nc = np.array(pairs).T
    sel = it >= it.max() - 300
    per_point = float(np.polyfit(it[sel], nc[sel], 1)[0])
    ev = json.loads((KIT_RES / EMULATOR[d] / "evidence/evidence_it_0_nlive2048.json").read_text())
    ratio, H = ev["n_dead"] / ev["n_live"], ev["information_H"]
    out[d]["nested"] = dict(n_live_run=500, calls=int(nc.max()), iterations=int(it.max()), calls_per_point=per_point,
                            slices=95 if d == "pp" else 110,
                            best=dict(emulator=EMULATOR[d], n_live=ev["n_live"], n_dead=ev["n_dead"], dead_per_live=ratio, H=H,
                                      lnZ=ev["lnZ"], sigma_lnZ=ev["sigma_lnZ"]),
                            cases={str(n): dict(sigma_lnZ=float(np.sqrt(H / n)), iterations=ratio * n, calls=ratio * n * per_point)
                                   for n in (500, 2048)})
# PROSPECT
settings = {}
for folder in sorted((here / "prospect").glob("output_pp_zone1b_*")):
    c = json.loads((folder / "convergence.json").read_text())
    pts = c["points"].values()
    steps = c["steps_per_iteration"]
    to01 = [p["calls_to"]["0.1"] for p in pts]
    settings[str(steps)] = dict(folder=folder.name, steps=steps, chains=len(to01), repetitions=2, iterations=20,
                                calls_per_point=2 * 20 * (steps + 1),
                                gap_at_20=[min(p["gap_at_schedule_end"] for p in pts), max(p["gap_at_schedule_end"] for p in pts)],
                                calls_to_01=[min(to01), max(to01)], A_osc=sorted({p["A_osc"] for p in pts}),
                                start_gap=[min(p["start_gap"] for p in pts), max(p["start_gap"] for p in pts)])
nomcmc = {}
for folder in sorted((here / "prospect/nomcmc").glob("output_*")):
    c = json.loads((folder / "convergence.json").read_text())
    start = float(re.search(r"_w(\d+)$", folder.name).group(1))
    nomcmc[str(start)] = [dict(omega=p["omega"], gap_at_20=p["gap_at_schedule_end"]) for p in c["points"].values()]
exact = json.loads((here / "prospect/exact_profile_pp_zone1b.json").read_text())
out["pp"]["prospect"] = dict(version="24.0.2", zone=[38.25, 60.5], settings=settings, nomcmc=nomcmc,
                             exact={k: v["logl"] for k, v in exact.items()})
(here / "sampler_cost.json").write_text(json.dumps(out, indent=1))
print(json.dumps({d: {k: v for k, v in out[d].items() if k != "prospect"} for d in out}, indent=1))
