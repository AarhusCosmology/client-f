"""PROSPECT's simulated annealing against the exact profile of the emulator, per profile point.

    PYTHONPATH=pylib:. python prospect/analyse_prospect.py prospect/output_pp_zone1b_example [more output folders]

Reads each output folder's state.pkl (the finished OptimiseTasks: per profile value, repetition and iteration, the best
-ln L found in that iteration) and prospect/exact_profile_pp_zone1b.json (multi-start L-BFGS-B with the network's
gradients). For each profile value and repetition: the running best after each iteration, its gap to the exact profile in
chi^2 (2 x Delta ln L), and the likelihood calls needed to get within 0.1, 0.5 and 1 in chi^2 (steps_per_iteration + 1
calls per iteration, the Num_evals PROSPECT prints), the gap after iteration 20 (SCHEDULE_END: max_iterations, where the
exponential schedule reaches the lower end of temperature_range; without chi2_tolerance PROSPECT does not stop by itself,
and these runs were stopped after 25-40 iterations), the omega of the best point, and the gap of the starting point (the
emulator at the InitialiseOptimiserTask's initial_bestfit: the best sample of the MCMC's bin, or the given position).
Writes <folder>/convergence.json.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
if (here / "exact_profile_pp_zone1b.json").exists():
    exact = {float(k): v["logl"] for k, v in json.loads((here / "exact_profile_pp_zone1b.json").read_text()).items()}
else:  # exact_profile.py still running: the points it has printed
    exact = {float(l.split()[1].rstrip(":")): float(l.split()[4]) for l in (here / "exact_profile.log").read_text().splitlines()
             if l.startswith("A_osc")}
THRESHOLDS = (0.1, 0.5, 1.0)
SCHEDULE_END = 20

for folder in map(Path, sys.argv[1:]):
    state = pickle.loads((folder / "state.pkl").read_bytes())
    tasks = [t for t in state.done.values() if t.type == "OptimiseTask"]
    steps = tasks[0].config.profile.steps_per_iteration
    per_iter = steps + 1
    import emulator as em
    starts = {}
    for t in state.done.values():
        if t.type == "InitialiseOptimiserTask":
            a = float(t.fixed_param_val)
            x = [a if n == "A_osc" else float(np.ravel(t.initial_bestfit[n])[0]) for n in em.NAMES]
            starts[(a, int(t.repetition_number))] = float(em.logp(x))
    runs = {}
    for t in tasks:
        key = (float(t.optimise_settings["fixed_param_val"]), int(t.optimise_settings["repetition_number"]))
        pos = t.optimiser.bestfit["position"]
        runs.setdefault(key, {})[int(t.optimise_settings["iteration_number"])] = (float(t.optimiser.bestfit["loglkl"]),
                                                                                 float(np.ravel(pos["omega"])[0]))
    out = {"steps_per_iteration": steps, "points": {}}
    print(f"\n{folder.name}: {steps} steps per iteration")
    print(f"{'A_osc':>6} {'rep':>3} {'iters':>5} {'omega':>6} {'gap@20':>8}   calls to within chi^2 of " + "  ".join(f"{s:>6}" for s in THRESHOLDS))
    for (a, rep), its in sorted(runs.items()):
        if a not in exact:
            continue
        n = max(its) + 1
        if sorted(its) != list(range(n)):
            print(f"{a:6.3f} {rep:3d}  iterations missing: {sorted(set(range(n)) - set(its))}")
            continue
        best = np.minimum.accumulate([its[i][0] for i in range(n)])
        omega = its[int(np.argmin([its[i][0] for i in range(n)]))][1]      # omega of the best point found
        gap = 2 * (best + exact[a])            # chi^2 above the exact profile (negative: better than the reference)
        need = {}
        for s in THRESHOLDS:
            hit = np.nonzero(gap <= s)[0]
            need[s] = int((hit[0] + 1) * per_iter) if hit.size else None
        at_end = float(gap[SCHEDULE_END - 1]) if n >= SCHEDULE_END else None
        print(f"{a:6.3f} {rep:3d} {n:5d} {omega:6.2f} {(f'{at_end:8.3f}' if at_end is not None else '       -')}   {'':>24}"
              + "  ".join(f"{(v if v is not None else '-'):>6}" for v in need.values()))
        out["points"][f"{a}_{rep}"] = dict(A_osc=a, repetition=rep, iterations=n, gap_chi2=gap.tolist(), gap_at_schedule_end=at_end, omega=omega,
                                           start_gap=2 * (exact[a] - starts[(a, rep)]) if (a, rep) in starts else None,
                                           calls_to=dict((str(s), v) for s, v in need.items()), exact_logl=exact[a])
    (folder / "convergence.json").write_text(json.dumps(out, indent=1))
