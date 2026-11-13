"""The exact profile of the PDP+ emulator in A_osc in the zone omega in [38.25, 60.5] (all other parameters free), at a
few points: the reference for PROSPECT's results (prospect/analyse_prospect.py).

    PYTHONPATH=pylib:. python exact_profile.py [A_osc ...]

Version 2 (8 November 2026). Version 1 ran L-BFGS-B in the parameters themselves, whose ranges differ by four orders of
magnitude; it stopped early, and PROSPECT's annealing found points 0.25 to 0.47 higher in chi^2 at A_osc = 0.01 and 0.02
(its log: prospect/exact_profile_v1_raw_coordinates.log). Now:
  - every search is in unit coordinates of the box (u = (x - lower) / (upper - lower));
  - starts: the 30 best samples of the zone's Metropolis-Hastings chains within 0.004 of A_osc, and the best points of
    every PROSPECT run in prospect/output_*/state.pkl at that A_osc;
  - L-BFGS-B with the network's own gradients (ftol 1e-15, gtol 1e-10), then from the best end point L-BFGS-B again and
    Powell (no gradients) until neither gains more than 1e-6 in ln L.
The reference is the best of all of these; the JSON also gives the best end point of each kind of start.
"""
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

import emulator as em

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here / "pylib"))                 # PROSPECT's classes, to read its state.pkl
VALUES = [float(v) for v in sys.argv[1:]] or [0.01, 0.02, 0.03, 0.04]
io, ia = em.NAMES.index("omega"), em.NAMES.index("A_osc")
box = em.BOX.copy()
box[io] = [38.25, 60.5]
lo, width = box[:, 0], box[:, 1] - box[:, 0]
names = open(here / "chains/zone1b/mcmc.1.txt").readline().split()[1:]
C = np.vstack([np.loadtxt(f) for f in sorted(glob.glob(str(here / "chains/zone1b/mcmc.[1-4].txt")))])
X = C[:, [names.index(n) for n in em.NAMES]]


def prospect_points(a):
    """The best point of every finished annealing iteration at A_osc = a, from all PROSPECT output folders."""
    pts = []
    for f in sorted((here / "prospect").glob("output_*/state.pkl")):
        state = pickle.loads(f.read_bytes())
        for t in state.done.values():
            if t.type == "OptimiseTask" and abs(float(t.optimise_settings["fixed_param_val"]) - a) < 1e-12:
                pos = t.optimiser.bestfit["position"]
                pts.append((float(t.optimiser.bestfit["loglkl"]), np.array([float(np.ravel(pos[n])[0]) if n in pos else a for n in em.NAMES])))
    pts.sort(key=lambda p: p[0])
    return [p[1] for p in pts[:10]]


@tf.function(input_signature=[tf.TensorSpec([None, len(em.NAMES)], tf.float64)])
def value_and_grad(x):
    with tf.GradientTape() as t:
        t.watch(x)
        v = em.MODEL(x, training=False)[:, 0]
    return v, t.gradient(v, x)


free = [k for k in range(len(em.NAMES)) if k != ia]


def search(x0, method):
    """Maximise over the free parameters from x0 (A_osc as in x0), in unit coordinates."""
    def to_x(u):
        x = x0.copy()
        x[free] = lo[free] + u * width[free]
        return x

    def f(u):
        v, g = value_and_grad(tf.constant(to_x(u)[None]))
        return -float(v[0]), -g.numpy()[0][free] * width[free]
    u0 = np.clip((x0[free] - lo[free]) / width[free], 0, 1)
    if method == "lbfgsb":
        r = minimize(f, u0, jac=True, method="L-BFGS-B", bounds=[(0, 1)] * len(free),
                     options=dict(maxiter=5000, maxfun=20000, ftol=1e-15, gtol=1e-10))
    else:
        r = minimize(lambda u: f(u)[0], u0, method="Powell", bounds=[(0, 1)] * len(free),
                     options=dict(maxiter=200000, xtol=1e-10, ftol=1e-14))
    return -r.fun, to_x(r.x)


out = {}
for a in VALUES:
    near = X[np.abs(X[:, ia] - a) < 0.004]
    vals = em.logp_many(near)
    starts = {"mcmc": near[np.argsort(vals)[::-1][:30]], "prospect": prospect_points(a)}
    best, kinds = (-np.inf, None), {}
    for kind, xs in starts.items():
        for s in xs:
            x0 = s.copy()
            x0[ia] = a
            v, x = search(x0, "lbfgsb")
            if v > kinds.get(kind, (-np.inf,))[0]:
                kinds[kind] = (v, x)
            if v > best[0]:
                best = (v, x)
    rounds = 0
    while rounds < 20:
        rounds += 1
        gain = 0.0
        for method in ("lbfgsb", "powell"):
            v, x = search(best[1], method)
            if v > best[0]:
                gain += v - best[0]
                best = (v, x)
        if gain < 1e-6:
            break
    out[a] = dict(logl=best[0], x=dict(zip(em.NAMES, map(float, best[1]))), polish_rounds=rounds,
                  best_by_start={k: float(v[0]) for k, v in kinds.items()}, starts={k: len(v) for k, v in starts.items()})
    print(f"A_osc {a}: exact profile {best[0]:.4f} at omega {best[1][io]:.3f} (best L-BFGS-B end point from MCMC starts "
          f"{kinds['mcmc'][0]:.4f}, from {len(starts['prospect'])} PROSPECT points "
          f"{kinds['prospect'][0] if 'prospect' in kinds else float('nan'):.4f}; {rounds} polishing rounds)", flush=True)
(here / "prospect" / "exact_profile_pp_zone1b.json").write_text(json.dumps(out, indent=1))
