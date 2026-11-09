"""Profile likelihoods with the true likelihood alone, node by node along chains (27 October 2026).

    python smoke/profile_chain.py SPEC NAME [--workers 48] [--chains LABEL ...] [--node-budget 400] [--stall 4]
                                  [--stall-gain 0.02] [--stop-above DCHI2] [--resume LOG]

Made for the World Cup profiles of the SIDR + DRMD paper: there the refinement of the emulator's profiles with the true
likelihood ended mostly at its limit of 156 calls per node, because the emulator's optima lay far from the true ones,
and four passes of moving known points to the nodes left several profiles jagged by 2-3 in chi^2. Here the emulator
plays no part. SPEC (JSON) gives what smoke/minimise_true.py's does (cobaya, job_env, fixed, nuisance) and a covariance
of every slow parameter, the profiled ones included ("covariance": {names, matrix}), and the chains:
[{label, profile, seed: {params}, nodes: [coordinates, in the order they are done]}].

A chain does its nodes one after the other. At a node the profiled parameter is fixed at the node's value and the
log-posterior (priors included) is maximised over the other slow parameters as minimise_true.py does: the nuisance
parameters optimised at every point, L-BFGS-B in whitened coordinates (here whitened by the covariance conditional on the
profiled parameter), central-difference gradients evaluated in parallel, restarts from the best point while a run gains
RESTART_GAIN. It starts from the previous node's optimum (the seed for the first node) with x_p set to the node's value,
either as it is or moved along the posterior's degeneracy (every other slow parameter shifted by Cov(x_n, x_p) / Var(x_p)
times the step in x_p, clipped into the prior box), whichever of the two points is better. The shift is what a chain
along a degeneracy needs (ln(10^10 A_s) on the World Cup data: 0.26 above the best fit one node away, where moving x_p
alone had left the polish passes at 2.5-3), but the covariance is the best fit's mode's, and on another branch it can be
far off (the second branch's f_idm, jobs of 27 October 2026: log10 z_stop moved from 5 to 4, 470 above in chi^2). A node ends when its best log-posterior has gained less than --stall-gain over its last --stall gradients
(the likelihood's noise keeps L-BFGS-B from its own tests, and a failed line search costs several gradients: in the
pilot on ln(10^10 A_s), job 6969833, every node used its 414 evaluations, and this rule would have ended them after
5-9 gradients, at most 0.08 in chi^2 above the full node), and once it has used --node-budget evaluations. With
--stop-above a chain ends after a node whose -2 ln posterior lies more than DCHI2 above the SPEC's "reference_logpost"
(the best fit, in the Cobaya file's convention): the tails beyond the intervals are not needed. Chains run in threads on
one pool: two chains of 11 free slow parameters take 46 of 48 workers per round.

Output logs/profile_chain_<NAME>_<job>.json, rewritten after every gradient: for every chain and node the best
evaluation (all sampled parameters, the log-posterior, chi^2 per likelihood), the log-posterior at the centre of every
gradient, the number of evaluations and how the node ended. --resume LOG continues each chain after the last node LOG
records as ended (the nodes must be the SPEC's, in the same order).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chi2_points_mp as cp  # noqa: E402
import minimise_true as mt  # noqa: E402


class NodeBudget(Exception):
    """The node has used its evaluations."""


class NodeStalled(Exception):
    """The node's best value has stopped improving."""


def chain_geometry(names, matrix, profile):
    """(free slow parameters, shift per unit of the profiled one, Cholesky factor of the conditional covariance)."""
    C = np.array(matrix, float)
    i = names.index(profile)
    s = [k for k in range(len(names)) if k != i]
    beta = C[s, i] / C[i, i]
    cond = C[np.ix_(s, s)] - np.outer(C[s, i], C[i, s]) / C[i, i]
    return [names[k] for k in s], beta, np.linalg.cholesky(cond)


def resumed_nodes(chain, previous):
    """The nodes of `chain` that an earlier log records as ended, checked against the chain's coordinates."""
    nodes = [n for n in previous.get("nodes", []) if n.get("ended")]
    for k, n in enumerate(nodes):
        if k >= len(chain["nodes"]) or abs(n["coordinate"] - chain["nodes"][k]) > 1e-9:
            raise SystemExit(f"--resume: node {k} of {chain['label']} is at {n['coordinate']} in the log and "
                             f"{chain['nodes'][k] if k < len(chain['nodes']) else 'nothing'} in the spec")
    return nodes


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("spec", type=Path)
    p.add_argument("name")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--maxiter", type=int, default=30, help="iterations per L-BFGS-B run")
    p.add_argument("--node-budget", type=int, default=400, help="evaluations per node")
    p.add_argument("--stall", type=int, default=4, help="gradients over which a node must gain --stall-gain (0: never)")
    p.add_argument("--stall-gain", type=float, default=0.02, help="in ln posterior")
    p.add_argument("--stop-above", type=float, help="end a chain after a node this far above the reference, in chi^2")
    p.add_argument("--chains", nargs="+", help="only these chains (labels) of SPEC")
    p.add_argument("--resume", type=Path, help="a log of an earlier job of the same SPEC")
    a = p.parse_args()
    spec = json.loads(a.spec.read_text())
    chains = spec["chains"]
    if a.chains:
        missing = set(a.chains) - {c["label"] for c in chains}
        if missing:
            raise SystemExit(f"no chains {sorted(missing)} in {a.spec}")
        chains = [c for c in chains if c["label"] in a.chains]
    if "classy" in (HERE / spec["cobaya"]).read_text():
        import classy
        sys.path.insert(0, str(HERE))
        import client_paths  # noqa: E402,F401  CLiENT's hpc/ tools
        from boltzmann_build import location_problem
        problem = location_problem("classy", classy.__file__)
        if problem:
            raise SystemExit(f"FATAL: {problem} (source {spec['job_env']} first)")
    fixed, nuisance = spec["fixed"], spec["nuisance"]
    if a.stop_above is not None and "reference_logpost" not in spec:
        raise SystemExit(f"--stop-above needs reference_logpost in {a.spec}")
    names, matrix = spec["covariance"]["names"], spec["covariance"]["matrix"]
    from cobaya.yaml import yaml_load_file
    prior = yaml_load_file(str(HERE / spec["cobaya"]))["params"]
    box = {n: (float(prior[n]["prior"]["min"]), float(prior[n]["prior"]["max"])) for n in names}
    previous = json.loads(a.resume.read_text())["chains"] if a.resume else {}
    ended_before = {c["label"]: resumed_nodes(c, previous.get(c["label"], {})) for c in chains}
    out = HERE / "logs" / f"profile_chain_{a.name}_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    out.parent.mkdir(exist_ok=True)
    log = dict(spec=str(a.spec), name=a.name, cobaya=spec["cobaya"], fixed=fixed, nuisance=nuisance, step=mt.STEP,
               node_budget=a.node_budget, stall=a.stall, stall_gain=a.stall_gain, stop_above=a.stop_above, resumed=str(a.resume) if a.resume else None, chains={})
    lock = threading.Lock()

    def save():
        with lock:
            out.write_text(json.dumps(log, indent=1))

    print(f"== {a.name}: {len(chains)} chains ({', '.join(c['label'] for c in chains)}), {len(nuisance)} nuisance, "
          f"fixed {fixed}; {a.workers} workers, {a.node_budget} evaluations per node", flush=True)
    from scipy.optimize import minimize
    with mp.get_context("spawn").Pool(a.workers, initializer=cp.start, initargs=(spec["cobaya"], None)) as pool:

        def run(chain):
            try:
                follow(chain)
            except Exception:                       # noqa: BLE001 - the other chains go on; the log keeps this one's nodes
                with lock:
                    log["chains"].setdefault(chain["label"], {})["error"] = traceback.format_exc()
                save()
                print(f"== {chain['label']}: stopped by an error\n{traceback.format_exc()}", flush=True)

        def follow(chain):
            label, prof = chain["label"], chain["profile"]
            slow, beta, L = chain_geometry(names, matrix, prof)
            lo, hi = np.array([box[n][0] for n in slow]), np.array([box[n][1] for n in slow])
            ended = ended_before[label]
            entry = log["chains"].setdefault(label, dict(profile=prof, seed_note=chain.get("note", ""), nodes=list(ended)))
            found = [n for n in ended if n["best"] is not None]
            prev = dict(found[-1]["best"]["params"]) if found else dict(chain["seed"])
            for c in chain["nodes"][len(ended):]:
                base = dict(prev, **fixed)
                x_plain = np.clip(np.array([float(base[n]) for n in slow]), lo, hi)
                x_shift = np.clip(x_plain + beta * (c - float(base[prof])), lo, hi)
                base[prof] = float(c)
                node = dict(coordinate=float(c), start=None, centres=[], bests=[], evaluations=0, best=None, runs=[])
                with lock:
                    entry["nodes"].append(node)
                state = dict(nuis={n: base[n] for n in nuisance})
                tries = pool.map(mt.evaluate, [(dict(base, **dict(zip(slow, map(float, x)))), nuisance) for x in (x_shift, x_plain)],
                                 chunksize=1)
                shifted = tries[0]["logpost"] >= tries[1]["logpost"] or not np.isfinite(tries[1]["logpost"])
                x_start = (x_shift if shifted else x_plain).copy()
                with lock:
                    node["evaluations"] += 2
                    node["start"] = dict(chosen="shifted" if shifted else "unshifted",
                                         shifted=tries[0]["logpost"] if np.isfinite(tries[0]["logpost"]) else None,
                                         unshifted=tries[1]["logpost"] if np.isfinite(tries[1]["logpost"]) else None)
                    for r in tries:
                        if np.isfinite(r["logpost"]) and (node["best"] is None or r["logpost"] > node["best"]["logpost"]):
                            node["best"] = r
                    if node["best"] is not None:
                        state["nuis"] = {n: node["best"]["params"][n] for n in nuisance}

                def to_params(z):
                    x = np.clip(x_start + L @ z, lo, hi)
                    return dict(base, **state["nuis"], **dict(zip(slow, map(float, x))))

                def fun_and_grad(z):
                    if node["evaluations"] >= a.node_budget:
                        raise NodeBudget
                    e = np.eye(len(slow))
                    tasks = ([to_params(z)] + [to_params(z + mt.STEP * e[i]) for i in range(len(slow))]
                             + [to_params(z - mt.STEP * e[i]) for i in range(len(slow))])
                    res = pool.map(mt.evaluate, [(t, nuisance) for t in tasks], chunksize=1)
                    with lock:
                        node["evaluations"] += len(res)
                        node["centres"].append(res[0]["logpost"] if np.isfinite(res[0]["logpost"]) else None)
                        for r in res:
                            if np.isfinite(r["logpost"]) and (node["best"] is None or r["logpost"] > node["best"]["logpost"]):
                                node["best"] = r
                        if node["best"] is not None:
                            state["nuis"] = {n: node["best"]["params"][n] for n in nuisance}
                        node["bests"].append(node["best"]["logpost"] if node["best"] is not None else None)
                        h = node["bests"]
                        stalled = (a.stall > 0 and len(h) > a.stall and h[-1] is not None and h[-1 - a.stall] is not None
                                   and h[-1] - h[-1 - a.stall] < a.stall_gain)
                    save()
                    if stalled:
                        raise NodeStalled
                    f0 = -res[0]["logpost"]
                    if not np.isfinite(f0):
                        return 1e10, np.zeros(len(slow))
                    n = len(slow)
                    fp = np.array([-r["logpost"] for r in res[1:n + 1]])
                    fm = np.array([-r["logpost"] for r in res[n + 1:]])
                    g = np.where(np.isfinite(fp) & np.isfinite(fm), (fp - fm) / (2 * mt.STEP),
                                 np.where(np.isfinite(fp), (fp - f0) / mt.STEP, np.where(np.isfinite(fm), (f0 - fm) / mt.STEP, 0.0)))
                    return f0, g

                # restarts from the best point, as minimise_true.py; the best of a node may be a gradient's side point
                for k in range(mt.MAX_RUNS):
                    before = node["best"]["logpost"] if node["best"] else -np.inf
                    if node["best"]:
                        x_start[:] = np.clip([node["best"]["params"][n] for n in slow], lo, hi)
                    try:
                        r = minimize(fun_and_grad, np.zeros(len(slow)), jac=True, method="L-BFGS-B",
                                     options=dict(maxiter=a.maxiter, ftol=1e-8, gtol=1e-3, maxls=8))
                        node["runs"].append(dict(message=str(r.message), iterations=int(r.nit)))
                    except NodeBudget:
                        node["runs"].append(dict(message="node budget", iterations=None))
                        break
                    except NodeStalled:
                        node["runs"].append(dict(message="stalled", iterations=None))
                        break
                    if node["best"] is None or node["best"]["logpost"] - before < mt.RESTART_GAIN:
                        break
                with lock:
                    node["ended"] = node["runs"][-1]["message"] if node["best"] is not None else "no finite value"
                save()
                b = node["best"]
                if b is None:
                    print(f"== {label}: {prof} = {c:.6g}: no finite value in {node['evaluations']} evaluations", flush=True)
                    continue
                prev = dict(b["params"])
                print(f"== {label}: {prof} = {c:.6g}: -2 ln posterior {-2 * b['logpost']:.3f} after {node['evaluations']} "
                      f"evaluations ({node['ended']})", flush=True)
                if a.stop_above is not None and 2 * (spec["reference_logpost"] - b["logpost"]) > a.stop_above:
                    print(f"== {label}: ends, {2 * (spec['reference_logpost'] - b['logpost']):.1f} above the reference", flush=True)
                    with lock:
                        entry["stopped_above"] = float(c)
                    save()
                    break

        threads = [threading.Thread(target=run, args=(c,)) for c in chains]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    print(f"written: {out}")


if __name__ == "__main__":
    main()
