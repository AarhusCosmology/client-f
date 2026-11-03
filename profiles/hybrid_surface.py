"""Joint two-coordinate hybrid scans and checkpointed residual-Newton refinement.

Surrogate nodes exchange starts with four grid neighbours. True refinement uses
independent rows: each row starts nearest the surrogate anchor and continues in
both directions along its second coordinate (with `chain_length`, from further
seeds too). Only a chain's predecessor supplies residual state; scheduling and
worker count cannot change that state.

With `continuation` (opt-in) a row's seed is its node nearest a given value of the
second coordinate, started from the best training point (or the scan's optimum
there), and every other node starts from its predecessor's true optimum instead of
the scan's: the network's optima are not used beyond the seeds.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from profiles.io import save


def validate_grid(bounds, indices, axes):
    bounds = np.asarray(bounds, float)
    ids = np.asarray(indices)
    if (ids.shape != (2,) or ids.dtype.kind not in "iu" or len(set(ids)) != 2
            or np.any(ids < 0) or np.any(ids >= len(bounds))):
        raise ValueError("a surface requires two distinct parameter indices")
    if len(axes) != 2:
        raise ValueError("a surface requires two axes")
    axes = [np.asarray(a, float) for a in axes]
    for k, a in zip(ids, axes):
        if (a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all()
                or np.any(np.diff(a) <= 0) or np.any(a < bounds[k, 0])
                or np.any(a > bounds[k, 1])):
            raise ValueError("surface axes must increase strictly within physical bounds")
    return ids.tolist(), axes


def scan_surface_hybrid(model, bounds, data, indices, axes, config, checkpoint=None,
                        *, surrogate=None, log=print):
    from profiles.hybrid import scan_hybrid, hybrid_settings
    indices, axes = validate_grid(bounds, indices, axes)
    names = [f"p{i}" for i in range(len(bounds))]
    md = SimpleNamespace(param_names=names, bounds=dict(zip(names, bounds)))
    result, _ = scan_hybrid(
        model, md, data, [names[i] for i in indices], axes, config,
        hybrid=hybrid_settings(config),
        surrogate=surrogate, surface_indices=indices, log=log)
    if checkpoint is not None:
        save(checkpoint, result)
    return result


def refine_surface_newton(model, source, bounds, ledger, out, config, *, surrogate=None, log=print):
    from profiles.hybrid import BatchedSurrogate, ConcurrentLedger, NodeLedger
    from profiles.residual_newton import NewtonConfig, NewtonRefiner, chain_plan, refused_start, run_seeded_chains

    ids, axes = validate_grid(bounds, source["indices"], source["axes"])
    shape = (len(axes[0]), len(axes[1]))
    starts = np.asarray(source["xopt"], float)
    if starts.shape != (*shape, len(bounds)) or not np.isfinite(starts).all():
        raise ValueError("invalid surface starting points")
    cfg = NewtonConfig(tol=config.corrected_tolerance(), maxcalls=config.maxfun)
    sb = surrogate if surrogate is not None else BatchedSurrogate(model, bounds)
    directory = Path(out) / "true_surface"
    directory.mkdir(parents=True, exist_ok=True)
    # Independent of worker count; guards accidental reuse of low-level checkpoints.
    contract = dict(method="joint residual Newton rows v1", indices=ids,
                    axes=[a.tolist() for a in axes], bounds=np.asarray(bounds).tolist(),
                    configuration=asdict(cfg), starts=starts.tolist(), best_x=list(source["best_x"]))
    chain_length = getattr(config, "chain_length", None)
    if chain_length is not None:  # absent by default, so earlier surfaces keep their signature
        contract["chain_length"] = int(chain_length)
    continuation = getattr(config, "continuation", None)
    if continuation is not None:
        from profiles.config import check_continuation

        seed_value, start_kind = check_continuation(continuation)
        if chain_length is not None:
            raise ValueError("continuation requires chain_length: null (one seed a row)")
        if not axes[1][0] <= seed_value <= axes[1][-1]:
            raise ValueError(f"continuation seed {seed_value:g} outside the second axis "
                             f"[{axes[1][0]:g}, {axes[1][-1]:g}]")
        seed_start = None
        if start_kind == "best":
            seed_start = np.asarray(source.get("seed_start", []), float)
            if seed_start.shape != (len(bounds),) or not np.isfinite(seed_start).all():
                raise ValueError("continuation from the best training point needs the point (source['seed_start'])")
        # Absent without the setting, so earlier surfaces keep their signature.
        contract["continuation"] = dict(seed=seed_value, start=start_kind,
                                        **({} if seed_start is None else dict(seed_start=seed_start.tolist())))
        j0 = int(np.argmin(abs(axes[1] - seed_value)))
    else:
        j0 = int(np.argmin(abs(axes[1] - source["best_x"][ids[1]])))
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    identity = directory / "method.json"
    if identity.exists():
        if json.loads(identity.read_text())["signature"] != signature:
            raise ValueError("surface checkpoint inputs or method changed")
    elif any(directory.glob("node_*.json")) or (directory / "anchor.json").exists():
        raise ValueError("surface checkpoints without a method record cannot be reused")
    else:
        save(identity, dict(signature=signature, **contract))

    def checkpoint(path, compute):
        if not path.exists():
            save(path, compute())
        return json.loads(path.read_text())

    shared = ConcurrentLedger(ledger, workers=config.workers)
    rows = {}
    try:
        def run(i, order, state):
            refiner = NewtonRefiner(sb, NodeLedger(shared, "corrected/node/surface"), cfg, log)
            records = {}
            previous = None
            if continuation is not None and order and order[0] != j0:
                # A chain's first node starts from the true optimum of the node before it,
                # toward the seed: done, since a chain starts when its seed is
                # (run_seeded_chains), and read from its checkpoint.
                k = order[0] + (1 if order[0] < j0 else -1)
                neighbour = json.loads((directory / f"node_{i:04d}_{k:04d}.json").read_text())["x"]
                previous = None if neighbour is None else np.asarray(neighbour, float)
            for j in order:
                values = np.array([axes[0][i], axes[1][j]])
                if continuation is None:
                    x_s = starts[i, j].copy()
                elif previous is not None:
                    x_s = previous.copy()
                else:
                    x_s = (starts[i, j] if seed_start is None else seed_start).copy()
                x_s[ids] = values
                # In-box start (a scan can end a few ulps outside a bound, which the ledger
                # would reject); clipped here, so the checkpoint signature is unchanged.
                x_s = np.clip(x_s, np.asarray(bounds)[:, 0], np.asarray(bounds)[:, 1])
                prior = state

                def compute():
                    refiner.ledger.stage = f"corrected/direct/surface/{i}/{j}"
                    direct = float(refiner.ledger(x_s))
                    if not np.isfinite(direct):
                        log(f"  surface node ({i}, {j}): the likelihood refuses its start "
                            f"({'the surrogate optimum' if continuation is None else 'the continued start'}, "
                            f"x={x_s.tolist()}); no value there", flush=True)
                        return refused_start(x_s, prior, fixed_values=values)
                    v, x, diag, successor = refiner.solve(
                        sb.box(x_s), ids, values, stage=f"corrected/node/surface/{i}/{j}",
                        prior=prior, t_start=direct)
                    return dict(logl=v, x=x, diagnostics=diag, state=successor,
                                direct_logl=direct, fixed_values=values)

                record = checkpoint(directory / f"node_{i:04d}_{j:04d}.json", compute)
                state = record["state"]
                if record["x"] is not None:           # a refused node hands on what it was given
                    previous = np.asarray(record["x"], float)
                records[i, j] = record
            return records

        # Each row: a seed at j0 and chains outward from it (with a chain_length, further
        # seeds along the way); a chain starts when its own seed is done.
        seeds, chains_of = [], {}
        for i in range(shape[0]):
            seeds.append((i, j0))
            for side in (list(range(j0-1, -1, -1)), list(range(j0+1, shape[1]))):
                own, further = chain_plan(side, chain_length)
                if own:
                    chains_of.setdefault((i, j0), []).append(own)
                for seed, chain in further:
                    seeds.append((i, seed))
                    if chain:
                        chains_of[(i, seed)] = [chain]

        def run_row(i, order, state):
            return i, {j: record for (_, j), record in run(i, order, state).items()}

        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            rows.update(run_seeded_chains(pool, seeds, chains_of, run_row))
        # Tie breaking is lexicographic, never completion-order dependent.
        valued = [rows[k] for k in sorted(rows) if rows[k]["logl"] is not None]
        if not valued:
            raise RuntimeError("the likelihood refuses the start of every node of the surface: inspect the backend")
        best = max(valued, key=lambda q: q["logl"])
        final = NewtonRefiner(sb, NodeLedger(shared, "corrected/anchor"), cfg, log)

        def solve_anchor():
            path = best["state"]["path"][-1:]
            prior = dict(best["state"], path=[(0.0, path[0][1])] if path else [])
            v, x, diag, state = final.solve(sb.box(np.array(best["x"])),
                stage="corrected/anchor", prior=prior, t_start=best["logl"])
            return dict(logl=v, x=x, diagnostics=diag, state=state)

        anchor = checkpoint(directory / "anchor.json", solve_anchor)
    finally:
        shared.close()
    pick = lambda key: [[rows[i, j][key] for j in range(shape[1])] for i in range(shape[0])]
    result = dict(axes=axes, indices=ids, names=source["names"],
        full_parameter_names=source.get("full_parameter_names"), logl=pick("logl"),
        xopt=pick("x"), direct_logl=pick("direct_logl"), diagnostics=pick("diagnostics"),
        best_logl=anchor["logl"], best_x=anchor["x"], anchor_diagnostics=anchor["diagnostics"],
        anchor_consistent=bool(max(q["logl"] for q in valued) <= anchor["logl"]+1e-6),
        method="joint surrogate-corrected residual Newton", configuration=asdict(cfg),
        workers=config.workers, objective="native", accuracy_certified=False,
        global_optimum_certified=False)
    if continuation is not None:
        result["continuation"] = contract["continuation"]
    save(Path(out) / "true_surface.json", result)
    return result
