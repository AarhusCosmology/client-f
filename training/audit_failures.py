"""Bounded recovery and explicitly synthetic audit training constraints."""
from pathlib import Path

import numpy as np


def point_key(x):
    return np.asarray(x, dtype=np.float32).tobytes()


def nearby_points(x, lower, upper, count, radius, rng):
    from training.audit import inside_float32

    width = upper - lower
    u = (np.asarray(x) - lower) / width
    points, seen = [], {point_key(x)}
    for _ in range(64):
        # Reflect at the domain boundary, rather than clipping every proposal to
        # the same corner. Each coordinate moves by at most radius box widths.
        proposal = u + rng.uniform(-radius, radius, len(x))
        proposal = np.where(proposal < 0, -proposal, proposal)
        proposal = np.where(proposal > 1, 2 - proposal, proposal)
        p = inside_float32(lower + proposal * width, lower, upper)
        key = point_key(p)
        if key not in seen:
            seen.add(key)
            points.append(p)
        if len(points) == count:
            break
    return np.asarray(points).reshape(-1, len(x))


def evaluate_with_recovery(points, evaluate, config, best, lower, upper, rng, budget):
    """One primary batch and at most one probe batch; never recursively probe failures.

    Return all actual evaluations, synthetic constraints, and per-primary diagnostics.
    Insufficient budget/representable probes never justifies a penalty.
    """
    x, y, d = evaluate(points)
    x, y = np.asarray(x, float), np.asarray(y, float).ravel()
    primary_y = y.copy()
    penalties, failures, groups, probes = [], [], [], []
    remaining = budget - len(y)
    for i in np.flatnonzero(~np.isfinite(y)):
        failure = dict(x=x[i].tolist(), outcome="skipped", probes=[], probe_values=[])
        failures.append(failure)
        if config.failure_policy != "penalize":
            continue
        if remaining < config.failure_probes:
            failure["outcome"] = "insufficient_budget"
            continue
        neighbours = nearby_points(x[i], lower, upper, config.failure_probes,
                                   config.failure_radius, rng)
        if len(neighbours) != config.failure_probes:
            failure["outcome"] = "insufficient_distinct_probes"
            continue
        groups.append((failure, len(probes), len(neighbours)))
        probes.extend(neighbours)
        remaining -= len(neighbours)
    if probes:
        px, py, pd = evaluate(np.asarray(probes))
        px, py = np.asarray(px, float), np.asarray(py, float).ravel()
        finite_values = np.concatenate([y[np.isfinite(y)], py[np.isfinite(py)]])
        best = max(best, max(finite_values, default=best))
        for failure, start, count in groups:
            vals = py[start:start+count]
            failure.update(probes=px[start:start+count].tolist(),
                           probe_values=[float(v) if np.isfinite(v) else None for v in vals])
            if np.isfinite(vals).any():
                failure["outcome"] = "finite_neighbour"
            else:
                target = min(np.float32(best - config.failure_penalty),
                             np.nextafter(np.float32(best), np.float32(-np.inf)))
                if not np.isfinite(target):
                    raise ValueError("audit penalty cannot be represented as a finite float32 target")
                failure["outcome"] = "penalized"
                failure["penalty_loglkl"] = float(target)
                penalties.append(dict(x=failure["x"], penalty_loglkl=float(target)))
        if d is not None or pd is not None:
            columns = np.asarray(d if d is not None else pd).shape[1]
            d = np.concatenate([np.full((len(y), columns), np.nan) if d is None else d,
                                np.full((len(py), columns), np.nan) if pd is None else pd])
        x, y = np.concatenate([x, px]), np.concatenate([y, py])
    return x, y, d, penalties, failures, primary_y


def active_penalties(penalties, true_inputs, lower, upper):
    """An actual finite observation supersedes a synthetic constraint at that point."""
    known = {point_key(x) for x in true_inputs}
    active = {}
    for p in penalties:
        x, value = np.asarray(p["x"], float), float(p["penalty_loglkl"])
        if (x.shape != np.shape(lower) or not np.isfinite(x).all() or not np.isfinite(value)
                or np.any(x < lower) or np.any(x > upper)):
            raise ValueError("invalid persisted audit penalty")
        if point_key(x) not in known:
            active[point_key(x)] = dict(x=x.tolist(), penalty_loglkl=value)
    return list(active.values())


def load_penalties(directory, before_iteration, names):
    """Load separate constraints from earlier audits; never read them as true data."""
    import pandas as pd

    penalties = []
    paths = sorted(Path(directory).glob("audit_penalties_it_*.csv"),
                   key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    for path in paths:
        if int(path.stem.rsplit("_", 1)[1]) >= before_iteration:
            continue
        frame = pd.read_csv(path)
        if list(frame.columns) != [*names, "penalty_loglkl"]:
            raise ValueError(f"{path}: audit penalty columns do not match parameters")
        penalties.extend(dict(x=x.tolist(), penalty_loglkl=float(y))
                         for x, y in zip(frame[list(names)].to_numpy(float), frame.penalty_loglkl))
    return penalties


def penalty_training_rows(penalties, n_real, ndim, fraction):
    """Repeat constraints within fine-tuning only, so sparse failures retain influence.

    Preserve every unique constraint at least once, and distribute repetitions evenly.
    The loss is still calibrated on genuine data; no duplicate is saved as a true call.
    """
    x = np.asarray([p["x"] for p in penalties], np.float32).reshape(-1, ndim)
    y = np.asarray([p["penalty_loglkl"] for p in penalties], np.float32)
    if len(x):
        rows = max(len(x), int(np.ceil(fraction * n_real)))
        indices = np.arange(rows) % len(x)
        x, y = x[indices], y[indices]
    return x, y
