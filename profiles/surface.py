"""The legacy scan of one two-coordinate surface (`surrogate_optimizer: legacy`), with neighbour
continuation and saved passes. The true stage of a surface is profiles/hybrid_surface.py."""

import json
from pathlib import Path
import numpy as np
from profiles.io import save
from profiles.surrogate import surrogate_optimizer


def scan_surface(model, bounds, data, indices, axes, config, checkpoint=None):
    optimize = surrogate_optimizer(model, bounds, seed=config.seed, gtol=config.gtol)
    starts = [
        *data[np.argsort(data[:, -1])[-min(8, len(data)) :], :-1],
        bounds.mean(axis=1),
    ]
    av, ax, ad = optimize(None, None, starts)
    shape = tuple(len(a) for a in axes)
    values = np.full(shape, -np.inf)
    xs = np.zeros(shape + (len(bounds),))
    ds = [[None for _ in axes[1]] for _ in axes[0]]
    saved = Path(checkpoint) if checkpoint else None
    if saved and saved.exists():
        old = json.loads(saved.read_text())
        values, xs, ds = (
            np.array(old["logl"]),
            np.array(old["xopt"]),
            old["diagnostics"],
        )
        av, ax, ad = (
            old["best_logl"],
            np.array(old["best_x"]),
            old["anchor_diagnostics"],
        )
        if old["quality"]["converged"]:
            return old
    stable = False
    for step in range(config.max_passes):
        before = values.copy()
        # Alternate sweep direction; adjacent optima and the unrestricted anchor
        # compete at every node. There is no assumption of a single ridge branch.
        order = [
            (i, j)
            for i in range(shape[0])
            for j in (range(shape[1]) if i % 2 == 0 else range(shape[1] - 1, -1, -1))
        ]
        if step % 2:
            order.reverse()
        for i, j in order:
            candidates = [ax]
            for k, l in [(i, j), (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                if (
                    0 <= k < shape[0]
                    and 0 <= l < shape[1]
                    and np.isfinite(values[k, l])
                ):
                    candidates.append(xs[k, l])
            v, x, d = optimize(indices, np.array([axes[0][i], axes[1][j]]), candidates)
            if v > values[i, j] or (
                v == values[i, j]
                and d["projected_gradient_inf"] < ds[i][j]["projected_gradient_inf"]
            ):
                values[i, j], xs[i, j], ds[i][j] = v, x, d
        best = np.unravel_index(np.argmax(values), shape)
        if values[best] > av + 1e-7:
            v, x, d = optimize(None, None, [xs[best], ax])
            if v >= av:
                av, ax, ad = v, x, d
        change = float(np.max(values - before))
        stable = bool(change <= 1e-4)
        consistent = bool(np.max(values) <= av + 1e-4)
        stationary = bool(
            ad["stationary"] and all(d["stationary"] for row in ds for d in row)
        )
        result = dict(
            axes=axes,
            indices=indices,
            logl=values,
            xopt=xs,
            diagnostics=ds,
            best_logl=av,
            best_x=ax,
            anchor_diagnostics=ad,
            quality=dict(
                passes=step + 1,
                stable=stable,
                stationary=stationary,
                anchor_consistent=consistent,
                converged=stable and consistent and stationary,
                max_change=change if np.isfinite(change) else None,
                global_optimum_certified=False,
            ),
        )
        if saved:
            save(saved, result)
        print(
            f"2D pass {step+1}: stationary={stationary}, stable={stable}, anchor={consistent}",
            flush=True,
        )
        if result["quality"]["converged"]:
            return result
    raise RuntimeError(
        "2D surrogate did not pass stationarity/stability checks; saved diagnostics retained"
    )
