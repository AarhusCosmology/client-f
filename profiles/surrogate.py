"""The network's optimum at one profile node, for the legacy scan and the designs of
`retrain: true` (profiles/workflow.py, profiles/surface_workflow.py), and the display grids of a study.

The optimiser is bounded L-BFGS-B in prior-box coordinates with explicit numerical-quality
diagnostics; they establish local stationarity, not globality. A nonstationary solution is
polished without relaxing gtol: large additive native-objective constants make relative
function stopping poorly suited to stationarity checks, so the polish uses a locally shifted
objective and ftol=0, and accepts only stationary solutions with <=1e-7 log-objective value
loss. No true likelihood calls occur here. The default profile scan is the hybrid optimiser's
(profiles/hybrid.py); `scan` here is `surrogate_optimizer: legacy`, with continuation passes.
"""

import numpy as np
from scipy.optimize import minimize


def _bounded_optimizer(model, bounds, *, precision='float64', **options):
    """Build the scaled objective used by profile studies (TensorFlow imported lazily)."""
    import tensorflow as tf
    from model.network import copy_model_dtype
    if precision == 'float64' and model.compute_dtype != 'float64':
        model = copy_model_dtype(model)
    dtype = getattr(tf, precision)
    lo, hi = np.asarray(bounds, float).T
    centre, half = (lo+hi)/2, (hi-lo)/2

    @tf.function(input_signature=[tf.TensorSpec([1,len(lo)],dtype)])
    def value_grad(point):
        with tf.GradientTape() as tape:
            tape.watch(point)
            value = model(point, training=False)
        return value, tape.gradient(value, point)

    def objective(u, i_fixed, v_fixed):
        mask = np.ones(len(lo), bool)
        x = np.empty(len(lo), dtype=precision)
        if i_fixed is not None:
            mask[i_fixed], x[i_fixed] = False, v_fixed
        x[mask] = centre[mask]+half[mask]*u
        value, gradient = value_grad(tf.constant(x[None,:],dtype=dtype))
        return -float(value.numpy()[0,0]), -gradient.numpy()[0,mask].astype(float)*half[mask]
    return ProfileOptimizer(objective, lo, hi, **options)


class ProfileOptimizer:
    def __init__(self, objective, lo, hi, *, seed=0, gtol=1e-4, ftol=1e-10,
                 maxiter=20000, maxfun=40000, retry_starts=2):
        self.objective = objective
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        if (self.lo.ndim != 1 or self.lo.shape != self.hi.shape or not len(self.lo)
                or not np.isfinite(self.lo).all() or not np.isfinite(self.hi).all()
                or np.any(self.hi <= self.lo)):
            raise ValueError("finite, strictly ordered prior bounds are required")
        if (not np.isfinite([gtol, ftol]).all() or gtol <= 0 or ftol < 0
                or maxiter < 1 or maxfun < 1 or retry_starts < 0):
            raise ValueError("invalid optimizer tolerances or budgets")
        self.centre, self.half = (self.lo + self.hi)/2, (self.hi - self.lo)/2
        self.options = dict(gtol=gtol, ftol=ftol, maxiter=maxiter, maxfun=maxfun)
        self.retry_starts, self.rng = retry_starts, np.random.default_rng(seed)

    def __call__(self, i_fixed, v_fixed, starts):
        mask = np.ones(len(self.lo), bool)
        if i_fixed is not None:
            if (not np.isfinite(v_fixed).all() or np.any(np.asarray(v_fixed) < self.lo[i_fixed])
                    or np.any(np.asarray(v_fixed) > self.hi[i_fixed])):
                raise ValueError("fixed coordinate outside prior bounds")
            mask[i_fixed] = False
        best, attempts = None, []

        def consider(u, source, result=None):
            nonlocal best
            u = np.asarray(u, float)
            if u.shape != (int(mask.sum()),) or not np.isfinite(u).all():
                attempts.append(dict(source=source, finite=False, message="non-finite coordinates"))
                return
            u = np.clip(u, -1, 1)
            value, gradient = self.objective(u, i_fixed, v_fixed)
            gradient = np.asarray(gradient, float)
            finite = bool(np.isfinite(value) and gradient.shape == u.shape
                          and np.isfinite(gradient).all())
            pg = float(np.max(np.abs(u - np.clip(u - gradient, -1, 1)), initial=0)) if finite else None
            rec = dict(source=source, finite=finite,
                       logl=-float(value) if finite else None,
                       projected_gradient_inf=pg,
                       stationary=bool(finite and pg <= self.options['gtol']),
                       optimizer_success=bool(result.success) if result is not None else None,
                       status=int(result.status) if result is not None else None,
                       message=str(result.message) if result is not None else "seed evaluation",
                       nit=int(result.nit) if result is not None else 0,
                       nfev=int(result.nfev) if result is not None else 1)
            attempts.append(rec)
            if finite and (best is None or value < best[0]
                           or (value == best[0] and pg < best[2]['projected_gradient_inf'])):
                best = float(value), u.copy(), rec

        def solve(start, source):
            s = np.asarray(start, float)
            if s.shape != self.lo.shape or not np.isfinite(s).all():
                raise ValueError("optimizer starts must be finite full parameter vectors")
            u = (np.clip(s, self.lo, self.hi)[mask] - self.centre[mask]) / self.half[mask]
            consider(u, source + ":seed")
            if not len(u):
                return  # One-dimensional profile: no nuisance optimization.
            r = minimize(self.objective, u, args=(i_fixed, v_fixed), jac=True,
                         method="L-BFGS-B", bounds=[(-1, 1)] * len(u), options=self.options)
            consider(r.x, source, r)

        for j, start in enumerate(starts):
            solve(start, f"start_{j}")
        retries = 0
        while (best is None or not best[2]['stationary']) and retries < self.retry_starts:
            solve(self.rng.uniform(self.lo, self.hi), f"retry_{retries}")
            retries += 1
        if best is None:
            raise RuntimeError("no finite profile candidate; check the surrogate and starting points")
        full = np.empty(len(self.lo))
        full[mask] = self.centre[mask] + best[1] * self.half[mask]
        full = np.clip(full, self.lo, self.hi)  # remove round-off at active bounds
        if i_fixed is not None:
            full[i_fixed] = v_fixed
        diagnostic = dict(best[2], retries=retries, attempts=attempts)
        return -best[0], full, diagnostic




class PolishedOptimizer:
    def __init__(self, base):
        self.base = base

    def __call__(self, i, v, starts):
        value, x, diag = self.base(i, v, starts)
        if diag["stationary"]:
            return value, x, diag
        base = self.base
        mask = np.ones(len(base.lo), bool)
        if i is not None:
            mask[i] = False
        u = (x[mask] - base.centre[mask]) / base.half[mask]
        offset = base.objective(u, i, v)[0]

        def objective(q):
            f, g = base.objective(q, i, v)
            return f - offset, g

        attempts = []
        for k in range(3):
            r = minimize(
                objective,
                u,
                jac=True,
                method="L-BFGS-B",
                bounds=[(-1, 1)] * len(u),
                options=dict(
                    base.options, ftol=0.0, gtol=base.options["gtol"] * 0.25, maxls=50
                ),
            )
            u = np.clip(r.x, -1, 1)
            f, g = base.objective(u, i, v)
            pg = float(np.max(abs(u - np.clip(u - g, -1, 1)), initial=0))
            stationary = pg <= base.options["gtol"]
            rec = dict(
                source=f"shifted_polish_{k}",
                finite=bool(np.isfinite(f) and np.isfinite(g).all()),
                logl=-float(f),
                projected_gradient_inf=pg,
                stationary=stationary,
                optimizer_success=bool(r.success),
                status=int(r.status),
                message=str(r.message),
                nit=int(r.nit),
                nfev=int(r.nfev),
            )
            attempts.append(rec)
            if stationary and -f >= value - 1e-7:
                xx = x.copy()
                xx[mask] = base.centre[mask] + base.half[mask] * u
                out = dict(
                    rec,
                    retries=diag.get("retries", 0),
                    attempts=diag["attempts"] + attempts,
                    polish_original_logl=value,
                    polish_logl_change=float(-f - value),
                )
                return -float(f), xx, out
        # At machine-precision objective ties, line search can stall even with
        # accurate autodiff gradients. Newton-polish those gradients using their
        # finite-difference Hessian; validate the value AND stationarity afterward.
        for k in range(8):
            f, g = base.objective(u, i, v)
            pg0 = float(np.max(abs(u - np.clip(u - g, -1, 1))))
            active = ((u <= -1 + 1e-10) & (g > 0)) | ((u >= 1 - 1e-10) & (g < 0))
            free = np.where(~active)[0]
            if not len(free):
                break
            h = np.empty((len(free), len(free)))
            for col, j in enumerate(free):
                up = u.copy()
                um = u.copy()
                up[j] = min(1.0, u[j] + 1e-5)
                um[j] = max(-1.0, u[j] - 1e-5)
                h[:, col] = (
                    base.objective(up, i, v)[1][free]
                    - base.objective(um, i, v)[1][free]
                ) / (up[j] - um[j])
            eig, vec = np.linalg.eigh((h + h.T) / 2)
            step = -(vec @ ((vec.T @ g[free]) / np.maximum(eig, 1e-6)))
            step *= min(1.0, 0.05 / max(np.linalg.norm(step), 1e-30))
            accepted = False
            for factor in [1.0, 0.5, 0.25, 0.125]:
                candidate = u.copy()
                candidate[free] += factor * step
                candidate = np.clip(candidate, -1, 1)
                ff, gg = base.objective(candidate, i, v)
                pg = float(np.max(abs(candidate - np.clip(candidate - gg, -1, 1))))
                if np.isfinite(ff) and -ff >= value - 1e-7 and pg < pg0:
                    u = candidate
                    accepted = True
                    rec = dict(
                        source=f"gradient_newton_polish_{k}",
                        finite=True,
                        logl=-float(ff),
                        projected_gradient_inf=pg,
                        stationary=pg <= base.options["gtol"],
                        optimizer_success=None,
                        status=None,
                        message="autodiff-gradient Newton polish, objective and projected-gradient checked",
                        nit=k + 1,
                        nfev=2 * len(free) + 1,
                    )
                    attempts.append(rec)
                    if rec["stationary"]:
                        xx = x.copy()
                        xx[mask] = base.centre[mask] + base.half[mask] * u
                        return (
                            -float(ff),
                            xx,
                            dict(
                                rec,
                                retries=diag.get("retries", 0),
                                attempts=diag["attempts"] + attempts,
                                polish_original_logl=value,
                                polish_logl_change=float(-ff - value),
                            ),
                        )
                    break
            if not accepted:
                break
        diag = dict(diag, polish_attempts=attempts)
        return value, x, diag


def surrogate_optimizer(*args, **kwargs):
    return PolishedOptimizer(_bounded_optimizer(*args, **kwargs))


def scan_profiles(optimise, starts, grids, indices, *, max_passes=15, value_tol=1e-4):
    """Continue each profile from its anchor and preserve maxima across passes.

    Stop when the anchor is consistent and profile values stabilize, but report
    local stationarity separately. Neither condition proves a global maximum.
    """
    if max_passes < 1 or not np.isfinite(value_tol) or value_tol <= 0:
        raise ValueError("positive pass budget and value tolerance required")
    if len(grids) != len(indices) or not len(indices):
        raise ValueError("one grid is required per profiled parameter")
    for grid in grids:
        if len(grid) < 2 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
            raise ValueError("profile grids must be finite and strictly increasing")
    anchor_v, anchor_x, anchor_diag = optimise(None, None, starts)
    results = [dict(grid=np.asarray(g), logl=np.full(len(g), -np.inf),
                    xopt=np.zeros((len(g), len(anchor_x))), diagnostics=[None]*len(g))
               for g in grids]
    stable = consistent = False
    for npass in range(1, max_passes + 1):
        largest_change = 0.0
        for i, rec in zip(indices, results):
            grid, prof, xopt, diagnostics = (rec[k] for k in ('grid', 'logl', 'xopt', 'diagnostics'))
            before = prof.copy()

            def update(j, candidates):
                if np.isfinite(prof[j]):
                    candidates = [xopt[j].copy(), *candidates]
                v, x, diagnostic = optimise(i, grid[j], candidates)
                old = prof[j]
                if v > old or (v == old and (diagnostics[j] is None or
                        diagnostic['projected_gradient_inf'] < diagnostics[j]['projected_gradient_inf'])):
                    prof[j], xopt[j], diagnostics[j] = v, x, diagnostic
                return prof[j] > old + value_tol

            centre_j = int(np.argmin(np.abs(grid - anchor_x[i])))
            for direction in (+1, -1):
                warm = anchor_x.copy()
                for j in range(centre_j, len(grid) if direction == 1 else -1, direction):
                    update(j, [warm, anchor_x])
                    warm = xopt[j].copy()
            for _ in range(4):
                moved = False
                for j in range(len(grid)):
                    moved = update(j, [xopt[m].copy() for m in (j-1, j+1)
                                       if 0 <= m < len(grid)]) or moved
                if not moved:
                    break
            largest_change = max(largest_change, float(np.max(prof - before)))

        best_rec = max(results, key=lambda r: float(np.max(r['logl'])))
        j = int(np.argmax(best_rec['logl']))
        if best_rec['logl'][j] > anchor_v + value_tol:
            v, x, diagnostic = optimise(None, None, [best_rec['xopt'][j], anchor_x])
            if v >= anchor_v:
                anchor_v, anchor_x, anchor_diag = v, x, diagnostic
        consistent = bool(max(float(np.max(r['logl'])) for r in results) <= anchor_v + value_tol)
        stable = bool(largest_change <= value_tol)
        print(f"pass {npass}: best logL {anchor_v:.7g}; anchor consistent={consistent}, "
              f"profiles stable={stable}", flush=True)
        if consistent and stable:
            break
    stationary = bool(anchor_diag['stationary'] and
                      all(d['stationary'] for r in results for d in r['diagnostics']))
    quality = dict(passes_completed=npass, max_passes=max_passes,
                   pass_cap_reached=not (consistent and stable), anchor_consistent=consistent,
                   profiles_stable=stable, all_selected_points_stationary=stationary,
                   converged=bool(consistent and stable and stationary),
                   global_optimum_certified=False, value_tolerance=value_tol,
                   anchor=anchor_diag)
    return anchor_v, anchor_x, results, quality


def scan(model, metadata, data, names, grids, config):
    bounds = np.array([metadata.bounds[n] for n in metadata.param_names])
    optimize = surrogate_optimizer(model, bounds, seed=config.seed, gtol=config.gtol)
    starts = [
        *data[np.argsort(data[:, -1])[-min(12, len(data)) :], :-1],
        bounds.mean(axis=1),
    ]
    value, anchor, records, quality = scan_profiles(
        optimize,
        starts,
        grids,
        [metadata.param_names.index(n) for n in names],
        max_passes=config.max_passes,
    )
    result = dict(
        names=names,
        full_parameter_names=metadata.param_names,
        best_logl=value,
        best_x=anchor,
        profiles={
            n: dict(
                grid=r["grid"],
                logl=r["logl"],
                xopt=r["xopt"],
                optimization=r["diagnostics"],
            )
            for n, r in zip(names, records)
        },
        optimization_quality=quality,
        objective="native",
        grid_fractions={n: 1.0 for n in names},
    )
    return result, optimize


def display_grids(profile, bounds_map, config):
    """Exactly requested fresh nodes; never subsample existing optima."""
    from profiles.metric import crossings

    grids = []
    diagnostics = {}
    for name in config.parameters:
        lo, hi = bounds_map[name]
        rec = profile["profiles"][name]
        i = profile["full_parameter_names"].index(name)
        center = profile["best_x"][i]
        interval = crossings(
            rec["grid"], 2 * (profile["best_logl"] - np.array(rec["logl"])), 1.0
        )
        requested = config.ranges.get(name)
        if requested == "prior":
            low, high = lo, hi
            kind = "full metadata bounds"
        elif requested is not None:
            low, high = requested
            kind = "explicit physical range"
            if low < lo or high > hi:
                raise ValueError(f"{name} requested range outside saved model domain")
        elif len(interval) == 2 and interval[0] < center < interval[1]:
            low = center - config.sigma_range * (center - interval[0])
            high = center + config.sigma_range * (interval[1] - center)
            kind = "surrogate 1-sigma widths"
        else:
            low, high = lo, hi
            kind = "full metadata bounds: no two-sided surrogate 1-sigma interval"
        actual = [max(lo, low), min(hi, high)]
        grids.append(np.linspace(*actual, config.profile_points))
        diagnostics[name] = dict(
            requested=[float(low), float(high)],
            actual=actual,
            clipped=bool(actual[0] != low or actual[1] != high),
            kind=kind,
            points=config.profile_points,
            surrogate_1sigma_crossings=interval,
        )
    return grids, diagnostics
