"""Hybrid profile optimiser: BEST-style batched walkers, CLiENT-style certification.

Surrogate stage
  * Every grid node and the global anchor are optimised together; each step is one
    batched float64 TensorFlow evaluation, as in BEST's Optimiser.
  * Each node keeps persistent walkers. A hop either perturbs a walker with a Gaussian
    step shaped by the anchor Laplace covariance (BEST's covariance preconditioning) or
    copies a neighbouring node's incumbent (CLiENT's continuation), then ascends
    locally. Metropolis acceptance at a fixed temperature lets walkers leave a basin;
    the node incumbent is elitist over every local optimum found. Step sizes adapt to
    a target acceptance rate. Hopping stops once no node improves for `patience` hops.
  * The local ascent is a batched projected BFGS in prior-box coordinates. Its Hessian
    model starts from the exact autodiff Hessian (whitening) and is carried across hops.
    Bounds are exact through an epsilon-active set. There is no relative-ftol stopping
    rule: rows stop on CLiENT's projected-gradient test, or when no step increases the
    objective in float64.
  * Every incumbent is certified: projected gradient <= gtol, and a second-order test
    on the exact Hessian of the free, inactive coordinates. Saddles are left along the
    direction of positive curvature before recertifying.

The true stage (profiles/residual_newton.py, profiles/hybrid_surface.py) shares the ledgers
below: ConcurrentLedger, a thread-safe view of the study's ledger, and NodeLedger, one node's
view of it with its own call counts.
"""

from collections import Counter
from dataclasses import dataclass, asdict
import threading
import time

import numpy as np


# --------------------------------------------------------------------------- surrogate


class BatchedSurrogate:
    """Float64 value, gradient and Hessian of the surrogate, batched, in box coordinates.

    u in [-1, 1] maps to x = centre + half * u. Gradients and Hessians are returned in u.
    `counts` records points evaluated per kind; a Hessian costs roughly N gradients.
    """

    def __init__(self, model, bounds):
        import tensorflow as tf
        from model.network import copy_model_dtype

        if model.compute_dtype != "float64":
            model = copy_model_dtype(model, "float64")
        bounds = np.asarray(bounds, float)
        self.lo, self.hi = bounds[:, 0], bounds[:, 1]
        self.centre, self.half = (self.lo + self.hi) / 2, (self.hi - self.lo) / 2
        self.n = len(bounds)
        self.counts = Counter()
        self.seconds = Counter()
        self._tf = tf
        spec = [tf.TensorSpec([None, self.n], tf.float64)]

        @tf.function(input_signature=spec)
        def value(x):
            return (model(x, training=False)[:, 0],)

        @tf.function(input_signature=spec)
        def gradient(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                v = model(x, training=False)[:, 0]
            return v, tape.gradient(v, x)

        @tf.function(input_signature=spec)
        def hessian(x):
            with tf.GradientTape() as outer:
                outer.watch(x)
                with tf.GradientTape() as inner:
                    inner.watch(x)
                    v = model(x, training=False)[:, 0]
                g = inner.gradient(v, x)
            return v, g, outer.batch_jacobian(g, x)

        self._fns = dict(value=value, gradient=gradient, hessian=hessian)

    def physical(self, U):
        return np.clip(self.centre + self.half * np.asarray(U, float), self.lo, self.hi)

    def box(self, X):
        return np.clip((np.asarray(X, float) - self.centre) / self.half, -1, 1)

    def _run(self, kind, U, chunk):
        U = np.asarray(U, float).reshape(-1, self.n)
        self.counts[kind] += len(U)
        start = time.monotonic()
        parts = [
            [t.numpy() for t in self._fns[kind](self._tf.constant(self.physical(U[s : s + chunk])))]
            for s in range(0, len(U), chunk)
        ]
        self.seconds[kind] += time.monotonic() - start
        if not parts:
            width = dict(value=1, gradient=2, hessian=3)[kind]
            shapes = [(0,), (0, self.n), (0, self.n, self.n)][:width]
            return [np.zeros(s) for s in shapes]
        return [np.concatenate(p) for p in zip(*parts)]

    def value(self, U):
        return self._run("value", U, 8192)[0]

    def value_grad(self, U):
        v, g = self._run("gradient", U, 4096)
        return v, g * self.half

    def value_grad_hess(self, U):
        v, g, h = self._run("hessian", U, 256)
        return v, g * self.half, h * np.outer(self.half, self.half)


def projected_gradient(U, g, fixed):
    """CLiENT's stationarity measure for maximisation over free coordinates of the box."""
    pg = np.where(fixed, 0.0, np.abs(U - np.clip(U + g, -1, 1)))
    return pg.max(axis=1, initial=0.0)


def positive_curvature(A, relative_floor=1e-8):
    """Nearest-in-spectrum positive definite matrix: |eigenvalues| with a relative floor."""
    A = 0.5 * (A + np.swapaxes(A, -1, -2))
    w, V = np.linalg.eigh(A)
    floor = np.maximum(1e-12, relative_floor * np.max(np.abs(w), axis=-1, keepdims=True))
    w = np.maximum(np.abs(w), floor)
    return (V * w[..., None, :]) @ np.swapaxes(V, -1, -2)


@dataclass
class Ascent:
    U: np.ndarray
    f: np.ndarray
    g: np.ndarray
    B: np.ndarray
    pg: np.ndarray
    iterations: np.ndarray
    status: np.ndarray


def ascend(sb, U, fixed, B=None, *, gtol=1e-4, max_iter=300, trust=0.5, c1=1e-4,
           backtracks=40):
    """Batched projected BFGS ascent of the surrogate, one independent problem per row.

    fixed marks coordinates held at their current value (the profiled ones). B holds
    positive definite models of -Hessian in box coordinates; None starts each row from
    the modified exact Hessian. Steps are capped at `trust` in the infinity norm.
    """
    U = np.clip(np.array(U, float), -1, 1)
    fixed = np.asarray(fixed, bool)
    R, N = U.shape
    f, g = sb.value_grad(U)
    if B is None:
        B = positive_curvature(-sb.value_grad_hess(U)[2])
    else:
        B = np.array(B, float)
    pg = projected_gradient(U, g, fixed)
    iterations = np.zeros(R, int)
    status = np.where(pg <= gtol, "stationary", "running").astype(object)
    status[~np.isfinite(f)] = "nonfinite"
    eye = np.eye(N)
    alphas = 0.5 ** np.arange(1, backtracks + 1)
    for _ in range(max_iter):
        run = np.flatnonzero(status == "running")
        if not len(run):
            break
        u, gr, fr, Br, fx = U[run], g[run], f[run], B[run], fixed[run]
        eps = np.minimum(1e-3, pg[run])[:, None]
        active = ~fx & (((u <= -1 + eps) & (gr < 0)) | ((u >= 1 - eps) & (gr > 0)))
        free = ~fx & ~active
        M = np.where(free[:, :, None] & free[:, None, :], Br, 0.0) + eye * (~free)[:, :, None]
        rhs = np.where(free, gr, 0.0)[..., None]
        try:
            d = np.linalg.solve(M, rhs)[..., 0]
        except np.linalg.LinAlgError:
            # numpy refuses the whole batch when one row's model is singular (a BFGS model
            # driven there by a region's wall: the SH0ES run's study of 21 October 2026 stopped
            # at its scan). Solve row by row; a singular row takes its free gradient, capped
            # below like every step, and starts its model afresh. Without a singular row this
            # path is never taken.
            d = np.empty((len(run), N))
            for i in range(len(run)):
                try:
                    d[i] = np.linalg.solve(M[i], rhs[i])[:, 0]
                except np.linalg.LinAlgError:
                    d[i] = rhs[i, :, 0]
                    Br[i] = eye
        d *= np.minimum(1.0, trust / np.maximum(np.abs(d).max(axis=1), 1e-300))[:, None]

        u1 = np.clip(u + d, -1, 1)
        f1, g1 = sb.value_grad(u1)
        ok = np.isfinite(f1) & (f1 - fr >= c1 * np.sum(gr * (u1 - u), axis=1)) & (f1 >= fr)
        # Near a stiff optimum the gain still available at the gtol threshold can be below
        # the network's float64 evaluation noise (SINU TT it10: curvature ~4e5 in box units,
        # log L ~ -3240, value noise ~1e-10, i.e. ~1e2 ulp), so value tests can no longer
        # rank points and a correct Newton step is rejected about half the time. Then
        # accept a full step that stays within that noise (1024 ulp, 7e-10 here) and
        # reduces the projected gradient, as the legacy polish does with its bounded loss.
        noise = 1024 * np.finfo(float).eps * np.maximum(1.0, np.abs(fr))
        ok |= (~ok & np.isfinite(f1) & (np.abs(f1 - fr) <= noise)
               & (projected_gradient(u1, g1, fx) < pg[run]))
        bad = np.flatnonzero(~ok)
        stalled = np.zeros(len(run), bool)
        # Backtrack in stages: three halvings first, the rest only for rows still failing.
        for stage in (alphas[:3], alphas[3:]):
            if not len(bad) or not len(stage):
                break
            cand = np.clip(u[bad, None, :] + stage[None, :, None] * d[bad, None, :], -1, 1)
            fv = sb.value(cand.reshape(-1, N)).reshape(len(bad), -1)
            good = np.isfinite(fv) & (fv - fr[bad, None] >= c1 * np.sum(
                gr[bad, None, :] * (cand - u[bad, None, :]), axis=2)) & (fv >= fr[bad, None])
            found = good.any(axis=1)
            rows = bad[found]
            if len(rows):
                u1[rows] = cand[found, np.argmax(good[found], axis=1)]
                f1[rows], g1[rows] = sb.value_grad(u1[rows])
            bad = bad[~found]
        stalled[bad] = True
        move = ~stalled
        # Damped (Powell) BFGS update of the -Hessian model: A s ~= y = -(g_new - g).
        s = (u1 - u)[move]
        y = -(g1 - gr)[move]
        Bm = Br[move]
        Bs = np.einsum("rij,rj->ri", Bm, s)
        sBs = np.sum(s * Bs, axis=1)
        sy = np.sum(s * y, axis=1)
        theta = np.where(sy >= 0.2 * sBs, 1.0, 0.8 * sBs / np.maximum(sBs - sy, 1e-300))
        r = theta[:, None] * y + (1 - theta)[:, None] * Bs
        sr = np.sum(s * r, axis=1)
        upd = (sBs > 1e-300) & (sr > 1e-300)
        Bm[upd] += (np.einsum("ri,rj->rij", r[upd], r[upd]) / sr[upd, None, None]
                    - np.einsum("ri,rj->rij", Bs[upd], Bs[upd]) / sBs[upd, None, None])
        idx = run[move]
        U[idx], f[idx], g[idx], B[idx] = u1[move], f1[move], g1[move], Bm
        iterations[run] += 1
        pg[run] = projected_gradient(U[run], g[run], fixed[run])
        status[run[stalled]] = "stalled"
        status[run[move & (pg[run] <= gtol)]] = "stationary"
    status[status == "running"] = "max_iter"
    return Ascent(U, f, g, B, pg, iterations, status)


def certify(sb, U, fixed, *, gtol=1e-4, trust=0.5, max_iter=300, rounds=4,
            curvature_rtol=1e-6, newton_rounds=10, newton_iter=3, gain_tol=1e-6):
    """Tight ascent, then a second-order test with saddle escape, one row per node.

    Coordinates at a bound with an outward gradient are excluded from the curvature
    test (first-order optimality holds there). A direction of positive curvature of the
    objective is followed both ways; the row re-ascends only if the objective improves.

    A row is stationary when its projected gradient is within `gtol` (box coordinates)
    or, failing that, when its Newton decrement, the predicted remaining gain
    g^T (-H)^{-1} g / 2 over the free coordinates at a local maximum, is within
    `gain_tol` (log-likelihood units). The second test does not depend on the
    parameterisation: on a very ill-conditioned target a fully converged row can keep a
    box-coordinate gradient above gtol along its stiffest directions. It is evaluated
    only for rows that fail the first, and never changes a point or value.
    """
    fixed = np.asarray(fixed, bool)
    res = ascend(sb, U, fixed, None, gtol=gtol, max_iter=max_iter, trust=trust)
    newton_restarts = np.zeros(len(res.U), int)
    for _ in range(newton_rounds):
        # BFGS can stall just above gtol; restart those rows from the exact Hessian.
        rows = np.flatnonzero(res.pg > gtol)
        if not len(rows):
            break
        again = ascend(sb, res.U[rows], fixed[rows], None, gtol=gtol, max_iter=newton_iter, trust=trust)
        for name in ("U", "f", "g", "B", "pg", "status"):
            getattr(res, name)[rows] = getattr(again, name)
        res.iterations[rows] += again.iterations
        newton_restarts[rows] += 1
    R, N = res.U.shape
    escapes = np.zeros(R, int)
    lam_min = np.zeros(R)
    lam_max = np.zeros(R)
    unresolved = np.zeros(R, bool)
    radii = trust * np.array([1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001])
    todo = np.arange(R)
    for _ in range(rounds + 1):
        if not len(todo):
            break
        u = res.U[todo]
        f, g, H = sb.value_grad_hess(u)
        at_bound = ~fixed[todo] & (((u <= -1 + 1e-9) & (g < 0)) | ((u >= 1 - 1e-9) & (g > 0)))
        free = ~fixed[todo] & ~at_bound
        both = free[:, :, None] & free[:, None, :]
        A = np.where(both, -H, 0.0)
        # Pad non-free coordinates above the free block's spectral radius (<= N max|a|),
        # keeping LAPACK's internal scaling benign; the smallest eigenvalue is then free.
        big = (2 * N * np.max(np.abs(A), axis=(1, 2)) + 1.0)[:, None, None]
        A = A + np.eye(N) * big * (~free)[:, :, None]
        w, V = np.linalg.eigh(0.5 * (A + np.swapaxes(A, 1, 2)))
        scale = np.max(np.where(np.abs(w) < big[:, :, 0] / 2, np.abs(w), 0.0), axis=1)
        lam_min[todo], lam_max[todo] = w[:, 0], scale
        saddle = free.any(axis=1) & (w[:, 0] < -curvature_rtol * np.maximum(1.0, scale))
        rows = np.flatnonzero(saddle)
        next_todo = []
        if len(rows):
            v = V[rows, :, 0]
            cand = np.clip(u[rows, None, :] + np.concatenate([radii, -radii])[None, :, None] * v[:, None, :], -1, 1)
            cand = np.where(fixed[todo[rows], None, :], u[rows, None, :], cand)
            fv = sb.value(cand.reshape(-1, N)).reshape(len(rows), -1)
            k = np.argmax(np.where(np.isfinite(fv), fv, -np.inf), axis=1)
            better = fv[np.arange(len(rows)), k] > f[rows] + 1e-12 * np.maximum(1.0, np.abs(f[rows]))
            unresolved[todo[rows[~better]]] = True
            go = rows[better]
            if len(go):
                start = cand[better, k[better]]
                again = ascend(sb, start, fixed[todo[go]], None, gtol=gtol, max_iter=max_iter, trust=trust)
                for name in ("U", "f", "g", "B", "pg", "status"):
                    getattr(res, name)[todo[go]] = getattr(again, name)
                res.iterations[todo[go]] += again.iterations
                escapes[todo[go]] += 1
                next_todo = todo[go]
        todo = np.asarray(next_todo, int)
    unresolved[todo] = True  # still saddles after the final allowed escape
    curvature_ok = lam_min >= -curvature_rtol * np.maximum(1.0, lam_max)
    gain = np.full(R, np.nan)  # not evaluated (gradient test passed) or not a local maximum
    rows = np.flatnonzero((res.pg > gtol) & curvature_ok & ~unresolved)
    if len(rows):
        u = res.U[rows]
        _, g, H = sb.value_grad_hess(u)
        free = ~fixed[rows] & ~(((u <= -1 + 1e-9) & (g < 0)) | ((u >= 1 - 1e-9) & (g > 0)))
        for j, r in enumerate(rows):
            f = free[j]
            if not f.any():
                gain[r] = 0.0
                continue
            A = -0.5 * (H[j] + H[j].T)[np.ix_(f, f)]
            w, V = np.linalg.eigh(A)
            # Strictly concave on the free block, allowing any conditioning above round-off;
            # a near-flat free direction with a gradient along it gives a large gain (fails).
            if w[0] > 1e-12 * abs(w[-1]):
                gain[r] = 0.5 * float(np.sum((V.T @ g[j][f]) ** 2 / w))
    stationary = (res.pg <= gtol) | (gain <= gain_tol)
    local_max = stationary & curvature_ok
    return res, dict(escapes=escapes, newton_restarts=newton_restarts, lambda_min=lam_min, lambda_max=lam_max,
                     local_max=local_max, negative_curvature_unresolved=unresolved & ~local_max,
                     stationary=stationary, remaining_gain=gain)


# ------------------------------------------------------------------------ basin hopping


@dataclass
class HybridConfig:
    walkers: int = 8
    hops_min: int = 4
    hops_max: int = 12
    patience: int = 2
    temperature: float = 1.0  # log-likelihood units: exp(-dlogL/T) Metropolis acceptance
    step0: float = 1.0  # in anchor-Laplace standard deviations
    step_bounds: tuple = (0.05, 5.0)
    exchange: float = 0.25  # probability a hop copies a neighbour's incumbent
    target_accept: float = 0.5
    reset_gap: float = 25.0  # walkers this far below the node best restart from it
    gtol: float = 1e-4
    gain_tol: float = 1e-6  # log L: a row whose Newton decrement is this small is stationary even
    # when its box-coordinate gradient exceeds gtol (stiff directions of ill-conditioned targets)
    value_tol: float = 1e-4
    trust: float = 0.5
    max_iter: int = 300  # certification ascent
    hop_gtol_factor: float = 10.0  # hop ascents only locate basins; certification polishes
    hop_max_iter: int = 100
    seed: int = 42


def hybrid_settings(config):
    """The HybridConfig of a profile study's settings: hybrid_walkers, seed, gtol and, when
    set, hybrid_hops as the limit on the hops (otherwise HybridConfig's own)."""
    hops = getattr(config, "hybrid_hops", None)
    return HybridConfig(walkers=config.hybrid_walkers, seed=config.seed, gtol=config.gtol,
                        **({} if hops is None else dict(hops_max=hops)))


def curvature_geometry(hessian, *, maximum_step=.5, relative_floor=1e-6):
    """Regularized inverse square root in prior-box coordinates, without dropping modes."""
    hessian = np.asarray(hessian, float)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1] or not np.isfinite(hessian).all():
        raise ValueError('finite square Hessian required')
    values, vectors = np.linalg.eigh(.5*(hessian+hessian.T))
    floor = max(1e-8, relative_floor*float(np.max(abs(values), initial=0)))
    scales = np.minimum(maximum_step, 1/np.sqrt(np.maximum(values, floor)))
    return vectors * scales, dict(eigenvalues=values.tolist(), eigenvalue_floor=floor,
                                 capped_modes=int(np.sum(scales == maximum_step)))


def laplace_geometry(sb, u_anchor):
    """Proposal shape: columns are anchor -Hessian eigenvectors times 1/sqrt(eigenvalue),
    capped at half a box so flat directions still move. Same map as the true stage uses."""
    H = sb.value_grad_hess(u_anchor[None])[2][0]
    geometry, info = curvature_geometry(-H)
    return geometry, info


def reoptimise_anchor(sb, res, second, N, cfg, updates=3):
    """While a node beats the anchor (row 0), re-optimise the anchor from the best node, in place.
    A search that does not beat it ends this: certify is deterministic, and nothing else changed.
    Returns the number of searches."""
    for done in range(updates):
        if np.max(res.f[1:]) <= res.f[0] + cfg.value_tol:
            return done
        k = 1 + int(np.argmax(res.f[1:]))
        a, s2 = certify(sb, res.U[k][None], np.zeros((1, N), bool), gtol=cfg.gtol, trust=cfg.trust,
                        gain_tol=cfg.gain_tol)
        if not a.f[0] > res.f[0]:
            return done + 1
        for name in ("U", "f", "g", "pg", "status", "iterations"):
            getattr(res, name)[0] = getattr(a, name)[0]
        for key in second:
            second[key][0] = s2[key][0]
    return updates


def hop(sb, fixed_mask, fixed_u, init, neighbours, geometry, cfg, rng, log=print):
    """Basin hopping for K nodes with W persistent walkers each, all rows in one batch."""
    K, N = fixed_mask.shape
    W = cfg.walkers
    node = np.repeat(np.arange(K), W)
    Fm, Fu = fixed_mask[node], fixed_u[node]

    def pin(U):
        return np.where(Fm, Fu, np.clip(U, -1, 1))

    loose = dict(gtol=cfg.gtol * cfg.hop_gtol_factor, max_iter=cfg.hop_max_iter, trust=cfg.trust)
    first = ascend(sb, pin(init), Fm, None, **loose)
    U, f, B = first.U, first.f, first.B
    best_f = np.full(K, -np.inf)
    best_U = np.zeros((K, N))
    found = np.zeros(K, int)

    def record(Ucand, fcand):
        """Elitist node incumbents over every local optimum found; returns gain per node."""
        before = best_f.copy()
        fc = np.where(np.isfinite(fcand), fcand, -np.inf).reshape(K, W)
        winner = np.argmax(fc, axis=1)
        top = fc[np.arange(K), winner]
        better = top > best_f
        best_f[better] = top[better]
        best_U[better] = Ucand.reshape(K, W, N)[better, winner[better]]
        found[better] += 1
        return np.where(np.isfinite(before), best_f - before, np.where(better, np.inf, 0.0))

    record(U, f)
    step = np.full(K, cfg.step0)
    accepted_total = np.zeros(K)
    proposed_total = np.zeros(K)
    stale = 0
    hops_done = 0
    for h in range(1, cfg.hops_max + 1):
        xi = rng.standard_normal((K * W, N))
        delta = step[node][:, None] * (xi @ geometry.T)
        proposal = U + delta
        far = f < best_f[node] - cfg.reset_gap
        proposal[far] = best_U[node[far]] + delta[far]
        exchange = (rng.random(K * W) < cfg.exchange) & ~far
        for r in np.flatnonzero(exchange):
            options = [j for j in neighbours[node[r]] if np.isfinite(best_f[j])]
            if options:
                proposal[r] = best_U[options[rng.integers(len(options))]]
            else:
                exchange[r] = False
        res = ascend(sb, pin(proposal), Fm, B, **loose)
        with np.errstate(over="ignore"):
            accept = np.isfinite(res.f) & (
                (res.f >= f) | (rng.random(K * W) < np.exp(np.minimum(0.0, res.f - f) / cfg.temperature)))
        U[accept], f[accept], B[accept] = res.U[accept], res.f[accept], res.B[accept]
        improvement = record(res.U, res.f)
        perturb = ~exchange & ~far
        acc = np.bincount(node[perturb], weights=accept[perturb], minlength=K)
        tot = np.bincount(node[perturb], minlength=K)
        accepted_total += acc
        proposed_total += tot
        rate = np.divide(acc, tot, out=np.full(K, cfg.target_accept), where=tot > 0)
        step = np.clip(np.where(rate > cfg.target_accept, step * 1.3, step / 1.3), *cfg.step_bounds)
        hops_done = h
        largest = float(np.max(improvement))
        stale = stale + 1 if largest <= cfg.value_tol else 0
        log(f"  hop {h}: nodes improved {int(np.sum(improvement > cfg.value_tol))}/{K}, "
            f"largest gain {largest:.3g}, acceptance {np.sum(acc) / max(1, np.sum(tot)):.2f}, "
            f"ascent iterations median {int(np.median(res.iterations))}", flush=True)
        if h >= cfg.hops_min and stale >= cfg.patience:
            break
    stable = stale >= cfg.patience
    return best_f, best_U, dict(hops=hops_done, stable=bool(stable), incumbents_found=found,
                                acceptance=np.divide(accepted_total, np.maximum(proposed_total, 1)),
                                final_step=step)


def scan_hybrid(model, metadata, data, names, grids, config=None, *, hybrid=None, log=print,
                surrogate=None, surface_indices=None):
    """Drop-in replacement for profiles.surrogate.scan; returns (result, BatchedSurrogate).

    `surrogate` may supply any object with BatchedSurrogate's interface (for example an
    analytic objective in tests); `model` is then unused.
    """
    cfg = hybrid or HybridConfig(**({} if config is None else dict(
        seed=getattr(config, "seed", 42), gtol=getattr(config, "gtol", 1e-4))))
    rng = np.random.default_rng(cfg.seed)
    t0 = time.monotonic()
    params = list(metadata.param_names)
    bounds = np.array([metadata.bounds[n] for n in params], float)
    sb = surrogate if surrogate is not None else BatchedSurrogate(model, bounds)
    N = len(params)
    W = cfg.walkers

    # Anchor: the legacy starting set (top-12 labels and the box centre), then hopping.
    starts = np.vstack([data[np.argsort(data[:, -1])[-min(12, len(data)):], :-1], bounds.mean(axis=1)])
    seed_run = ascend(sb, sb.box(starts), np.zeros((len(starts), N), bool), None,
                      gtol=cfg.gtol * cfg.hop_gtol_factor, max_iter=cfg.hop_max_iter, trust=cfg.trust)
    u0 = seed_run.U[np.argmax(seed_run.f)]
    geometry, geo_info = laplace_geometry(sb, u0)
    order = np.argsort(-seed_run.f)
    seeded = min(W // 2, len(order))
    init = np.vstack([seed_run.U[order[:seeded]]] +
                     [u0 + cfg.step0 * (rng.standard_normal(N) @ geometry.T) for _ in range(W - seeded)])
    log("anchor hopping", flush=True)
    af, aU, _ = hop(sb, np.zeros((1, N), bool), np.zeros((1, N)), init, [[]], geometry, cfg, rng, log)
    u_anchor = aU[0]

    # All profile nodes plus the anchor (node 0) in one batch.
    specs = [(None, None, None)]
    if surface_indices is None:
        for p, (name, grid) in enumerate(zip(names, grids)):
            i = params.index(name)
            specs += [(p, j, (i, float(v))) for j, v in enumerate(grid)]
    else:
        specs += [(i, j, (surface_indices, np.array([a, b], float)))
                  for i, a in enumerate(grids[0]) for j, b in enumerate(grids[1])]
    K = len(specs)
    fixed_mask = np.zeros((K, N), bool)
    fixed_u = np.zeros((K, N))
    for k, (_, _, fix) in enumerate(specs):
        if fix is not None:
            i, v = fix
            fixed_mask[k, i] = True
            fixed_u[k, i] = (v - sb.centre[i]) / sb.half[i]
    index = {(p, j): k for k, (p, j, _) in enumerate(specs) if p is not None}
    neighbours = [list(range(1, K))] + [
        [index[(p, jj)] for jj in (j - 1, j + 1) if (p, jj) in index] + [0]
        for p, j, _ in specs[1:]]
    if surface_indices is not None:
        neighbours = [list(range(1, K))] + [
            [index[q] for q in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)) if q in index] + [0]
            for i, j, _ in specs[1:]]
    init = np.empty((K * W, N))
    for k in range(K):
        init[k * W] = u_anchor
        init[k * W + 1: (k + 1) * W] = u_anchor + cfg.step0 * (rng.standard_normal((W - 1, N)) @ geometry.T)
    log(f"profile hopping: {K - 1} nodes + anchor, {W} walkers each", flush=True)
    best_f, best_U, hop_info = hop(sb, fixed_mask, fixed_u, init, neighbours, geometry, cfg, rng, log)
    if af[0] > best_f[0]:
        best_f[0], best_U[0] = af[0], u_anchor

    # Certify every incumbent; if a node beats the anchor, re-optimise the anchor from it.
    log("certification", flush=True)
    res, second = certify(sb, best_U, fixed_mask, gtol=cfg.gtol, trust=cfg.trust, max_iter=cfg.max_iter,
                          gain_tol=cfg.gain_tol)
    anchor_updates = reoptimise_anchor(sb, res, second, N, cfg)
    consistent = bool(np.max(res.f[1:]) <= res.f[0] + cfg.value_tol)

    def diagnostic(k):
        return dict(source="hybrid", logl=float(res.f[k]),
                    projected_gradient_inf=float(res.pg[k]),
                    stationary=bool(second["stationary"][k]),
                    remaining_gain=(float(second["remaining_gain"][k])
                                    if np.isfinite(second["remaining_gain"][k]) else None),
                    local_max_certified=bool(second["local_max"][k]),
                    lambda_min=float(second["lambda_min"][k]),
                    lambda_max=float(second["lambda_max"][k]),
                    saddle_escapes=int(second["escapes"][k]),
                    newton_restarts=int(second["newton_restarts"][k]),
                    negative_curvature_unresolved=bool(second["negative_curvature_unresolved"][k]),
                    ascent_status=str(res.status[k]), certify_iterations=int(res.iterations[k]),
                    incumbents_found=int(hop_info["incumbents_found"][k]),
                    acceptance=float(hop_info["acceptance"][k]))

    X = sb.physical(res.U)
    for k, (_, _, fix) in enumerate(specs):
        if fix is not None:
            X[k, fix[0]] = fix[1]
    if surface_indices is None:
        profiles = {}
        for p, (name, grid) in enumerate(zip(names, grids)):
            ks = [index[(p, j)] for j in range(len(grid))]
            profiles[name] = dict(grid=np.asarray(grid, float), logl=res.f[ks], xopt=X[ks],
                                  optimization=[diagnostic(k) for k in ks])
    stationary = bool(np.all(second["stationary"]))
    quality = dict(
        method="hybrid basin hopping + projected BFGS",
        hops=hop_info["hops"], profiles_stable=hop_info["stable"], anchor_consistent=consistent,
        anchor_updates=anchor_updates, all_selected_points_stationary=stationary,
        stationary_by_gradient=int(np.sum(res.pg <= cfg.gtol)),
        stationary_by_remaining_gain=int(np.sum(second["stationary"] & (res.pg > cfg.gtol))),
        all_local_max_certified=bool(np.all(second["local_max"])),
        saddle_escapes=int(np.sum(second["escapes"])),
        converged=bool(consistent and stationary and hop_info["stable"]),
        global_optimum_certified=False, value_tolerance=cfg.value_tol,
        anchor=diagnostic(0), geometry=geo_info, configuration=asdict(cfg),
        evaluations=dict(sb.counts), evaluation_seconds=dict(sb.seconds),
        seconds=time.monotonic() - t0)
    if surface_indices is not None:
        shape = (len(grids[0]), len(grids[1]))
        quality.update(stable=hop_info["stable"], stationary=stationary)
        result = dict(axes=[np.asarray(a) for a in grids], indices=list(surface_indices),
                      names=list(names), full_parameter_names=params,
                      logl=res.f[1:].reshape(shape), xopt=X[1:].reshape(*shape, N),
                      diagnostics=[[diagnostic(index[i, j]) for j in range(shape[1])]
                                   for i in range(shape[0])],
                      best_logl=float(res.f[0]), best_x=X[0], anchor_diagnostics=diagnostic(0),
                      quality=quality, objective="native")
        return result, sb
    result = dict(names=list(names), full_parameter_names=params, best_logl=float(res.f[0]),
                  best_x=X[0], profiles=profiles, optimization_quality=quality,
                  objective="native", grid_fractions={n: 1.0 for n in names})
    return result, sb


# --------------------------------------------------------------------------- true stage


class ConcurrentLedger:
    """Thread-safe front end to an EvaluationLedger (or WorkflowLedger).

    Bookkeeping and journal writes happen under one lock; the backend call does not, so
    several nodes wait on the backend pool at once. Requests for a coordinate already in
    flight wait for that call instead of repeating it.

    The journal is written through one append handle, flushed after every event (the
    same durability as EvaluationLedger's open/append/close per event): on this machine
    each open() costs ~50 ms, which under the lock would serialise all nodes. A
    WorkflowLedger's count summary is refreshed at most every `persist_seconds`, and on
    close().
    """

    def __init__(self, ledger, persist_seconds=5.0, workers=1):
        import json
        from concurrent.futures import ThreadPoolExecutor

        self.ledger = ledger
        self.bounds = ledger.bounds
        self.lock = threading.Lock()
        self.inflight = {}
        self.refused = set()         # points that returned no value: waiters and later requests do not call again
        self.persist_seconds = persist_seconds
        self._last_persist = time.monotonic()
        self._handle = open(ledger.journal, "a")
        # Runs the backend calls of call_many; `workers` matches the backend pool size.
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(workers)))

        def event(**fields):
            self._handle.write(json.dumps(dict(time=time.time(), stage=ledger.stage, **fields),
                                          allow_nan=False) + "\n")
            self._handle.flush()

        ledger._event = event  # instance override; the class method is untouched

    def close(self):
        self._executor.shutdown(wait=True)
        with self.lock:
            del self.ledger._event
            self._handle.close()
            if hasattr(self.ledger, "persist"):
                self.ledger.persist()

    def _maybe_persist(self):
        if hasattr(self.ledger, "persist") and time.monotonic() - self._last_persist >= self.persist_seconds:
            self.ledger.persist()
            self._last_persist = time.monotonic()

    def _begin(self, x, stage):
        """Under the lock, resolve a request: ("done", value, kind), ("wait", event) for a
        coordinate already in flight, or ("run", key, number) after journaling its start."""
        from profiles.evaluation import CallBudgetExceeded

        L = self.ledger
        with self.lock:
            L.stage = stage
            L._count("requests")
            if (x.shape != (len(self.bounds),) or not np.isfinite(x).all()
                    or np.any(x < self.bounds[:, 0]) or np.any(x > self.bounds[:, 1])):
                L._count("invalid_requests")
                L._event(event="invalid_request")
                return ("done", -np.inf, "invalid")
            key = L._key(x)
            if L.deterministic and key in L.cache:
                L._count("cache_hits")
                L._event(event="cache_hit", x=x.tolist(), value=L.cache[key])
                return ("done", L.cache[key], "cache")
            if L.deterministic and key in self.refused:
                L._count("cache_hits")
                L._event(event="cache_hit", x=x.tolist(), value=None)
                return ("done", -np.inf, "cache")
            waiting = self.inflight.get(key)
            if waiting is not None:
                L._count("coalesced_requests")
                return ("wait", waiting)
            if L.counts["backend_calls"] >= L.max_calls:
                L._count("budget_rejections")
                raise CallBudgetExceeded("total true-likelihood call budget reached")
            L._count("backend_calls")
            number = L.counts["backend_calls"]
            L._event(event="start", call=number, x=x.tolist())
            self.inflight[key] = threading.Event()
            return ("run", key, number)

    def _abandon(self, key, number, stage, reason):
        """Close a journaled start whose backend call will not be made."""
        L = self.ledger
        with self.lock:
            L.stage = stage
            L._count("exceptions")
            L._event(event="exception", call=number, error=reason, seconds=0.0)
            self.inflight.pop(key).set()

    def _run(self, x, stage, key, number):
        L = self.ledger
        start = time.monotonic()
        try:
            value = float(L.objective(x))
        except Exception as exc:
            with self.lock:
                L.stage = stage
                L._count("exceptions")
                L._event(event="exception", call=number, error=repr(exc), seconds=time.monotonic() - start)
                self.inflight.pop(key).set()
            raise
        with self.lock:
            L.stage = stage
            finite = bool(np.isfinite(value))
            L._count("finite_returns" if finite else "nonfinite_returns")
            L._event(event="return", call=number, value=value if finite else None,
                     seconds=time.monotonic() - start)
            if finite and L.deterministic:
                L.cache[key] = value
            elif L.deterministic:
                self.refused.add(key)
            self.inflight.pop(key).set()
            self._maybe_persist()
        return (value if finite else -np.inf), "backend"

    def call(self, x, stage):
        x = np.asarray(x, float)
        while True:
            state = self._begin(x, stage)
            if state[0] == "done":
                return state[1], state[2]
            if state[0] == "wait":
                state[1].wait()  # then retry: the value is cached unless that call failed
                continue
            return self._run(x, stage, *state[1:])

    def call_many(self, points, stage):
        """Evaluate points concurrently on the executor, returning [(value, kind)] in order.

        Starts are journaled in the given order before any backend call, so one thread's
        calls replay in a deterministic order. Every started call is closed in the journal
        (returned, failed, or abandoned when a budget stop interrupts the batch), and all
        submitted calls finish before an exception is re-raised.
        """
        points = [np.asarray(x, float) for x in points]
        states, error = [], None
        for x in points:
            try:
                states.append(self._begin(x, stage))
            except Exception as exc:
                error = exc
                break
        if error is not None:
            for state in states:
                if state[0] == "run":
                    self._abandon(state[1], state[2], stage, "not run: " + repr(error))
            raise error
        futures = {i: self._executor.submit(self._run, points[i], stage, *state[1:])
                   for i, state in enumerate(states) if state[0] == "run"}
        results, first_error = [], None
        for i, state in enumerate(states):
            try:
                if state[0] == "done":
                    results.append((state[1], state[2]))
                elif state[0] == "run":
                    results.append(futures[i].result())
                else:
                    results.append(self.call(points[i], stage))
            except Exception as exc:
                first_error = first_error or exc
                results.append((-np.inf, "exception"))
        if first_error is not None:
            raise first_error
        return results


class NodeLedger:
    """The ledger interface the true stage uses, with call counts
    private to one node or one worker thread; `stage` may be reassigned per call."""

    def __init__(self, shared, stage):
        self.shared, self.stage, self.bounds = shared, stage, shared.bounds
        self.counts = Counter()
        self.prior_by_stage = getattr(shared.ledger, 'prior_by_stage', {})
        self.prior_values_by_stage = getattr(shared.ledger, 'prior_values_by_stage', {})

    def missing_calls(self, points):
        with self.shared.lock:
            ledger = self.shared.ledger
            # A duplicate of a miss is counted though it waits for the first: an upper bound.
            return sum(ledger._key(x) not in ledger.cache and ledger._key(x) not in self.shared.refused for x in points)


    def _tally(self, kind):
        self.counts["requests"] += 1
        if kind == "backend":
            self.counts["backend_calls"] += 1
        elif kind == "cache":
            self.counts["cache_hits"] += 1

    def __call__(self, x):
        value, kind = self.shared.call(x, self.stage)
        self._tally(kind)
        return value

    def call_many(self, points):
        results = self.shared.call_many(points, self.stage)
        for _, kind in results:
            self._tally(kind)
        return [value for value, _ in results]
