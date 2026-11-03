"""Surrogate-corrected true-likelihood profile refinement, residual Newton method.

The true stage of every profile study.

Around an incumbent u_b (box coordinates, d = u - u_b) the truth is modelled as

    t(u) ~ s(u) + r(u_b) + g . d + 0.5 d^T Q d,

with s the network (all its nonlinearity kept), g the gradient of the residual r = t - s,
measured by finite differences, and Q a correction to the network's curvature, learned and
carried along the profile. Coordinates are whitened by the network's local curvature
(sigma capped at `max_step` box half-widths, as profiles.hybrid.curvature_geometry).

Per node (profiled coordinate fixed):
  0. Start at the surrogate optimum; its true value is the workflow's direct call.
  1. Transfer: predict g from the previous nodes of the chain (the residual gradient at an
     optimum of t is exactly -grad s there; extrapolated linearly along the profile) and
     spend one call at the model's maximum if it predicts more than `transfer_min`. A
     transfer that fails pauses transfers for two nodes.
  2. Gradient round, one concurrent batch: forward differences of t - s, `step` sigmas along
     the network's curvature eigenvectors (capped at `fd_max_box`), inward along the axis
     for every free coordinate on a bound (even if the predicted gradient is outward), and
     central differences of `flat_step` sigmas along the `flat_central` flattest
     directions, which also measure the curvature there (written into Q). g is the
     whitened least-squares fit, ridged towards the prediction.
  3. Trust-region step on the model (radius starts at `trust0` sigmas; the network is least
     reliable along its flattest direction). One call. Q is corrected along the step from
     the actual/predicted mismatch, and by a PSB secant update when the next gradient round
     is `secant_min` sigmas or more away. A good step to the edge of the region steps again
     with the gradient carried by Q; a failed one shrinks the region, and a failure within
     `escalate` sigmas mirrors every probe (central differences, full diagonal curvature).
  4. Stop when a freshly measured gradient predicts at most `tol`, or after a good step when
     the curvature error bound 0.5 (L * distance since the gradient round)^2 is below `tol`
     (L: the model's curvature error, floored at `curvature`, learned from secant and step
     mismatches beyond the noise, carried along the chain with factor `L_memory`).

Each profile starts at its node nearest the surrogate's best fit, with no prior, and runs
two chains outward from it; a node learns only from its predecessors in its own chain, so
results do not depend on timing or on `workers`. The global best fit is solved last, from
the best node with every coordinate free. On resume, finished nodes are loaded from their
checkpoints (each holds the state its successor needs) and an interrupted node is re-run;
its journaled calls come back from the ledger's cache, so no call is repeated.

The reported node value is always an evaluated true likelihood (a lower bound on the
profile). Stopping is model-based and does not certify accuracy.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import time

import numpy as np

# Consecutive steps served from the ledger's cache (no budget spent) after which a node stops
# stepping with its current gradient. A module constant, not a NewtonConfig field, so saved
# checkpoint signatures (which include the configuration) are unchanged. Finished studies had
# at most a few cache hits per node.
FREE_STEPS_MAX = 50


def _recorded(gain):
    """A step's gain as the node's history holds it: None when the true value there was not
    finite. The history is a record only, and a checkpoint is JSON without infinities: a node
    with one such step could not be saved, and every resume stopped at the same call."""
    gain = float(gain)
    return gain if np.isfinite(gain) else None


@dataclass
class NewtonConfig:
    tol: float = 1e-2  # log L: predicted remaining gain at which a node stops (a few times
    # the gain that gradient noise alone predicts: CLASS++ values scatter by ~1e-3 - 2.5e-3)
    step: float = 0.25  # forward-difference step, local sigmas
    trust0: float = 0.5  # initial trust radius, local sigmas (grows on success). The network's
    # flattest direction is its least reliable: on the SINU TT profiles it is the tau_reio-A_s
    # degeneracy, and 1 sigma along it reaches the tau prior edge, where the truth drops by
    # tens of log L the network does not know about.
    trust_max: float = 8.0
    trust_min: float = 0.05
    curvature: float = 0.2  # floor on L, the model's curvature error relative to C (the SINU
    # networks' real error is 0.3 - 0.5; 0.1 let a node stop 0.1 log L short after a 1-sigma move)
    noise: float = 2.5e-3  # log L allowance for true-likelihood noise in mismatches
    transfer_min: float = 1.5e-2  # log L: predicted transfer gain worth one call
    ridge: float = 1e-4  # whitened ridge towards the predicted gradient, relative to the probes
    secant: bool = True  # learn and carry the curvature correction Q
    q_max: float = 0.9  # largest flattening by Q relative to the local surrogate curvature
    q_stiff: float = 50.0  # largest stiffening by Q (only makes steps more cautious)
    fd_max_box: float = 0.1  # cap on a forward-difference step, box half-widths
    flat_step: float = 0.5  # central-difference step along the flattest directions, sigmas
    flat_max_box: float = 0.25  # its cap, box half-widths (curvature needs a long baseline)
    secant_min: float = 0.5  # shortest move (sigmas) whose gradient change updates Q
    escalate: float = 0.5  # a step failing within this many sigmas triggers central differences
    flat_central: int = 2  # flattest surrogate directions probed with central differences
    max_model_steps: int = 5  # successful boundary steps before the gradient is re-measured
    L_memory: float = 0.8  # factor on the previous node's L carried to the next
    extrapolate: bool = True  # transfer with the chain's extrapolated gradient
    max_step: float = 0.5  # sigma cap in box half-widths (as curvature_geometry)
    maxcalls: int = 150
    max_rounds: int = 12


class QuadCorrected:
    """s(u) + g . d + 0.5 d^T Q d, d = u - u0, with BatchedSurrogate's API."""

    def __init__(self, sb, g, Q, u0):
        self.sb, self.g, self.Q, self.u0, self.n = sb, np.asarray(g, float), np.asarray(Q, float), np.asarray(u0, float), sb.n

    def _corr(self, U):
        D = np.atleast_2d(U) - self.u0
        QD = D @ self.Q
        return D @ self.g + 0.5 * np.sum(D * QD, axis=1), self.g + QD

    def value(self, U):
        return self.sb.value(np.atleast_2d(U)) + self._corr(U)[0]

    def value_grad(self, U):
        f, g = self.sb.value_grad(np.atleast_2d(U))
        c, cg = self._corr(U)
        return f + c, g + cg

    def value_grad_hess(self, U):
        f, g, H = self.sb.value_grad_hess(np.atleast_2d(U))
        c, cg = self._corr(U)
        return f + c, g + cg, H + self.Q


class Frame:
    """Local whitening on the probed coordinates P at one point: z = K d_P, d_P = T z."""

    def __init__(self, sb, u, P, floor):
        _, gs, H = sb.value_grad_hess(u[None])
        self.gs, self.P = gs[0], P
        Hp = H[0][np.ix_(P, P)]
        lam, V = np.linalg.eigh(-0.5 * (Hp + Hp.T))
        lam = np.maximum(np.abs(lam), floor)
        self.lam = lam
        self.T = V / np.sqrt(lam)
        self.K = (V * np.sqrt(lam)).T

    def norm(self, d):
        return float(np.linalg.norm(self.K @ d[self.P]))


class NewtonRefiner:
    def __init__(self, sb, ledger, config=None, log=print):
        self.sb, self.ledger, self.cfg, self.log = sb, ledger, config or NewtonConfig(), log
        self.floor = 1.0 / self.cfg.max_step ** 2

    def _physical(self, u):
        x = self.sb.physical(u)
        if getattr(self, "_fixed_indices", None) is not None:
            x[self._fixed_indices] = self._fixed_values
        return x

    def call(self, u, stage):
        self.ledger.stage = stage
        return float(self.ledger(self._physical(u)))

    def call_many(self, us, stage):
        self.ledger.stage = stage
        xs = [self._physical(u) for u in us]
        if hasattr(self.ledger, "call_many"):
            return [float(t) for t in self.ledger.call_many(xs)]
        return [float(self.ledger(x)) for x in xs]

    def solve(self, u_start, fixed_index=None, fixed_value=None, stage="node", prior=None, t_start=None):
        """Return (native value, x, diagnostics, successor state).

        fixed_index/value accept scalars or matching arrays. Multi-fixed chains
        vary only the last coordinate; preceding fixed values must remain constant.
        """
        from profiles.hybrid import ascend

        cfg, sb, N = self.cfg, self.sb, self.sb.n
        fixed = np.zeros(N, bool)
        u0 = np.clip(np.asarray(u_start, float), -1, 1)
        self._fixed_indices = self._fixed_values = None
        if fixed_index is not None:
            ids = np.atleast_1d(fixed_index)
            vals = np.atleast_1d(fixed_value).astype(float)
            if (ids.dtype.kind not in "iu" or ids.ndim != 1 or len(ids) == 0
                    or len(set(ids.tolist())) != len(ids) or np.any(ids < 0) or np.any(ids >= N)
                    or vals.shape != ids.shape or not np.isfinite(vals).all()
                    or np.any(vals < getattr(sb, "lo", sb.centre - sb.half)[ids])
                    or np.any(vals > getattr(sb, "hi", sb.centre + sb.half)[ids])):
                raise ValueError("invalid fixed coordinates or values")
            self._fixed_indices, self._fixed_values = ids, vals
            fixed_index = int(ids[0]) if np.ndim(fixed_index) == 0 else ids
            fixed_value = np.asarray(fixed_value, float)
            fixed[fixed_index] = True
            u0[fixed_index] = (fixed_value - sb.centre[fixed_index]) / sb.half[fixed_index]
        F = ~fixed
        calls0 = self.ledger.counts["backend_calls"]
        previous_attempts = getattr(self.ledger, 'prior_by_stage', {}).get(stage, 0)
        # A torn record without a known stage is reserved once by WorkflowLedger's
        # lifetime budget. It cannot be charged to every subsequent node as well.
        uncertain_attempts = 0
        used = lambda: previous_attempts + self.ledger.counts["backend_calls"] - calls0
        def needed(us):
            xs = [self._physical(u) for u in us]
            if not previous_attempts:
                return len(xs)  # preserve the validated fresh-run probe policy
            if hasattr(self.ledger, 'missing_calls'):
                return self.ledger.missing_calls(xs)
            if getattr(self.ledger, 'deterministic', False):
                return sum(self.ledger._key(x) not in self.ledger.cache for x in xs)
            return len(xs)

        prior = prior or {}
        path = [(float(c), np.asarray(g, float)) for c, g in prior.get("path", [])]
        # Multi-fixed nodes transfer along the last fixed coordinate. Their driver must
        # keep all preceding fixed values constant within each continuation chain.
        coord = float(u0[self._fixed_indices[-1]]) if fixed_index is not None else 0.0
        if path and cfg.extrapolate and len(path) >= 2 and fixed_index is not None:
            (c1, g1), (c2, g2) = path[-2], path[-1]
            # Linear extrapolation along the optimum path, only from a baseline at least half
            # as long as the extrapolation (otherwise the previous gradient as it is).
            ok = abs(c2 - c1) >= 0.5 * abs(coord - c2) and c2 != c1
            g_pred = g2 + (g2 - g1) * (coord - c2) / (c2 - c1) if ok else g2
        elif path:
            g_pred = path[-1][1]
        else:
            g_pred = np.zeros(N)
        g_pred = g_pred * F
        Q = np.asarray(prior.get("Q", np.zeros((N, N))), float) if cfg.secant else np.zeros((N, N))
        Q = Q * np.outer(F, F)
        L_prior = float(prior.get("L", 0.0))
        L_node = []
        counts = dict(transfer=0, gradient_rounds=0, gradient_calls=0, steps=0, failed_steps=0)
        history = []

        def pin(u):
            return np.where(fixed, u0, np.clip(u, -1, 1))

        def L_now():
            return max(cfg.curvature, L_prior, max(L_node, default=0.0))

        def trust_step(g, u_b, mask, trust, frame):
            """Maximise the model within `trust` local sigmas of u_b (coordinates in mask held)."""
            model = QuadCorrected(sb, g, Q, u_b)
            uA = pin(ascend(model, u_b[None], mask[None], gtol=1e-7, max_iter=300).U[0])
            nA = frame.norm(uA - u_b)
            cands = [uA if nA <= trust else pin(u_b + (uA - u_b) * trust / nA)]
            _, gm, Hm = model.value_grad_hess(u_b[None])
            P = frame.P & ~mask
            sub = P[frame.P]
            T = frame.T[sub]
            gz = T.T @ gm[0][P]
            Bz = -(T.T @ Hm[0][np.ix_(P, P)] @ T)
            w, E = np.linalg.eigh(0.5 * (Bz + Bz.T))
            w = np.maximum(w, 0.05)
            ge = E.T @ gz
            # Gain of the local quadratic model's unconstrained Newton step: what the
            # measured gradient says is left, whatever the trust radius.
            pred_q = 0.5 * float(np.sum(ge ** 2 / w))
            if nA > trust:
                lo, hi = 0.0, float(np.linalg.norm(ge)) / trust + 1.0
                if np.linalg.norm(ge / w) > trust:
                    for _ in range(100):
                        mu = 0.5 * (lo + hi)
                        lo, hi = (mu, hi) if np.linalg.norm(ge / (w + mu)) > trust else (lo, mu)
                    mu = hi
                else:
                    mu = 0.0
                d = np.zeros(N)
                d[P] = T @ (E @ (ge / (w + mu)))
                cands.append(pin(u_b + d))
            m = model.value(np.vstack([u_b] + cands))
            k = int(np.argmax(m[1:]))
            u = cands[k]
            return u, float(m[1 + k] - m[0]), frame.norm(u - u_b), pred_q

        def update_Q_along(d, frame, excess):
            """Rank-one correction of Q along d so the model reproduces the observed change."""
            if not cfg.secant:
                return
            P = frame.P
            z = frame.K @ d[P]
            n2 = float(z @ z)
            c = float(np.clip(2 * excess / n2, -cfg.q_stiff, cfg.q_max))
            Qz = frame.T.T @ Q[np.ix_(P, P)] @ frame.T + c * np.outer(z, z) / n2
            Q[np.ix_(P, P)] = frame.K.T @ Qz @ frame.K

        def update_Q_secant(s, y, frame):
            """PSB: least change of Q (whitened Frobenius norm) with Q s = y on P."""
            P = frame.P
            sz = frame.K @ s[P]
            yz = frame.T.T @ y[P]
            Qz = frame.T.T @ Q[np.ix_(P, P)] @ frame.T
            r = yz - Qz @ sz
            ss = float(sz @ sz)
            # What two forward-difference gradients scatter by from value noise alone.
            noise_y = 2 * cfg.noise * np.sqrt(len(sz)) / cfg.step
            error = float(np.sqrt(max(0.0, float(r @ r) - noise_y ** 2) / ss))
            if cfg.secant:
                Qz = Qz + (np.outer(r, sz) + np.outer(sz, r)) / ss - (r @ sz) * np.outer(sz, sz) / ss ** 2
                Q[np.ix_(P, P)] = frame.K.T @ Qz @ frame.K
            return error

        def clamp_Q(frame):
            """Bound Q's eigenvalues, relative to the local surrogate curvature, to +-q_max:
            the correction is learned at other points, where the network's own curvature
            along a direction can differ by orders of magnitude."""
            P = frame.P
            Qz = frame.T.T @ Q[np.ix_(P, P)] @ frame.T
            w, E = np.linalg.eigh(0.5 * (Qz + Qz.T))
            Qz = (E * np.clip(w, -cfg.q_stiff, cfg.q_max)) @ E.T
            Q[np.ix_(P, P)] = frame.K.T @ Qz @ frame.K

        if t_start is None and used() + needed([u0]) > cfg.maxcalls:
            from profiles.evaluation import CallBudgetExceeded
            raise CallBudgetExceeded("node lifetime budget exhausted before a usable start")
        t0 = self.call(u0, stage) if t_start is None else float(t_start)
        u_best, t_best = u0.copy(), t0
        g_full = g_pred.copy()
        # 1. Transfer.
        if np.any(Q):
            clamp_Q(Frame(sb, u0, F, self.floor))
        skip = int(prior.get("transfer_skip", 0))
        transfer_skip = max(0, skip - 1)
        if (np.any(g_pred) or np.any(Q)) and not skip and used() < cfg.maxcalls:
            frame = Frame(sb, u0, F, self.floor)
            try:
                u_T, pred, norm, _ = trust_step(g_pred, u0, fixed, cfg.trust0, frame)
            except np.linalg.LinAlgError:
                pred = -np.inf  # the inherited model cannot be maximised here: no transfer
            if pred > cfg.transfer_min:
                t_T = self.call(u_T, stage)
                counts["transfer"] += 1
                history.append(dict(kind="transfer", predicted=pred, actual=_recorded(t_T - t0), norm=norm))
                if t_T - t0 < 0.25 * pred:
                    transfer_skip = 2  # the chain's transfer is not holding here: pause it
                if t_T > t_best:
                    u_best, t_best = u_T, t_T
                    g_full = g_pred + (Q @ (u_T - u0))
        reason, trust = "budget", cfg.trust0
        last = None  # (centre, gradient, probed set) of the previous gradient round

        def probe_points(u_b, P, edge_lo, edge_hi, frame):
            """Forward-difference points, and the mirror of each (for central differences)."""
            edge = P & (edge_lo | edge_hi)  # on a bound, pulled inward: probe inward along the axis
            inner = P & ~edge
            pts, mirrors, flat = [], [], []
            cap = lambda d, c=cfg.fd_max_box: d * min(1.0, c / max(np.abs(d).max(), 1e-300))
            if inner.any():
                fi = Frame(sb, u_b, inner, self.floor)
                flat = list(np.argsort(fi.lam)[:cfg.flat_central])  # eigh order: flattest first
                for i in range(int(inner.sum())):
                    d = np.zeros(N)
                    if i in flat:
                        d[inner] = cfg.flat_step * fi.T[:, i]
                        d = cap(d, cfg.flat_max_box)
                    else:
                        d[inner] = cfg.step * fi.T[:, i]
                        d = cap(d)
                    up, um = pin(u_b + d), pin(u_b - d)
                    first = frame.norm(up - u_b) >= 0.999 * frame.norm(um - u_b)
                    pts.append(up if first else um)
                    mirrors.append(um if first else up)
            H = sb.value_grad_hess(u_b[None])[2][0]
            for e in np.flatnonzero(edge):
                sigma = min(cfg.max_step, 1 / np.sqrt(max(abs(H[e, e]), self.floor)))
                d = np.zeros(N)
                d[e] = cfg.step * sigma * (1.0 if edge_lo[e] else -1.0)
                d = cap(d)
                pts.append(pin(u_b + d))
                mirrors.append(pin(u_b + 2 * d))  # one-sided second difference
            return pts, mirrors, flat

        def fit(u_b, t_b, pts, vals, frame, g_prior, dirs=None):
            """Residual gradient (and, with `dirs`, curvature along each direction) by whitened
            least squares on t - s, ridged towards g_prior."""
            P = frame.P
            ok = np.isfinite(vals)
            g_new = g_prior.copy()
            if not ok.any():
                return g_new, None
            pts = np.array(pts)[ok]
            Z = (pts - u_b)[:, P] @ frame.K.T
            dr = (vals[ok] - sb.value(pts)) - (t_b - float(sb.value(u_b[None])[0]))
            nP = Z.shape[1]
            alpha = cfg.ridge * cfg.step ** 2
            if dirs is None:
                A, b, prior_vec = Z, dr, frame.T.T @ g_prior[P]
                reg = alpha * np.ones(nP)
            else:
                A = np.hstack([Z, 0.5 * (Z @ dirs.T) ** 2])
                b = dr
                Qz = frame.T.T @ Q[np.ix_(P, P)] @ frame.T
                prior_vec = np.concatenate([frame.T.T @ g_prior[P], np.einsum("ij,jk,ik->i", dirs, Qz, dirs)])
                reg = np.concatenate([alpha * np.ones(nP), alpha * cfg.step ** 2 * np.ones(len(dirs))])
            sol = np.linalg.solve(A.T @ A + np.diag(reg), A.T @ b + reg * prior_vec)
            g_new[P] = frame.K.T @ sol[:nP]
            return g_new, (None if dirs is None else sol[nP:])

        def set_curvature(dirs, c, frame):
            """Make Q's curvature along each whitened unit direction equal the measured one."""
            if not cfg.secant:
                return
            P = frame.P
            Qz = frame.T.T @ Q[np.ix_(P, P)] @ frame.T
            for v, ci in zip(dirs, c):
                Qz = Qz + (np.clip(ci, -cfg.q_stiff, cfg.q_max) - v @ Qz @ v) * np.outer(v, v)
            Q[np.ix_(P, P)] = frame.K.T @ Qz @ frame.K

        # One node's linear algebra failing (numpy raises LinAlgError for a singular system and
        # for one with a NaN in it: the axion run's (A_osc, omega) and (A_osc, phi) studies, one
        # node each at large amplitude, in the ascent of the corrected model) ends that node, not
        # the study: the best true point it has found stands, and its measurements are not
        # passed on along its chain. Nothing changes for a node that does not fail.
        singular = False
        try:
            while used() < cfg.maxcalls and counts["gradient_rounds"] < cfg.max_rounds:
                u_b, t_b = u_best.copy(), t_best
                if last is not None and np.array_equal(u_b, last[0]):
                    reason = "stalled"  # no step from this gradient succeeded
                    break
                # 2. Gradient round.
                gs = sb.value_grad(u_b[None])[1][0]
                edge_lo, edge_hi = u_b <= -1 + 1e-9, u_b >= 1 - 1e-9
                # A surrogate or inherited residual slope cannot establish an active bound.
                # Probe every free bound coordinate inward at this centre before holding it.
                # The physical bounds and the profiled coordinate remain unchanged.
                held = np.zeros(N, bool)
                P = F.copy()
                if not P.any():
                    reason = "converged"
                    break
                frame = Frame(sb, u_b, P, self.floor)
                points, mirrors, flat = probe_points(u_b, P, edge_lo, edge_hi, frame)
                # The network's curvature is least reliable along its flattest directions (for
                # the SINU TT profiles: tau_reio): probe those both ways, so the model's
                # curvature there is measured rather than assumed.
                extra = [mirrors[i] for i in flat]
                if used() + needed(points + extra) > cfg.maxcalls:
                    break
                values = np.array(self.call_many(points + extra, stage))
                counts["gradient_rounds"] += 1
                counts["gradient_calls"] += len(points) + len(extra)
                for u, t in zip(points + extra, values):
                    if t > t_best:
                        u_best, t_best = u, float(t)
                # Failed inward probes cannot establish that a bound is active.
                n_edge = int(np.sum(P & (edge_lo | edge_hi)))
                if n_edge and not np.isfinite(values[len(points)-n_edge:len(points)]).all():
                    reason = "unverified_bound"
                    break
                if not np.isfinite(values).all():
                    reason = "unverified_interior"
                    break
                if extra:
                    Zf = (np.array([points[i] for i in flat]) - u_b)[:, P] @ frame.K.T
                    fdirs = Zf / np.maximum(np.linalg.norm(Zf, axis=1, keepdims=True), 1e-300)
                    g_new, c = fit(u_b, t_b, points + extra, values, frame, g_full, fdirs)
                    set_curvature(fdirs, c, frame)
                    history.append(dict(kind="flat", curvature=[float(v) for v in c]))
                else:
                    g_new, _ = fit(u_b, t_b, points, values, frame, g_full)
                values = values[:len(points)]
                clamp_Q(frame)
                if last is not None and np.array_equal(last[2], P):
                    s = u_b - last[0]
                    if frame.norm(s) > cfg.secant_min:
                        err = update_Q_secant(s, g_new - last[1], frame)
                        clamp_Q(frame)
                        L_node.append(err)
                        history.append(dict(kind="secant", error=err, norm=frame.norm(s)))
                last, g_full = (u_b.copy(), g_new.copy(), P.copy()), g_new
                # A coordinate on a bound that the measured gradient pushes outward is held too.
                g_meas = gs + g_full
                held = held | (F & ((edge_lo & (g_meas < 0)) | (edge_hi & (g_meas > 0))))
                central, model_steps = False, 0
                fd_frame = frame  # metric at the gradient's centre, for the distance moved since
                failed_at, free = None, 0
                if used() >= cfg.maxcalls:
                    # The verdict of a fresh gradient costs no call, but it is reached inside the
                    # loop below, which a node whose round ended exactly at its budget never enters:
                    # it was reported out of budget (flag 1) and gave its successor a state without
                    # the converged gradient. The same test as the loop's first pass, and nothing else.
                    u_N, pred, norm, pred_q = trust_step(g_full, u_b, fixed | held, trust, frame)
                    if pred <= cfg.tol and pred_q <= cfg.tol and t_best - t_b <= cfg.tol:
                        history.append(dict(kind="round", predicted=pred, norm=norm, trust=trust, newton=pred_q))
                        reason = "converged"
                # 3./4. Steps with this gradient.
                while used() < cfg.maxcalls:
                    previous_failure, failed_at = failed_at, None
                    u_N, pred, norm, pred_q = trust_step(g_full, u_b, fixed | held, trust, frame)
                    history.append(dict(kind="round", predicted=pred, norm=norm, trust=trust, newton=pred_q))
                    if pred <= cfg.tol:
                        # Converged only on a freshly measured gradient whose own Newton gain is
                        # small; a small prediction from a collapsed region or a carried-along
                        # gradient means: shrink, or measure again.
                        if model_steps == 0 and pred_q <= cfg.tol:
                            if t_best - t_b <= cfg.tol:
                                reason = "converged"
                            # An improving probe takes precedence over a stationary model:
                            # remeasure at that incumbent rather than certify this centre.
                            break
                        if pred_q > cfg.tol and trust / 4 >= cfg.trust_min:
                            trust /= 4
                            continue
                        trust = cfg.trust0
                        if model_steps:
                            u_best, t_best = (u_b, t_b) if t_b >= t_best else (u_best, t_best)
                        break
                    if (previous_failure is not None and trust == previous_failure[1]
                            and np.array_equal(u_N, previous_failure[0]) and np.array_equal(Q, previous_failure[2])):
                        # The last step failed and left the state unchanged, and the radius did not
                        # shrink: a step pinned to a bound can be longer than the radius in the
                        # whitened metric, and norm / 4 below then returns the same radius. The same
                        # point would follow, served from the ledger's cache at no cost, without end.
                        # Only in this state shrink the radius itself. A run that ended never reached
                        # it (the old rule repeats it forever), so no finished result changes.
                        history[-1].update(repeat=True)
                        trust /= 4
                        if trust < cfg.trust_min:
                            trust = cfg.trust0
                            break
                        continue
                    before, proposal = used(), (u_N.copy(), trust, Q.copy())
                    t_N = self.call(u_N, stage)
                    # Safety net for any other cycle of cached steps: they spend no budget.
                    free = 0 if used() > before else free + 1
                    if free > FREE_STEPS_MAX:
                        history[-1].update(actual=_recorded(t_N - t_b), repeat=True)
                        trust = cfg.trust0
                        break
                    counts["steps"] += 1
                    actual = t_N - t_b
                    history[-1].update(actual=_recorded(actual))
                    if t_N > t_best:
                        u_best, t_best = u_N, t_N
                    excess = actual - pred
                    slack = 2 * cfg.noise + 0.05 * norm
                    success = actual > 0 and actual >= 0.25 * pred
                    # A step without a finite value says nothing about the curvature: as an excess
                    # of -inf it wrote the stiffest curvature allowed along the step into Q, and
                    # the node, and those started from its state, ran out of budget.
                    if np.isfinite(actual) and norm > 0.3 and abs(excess) > slack:
                        if success:
                            L_node.append(2 * (abs(excess) - slack) / norm ** 2)
                        update_Q_along(u_N - u_b, frame, excess)
                        clamp_Q(frame)
                    if success:
                        boundary_step = norm >= 0.9 * trust
                        if actual >= 0.75 * pred and boundary_step:
                            trust = min(2 * trust, cfg.trust_max)
                            if model_steps < cfg.max_model_steps and t_N >= t_best:
                                # A good step to the edge of the region: the model still holds,
                                # so step again from there with the residual gradient carried
                                # along by Q, before paying for a new gradient round.
                                g_full = g_full + Q @ (u_N - u_b)
                                u_b, t_b = u_N.copy(), t_N
                                frame = Frame(sb, u_b, P, self.floor)
                                model_steps += 1
                                counts["model_steps"] = counts.get("model_steps", 0) + 1
                                continue
                        # The curvature bound estimates error at a model optimum, not remaining
                        # improvement along a step truncated by the trust radius. If a probe won,
                        # it says nothing about that returned incumbent either. Both cases need
                        # a fresh gradient round at u_best (or retain a budget flag).
                        moved = fd_frame.norm(u_N - last[0])
                        if (not boundary_step and np.array_equal(u_N, u_best)
                                and 0.5 * (L_now() * moved) ** 2 <= cfg.tol):
                            reason = "converged"
                        break
                    counts["failed_steps"] += 1
                    if (norm <= cfg.escalate and not central and model_steps == 0
                            and used() + needed(mirrors) <= cfg.maxcalls):
                        # The model fails close by: the forward differences are suspect (a
                        # direction where the truth is much more curved than the network).
                        # Mirror every probe: central differences give the gradient and the
                        # curvature along each probe direction, which is written into Q.
                        mvals = np.array(self.call_many(mirrors, stage))
                        counts["central_rounds"] = counts.get("central_rounds", 0) + 1
                        counts["gradient_calls"] += len(mirrors)
                        both = points + mirrors
                        Zf = (np.array(points) - u_b)[:, P] @ frame.K.T
                        dirs = Zf / np.maximum(np.linalg.norm(Zf, axis=1, keepdims=True), 1e-300)
                        if not np.isfinite(mvals).all():
                            for u, t in zip(mirrors, mvals):
                                if t > t_best:
                                    u_best, t_best = u, float(t)
                            reason = "unverified_interior"
                            break
                        g_c, c = fit(u_b, t_b, both, np.concatenate([values, mvals]), frame, g_full, dirs)
                        set_curvature(dirs, c, frame)
                        clamp_Q(frame)
                        for u, t in zip(mirrors, mvals):
                            if t > t_best:
                                u_best, t_best = u, float(t)
                        history.append(dict(kind="central", curvature_min=float(np.min(c)), curvature_max=float(np.max(c))))
                        g_full, central, trust = g_c, True, max(norm, 0.5)
                        g_meas = gs + g_full
                        held = F & ((edge_lo & (g_meas < 0)) | (edge_hi & (g_meas > 0)))
                        last = (u_b.copy(), g_c.copy(), P.copy())
                        continue
                    failed_at = proposal
                    trust = norm / 4
                    if trust < cfg.trust_min:
                        trust = cfg.trust0
                        break
                if reason in ("converged", "unverified_interior"):
                    break
        except np.linalg.LinAlgError as error:
            reason, singular = "singular", True
            self.log(f"  {stage}: {error} in the Newton step; the node ends with its best true value so far", flush=True)
        # An exhausted/interrupted replay must not lose the best previously returned point.
        for old_x, old_t in getattr(self.ledger, 'prior_values_by_stage', {}).get(stage, []):
            if old_t > t_best:
                u_best, t_best = sb.box(np.asarray(old_x)), float(old_t)
                reason = "budget"  # no gradient verification at this restored incumbent
        x = sb.physical(u_best)
        coord_final = coord
        if fixed_index is not None:
            x[fixed_index] = fixed_value
        if singular:
            state = dict(path=[(c, g.tolist()) for c, g in path],
                         Q=np.asarray(prior.get("Q", np.zeros((N, N))), float).tolist(),
                         transfer_skip=transfer_skip, L=L_prior)
        else:
            g_state = g_full if last is None else last[1] + Q @ (u_best - last[0])
            if reason == "converged":
                # At an optimum of t, grad r = -grad s exactly: the best estimate of the
                # residual gradient there, independent of where the probes were taken.
                interior = F & (u_best > -1 + 1e-9) & (u_best < 1 - 1e-9)
                g_state = np.where(interior, -sb.value_grad(u_best[None])[1][0], g_state)
            new_path = (path + [(coord_final, g_state)])[-2:] if fixed_index is not None else [(0.0, g_state)]
            state = dict(path=[(c, g.tolist()) for c, g in new_path], Q=Q.tolist(), transfer_skip=transfer_skip,
                         L=max(max(L_node, default=0.0), cfg.L_memory * L_prior))
        diag = dict(reason=reason, flag=0 if reason == "converged" else 1, backend_calls=used() - previous_attempts - uncertain_attempts,
                    lifetime_node_attempts=used() - uncertain_attempts, previous_node_attempts=previous_attempts,
                    uncertain_budget_reservations=uncertain_attempts,
                    residual_curvature=L_now(), trust_radius=trust, history=history, **counts)
        return t_best, x, diag, state


def chain_plan(order, chain_length):
    """Split the nodes on one side of a seed, nearest first, into the seed's own chain and
    further (seed, chain) pieces: at most `chain_length` nodes are solved in sequence from
    one cold start. None keeps a single chain; 0 makes every node a seed."""
    if chain_length is None:
        return list(order), []
    L = int(chain_length)
    return list(order[:L]), [(order[s], list(order[s + 1:s + 1 + L])) for s in range(L, len(order), L + 1)]


def refused_start(x_start, prior, **row):
    """The row of a node whose start the likelihood refuses (a prior edge, a Boltzmann failure):
    no value, and the state it was given goes on along its chain, as from a singular node. Its
    checkpoint keeps it on a resume, since the objective is deterministic."""
    return dict(row, logl=None, x=None, direct_logl=None, state=prior,
                diagnostics=dict(reason="refused start", flag=1, backend_calls=0,
                                 start=np.asarray(x_start, float).tolist()))


def run_seeded_chains(pool, seeds, chains_of, run):
    """Solve every seed at once, then each seed's chains as soon as that seed is done (not
    when every seed is). `run(k, order, state)` returns (k, {j: row}); a chain starts from
    its seed's final state. Returns {(k, j): row}. Which state a node starts from is fixed
    by the plan, so the rows do not depend on completion order."""
    from concurrent.futures import FIRST_COMPLETED, wait

    rows, pending = {}, {}
    for k, j in seeds:
        pending[pool.submit(run, k, [j], None)] = (k, j)
    while pending:
        finished, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in finished:
            origin = pending.pop(future)
            k, out_rows = future.result()
            rows.update({(k, j): row for j, row in out_rows.items()})
            if origin is not None:
                state = out_rows[origin[1]]["state"]
                for order in chains_of.get(origin, []):
                    pending[pool.submit(run, k, order, state)] = None
    return rows


def refine_profiles_newton(model, source, bounds, ledger, out, config=None, log=print, direct=True,
                           workers=1, surrogate=None, nodes=None, chain_length=None):
    """The true stage of a 1D study: stage names `corrected/{direct,node}/NAME/J`. `nodes` (grid
    indices) restricts the refinement to a subset; the others are reported as null.
    `chain_length` caps how many nodes are solved in sequence from one cold start (None:
    two chains per profile, outward from its seed; see chain_plan)."""
    from concurrent.futures import ThreadPoolExecutor
    from profiles.hybrid import BatchedSurrogate, ConcurrentLedger, NodeLedger
    from profiles.io import save

    t0 = time.monotonic()
    bounds = np.asarray(bounds, float)
    names = source["full_parameter_names"]
    directory = Path(out) / "true"
    directory.mkdir(parents=True, exist_ok=True)
    sb = surrogate if surrogate is not None else BatchedSurrogate(model, bounds)
    cfg = config or NewtonConfig()
    def checkpoint(path, compute):
        if path.exists():
            return json.loads(path.read_text())
        save(path, compute())
        return json.loads(path.read_text())

    shared = ConcurrentLedger(ledger, workers=workers)
    try:
        profiled = source["names"]
        ax = np.array(source["best_x"], float)
        plan = []
        for k, name in enumerate(profiled):
            i = names.index(name)
            grid = np.asarray(source["profiles"][name]["grid"], float)
            keep = sorted(set(range(len(grid)) if nodes is None else nodes))
            j0 = min(keep, key=lambda j: (abs(grid[j] - ax[i]), j))
            up = [j for j in keep if grid[j] > grid[j0]]
            down = [j for j in keep[::-1] if grid[j] < grid[j0]]
            plan.append((k, j0, sorted(up, key=lambda j: grid[j]), sorted(down, key=lambda j: -grid[j])))
        rows = {}

        def run(k, order, state):
            """Solve nodes `order` of profile k in sequence, each from its predecessor's state."""
            name, i = profiled[k], names.index(profiled[k])
            grid = np.asarray(source["profiles"][name]["grid"], float)
            refiner = NewtonRefiner(sb, NodeLedger(shared, f"corrected/node/{name}"), cfg, log)
            out_rows = {}
            for j in order:
                x_s = np.array(source["profiles"][name]["xopt"][j], float)
                x_s[i] = grid[j]
                x_s = np.clip(x_s, bounds[:, 0], bounds[:, 1])  # see clip_profile_starts
                prior = state

                def compute():
                    d = None
                    if direct:
                        refiner.ledger.stage = f"corrected/direct/{name}/{j}"
                        d = float(refiner.ledger(x_s))
                        if not np.isfinite(d):
                            log(f"  {name} node {j}: the likelihood refuses its start, the surrogate optimum "
                                f"(x={x_s.tolist()}); no value there", flush=True)
                            return refused_start(x_s, prior, coordinate=float(grid[j]), surrogate_x=x_s)
                    v, x, diag, st = refiner.solve(sb.box(x_s), i, float(grid[j]),
                                                   stage=f"corrected/node/{name}/{j}", prior=prior, t_start=d)
                    return dict(logl=v, x=x, diagnostics=diag, state=st, coordinate=float(grid[j]),
                                direct_logl=d, surrogate_x=x_s)

                row = checkpoint(directory / f"node_{k:03d}_{j:05d}.json", compute)
                state = row["state"]
                out_rows[j] = row
                log(f"  {name} node {j} ({grid[j]:+.2f}): {row['diagnostics']['backend_calls']} calls, "
                    f"{row['diagnostics']['reason']}", flush=True)
            return k, out_rows

        # Each profile starts at its node nearest the surrogate's best fit (no prior), then
        # runs two chains outward from it; with a chain_length, further seeds along the
        # way start further chains (chain_plan). The global best fit is solved last, from
        # the best node, instead of first from the surrogate's best fit: that start is
        # where the network is least trustworthy along the profiled direction, and a
        # profile's own maximum is the better place to begin.
        seeds, chains_of, order_of = [], {}, {}
        for k, j0, up, down in plan:
            seeds.append((k, j0))
            order_of[k] = [j0]
            for side in (up, down):
                own, further = chain_plan(side, chain_length)
                if own:
                    chains_of.setdefault((k, j0), []).append(own)
                order_of[k] += own
                for seed, chain in further:
                    seeds.append((k, seed))
                    if chain:
                        chains_of[(k, seed)] = [chain]
                    order_of[k] += [seed] + chain
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            solved = run_seeded_chains(pool, seeds, chains_of, run)
        for (k, j), row in solved.items():
            rows.setdefault(k, {})[j] = row
        records, directs = {}, {}
        for k, name in enumerate(profiled):
            grid = source["profiles"][name]["grid"]
            these = [rows.get(k, {}).get(j) for j in range(len(grid))]
            pick = lambda key: [None if q is None else q[key] for q in these]
            records[name] = dict(grid=grid, logl=pick("logl"), xopt=pick("x"), optimization=pick("diagnostics"))
            directs[name] = dict(grid=grid, logl=pick("direct_logl"), xopt=source["profiles"][name]["xopt"])
        # Ties go to the first node in plan order (seed, then its chains outward), whatever
        # the completion order was.
        valued = [(rows[k][j]["logl"], rows[k][j]) for k in sorted(order_of) for j in order_of[k]
                  if rows[k][j]["logl"] is not None]
        if not valued:
            raise RuntimeError("the likelihood refuses the start of every node of the true stage: inspect the backend")
        best = max(valued, key=lambda q: q[0])[1]
        final = NewtonRefiner(sb, NodeLedger(shared, "corrected/anchor"), cfg, log)

        def solve_anchor():
            path = best["state"]["path"][-1:]
            prior = dict(best["state"], path=[(0.0, path[0][1])] if path else [])
            v, x, d, state = final.solve(sb.box(np.clip(np.array(best["x"], float), bounds[:, 0], bounds[:, 1])),
                                         stage="corrected/anchor",
                                         prior=prior, t_start=best["logl"])
            return dict(logl=v, x=x, diagnostics=d, state=state)

        anchor = checkpoint(directory / "anchor.json", solve_anchor)
    finally:
        shared.close()
    payload = dict(names=source["names"], full_parameter_names=names, best_logl=anchor["logl"],
                   best_x=anchor["x"], objective="native", global_optimum_certified=False,
                   anchor_diagnostics=anchor["diagnostics"], method="surrogate-corrected residual Newton",
                   configuration=asdict(cfg), workers=workers)
    consistent = all(max((v for v in r["logl"] if v is not None), default=-np.inf) <= anchor["logl"] + 1e-6
                     for r in records.values())
    save(Path(out) / "true_profiles.json", dict(payload, profiles=records, anchor_consistent=consistent))
    if direct:
        save(Path(out) / "direct_profiles.json", dict(payload, profiles=directs))
    save(directory / "complete.json", dict(
        anchor_consistent=consistent, seconds=time.monotonic() - t0, workers=workers,
        budget_exhausted_nodes=sum(d.get("flag") != 0 for r in records.values() for d in r["optimization"] if d)))
