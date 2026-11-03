"""Where a run's training box should be centred: the best fit of its other parameters with
some of them (the cosmology) held fixed.

    mpirun -n N python analysis/find_centre.py runs/RUN/cobaya.yaml --fix H0 logA ns omegabh2 omegach2 tau
                                              --covmat FILE --out results/RUN_centre [--start FILE]

    python analysis/make_box.py RUN --centre results/RUN_centre/centre.json   # then the box

A likelihood with many nuisance parameters comes with their priors, which are far wider than
the posterior, and sometimes with a covariance of an earlier chain (a proposal matrix), which
gives widths but no centres. A box of a few widths needs the centres. With the cosmology fixed
at published values the Boltzmann code runs once per rank and is then served from Cobaya's
cache, so the log-posterior as a function of the remaining parameters costs only the
likelihoods: their best fit takes minutes.

Method. The free parameters are moved in the coordinates z in which the given covariance is
the unit matrix (theta = start + L z, L its Cholesky factor; a parameter the covariance does
not have gets its own axis, scaled by the proposal width of the Cobaya file). There the
problem is close to a unit quadratic whatever the correlations, and L-BFGS-B needs few
iterations. Every iteration is one batch of 2 n + 1 points (central differences of step
`--step` in z), spread over the ranks. A point outside a prior range is evaluated at the
range's edge, with a quadratic penalty for the distance, so the likelihood is never asked for
a point its prior refuses.

Prior edges. Differences taken across an edge see that penalty, and a fit that runs into an
edge stalls within a step of it with the other parameters not yet settled. So the fit is
done in rounds: a parameter that starts or ends within 1.5 steps of an edge is put on the
edge and held there, and the others are fitted again; when no further parameter arrives at
an edge, every held one is tried one step inside, and released if the log-posterior is
higher there. An amplitude consistent with zero ends at zero, held; a reference value that
merely sits on an edge (HiLLik's SPT3G_beta_pol_* start at 1) is released.

Written to --out: centre.json with, for every parameter, the value, whether it sits at a
prior edge, the width from the covariance, and the width of the log-posterior along that
parameter alone at the best fit (conditional: never above the marginal one of the
covariance when the covariance is right; one far above it says the covariance is too narrow
there). A parameter at an edge has no such width; it gets the distance inside at which the
log-posterior is 1 lower. Also -2 log(likelihood) of every likelihood at the best fit (a chi2
only where the likelihood drops its normalisation: HiLLik's ACT part keeps it), and the
history of the fit.

--start FILE (a centre.json, or {name: value}) starts from other values than the Cobaya
file's reference point. --fix NAME=VALUE fixes at another value than the reference.

A rank that raises ends the job (stop_peers_on_error).
"""
import argparse
import json
import os
import sys
import time
import traceback
from functools import wraps
from pathlib import Path

import numpy as np
from mpi4py import MPI

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.environ.get("CLIENT_REPO", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(HERE))


def stop_peers_on_error(function):
    """A rank that raises stops the others, as client.py does for its main: the traceback, then
    comm.Abort. A worker that raises leaves rank 0 waiting for its values, and the node is held
    until the time limit. Cobaya's own hook for uncaught exceptions ends such a job only once a
    model has been built, and never for a SystemExit; this does not rely on it. An exit with
    status 0 (--help) is not a failure and aborts nothing."""
    @wraps(function)
    def guarded(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            comm = MPI.COMM_WORLD
            if comm.size > 1 and not (isinstance(error, SystemExit) and error.code in (None, 0)):
                traceback.print_exc()
                sys.stdout.flush()
                sys.stderr.flush()
                comm.Abort(1)
            raise
    return guarded


def read_covmat(path):
    """(names, matrix) of a Cobaya covariance file: '# name name ...', then the rows."""
    lines = Path(path).read_text().splitlines()
    names = lines[0].lstrip("#").split()
    matrix = np.loadtxt(lines[1:1 + len(names)])
    if matrix.shape != (len(names), len(names)):
        raise ValueError(f"{path}: {len(names)} names but a matrix of shape {matrix.shape}")
    return names, matrix


def prior_of(info):
    """(kind, lower, upper, loc, scale) of a Cobaya prior entry."""
    prior = info["prior"]
    if "min" in prior or "max" in prior:
        return "flat", prior.get("min"), prior.get("max"), None, None
    if prior.get("dist", "uniform") == "norm":
        return "gauss", None, None, float(prior["loc"]), float(prior["scale"])
    raise ValueError(f"prior {prior} is neither a range nor a normal distribution")


def reference(info):
    """The reference value of a Cobaya parameter: ref (its loc), else the prior's centre."""
    ref = info.get("ref")
    if isinstance(ref, dict):
        ref = ref.get("loc")
    if ref is not None:
        return float(ref)
    kind, lower, upper, loc, _ = prior_of(info)
    return loc if kind == "gauss" else 0.5 * (float(lower) + float(upper))


def fallback_width(info):
    """A width for a parameter the covariance does not have: the proposal, else the prior's."""
    kind, lower, upper, _, scale = prior_of(info)
    if info.get("proposal") is not None:
        return float(info["proposal"])
    return scale if kind == "gauss" else (float(upper) - float(lower)) / 20.0


def scaling(names, infos, covmat):
    """Covariance of the free parameters `names`: the block of the given matrix, and a
    diagonal entry from the Cobaya file for a parameter it does not have. Returns
    (matrix, widths, names not in the covariance)."""
    known, matrix = covmat if covmat is not None else ([], np.zeros((0, 0)))
    index = {name: k for k, name in enumerate(known)}
    out = np.zeros((len(names), len(names)))
    missing = []
    for a, name in enumerate(names):
        if name in index:
            for b, other in enumerate(names):
                if other in index:
                    out[a, b] = matrix[index[name], index[other]]
        else:
            missing.append(name)
            out[a, a] = fallback_width(infos[name]) ** 2
    return out, np.sqrt(np.diag(out)), missing


class Problem:
    """f(z) = -log-posterior at theta(z), clipped into the prior ranges, plus the penalty."""

    def __init__(self, names, start, lower, upper, cholesky, widths, penalty=100.0):
        self.names, self.start, self.cholesky, self.widths, self.penalty = names, start, cholesky, widths, penalty
        pad = 1e-9 * np.where(np.isfinite(upper - lower), upper - lower, 1.0)
        self.lower, self.upper = lower + pad, upper - pad

    def theta(self, z):
        return self.start + np.atleast_2d(z) @ self.cholesky.T

    def clipped(self, z):
        theta = self.theta(z)
        inside = np.clip(theta, self.lower, self.upper)
        return inside, 0.5 * self.penalty * np.sum(((theta - inside) / self.widths) ** 2, axis=1)


def evaluator(comm, truth, free, fixed_values):
    """evaluate(points of the free parameters) -> log-posteriors, on all ranks; and the
    function that releases the workers. Rank 0 calls both; a worker serves until released."""
    position = [truth.param_names.index(name) for name in free]
    template = np.array([fixed_values.get(name, np.nan) for name in truth.param_names], float)

    def local(points):
        out = np.empty(len(points))
        for k, point in enumerate(points):
            full = template.copy()
            full[position] = point
            out[k] = truth.loglkl(full)
        return out

    def evaluate(points):
        points = np.asarray(points, float)
        chunks = np.array_split(points, comm.size)
        mine = comm.scatter(chunks, root=0)
        values = comm.gather(local(mine), root=0)
        return np.concatenate(values)

    def serve():
        while True:
            mine = comm.scatter(None, root=0)
            if mine is None:
                return
            comm.gather(local(mine), root=0)

    def release():
        comm.scatter([None] * comm.size, root=0)

    return evaluate, serve, release


def fit(problem, evaluate, step, max_iterations, log=print, settled=1e-3):
    """L-BFGS-B on f(z) with central differences; returns (z, f, state, last batch, result)."""
    from scipy.optimize import minimize

    n = len(problem.names)
    offsets = np.concatenate([np.zeros((1, n)), step * np.eye(n), -step * np.eye(n)])
    state = dict(history=[], evaluations=0, worst=None, last=None)

    def batch(z):
        points, penalty = problem.clipped(z + offsets)
        values = -evaluate(points) + penalty
        state["evaluations"] += len(points)
        finite = values[np.isfinite(values)]
        if not np.isfinite(values[0]):
            raise RuntimeError("the log-posterior is not finite at a point inside the prior ranges; "
                               "start the fit somewhere else (--start)")
        # a neighbour the likelihood cannot evaluate counts as far worse than anything seen
        values = np.where(np.isfinite(values), values, finite.max() + 1e4)
        return values

    def objective(z):
        values = batch(z)
        gradient = (values[1:n + 1] - values[n + 1:]) / (2 * step)
        state["last"] = (z.copy(), values)
        return values[0], gradient

    def progress(intermediate_result):
        f = float(intermediate_result.fun)
        state["history"].append(f)
        state["x"] = np.array(intermediate_result.x)
        log(f"   iteration {len(state['history']):3d}: -log-posterior {f:14.4f}   "
            f"|z| {np.linalg.norm(intermediate_result.x):7.2f}   {state['evaluations']} evaluations")
        # Settled: three iterations that together gained less than `settled`. Without this the
        # line search goes on for hundreds of batches on differences at rounding level.
        history = state["history"]
        if len(history) > 3 and history[-4] - history[-1] < settled:
            state["settled"] = True
            raise StopIteration

    f0, g0 = objective(np.zeros(n))
    log(f"   start:         -log-posterior {f0:14.4f}")
    if 0.5 * float(g0 @ g0) < settled:
        # At a unit quadratic the gain still to be had is |gradient|^2 / 2: nothing to fit.
        from types import SimpleNamespace

        result = SimpleNamespace(x=np.zeros(n), success=True, message=f"settled at the start: less than {settled:g} to gain")
        return result.x, float(f0), state, state["last"][1], result
    result = minimize(objective, np.zeros(n), jac=True, method="L-BFGS-B", callback=progress,
                      options=dict(maxiter=max_iterations, maxfun=3 * max_iterations, ftol=1e-12, gtol=2e-3, maxcor=30))
    z = state.get("x", result.x) if state.get("settled") else result.x
    if state.get("settled"):
        result.success, result.message = True, f"settled: three iterations gained less than {settled:g}"
    values = batch(z)                                    # at the answer itself, for the curvatures
    return z, float(values[0]), state, values, result


def fit_with_edges(names, start, lower, upper, matrix, widths, evaluate, step, max_iterations, log=print, rounds=10):
    """The fit in rounds, with the parameters at a prior edge held there (see the module text).
    evaluate(points of all the parameters) -> log-posteriors. Returns a dict: theta, value
    (-log-posterior), held ({index: "lower" | "upper"}), curvature (along the covariance's axes
    of the parameters not held, from the last round), evaluations, history, converged, message."""
    n = len(names)
    theta = np.array(start, float)
    reach = 1.5 * step * widths
    held, released, total, history = {}, set(), 0, []
    outcome = None

    def near(k):
        if theta[k] - lower[k] <= reach[k]:
            return "lower"
        if upper[k] - theta[k] <= reach[k]:
            return "upper"
        return None

    def hold(candidates):
        new = []
        for k in candidates:
            side = near(k)
            if side and k not in held and k not in released:
                held[k] = side
                theta[k] = lower[k] if side == "lower" else upper[k]
                new.append(k)
        return new

    hold(range(n))
    for number in range(1, rounds + 1):
        free = [k for k in range(n) if k not in held]
        log(f"   round {number}: {len(free)} parameters fitted, {len(held)} held at a prior edge"
            + (f" ({', '.join(names[k] for k in sorted(held))})" if held else ""))
        problem = Problem([names[k] for k in free], theta[free], lower[free], upper[free],
                          np.linalg.cholesky(matrix[np.ix_(free, free)]), widths[free])

        def evaluate_free(points, free=free):
            full = np.tile(theta, (len(points), 1))
            full[:, free] = points
            return evaluate(full)

        z, value, state, values, result = fit(problem, evaluate_free, step, max_iterations, log=log)
        total += state["evaluations"] + len(values)
        history += state["history"]
        theta[free] = problem.clipped(z)[0][0]
        m = len(free)
        outcome = dict(value=value, converged=bool(result.success), message=str(result.message),
                       curvature=(values[1:m + 1] + values[m + 1:] - 2 * values[0]) / step ** 2)
        if hold(free):
            continue
        if not held:
            break
        # every held parameter one step inside: released where the log-posterior is higher there
        order = sorted(held)
        inside = np.tile(theta, (len(order) + 1, 1))
        for row, k in enumerate(order, start=1):
            inside[row, k] += (2 * step * widths[k]) * (1 if held[k] == "lower" else -1)
        tried = -evaluate(inside)
        total += len(inside)
        outcome["value"] = float(tried[0])
        back = [k for row, k in enumerate(order, start=1) if np.isfinite(tried[row]) and tried[row] < tried[0] - 1e-9]
        if not back:
            break
        for k in back:
            theta[k] = inside[order.index(k) + 1, k]
            del held[k]
            released.add(k)
        log(f"   released from its edge: {', '.join(names[k] for k in back)}")
    else:
        outcome["converged"], outcome["message"] = False, f"parameters still arriving at or leaving prior edges after {rounds} rounds"
    return dict(theta=theta, held=held, evaluations=total, history=history, **outcome)


def one_sided_widths(theta, value, held, lower, upper, scales, evaluate, drop=1.0, ladder=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)):
    """For every parameter held at a prior edge: how far inside the log-posterior is `drop`
    lower than at the edge, the others staying where they are. A posterior that falls away
    from an edge has no two-sided width; for one that falls exponentially this distance is its
    scale (its mean and its standard deviation). Returns {index: distance, or None when the
    log-posterior has not dropped that much three scales inside, or "inert" when it is the
    same at every distance: the likelihood does not depend on the parameter}."""
    order = sorted(held)
    if not order:
        return {}
    points = np.tile(theta, (len(order) * len(ladder), 1))
    for a, k in enumerate(order):
        room = upper[k] - lower[k]
        for b, fraction in enumerate(ladder):
            distance = min(fraction * scales[k], 0.999 * room) if np.isfinite(room) else fraction * scales[k]
            points[a * len(ladder) + b, k] += distance if held[k] == "lower" else -distance
    rise = (-evaluate(points) - value).reshape(len(order), len(ladder))
    out = {}
    for a, k in enumerate(order):
        distances = np.abs(points[a * len(ladder):(a + 1) * len(ladder), k] - theta[k])
        out[k] = None
        if np.all(np.abs(rise[a]) < 1e-9):          # identical all the way: the likelihood does not use it
            out[k] = "inert"
            continue
        for b in range(len(ladder)):
            if np.isfinite(rise[a, b]) and rise[a, b] >= drop:
                before_d, before_r = (distances[b - 1], max(rise[a, b - 1], 0.0)) if b else (0.0, 0.0)
                out[k] = float(before_d + (drop - before_r) / (rise[a, b] - before_r) * (distances[b] - before_d))
                break
    return out


@stop_peers_on_error
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cobaya", type=Path)
    parser.add_argument("--fix", nargs="*", default=[], help="NAME (at its reference value) or NAME=VALUE")
    parser.add_argument("--covmat", type=Path, help="a Cobaya covariance file with widths and correlations of the parameters")
    parser.add_argument("--start", type=Path, help="a centre.json or {name: value} to start from")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", type=float, default=0.1, help="finite-difference step, in widths")
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    from loading import build_in_turn  # noqa: E402
    from likelihood.cobaya import CobayaLikelihood  # noqa: E402

    truth = build_in_turn(comm, lambda: CobayaLikelihood(str(args.cobaya)), label="[centre]")   # CLIENT_LOAD_GROUP
    infos = {name: truth.cobaya_info["params"][name] for name in truth.param_names}
    fixed = {}
    for item in args.fix:
        name, _, value = item.partition("=")
        if name not in infos:
            raise SystemExit(f"--fix {name}: not a sampled parameter of {args.cobaya}")
        fixed[name] = float(value) if value else reference(infos[name])
    free = [name for name in truth.param_names if name not in fixed]
    evaluate, serve, release = evaluator(comm, truth, free, fixed)
    if rank != 0:
        serve()
        return

    started = time.time()
    try:
        start = {name: reference(infos[name]) for name in free}
        if args.start:
            given = json.loads(args.start.read_text())
            given = {k: v["value"] for k, v in given["parameters"].items()} if "parameters" in given else given
            start.update({name: float(given[name]) for name in free if name in given})
        priors = {name: prior_of(infos[name]) for name in free}
        lower = np.array([-np.inf if priors[n][1] is None else float(priors[n][1]) for n in free])
        upper = np.array([np.inf if priors[n][2] is None else float(priors[n][2]) for n in free])
        covmat = read_covmat(args.covmat) if args.covmat else None
        matrix, widths, missing = scaling(free, infos, covmat)
        first = np.array([start[n] for n in free])
        print(f"[centre] {args.cobaya}: {len(truth.param_names)} sampled parameters, {len(fixed)} fixed "
              f"({', '.join(f'{k} = {v:g}' for k, v in fixed.items())}), {len(free)} free; {comm.size} ranks", flush=True)
        if missing:
            print(f"[centre] not in the covariance, scaled by their proposal width: {', '.join(missing)}", flush=True)
        outside = [n for n, v, lo, hi in zip(free, first, lower, upper) if not lo <= v <= hi]
        if outside:
            raise SystemExit(f"start values outside their prior ranges: {outside}")

        say = lambda text: print(text, flush=True)
        done = fit_with_edges(free, first, lower, upper, matrix, widths, evaluate, args.step, args.iterations, log=say)
        n = len(free)
        theta, value, held = done["theta"], done["value"], done["held"]
        curvature_z = done["curvature"]

        # the width of the log-posterior along every parameter alone, at the best fit
        h = 0.3 * widths
        up, down = np.clip(theta + np.diag(h), lower, upper), np.clip(theta - np.diag(h), lower, upper)
        along = -evaluate(np.concatenate([up, down]))
        up_step, down_step = np.diag(up) - theta, theta - np.diag(down)
        conditional = np.full(n, np.nan)
        for k in range(n):
            a, b = up_step[k], down_step[k]
            if a > 0 and b > 0:        # second derivative from two unequal steps
                second = 2 * ((along[k] - value) / a + (along[n + k] - value) / b) / (a + b)
            else:                      # at an edge: no two-sided difference
                second = np.nan
            conditional[k] = 1 / np.sqrt(second) if np.isfinite(second) and second > 0 else np.nan

        inside = one_sided_widths(theta, value, held, lower, upper, widths, evaluate)

        full = np.array([fixed.get(name, np.nan) for name in truth.param_names], float)
        full[[truth.param_names.index(name) for name in free]] = theta
        at_best = truth.cobaya_model.logposterior(dict(zip(truth.param_names, full)))
        chi2 = {name: float(-2 * v) for name, v in zip(truth.cobaya_model.likelihood, at_best.loglikes)}
        parameters = {}
        for name in truth.param_names:
            if name in fixed:
                parameters[name] = dict(value=fixed[name], fixed=True)
                continue
            k = free.index(name)
            parameters[name] = dict(value=float(theta[k]), fixed=False, start=float(first[k]), at_prior_edge=held.get(k),
                                    width=float(widths[k]), width_from="proposal" if name in missing else "covariance",
                                    conditional_width=None if not np.isfinite(conditional[k]) else float(conditional[k]),
                                    one_sided_width=None if inside.get(k) == "inert" else inside.get(k),
                                    inert=bool(inside.get(k) == "inert" or (along[k] == value and along[n + k] == value
                                                                           and up_step[k] > 0 and down_step[k] > 0)),
                                    moved_in_widths=float((theta[k] - first[k]) / widths[k]),
                                    prior=dict(zip(("kind", "lower", "upper", "loc", "scale"), priors[name])))
        out = dict(cobaya=str(args.cobaya), covmat=str(args.covmat) if args.covmat else None, ranks=comm.size,
                   seconds=time.time() - started, evaluations=done["evaluations"] + 2 * n + 7 * len(held), iterations=len(done["history"]),
                   converged=done["converged"], message=done["message"], logpost=float(at_best.logpost),
                   logprior=float(np.sum(at_best.logpriors)), chi2=chi2, chi2_total=float(sum(chi2.values())),
                   history=done["history"], curvature_in_z=dict(min=float(np.nanmin(curvature_z)), median=float(np.nanmedian(curvature_z)),
                                                                 max=float(np.nanmax(curvature_z))),
                   step=args.step, parameters=parameters, written=time.strftime("%Y-%m-%d %H:%M:%S"))
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "centre.json").write_text(json.dumps(out, indent=1) + "\n")

        print(f"\n[centre] {out['message']}; {out['iterations']} iterations, {out['evaluations']} evaluations, {out['seconds']:.0f} s", flush=True)
        print(f"[centre] log-posterior {out['logpost']:.3f} (log-prior {out['logprior']:.3f}); -2 log(likelihood) by likelihood "
              "(a chi2 only where the likelihood drops its normalisation constant):")
        for name, v in chi2.items():
            print(f"            {name:44s} {v:12.2f}")
        print(f"            {'total':44s} {out['chi2_total']:12.2f}")
        print(f"[centre] curvature along the covariance's own axes (1 if the covariance is the posterior's, above 1 with "
              f"parameters fixed): min {out['curvature_in_z']['min']:.2f}, median {out['curvature_in_z']['median']:.2f}, "
              f"max {out['curvature_in_z']['max']:.2f}")
        wide = [(name, p["conditional_width"] / p["width"]) for name, p in parameters.items()
                if not p["fixed"] and p["conditional_width"] and p["conditional_width"] > 1.3 * p["width"]]
        if wide:
            print("[centre] WIDER than the covariance says (conditional width / width): "
                  + ", ".join(f"{name} {ratio:.1f}" for name, ratio in wide))
        edges = [f"{name} ({p['at_prior_edge']}; the log-posterior is 1 lower "
                 + (f"{p['one_sided_width']:.3g}" if p["one_sided_width"] else "more than three widths") + " inside)"
                 for name, p in parameters.items() if not p["fixed"] and p["at_prior_edge"] and not p["inert"]]
        print(f"[centre] at a prior edge: {', '.join(edges) or 'none'}")
        inert = [name for name, p in parameters.items() if not p["fixed"] and p["inert"]]
        if inert:
            print(f"[centre] WARNING: the log-posterior does not depend on {', '.join(inert)}: no likelihood uses "
                  "them as the file stands. Fix them in the Cobaya file; a sampled parameter that does nothing "
                  "only adds a flat direction to the box.")
        moved = sorted(((abs(p["moved_in_widths"]), name) for name, p in parameters.items() if not p["fixed"]), reverse=True)[:8]
        print("[centre] moved farthest from the start, in widths: " + ", ".join(f"{name} {d:.1f}" for d, name in moved))
        print(f"[centre] written: {args.out / 'centre.json'}", flush=True)
    except BaseException:
        # The workers are not released here. When rank 0's own share of a batch raises they are
        # in the middle of that batch and answer with a gather, not a scatter; the guard of main
        # stops them, wherever they are.
        raise
    else:
        release()


if __name__ == "__main__":
    main()
