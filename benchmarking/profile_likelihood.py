"""1D profile likelihoods from a trained surrogate.

    python benchmarking/profile_likelihood.py RUN_DIR -it 10 -o profiles_it10.json

For each requested parameter this walks a grid, maximising the surrogate over all other
parameters at each grid point, and writes the resulting profile to JSON. With ``--true``
it also evaluates the real likelihood at each profile-optimal point, which is what makes
the residuals in the output meaningful.

WHY THE OPTIMISER IS NOT A PLAIN MULTI-START. A naive multi-start L-BFGS-B on this problem
produces profile points that beat the global anchor, i.e. negative Delta chi-squared,
because in 30-odd dimensions a random start rarely finds the ridge at all and the best of
a handful of bad optima is still bad. Three things fix it, and all three are needed:

  * profiles are walked OUTWARD from the anchor, each grid point warm-started from its
    neighbour, so the optimiser never has to re-find the ridge from scratch;
  * neighbour relaxation re-optimises each point from its neighbours' solutions until
    nothing improves, which repairs points that fell off the ridge;
  * the anchor is re-polished whenever any profile point beats it, and the whole sweep
    repeats. Without this the reported Delta chi-squared is measured against a maximum the
    optimiser has already exceeded.

GRADIENTS COME FROM THE NETWORK. The surrogate is auto-differentiable, so L-BFGS-B gets
analytic gradients rather than finite differences. This is one of the concrete pay-offs of
emulating the likelihood rather than the observables.

THE -inf PRIOR TERM IS DELIBERATELY NOT APPLIED. ``SurrogateLikelihood.logpost`` returns
-inf outside the metadata bounds, which would make the gradient NaN; L-BFGS-B enforces the
box itself, so ``loglkl`` is the right function to hand it.

GRID CHOICE MATTERS MORE THAN IT LOOKS. ``--grid-frac`` spans the central fraction of the
prior box. The default of 1.0 covers the whole box, which is the right choice for seeing
where an emulator is optimistic, but it is a poor choice for locating Delta chi-squared
crossings: at 25 points over a +/-10 sigma box a 1 sigma interval spans about 2.4 cells,
and linear interpolation of a locally parabolic curve then misestimates its width by
several per cent. If the profiles are destined for ``profiles/metric.py``, use a narrower
span and more points (61 points over ``--grid-frac 0.5`` was measured to reduce that error
by a factor of 34) and check the floor with ``profiles/metric.py --floor``.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profiles.surrogate import ProfileOptimizer, scan_profiles


def grid_for(lo, hi, n, frac):
    """``n`` points over the central ``frac`` of ``[lo, hi]``, centred on the box centre."""
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * frac
    return np.linspace(centre - half, centre + half, n)


def finite_or_none(value):
    """`value` for JSON: a value that is not finite, a point the likelihood refused, is None."""
    if isinstance(value, dict):
        return {k: finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_or_none(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", help="run directory containing metadata.json and trained_models/")
    p.add_argument("-it", "--iteration", type=int, required=True)
    p.add_argument("-o", "--output", required=True, help="output json path")
    p.add_argument("--model", default=None,
                   help="model filename; default model_it_<iteration>.keras")
    p.add_argument("--train", default=None,
                   help="training csv used ONLY to seed the anchor search; default "
                        "data_it_<iteration>.csv. Give this explicitly for a final "
                        "iteration, whose data_it_<N+1>.csv is never written.")
    p.add_argument("-p", "--params", default=None,
                   help="comma-separated parameters to profile; default all")
    p.add_argument("--region", choices=["run", "none"], default="run",
                   help="run (default): the network inside the region of its boosts, as CLiENT's other consumers use it, "
                        "when it has one; none: the bare network")
    p.add_argument("-n", "--n-grid", type=int, default=25)
    p.add_argument("--grid-frac", type=float, default=1.0,
                   help="fraction of the prior box the grid spans, centred (default 1.0)")
    p.add_argument("--max-passes", type=int, default=15)
    p.add_argument("--true", action="store_true",
                   help="also evaluate the true likelihood at each profile optimum")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", choices=['float32', 'float64'], default='float32',
                   help="evaluation precision; float64 retains weights but improves line searches")
    p.add_argument("--gtol", type=float, default=1e-4,
                   help="projected-gradient tolerance in scaled coordinates")
    p.add_argument("--ftol", type=float, default=1e-10)
    p.add_argument("--value-tol", type=float, default=1e-4,
                   help="absolute log-likelihood tolerance for profile/anchor stability")
    p.add_argument("--retry-starts", type=int, default=2,
                   help="independent starts for nodes failing the stationarity check")
    a = p.parse_args()
    if a.n_grid < 2 or not 0 < a.grid_frac <= 1 or a.max_passes < 1:
        p.error("require n-grid >= 2, 0 < grid-frac <= 1, and max-passes >= 1")
    if (not np.isfinite([a.gtol, a.ftol, a.value_tol]).all()
            or a.gtol <= 0 or a.ftol < 0 or a.value_tol <= 0 or a.retry_starts < 0):
        p.error("invalid optimizer tolerances or retry budget")

    import tensorflow as tf
    from config.config import Config
    from run.run import settings_file
    from likelihood.surrogate import SurrogateLikelihood, SurrogateMetadata
    from model.network import load_model, copy_model_dtype, restrict_model
    from sampling.targets import load as load_region, resolve

    run = Path(a.run_dir)
    md = SurrogateMetadata.load(run / "metadata.json")
    names, ndim = md.param_names, len(md.param_names)
    lo = np.array([md.bounds[n][0] for n in names], dtype=np.float64)
    hi = np.array([md.bounds[n][1] for n in names], dtype=np.float64)

    wanted = names if a.params is None else [s.strip() for s in a.params.split(",")]
    missing = [n for n in wanted if n not in names]
    if len(set(wanted)) != len(wanted):
        p.error("profile parameter names must be unique")
    if missing:
        p.error(f"unknown parameter(s): {', '.join(missing)}")
    idx = [names.index(n) for n in wanted]

    model_name = a.model or f"model_it_{a.iteration}.keras"
    model = load_model(run / "trained_models" / model_name)
    region = resolve(run, a.iteration, None if a.region == "none" else "run")
    if region is not None:
        model = restrict_model(model, load_region(run, region), dtype=a.precision)
    elif a.precision == 'float64':
        model = copy_model_dtype(model)
    surrogate = SurrogateLikelihood(model=model, metadata=md)
    tf_dtype = getattr(tf, a.precision)

    true_lk = None
    if a.true:
        cfg = Config.from_yaml(str(settings_file(run)))
        from likelihood.base import build_likelihood
        true_lk = build_likelihood(
            cfg.likelihood.wrapper,
            str(run / "likelihood_input" / Path(cfg.likelihood.input).name))

    @tf.function(reduce_retracing=True)
    def value_and_grad(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = surrogate.loglkl(x)
        return y, tape.gradient(y, x)

    # OPTIMISE IN SCALED COORDINATES u = (x - centre) / halfwidth, so every free coordinate
    # is O(1) on [-1, 1]. In raw units these parameters span omega_b ~ 0.02 to
    # ps_A_100_100 ~ 200, and L-BFGS-B builds its inverse-Hessian approximation from that
    # geometry: the line search stalls far short of the optimum. Measured on the 27D
    # Gaussian it5 network, the raw-unit search reported -23.3518 after 60 passes while the
    # scaled one reaches -22.6676 -- the network's true maximum -- in 0.1 s. This is
    # conditioning, not float32 precision: the gradients are exact autodiff derivatives and
    # their noise floor (~1e-7 relative) is four orders below the 0.68 log-unit gap.
    centre_all = 0.5 * (lo + hi)
    halfw_all = 0.5 * (hi - lo)
    if not np.isfinite(np.r_[lo, hi]).all() or np.any(halfw_all <= 0):
        p.error("finite, strictly ordered prior bounds are required")

    def objective(free_u, i_fixed, v_fixed):
        full = np.empty(ndim, dtype=a.precision)
        mask = np.ones(ndim, dtype=bool)
        if i_fixed is not None:
            full[i_fixed], mask[i_fixed] = v_fixed, False
        full[mask] = centre_all[mask] + free_u * halfw_all[mask]
        y, g = value_and_grad(tf.constant(full[None, :], dtype=tf_dtype))
        grad = np.asarray(g.numpy()[0][mask], dtype=np.float64) * halfw_all[mask]
        return -float(y.numpy()[0]), -grad

    optimise = ProfileOptimizer(objective, lo, hi, seed=a.seed, gtol=a.gtol,
                                ftol=a.ftol, retry_starts=a.retry_starts)

    rng = np.random.default_rng(a.seed)
    train_name = a.train or f"data_it_{a.iteration}.csv"
    train_path = run / "training_data" / train_name
    if not train_path.exists():
        p.error(f"{train_path} not found; pass --train explicitly "
                "(a final iteration has no data_it_<N+1>.csv)")
    train = np.loadtxt(train_path, delimiter=",", skiprows=1)
    top = train[np.argsort(train[:, -1])[-20:], :ndim]
    starts = [*top, 0.5 * (lo + hi), *rng.uniform(lo, hi, size=(10, ndim))]

    t0 = time.monotonic()
    grids = [grid_for(lo[i], hi[i], a.n_grid, a.grid_frac) for i in idx]
    anchor_v, anchor_x, records, quality = scan_profiles(
        optimise, starts, grids, idx, max_passes=a.max_passes, value_tol=a.value_tol)
    results = dict(zip(wanted, records))
    print(f"best fit found: logL = {anchor_v:.7g} "
          f"({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
    if not quality['converged']:
        print("WARNING: numerical checks incomplete; inspect optimization_quality "
              "and per-node diagnostics in the JSON", flush=True)

    out = {}
    for nm in wanted:
        r = results[nm]
        rec = dict(grid=r["grid"].tolist(), logl=r["logl"].tolist(),
                   xopt=r["xopt"].tolist(),
                   optimization=r["diagnostics"])
        if true_lk is not None:
            tv = np.array([true_lk.loglkl(x) for x in r["xopt"]], dtype=float)
            raw = 2.0 * (tv - r["logl"])
            rec.update(true_logl=tv.tolist(), raw=raw.tolist())
            print(f"{nm:>16}: raw mean {raw.mean():+7.3f}  sd {raw.std():6.3f}  "
                  f"min {raw.min():+7.3f}  max {raw.max():+7.3f}")
        out[nm] = rec

    payload = {"iteration": a.iteration, "model": model_name, "names": wanted,
               "n_grid": a.n_grid, "grid_frac": a.grid_frac,
               "best_logl": anchor_v, "best_x": anchor_x.tolist(), "profiles": out,
               "full_parameter_names": names, "schema_version": 2,
               "optimization_quality": quality,
               "optimizer_settings": dict(optimise.options, retry_starts=a.retry_starts,
                                          seed=a.seed, precision=a.precision),
               "true_evaluation_kind": "at_surrogate_optima" if a.true else None,
               "region": region}
    if true_lk is not None:
        tb = float(true_lk.loglkl(anchor_x))
        payload["true_at_best"] = tb
        print(f"\nemulator best {anchor_v:.4f}, true there {tb:.4f} "
              f"(bias {2 * (tb - anchor_v):+.3f} in chi-squared)")
        # MontePythonLikelihood has close(); CobayaLikelihood does not.
        getattr(true_lk, "close", lambda: None)()

    Path(a.output).write_text(json.dumps(finite_or_none(payload), allow_nan=False))
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
