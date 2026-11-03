"""A surrogate posterior chain with derived-parameter columns, in GetDist's format."""

import time
from pathlib import Path

import numpy as np

from derived.emulator import predict_derived

# Below this fraction the importance weights are carried by a handful of points.
MIN_RELIABLE_ESS = 0.1


def sample_posterior(surrogate, bounds, sampling, n_steps, thin, seed=None, diagnostics=None):
    """Sample the surrogate at T = 1, the posterior rather than the tempered target.

    Returns (chain, log_prob) of shapes (steps, walkers, ndim) and (steps, walkers),
    burn-in already discarded by the sampler. Multi-region output is an equal-weight
    mixture resample; its axes are storage dimensions, not chronological walkers.
    """
    from sampling.base import build_sampler

    bounds = np.asarray(bounds, dtype=float)
    region = getattr(surrogate, 'region', None)
    if region is not None and len(region) > 1:
        from sampling.mixture import sample_region_mixture

        return sample_region_mixture(surrogate, bounds, sampling, n_steps, thin, seed, diagnostics)
    sampler = build_sampler(
        name=sampling.sampler,
        n_walkers=sampling.n_walkers,
        ndim=len(bounds),
        log_prob_fn=surrogate.logpost,
    )
    rng = np.random.default_rng(seed)
    start = time.monotonic()
    sampler.run(
        n_steps=n_steps,
        initial_positions=(surrogate.region.start(sampling.n_walkers, rng)
                           if getattr(surrogate, 'region', None) is not None else
                           rng.uniform(bounds[:, 0], bounds[:, 1], size=(sampling.n_walkers, len(bounds)))) ,
        burn_in=sampling.burn_in,
        chunk_size=sampling.chunk_size,
        thin=thin,
    )
    chain = np.array(sampler.chain(), copy=True)
    log_prob = np.array(sampler.log_prob(), copy=True)
    acceptance = float(np.mean(sampler.acceptance_fraction().numpy()))
    print(f"Sampled {chain.shape[0]} x {chain.shape[1]} walkers in "
          f"{time.monotonic() - start:.1f}s (acceptance {acceptance:.2f}, "
          f"max(tau) {sampler.max_tau:.1f} steps)")
    sampler.reset()
    if diagnostics is not None:
        diagnostics.update(method='single ensemble', acceptance=acceptance)
    return chain, log_prob


def getdist_name(name):
    # As benchmarking/getdist_utils.py: MontePython names such as 100*theta_s carry
    # characters GetDist reads as markup ('*' marks a derived parameter).
    return "".join(c for c in name if not c.isspace() and c not in "*?")


def write_getdist_chain(root, chain, log_prob, param_names, param_labels, bounds,
                        derived, derived_names, derived_labels):
    """Write root.txt, root.paramnames and root.ranges.

    Columns: weight, -log posterior, the sampled parameters, then the derived ones.
    Columns of the storage array are written one after another; for a mixture resample
    they are not chronological walkers. Every row has weight 1; a sample from
    an ensemble sampler is not an independent chain, so no convergence statistic is
    implied by the split.
    """
    root = Path(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    n_steps, n_walkers, ndim = chain.shape

    points = chain.transpose(1, 0, 2).reshape(-1, ndim)
    minus_logpost = -log_prob.T.reshape(-1)
    values = derived.reshape(n_steps, n_walkers, -1).transpose(1, 0, 2).reshape(
        n_steps * n_walkers, -1
    )
    table = np.column_stack([np.ones(len(points)), minus_logpost, points, values])
    np.savetxt(root.with_suffix(".txt"), table, fmt="%.10e")

    lines = [f"{getdist_name(n)}\t{label}" for n, label in zip(param_names, param_labels)]
    lines += [f"{getdist_name(n)}*\t{label}" for n, label in zip(derived_names, derived_labels)]
    root.with_suffix(".paramnames").write_text("\n".join(lines) + "\n")

    ranges = [f"{getdist_name(n)}\t{lo}\t{hi}" for n, (lo, hi) in zip(param_names, bounds)]
    root.with_suffix(".ranges").write_text("\n".join(ranges) + "\n")


def chain_with_derived(run, iteration, derived_model, derived_metadata, n_steps=None,
                       thin=1, seed=None, region="run"):
    """Sample iteration `iteration`'s surrogate, attach emulated derived parameters,
    and write run/posterior_chains/surrogate_it_<N>.{txt,paramnames,ranges}.

    Returns (root, chain, log_prob, derived) with derived of shape (steps, walkers, k).
    """
    from config.config import Config
    from run.run import settings_file

    run = Path(run)
    cfg = Config.from_yaml(settings_file(run))
    from sampling.targets import load_snapshot
    surrogate, target = load_snapshot(run, iteration, region)
    region_entry = target["region"]
    metadata = surrogate.metadata
    print(f"Surrogate region: {region_entry or 'none (bare network)'}")
    bounds = [metadata.bounds[name] for name in metadata.param_names]
    n_steps = cfg.sampling.n_steps if n_steps is None else n_steps
    print(f"Sampling the iteration-{iteration} surrogate at T = 1: "
          f"{cfg.sampling.n_walkers} walkers, {n_steps} steps, burn-in "
          f"{cfg.sampling.burn_in}, thin {thin}")
    diagnostics = {}
    chain, log_prob = sample_posterior(surrogate, bounds, cfg.sampling, n_steps, thin,
                                       seed=seed, diagnostics=diagnostics)

    start = time.monotonic()
    flat = chain.reshape(-1, chain.shape[-1])
    derived = predict_derived(derived_model, flat).reshape(chain.shape[:2] + (-1,))
    print(f"Emulated {len(derived_metadata.names)} derived parameters at {len(flat)} "
          f"chain points in {time.monotonic() - start:.1f}s")

    root = run / "posterior_chains" / f"surrogate_it_{iteration}"
    write_getdist_chain(
        root, chain, log_prob, metadata.param_names, metadata.param_labels, bounds,
        derived, derived_metadata.names, derived_metadata.labels,
    )
    from run.storage import write_json
    from sampling.provenance import sampler_identity
    write_json(root.with_suffix('.target.json'), target)
    write_json(root.with_suffix('.sampling.json'),
               dict(sampler_identity(surrogate.region), target=target, **diagnostics))
    print(f"Wrote {root}.txt (+ .paramnames, .ranges)")
    return root, chain, log_prob, derived


def weights_problem(root):
    """Why the chain written at `root` cannot carry importance weights to the truth, or None:
    what its sidecar says of the sampler (sampling.provenance.sampling_correction_problem). A
    region of several ellipsoids is mixed by the network's evidence in each cell, and a cell
    whose estimate is not reliable leaves the shares of the modes, and any mean over them,
    uncertain. hpc/analysis/true_check.py withholds its corrections on the same test."""
    import json

    from sampling.provenance import sampling_correction_problem

    try:
        info = json.loads(Path(root).with_suffix('.sampling.json').read_text())
    except (OSError, ValueError):
        return 'the chain has no readable record of how it was sampled'
    return sampling_correction_problem(info)


def true_likelihood(run):
    """The run's own backend, with the run's effective prior bounds."""
    from config.config import Config
    from likelihood.base import build_likelihood
    from run.run import settings_file

    run = Path(run)
    cfg = Config.from_yaml(settings_file(run))
    likelihood = build_likelihood(
        wrapper=cfg.likelihood.wrapper,
        input_path=str(run / "likelihood_input" / Path(cfg.likelihood.input).name),
    )
    if cfg.prior.n_sigma is not None:
        likelihood.restrict_prior_bounds(cfg.prior.n_sigma)
    return likelihood


def pick_points(chain, log_prob, derived, n_points, rng):
    """n_points distinct chain rows: positions, surrogate log-posterior, emulated derived."""
    flat = chain.reshape(-1, chain.shape[-1])
    n_points = min(n_points, len(flat))
    index = rng.choice(len(flat), size=n_points, replace=False)
    return (
        flat[index].astype(np.float64),
        log_prob.reshape(-1)[index].astype(np.float64),
        derived.reshape(len(flat), -1)[index],
    )


def compare_exact(names, emulated, exact, derived_chain, surrogate_logpost, true_loglkl, weights_problem=None):
    """Emulated against exact derived values at posterior points, in units of each
    parameter's posterior width, plus what the true log-likelihood says about the
    surrogate at the same points.

    `is_shift` is how far importance weights exp(true - surrogate) move each derived
    parameter's posterior mean, also in posterior widths: the correction the emulated
    log-likelihood's own error calls for, as opposed to the derived emulator's. The rule is the
    true check's (hpc/analysis/true_check.py). A refused point (-inf) is a draw of the surrogate's
    posterior with weight zero: it stays in the baseline, the mean over all draws, where its exact
    value does not exist and its emulated value stands in (`baseline_emulated` counts them). A
    point without any value (NaN, +inf) has an unknown weight: shifts and ESS are withheld.

    `weights_problem` (weights_problem() of the chain) says the points are not a sample that
    such weights can be applied to: the shifts and the effective fraction are then NaN and the
    summary carries the reason. The errors of the emulated values are pointwise and stay.
    """
    sigma = derived_chain.reshape(-1, derived_chain.shape[-1]).std(axis=0)
    true_loglkl = np.asarray(true_loglkl, dtype=float)
    delta = true_loglkl - surrogate_logpost
    finite = np.isfinite(delta)
    refused = np.isneginf(true_loglkl)
    failed = ~np.isfinite(true_loglkl) & ~refused
    if failed.any():
        reason = f"{int(failed.sum())} of the {len(failed)} true evaluations failed (NaN or +inf), so their weights are not known"
        weights_problem = f"{weights_problem}; {reason}" if weights_problem else reason
    elif not finite.any():
        weights_problem = weights_problem or "no true value at these points is finite"
    weights = np.zeros_like(delta)
    if finite.any():
        weights[finite] = np.exp(delta[finite] - delta[finite].max())
    ess_fraction = weights.sum() ** 2 / (len(weights) * np.sum(weights**2)) if finite.any() else float("nan")

    rows = []
    for j, name in enumerate(names):
        ok = np.isfinite(exact[:, j]) & finite
        if not ok.any():
            rows.append({"name": name, "n": 0, "posterior_sigma": sigma[j], "baseline_emulated": len(ok),
                         **dict.fromkeys(("rms_error_over_sigma", "max_error_over_sigma", "mean_error_over_sigma",
                                          "is_shift_over_sigma"), float("nan"))})
            continue
        error = emulated[ok, j] - exact[ok, j]
        w = weights[ok]
        known = np.isfinite(exact[:, j])
        baseline = np.mean(np.where(known, exact[:, j], emulated[:, j]))
        rows.append({
            "name": name,
            "n": int(ok.sum()),
            "posterior_sigma": sigma[j],
            "baseline_emulated": int((~known).sum()),
            "rms_error_over_sigma": np.sqrt(np.mean(error**2)) / sigma[j],
            "max_error_over_sigma": np.max(np.abs(error)) / sigma[j],
            "mean_error_over_sigma": np.mean(error) / sigma[j],
            "is_shift_over_sigma": float("nan") if weights_problem else (
                np.sum(w * exact[ok, j]) / np.sum(w) - baseline) / sigma[j],
        })
    loglkl_summary = {
        "n": int(finite.sum()),
        "refused": int(refused.sum()),
        "failed": int(failed.sum()),
        "mean_true_minus_surrogate": float(np.mean(delta[finite])) if finite.any() else float("nan"),
        "std_true_minus_surrogate": float(np.std(delta[finite])) if finite.any() else float("nan"),
        "importance_ess_fraction": float("nan") if weights_problem else float(ess_fraction),
    }
    if weights_problem:
        loglkl_summary["importance_unavailable"] = weights_problem
    return rows, loglkl_summary


def format_exact(rows, loglkl_summary):
    unavailable = loglkl_summary.get("importance_unavailable")
    lines = [
        f"   true - surrogate log-likelihood at {loglkl_summary['n']} points"
        + (f" ({loglkl_summary['refused']} more refused: weight zero, and the emulated value in the IS baseline)"
           if loglkl_summary.get("refused") else "")
        + ": mean "
        f"{loglkl_summary['mean_true_minus_surrogate']:.4g}, std "
        f"{loglkl_summary['std_true_minus_surrogate']:.4g}; "
        + (f"no importance weights ({unavailable}): the IS shifts are not given" if unavailable else
           f"importance-weight ESS {100 * loglkl_summary['importance_ess_fraction']:.1f}%"),
        f"   {'parameter':<20}{'sigma':>11}{'rms err/sig':>13}{'max err/sig':>13}"
        f"{'bias/sig':>10}{'IS shift/sig':>14}",
    ]
    for r in rows:
        lines.append(
            f"   {r['name']:<20}{r['posterior_sigma']:>11.4g}{r['rms_error_over_sigma']:>13.4f}"
            f"{r['max_error_over_sigma']:>13.4f}{r['mean_error_over_sigma']:>10.4f}"
            f"{r['is_shift_over_sigma']:>14.4f}"
        )
    if loglkl_summary["importance_ess_fraction"] < MIN_RELIABLE_ESS:
        lines.append(
            f"   WARNING: an importance-weight ESS below {100 * MIN_RELIABLE_ESS:.0f}% means "
            "a few points carry the weights, so the IS shifts are not reliable; the "
            "surrogate log-likelihood is itself far from the truth on its posterior "
            "(an unconverged run)"
        )
    return "\n".join(lines)


def summarize(names, derived):
    """Posterior mean and standard deviation of each derived parameter."""
    flat = derived.reshape(-1, derived.shape[-1])
    lines = [f"   {'parameter':<20}{'mean':>14}{'std':>12}"]
    for j, name in enumerate(names):
        lines.append(f"   {name:<20}{np.mean(flat[:, j]):>14.6g}{np.std(flat[:, j]):>12.4g}")
    return "\n".join(lines)
