"""python client.py evidence RUN_DIR [-i N] [--n-live 2048] [--prior a=lo:hi,...] [--prior-file F]

Bayesian evidence of a CLiENT run from its trained network, with the BEST nested sampler
(https://github.com/AndreasNygaard/best-inference; installed, on $BEST_PATH, or
--best-path). Only the network is evaluated: no Boltzmann code, backend or Cobaya model is
built, so it takes minutes rather than the CPU-years a true nested-sampling run costs.

Two numbers are reported (evidence/priors.py has the conventions):
  ln Z_box      for a uniform prior on the sampling box, what BEST computes;
  ln Z          in the run's own prior convention: Cobaya's priors, or MontePython's
                uniform [min, max] from the .param (the run's domain where a side is None),
                with any --prior / --prior-file range replacing a parameter's. It does not
                depend on the box as long as the box holds the posterior mass; a warning
                names any face of the box inside the prior that the posterior reaches.

The box is the run's domain (metadata.json), the region the network was trained and the
posterior is sampled in, unless --n-sigma (centre +/- K proposal sigma, clipped to the
domain; --keep-bounds exempts parameters) or --box (the lower/upper of an earlier output)
is given. A network with a region of the run's boosts (boost/region_it_M.npz at or before
the iteration) is integrated inside it, as profiles and derived chains use it; --region E puts
the region's edge at E times the reach of the boost points instead of the run's own, and
--region none integrates the bare network. Writes evidence/evidence_it_N_nliveM[_TAG].json
(_edgeE added for --region E) and a weighted GetDist chain of the nested samples beside it.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run.run import iterations, latest_model, settings_file


# Live points replaced per iteration, as a share of them: 1/16, 128 of the default 2048 (since 14
# October 2026; 10 before). BEST draws each replacement above the likelihood of the point it
# replaces and sorts old and new together before it removes any, so its count of the prior
# volume holds for a batch. Measured against 10: unbiased on correlated Gaussians with exact
# evidences (16 and 49 parameters), the same ln Z within the errors on SINU network 31
# (+0.28 +- 0.17) and the 49-parameter network 22 (-0.40 +- 0.28), the same bias of BEST on the
# exact 31-parameter ridge (-0.54 +- 0.09 and -0.69 +- 0.12); 2.5 times faster on the laptop
# and 4.3 on the cluster (141 against 613 minutes).
UPDATES_SHARE = 16


def domain_fingerprint(md):
    return hashlib.sha256(json.dumps(dict(names=md.param_names,
        bounds=[md.bounds[n] for n in md.param_names]), sort_keys=True).encode()).hexdigest()


def sampling_box(md, n_sigma=None, keep=(), box_file=None):
    names = md.param_names
    lo = np.array([md.bounds[n][0] for n in names], float)
    hi = np.array([md.bounds[n][1] for n in names], float)
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError("the run's domain is unbounded in some parameter; give --box or --n-sigma")
    if box_file is not None:
        saved = json.loads(Path(box_file).read_text())
        if saved.get("domain_sha256", domain_fingerprint(md)) != domain_fingerprint(md):
            raise ValueError("--box belongs to a different saved model domain")
        ordered = saved.get("param_names", saved.get("priors"))
        if isinstance(ordered, list) and ordered and isinstance(ordered[0], dict):
            ordered = [row.get("name") for row in ordered]
        if isinstance(ordered, dict):
            ordered = list(ordered)
        if ordered != names:
            raise ValueError("--box ordered param_names must match the model's parameters")
        a, b = np.asarray(saved["lower"], float), np.asarray(saved["upper"], float)
        if (a.shape != lo.shape or b.shape != hi.shape or not np.isfinite([a, b]).all()
                or np.any(a >= b) or np.any(a < lo) or np.any(b > hi)):
            raise ValueError("--box needs finite ordered bounds contained in the model domain")
        return (a, b), f"from {Path(box_file).name}"
    if n_sigma is None:
        return (lo, hi), "run domain"
    if not np.isfinite(n_sigma) or n_sigma <= 0:
        raise ValueError("--n-sigma must be positive and finite")
    unknown = set(keep) - set(names)
    if unknown:
        raise ValueError(f"--keep-bounds names unknown parameters: {', '.join(sorted(unknown))}")
    centre = np.array([np.nan if p.center is None else p.center for p in md.parameters], float)
    sigma = np.array([np.nan if p.sigma is None else p.sigma for p in md.parameters], float)
    free = np.array([n not in keep for n in names])
    known = np.isfinite(centre) & np.isfinite(sigma)
    if not known[free].all():
        missing = [n for n, k, f in zip(names, known, free) if f and not k]
        raise ValueError(f"no proposal centre or sigma for {', '.join(missing)}; name them in --keep-bounds")
    box_lo = np.where(free, np.maximum(lo, centre - n_sigma * sigma), lo)
    box_hi = np.where(free, np.minimum(hi, centre + n_sigma * sigma), hi)
    return (box_lo, box_hi), f"+/-{n_sigma:g} proposal sigma around the proposal centre, within the run domain"


def reference_value(run, iteration):
    """(value, where from): the best log-likelihood of the network's training set, at single
    precision, or (0.0, None) when the run has no training set left. Any constant near the peak
    will do: BEST is given the network minus it, and it is added back to what BEST returns. So
    when the network's own file is gone (a run tidied up), the nearest iteration's serves."""
    import pandas as pd

    found = iterations(run / "training_data", "data_it_*.csv")
    for number in sorted(found, key=lambda n: (abs(n - iteration), n)):
        values = pd.read_csv(found[number], usecols=["loglkl"])["loglkl"].to_numpy(float)
        values = values[np.isfinite(values)]
        if len(values):
            return float(np.float32(values.max())), f"training_data/{found[number].name}"
    return 0.0, None


def region_setting(text):
    """--region as the command takes it: `run`, `none`, or the region's edge in reaches."""
    if text in ("run", "none"):
        return text
    try:
        edge = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: run, none, or the region's edge, a positive number") from None
    if not np.isfinite(edge) or edge <= 0:
        raise argparse.ArgumentTypeError(f"{text!r}: the region's edge must be a positive number")
    return edge


def write_chain(root, names, labels, points, weights, logL, box):
    # GetDist refuses a name with a space, * or ?, and MontePython's 100*theta_s has one: the
    # chain of every such run could not be loaded. The names as the derived chain writes them.
    from derived.chain import getdist_name

    names = [getdist_name(name) for name in names]
    keep = weights > 0
    rows = np.column_stack([weights[keep] / weights[keep].max(), -logL[keep], points[keep]])
    np.savetxt(f"{root}.txt", rows, fmt="%.10e")
    Path(f"{root}.paramnames").write_text("".join(f"{n}\t{l}\n" for n, l in zip(names, labels)))
    Path(f"{root}.ranges").write_text("".join(f"{n}\t{lo:.10g}\t{hi:.10g}\n" for n, lo, hi in zip(names, *box)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path, help="run directory")
    parser.add_argument("-i", "--iteration", type=int, help="network iteration (default: the latest)")
    parser.add_argument("--n-live", type=int, default=2048,
                        help="live points (default 2048). A power of two: BEST adds up the prior volume in single "
                             "precision, n_live_updates / n_live at every iteration, which is exact only then")
    parser.add_argument("--n-live-updates", type=int, default=None,
                        help="live points replaced per iteration (default n_live / 16: 128 of 2048). BEST's time is "
                             "its number of network calls in sequence, not the points in them; 128 was 2.5 to 4.3 "
                             "times faster than 10 on real networks and gave the same ln Z (docs/evidence.md)")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="fraction of the evidence left in the live points at which BEST stops")
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prior", default="", help="uniform prior ranges, e.g. omega_b=0.019:0.025,omega_cdm=0.095:0.145")
    parser.add_argument("--prior-file", type=Path, help="YAML of prior ranges: {name: [low, high]}")
    box = parser.add_mutually_exclusive_group()
    box.add_argument("--n-sigma", type=float, help="box: proposal centre +/- this many proposal sigma")
    box.add_argument("--box", type=Path, help="box: the lower/upper of an earlier evidence output")
    parser.add_argument("--keep-bounds", default="", help="with --n-sigma: parameters that keep the run domain")
    parser.add_argument("--best-path", help="BEST checkout, if not installed or on $BEST_PATH")
    parser.add_argument("--region", nargs="?", type=region_setting, const="run",
                        help="run (or the bare option): the network only inside the region of the run's "
                             "boosts (acquisition.boost.region; boost/region_it_N.npz at or before the "
                             "iteration), and stop when the run has none; a number E: the same region with "
                             "its edge at E times the reach of the boost points (the output's name gets "
                             "_edgeE); none: the bare network over the box. Default: the region when the "
                             "run has one")
    parser.add_argument("--tag", default="", help="suffix for the output names")
    parser.add_argument("--quiet", action="store_true", help="no BEST progress output")
    args = parser.parse_args(argv)
    if not 0.0 < args.tolerance < 1.0:
        parser.error("--tolerance is a fraction of the evidence: above 0 and below 1")
    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")
    if args.n_live_updates is None:
        args.n_live_updates = max(1, args.n_live // UPDATES_SHARE)
    if not 1 <= args.n_live_updates <= args.n_live // 2:
        parser.error("--n-live-updates must be at least 1 and at most half of --n-live")

    import yaml

    from evidence import priors
    from evidence.sampler import EDGE_FRACTION, EDGE_WARNING, edge_mass, import_best, run_nested

    import_best(args.best_path)  # fail before loading anything
    run = args.run
    from sampling.targets import load_snapshot, region_file

    iteration = latest_model(run) if args.iteration is None else args.iteration
    saved_region = region_file(run, iteration)
    edge = args.region if isinstance(args.region, float) else None
    if args.region not in (None, "none") and saved_region is None:
        parser.error("--region: the run has no saved boost region at or before this iteration")
    setting = None if args.region == "none" else ("run" if edge is None else edge)
    surrogate, target = load_snapshot(run, iteration, setting)
    md, region_entry = surrogate.metadata, target["region"]
    names = md.param_names
    # Only the likelihood section matters: which backend, and its input (the run's copy).
    likelihood = yaml.safe_load(settings_file(run).read_text())["likelihood"]
    backend = likelihood["wrapper"]
    source = run / "likelihood_input" / Path(likelihood["input"]).name
    if not source.exists():
        source = Path(likelihood["input"])
    if backend == "cobaya":
        backend_ranges = priors.cobaya_ranges(source, names)
    elif backend == "montepython":
        backend_ranges = priors.montepython_ranges(source, dict(zip(names, md.scales)))
    else:
        parser.error(f"evidence supports runs with the cobaya or montepython wrapper, not {backend!r}")
    user = priors.parse_ranges(args.prior, args.prior_file)
    domain = ([md.bounds[n][0] for n in names], [md.bounds[n][1] for n in names])
    keep = {s.strip() for s in args.keep_bounds.split(",") if s.strip()}
    (box_lo, box_hi), box_kind = sampling_box(md, args.n_sigma, keep, args.box)
    (box_lo, box_hi), table, correction = priors.resolve(backend, names, domain, (box_lo, box_hi),
                                                         backend_ranges, user)
    box_lo, box_hi = np.asarray(box_lo), np.asarray(box_hi)
    lnV = float(np.sum(np.log(box_hi - box_lo)))

    import tensorflow as tf

    if region_entry is not None:
        region_note = f"network inside the region of {region_entry['file']} (edge {region_entry['edge']:g})"
    elif saved_region is not None:
        region_note = f"--region none: the bare network, not the region of boost/{saved_region.name}"
    else:
        region_note = "no boost region for this network: the bare network"
    model_sha256 = target["model_sha256"]
    reference, reference_from = reference_value(run, iteration)

    @tf.function(reduce_retracing=True)
    def log_prob(x):
        x = tf.cast(x, tf.float32)
        lo = tf.constant([md.bounds[n][0] for n in names], tf.float32)
        hi = tf.constant([md.bounds[n][1] for n in names], tf.float32)
        inside = tf.reduce_all((x >= lo) & (x <= hi), axis=-1)
        # BEST adds ln Z up in single precision. At a real log-likelihood of -15656 such numbers
        # are 1e-3 apart and part of every addition is rounded away, so BEST gets the network
        # minus a reference near the peak. The reference is a single-precision number itself:
        # the difference is then exact, and adding it back in double precision loses nothing.
        return tf.where(inside, surrogate.loglkl(x) - tf.constant(reference, tf.float32),
                        tf.constant(-np.inf, tf.float32))

    print(f"run {run.name}, iteration {iteration}, backend {backend}; box: {box_kind}"
          + (f"; user prior ranges for {', '.join(sorted(user))}" if user else "")
          + f"; {region_note}; {args.n_live} live points, {args.n_live_updates} replaced per iteration"
          + f"; BEST stops with {args.tolerance:g} of the evidence left in the live points"
          + (f"; BEST is given, and displays, ln L relative to {reference:.10g}, the best of {reference_from}"
             if reference_from else
             "; no training set in the run for a reference: BEST is given ln L as it is"),
          flush=True)
    if args.n_live & (args.n_live - 1):
        print(f"note: --n-live {args.n_live} is not a power of two. BEST adds ln X up in single precision, "
              f"{args.n_live_updates}/{args.n_live} at every iteration, and that sum is exact only for a power of two "
              "(docs/evidence.md, \"Single precision\"); 2048 is the default for that reason.", flush=True)
    result = run_nested(log_prob, box_lo, box_hi, n_live=args.n_live, n_live_updates=args.n_live_updates,
                        tolerance=args.tolerance, n_max_iter=args.max_iterations, seed=args.seed,
                        verbose=not args.quiet)
    lnZ_box = result["lnZ_box"] + reference
    lnZ = lnZ_box + lnV + correction

    # A face of the box inside the prior range cuts posterior mass the evidence misses; a
    # face on the prior's own edge (the MontePython domain, a Cobaya bound) does not.
    edges = edge_mass(result["points"], result["weights"], box_lo, box_hi)
    inside = []
    for i, row in enumerate(table):
        for side, box_value, prior_value in ((0, box_lo[i], row["lower"]), (1, box_hi[i], row["upper"])):
            interior = prior_value is None or abs(box_value - prior_value) > 1e-9 * max(1.0, abs(box_value))
            if interior and edges[i, side] > EDGE_WARNING:
                inside.append(dict(parameter=row["name"], side=("lower", "upper")[side],
                                   mass=float(edges[i, side])))
    for i, row in enumerate(table):
        row.update(box_lower=float(box_lo[i]), box_upper=float(box_hi[i]),
                   edge_mass_lower=float(edges[i, 0]), edge_mass_upper=float(edges[i, 1]))

    out_dir = run / "evidence"
    out_dir.mkdir(exist_ok=True)
    # An edge of one's own is another target than the run's region: its output stands beside
    # the default one, never in its place.
    tag = "_".join(filter(None, [args.tag, None if edge is None else f"edge{edge:g}"]))
    stem = f"evidence_it_{iteration}_nlive{args.n_live}" + (f"_{tag}" if tag else "")
    write_chain(out_dir / stem, names, md.param_labels, result["points"], result["weights"],
                result["logL"] + reference, (box_lo, box_hi))

    report = dict(
        target=dict(target, lower=box_lo.tolist(), upper=box_hi.tolist()),
        run=run.name, iteration=iteration, backend=backend, lnZ=lnZ, lnZ_box=lnZ_box,
        sigma_lnZ=result["sigma"], lnV_box=lnV, ln_prior_correction=correction,
        lnL_reference=reference, lnL_reference_from=reference_from,
        information_H=result["H"], box=box_kind, lower=box_lo.tolist(), upper=box_hi.tolist(),
        domain_sha256=domain_fingerprint(md), param_names=names, priors=table, box_faces_reached=inside, edge_fraction=EDGE_FRACTION,
        n_live=args.n_live, n_live_updates=args.n_live_updates, tolerance=args.tolerance,
        live_fraction=result["live_fraction"],
        n_dead=result["n_dead"], iteration_cap_reached=result["iteration_cap_reached"], seed=args.seed,
        device=result["device"], seconds=result["seconds"],
        region=region_entry, model_sha256=model_sha256,
        convention=("Cobaya's normalised priors (what Cobaya's nested samplers report)" if backend == "cobaya" else
                    "MontePython: uniform priors on the ranges listed; nuisance Gaussian priors as "
                    "MontePython adds them, unnormalised"),
    )
    # Whole or not at all: on the cluster the tasks of an array each read every output of the
    # run as they end (the summary), while others are being written.
    from run.storage import atomic_path

    with atomic_path(out_dir / f"{stem}.json") as temporary:
        temporary.write_text(json.dumps(report, indent=1))

    print(f"\n{'=' * 72}")
    print(f"ln Z      = {lnZ:+.4f} +/- {result['sigma']:.4f}   ({backend} prior convention)")
    print(f"ln Z_box  = {lnZ_box:+.4f}   (uniform on the box; ln V_box = {lnV:+.4f})")
    print(f"H = {result['H']:.2f} nats; {result['n_dead']} dead points, the live points hold "
          f"{result['live_fraction']:.1e} of the evidence; {result['seconds'] / 60:.1f} min on {result['device']}")
    if reference_from is None and abs(float(np.max(result["logL"]))) > 100.0:
        print(f"Warning: BEST added ln Z up in single precision at ln L near {float(np.max(result['logL'])):.0f}, "
              "where part of every addition is rounded away, because the run has no training set to take a "
              "reference from: ln Z above is too low by an amount that grows with the number of live points "
              "(docs/evidence.md, \"Single precision, and when BEST stops\").")
    changed = priors.changed_ranges(table)
    if changed:
        print("prior ranges not taken from the backend as written:\n  " + "\n  ".join(changed))
    for face in inside:
        print(f"WARNING: {face['mass']:.1e} of the posterior lies within {EDGE_FRACTION:.0%} of the box's "
              f"{face['side']} face in {face['parameter']}, inside its prior range: ln Z misses the mass beyond it.")
    if result["iteration_cap_reached"]:
        print("WARNING: BEST stopped at --max-iterations, not at its tolerance: ln Z is incomplete.")
    print("=" * 72)
    print(f"wrote {out_dir / stem}.json and the weighted chain {stem}.txt")


if __name__ == "__main__":
    main()
