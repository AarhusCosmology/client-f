"""The true likelihood at posterior samples of chosen networks: the direct measure of a
network's error where its own posterior is.

    mpirun -n N python analysis/true_check.py RUN_DIR --cobaya runs/RUN/cobaya.yaml --domain runs/RUN/domain_bounds.json
                                             [--networks 10 15] [--points 500] [--seed 1] [--discard-chains]

--domain is the run's training box when the launcher gave it one (launch.py); a run whose box
came from prior.n_sigma has none, and the points are inside its box in any case.

Rank 0 takes `points` samples at random from RUN_DIR/analysis/posterior_it_N.npz (the T = 1
chains analysis/report.py wrote; a network without one is sampled here), every rank evaluates
the true likelihood at them through CLiENT's own dispatcher, and rank 0 writes
RUN_DIR/analysis/true_check.json and true_check_it_N.npz (points, network values, true values).
Networks checked by an earlier call stay in true_check.json.

A chain is used as the network's posterior only when it was sampled as one. A network whose
region has several ellipsoids is sampled cell by cell and weighted by its evidence in each
(CLiENT's sampling/mixture.py); a saved chain of such a region that does not record that
sampler is put aside and sampled again here. The record travels with the chain, into
true_check.json (`sampling`) and true_check_it_N.npz. Where it marks the weight of a cell as
not reliable, the samples cannot carry a mean over the posterior: the log says so before the
true evaluations, which are made and saved with their pointwise errors all the same, and the
corrected moments, the effective points and the evidence correction are withheld.

For each network, with D = truth minus network at its own posterior samples:
  mean, spread and rms of D     a constant offset changes nothing for one mode; the spread does
  effective points              (sum w)^2 / sum w^2 with w = exp(D): how many of the samples carry
                                the true posterior. Near all of them: the network's posterior
                                is the true one. A few: it is not yet.
  corrected mean and width      of every parameter with the weights w, against the network's
                                own, in units of its width: what the remaining error does to
                                the result. Only as good as the effective points behind it.

A point the likelihood refuses (refused.py) has no D. It comes back as -inf, or, with
CLIENT_REFUSED_VALUE in the job's environment (the job script reads the run's job.env), as the
refused value, which is a convention of the training set and not a likelihood: a difference of
tens of thousands if it were taken as one. Refusals are excluded only from finite error
moments. They have zero importance weight, while the baseline mean and width use ALL
surrogate draws. NaN/+inf evaluations have unknown weights: corrected moments and ESS
are withheld until they are resolved. `not_finite` includes refusals and failures; finite
error moments use points - not_finite draws. The
refused value is the run's own (RUN_DIR/refused_value.json; the environment's when the run has
no such file), whatever this job's environment says, so that a run gives the same numbers
either way. true_check_it_N.npz holds every point, with the refused ones marked in `refused`.
One thing still follows the environment: with CLIENT_REFUSED_VALUE in it the likelihood object
gives the refused value to a NaN as well (refused.py), which is then counted as refused.
That opt-in policy loses the distinction between numerical failure and zero density; the
true check cannot reconstruct it. Use the backend journal to investigate such points.

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
sys.path.insert(0, str(HERE / "analysis"))


def stop_peers_on_error(function):
    """A rank that raises stops the others, as client.py does for its main: the traceback, then
    comm.Abort. Rank 0 loads the run and samples a chain while the other ranks already wait for
    the points, and a rank left waiting for one that has raised holds the node until the time
    limit. Cobaya's own hook for uncaught exceptions ends such a job only once a model has been
    built, and never for a SystemExit; this does not rely on it. An exit with status 0 (--help)
    is not a failure and aborts nothing."""
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


def refused_points(true, *values):
    """Which true values are refusals of the likelihood and not values of it: -inf, and any of
    `values`, the finite value CLIENT_REFUSED_VALUE gives a refused point (None: there is none).
    By equality, as the launcher's journal counts them: the true values arrive here in double
    precision as the likelihood object returned them, and the refused value is one no real
    value equals (refused.py)."""
    true = np.asarray(true, float)
    refused = np.isneginf(true)
    for value in values:
        if value is not None:
            refused |= true == value
    return refused


def network_entry(iteration, x, network, true, best, names, refused, refused_value=None):
    """A network's entry of true_check.json: from its points x, its own values there, the true
    values, the best training value of the run and the parameter names. Points marked `refused`
    (refused_points) get zero weight. Compare weighted true moments to all surrogate draws;
    unresolved NaN/+inf values withhold corrected moments and ESS. Finite error moments
    remain conditional on genuine finite returns."""
    x, network, true = np.asarray(x), np.asarray(network, float), np.asarray(true, float)
    refused = np.asarray(refused, bool)
    if not np.isfinite(network).all() or not np.isfinite(x).all():
        raise ValueError("posterior samples and network values must be finite")
    ok = np.isfinite(true) & ~refused
    if not ok.any():
        # Rank 0 is in the loop over the networks: a raise here would lose the others' results too.
        failed = int((~refused).sum())
        return dict(effective_points=None, corrected_shift=None, corrected_width=None, largest_shift=None,
                    largest_shift_parameter=None, failed=failed,
                    correction_unavailable=(f"none of its {len(true)} points has a true value "
                                            f"({int(refused.sum())} refused, {failed} NaN or +inf)"),
                    iteration=iteration, points=int(len(x)), not_finite=int(len(true)), refused=int(refused.sum()),
                    refused_value=refused_value, truth_minus_network=None, median_depth_below_best=None,
                    above_best_training_value=0, best_true_here=None)
    d = true[ok] - network[ok]
    w = np.exp(d - d.max())
    ess = float(w.sum() ** 2 / np.sum(w ** 2))
    xs = x[ok].astype(float)
    mean, sd = x.astype(float).mean(axis=0), x.astype(float).std(axis=0)
    wmean = (w[:, None] * xs).sum(axis=0) / w.sum()
    wsd = np.sqrt((w[:, None] * (xs - wmean) ** 2).sum(axis=0) / w.sum())
    failed = int((~ok & ~refused).sum())
    shift = np.divide(wmean - mean, sd, out=np.full_like(sd, np.nan), where=sd > 0)
    width = np.divide(wsd, sd, out=np.full_like(sd, np.nan), where=sd > 0) - 1.0
    finite_shift = np.flatnonzero(np.isfinite(shift))
    largest = int(finite_shift[np.argmax(np.abs(shift[finite_shift]))]) if len(finite_shift) else None
    moments = dict(effective_points=ess,
                   corrected_shift=dict(zip(names, [float(v) if np.isfinite(v) else None for v in shift])),
                   corrected_width=dict(zip(names, [float(v) if np.isfinite(v) else None for v in width])),
                   largest_shift=None if largest is None else float(abs(shift[largest])),
                   largest_shift_parameter=None if largest is None else names[largest])
    if failed:
        moments = dict.fromkeys(moments)
    return dict(**moments, failed=failed,
                correction_unavailable=f"{failed} unresolved true evaluations" if failed else None,
                iteration=iteration, points=int(len(x)), not_finite=int((~ok).sum()),
                refused=int(refused.sum()), refused_value=refused_value,
                truth_minus_network=dict(mean=float(d.mean()), sd=float(d.std()), rms=float(np.sqrt(np.mean(d ** 2))),
                                         largest=float(np.abs(d).max())),
                median_depth_below_best=float(np.median(best - true[ok])),
                above_best_training_value=int(np.sum(true[ok] > best)), best_true_here=float(true[ok].max()))


def narrower_than_the_run(prior_bounds, names, lower, upper):
    """The parameters in which the likelihood's box ends inside the box the run was trained in
    (its metadata.json). The posterior samples lie in the run's box: beyond a narrower one the
    prior gives -inf, which would be counted as a point the likelihood refused, without a
    word. That happens when a run's domain file is made again, narrower, after the run. A
    wider box, or no bound at all (a run whose box came from prior.n_sigma), is no trouble."""
    out = []
    for name, lo, hi in zip(names, lower, upper):
        prior_lo, prior_hi = prior_bounds[name]
        slack = 1e-9 * max(abs(lo), abs(hi), hi - lo)
        if (prior_lo is not None and prior_lo > lo + slack) or (prior_hi is not None and prior_hi < hi - slack):
            out.append(name)
    return out


def report_line(entry, ndim):
    """A network's line of the log, from its entry. A function of its own so that a test runs
    it: written inside main, two of its pieces once stood side by side without a +, which
    calls a string, and the job stopped there for every network with a correction, after its
    true evaluations and before true_check.json."""
    d, used = entry["truth_minus_network"], entry["points"] - entry["not_finite"]
    if entry.get("correction_unavailable"):
        corrected = "correction unavailable: " + entry["correction_unavailable"]
    else:
        shift = (f"largest corrected mean shift {entry['largest_shift']:.2f} sigma ({entry['largest_shift_parameter']})"
                 if entry["largest_shift"] is not None else "no corrected mean shift (zero widths)")
        corrected = f"{entry['effective_points']:.0f} effective points; {shift}"
    left_out = (f" ({entry['refused']} refused, {entry['not_finite'] - entry['refused']} NaN or +inf)"
                if entry["not_finite"] else "")
    if d is None:
        return f"network {entry['iteration']}: {used} of {entry['points']} with a true value{left_out}; {corrected}"
    return (f"network {entry['iteration']}: {used} of {entry['points']} with a true value{left_out}; "
            f"truth minus network: mean {d['mean']:+.2f}, spread {d['sd']:.2f}, rms {d['rms']:.2f}, "
            f"largest {d['largest']:.1f}; {corrected}; median depth below the best training value "
            f"{entry['median_depth_below_best']:.1f} (n / 2 = {ndim / 2:.1f})")


def chain_line(iteration, drawn, total, problem=None):
    """A network's line of the log before its true evaluations: how many of its posterior
    samples are evaluated, and, when the chain's own record says its samples cannot carry a
    mean over the posterior (`problem`: sampling.provenance.sampling_correction_problem), what
    that will cost. Said here, before the evaluations, and not only in the result after them."""
    line = f"network {iteration}: {drawn} of its {total} posterior samples"
    if problem:
        line += (f"\n  WARNING, network {iteration}: {problem}. Its true values and the pointwise errors will be saved; the "
                 "corrected moments, the effective points and the evidence correction will not. A chain that was too short "
                 "is replaced by running analysis/report.py with more --steps; a cell whose shape the bridge's Gaussian "
                 "mixture does not follow stays marked at any length.")
    return line


@stop_peers_on_error
def main():
    from domain import apply_domain, read_domain
    from loading import build_in_turn
    from refused import read_record, refuse_logged_errors
    from likelihood.cobaya import CobayaLikelihood
    from run.storage import atomic_path
    from utils.mpi_utils import broadcast_and_evaluate

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--cobaya", type=Path, required=True)
    parser.add_argument("--domain", type=Path, help="the run's domain_bounds.json, when the launcher gave it one")
    parser.add_argument("--networks", type=int, nargs="*", help="default: the last network before the boosts and the last network")
    parser.add_argument("--points", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20000, help="steps of a T = 1 chain sampled here")
    parser.add_argument("--discard-chains", action="store_true", help="do not keep the T = 1 chains sampled here (posterior_it_N.npz)")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    truth = build_in_turn(comm, lambda: CobayaLikelihood(str(args.cobaya)), label="[true check]")   # CLIENT_LOAD_GROUP
    if args.domain is not None:
        apply_domain(truth, read_domain(args.domain))
    refuse_logged_errors(truth, rank, label="[true check]", value_file=args.run / "refused_value.json")   # design: the run's value

    points, blocks, run, refused_value = None, [], None, None
    sampled_targets, sampled_diagnostics = {}, {}
    if rank == 0:
        from report import Run, sample_posterior          # TensorFlow on rank 0 only

        run = Run(args.run, Path(os.environ.get("CLIENT_REPO", Path(__file__).resolve().parents[2])))
        if run.names != truth.param_names:
            raise RuntimeError(f"the run's parameters {run.names} are not the likelihood's {truth.param_names}")
        narrower = narrower_than_the_run(truth.prior_bounds, run.names, run.lower, run.upper)
        if narrower:
            raise RuntimeError(f"the likelihood's box is narrower than the box the run was trained in (metadata.json) in "
                               f"{', '.join(narrower[:8])}" + (f" and {len(narrower) - 8} more" if len(narrower) > 8 else "")
                               + ": posterior samples beyond it would be counted as refused points. Give the domain file the "
                               "run was made with (--domain).")
        # The refused value is the run's, not this job's: it is in the run's training set, and with
        # it in this job's environment a refused point comes back carrying it, without it as -inf.
        record = read_record(args.run / "refused_value.json")
        recorded = record.get("value") if isinstance(record, dict) else None
        refused_value = truth.refused_value if recorded is None else float(recorded)
        if refused_value is not None:
            print(f"[true check] the run's refused value is {refused_value:g}"
                  + ("" if recorded is not None else " (CLIENT_REFUSED_VALUE; the run has no refused_value.json)")
                  + ": a point that comes back with it, or as -inf, has zero importance weight", flush=True)
        if truth.refused_value not in (None, refused_value):
            print(f"[true check] CLIENT_REFUSED_VALUE gives refused points {truth.refused_value:g} in this job, which is "
                  "not the run's value: a point with either is counted as refused", flush=True)
        out = run.dir / "analysis"
        out.mkdir(exist_ok=True)
        networks = args.networks
        if not networks:
            last = max(run.models)
            before = run.boost["start_iteration"] if run.boost and run.boost["start_iteration"] in run.models else None
            networks = sorted({last} | ({before} if before is not None else set()))
        rng = np.random.default_rng(args.seed)
        for it in networks:
            file = out / f"posterior_it_{it}.npz"
            from sampling.provenance import cached_samples, save_samples, sampler_identity, sampling_correction_problem
            target = run.target_identity(it)
            sampled_targets[it] = target
            sampler = sampler_identity(run.surrogate(it, region=True).region)
            info = {}
            samples = cached_samples(file, target, sampler=sampler, diagnostics=info)
            if samples is None:
                import tensorflow as tf

                tf.keras.utils.set_random_seed(1000 + it)
                chain, _ = sample_posterior(run, it, args.steps, 256, rng)
                samples = chain.reshape(-1, run.ndim)
                info = dict(sampler, **run.mixtures.get(it, {}))
                if not args.discard_chains:
                    save_samples(file, samples, run.names, target, sampling=info)
            sampled_diagnostics[it] = info
            x = samples[rng.choice(len(samples), min(args.points, len(samples)), replace=False)].astype(np.float32)
            blocks.append((it, x, run.predict(it, x, region=True)))
            print(chain_line(it, len(x), len(samples), sampling_correction_problem(info)), flush=True)
        points = np.concatenate([x for _, x, _ in blocks]).astype(float)
        started = time.time()

    inputs, values = broadcast_and_evaluate(samples=points, evaluator=truth.evaluate_with_derived)

    if rank == 0:
        targets = np.asarray(truth.split_evaluations(values)[0], float).ravel()
        seconds = time.time() - started
        report = dict(run=run.dir.name, points=args.points, seed=args.seed, ranks=comm.size, seconds=seconds,
                      best_training_value=run.best, parameters=run.names, networks=[])
        print(f"\n{len(targets)} true evaluations in {seconds:.0f} s on {comm.size} ranks; best training value of the run {run.best:.2f}\n")
        start = 0
        for it, x, network in blocks:
            true = targets[start:start + len(x)]
            start += len(x)
            refused = refused_points(true, refused_value, truth.refused_value)     # the run's value, and this job's
            entry = network_entry(it, x, network, true, run.best, run.names, refused, refused_value)
            entry["target"] = sampled_targets[it]
            entry["sampling"] = sampled_diagnostics[it]
            sampling_problem = sampling_correction_problem(entry["sampling"])
            if sampling_problem:
                for key in ("effective_points", "corrected_shift", "corrected_width", "largest_shift", "largest_shift_parameter"):
                    entry[key] = None
                entry["correction_unavailable"] = "; ".join(filter(None, [entry["correction_unavailable"], sampling_problem]))
            report["networks"].append(entry)
            with atomic_path(run.dir / "analysis" / f"true_check_it_{it}.npz") as temporary:
                np.savez_compressed(temporary, x=x, network=network, truth=true, refused=refused,
                                    names=np.array(run.names), target_json=json.dumps(entry["target"], sort_keys=True),
                                    sampling_json=json.dumps(entry["sampling"], sort_keys=True),
                                    sampling_correction_unavailable=sampling_problem or "")
            print(report_line(entry, run.ndim), flush=True)
        # Networks checked earlier stay in the file: a later call for other networks adds to it.
        file = run.dir / "analysis" / "true_check.json"
        try:
            earlier = json.loads(file.read_text())["networks"] if file.exists() else []
        except (ValueError, KeyError):
            earlier = []
        done = {entry["iteration"] for entry in report["networks"]}
        report["networks"] = sorted([e for e in earlier if e["iteration"] not in done] + report["networks"],
                                    key=lambda entry: entry["iteration"])
        with atomic_path(file) as temporary:
            temporary.write_text(json.dumps(report, indent=1))
        print(f"\nwritten: {run.dir / 'analysis' / 'true_check.json'}")


if __name__ == "__main__":
    main()
