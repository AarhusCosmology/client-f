import argparse
from functools import wraps
from run.storage import PhaseStore, RunLock, save_model
import time
from pathlib import Path

import numpy as np

from likelihood.base import build_likelihood
from utils.mpi_utils import (
    bcast,
    bcast_idle,
    broadcast_and_evaluate,
    get_size,
    is_master,
    is_mpi_available,
    print_master,
)


def draw_peak_samples(cfg, likelihood, inputs, targets, master,
                      n_draw, seed, context=""):
    """Points drawn from the Gaussian the quadratic response surface implies.

    Returns the NEW (inputs, targets) only; the caller decides where they go. The
    surface's vertex is a good estimate of where the maximum is, but a poor training
    point on its own: its log-likelihood sits hundreds of units above the rest of the
    design, so the msre weighting collapses onto it and the network fits one point. A
    *sample* from the same fit lands 0.5*T*ndim +/- 0.5*T*sqrt(2*ndim) below the vertex
    instead -- a few loss margins down, spread over a comparable width -- which is the
    regime the weighting expects.

    Drawn at the acquisition temperature, so the injected points span the log-likelihood
    range the sampler will actually visit. Costs n_draw true evaluations per call.

    Note the interaction with training.msre_ess_floor: the injected set is bimodal, and
    the floor targets an effective sample size over the *whole* training set, so it is
    forced to a very large c unless n_draw > alpha * n_train / (1 - alpha). See the
    README.
    """
    if master:
        from dataset.peak_samples import INJECT_WINDOW, gaussian_peak_samples

        bounds = np.array(
            [[lo, hi] for lo, hi in likelihood.prior_bounds.values()], dtype=float
        )
        injected, info = gaussian_peak_samples(
            inputs,
            np.asarray(targets).reshape(-1),
            n_draw=n_draw,
            temperature=cfg.acquisition.target_temperature,
            bounds=bounds,
            seed=seed,
            window=INJECT_WINDOW,
        )
        fitted = (f"{len(targets)} points" if info["window_depth"] is None else
                  f"the {info['n_fit']} of {info['n_all']} points within "
                  f"{info['window_depth']:.0f} log-units of the best ({info['best_fitted']:.1f}; "
                  f"{info['n_projected']} wrong-sign curvatures, width capped along "
                  f"{info['n_capped']} directions)")
        print_master(
            f"{context}Injecting {len(injected)} points from the quadfit Gaussian at "
            f"T={cfg.acquisition.target_temperature:g}: fitted on "
            f"{fitted}, vertex value "
            f"{info['model_value']:.1f}, predicted deficit {info['deficit'].mean():.1f} "
            f"+/- {info['deficit'].std():.1f} (law {info['deficit_mean_expected']:.1f} "
            f"+/- {info['deficit_sd_expected']:.1f}), draw acceptance "
            f"{info['acceptance']:.2f}, fit rms {info['resid']:.1f}"
        )
    else:
        injected = None

    journal = getattr(likelihood, "_client_calls", None)
    previous_stage = None if journal is None else journal.stage
    if journal is not None:
        journal.stage = previous_stage + "/injection"
    try:
        new_inputs, new_targets, new_derived = evaluate_true(likelihood, injected)
    finally:
        if journal is not None:
            journal.stage = previous_stage
    if not master:
        return None, None, None

    # A point the backend refuses (a Boltzmann code's error, turned into -inf by the
    # backend) is left out, as everywhere else in the loop; all of them may be.
    ok = np.isfinite(new_targets)
    if ok.any():
        print_master(
            f"   injected log-likelihood range "
            f"[{new_targets[ok].min():.1f}, {new_targets[ok].max():.1f}]"
        )
    if not ok.all():
        print_master(
            f"   {ok.size - np.count_nonzero(ok)} of the {ok.size} injected points have no "
            f"finite log-likelihood and are left out"
        )
    return new_inputs[ok], new_targets[ok], new_derived[ok]


def evaluate_true(likelihood, samples):
    """True evaluations of master-owned `samples` across MPI ranks.

    Returns (inputs, targets, derived) with derived of shape (n, n_derived): every
    true evaluation already computes the backend's derived parameters, so they are
    kept at no extra cost. Workers get empty arrays back, as from
    broadcast_and_evaluate.
    """
    inputs, values = broadcast_and_evaluate(samples=samples, evaluator=_evaluator(likelihood))
    targets, derived = likelihood.split_evaluations(values)
    return inputs, targets, derived


def _evaluator(likelihood):
    """The likelihood's evaluate_with_derived, through the run's call journal when it has one."""
    journal = getattr(likelihood, "_client_calls", None)
    return likelihood.evaluate_with_derived if journal is None else lambda x: journal.evaluate(likelihood, x)


def draw_boost_points(cfg, predict, lower, upper, tempered, iteration, dataset, directory, master):
    """The points of a boost iteration: the master draws them while the workers sleep.

    Returns (points, draw, boosting) on every rank; points and draw are None on a worker.
    boosting comes back False, on every rank, when the network has no mode to draw posterior
    samples from (sampling.modes.NoModeFound): it is not ready for a boost, and the iteration
    then acquires as an ordinary one does, which is what such a network needs. The master
    writes RUN/boost/fallback_it_N.json. predict, tempered, dataset and directory are the
    master's; a worker passes None.
    """
    points = draw = None
    boosting = True
    if master:
        from dataset.boost import draw_boost, record_fallback
        from sampling.modes import NoModeFound

        try:
            points, draw = draw_boost(
                predict, lower, upper, tempered, cfg.sampling.temperature, cfg.acquisition.boost,
                seed=None if cfg.seed is None else cfg.seed + 104729 * (iteration + 1),
                log=print_master,
                dataset=dataset,
            )
        except NoModeFound as error:
            points = draw = None
            boosting = False
            record_fallback(directory, iteration, error)
            print_master(
                f"   {error}. The network is not ready for a boost: this iteration acquires "
                "as an ordinary one does instead"
            )
    # The workers slept through the master's sampling; they learn here whether it is a boost.
    return points, draw, bcast_idle(boosting)


def tempered_convergence(cfg, metrics_tracker, iteration, use_convergence, boosts_follow):
    """The criterion of the ordinary loop on its own true evaluations (convergence.metric:
    tempered_batch_scatter; convergence/convergence.py). Master only, in an ordinary iteration,
    once network `iteration` is trained. Prints the verdict, records it, and returns
    (stop, hand over, verdict): stop when the loop has converged and nothing follows; hand
    over when it has converged and the run has boosts still to begin (`boosts_follow`).
    With the number of iterations given (-i) the verdict is printed and nothing follows."""
    from convergence.convergence import TEMPERED_BATCH_SCATTER, tempered_verdict

    scatters = {k: m.held_out_sd for k, m in metrics_tracker.acquisition_metrics.items()}
    verdict = tempered_verdict(scatters, iteration, cfg.sampling.temperature,
                               cfg.convergence.threshold, cfg.convergence.consecutive)
    first = iteration - cfg.convergence.consecutive
    if first < 0:
        print_master(f"{TEMPERED_BATCH_SCATTER}: not yet, it takes the tempered batches of "
                     f"{cfg.convergence.consecutive} networks")
        return False, False, verdict
    shown = ", ".join("not known" if v is None else f"{v:.3g}" for v in verdict["used"])
    print_master(
        f"{TEMPERED_BATCH_SCATTER}: {shown} at the tempered batch"
        + (f"es of networks {first} to {iteration - 1}" if cfg.convergence.consecutive > 1 else f" of network {first}")
        + f", against {cfg.convergence.threshold:g} x T = {verdict['limit']:.3g}: "
        + ("converged" if verdict["converged"] else "not converged")
    )
    if verdict["value"] is not None:
        metrics_tracker.add_convergence_metrics(
            iteration=iteration,
            metric_value=verdict["value"],
            converged=verdict["converged"],
            metric_name=TEMPERED_BATCH_SCATTER,
        )
        metrics_tracker.save_all_metrics()
    if not verdict["converged"]:
        return False, False, verdict
    if not use_convergence:
        print_master("   nothing follows from it: the number of iterations was given (-i)")
        return False, False, verdict
    if boosts_follow:
        return False, True, verdict
    print_master("Convergence criterion met, stopping...")
    return True, False, verdict


def audit_network(cfg, likelihood, master, run_audit=False, evaluation_journal=None, penalty_source=None, **arguments):
    """Audit collectives transport evaluation errors before all ranks stop together."""
    from utils.mpi_utils import _MESSAGE_TOLD, broadcast_and_evaluate_checked

    def batch(points):
        x, values, errors = broadcast_and_evaluate_checked(
            points, _evaluator(likelihood), 1 + likelihood.n_derived)
        y, d = likelihood.split_evaluations(values)
        return x, y, d, errors

    if not master:
        while True:
            command = bcast_idle(None)
            if isinstance(command, dict) and "audit_error" in command:
                raise RuntimeError(command["audit_error"])
            if not command:
                return None
            batch(None)
    if not run_audit:
        bcast_idle(False)
        return None
    from training.audit import audit_optima

    def evaluate(points):
        bcast_idle(True)
        x, y, d, errors = batch(points)
        if evaluation_journal is not None:
            import json
            path = Path(evaluation_journal)
            path.parent.mkdir(parents=True, exist_ok=True)
            by_point = {np.asarray(e["x"], float).tobytes(): e for e in errors}
            with path.open("a") as stream:
                for point, value in zip(x, y):
                    error = by_point.get(np.asarray(point, float).tobytes())
                    record = dict(x=point.tolist(), value=float(value) if np.isfinite(value) else None,
                                  status=("recoverable_exception" if error["recoverable"] else "fatal_exception")
                                  if error else ("finite" if np.isfinite(value) else "nonfinite"),
                                  error=error)
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
        fatal = [e for e in errors if not e["recoverable"]]
        if fatal:
            raise RuntimeError(f"Fatal audit backend evaluation: {fatal[0]}")
        return x, y, d

    termination = False
    try:
        if penalty_source is not None and cfg.acquisition.audit.failure_policy == "penalize":
            from training.audit_failures import load_penalties
            arguments["penalties"] = load_penalties(*penalty_source)
        return audit_optima(evaluate=evaluate, config=cfg.acquisition.audit,
                            log=print_master, **arguments)
    except Exception as error:
        # Trimmed for the workers' receive (32 KiB): a long TensorFlow error does not fit it.
        termination = {"audit_error": str(error)[:_MESSAGE_TOLD]}
        raise
    finally:
        bcast_idle(termination)


def check_settings_for(cfg, ndim):
    """What the settings parser cannot know and the loop used to meet only after the initial
    design was paid for: the names Keras resolves, and the walkers against the dimension."""
    import tensorflow as tf

    problems = []
    # CLiENT's own: msre (training/losses.py) and alsing (model/activations.py).
    for setting, name, own, resolve in (("training.loss", cfg.training.loss, "msre", tf.keras.losses.get),
                                        ("model.activation", cfg.model.activation, "alsing", tf.keras.activations.get)):
        if name != own:
            try:
                resolve(name)
            except (ValueError, TypeError):
                problems.append(f"{setting}: {name!r} is neither CLiENT's own nor a name Keras knows")
    ensembles = [("sampling.n_walkers", cfg.sampling.n_walkers)]
    if cfg.acquisition.boost is not None:
        ensembles.append(("acquisition.boost.walkers", cfg.acquisition.boost.walkers))
    for setting, walkers in ensembles:
        if walkers < 2 * ndim:
            problems.append(f"{setting}: {walkers} walkers for {ndim} parameters; the ensemble "
                            f"sampler needs at least twice the dimension ({2 * ndim})")
    if problems:
        raise ValueError("The settings do not fit this problem: " + "; ".join(problems))


def build_initial_design(cfg, likelihood, master):
    """The initial design: a Latin hypercube (or whatever prior.sampling_strategy asks).

    Returns the samples on master only (None elsewhere), matching sample_prior's contract
    so the caller's broadcast_and_evaluate is unchanged.
    """
    if not master:
        return None
    from sampling.prior_sampler import sample_prior

    print_master(
        f"Generating {cfg.prior.n_samples} {cfg.prior.sampling_strategy} samples")
    return sample_prior(likelihood=likelihood, n_samples=cfg.prior.n_samples,
                        strategy=cfg.prior.sampling_strategy)


def inject_peak_samples(cfg, likelihood, inputs, targets, derived, master):
    """One-shot injection after the initial design (`prior.n_inject`)."""
    new_inputs, new_targets, new_derived = draw_peak_samples(
        cfg, likelihood, inputs, targets, master,
        n_draw=cfg.prior.n_inject, seed=cfg.seed,
    )
    if not master:
        return inputs, targets, derived

    inputs = np.vstack([inputs, new_inputs])
    targets = np.concatenate([targets, new_targets])
    derived = np.vstack([derived, new_derived])
    print_master(f"   training set now {len(targets)} points")
    return inputs, targets, derived


def _stop_peers_on_error(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException:
            import traceback
            from utils.mpi_utils import _get_communicator
            comm = _get_communicator()
            if comm is not None:
                traceback.print_exc()
                comm.Abort(1)
            raise
    return guarded


def schedule_note(environ=None):
    """A warning when CLIENT_MPI_SCHEDULE names no schedule (static, chunked), or None. The
    default, static, is then used."""
    import os

    environ = os.environ if environ is None else environ
    schedule = environ.get("CLIENT_MPI_SCHEDULE", "").strip().lower()
    if schedule and schedule not in ("static", "chunked"):
        return (f"CLIENT_MPI_SCHEDULE={environ['CLIENT_MPI_SCHEDULE']!r} is not a schedule (static, chunked): "
                "the default, static, is used.")
    return None


def rewind_to_start(run, args):
    """-s N in a run that has gone further than network N. The loop would do iteration N again
    in place and leave the later networks and training sets where they are: every phase record
    stays complete, and the next start without -s takes the latest files, which are the earlier
    line's and do not hold the points of the iteration done again. So it is refused, unless
    --rewind sets what belongs to the later iterations aside first (run.storage.set_later_aside)."""
    from run.run import latest_iteration
    from run.storage import set_later_aside

    if run.is_new or args.start is None:
        if args.rewind:
            raise RuntimeError("--rewind goes with -s N on a run directory: the network to go back to must be named")
        return
    latest = latest_iteration(run.run_dir)
    if args.start >= latest:
        if args.rewind:
            print_master(f"--rewind: the run holds nothing after iteration {args.start}; nothing is set aside")
        return
    if not args.rewind:
        raise RuntimeError(
            f"{run.run_dir} has gone on to iteration {latest}: -s {args.start} would do iteration {args.start} "
            "again in place and leave the later networks and training sets, which the next start without -s "
            "would continue from, without the points of the iteration done again. Add --rewind to set the "
            f"files of the iterations after {args.start} aside under {run.run_dir.name}/interrupted/ (nothing is "
            f"deleted) and go on from network {args.start}; to keep this run as it is, make another directory "
            "from it (for boosts from an earlier network: hpc/analysis/branch_run.py).")
    set_later_aside(run.run_dir, args.start, log=print_master)


def parse_arguments():
    """The command line. --help, a wrong argument and a combined emulator stop every rank alike
    here, before any rank waits for another: ordinary exits, not a job to abort."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_or_dir", help="Input YAML file (new run) or run directory (continue)"
    )
    parser.add_argument(
        "-n", "--name", help="Run name/tag for organization (new runs only)"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="results",
        help="Base output directory (new runs only)",
    )
    parser.add_argument(
        "-r",
        "--retrain",
        action="store_true",
        help="Force retrain even if a saved model exists",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        help="Starting iteration (continue only, auto-detected if omitted)",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        help="Number of (additional) iterations to run (overrides convergence criterion)",
    )
    parser.add_argument("--rewind", action="store_true",
                        help="With -s N in a run that has gone further: set the files of the iterations after N "
                             "aside under RUN/interrupted/ (nothing is deleted) and go on from network N")
    parser.add_argument("--accept-identity-change", action="store_true",
                        help="Continue a run whose recorded objective identity differs from the current one, "
                             "after checking that the objective itself is unchanged; the old record is kept")
    args = parser.parse_args()
    if (Path(args.input_or_dir) / "combine.json").exists():
        raise SystemExit(f"{args.input_or_dir} is a combined emulator (client.py combine, its combine.json), "
                         "not a run to continue: continue its parts")
    return args


def main():
    run_client(parse_arguments())


@_stop_peers_on_error
def run_client(args):
    using_mpi = is_mpi_available()
    master = is_master()
    mpi_status = f"Enabled ({get_size()} processes)" if using_mpi else "Disabled"
    print_master(f"\nMPI status: {mpi_status}\n")

    # ---- Configuration (master loads the run and broadcasts it to workers) ----
    if master:
        from run.metrics import MetricsTracker
        from run.run import Run

        run = Run.from_args(args)
        if run.is_new:
            run.create_directories(args.input_or_dir)

        run_lock = RunLock(run.run_dir)
        import atexit
        atexit.register(run_lock.close)
        from run.storage import recover_rewind
        if not run.is_new and recover_rewind(run.run_dir, log=print_master):
            # The directory may have held only part of the old line when from_args first
            # chose an automatic start. Choose again after finishing the recorded rewind.
            run = Run.from_args(args)
        phases = PhaseStore(run.run_dir)
        # A phase the previous job began and did not commit is set aside and done again; where
        # the run starts is read once more, from what is committed.
        if not run.is_new and phases.recover(log=print_master):
            run = Run.from_args(args)
        rewind_to_start(run, args)
        phases.check()

        metrics_tracker = MetricsTracker(
            results_dir=run.run_dir,
            start_iteration=run.start_iteration,
            preserve_start_metrics=False if run.is_new else True,
        )

        print_master(f"Run ID: {run.run_id}")
        print_master(f"Run mode: {run.mode}")
        print_master(f"Run directory: {run.run_dir}\n")
    else:
        run = None
        metrics_tracker = None

    run = bcast(run)
    cfg = run.config

    # Seed before anything draws.
    if cfg.seed is not None:
        if master:
            import tensorflow as tf

            tf.keras.utils.set_random_seed(cfg.seed)
        else:
            # What set_random_seed does besides TensorFlow's own seed. A rank that only
            # evaluates the likelihood has no use for that one, and importing TensorFlow
            # for it cost 3.7 s and about 450 MB on every rank.
            import random

            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
        print_master(f"Seed: {cfg.seed} (AIES proposals excepted -- XLA discards them)")
    note = schedule_note()
    if note:
        print_master(f"WARNING: {note}")

    # ---- Likelihood ----
    print_master("initializing likelihood...")

    likelihood = build_likelihood(
        wrapper=cfg.likelihood.wrapper,
        input_path=cfg.likelihood.input,
    )
    if cfg.prior.n_sigma is not None:
        likelihood.restrict_prior_bounds(cfg.prior.n_sigma)
    if likelihood.n_derived:
        print_master(
            f"Recording {likelihood.n_derived} derived parameters with every true "
            f"evaluation: {', '.join(likelihood.derived_names)}"
        )

    from run.objective_calls import ObjectiveCalls
    from utils.mpi_utils import _get_rank
    likelihood._client_calls = ObjectiveCalls(run.run_dir, _get_rank())
    if master:
        from run.identity import identity, verify
        verify(run.run_dir, identity(likelihood, cfg), new=run.is_new,
               accept_change=args.accept_identity_change)

    # ---- Surrogate metadata ----
    if master:
        from dataset.dataset import TrainingDataset
        from likelihood.surrogate import SurrogateLikelihood, SurrogateMetadata

        surrogate_metadata = SurrogateMetadata.from_likelihood(likelihood)
        check_settings_for(cfg, len(surrogate_metadata.param_names))
        if run.is_new:
            surrogate_metadata.save(run.run_dir / "metadata.json")

        # Written on continued runs too, unlike metadata.json: a run recorded before
        # derived parameters existed starts recording them when it is continued.
        derived_metadata = None
        if not run.is_new:
            from derived.metadata import changed_since_recorded

            change = changed_since_recorded(run.derived_data_dir, likelihood.derived_names)
            if change is not None:
                added, removed = change
                print_master(
                    "Warning: the derived parameters differ from those this run was "
                    f"recorded with (added: {', '.join(added) or 'none'}; removed: "
                    f"{', '.join(removed) or 'none'}). With some backends that changes the "
                    "likelihood itself: Cobaya + CAMB computes sigma8 with high-precision "
                    "transfer functions, which moved Planck TT by 1.5 in log L at a fixed "
                    "point. Earlier targets may not be comparable with new ones."
                )
        if likelihood.n_derived:
            from derived.metadata import DerivedMetadata

            derived_metadata = DerivedMetadata.from_likelihood(likelihood)
            derived_metadata.save(run.derived_data_dir)
            if cfg.derived is not None:
                print_master(
                    "Derived-parameter emulator retrained every iteration, on "
                    f"{', '.join(derived_metadata.inputs)}"
                )
    else:
        surrogate_metadata = None


    # ---- Training data (new run) ----
    if run.is_new:
        prior_samples = build_initial_design(cfg, likelihood, master)

        if master:
            phases.begin("initial")
        inputs, targets, derived = evaluate_true(likelihood, prior_samples)

        if master:
            valid = np.isfinite(targets)
            if not valid.all():
                inputs, targets, derived = inputs[valid], targets[valid], derived[valid]
                print_master(
                    f"Warning: filtered {valid.size - np.count_nonzero(valid)} targets with non-finite loglkl values"
                )


        if cfg.prior.n_inject > 0:
            inputs, targets, derived = inject_peak_samples(
                cfg, likelihood, inputs, targets, derived, master
            )

        if master:
            dataset = TrainingDataset(
                inputs=inputs,
                targets=targets,
                likelihood=likelihood,
                n_neighbors=cfg.acquisition.n_neighbors,
                target_temperature=cfg.acquisition.target_temperature,
                derived=derived,
            )
            dataset.save(run.training_data_dir / "data_it_0.csv")
            dataset.save_derived(run.derived_data_dir / "derived_it_0.csv")
            phases.complete("initial", [run.training_data_dir / "data_it_0.csv",
                                        run.derived_data_dir / "derived_it_0.csv",
                                        run.run_dir / "metadata.json"])
        else:
            dataset = None

    # --- Training data (continue run) ----
    else:
        if master:
            dataset = TrainingDataset.load(
                training_data_dir=run.training_data_dir,
                likelihood=likelihood,
                n_neighbors=cfg.acquisition.n_neighbors,
                target_temperature=cfg.acquisition.target_temperature,
                iteration=run.start_iteration,
                derived_data_dir=run.derived_data_dir,
            )
            print_master(
                f"Loaded {len(dataset.inputs)} training samples from data_it_{run.start_iteration}.csv"
            )
            # Points selected by `client.py augment` for this training set (dataset/augment.py).
            augment_path = run.run_dir / "augmentation" / f"augment_it_{run.start_iteration}.csv"
            if augment_path.exists():
                from dataset.augment import read_augmentation

                extra_inputs, extra_targets = read_augmentation(augment_path, likelihood.param_names)
                dataset.add_data(inputs=extra_inputs, targets=extra_targets)
                print_master(
                    f"   plus {len(extra_targets)} augmentation points from {augment_path.name}: "
                    f"{len(dataset.inputs)} training samples"
                )
            start_model = run.trained_models_dir / f"model_it_{run.start_iteration}.keras"
            # Points the members' pilot added after training this iteration's saved network
            # (training.member_pilot, training/members.py), before its audit's.
            pilot_path = run.trained_models_dir / "members" / f"pilot_it_{run.start_iteration}.csv"
            if pilot_path.exists() and start_model.exists() and not run.retrain:
                from training.members import load_pilot

                extra_inputs, extra_targets, extra_derived = load_pilot(
                    run.run_dir, run.start_iteration, likelihood.param_names, likelihood.derived_names)
                if len(extra_targets):
                    dataset.add_data(inputs=extra_inputs, targets=extra_targets, derived=extra_derived)
                    print_master(
                        f"   plus {len(extra_targets)} pilot points from {pilot_path.name}: "
                        f"{len(dataset.inputs)} training samples"
                    )
            # Points the audit added while training this iteration's saved network
            # (training/audit.py). A network that will be retrained is audited again.
            audit_path = run.run_dir / "audit" / f"audit_it_{run.start_iteration}.csv"
            if audit_path.exists() and start_model.exists() and not run.retrain:
                from training.audit import load_audit

                extra_inputs, extra_targets, extra_derived = load_audit(
                    audit_path.parent, run.start_iteration, likelihood.param_names,
                    likelihood.derived_names)
                if len(extra_targets):
                    dataset.add_data(inputs=extra_inputs, targets=extra_targets,
                                     derived=extra_derived)
                    print_master(
                        f"   plus {len(extra_targets)} audit points from {audit_path.name}: "
                        f"{len(dataset.inputs)} training samples"
                    )
            if dataset.derived_names:
                n_known = int(np.count_nonzero(np.isfinite(dataset.derived).any(axis=1)))
                print_master(
                    f"   derived parameters known for {n_known}/{len(dataset.inputs)} of them"
                )
        else:
            dataset = None

    # ---- Convergence metric ----
    previous_chain_summary = None
    if master:
        from convergence.convergence import build_convergence_metric

        metric = build_convergence_metric(cfg.convergence.metric)
        if not run.is_new and run.start_iteration > 0:
            previous_chain_summary = metric.load_chain_summary(
                run.convergence_stats_dir, run.start_iteration - 1
            )
            print_master(
                f"Loaded previous chain summary from chain_summary_it_{run.start_iteration - 1}.npz"
            )
    else:
        metric = None

    final_iteration = run.final_iteration
    use_convergence = run.use_convergence

    # Opt-in (convergence.metric: tempered_batch_scatter): the ordinary loop converges on the
    # error of its networks at their tempered batches, and hands over to the boosts when the
    # run has any (convergence/convergence.py, docs/tempered_convergence.md).
    from convergence.convergence import TEMPERED_BATCH_SCATTER

    batch_criterion = cfg.convergence.metric == TEMPERED_BATCH_SCATTER
    # The iteration at which the boosts begin: acquisition.boost.start_iteration, or the
    # earlier one at which the ordinary loop of this run converged (RUN/boost/start.json).
    boost_start = None if cfg.acquisition.boost is None else cfg.acquisition.boost.start_iteration
    if batch_criterion and cfg.acquisition.boost is not None:
        if master:
            from dataset.boost import recorded_start

            earlier = recorded_start(run.run_dir / "boost")
            if earlier is not None and earlier < boost_start:
                boost_start = earlier
                print_master(f"The ordinary loop of this run converged at iteration {earlier}: its boosts began there")
        boost_start = bcast(boost_start)
    # Opt-in (acquisition.boost.boosts): a run of no given length ends after that many boosts.
    if use_convergence and cfg.acquisition.boost is not None and cfg.acquisition.boost.boosts is not None:
        final_iteration = min(final_iteration, boost_start + cfg.acquisition.boost.boosts)

    # Hoist loop-invariant values and master-only imports out of the iteration loop.
    if master:
        ndim = likelihood.ndim
        prior_bounds = np.asarray(list(likelihood.prior_bounds.values()), dtype=float)
        prior_lower = prior_bounds[:, 0]
        prior_upper = prior_bounds[:, 1]
        inverse_sampling_temperature = 1.0 / cfg.sampling.temperature

        import tensorflow as tf

        from dataset.acquisition import select_points
        from model.network import build_model, load_model, transfer_warm_start
        from sampling.base import build_sampler
        from training.losses import build_loss
        from training.training import save_history, train_model

        def loss_for(values):
            """The training loss for a network trained on the targets `values`."""
            return build_loss(
                name=cfg.training.loss,
                sigma_level=cfg.training.sigma_level,
                chi2_dof=ndim,
                max_loglkl=float(np.max(values)),
                targets=values,
                ess_floor=cfg.training.msre_ess_floor,
            )

    # ---- Main loop ----
    for iteration in range(run.start_iteration, final_iteration + 1):
        if iteration > final_iteration:  # the run was shortened at the hand-over to its boosts
            break
        print_master(f"\n--- Iteration {iteration}/{final_iteration} ---")
        if master:
            iteration_start_time = time.monotonic()
            model_path = run.trained_models_dir / f"model_it_{iteration}.keras"
            # Train for new/retrain runs, after acquisition updates, or if a continued run has no saved model.
            should_train = (
                run.is_new
                or run.retrain
                or iteration != run.start_iteration
                or not model_path.exists()
            )
            if should_train:
                if not model_path.exists() and not run.is_new:
                    print_master(
                        f"No saved model found for iteration {iteration}; retraining..."
                    )
                print_master(f"Training model on {len(dataset.inputs)} samples...")
            else:
                print_master(f"Loading existing model: {model_path}")

        # ---- Training ----
        if master and should_train:
            phases.begin(f"training_it_{iteration}")
            tf.keras.backend.clear_session()

            # Shuffle the training data to avoid biasing the validation set
            shuffle_indices = np.random.permutation(len(dataset.inputs))
            inputs, targets = (
                dataset.inputs[shuffle_indices],
                dataset.targets[shuffle_indices],
            )
            # training.members > 1 (training/members.py): the rows acquired since the previous
            # training, from which the selection set is held out below.
            newest = None
            member_results = None
            if cfg.training.members > 1 and not cfg.training.member_pilot:
                from training.members import newest_rows, previous_rows

                newest = newest_rows(len(dataset.inputs),
                                     previous_rows(run.run_dir, iteration))[shuffle_indices]

            # ---- Training domain ----
            # Opt-in (acquisition.boost.train_domain, training/domain.py): from the first
            # training after the boosts' start on, the network is trained only on the points
            # inside a fixed ellipsoid around those near the best, built at that first
            # training. The others stay in the training set's files and its other uses.
            if (cfg.acquisition.boost is not None
                    and cfg.acquisition.boost.train_domain is not None
                    and iteration > boost_start):
                from run.storage import atomic_path
                from training.domain import TrainDomain, saved_domain

                domain_edge = cfg.acquisition.boost.train_domain
                domain_path, train_domain = saved_domain(run.run_dir / "boost", iteration)
                built = train_domain is None
                if built:
                    try:
                        train_domain = TrainDomain.build(
                            dataset.inputs, dataset.targets,
                            depth=cfg.acquisition.target_temperature * ndim)
                    except ValueError as error:
                        print_master(f"No training domain this iteration: {error}; every point is trained on")
                    else:
                        domain_path = run.run_dir / "boost" / f"train_domain_it_{iteration}.npz"
                        with atomic_path(domain_path) as temporary:
                            train_domain.save(temporary, iteration=iteration)
                if train_domain is not None:
                    keep = train_domain.kept(inputs, domain_edge, domain_path.name)
                    inputs, targets = inputs[keep], targets[keep]
                    if newest is not None:
                        newest = newest[keep]
                    print_master(
                        f"Training domain ({domain_path.name}"
                        + (", built from this training set" if built else "") + "): "
                        f"{int(keep.sum())} of {len(keep)} points inside {domain_edge:g} times the "
                        f"farthest of the {train_domain.rows} within {train_domain.depth:g} of the "
                        f"best; log L {float(targets.min()):.1f} to {float(targets.max()):.2f}. "
                        f"The other {int((~keep).sum())} are not trained on"
                    )

            # ---- Members' selection set (training.members) ----
            # A share of the newest rows, held out from every member's training this time: unseen
            # by all of them, the warm-started one included. Drawn with its own generator.
            selection = None
            if newest is not None:
                from training.members import member_seed, selection_rows

                held = selection_rows(newest, cfg.training.selection_share,
                                      np.random.default_rng(member_seed(cfg.seed, iteration, 0)))
                if held.any():
                    selection = (inputs[held], targets[held])
                    inputs, targets = inputs[~held], targets[~held]
                    print_master(
                        f"Members: {cfg.training.members} trainings of this network; {int(held.sum())} of the "
                        f"{int(newest.sum())} points acquired since the previous training held out to choose "
                        f"between them (training.selection_share {cfg.training.selection_share:g})")
                else:
                    print_master("Members: no point acquired since the previous training, so nothing to choose "
                                 "on; this network is trained once")

            # ---- Background ----
            # Opt-in (acquisition.boost.background): once the training set holds boost
            # points the network is a fixed quadratic plus a network for the difference.
            # The quadratic is the previous model's when it has one; otherwise it is fitted
            # here, and this one training starts from random weights.
            background = None
            cold_start = False
            # Loaded once, by whichever of the background and the warm start needs it first.
            previous_model_path = run.trained_models_dir / f"model_it_{iteration - 1}.keras"
            previous_model = None
            if (cfg.acquisition.boost is not None
                    and cfg.acquisition.boost.background == "quadratic"
                    and iteration > boost_start):
                from model.network import fit_background, model_background

                if previous_model_path.exists():
                    previous_model = load_model(previous_model_path)
                    background = model_background(previous_model)
                if background is None:
                    try:
                        background = fit_background(
                            inputs, targets,
                            depth=cfg.acquisition.target_temperature * ndim)
                    except ValueError as error:
                        print_master(f"No quadratic background this iteration: {error}")
                    else:
                        cold_start = True
                        print_master(
                            f"Quadratic background fitted to the {background['rows']} "
                            f"training points within "
                            f"{cfg.acquisition.target_temperature * ndim:g} log-units of "
                            f"the best (scale kappa = {background['kappa']:.3f}); it stays "
                            f"fixed from here on, and this training starts from random "
                            f"weights"
                        )

            model = build_model(
                inputs=inputs,
                targets=targets,
                n_layers=cfg.model.n_layers,
                n_neurons=cfg.model.n_neurons,
                activation=cfg.model.activation,
                **({} if background is None else {"background": background}),
            )

            # ---- Warm start ----
            # Keep newly adapted scalers and compensate the first/last Dense layers
            # so the warm start preserves raw predictions before fine-tuning.
            keep_last = False
            warm_started = False
            if cfg.training.warm_start and iteration > 0 and not cold_start:
                if previous_model_path.exists():
                    if previous_model is None:
                        previous_model = load_model(previous_model_path)
                    try:
                        transfer_warm_start(previous_model, model)
                    except ValueError as error:
                        print_master(f"Warm start SKIPPED: {error}; using random initialisation")
                    else:
                        warm_started = True
                        keep_last = boost_start is not None and iteration > boost_start
                        print_master(
                            f"Warm start: inherited {len(model.trainable_weights)} weight "
                            f"tensors from iteration {iteration - 1} "
                            f"(scalers re-adapted)"
                            + ("; the last weights are kept (boost phase)" if keep_last else "")
                        )
                else:
                    print_master(
                        f"Warm start: no model at {previous_model_path.name}, "
                        "training from a random initialisation"
                    )
            previous_model = None
            loss = loss_for(targets)
            # The first training with points from `client.py augment` runs a longer
            # schedule: a warm start assumes a training set that changed little.
            from dataset.augment import mark_trained, pending_augmentations

            fresh_augmentations = pending_augmentations(run.run_dir, dataset.inputs)
            schedule_factor = cfg.training.augment_schedule_factor if fresh_augmentations else 1.0
            if schedule_factor != 1.0:
                print_master(
                    f"First training with the points of "
                    f"{', '.join(path.name for path in fresh_augmentations)}: lr_patience, "
                    f"lr_grace and patience x{schedule_factor:g} "
                    f"(training.augment_schedule_factor)"
                )
            training_start_time = time.monotonic()
            schedule = dict(
                learning_rate=cfg.training.learning_rate,
                n_epochs=cfg.training.n_epochs,
                batch_size=cfg.training.batch_size,
                validation_split=cfg.training.validation_split,
                patience=int(round(cfg.training.patience * schedule_factor)),
                lr_schedule=cfg.training.lr_schedule,
                lr_factor=cfg.training.lr_factor,
                lr_patience=int(round(cfg.training.lr_patience * schedule_factor)),
                lr_min=cfg.training.lr_min,
                lr_grace=int(round(cfg.training.lr_grace * schedule_factor)),
            )
            member_tasks = []
            if selection is not None or (cfg.training.members > 1 and cfg.training.member_pilot):
                from training.members import dispatch as dispatch_members
                from training.members import member_ranks, member_seed, oversubscription

                ranks = member_ranks(get_size(), cfg.training.members)
                members_dir = run.trained_models_dir / "members"
                members_dir.mkdir(parents=True, exist_ok=True)
                import os

                threads = cfg.training.member_threads or int(os.environ.get("TF_NUM_INTRAOP_THREADS") or 4)
                for k in range(1, cfg.training.members):
                    member_tasks.append(dict(
                        member=k, seed=member_seed(cfg.seed, iteration, k), threads=threads,
                        inputs=inputs, targets=targets,
                        selection_inputs=None if selection is None else selection[0],
                        selection_targets=None if selection is None else selection[1], background=background,
                        model=dict(n_layers=cfg.model.n_layers, n_neurons=cfg.model.n_neurons,
                                   activation=cfg.model.activation),
                        loss=dict(name=cfg.training.loss, sigma_level=cfg.training.sigma_level,
                                  chi2_dof=ndim, ess_floor=cfg.training.msre_ess_floor),
                        schedule=schedule,
                        model_path=str(members_dir / f"model_it_{iteration}_member{k}.keras"),
                        history_path=str(run.training_history_dir / f"history_it_{iteration}_member{k}.csv"),
                        rank=ranks[k - 1] if k - 1 < len(ranks) else 0))
                if get_size() > 1:
                    from utils.mpi_utils import _get_communicator

                    dispatch_members(_get_communicator(), {t["rank"]: t for t in member_tasks if t["rank"]})
                print_master(
                    f"   members 1-{cfg.training.members - 1} from random weights"
                    + (f" on ranks {', '.join(str(r) for r in ranks)}" if ranks else "")
                    + (f", {sum(1 for t in member_tasks if not t['rank'])} after this one on rank 0"
                       if any(not t["rank"] for t in member_tasks) else "")
                    + f", {threads} TensorFlow threads each")
                warning = oversubscription(get_size(), cfg.training.members, threads,
                                           int(os.environ.get("TF_NUM_INTRAOP_THREADS") or threads),
                                           os.cpu_count() or 1)
                if warning:
                    print_master(warning)
            elif cfg.training.members > 1 and get_size() > 1:
                from training.members import dispatch as dispatch_members
                from utils.mpi_utils import _get_communicator

                dispatch_members(_get_communicator(), {})
            history, training_metrics = train_model(
                model=model,
                inputs=inputs,
                targets=targets,
                loss=loss,
                learning_rate=cfg.training.learning_rate,
                n_epochs=cfg.training.n_epochs,
                batch_size=cfg.training.batch_size,
                validation_split=cfg.training.validation_split,
                patience=int(round(cfg.training.patience * schedule_factor)),
                lr_schedule=cfg.training.lr_schedule,
                lr_factor=cfg.training.lr_factor,
                lr_patience=int(round(cfg.training.lr_patience * schedule_factor)),
                lr_min=cfg.training.lr_min,
                lr_grace=int(round(cfg.training.lr_grace * schedule_factor)),
                # A warm-started network's validation loss is an in-sample one at first: the
                # split is drawn afresh from a set it was mostly trained on, so its lowest
                # validation loss comes before it has learnt the new points. In the boost
                # phase the last weights are kept. Measured on the 49-parameter, SINU and
                # HiLLiK runs (15 October 2026): where the best epoch was one of the first
                # few, the last weights had 0.48 to 0.86 of the error at true values.
                restore_best=not keep_last,
            )
            if selection is not None:
                from training.members import keep_member

                model, history, training_metrics = keep_member(
                    cfg, run, iteration, model, history, training_metrics, loss, selection, member_tasks,
                    warm=warm_started)
            elif member_tasks:
                from training.members import gather as gather_members

                # Chosen by the pilot, after the region: rank 0's network stands in until then.
                member_results = gather_members(cfg, run, iteration, model, history, training_metrics, loss,
                                                None, member_tasks, warm=warm_started)
            training_time = time.monotonic() - training_start_time
            print_master(f"Finished training in {training_time:.2f}s")
            save_model(model, model_path)
            mark_trained(fresh_augmentations, iteration, training_metrics["epoch"], schedule_factor)
            save_history(
                history.history,
                run.training_history_dir / f"history_it_{iteration}.csv",
            )
            metrics_tracker.add_training_metrics(
                iteration=iteration,
                epoch=training_metrics["epoch"],
                loss=training_metrics["loss"],
                val_loss=training_metrics["val_loss"],
                training_time=training_time,
            )
            metrics_tracker.save_all_metrics()

        # ---- Members on the other ranks (training.members) ----
        # A worker waits here, asleep, for rank 0's message: a member to train or nothing.
        if cfg.training.members > 1 and get_size() > 1:
            from training.members import dispatch as dispatch_members
            from training.members import serve as serve_member
            from utils.mpi_utils import _get_communicator

            if not master:
                serve_member(_get_communicator())
            elif not should_train:
                dispatch_members(_get_communicator(), {})

        # ---- Region of the boosts ----
        # Opt-in (acquisition.boost.region, sampling/region.py): from the second boost on,
        # the audit's searches, this iteration's chain and the boost's own sampling see the
        # network only inside the region that the earlier boosts' posterior points reach.
        region = None
        if (master and cfg.acquisition.boost is not None
                and cfg.acquisition.boost.region is not None
                and iteration > boost_start):
            from run.storage import atomic_path
            from sampling.region import Region, boost_region

            region_path = run.run_dir / "boost" / f"region_it_{iteration}.npz"
            saved_region = not should_train and region_path.exists()
            if saved_region:
                # The saved network is seen in the region it was audited and sampled in. Built
                # again here, from a training set that now holds its own audit's points, the
                # region could differ, and the file is a committed one.
                region, region_info = Region.load(region_path), dict(dropped=0)
            else:
                region, region_info = boost_region(
                    run.run_dir / "boost", likelihood.param_names, prior_lower, prior_upper,
                    cfg.acquisition.boost.region, before=iteration,
                    best=float(np.max(dataset.targets)),
                    depth=0.5 * cfg.acquisition.target_temperature * ndim,
                    quantile=cfg.acquisition.boost.region_quantile,
                )
            left_out = region_info.get("left_out") or []
            aside = (f"{len(left_out)} cloud(s) of {' + '.join(str(n) for n in left_out)} verified boost point(s) left out "
                     "(a single point, or too few while no cloud is well sampled, gives no shape)") if left_out else ""
            if region is None:
                print_master("Region: " + (aside + "; no region yet" if aside else "no verified boost points yet")
                             + "; the network is used over the whole box")
            else:
                if not saved_region:
                    with atomic_path(region_path) as temporary:
                        region.save(temporary, iteration=iteration, **region_info)
                print_master(
                    (f"Region of the saved network ({region_path.name}): " if saved_region else "Region: ") +
                    f"{' + '.join(str(n) for n in region.rows)} verified boost points "
                    f"in {len(region)} mode(s), farthest at "
                    f"{', '.join(f'{r:.2f}' for r in region.reach)} sigma of its cloud"
                    + (f" (the {region_info['quantile']:g} quantile of its points' distances)"
                       if region_info.get("quantile") is not None else "")
                    + f"; edge at {region.inflate:g} times that"
                    + (f"; {region_info['dropped']} boost points left out (true value not "
                       f"finite or too low)" if region_info["dropped"] else "")
                    + (f"; {region_info['small_clouds']} sparse cloud(s) with the shape of a well-sampled one"
                       if region_info.get("small_clouds") else "")
                    + ("; " + aside if aside else "")
                )
                if cfg.acquisition.boost.train_domain is not None:
                    # The network is trained inside the domain and used inside the region:
                    # where the region reaches beyond the domain it is used without data.
                    from training.domain import region_coverage_message, region_outside, saved_domain

                    _, domain = saved_domain(run.run_dir / "boost", iteration)
                    if domain is not None:
                        coverage = region_outside(domain, cfg.acquisition.boost.train_domain, region,
                                                  np.random.default_rng(iteration), diagnostics=True)
                        print_master("   " + region_coverage_message(coverage))

        # ---- Members' pilot (training.member_pilot) ----
        # Every member's own posterior points, evaluated once with the true likelihood on all
        # ranks; the member with the smallest error over them is kept (training/members.py).
        if cfg.training.members > 1 and cfg.training.member_pilot:
            from training.members import dispatch_pilot, serve_pilot
            from utils.mpi_utils import _get_communicator

            if getattr(likelihood, "_client_calls", None) is not None:
                likelihood._client_calls.stage = f"members/it{iteration}"
            if master and should_train and member_results:
                from training.members import kept_network, run_pilot

                kept, _, pilot_inputs, pilot_targets, pilot_derived = run_pilot(
                    cfg, run, iteration, member_results,
                    None if region is None else run.run_dir / "boost" / f"region_it_{iteration}.npz",
                    dataset.inputs, dataset.targets, list(likelihood.param_names),
                    evaluate=lambda points: evaluate_true(likelihood, points), warm=warm_started,
                    derived_names=likelihood.derived_names, ignore=getattr(likelihood, "refused_value", None))
                valid = np.isfinite(pilot_targets)          # as the acquisition adds its points
                pilot_inputs, pilot_targets, pilot_derived = (pilot_inputs[valid], pilot_targets[valid],
                                                              pilot_derived[valid])
                if len(pilot_targets):
                    dataset.add_data(inputs=pilot_inputs, targets=pilot_targets, derived=pilot_derived)
                    print_master(f"   {len(pilot_targets)} pilot points added: {len(dataset.inputs)} training samples")
                if kept != 0:
                    model, history, training_metrics = kept_network(member_results, kept, model, history,
                                                                    training_metrics)
                    save_model(model, model_path)
                    save_history(history.history, run.training_history_dir / f"history_it_{iteration}.csv")
                    metrics_tracker.add_training_metrics(
                        iteration=iteration, epoch=training_metrics["epoch"], loss=training_metrics["loss"],
                        val_loss=training_metrics["val_loss"], training_time=training_time)
                    metrics_tracker.save_all_metrics()
            elif master:
                if get_size() > 1:
                    dispatch_pilot(_get_communicator(), {}, evaluate=False)
            elif get_size() > 1:
                serve_pilot(_get_communicator(), evaluate=lambda: evaluate_true(likelihood, None))

        # The acquisition chain of this iteration's network, as a function: the sampling step
        # runs it, and before that the audit's chain search does (acquisition.audit.chain).
        audit_chain = {}

        def run_chain(target):
            def tempered_logpost_fn(positions):
                return target.logpost(positions) * inverse_sampling_temperature

            sampler = build_sampler(
                name=cfg.sampling.sampler,
                n_walkers=cfg.sampling.n_walkers,
                ndim=ndim,
                log_prob_fn=tempered_logpost_fn,
            )
            if region is None:
                initial_positions = np.random.uniform(
                    low=prior_lower,
                    high=prior_upper,
                    size=(cfg.sampling.n_walkers, ndim),
                )
            else:
                # Uniform over the box would start every walker on the region's wall.
                initial_positions = region.start(cfg.sampling.n_walkers, np.random.default_rng(
                    None if cfg.seed is None else cfg.seed + 15485863 * (iteration + 1)))
            sampling_start_time = time.monotonic()
            sampler.run(
                n_steps=cfg.sampling.n_steps,
                initial_positions=initial_positions,
                burn_in=cfg.sampling.burn_in,
                chunk_size=cfg.sampling.chunk_size,
                thin=cfg.sampling.thin,
            )
            return sampler, time.monotonic() - sampling_start_time

        # ---- Audit of the network's optima ----
        # Opt-in. Before the chain, which starts its walkers uniformly over the box and
        # would otherwise be the one to find a false optimum, at the price of a batch.
        if cfg.acquisition.audit is not None:
            likelihood._client_calls.stage = f"audit/it{iteration}"
            run_audit = bool(
                master and should_train
                and iteration >= cfg.acquisition.audit.start_iteration
            )
            # Opt-in (acquisition.audit.chain), in a boost iteration: a round of the audit whose
            # own climbs flag nothing runs this network's chain, cools its walkers and looks
            # where they climb to, as the boost's mode search will. A maximum that only the
            # chain reaches is then evaluated here, at a few points, and not by a third of
            # the boost (docs/audit_optima.md, "The chain search").
            chain_search = {}
            if (run_audit and cfg.acquisition.audit.chain and cfg.acquisition.boost is not None
                    and boost_start is not None and iteration >= boost_start):
                def chain_summits(audited):
                    from sampling.modes import cooled_peaks
                    from training.audit import weights_digest

                    if audit_chain:
                        audit_chain.pop("sampler").reset()
                        audit_chain.clear()
                    target = SurrogateLikelihood(
                        model=audited, metadata=surrogate_metadata,
                        **({} if region is None else {"region": region}))
                    chain_sampler, seconds = run_chain(target)
                    audit_chain.update(sampler=chain_sampler, seconds=seconds, weights=weights_digest(audited))
                    walkers = chain_sampler.chain()
                    return cooled_peaks(
                        target.loglkl, prior_lower, prior_upper, walkers[-max(1, len(walkers) // 10):],
                        cfg.sampling.temperature, walkers=cfg.acquisition.boost.walkers,
                        cool_steps=cfg.acquisition.boost.cool_steps,
                        seed=None if cfg.seed is None else cfg.seed + 104729 * (iteration + 1))

                chain_search = dict(peaks=chain_summits, cloud=True)
            audited = audit_network(
                cfg, likelihood, master, run_audit=run_audit,
                evaluation_journal=run.run_dir / "audit" / f"audit_evaluations_it_{iteration}.jsonl",
                **(dict(
                    model=model, inputs=inputs, targets=targets,
                    penalty_source=(run.run_dir / "audit", iteration, likelihood.param_names),
                    validation_split=cfg.training.validation_split,
                    lower=prior_lower, upper=prior_upper,
                    build_loss=loss_for,
                    batch_size=cfg.training.batch_size,
                    n_append=cfg.acquisition.n_append,
                    seed=None if cfg.seed is None else cfg.seed + 7919 * (iteration + 1),
                    **({} if region is None else {"project": region.project_unit}),
                    **chain_search,
                ) if run_audit else {}),
            )
            if audited is not None:
                audit_inputs, audit_targets, audit_derived, audit_report = audited
                audit_dir = run.run_dir / "audit"
                audit_dir.mkdir(exist_ok=True)
                if len(audit_targets):
                    dataset.add_data(inputs=audit_inputs, targets=audit_targets,
                                     derived=audit_derived)
                    print_master(
                        f"   {len(audit_targets)} audit points added: "
                        f"{len(dataset.inputs)} training samples"
                    )
                if len(audit_targets) or audit_report["penalized"] or audit_report["penalties_reapplied"]:
                    save_model(model, model_path)
                from training.audit import save_audit

                save_audit(audit_dir, iteration, likelihood.param_names,
                           audit_inputs, audit_targets, audit_report,
                           derived_names=likelihood.derived_names, derived=audit_derived)

        # ---- Derived-parameter emulator ----
        # Right after the log-likelihood network, so that a likelihood on derived
        # parameters can use it in this iteration's sampling. Its split draws from its
        # own generator, so the acquisition draws match a run without it.
        if master and derived_metadata is not None and cfg.derived is not None:
            from derived.emulator import model_path as derived_model_path
            from derived.emulator import train_and_save as train_derived

            if should_train or not derived_model_path(run.derived_data_dir, iteration).exists():
                if np.isfinite(dataset.derived).any():
                    train_derived(
                        dataset.inputs, dataset.derived, dataset.targets,
                        likelihood.param_names, derived_metadata, cfg.derived,
                        run.derived_data_dir, iteration,
                        rng=np.random.default_rng(
                            None if cfg.seed is None else cfg.seed + iteration
                        ),
                        log=print_master,
                    )
                else:
                    print_master(
                        "No training point has derived values yet; "
                        "derived-parameter emulator not trained"
                    )

        if master and should_train:
            artifacts = [model_path, run.training_history_dir / f"history_it_{iteration}.csv",
                         run.run_dir / "boost" / f"region_it_{iteration}.npz",
                         run.run_dir / "boost" / f"train_domain_it_{iteration}.npz"]
            artifacts += list((run.run_dir / "audit").glob(f"*_it_{iteration}.csv"))
            artifacts += list((run.run_dir / "audit").glob(f"audit_it_{iteration}.json"))
            artifacts += sorted((run.trained_models_dir / "members").glob(f"model_it_{iteration}_member*.keras"))
            artifacts += sorted((run.trained_models_dir / "members").glob(f"members_it_{iteration}.json"))
            artifacts += sorted((run.trained_models_dir / "members").glob(f"pilot_it_{iteration}.csv"))
            artifacts += sorted(run.training_history_dir.glob(f"history_it_{iteration}_member*.csv"))
            phases.complete(f"training_it_{iteration}", artifacts)

        # ---- Surrogate ----
        if master:
            # Reuse the freshly trained model in memory; only load from disk for already-trained iterations.
            if not should_train:
                tf.keras.backend.clear_session()
                model = load_model(model_path)
            surrogate = SurrogateLikelihood(
                model=model,
                metadata=surrogate_metadata,
                **({} if region is None else {"region": region}),
            )

            # ---- Sampling ----
            if audit_chain:
                from training.audit import weights_digest

                if audit_chain["weights"] == weights_digest(model):
                    # acquisition.audit.chain: the audit's last look ran the chain of this very
                    # network (nothing was flagged, or what was changed nothing after it)
                    sampler, sampling_elapsed_time = audit_chain["sampler"], audit_chain["seconds"]
                    print_master("Sampling: the chain of the audit's last look is this network's; it is not run again")
                else:
                    audit_chain["sampler"].reset()
                    sampler, sampling_elapsed_time = run_chain(surrogate)
                audit_chain = {}
            else:
                sampler, sampling_elapsed_time = run_chain(surrogate)

            # chain() and log_prob() are numpy views of the sampler's own buffer, so
            # they cost nothing here; the sampler must therefore not be reset until
            # acquisition has finished with them.
            chain = sampler.chain()
            logposts = sampler.log_prob()

            acceptance = sampler.acceptance_fraction().numpy()
            steps_run = sampler.n_steps_run
            max_tau = sampler.max_tau

            mean_acceptance = float(np.mean(acceptance))
            tau_note = f", max(tau): {max_tau:.1f}"
            print_master(
                f"Finished sampling in {sampling_elapsed_time:.2f}s "
                f"({steps_run} steps, acceptance rate: {mean_acceptance:.2g}{tau_note})"
            )

            metrics_tracker.add_sampling_metrics(
                iteration=iteration,
                steps_per_walker=steps_run,
                acceptance_rate=mean_acceptance,
                sampling_time=sampling_elapsed_time,
            )
            metrics_tracker.save_all_metrics()

        # ---- Convergence check ----
        converged = False
        handover = False  # tempered_batch_scatter: the ordinary loop has converged, the boosts begin
        if master:
            chain_summary = metric.summarize(chain)
            metric.save_chain_summary(
                convergence_stats_dir=run.convergence_stats_dir,
                iteration=iteration,
                chain_summary=chain_summary,
            )
            if previous_chain_summary is None:
                print_master(
                    "No previous chain summary available, skipping convergence check"
                )
            else:
                metric_value = metric.compute_from_summaries(
                    current_chain_summary=chain_summary,
                    previous_chain_summary=previous_chain_summary,
                )
                if batch_criterion:
                    print_master(f"{metric.name}: {metric_value:.3g} (not this run's criterion)")
                else:
                    # As for the batch criterion below: once the boosts have begun nothing stops
                    # the run but the number of iterations asked for. The drift is still recorded.
                    in_boosts = boost_start is not None and iteration >= boost_start
                    print_master(
                        f"{metric.name}: {metric_value:.3g} "
                        + ("(the boosts have begun: not a criterion)" if in_boosts
                           else f"(threshold: {cfg.convergence.threshold:.3g})")
                    )
                    converged = metric_value < cfg.convergence.threshold and not in_boosts
                    metrics_tracker.add_convergence_metrics(
                        iteration=iteration,
                        metric_value=metric_value,
                        converged=converged,
                        metric_name=metric.name,
                    )
                    metrics_tracker.save_all_metrics()
                    if use_convergence and converged:
                        print_master("Convergence criterion met, stopping...")
            previous_chain_summary = chain_summary
            # The criterion on true evaluations belongs to the ordinary loop alone: once the
            # boosts have begun, every boost reports its own error and the user decides.
            if batch_criterion and not (boost_start is not None and iteration >= boost_start):
                converged, handover, verdict = tempered_convergence(
                    cfg, metrics_tracker, iteration, use_convergence,
                    boosts_follow=boost_start is not None,
                )

        # Broadcast the stopping decision so all ranks leave the loop together.
        # The workers have been idle through training, sampling and the convergence
        # check, so this is one of the two long waits per iteration: bcast_idle
        # parks them on a sleep-poll instead of spinning against rank 0's training.
        if use_convergence:
            if batch_criterion:
                converged, handover = bcast_idle((converged, handover))
            else:
                converged = bcast_idle(converged)
            if handover:
                # The boosts begin with this network, on every rank; with
                # acquisition.boost.boosts the run ends after that many.
                boost_start = iteration
                if cfg.acquisition.boost.boosts is not None:
                    final_iteration = min(final_iteration, iteration + cfg.acquisition.boost.boosts)
                if master:
                    from dataset.boost import record_start

                    record_start(run.run_dir / "boost", iteration, cfg.acquisition.boost.start_iteration,
                                 verdict, cfg.sampling.temperature, cfg.convergence.threshold)
                    print_master(
                        f"The ordinary loop has converged: the boosts begin with this network "
                        f"(acquisition.boost.start_iteration: {cfg.acquisition.boost.start_iteration}); "
                        + (f"the run ends after {cfg.acquisition.boost.boosts} boosts, with network {final_iteration}"
                           if cfg.acquisition.boost.boosts is not None else f"the run goes on to network {final_iteration}")
                    )
            if converged:
                if master:
                    iteration_elapsed_time = time.monotonic() - iteration_start_time
                    metrics_tracker.add_iteration_metrics(
                        iteration=iteration,
                        iteration_time=iteration_elapsed_time,
                    )
                    metrics_tracker.save_all_metrics()
                break

        # ---- Acquisition ----
        if iteration < final_iteration:
            # Opt-in (acquisition.boost): from its start_iteration on, the new points are
            # posterior samples of the network, the same number from every mode, in place
            # of the tempered batch (dataset/boost.py). A network without a mode to draw
            # them from gets an ordinary iteration instead (draw_boost_points).
            boost_config = cfg.acquisition.boost
            boosting = boost_config is not None and iteration >= boost_start
            if boosting:
                print_master(
                    f"Boost: {boost_config.n} posterior samples of the network (T = 1) "
                    f"in place of the tempered batch"
                )
                if master:
                    acquisition_start_time = time.monotonic()
                new_samples, boost_draw, boosting = draw_boost_points(
                    cfg,
                    surrogate.loglkl if master else None,
                    prior_lower if master else None,
                    prior_upper if master else None,
                    chain[-max(1, len(chain) // 10):] if master else None,
                    iteration,
                    dataset if master else None,
                    run.run_dir / "boost" if master else None,
                    master,
                )
                if master and boosting:
                    chain = None
                    logposts = None
                    sampler.reset()
                    sampler = None
            if not boosting:
                print_master(
                    f"Selecting {cfg.acquisition.n_append} new samples from surrogate chain"
                )

            if master and not boosting:
                acquisition_start_time = time.monotonic()
                n_samples = logposts.size
                new_samples, acq_metrics = select_points(
                    dataset=dataset,
                    chain=chain,
                    logposts=logposts,
                    n_append=cfg.acquisition.n_append,
                    mcmc_temperature=cfg.sampling.temperature,
                    pool_factor=cfg.acquisition.pool_factor,
                )

                # The sampler output is no longer needed after acquisition; release it
                # before likelihood evaluations. chain/logposts are views into the
                # sampler's buffer, so the sampler has to go too or nothing is freed.
                chain = None
                logposts = None
                sampler.reset()
                sampler = None

                n_unique = acq_metrics["n_unique"]
                n_duplicates = n_samples - n_unique
                unique_fraction = n_unique / n_samples
                print_master(
                    f"Identified {n_unique} unique samples from {n_samples} total samples "
                    f"({n_duplicates} duplicates, unique fraction: {unique_fraction:.3g}, "
                    f"max multiplicity: {acq_metrics['max_multiplicity']})"
                )
                print_master(
                    f"Estimated the target density from a "
                    f"{acq_metrics['n_reference']}-point reference sample "
                    f"({len(dataset.inputs)} training points)"
                )
                if len(new_samples) < cfg.acquisition.n_append:
                    print_master(
                        f"Warning: selected only {len(new_samples)}/"
                        f"{cfg.acquisition.n_append} requested acquisition samples"
                    )
            elif not boosting:
                new_samples = None

            likelihood._client_calls.stage = f"{'boost' if boosting else 'acquisition'}/it{iteration}"
            if master:
                phases.begin(f"acquisition_it_{iteration}")
            new_inputs, new_targets, new_derived = evaluate_true(likelihood, new_samples)

            # The network chose these points and had not seen them: its error there is a
            # test of this iteration's network that the loop gets for nothing.
            held_out = None
            # A launcher may give points the likelihood cannot evaluate a fixed value (hpc/refused.py).
            not_evaluated = getattr(likelihood, "refused_value", None)
            if master and boosting:
                from dataset.boost import finish_boost

                held_out = finish_boost(run.run_dir / "boost", iteration, likelihood.param_names, boost_draw,
                                        new_inputs, new_targets, log=print_master, ignore=not_evaluated).get("error")
            elif master and len(new_targets):
                from utils.held_out import describe, held_out_error

                held_out = held_out_error(
                    np.asarray(surrogate.network(np.asarray(new_inputs, dtype=np.float32))), new_targets,
                    ignore=not_evaluated)
                print_master(f"   network {iteration} at its tempered batch, which it had not seen: network minus truth "
                             + describe(held_out))

            if master:
                n_attempted = len(new_targets)
                valid = np.isfinite(new_targets)
                if not valid.all():
                    new_inputs, new_targets = new_inputs[valid], new_targets[valid]
                    new_derived = new_derived[valid]
                    print_master(
                        f"Warning: filtered {valid.size - np.count_nonzero(valid)} targets with non-finite loglkl values"
                    )

                n_current_inputs = len(dataset.inputs)
                dataset.add_data(inputs=new_inputs, targets=new_targets, derived=new_derived)
                n_new_inputs = len(dataset.inputs) - n_current_inputs
                acquisition_elapsed_time = time.monotonic() - acquisition_start_time
                print_master(
                    f"Added {n_new_inputs} new training samples in {acquisition_elapsed_time:.2f}s for a total of {len(dataset.inputs)} samples"
                )
                metrics_tracker.add_acquisition_metrics(
                    iteration=iteration,
                    n_evaluated=n_attempted,
                    n_added=n_new_inputs,
                    acquisition_time=acquisition_elapsed_time,
                    dataset_size=len(dataset.inputs),
                    held_out=held_out,
                )
                metrics_tracker.save_all_metrics()

            # Re-fit the quadratic on EVERYTHING gathered so far and draw a fresh sample
            # from it. prior.n_inject fits the initial design once; this tracks the fit as
            # the training set improves, at n_inject evaluations per iteration.
            # (Not in a boost iteration: its points are the n it was asked for.)
            if cfg.acquisition.n_inject > 0 and not boosting:
                inj_seed = None if cfg.seed is None else cfg.seed + 1000 + iteration
                inj_inputs, inj_targets, inj_derived = draw_peak_samples(
                    cfg, likelihood,
                    dataset.inputs if master else None,
                    dataset.targets if master else None,
                    master,
                    n_draw=cfg.acquisition.n_inject,
                    seed=inj_seed,
                    context=f"it {iteration + 1}: ",
                )
                if master:
                    dataset.add_data(inputs=inj_inputs, targets=inj_targets,
                                     derived=inj_derived)
                    print_master(
                        f"   training set now {len(dataset.inputs)} points"
                    )

            if master:
                # This size includes injections; n_evaluated remains the acquisition batch.
                metrics_tracker.acquisition_metrics[iteration].dataset_size = len(dataset.inputs)
                dataset.save(run.training_data_dir / f"data_it_{iteration + 1}.csv")
                dataset.save_derived(
                    run.derived_data_dir / f"derived_it_{iteration + 1}.csv"
                )
                phases.complete(f"acquisition_it_{iteration}", [
                    run.training_data_dir / f"data_it_{iteration + 1}.csv",
                    run.derived_data_dir / f"derived_it_{iteration + 1}.csv"])
                from run.objective_calls import summarize
                summarize(run.run_dir)

        if master:
            iteration_elapsed_time = time.monotonic() - iteration_start_time
            metrics_tracker.add_iteration_metrics(
                iteration=iteration,
                iteration_time=iteration_elapsed_time,
            )
            metrics_tracker.save_all_metrics()

            # Release remaining per-iteration objects.
            chain = None
            logposts = None
            surrogate = None
            model = None

    # ---- Finalization ----
    if master:
        metrics_tracker.save_all_metrics()
        print_master(f"\nRun completed: {run.run_id}")
        print_master(f"Results saved in: {run.run_dir}")

    # The workers skip training, sampling and the convergence check, so on the LAST
    # iteration they leave the loop while rank 0 still has all three to do -- there is
    # no acquisition left to park them in. Without this they drop straight into
    # MPI_Finalize, whose barrier spins, and seven ranks then contend with the final
    # iteration's training exactly as they did before bcast_idle existed. Measured on
    # a 150-epoch smoke run, one epoch stalled for 954 s.
    #
    # Park them on a sleeping wait instead; the master arrives here only once it is
    # genuinely finished.
    bcast_idle(True)
    if master:
        from run.objective_calls import summarize
        summarize(run.run_dir)
        run_lock.close()


if __name__ == "__main__":
    import importlib
    import sys

    # `python client.py COMMAND ...`: the module whose main(argv) runs it.
    commands = {
        "profiles": "profiles.__main__",
        "derived": "derived.__main__",
        "evidence": "evidence.__main__",
        "augment": "dataset.augment",
        "derived-view": "derived.view_cli",
        "modes": "sampling.detect",
        "combine": "run.combine",
        "export": "run.export",
    }
    if len(sys.argv) > 1 and sys.argv[1] in commands:
        importlib.import_module(commands[sys.argv[1]]).main(sys.argv[2:])
    else:
        main()
