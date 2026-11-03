"""Orchestrate independently restartable profile stages from an ordinary CLiENT run."""

import json
from pathlib import Path
import numpy as np
import yaml
from profiles.io import save
from run.run import iterations
from run.storage import atomic_path, digest as sha


def read(path):
    return json.loads(Path(path).read_text())


def array_save(path, value):
    with atomic_path(path) as temporary, temporary.open("wb") as f:
        np.save(f, value)


def preflight(run, config):
    config.validate()
    run = Path(run).resolve()
    md = read(run / "metadata.json")
    names = [p["name"] for p in md["parameters"]]
    bounds = np.array([[p["lower"], p["upper"]] for p in md["parameters"]], float)
    if (
        len(names) < 2
        or len(set(names)) != len(names)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        raise ValueError("need >=2 parameters and finite ordered metadata bounds")
    if set(config.parameters) - set(names):
        raise ValueError(f"unknown parameters: {set(config.parameters)-set(names)}")
    for n, r in config.ranges.items():
        if r != "prior" and (
            r[0] < bounds[names.index(n), 0] or r[1] > bounds[names.index(n), 1]
        ):
            raise ValueError(f"{n} range outside saved metadata domain")
    if config.dimension == 2 and config.retrain and len(names) < 3:
        raise ValueError(
            "2D augmentation requires at least one nuisance coordinate; use retrain=false and added_points=0 for a two-parameter objective"
        )
    iteration = config.iteration
    if iteration is None:
        ids = [
            n for n in iterations(run / "trained_models", "model_it_*.keras")
            if (run / "training_data" / f"data_it_{n}.csv").exists()
        ]
        if not ids:
            raise ValueError(
                "no matching model/data iteration; choose an existing iteration"
            )
        iteration = max(ids)
    model = run / "trained_models" / f"model_it_{iteration}.keras"
    data = run / "training_data" / f"data_it_{iteration}.csv"
    configs = (
        [run / config.run_config]
        if config.run_config
        else list(run.glob("*.yaml")) + list(run.glob("*.yml"))
    )
    if len(configs) != 1:
        raise ValueError("ambiguous run YAML: set run_config in profile config")
    cfg = yaml.safe_load(configs[0].read_text())
    backend = run / "likelihood_input" / Path(cfg["likelihood"]["input"]).name
    with data.open() as f:
        header = f.readline().strip().split(",")
    if header[:-1] != names:
        raise ValueError("training CSV column order differs from metadata")
    # Match the original network input representation before adding any labels.
    training = np.loadtxt(
        data, delimiter=",", skiprows=1, ndmin=2, dtype=np.float32
    ).astype(float)
    if training.shape[1] != len(names) + 1 or not np.isfinite(training).all():
        raise ValueError("invalid original training data")
    tolerance = (
        2 * np.finfo(np.float32).eps * np.maximum(1, np.max(abs(bounds), axis=1))
    )
    if np.any(training[:, :-1] < bounds[:, 0] - tolerance) or np.any(
        training[:, :-1] > bounds[:, 1] + tolerance
    ):
        raise ValueError("original data outside metadata bounds")
    if config.maxfun < len(names) + 3:
        raise ValueError(f"maxfun must be >= {len(names)+3} for anchor interpolation")
    files = [run / "metadata.json", configs[0], model, data, backend]
    backend_identity = None
    if config.backend_module:
        import importlib.util

        spec = importlib.util.find_spec(config.backend_module)
        if spec is None or not spec.origin:
            raise ValueError("backend_module is not importable")
        backend_identity = dict(
            module=config.backend_module,
            path=str(Path(spec.origin).resolve()),
            sha256=sha(spec.origin),
        )
        if (
            config.backend_sha256
            and config.backend_sha256 != backend_identity["sha256"]
        ):
            raise ValueError(
                "backend binary hash differs from requested backend_sha256"
            )
    from profiles.region import resolve

    region = resolve(run, iteration, config.region)
    manifest = dict(
        backend_identity=backend_identity,
        run=str(run),
        iteration=iteration,
        configuration=config.as_dict(),
        sources={str(p): sha(p) for p in files},
        objective="native backend objective, matching ordinary client training",
        bounds=dict(zip(names, bounds.tolist())),
        original_training_rows=len(training),
        historical_backend_calls=None,
        # Where the network is searched (profiles/region.py): None, the whole box; or the
        # region of the run's boosts that belongs to this network, by file, hash and edge.
        region=region,
    )
    return manifest, training, model, backend, cfg["likelihood"]["wrapper"]


def explicit_grids(bounds_map, config):
    """The nodes display_grids makes when every range is explicit ("prior" or [low, high]),
    without the full-range surrogate scan it would otherwise read (only for the 1-sigma
    crossings it reports, which are null here)."""
    grids, diagnostics = [], {}
    for name in config.parameters:
        lo, hi = bounds_map[name]
        requested = config.ranges[name]
        if requested == "prior":
            low, high, kind = lo, hi, "full metadata bounds"
        else:
            low, high = requested
            kind = "explicit physical range"
            if low < lo or high > hi:
                raise ValueError(f"{name} requested range outside saved model domain")
        actual = [max(lo, low), min(hi, high)]
        grids.append(np.linspace(*actual, config.profile_points))
        diagnostics[name] = dict(
            requested=[float(low), float(high)],
            actual=actual,
            clipped=bool(actual[0] != low or actual[1] != high),
            kind=kind,
            points=config.profile_points,
            surrogate_1sigma_crossings=None,
        )
    return grids, diagnostics


def canonical_points(points, bounds, excluded):
    """Round inputs before evaluating labels; reject collisions before spending calls."""
    bounds = np.asarray(bounds, float)
    lo = bounds[:, 0].astype(np.float32)
    hi = bounds[:, 1].astype(np.float32)
    lo = np.where(lo < bounds[:, 0], np.nextafter(lo, np.float32(np.inf)), lo)
    hi = np.where(hi > bounds[:, 1], np.nextafter(hi, np.float32(-np.inf)), hi)
    x = np.clip(np.asarray(points, dtype=np.float32), lo, hi).astype(float)
    rows = [tuple(r) for r in x]
    if len(rows) != len(set(rows)) or set(rows) & set(map(tuple, excluded)):
        raise ValueError(
            "coordinate collision after float32 conversion; use a new output/configuration, no labels evaluated for this batch"
        )
    return x


def audit_row_ids(labels, count):
    """The saved training rows whose labels the audit recomputes: the `count` with the
    highest label."""
    count = min(count, len(labels))
    return np.sort(np.argsort(-np.asarray(labels, float), kind="stable")[:count])


def audit_labels(ledger, original, bounds, c):
    """Recompute a few saved labels with the current backend. The tolerance is absolute in
    Delta chi-squared, so the rows checked are the best ones: far from the peak two
    machines' backends differ by more than it without any error in either."""
    ids = audit_row_ids(original[:, -1], c.audit_points)
    ledger.stage = "audit"
    points = np.clip(original[ids, :-1], bounds[:, 0], bounds[:, 1])
    values = ledger.evaluate_many(points)
    errors = 2 * abs(values - original[ids, -1])
    refused = ~np.isfinite(values)
    # Ordinary client stores targets as float32. Account for that rounding.
    rounding = 2 * abs(np.spacing(original[ids, -1].astype(np.float32))).astype(
        float
    )
    passed = bool(np.all(errors <= c.audit_tolerance + rounding))
    return dict(
        rows="best",
        indices=ids,
        delta_chi2_errors=[None if r else e for r, e in zip(refused.tolist(), errors.tolist())],
        rounding_allowance=rounding,
        refused_rows=ids[refused],
        passed=passed,
    )


def run_workflow(run, output, config, *, resume=False, dry_run=False):
    manifest, original, model_path, backend_path, wrapper = preflight(run, config)
    if dry_run:
        nodes = (
            config.profile_points * len(config.parameters)
            if config.dimension == 1
            else config.profile_points**2
        )
        validation = (
            (
                config.validation_global
                + 3
                * (
                    config.validation_nodes * len(config.parameters)
                    if config.dimension == 1
                    else config.validation_nodes**2
                )
            )
            if config.retrain
            else 0
        )
        print(
            json.dumps(
                dict(
                    manifest=manifest,
                    planned_added_training=config.added_points,
                    planned_validation=(
                        (
                            config.validation_global
                            + 3
                            * (
                                config.validation_nodes * len(config.parameters)
                                if config.dimension == 1
                                else config.validation_nodes**2
                            )
                        )
                        if config.retrain
                        else 0
                    ),
                    planned_grid_nodes=(
                        config.profile_points * len(config.parameters)
                        if config.dimension == 1
                        else config.profile_points**2
                    ),
                    conservative_grid_and_anchor_ceiling=nodes * (config.maxfun + 1)
                    + 2 * config.maxfun,
                    conservative_whole_study_ceiling=nodes * (config.maxfun + 1)
                    + 2 * config.maxfun
                    + config.audit_points
                    + config.added_points
                    + validation,
                    true_optimization_calls="data dependent; lifetime total_call_budget may stop before this conservative ceiling",
                ),
                indent=2,
            )
        )
        return
    out = Path(output).resolve()
    run = Path(run).resolve()
    if out == run or out in run.parents:
        raise ValueError("output cannot be the source run or its parent")
    out.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (out / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another profile process owns this output directory")
        if (out / "manifest.json").exists():
            if not resume:
                raise ValueError("existing study: use --resume or a new output")
            stored_manifest = read(out / "manifest.json")
            # A study depends on the region its network was searched in, which the manifest
            # holds resolved (None for the bare network), not on the setting that led to it.
            from profiles.config import EXECUTION_SETTINGS

            def comparable(m):
                # The run's files by their place in the run, not by where the run stood: a study
                # depends on what the files hold (their hashes), and a run directory that was
                # moved or synced to another place with the same files is the same source.
                # (The absolute paths were compared, and such a study could not be resumed.)
                def in_run(path):
                    try:
                        return str(Path(path).relative_to(m["run"]))
                    except ValueError:
                        return path

                return dict(m, run=None, sources={in_run(k): v for k, v in m["sources"].items()},
                            configuration={k: v for k, v in m["configuration"].items()
                                           if k not in EXECUTION_SETTINGS and k != "region"})

            if comparable(stored_manifest) != comparable(manifest):
                raise ValueError("source/configuration changed; resume refused")
            if stored_manifest["run"] != manifest["run"]:
                print(f"The study was begun with the run at {stored_manifest['run']}; it is at {manifest['run']} now, "
                      "with the same files (their hashes are those of the manifest).")
        else:
            if any(p.name != ".lock" for p in out.iterdir()):
                raise ValueError("output is not empty")
            save(out / "manifest.json", manifest)
        if (out / "complete.json").exists():
            print(f"Already complete: {out}")
            return
        _execute(
            run, out, config, manifest, original, model_path, backend_path, wrapper
        )


def _execute(run, out, c, manifest, original, model_path, backend_path, wrapper):
    from likelihood.surrogate import SurrogateMetadata
    from model.network import load_model
    from profiles.evaluation import BackendPool, WorkflowLedger, aggregate_counts, record_refused
    from profiles.surrogate import scan, display_grids, surrogate_optimizer
    from profiles.training import (
        ProfileTrainingPreset,
        broad_design,
        validation_set,
        fit_profile_model,
    )
    from dataset.peak_samples import _fit_quadratic
    from dataset.profile_augmentation import make_design

    md = SurrogateMetadata.load(run / "metadata.json")
    bounds = np.array([md.bounds[n] for n in md.param_names])
    pool = None
    ledger = None

    def status(name):
        (out / "status.txt").write_text(name + "\n")
        print(name, flush=True)

    def cached_json(name, compute):
        path = out / name
        if path.exists():
            return read(path)
        result = compute()
        save(path, result)
        return read(path)

    region = None
    if manifest.get("region"):
        from profiles import region as boosts

        region = boosts.load(run, manifest["region"])

    def see(network):
        """What the searches see of a network: inside the region of its boosts, the network."""
        if region is None:
            return network
        return boosts.restrict(network, region)

    def scan_checked(model, data, grids):
        def scan_once(grids):
            if c.surrogate_optimizer == "legacy":
                result, _ = scan(model, md, data, c.parameters, grids, c)
                return result
            from profiles.hybrid import hybrid_settings, scan_hybrid

            result, _ = scan_hybrid(
                model, md, data, c.parameters, grids,
                hybrid=hybrid_settings(c),
            )
            return result

        # In a region, nodes whose optimum is not inside it are cut from the ranges first.
        result = (scan_once(grids) if region is None
                  else boosts.scan(scan_once, region, grids, c.parameters))
        if not result["optimization_quality"]["converged"]:
            save(out / "failed_surrogate_diagnostics.json", result)
            raise RuntimeError(
                "surrogate stationarity/stability check failed; inspect failed_surrogate_diagnostics.json"
                + (boosts.at_edge(region, result) if region is not None else "")
            )
        return result

    try:
        pool = BackendPool(
            c.workers,
            wrapper=wrapper,
            input_path=str(backend_path),
            names=md.param_names,
            bounds=bounds,
            module=c.backend_module,
            expected_hash=(
                manifest["backend_identity"]["sha256"]
                if manifest["backend_identity"]
                else None
            ),
        )
        ledger = WorkflowLedger(pool, bounds, out / "calls", c.total_call_budget)
        status("audit original objective convention")
        if not (out / "audit.json").exists():
            result = audit_labels(ledger, original, bounds, c)
            save(out / ("audit.json" if result["passed"] else "failed_audit.json"), result)
            if not result["passed"]:
                if len(result["refused_rows"]):
                    raise RuntimeError(
                        f"the current backend refuses saved training row(s) {result['refused_rows'].tolist()}, among the best the "
                        "run has: it is not the backend the labels came from; failed_audit.json has every row")
                errors = np.array(result["delta_chi2_errors"], float)
                worst = int(np.argmax(errors - result["rounding_allowance"]))
                raise RuntimeError(
                    "saved labels do not match current backend; check original objective/backend installation"
                    f" (training row {result['indices'][worst]}: Delta chi-squared differs by "
                    f"{errors[worst]:.3g}, saved log-objective "
                    f"{original[result['indices'][worst], -1]:.6g}; failed_audit.json has every row)"
                )
        model = load_model(model_path)
        if region is not None:
            print("region: " + boosts.describe(region, manifest["region"]), flush=True)
        elif c.region is not None and any(
                int(p.stem.rsplit("_", 1)[1]) < manifest["iteration"] for p in (run / "boost").glob("boost_it_*.csv")):
            # A boost run made without acquisition.boost.region: there is no region to use.
            print("region: this network was trained on boosts, and the run saved no region of them "
                  "(acquisition.boost.region was not set): it is searched over the whole box. Such a network can "
                  "have maxima where no boost point lies; check every node's true value against the scan's.",
                  flush=True)
        if c.dimension == 2:
            from profiles.surface_workflow import execute_surface

            execute_surface(out, c, original, model, md, ledger, status, see=see, region=region)
            status("complete")
            return
        searched = see(model)
        fullgrids = [np.linspace(*md.bounds[n], c.scan_points) for n in c.parameters]
        # Without training, the full-range baseline scan only chooses display ranges. When
        # every range is explicit it has nothing to choose, so it is skipped (a study that
        # already has one keeps using it, so resumed outputs are unchanged).
        explicit = not c.retrain and all(n in c.ranges for n in c.parameters)
        if explicit and not (out / "baseline_profiles.json").exists():
            baseline = None
        else:
            status("baseline surrogate profiles")
            baseline = cached_json(
                "baseline_profiles.json", lambda: scan_checked(searched, original, fullgrids)
            )
        if c.retrain:
            optimizer = surrogate_optimizer(searched, bounds, seed=c.seed, gtol=c.gtol)
            fit = _fit_quadratic(original[:, :-1], original[:, -1], ridge_eig=1e-3)
            status("profile augmentation")
            if (out / "augmented_data.npy").exists():
                data = np.load(out / "augmented_data.npy")
            else:
                data = original.copy()
                nb = int(round(c.added_points * c.broad_fraction))
                for label, n in [("broad", nb), ("targeted", c.added_points - nb)]:
                    if not n:
                        continue
                    pointfile = out / f"{label}_points.npy"
                    if pointfile.exists():
                        points = np.load(pointfile)
                    else:
                        if label == "broad":
                            points, details = broad_design(
                                original, baseline, md, bounds, fit, optimizer, c.seed, n
                            )
                        else:
                            points, details = make_design(
                                data[:, :-1],
                                data[:, -1],
                                baseline,
                                md.param_names,
                                bounds,
                                n,
                                optimizer,
                                seed=c.seed + 10000,
                            )
                        points = canonical_points(points, bounds, data[:, :-1])
                        array_save(pointfile, points)
                        save(out / f"{label}_design.json", details)
                    ledger.stage = "augmentation/" + label
                    values = ledger.evaluate_many(points)
                    keep = record_refused(out, "augmentation/" + label, points, values)
                    data = np.vstack([data, np.column_stack([points[keep], values[keep]])])
                assert len(data) <= len(original) + c.added_points
                array_save(out / "augmented_data.npy", data)
                np.savetxt(
                    out / "augmented_data.csv",
                    data,
                    delimiter=",",
                    header=",".join(md.param_names + ["loglkl"]),
                    comments="",
                )
            status("independent pooled validation")
            if (out / "validation.npz").exists():
                validation = dict(np.load(out / "validation.npz"))
            else:
                ledger.stage = "validation"

                class Adapter:
                    @staticmethod
                    def evaluate_many(points):
                        values = ledger.evaluate_many(points)
                        record_refused(out, "validation", points, values)
                        return values

                g, r, pp, npairs, details = validation_set(
                    original,
                    baseline,
                    md,
                    bounds,
                    fit,
                    optimizer,
                    Adapter(),
                    seed=c.seed + 20000,
                    global_count=c.validation_global,
                    nodes_per_parameter=c.validation_nodes,
                    canonicalize=lambda x: canonical_points(x, bounds, data[:, :-1]),
                )
                validation = dict(
                    global_data=g, ridge_data=r, profile_pairs=pp, nuisance_pairs=npairs
                )
                with (out / "validation.tmp").open("wb") as f:
                    np.savez(f, **validation)
                (out / "validation.tmp").replace(out / "validation.npz")
                save(out / "validation_design.json", details)
            status("frozen-scaler profile training")
            training = out / "training"
            preset = ProfileTrainingPreset(
                **{k: getattr(c, k) for k in ProfileTrainingPreset.__dataclass_fields__}
            )
            if not (training / "training.json").exists():
                fit_profile_model(
                    load_model(model_path),
                    data,
                    validation,
                    training,
                    preset=preset,
                    original_rows=len(original),
                    counts=aggregate_counts(out / "calls"),
                )
            selected = see(load_model(training / "model_contrast.keras"))
            status("augmented surrogate profiles")
            full = cached_json(
                "augmented_full_profiles.json",
                lambda: scan_checked(selected, data, fullgrids),
            )
        else:
            selected, data, full = searched, original, baseline
        grids, ranges = (explicit_grids(md.bounds, c) if full is None
                         else display_grids(full, md.bounds, c))
        save(out / "ranges.json", ranges)
        surrogate = cached_json(
            "surrogate_profiles.json", lambda: scan_checked(selected, data, grids)
        )
        if surrogate.get("region_cuts"):
            # The ranges as computed: without the nodes found outside the region.
            for name, cut in surrogate["region_cuts"].items():
                ranges[name] = dict(ranges[name], actual=cut["kept"], region=cut)
            save(out / "ranges.json", ranges)
        status("direct and true profiles (residual Newton)")
        from profiles.evaluation import clip_profile_starts

        # True stages start from in-box points (a scan can end a few ulps outside a bound);
        # surrogate_profiles.json keeps the scan's own values, any change is recorded.
        surrogate, clipped = clip_profile_starts(surrogate, bounds)
        if clipped:
            save(out / "true_start_clipping.json", dict(changes=clipped))
        from profiles.residual_newton import NewtonConfig, refine_profiles_newton

        refine_profiles_newton(
            selected, surrogate, bounds, ledger, out,
            NewtonConfig(tol=c.corrected_tolerance(), maxcalls=c.maxfun),
            workers=c.workers, chain_length=c.chain_length,
        )
        status("plots and tables")
        from profiles.reporting import write_outputs

        save(out / "counts.json", aggregate_counts(out / "calls"))
        write_outputs(out)
        save(
            out / "complete.json",
            dict(
                training_performed=c.retrain,
                added_points=c.added_points,
                parameters=c.parameters,
                points_per_parameter=c.profile_points,
                counts=aggregate_counts(out / "calls"),
                objective="native",
                accuracy_certified=False,
            ),
        )
        status("complete")
    except BaseException as exc:
        save(
            out / "failure.json",
            dict(
                error=repr(exc),
                stage=(
                    (out / "status.txt").read_text()
                    if (out / "status.txt").exists()
                    else "initializing"
                ),
            ),
        )
        raise
    finally:
        if ledger is not None:
            ledger.persist()
            save(out / "counts.json", aggregate_counts(out / "calls"))
        if pool is not None:
            pool.close()
