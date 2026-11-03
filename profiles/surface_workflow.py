"""One pair per run. Reuse the 1D training selector, backend pool and call ledger."""

import json
from pathlib import Path
import numpy as np
from profiles.io import save
from profiles.workflow import canonical_points, array_save
from profiles.surface import scan_surface
from profiles.surrogate import scan, display_grids, surrogate_optimizer
from profiles.surface_design import surface_design, nuisance_draw
from profiles.evaluation import aggregate_counts, record_refused


def execute_surface(out, c, data, model, md, ledger, status, see=None, region=None):
    """`see(network)` is what the searches are given of a network (profiles/workflow.py: the
    network inside the region of its boosts, `region`); the training is given the network
    itself."""
    from model.network import load_model
    from profiles.hybrid import scan_hybrid, hybrid_settings
    from profiles.hybrid_surface import refine_surface_newton, scan_surface_hybrid
    from profiles.training import ProfileTrainingPreset, fit_profile_model, without_refused

    searched = model if see is None else see(model)

    legacy = c.surrogate_optimizer == "legacy"

    def range_scan(model, md, data, names, grids, config):
        if legacy:
            return scan(model, md, data, names, grids, config)
        return scan_hybrid(model, md, data, names, grids, config, hybrid=hybrid_settings(c))
    bounds = np.array([md.bounds[n] for n in md.param_names])
    indices = [md.param_names.index(n) for n in c.parameters]

    def finished_scan(path, scan_axes, pair, names):
        """The scan an earlier start of this study finished and checked, or None: the file as
        the workflow saves it after the check (with the names), for these axes. The study's
        manifest has already been compared, so network, region and settings are the same."""
        try:
            saved = json.loads(Path(path).read_text())
            usable = (saved["names"] == list(pair) and saved["full_parameter_names"] == list(names)
                      and saved["quality"]["converged"] is True
                      and [list(map(float, a)) for a in saved["axes"]] == [np.asarray(a, float).tolist() for a in scan_axes]
                      and np.asarray(saved["xopt"], float).shape == (len(scan_axes[0]), len(scan_axes[1]), len(names)))
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return saved if usable else None

    def scan_checked(scan_axes, checkpoint):
        # The legacy scan raises on its own when a pass limit is reached. The hybrid scan
        # returns its result whether it converged or not, so check it here, as the 1D workflow
        # does, before any node is refined from its starts: an unconverged scan can end at
        # spurious network maxima (box-corner artefacts far above every training label).
        search = scan_surface if legacy else scan_surface_hybrid
        result = search(searched, bounds, data, indices, scan_axes, c, checkpoint)
        if not result["quality"]["converged"]:
            save(out / "failed_surrogate_diagnostics.json", result)
            Path(checkpoint).unlink(missing_ok=True)
            raise RuntimeError(
                "2D surrogate stationarity/stability check failed; inspect "
                "failed_surrogate_diagnostics.json")
        return result

    status("2D ranges")
    if all(n in c.ranges for n in c.parameters):
        axes = []
        ranges = {}
        for n in c.parameters:
            r = c.ranges[n]
            low, high = md.bounds[n] if r == "prior" else r
            axes.append(np.linspace(low, high, c.profile_points))
            ranges[n] = dict(
                actual=[low, high],
                clipped=False,
                kind=(
                    "explicit physical range"
                    if r != "prior"
                    else "full metadata bounds"
                ),
            )
    else:
        # An earlier start's scan is read, as its axes must be identical bit for bit for the node
        # checkpoints: a scan done again need not be, on another device or build. It is saved
        # only once checked, and the manifest has been compared, so the network is the same.
        path = out / "range_profiles.json"
        if not path.exists():
            scanned, _ = range_scan(
                searched,
                md,
                data,
                c.parameters,
                [np.linspace(*md.bounds[n], c.scan_points) for n in c.parameters],
                c,
            )
            if not scanned["optimization_quality"]["converged"]:
                raise RuntimeError("range scan did not converge")
            save(path, scanned)
        preliminary = json.loads(path.read_text())
        axes, ranges = display_grids(preliminary, md.bounds, c)
    save(out / "ranges.json", ranges)
    design_axes = [np.linspace(a[0], a[-1], c.design_points) for a in axes]
    if c.retrain:
        status("2D coarse surrogate design")
        coarse = scan_checked(design_axes, out / "coarse_surface.json")
    original_rows = len(data)
    if c.retrain:
        optimize = surrogate_optimizer(searched, bounds, seed=c.seed, gtol=c.gtol)
        status("2D pair-conditioned acquisition")
        if (out / "augmented_data.npy").exists():
            data = np.load(out / "augmented_data.npy")
        else:
            path = out / "added_points.npy"
            if path.exists():
                points = np.load(path)
            else:
                starts, records, geo = surface_design(
                    data, coarse, bounds, c.added_points, c.broad_fraction, c.seed
                )
                rng = np.random.default_rng(c.seed + 1)
                points = []
                for start, record in zip(starts, records):
                    _, anchor, d = optimize(
                        indices, start[indices], [start, coarse["best_x"]]
                    )
                    if not d["stationary"]:
                        raise RuntimeError("nonstationary 2D acquisition anchor")
                    # Targeted pool includes its own broad draws, as in 1D.
                    mode = (
                        "broad"
                        if record["pool"] == "broad" or rng.random() < 0.5
                        else "near"
                    )
                    p, fallback = nuisance_draw(anchor, geo, bounds, rng, mode)
                    points.append(p)
                    record.update(mode=mode, gibbs_fallback=fallback)
                points = canonical_points(
                    np.array(points).reshape(-1, len(bounds)), bounds, data[:, :-1]
                )
                array_save(path, points)
                save(
                    out / "acquisition_design.json",
                    dict(draws=records, n_points=c.added_points),
                )
            ledger.stage = "augmentation/pair"
            if len(points):
                values = ledger.evaluate_many(points)
                keep = record_refused(out, "augmentation/pair", points, values)
                data = np.vstack([data, np.column_stack([points[keep], values[keep]])])
            array_save(out / "augmented_data.npy", data)
            np.savetxt(
                out / "augmented_data.csv",
                data,
                delimiter=",",
                header=",".join(md.param_names + ["loglkl"]),
                comments="",
            )
        status("2D independent validation")
        if (out / "validation.npz").exists():
            validation = dict(np.load(out / "validation.npz"))
        else:
            design_path = out / "validation_coordinates.npz"
            if design_path.exists():
                saved = dict(np.load(design_path))
                x = saved["coordinates"]
                nuisance_pairs = saved["nuisance_pairs"]
            else:
                rng = np.random.default_rng(c.seed + 20000)
                # Broad/global held-out points drawn near the original design.
                global_points = []
                trials = 0
                std = np.std(data[:, :-1], axis=0)
                while len(global_points) < c.validation_global:
                    trials += 1
                    if trials > 10000 * c.validation_global:
                        raise RuntimeError("validation rejection limit")
                    p = data[
                        rng.integers(original_rows), :-1
                    ] + 0.02 * std * rng.normal(size=len(bounds))
                    if np.all(p >= bounds[:, 0]) and np.all(p <= bounds[:, 1]):
                        global_points.append(p)
                starts, _, geo = surface_design(
                    data, coarse, bounds, c.validation_nodes**2, 1.0, c.seed + 21000
                )
                ridge = []
                nuisance_pairs = []
                for start in starts:
                    _, anchor, d = optimize(
                        indices, start[indices], [start, coarse["best_x"]]
                    )
                    if not d["stationary"]:
                        raise RuntimeError("nonstationary validation anchor")
                    j = len(ridge)
                    ridge.append(anchor)
                    for _ in range(2):
                        for attempt in range(128):
                            p, _ = nuisance_draw(anchor, geo, bounds, rng, "near")
                            key = tuple(p.astype(np.float32))
                            if key not in {tuple(q.astype(np.float32)) for q in ridge}:
                                break
                        else:
                            raise RuntimeError(
                                "distinct nuisance validation draw limit"
                            )
                        ridge.append(p)
                        nuisance_pairs.append([j, len(ridge) - 1])
                x = canonical_points(
                    np.vstack([global_points, ridge]), bounds, data[:, :-1]
                )
                with (out / "validation_coordinates.tmp").open("wb") as f:
                    np.savez(f, coordinates=x, nuisance_pairs=np.array(nuisance_pairs))
                (out / "validation_coordinates.tmp").replace(design_path)
            ledger.stage = "validation"
            y = ledger.evaluate_many(x)
            record_refused(out, "validation", x, y)
            g = c.validation_global
            centres = np.arange(0, 3 * c.validation_nodes**2, 3)
            pair_anchor = int(centres[len(centres) // 2])
            g_data, r_data, pp, npairs = without_refused(
                np.column_stack([x[:g], y[:g]]),
                np.column_stack([x[g:], y[g:]]),
                [[pair_anchor, j] for j in centres if j != pair_anchor],
                nuisance_pairs,
            )
            validation = dict(
                global_data=g_data, ridge_data=r_data, profile_pairs=pp, nuisance_pairs=npairs
            )
            with (out / "validation.tmp").open("wb") as f:
                np.savez(f, **validation)
            (out / "validation.tmp").replace(out / "validation.npz")
        status("2D frozen-scaler training")
        if not (out / "training/training.json").exists():
            preset = ProfileTrainingPreset(
                **{k: getattr(c, k) for k in ProfileTrainingPreset.__dataclass_fields__}
            )
            fit_profile_model(
                model,
                data,
                validation,
                out / "training",
                preset=preset,
                original_rows=original_rows,
                counts=aggregate_counts(out / "calls"),
            )
        model = load_model(out / "training/model_contrast.keras")
        searched = model if see is None else see(model)
    status("2D surrogate grid")
    surface = None if legacy else finished_scan(out / "surrogate_surface.json", axes, c.parameters, md.param_names)
    if surface is not None:
        # The node checkpoints of the true stage are tied to the scan's optima bit for bit
        # (their signature holds every start). A scan done again on a resume had to come out
        # identical for the study to go on; the finished one is read instead.
        print("The surrogate grid of the earlier start of this study is used again (surrogate_surface.json).")
    else:
        surface = scan_checked(axes, out / "surrogate_surface.json")
        surface.update(names=c.parameters, full_parameter_names=md.param_names)
        save(out / "surrogate_surface.json", surface)
    if region is not None:
        # A node whose optimum is at the region's edge or beyond it cannot be computed from
        # this network (profiles/region.py): its optimum is a point of the wall, which the scan
        # certifies, and the true stage spent most of a study's calls on such nodes without
        # one of them converging. A range is cut at its ends; a rectangle is not cut, so the
        # study stops here, before any true call, and says which nodes.
        from profiles.region import INSIDE

        optimum = np.asarray(surface["xopt"], float)
        rho = region.rho(optimum.reshape(-1, optimum.shape[-1])).reshape(optimum.shape[:2])
        beyond = np.argwhere(rho > INSIDE * region.inflate)
        if len(beyond):
            listed = "; ".join(f"({axes[0][i]:g}, {axes[1][j]:g}) rho {rho[i, j]:.3g}" for i, j in beyond[:6])
            raise RuntimeError(
                f"{len(beyond)} of {rho.size} nodes of the surface in {c.parameters[0]} and {c.parameters[1]} have "
                f"their optimum outside the region of the network's boosts (edge {region.inflate:g}): {listed}"
                + (f" and {len(beyond) - 6} more" if len(beyond) > 6 else "")
                + ". Give ranges inside it (surrogate_surface.json has every node's optimum), or set "
                "`region` in the profile settings (a larger edge, or null for the bare network)")
    status("2D direct and optimized true grid")
    continuation = getattr(c, "continuation", None)
    if continuation is not None and continuation.get("start", "best") == "best":
        # The rows' seeds start from the best training point (profiles/hybrid_surface.py);
        # the point enters the true stage's signature.
        surface = dict(surface, seed_start=data[int(np.argmax(data[:, -1])), :-1].tolist())
    refine_surface_newton(searched, surface, bounds, ledger, out, c)
    save(out / "counts.json", aggregate_counts(out / "calls"))
    from profiles.surface_reporting import write_surface_outputs

    write_surface_outputs(out)
    save(
        out / "complete.json",
        dict(
            dimension=2,
            pair=c.parameters,
            grid_shape=[len(a) for a in axes],
            added_points=c.added_points,
            training_performed=c.retrain,
            counts=aggregate_counts(out / "calls"),
            accuracy_certified=False,
        ),
    )
