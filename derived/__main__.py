"""python client.py derived RUN_DIR [--iteration N] [--retrain] [--chain [-n STEPS] [--thin K]]

Train the derived-parameter emulator on the derived values a CLiENT run recorded with
its true evaluations, without re-running anything. Uses the run YAML's `derived:`
section, or the DerivedConfig defaults when there is none, and writes
derived_data/derived_model_it_N.keras with its training history and a held-out
accuracy report. An emulator that already exists for the iteration is reused unless
--retrain is given.

--chain then samples that iteration's surrogate posterior (T = 1), evaluates the
emulator at every chain point, and writes posterior_chains/surrogate_it_N.txt with
.paramnames and .ranges, which GetDist loads with derived parameters marked as such.
"""

import argparse
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# 216 walkers x 100,000 steps unthinned is 21.6M rows: about 8 GB of GetDist text. Far
# more rows than a derived-parameter posterior needs, since consecutive steps of a
# chain with tau ~ 10^3 are nearly identical.
MAX_CHAIN_ROWS = 500_000


def default_thin(sampling, n_steps):
    return max(sampling.thin, -(-n_steps * sampling.n_walkers // MAX_CHAIN_ROWS))


def latest_iteration(derived_data_dir, trained_models_dir=None):
    """The latest iteration with recorded derived values (and, if a directory is
    given, a trained surrogate as well)."""
    iterations = [
        int(match.group(1))
        for path in Path(derived_data_dir).glob("derived_it_*.csv")
        if (match := re.fullmatch(r"derived_it_(\d+)", path.stem))
    ]
    if trained_models_dir is not None:
        iterations = [
            i for i in iterations
            if (Path(trained_models_dir) / f"model_it_{i}.keras").exists()
        ]
    if not iterations:
        raise FileNotFoundError(
            f"No derived_it_<N>.csv in {derived_data_dir}"
            + ("" if trained_models_dir is None else " with a matching trained model")
            + ": this run recorded no derived parameters (made before they were "
            "recorded, or its likelihood has none)."
        )
    return max(iterations)


def metadata_from_csv(path, param_names):
    from derived.metadata import DerivedMetadata
    from likelihood.base import ParameterInfo

    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    return DerivedMetadata(
        parameters=tuple(
            ParameterInfo(name=name, label=name) for name in header if name not in param_names
        ),
        inputs=(),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the derived-parameter emulator of a CLiENT run, and "
                    "optionally write a posterior chain with derived columns."
    )
    parser.add_argument("run", type=Path, help="run directory")
    parser.add_argument(
        "-i", "--iteration", type=int,
        help="iteration to use (default: the latest with derived values"
             " and, with --chain, a trained surrogate)",
    )
    parser.add_argument("--retrain", action="store_true",
                        help="train the emulator even if one exists for the iteration")
    parser.add_argument("--region", choices=("run", "none"),
                        help="chain target. run: the network inside the region of the run's boosts, and stop "
                             "when the run has none; none: the unrestricted network. Default: the region "
                             "when the run has one (as `client.py evidence`)")
    parser.add_argument("--chain", action="store_true",
                        help="sample the surrogate posterior and write it with derived columns")
    parser.add_argument("-n", "--n-steps", type=int,
                        help="chain steps per walker (default: sampling.n_steps)")
    parser.add_argument(
        "--thin", type=int,
        help="keep every K-th chain step (default: sampling.thin, raised so the written "
             f"chain holds at most about {MAX_CHAIN_ROWS:,} rows)",
    )
    parser.add_argument(
        "--exact", type=int, default=0, metavar="N",
        help="evaluate the TRUE likelihood at N chain points and compare the emulated "
             "derived values and log-likelihood with it (implies --chain; run under "
             "mpirun to spread the evaluations)",
    )
    args = parser.parse_args(argv)
    args.chain = args.chain or args.exact > 0

    from utils.mpi_utils import broadcast_and_evaluate, is_master

    if not is_master():
        # Workers only share the true evaluations; rank 0 does everything else and
        # releases them when the points are ready.
        if args.exact:
            from derived.chain import true_likelihood

            likelihood = true_likelihood(args.run)
            broadcast_and_evaluate(None, likelihood.evaluate_with_derived)
        return

    import tensorflow as tf

    from config.config import Config, DerivedConfig
    from run.run import settings_file
    from dataset.dataset import TrainingDataset
    from derived.emulator import load_derived_model, model_path, train_and_save
    from derived.metadata import DerivedMetadata
    from likelihood.surrogate import SurrogateMetadata

    run = args.run
    derived_dir = run / "derived_data"
    cfg = Config.from_yaml(settings_file(run))
    derived_cfg = cfg.derived or DerivedConfig()

    param_names = SurrogateMetadata.load(run / "metadata.json").param_names
    if args.iteration is not None:
        iteration = args.iteration
    else:
        iteration = latest_iteration(
            derived_dir, run / "trained_models" if args.chain else None
        )
    try:
        metadata = DerivedMetadata.load(derived_dir)
    except FileNotFoundError:
        # Names from the data file itself; with no recorded dependencies, 'auto'
        # falls back to every input.
        metadata = metadata_from_csv(derived_dir / f"derived_it_{iteration}.csv", param_names)
        print(f"No derived_parameters.json in {derived_dir}; using the columns of "
              f"derived_it_{iteration}.csv and all inputs")

    if cfg.seed is not None:
        tf.keras.utils.set_random_seed(cfg.seed)
    seed = None if cfg.seed is None else cfg.seed + iteration

    path = model_path(derived_dir, iteration)
    if path.exists() and not args.retrain:
        print(f"Using the existing emulator {path.name} (--retrain to replace it)")
        model = load_derived_model(path)
    else:
        # TrainingDataset needs only the names from its likelihood.
        names = SimpleNamespace(param_names=param_names, derived_names=metadata.names)
        dataset = TrainingDataset.load(
            run / "training_data",
            names,
            n_neighbors=cfg.acquisition.n_neighbors,
            target_temperature=cfg.acquisition.target_temperature,
            iteration=iteration,
            derived_data_dir=derived_dir,
        )
        print(f"Run {run.name}, iteration {iteration}: {len(dataset.inputs)} training points")
        model, _ = train_and_save(
            dataset.inputs, dataset.derived, dataset.targets, param_names, metadata,
            derived_cfg, derived_dir, iteration, rng=np.random.default_rng(seed),
        )

    if args.chain:
        from derived.chain import chain_with_derived, summarize
        from sampling.targets import region_file

        # asked for by name, the region must be there: without one the chain was the bare
        # network's, without a word
        if args.region == "run" and region_file(run, iteration) is None:
            parser.error("--region run: the run has no saved boost region at or before this iteration")
        n_steps = cfg.sampling.n_steps if args.n_steps is None else args.n_steps
        thin = default_thin(cfg.sampling, n_steps) if args.thin is None else args.thin
        root, chain, log_prob, derived = chain_with_derived(
            run, iteration, model, metadata, n_steps=n_steps, thin=thin, seed=seed,
                region=None if args.region == "none" else "run",
        )
        print("Derived parameters on the surrogate posterior:\n"
              + summarize(metadata.names, derived))

    if args.exact:
        import pandas as pd

        from derived.chain import compare_exact, format_exact, pick_points, true_likelihood, weights_problem

        points, surrogate_logpost, emulated = pick_points(
            chain, log_prob, derived, args.exact, np.random.default_rng(seed)
        )
        likelihood = true_likelihood(run)
        print(f"Evaluating the true likelihood at {len(points)} posterior points")
        _, values = broadcast_and_evaluate(points, likelihood.evaluate_with_derived)
        true_loglkl, exact_all = likelihood.split_evaluations(values)

        # Match by name: the backend may report derived parameters the emulator
        # was not trained on, or the other way round.
        columns = [likelihood.derived_names.index(n) if n in likelihood.derived_names else None
                   for n in metadata.names]
        exact = np.column_stack([
            exact_all[:, c] if c is not None else np.full(len(points), np.nan)
            for c in columns
        ])
        rows, loglkl_summary = compare_exact(
            metadata.names, emulated, exact, derived, surrogate_logpost, true_loglkl,
            weights_problem=weights_problem(root),
        )
        print("Exact check on the surrogate posterior:\n" + format_exact(rows, loglkl_summary))

        table = pd.DataFrame(points, columns=param_names)
        table["loglkl_true"] = true_loglkl
        table["logpost_surrogate"] = surrogate_logpost
        for j, name in enumerate(metadata.names):
            table[f"{name}_exact"] = exact[:, j]
            table[f"{name}_emulated"] = emulated[:, j]
        out = Path(f"{root}_exact.csv")
        table.to_csv(out, index=False)
        pd.DataFrame(rows).assign(**{k: v for k, v in loglkl_summary.items()}).to_csv(
            Path(f"{root}_exact_summary.csv"), index=False
        )
        print(f"Wrote {out.name} and {out.stem}_summary.csv")


if __name__ == "__main__":
    main()
