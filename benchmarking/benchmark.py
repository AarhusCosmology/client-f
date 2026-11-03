# benchmarking/benchmark.py

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark surrogate likelihood")
    parser.add_argument("run_dir", help="Path to run directory")
    parser.add_argument(
        "-i",
        "--iteration",
        type=int,
        default=None,
        help="Iteration to benchmark (auto-detects latest if not specified)",
    )
    parser.add_argument(
        "-n",
        "--n-steps",
        type=int,
        default=None,
        help="Number of MCMC steps (defaults to n_steps from config)",
    )
    parser.add_argument("-t", "--thin", type=int, default=1, help="Thinning factor for chains")
    parser.add_argument("-p", "--params", nargs="+", default=None, help="Parameter indices to include in analysis")
    parser.add_argument(
        "-c",
        "--chains",
        default=None,
        help="Path to MontePython or Cobaya chains directory for comparison",
    )
    parser.add_argument(
        "--region",
        choices=["run", "none"],
        default="run",
        help="run (default): the network inside the region of its boosts, as CLiENT's other consumers use it, "
             "when it has one; none: the bare network",
    )
    parser.add_argument("--no-training-data", action="store_true", help="Skip loading training data visualization")
    parser.add_argument("--no-training-history", action="store_true", help="Skip loading training history")
    return parser.parse_args()


def load_run_config(run_dir):
    import yaml

    from run.run import settings_file

    config_yaml = settings_file(run_dir)
    with open(config_yaml, encoding="utf-8") as f:
        return config_yaml, yaml.safe_load(f)


def resolve_iteration(run_dir, requested_iteration):
    if requested_iteration is not None:
        return requested_iteration

    trained_models_dir = run_dir / "trained_models"
    if not trained_models_dir.exists():
        raise FileNotFoundError(f"trained_models directory not found in {run_dir}")

    model_files = list(trained_models_dir.glob("model_it_*.keras"))
    if not model_files:
        raise FileNotFoundError(f"No trained models matching model_it_*.keras found in {trained_models_dir}")

    iterations = []
    for model_file in model_files:
        try:
            iterations.append(int(model_file.stem.split("_")[-1]))
        except ValueError:
            continue

    if not iterations:
        raise ValueError(f"Could not parse iteration numbers from model files in {trained_models_dir}")

    iteration = max(iterations)
    print(f"Auto-detected latest iteration: {iteration}")
    return iteration


def load_surrogate(run_dir, iteration, region="run"):
    """(surrogate, its metadata, the target's identity), from a snapshot like every other consumer."""
    from sampling.targets import load_snapshot

    surrogate, target = load_snapshot(run_dir, iteration, None if region == "none" else region)
    return surrogate, surrogate.metadata, target


def sampling_note(info):
    """The log's line on how the surrogate's chain was sampled. A chain that was reused says it
    too, and its comparisons are only as good as the weights it says are not reliable."""
    from sampling.provenance import sampling_correction_problem

    problem = sampling_correction_problem(info)
    return f"Surrogate chain: {info['method']}" + (
        f". WARNING: {problem}; the posterior comparisons below rest on those weights" if problem else "")


def load_training_samples(run_dir, iteration, param_names):
    import pandas as pd

    data_path = run_dir / "training_data" / f"data_it_{iteration}.csv"
    if not data_path.exists():
        return None

    return pd.read_csv(data_path)[param_names].to_numpy()


def chain_key(target, sampling_config, sampler):
    """What a saved chain was sampled from and with: a network replaced at the same path (a
    branch's), another region, other walkers or another sampling method do not reuse it."""
    import hashlib

    settings = {key: sampling_config.get(key) for key in ("sampler", "n_walkers", "burn_in", "chunk_size")}
    text = json.dumps(dict(model=target["model_sha256"], region=target["region"], settings=settings, sampler=sampler), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_or_run_chain(run_dir, iteration, sampling_config, n_steps, thin, prior_bounds, surrogate, target):
    from types import SimpleNamespace

    import numpy as np

    from derived.chain import sample_posterior
    from run.storage import atomic_path
    from sampling.provenance import read_sampling, sampler_identity

    sampler = sampler_identity(surrogate.region)
    sampler_name = sampling_config["sampler"]
    output_dir = run_dir / "benchmark_chains"
    output_dir.mkdir(exist_ok=True)
    chain_path = output_dir / (
        f"benchmark_chain_it_{iteration}_{sampler_name}_"
        f"steps{n_steps}_thin{thin}_{chain_key(target, sampling_config, sampler)}.npz"
    )

    if chain_path.exists():
        data = np.load(chain_path)
        info = read_sampling(data)
        # A chain without the record of how it was sampled is sampled again: its weights could
        # not be vouched for.
        if info is not None:
            chain = data["chain"]
            log_prob = data["log_prob"]
            print(f"Loaded chain: {chain.shape}")
            return chain_path, chain, log_prob, info

    n_walkers = sampling_config["n_walkers"]
    burn_in = sampling_config["burn_in"]
    print(f"Running {sampler_name} sampler: {n_walkers} walkers, {burn_in} burn-in, {n_steps} steps")

    bounds = np.asarray(list(prior_bounds.values()), dtype=float)
    if not np.all(np.isfinite(bounds)):
        raise ValueError("Benchmark sampling requires finite prior bounds for all parameters")

    # The derived chain's sampling: from the region when there is one, cell by cell when it
    # has several. Thinned at write time, so the buffer is only n_steps/thin long.
    settings = SimpleNamespace(sampler=sampler_name, n_walkers=n_walkers, burn_in=burn_in,
                               chunk_size=sampling_config.get("chunk_size", 5000))
    diagnostics = {}
    chain, log_prob = sample_posterior(surrogate, bounds, settings, n_steps, thin, diagnostics=diagnostics)
    info = dict(sampler, **diagnostics)
    with atomic_path(chain_path) as temporary:
        np.savez(temporary, chain=chain, log_prob=log_prob, sampling_json=json.dumps(info))
    print(f"Chain shape: {chain.shape}")
    return chain_path, chain, log_prob, info


def sample_inputs(chain, log_prob, sampler_name, info):
    """GetDist's input. A mixture of cells is an equal-weight resample, whose axes are storage,
    not chronological walkers: one block, and no convergence statistics from it."""
    from benchmarking.getdist_utils import getdist_sample_inputs

    if info.get("resampled"):
        return chain.reshape(-1, chain.shape[-1]), -log_prob.reshape(-1), False
    return getdist_sample_inputs(chain, log_prob, sampler_name)


def make_getdist_samples(chain, log_prob, sampler_name, getdist_names, param_labels, getdist_ranges, info):
    from getdist import MCSamples

    samples, loglikes, convergence_available = sample_inputs(chain, log_prob, sampler_name, info)
    return (
        MCSamples(
            samples=samples,
            names=getdist_names,
            labels=param_labels,
            loglikes=loglikes,
            ranges=getdist_ranges,
        ),
        convergence_available,
    )


def load_reference_samples(chains_path, param_names, getdist_names, param_labels, getdist_ranges, scales):
    from getdist import MCSamples

    from benchmarking.chains import detect_chain_format, load_cobaya_chains, load_montepython_chains

    if not chains_path:
        return None, None

    try:
        chain_format = detect_chain_format(chains_path)
        print(f"Detected {chain_format} chain format")

        if chain_format == "cobaya":
            samples_list, loglikes_list, weights_list = load_cobaya_chains(chains_path, param_names, thin=1)
        else:
            samples_list, loglikes_list, weights_list = load_montepython_chains(
                chains_path,
                param_names,
                thin=1,
                scales=scales,
            )

        reference_samples = MCSamples(
            samples=samples_list,
            weights=weights_list,
            names=getdist_names,
            labels=param_labels,
            loglikes=[-loglikes for loglikes in loglikes_list],
            ranges=getdist_ranges,
        )
        print(f"Loaded {sum(len(samples) for samples in samples_list)} samples from {chain_format} chains")
        return reference_samples, chain_format
    except Exception as e:
        print(f"Warning: Could not load chains: {e}")
        return None, None


def main():
    args = parse_args()

    from benchmarking.diagnostics import DiagnosticsConfig, TeeOutput, print_diagnostics
    from benchmarking.getdist_utils import (
        getdist_names_for_params,
        getdist_ranges_for_params,
        select_plot_params,
    )
    from benchmarking.plotting import save_training_history_plot, save_triangle_plot

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir)

    config_yaml, config = load_run_config(run_dir)
    n_steps = config["sampling"]["n_steps"] if args.n_steps is None else args.n_steps
    iteration = resolve_iteration(run_dir, args.iteration)
    surrogate, metadata, target = load_surrogate(run_dir, iteration, args.region)
    print(f"Surrogate region: {target['region'] or 'none (bare network)'}")

    param_names = metadata.param_names
    prior_bounds = metadata.bounds
    param_labels = metadata.param_labels
    getdist_names = getdist_names_for_params(param_names)
    getdist_ranges = getdist_ranges_for_params(param_names, getdist_names, prior_bounds)

    training_samples = load_training_samples(run_dir, iteration, param_names)
    sampler_name = config["sampling"]["sampler"]
    chain_path, chain, log_prob, sampling_info = load_or_run_chain(
        run_dir,
        iteration,
        config["sampling"],
        n_steps,
        args.thin,
        prior_bounds,
        surrogate,
        target,
    )
    samples, surrogate_convergence_available = make_getdist_samples(
        chain,
        log_prob,
        sampler_name,
        getdist_names,
        param_labels,
        getdist_ranges,
        sampling_info,
    )

    plot_params, param_indices = select_plot_params(args.params, param_names, getdist_names)
    reference_samples, reference_sampler = load_reference_samples(
        args.chains,
        param_names,
        getdist_names,
        param_labels,
        getdist_ranges,
        metadata.scales,
    )

    diagnostics_config = DiagnosticsConfig(
        iteration=iteration,
        config_yaml=config_yaml,
        run_dir=run_dir,
        thin=args.thin,
        n_steps=n_steps,
        chains_path=args.chains,
        surrogate_sampler=sampler_name,
        reference_sampler=reference_sampler,
        surrogate_convergence_available=surrogate_convergence_available,
    )

    log_file_path = run_dir / "benchmark_results" / f"{timestamp}_diagnostics_it_{iteration}.log"
    with TeeOutput(str(log_file_path)):
        print(sampling_note(sampling_info))
        print_diagnostics(
            samples,
            reference_samples,
            param_names,
            getdist_names,
            diagnostics_config,
            surrogate,
        )

    figure_dir = run_dir / "benchmark_figures"
    save_triangle_plot(
        figure_dir,
        timestamp,
        iteration,
        samples,
        reference_samples,
        plot_params,
        param_indices,
        getdist_ranges,
        training_samples=None if args.no_training_data else training_samples,
        surrogate_sampler=sampler_name,
        reference_sampler=reference_sampler,
    )

    if not args.no_training_history:
        save_training_history_plot(
            figure_dir,
            timestamp,
            iteration,
            run_dir / f"training_history/history_it_{iteration}.csv",
        )

    print(f"\n=== Benchmark Complete ===\nChain: {chain_path}\nPlots: {figure_dir}\nLog: {log_file_path}")


if __name__ == "__main__":
    main()
