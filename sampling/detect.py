"""python client.py modes RUN_DIR [-i N] [--mixing M] [--barrier-only] [--tempered-steps S] [--seed S] [--tag T]

The modes of a saved network's posterior, before deciding to boost or to quote a result: how
many there are, what separates them, whether a T = 1 chain mixes between them, and their
weights. Only the network is evaluated: no likelihood, no Boltzmann code, no MPI.

What a boost iteration does first, on its own: a tempered chain at the run's sampling
temperature, an ensemble cooled from it to T = 1, every walker climbed to its maximum, the
maxima grouped into modes, one T = 1 chain per mode, and the evidence of each mode by bridge
sampling (sampling/modes.py). By default the modes are decided by measured mixing (--mixing;
the run's own acquisition.boost.mixing when it has one, else 0.1: two candidate basins are one
mode when a free T = 1 chain has forgotten which of them it started in to within that
memory); --barrier-only uses the fixed barrier of the run's acquisition.boost block instead
(docs/boost_modes_mixing.md). The other settings (walkers, steps, barrier, ...) are the
run's boost settings, or the defaults when the run has none; with a saved region at or before
the iteration (a run with acquisition.boost.region), the network is used inside that region,
as the loop does.

Writes RUN_DIR/modes/modes_it_N[_TAG].json (the modes: weight, evidence and its spread, peak,
mean and standard deviation; the mixing test pair by pair) and .npz (the T = 1 samples of
every mode, x0, f0, x1, ..., and `weights`: together the posterior, each mode's samples
carrying its weight).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from run.run import latest_model, settings_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path, help="run directory")
    parser.add_argument("-i", "--iteration", type=int, help="network iteration (default: the latest)")
    parser.add_argument("--mixing", type=float,
                        help="memory left after the chain below which two basins are one mode "
                             "(default: the run's acquisition.boost.mixing, else 0.1)")
    parser.add_argument("--barrier-only", action="store_true",
                        help="decide by the fixed barrier, without the mixing test")
    parser.add_argument("--tempered-steps", type=int,
                        help="steps of the tempered chain (default: the run's sampling.n_steps)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", default="", help="suffix for the output names")
    args = parser.parse_args(argv)
    if args.mixing is not None and not 0 < args.mixing < 1:
        parser.error("--mixing must be between 0 and 1")

    import tensorflow as tf

    from config.config import BoostConfig, Config
    from sampling.base import build_sampler
    from sampling.modes import posterior_modes
    from sampling.targets import load_snapshot

    run = args.run
    cfg = Config.from_yaml(settings_file(run))
    boost = cfg.acquisition.boost or BoostConfig(n=1, start_iteration=0)
    mixing = args.mixing if args.mixing is not None else boost.mixing if boost.mixing is not None else 0.1
    iteration = latest_model(run) if args.iteration is None else args.iteration
    seed = cfg.seed if args.seed is None else args.seed
    if seed is not None:                              # before the network is loaded, as it always was
        tf.keras.utils.set_random_seed(seed + 31 * iteration)
    rng = np.random.default_rng(None if seed is None else seed + 31 * iteration)
    # The network, the run's metadata and the region that belongs to the network (by the rule
    # of the other consumers: the latest one saved at or before its iteration), from one
    # verified private copy, with the identity of those files for the output: a network
    # replaced while the search runs cannot be the one the result is recorded for.
    surrogate, target = load_snapshot(run, iteration)
    metadata, region = surrogate.metadata, surrogate.region
    saved = None if region is None else Path(target["region"]["file"])
    names = metadata.param_names
    lower = np.array([metadata.bounds[n][0] for n in names], float)
    upper = np.array([metadata.bounds[n][1] for n in names], float)
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
        parser.error("the run's domain is unbounded in some parameter")

    temperature = float(cfg.sampling.temperature)
    steps = cfg.sampling.n_steps if args.tempered_steps is None else args.tempered_steps
    burn_in = min(cfg.sampling.burn_in, steps // 4)
    print(f"run {run.name}, network {iteration}: tempered chain at T = {temperature:g} "
          f"({cfg.sampling.n_walkers} walkers, {steps} steps)"
          + ("; network used inside the region of " + saved.name if region is not None else ""), flush=True)
    started = time.monotonic()
    sampler = build_sampler(cfg.sampling.sampler, cfg.sampling.n_walkers, len(names),
                            lambda x: surrogate.logpost(x) / temperature)
    start = (rng.uniform(lower, upper, size=(cfg.sampling.n_walkers, len(names))) if region is None
             else region.start(cfg.sampling.n_walkers, rng))
    sampler.run(n_steps=steps, initial_positions=start.astype(np.float32), burn_in=burn_in,
                chunk_size=cfg.sampling.chunk_size, thin=cfg.sampling.thin, progress=False)
    chain = np.array(sampler.chain())
    tempered = chain[-max(1, len(chain) // 10):]
    sampler.reset()

    modes, info = posterior_modes(
        surrogate.loglkl, lower, upper, tempered, temperature, barrier=boost.barrier,
        min_share=boost.min_share, walkers=boost.walkers, cool_steps=boost.cool_steps,
        steps=boost.steps, components=boost.components, seed=seed, log=lambda text: print(text, flush=True),
        **({} if args.barrier_only else {"mixing": mixing, "candidate_barrier": boost.candidate_barrier}),
        **({} if boost.mode_depth is None else {"mode_depth": boost.mode_depth}))

    out = run / "modes"
    out.mkdir(exist_ok=True)
    stem = f"modes_it_{iteration}" + (f"_{args.tag}" if args.tag else "")
    arrays = {"weights": np.array([m["weight"] for m in modes]), "names": np.array(names)}
    for k, m in enumerate(modes):
        arrays[f"x{k}"], arrays[f"f{k}"] = m["x"], m["f"]
    from run.storage import atomic_path, write_json

    with atomic_path(out / f"{stem}.npz") as temporary:
        np.savez_compressed(temporary, target_json=json.dumps(target, sort_keys=True), **arrays)
    report = dict(
        run=run.name, iteration=iteration, temperature=temperature, region=saved.name if region is not None else None,
        target=target,
        decided_by="fixed barrier" if args.barrier_only else "measured mixing", parameters=names,
        modes=[dict(weight=m["weight"], log_evidence=m["log_evidence"], log_evidence_spread=m["log_evidence_spread"],
                    efficiency=m["efficiency"], walkers=m["walkers"], samples=int(len(m["x"])),
                    peak=np.asarray(m["peak"], float).tolist(), peak_value=m["peak_value"],
                    mean=m["x"].mean(axis=0).tolist(), std=m["x"].std(axis=0).tolist()) for m in modes],
        search=info, seconds=time.monotonic() - started)
    write_json(out / f"{stem}.json", report)

    print(f"\n{len(modes)} mode{'s' if len(modes) > 1 else ''} ({report['decided_by']}):")
    for k, m in enumerate(modes):
        print(f"  mode {k + 1}: weight {100 * m['weight']:.1f}%, log evidence {m['log_evidence']:.2f} "
              f"+- {m['log_evidence_spread']:.2f}, highest value {m['peak_value']:.2f}")
        if len(modes) > 1:
            first = modes[0]
            spread = np.sqrt(0.5 * (first["x"].var(axis=0) + m["x"].var(axis=0)))
            apart = np.abs(m["x"].mean(axis=0) - first["x"].mean(axis=0)) / np.maximum(spread, 1e-300)
            if k:
                order = np.argsort(-apart)[:3]
                print("          differs from mode 1 most in " + ", ".join(f"{names[j]} ({apart[j]:.1f} sigma)" for j in order))
    for pair in info.get("mixing", {}).get("pairs", []):
        a, b = pair["basins"]
        print(f"  basins {a + 1} and {b + 1}: {pair['dip']:.1f} log-units between their maxima, memory "
              f"{pair['memory_after_chain']:.2f} after the chain, {pair['exchanged']} exchanges: "
              f"{'one mode' if pair['one_mode'] else 'separate modes'}")
    if info["dropped_walkers"]:
        print(f"  {info['dropped_walkers']} of {info['walkers']} cooled walkers in smaller groups were left out")
    print(f"written: {out / stem}.json and .npz ({report['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
