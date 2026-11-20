"""The posterior of a network with a prior flat in 10^p, for a parameter p that the run sampled with a prior flat in p,
and its check against the true likelihood (9 November 2026; first use: log10Gamma of the DNCDM proxy runs).

    python analysis/linear_prior_posterior.py sample RUN_DIR [--network N] [--parameter log10Gamma]
                                                    [--steps 20000] [--walkers 256] [--seed 1]
    python analysis/linear_prior_posterior.py check RUN_DIR --cobaya runs/RUN/cobaya.yaml [--network N]
                                                    [--parameter log10Gamma] [--points 500] [--workers 8] [--seed 1]

Run from this folder with the run's job.env sourced (the true likelihood needs its CLASS++ build).

A network learns the run's log posterior, ln L plus the log of the prior it was trained with. A prior flat in 10^p
instead of p multiplies the density by 10^p, i.e. adds ln(10) p to the target, up to a constant. `sample` samples the
network with that term added, inside its region and cell by cell where the region has several ellipsoids, with the
sampler of CLiENT's report (hpc/analysis/report.py, sample_posterior), and writes

  RUN_DIR/analysis/linear_prior/posterior_<p>_it_N.npz    the samples (x, names, sampling_json, prior)
  RUN_DIR/analysis/linear_prior/posterior_<p>_it_N.md     quantiles of p and 10^p, means and widths of the others

`check` evaluates the true likelihood (Cobaya, one model per process) at `points` of those samples, drawn at random,
and the network at the same points. The added prior term is the same in both, so D = truth - network is the network's
error, and w = exp(D) turns the samples into ones of the true posterior. It writes check_<p>_it_N.npz (points, network,
truth) and appends to the .md: the spread of D, the effective samples (sum w)^2 / sum w^2, and the quantiles of p with
the weights. Only as good as the effective samples behind them.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
import client_paths  # noqa: E402,F401  CLiENT and its hpc/ tools

LN10 = float(np.log(10.0))
QUANTILES = (0.025, 0.05, 0.16, 0.5, 0.84, 0.95, 0.975)


def out_dir(run):
    d = Path(run) / "analysis" / "linear_prior"
    d.mkdir(parents=True, exist_ok=True)
    return d


def wquant(x, w, q):
    i = np.argsort(x)
    c = np.cumsum(w[i])
    return np.interp(q, c / c[-1], x[i])


def summary_lines(x, names, p, w=None):
    w = np.ones(len(x)) if w is None else w
    v = x[:, names.index(p)]
    lines = [f"| quantile | {p} | 10^{p} |", "|---|---|---|"]
    lines += [f"| {q} | {wquant(v, w, q):.3f} | {10 ** wquant(v, w, q):.3g} |" for q in QUANTILES]
    lines += ["", "| parameter | mean | standard deviation |", "|---|---|---|"]
    for k, n in enumerate(names):
        m = np.average(x[:, k], weights=w)
        lines.append(f"| {n} | {m:.6g} | {np.sqrt(np.average((x[:, k] - m) ** 2, weights=w)):.3g} |")
    return lines


# ------------------------------------------------------------------------------------------------- sample
def sample(a):
    import tensorflow as tf
    from report import Run, sample_posterior   # CLiENT hpc/analysis/report.py (client_paths puts it on sys.path)

    class Tilted:
        """The region surrogate with ln(10) p added: what sample_posterior and sampling/mixture.py use of it."""

        def __init__(self, base, column):
            self.base, self.column, self.region = base, column, base.region
            self.metadata = base.metadata

        def logpost(self, x):
            x = tf.convert_to_tensor(x)
            value = self.base.logpost(x)
            return value + tf.cast(LN10 * x[:, self.column], value.dtype)

    class TiltedRun(Run):
        def surrogate(self, iteration, region=False):
            s = super().surrogate(iteration, region=region)
            return Tilted(s, self.names.index(a.parameter)) if region else s

    run = TiltedRun(a.run, client_paths.CLIENT_REPO)
    network = a.network if a.network is not None else max(run.models)
    if a.parameter not in run.names:
        raise SystemExit(f"{a.parameter} is not a parameter of {a.run}: {run.names}")
    rng = np.random.default_rng(a.seed)
    t = time.time()
    chain, acceptance = sample_posterior(run, network, a.steps, a.walkers, rng)
    x = np.asarray(chain, float).reshape(-1, run.ndim)
    cells = run.mixtures.get(network)
    out = out_dir(a.run)
    stem = f"{a.parameter}_it_{network}"
    prior = f"flat in 10^{a.parameter}: ln(10) {a.parameter} added to network {network}'s log posterior"
    np.savez(out / f"posterior_{stem}.npz", x=x.astype(np.float32), names=np.array(run.names), prior=prior,
             sampling_json=json.dumps(dict(steps=a.steps, walkers=a.walkers, seed=a.seed, acceptance=acceptance,
                                           cells=cells, seconds=time.time() - t), default=float))
    lines = [f"# {Path(a.run).name}, network {network}: prior {prior}", "",
             f"{len(x)} samples ({a.walkers} walkers, {a.steps} steps), acceptance {acceptance:.2f}, "
             f"{time.time() - t:.0f} s" + (f", {len(cells['cells'])} region cells" if cells else ""), ""]
    lines += summary_lines(x, run.names, a.parameter)
    (out / f"posterior_{stem}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ------------------------------------------------------------------------------------------------- check
_model = {}


def _init(yaml_path):
    os.chdir(KIT)                                     # the yamls' python_path (likelihoods/) is relative to this folder (papers/neutrino_decay/)
    try:
        from cobaya.model import get_model
        from cobaya.yaml import yaml_load_file

        info = yaml_load_file(str(yaml_path))
        info.pop("sampler", None)
        info.pop("output", None)
        _model["m"] = get_model(info)
    except Exception as error:                        # noqa: BLE001 - reported by the first task, which stops the pool
        _model["error"] = f"{type(error).__name__}: {error}"


def _truth(job):
    i, names, x = job
    if "error" in _model:
        return i, None, f"model: {_model['error']}"
    try:
        m = _model["m"]
        if sorted(m.parameterization.sampled_params()) != sorted(names):
            raise ValueError(f"Cobaya samples {list(m.parameterization.sampled_params())}, the run {names}")
        return i, float(m.logposterior(dict(zip(names, map(float, x))), make_finite=False).logpost), None
    except Exception as error:                        # noqa: BLE001 - recorded with the point
        return i, None, f"{type(error).__name__}: {error}"


def check(a):
    from report import Run

    run = Run(a.run, client_paths.CLIENT_REPO)
    network = a.network if a.network is not None else max(run.models)
    stem = f"{a.parameter}_it_{network}"
    out = out_dir(a.run)
    saved = np.load(out / f"posterior_{stem}.npz")
    names = [str(n) for n in saved["names"]]
    if names != run.names:
        raise SystemExit("the samples' parameters are not the run's")
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(saved["x"]), size=min(a.points, len(saved["x"])), replace=False)
    points = saved["x"][pick].astype(float)
    net = run.predict(network, points, region=False)
    order = np.argsort(-points[:, names.index(a.parameter)])         # the slow points (large p for Gamma) first
    truth = np.full(len(points), np.nan)
    errors = {}
    t = time.time()
    print(f"== {len(points)} points of {out / f'posterior_{stem}.npz'} on {a.workers} workers", flush=True)
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(str(Path(a.cobaya).resolve()),)) as pool:
        for n, (i, value, error) in enumerate(pool.imap_unordered(
                _truth, [(int(k), names, points[k]) for k in order], chunksize=1), 1):
            if error and error.startswith("model:"):
                pool.terminate()
                raise SystemExit(f"FATAL: the Cobaya model could not be built: {error}")
            if error:
                errors[i] = error
            else:
                truth[i] = value
            if n % 25 == 0:
                print(f"   {n}/{len(points)} after {time.time() - t:.0f} s", flush=True)
    np.savez(out / f"check_{stem}.npz", points=points, network=net, truth=truth, names=np.array(names),
             errors=json.dumps(errors))
    ok = np.isfinite(truth)
    d = truth[ok] - net[ok]
    w = np.exp(d - d.max())
    ess = w.sum() ** 2 / (w ** 2).sum()
    lines = ["", f"## True check: {ok.sum()} of {len(points)} points evaluated ({len(errors)} errors), "
             f"{time.time() - t:.0f} s on {a.workers} workers", "",
             f"truth - network: mean {d.mean():+.3f}, standard deviation {d.std():.3f}, range [{d.min():+.2f}, {d.max():+.2f}]; "
             f"effective samples {ess:.0f} of {ok.sum()}", "",
             "Weighted with exp(truth - network):", ""]
    lines += summary_lines(points[ok], names, a.parameter, w)
    lines += ["", "Unweighted, the same points:", ""] + summary_lines(points[ok], names, a.parameter)[:2 + len(QUANTILES)]
    with open(out / f"posterior_{stem}.md", "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("sample", "check"):
        s = sub.add_parser(name)
        s.add_argument("run", type=Path)
        s.add_argument("--network", type=int)
        s.add_argument("--parameter", default="log10Gamma")
        s.add_argument("--seed", type=int, default=1)
        if name == "sample":
            s.add_argument("--steps", type=int, default=20000)
            s.add_argument("--walkers", type=int, default=256)
        else:
            s.add_argument("--cobaya", type=Path, required=True)
            s.add_argument("--points", type=int, default=500)
            s.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    sample(a) if a.command == "sample" else check(a)


if __name__ == "__main__":
    main()
