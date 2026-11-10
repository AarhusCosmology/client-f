"""Convergence and errors of a CLiENT run, finished or still running, from what the run saved.

    python analysis/report.py RUN_DIR [--out DIR] [--journal logs/evaluations_<job>.jsonl ...]
                              [--no-posteriors] [--networks N ...] [--last K] [--corner N ...]
                              [--corner-parameters 6] [--reference FILE] [--steps 20000] [--client PATH]
                              [--linear-prior NAME ...]

Writes RUN_DIR/analysis/ (or --out): report.md (the tables), report.json (every number),
errors.png, convergence.png, posterior.png, training.png, corner.png, and posterior_it_N.npz
(T = 1 samples of every network, for other plots and for analysis/true_check.py). With
published values for the parameters (--reference, or runs/<RUN>/reference_values.json when it
exists) it also writes marginals.png and a table of the last sampled network's posterior
against them: is any parameter far from where published analyses put it, or against an edge
of the training box that is not a prior edge.

No true likelihood is evaluated. Saved files are read and saved networks are evaluated:

  held-out errors   Network N chose the points of iteration N's acquisition and had not seen
                    their true values. Network N at those points against the truth is a test of
                    network N where its own tempered posterior (or, in a boost, its posterior) is.
                    Given for the tempered batch, for the injected points and for the boost
                    points, and by depth below the best true value of the run.
  the audit         true calls, maxima above the threshold before and after, whether it cleared.
  training          epochs, validation loss, learning-rate reductions, time.
  posterior         every network sampled at T = 1 (inside its region when the run has one):
                    mean and width of every parameter, their distance from the last network's
                    in units of its width, and the credible metric against the last network,
                    with the same numbers between two halves of the last network's own chain
                    as the noise floor. Agreement between networks is convergence, not accuracy:
                    for accuracy on the posterior see the boost points here and true_check.py.
  evaluations       the launcher's journal: points, failures and seconds of every batch.

With --linear-prior, the named parameters p, which the run sampled with a prior flat in p (log10 of a
quantity, such as log10Gamma), get a prior flat in 10^p instead: ln(10) p is added to every network's log
posterior where it is sampled (LinearPrior). The held-out errors stay those of the networks themselves. Everything
then goes to RUN_DIR/analysis/linear_prior_<p>[+<q>...] (analysis_folder), and the chains record the prior in their
target, so that no reader of analysis/ takes them for the run's posterior. analysis/true_check.py checks them with
the same option.

It runs anywhere the run directory is.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def iteration_of(path):
    return int(re.search(r"_it_(\d+)", str(path)).group(1))


def by_iteration(pattern):
    return {iteration_of(f): Path(f) for f in glob.glob(str(pattern))}


LN10 = float(np.log(10.0))


def linear_prior_names(names, wanted):
    """The parameters named for --linear-prior, in the run's order and once each. A name that is not a parameter of
    the run ends the program: a misspelt one would give the run's own posterior under the new prior's name."""
    wanted = list(wanted or ())
    unknown = [name for name in dict.fromkeys(wanted) if name not in names]
    if unknown:
        raise SystemExit(f"--linear-prior names what the run does not sample: {', '.join(unknown)}. "
                         f"Its parameters: {', '.join(names)}")
    return tuple(name for name in names if name in wanted)


def analysis_folder(run_dir, linear_prior=()):
    """Where the report and the true check keep their chains and results: RUN_DIR/analysis, and for a posterior with
    a prior flat in 10^p (--linear-prior) RUN_DIR/analysis/linear_prior_<p>[+<q>...], beside the run's own, which the
    corner plot, the evidence summary and the profiles' make_config read."""
    folder = Path(run_dir) / "analysis"
    return folder / ("linear_prior_" + "+".join(linear_prior)) if linear_prior else folder


class LinearPrior:
    """A sampled target with a prior flat in 10^p for the parameters in `columns`, which the run sampled with a prior
    flat in p: the target's log density plus ln(10) times their sum. The region is the target's, and where the target
    is -inf (outside the region) it stays -inf. A network learned the log posterior with the run's prior, so this is
    the posterior with the new prior, up to a constant."""

    def __init__(self, target, columns):
        self.target, self.columns = target, list(columns)
        self.region = target.region

    def logpost(self, x):
        import tensorflow as tf

        x = tf.convert_to_tensor(x)
        value = self.target.logpost(x)
        tilt = LN10 * tf.reduce_sum(tf.gather(x, self.columns, axis=1), axis=1)
        return value + tf.cast(tf.reshape(tilt, tf.shape(value)), value.dtype)


def stats(error):
    """Network minus truth at a set of points."""
    error = np.asarray(error, float)
    error = error[np.isfinite(error)]
    if not len(error):
        return dict(n=0)
    # With D = truth minus network and weights w = exp(D): the share of the points that carry the
    # true distribution when the points were drawn from the network's, and the Kullback-Leibler
    # divergence from the network's distribution to the true one. They mean that only for points
    # that are samples of the network's own posterior: boost points (thinned by the density
    # filter when it is on, so approximately) and T = 1 probes; not for a tempered batch.
    d = -error
    w = np.exp(d - d.max())
    low, high = np.quantile(error, [0.16, 0.84])
    return dict(n=int(len(error)), bias=float(error.mean()), sd=float(error.std()),
                rms=float(np.sqrt(np.mean(error ** 2))), largest=float(np.abs(error).max()),
                robust_sd=float(0.5 * (high - low)),
                effective_fraction=float(w.sum() ** 2 / np.sum(w ** 2) / len(w)),
                kl=float(d.max() + np.log(np.mean(w)) - d.mean()))


def keys(frame, names):
    """Rows as byte strings of their single-precision coordinates, to match rows between files."""
    values = np.ascontiguousarray(frame[names].to_numpy(np.float32))
    return [row.tobytes() for row in values]


class Run:
    def __init__(self, directory, client, linear_prior=()):
        import yaml

        self.dir = Path(directory)
        sys.path.insert(0, str(client))
        from likelihood.surrogate import SurrogateMetadata

        from run.run import settings_file
        from run.storage import digest

        self._metadata_sha256 = digest(self.dir / "metadata.json")
        self.metadata = SurrogateMetadata.load(self.dir / "metadata.json")
        if digest(self.dir / "metadata.json") != self._metadata_sha256:
            raise RuntimeError("Run metadata changed while loading the report")
        self.names = self.metadata.param_names
        self.lower = np.array([self.metadata.bounds[n][0] for n in self.names], float)
        self.upper = np.array([self.metadata.bounds[n][1] for n in self.names], float)
        self.ndim = len(self.names)
        self.linear_prior = linear_prior_names(self.names, linear_prior)      # sampled with a prior flat in 10^p
        self.config = yaml.safe_load(settings_file(self.dir).read_text())
        self.temperature = float(self.config["sampling"]["temperature"])
        self.boost = (self.config.get("acquisition") or {}).get("boost") or None
        # A run whose ordinary loop converged before the configured start began its boosts there
        # (convergence.metric: tempered_batch_scatter; CLiENT writes boost/start.json).
        early = self.dir / "boost" / "start.json"
        if self.boost and early.exists():
            self.boost = dict(self.boost, start_iteration=min(int(self.boost["start_iteration"]),
                                                              int(json.loads(early.read_text())["start_iteration"])))
        self.data = {k: pd.read_csv(f) for k, f in sorted(by_iteration(self.dir / "training_data" / "data_it_*.csv").items())}
        self.models = by_iteration(self.dir / "trained_models" / "model_it_*.keras")
        self.audits = {k: json.loads(f.read_text()) for k, f in by_iteration(self.dir / "audit" / "audit_it_*.json").items()}
        self.audit_points = by_iteration(self.dir / "audit" / "audit_it_*.csv")
        self.boosts = {k: json.loads(f.read_text()) for k, f in by_iteration(self.dir / "boost" / "boost_it_*.json").items()}
        self.boost_points = by_iteration(self.dir / "boost" / "boost_it_*.csv")
        self.regions = by_iteration(self.dir / "boost" / "region_it_*.npz")
        self.histories = by_iteration(self.dir / "training_history" / "history_it_*.csv")
        self.summaries = by_iteration(self.dir / "convergence_stats" / "chain_summary_it_*.npz")
        try:                                                     # a running loop may be rewriting it this instant
            self.metrics = json.loads((self.dir / "metrics.json").read_text())
        except (OSError, ValueError):
            self.metrics = {}
        self.best = max((float(np.nanmax(d["loglkl"])) for d in self.data.values()), default=float("nan"))
        self._model = (None, None)
        self.mixtures = {}              # per network: the cells of a region with several ellipsoids, as sampled

    def metric(self, group, iteration):
        rows = [r for r in self.metrics.get(group, []) if r.get("iteration") == iteration]
        return rows[-1] if rows else {}

    def surrogate(self, iteration, region=False):
        from likelihood.surrogate import SurrogateLikelihood
        from sampling.targets import load_snapshot

        if self._model[0] != iteration:
            surrogate, identity = load_snapshot(self.dir, iteration)
            if identity["metadata_sha256"] != self._metadata_sha256:
                raise RuntimeError("Run metadata changed since the report began")
            self._model = (iteration, surrogate, identity)
        surrogate = self._model[1]
        if region:
            return surrogate
        return SurrogateLikelihood(model=surrogate.model, metadata=surrogate.metadata)

    def target_identity(self, iteration):
        from copy import deepcopy

        self.surrogate(iteration, region=True)  # pin the model and its region together
        identity = deepcopy(self._model[2])
        if getattr(self, "linear_prior", ()):
            identity["linear_prior"] = list(self.linear_prior)      # another target: its chains are not the run's
        return identity

    def predict(self, iteration, points, region=False):
        import tensorflow as tf

        surrogate = self.surrogate(iteration, region=region)
        evaluate = surrogate.logpost if region else surrogate.network
        points = np.asarray(points, np.float32)
        return np.concatenate([np.asarray(evaluate(tf.constant(points[i:i + 16384])), float)
                               for i in range(0, len(points), 16384)]) if len(points) else np.zeros(0)

    def acquired(self, iteration):
        """The rows that joined the training set during iteration N, by origin:
        audit (network N's audit), boost or batch + injected (its acquisition)."""
        if iteration not in self.data or iteration + 1 not in self.data:
            return None
        new = self.data[iteration + 1].iloc[len(self.data[iteration]):].reset_index(drop=True)
        origin = np.full(len(new), "batch", dtype=object)
        row_keys = keys(new, self.names)
        for label, files in (("audit", self.audit_points), ("boost", self.boost_points)):
            if iteration in files:
                known = set(keys(pd.read_csv(files[iteration]), self.names))
                origin[[k in known for k in row_keys]] = label
        rest = np.flatnonzero(origin == "batch")
        if iteration in self.boost_points:                       # a boost iteration has no batch and no injection
            origin[rest] = "boost"
        else:
            added = self.metric("acquisition", iteration).get("n_added")
            if added is not None and len(rest) > added:
                origin[rest[int(added):]] = "injected"
        return new, origin


def held_out(run, iteration):
    """Network N at the points it acquired, against their true values. Points the likelihood
    refused, which carry the run's refused value (launch.py, CLIENT_REFUSED_VALUE), are counted
    and left out of the errors: their value is a convention, not a likelihood."""
    got = run.acquired(iteration)
    if got is None or iteration not in run.models:
        return {}
    new, origin = got
    truth = new["loglkl"].to_numpy(float)
    error = run.predict(iteration, new[run.names].to_numpy(float)) - truth
    depth = run.best - truth
    floor = getattr(run, "refused_value", None)
    refused = np.isclose(truth, floor, rtol=1e-6, atol=0) if floor is not None else np.zeros(len(truth), bool)
    edges = [0.0, run.ndim, 0.5 * run.temperature * run.ndim, run.temperature * run.ndim, np.inf]
    out = {}
    for label in ("batch", "injected", "boost", "audit"):
        of_label = origin == label
        if not of_label.any():
            continue
        pick = of_label & ~refused
        if not pick.any():
            out[label] = dict(n=0, refused=int(of_label.sum()))
            continue
        entry = dict(stats(error[pick]), refused=int((of_label & refused).sum()), median_depth=float(np.median(depth[pick])),
                     depth_10=float(np.quantile(depth[pick], 0.1)), depth_90=float(np.quantile(depth[pick], 0.9)),
                     centred_sd=float(np.nanstd(error[pick])))
        entry["bands"] = [dict(stats(error[pick & (depth >= lo) & (depth < hi)]), below_best=[lo, None if np.isinf(hi) else hi])
                          for lo, hi in zip(edges[:-1], edges[1:])]
        out[label] = entry
    return out


def training(run, iteration):
    if iteration not in run.histories:
        return {}
    h = pd.read_csv(run.histories[iteration])
    out = dict(epochs=int(len(h)), val_loss_min=float(h["val_loss"].min()), val_loss_last=float(h["val_loss"].iloc[-1]),
               epoch_of_min=int(h["val_loss"].idxmin()) + 1)
    if "learning_rate" in h:
        rate = h["learning_rate"].to_numpy(float)
        out.update(learning_rate_last=float(rate[-1]), learning_rate_reductions=int(np.sum(np.diff(rate) < -1e-12)))
    out["seconds"] = run.metric("training", iteration).get("training_time")
    return out


def sample_posterior(run, iteration, steps, walkers, rng):
    """T = 1 samples of network N (inside its region when it has one): (samples, acceptance). With the run's
    linear_prior, of its posterior with a prior flat in 10^p for those parameters (LinearPrior)."""
    from sampling.base import build_sampler

    surrogate = run.surrogate(iteration, region=True)
    linear = getattr(run, "linear_prior", ())
    if linear:
        surrogate = LinearPrior(surrogate, [run.names.index(name) for name in linear])
    burn = steps // 4
    if surrogate.region is not None and len(surrogate.region) > 1:
        # A region of several ellipsoids (a posterior with several modes). One ensemble started
        # on all of them keeps the shares of its start wherever it does not cross between them:
        # the report's posterior and the true check's mean of exp(truth - network) were then
        # those of the start. Every cell is sampled by itself and weighted by the network's
        # evidence in it (CLiENT's sampling/mixture.py, as the derived chain does).
        from types import SimpleNamespace

        from sampling.mixture import sample_region_mixture

        cells = {}
        chain, _ = sample_region_mixture(
            surrogate, np.column_stack([run.lower, run.upper]),
            SimpleNamespace(n_walkers=walkers, burn_in=burn, chunk_size=5000, progress=False),
            steps, max(1, (steps - burn) // 250), seed=int(rng.integers(2 ** 31)), diagnostics=cells)
        run.__dict__.setdefault("mixtures", {})[iteration] = cells
        return chain, float(np.mean([c["acceptance"] for c in cells["cells"]]))
    if surrogate.region is not None:
        start = surrogate.region.start(walkers, rng).astype(np.float32)
    else:
        summary = np.load(run.summaries[iteration])
        cov = np.atleast_2d(summary["cov"]) / max(run.temperature, 1.0)
        start = rng.multivariate_normal(summary["mean"], cov, size=walkers, check_valid="ignore")
        pad = 1e-6 * (run.upper - run.lower)
        start = np.clip(start, run.lower + pad, run.upper - pad).astype(np.float32)
    sampler = build_sampler("aies", walkers, run.ndim, surrogate.logpost)
    sampler.run(n_steps=steps, initial_positions=start, burn_in=burn, chunk_size=5000,
                thin=max(1, (steps - burn) // 250), progress=False)
    chain = np.array(sampler.chain())
    acceptance = float(np.mean(np.asarray(sampler.acceptance_fraction())))
    sampler.reset()
    return chain, acceptance


def summarise(samples):
    # Double precision: a single-precision sum over 1e5 samples rounds each term to about 1e-2
    # of its size, and a parameter known far better than that (100 theta to 4e-4) came out
    # with its mean two widths off and its width doubled.
    samples = np.asarray(samples, dtype=np.float64)
    q = np.quantile(samples, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0)
    return dict(mean=samples.mean(axis=0), sd=samples.std(axis=0), q=q)


def compare(a, reference):
    """Distance of posterior a from the reference: mean shift in reference widths, width error,
    and the credible metric (shift of both interval edges over the reference 68% width)."""
    width = reference["q"][3] - reference["q"][1]
    shift = (a["mean"] - reference["mean"]) / reference["sd"]
    cm68 = (np.abs(a["q"][1] - reference["q"][1]) + np.abs(a["q"][3] - reference["q"][3])) / width
    cm95 = (np.abs(a["q"][0] - reference["q"][0]) + np.abs(a["q"][4] - reference["q"][4])) / width
    return dict(shift=shift, width=a["sd"] / reference["sd"] - 1.0, cm68=cm68, cm95=cm95)


def worst(comparison, names):
    out = {}
    for key in ("shift", "width", "cm68", "cm95"):
        value = np.abs(comparison[key])
        out[f"{key}_max"] = float(value.max())
        out[f"{key}_median"] = float(np.median(value))
        out[f"{key}_worst"] = names[int(value.argmax())]
    return out


def load_reference(path, run):
    """Published values of the run's parameters: {name: dict(centre, minus, plus, prior, source)}."""
    wanted = json.loads(Path(path).read_text())["parameters"]
    return {name: wanted[name] for name in run.names if name in wanted}


def against_reference(run, posterior, reference):
    """The posterior of one network against published values, parameter by parameter: the shift
    of its mean in published sigmas (the side it lies on), its width over the published one, and
    how many of its own widths it keeps from the nearest edge of the training box that is not
    also an edge of the prior (None when both are). Flags: a shift beyond 3, or less than 3
    widths from such an edge."""
    rows = []
    for k, name in enumerate(run.names):
        ref = reference.get(name)
        if ref is None:
            continue
        mean, sd = float(posterior["mean"][k]), float(posterior["sd"][k])
        minus, plus = float(ref.get("minus") or 0.0), float(ref.get("plus") or 0.0)
        side = plus if (mean >= ref["centre"] or minus == 0) else minus
        prior = ref.get("prior") or [None, None]
        gaps = []
        for edge, prior_edge, sign in ((run.lower[k], prior[0], 1.0), (run.upper[k], prior[1], -1.0)):
            at_prior = prior_edge is not None and abs(edge - prior_edge) <= 1e-9 * max(1.0, abs(edge))
            if not at_prior:
                gaps.append(sign * (mean - edge) / sd)
        row = dict(name=name, centre=float(ref["centre"]), minus=minus, plus=plus, source=ref.get("source", ""),
                   mean=mean, sd=sd, shift=(mean - ref["centre"]) / side if side else None,
                   width=sd / (0.5 * (minus + plus) if minus and plus else max(minus, plus)) if max(minus, plus) else None,
                   edge=min(gaps) if gaps else None, lower=float(run.lower[k]), upper=float(run.upper[k]), prior=prior)
        row["flag"] = ", ".join(text for text, on in (("shift", row["shift"] is not None and abs(row["shift"]) > 3),
                                                      ("box edge", row["edge"] is not None and row["edge"] < 3)) if on)
        rows.append(row)
    return rows


def fmt(value, spec=".2f", none="-"):
    return none if value is None or (isinstance(value, float) and not np.isfinite(value)) else format(value, spec)


def journal_note(batches):
    """The audit's share of the launcher's journal: none in a journal from before its batches
    were recorded there, whose run may have made audit calls all the same."""
    audit = sum(b["points"] for b in batches if b.get("kind") == "audit")
    return f"{audit} of the points are the audit's." if audit else "No audit call is in the journal."


def refused_value_of(directory, journals=()):
    """The value the run's refused points carry (launch.py's CLIENT_REFUSED_VALUE), or None. From
    the run itself (refused_value.json, which the launcher writes into it), so that the report
    run by hand without --journal knows it too; from the launcher's journals, when they agree
    on one, for a run from before that file existed."""
    try:
        return float(json.loads((Path(directory) / "refused_value.json").read_text())["value"])
    except (OSError, ValueError, KeyError, TypeError):      # no such file, or not one of the launcher's
        pass
    values = {json.loads(line).get("refused_value") for file in journals
              for line in Path(file).read_text().splitlines() if line.strip()} - {None}
    return values.pop() if len(values) == 1 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--journal", type=Path, nargs="*", default=[], help="logs/evaluations_<job>.jsonl of the run's jobs")
    parser.add_argument("--no-posteriors", action="store_true", help="skip the T = 1 sampling of the networks")
    parser.add_argument("--networks", type=int, nargs="*", help="networks to sample (default: all)")
    parser.add_argument("--last", type=int, help="sample only the last K of them: minutes instead of minutes per network")
    parser.add_argument("--corner", type=int, nargs="*", help="networks in the corner plot (default: before the boosts and the last)")
    parser.add_argument("--corner-parameters", type=int, default=6, help="the first N parameters are in the corner plot")
    parser.add_argument("--reference", type=Path, help="published values of the parameters (default: runs/<RUN>/reference_values.json)")
    parser.add_argument("--steps", type=int, default=20000, help="steps of each T = 1 chain")
    parser.add_argument("--walkers", type=int, default=256)
    parser.add_argument("--client", type=Path, default=Path(os.environ.get("CLIENT_REPO", Path(__file__).resolve().parents[2])))
    parser.add_argument("--linear-prior", nargs="+", default=[], metavar="NAME",
                        help="parameters p sampled with a prior flat in p (log10 of a quantity): sample with a prior flat in "
                             "10^p instead, and write to analysis/linear_prior_<names>/")
    args = parser.parse_args()

    started = time.monotonic()
    run = Run(args.run, args.client, linear_prior=args.linear_prior)
    out = args.out or analysis_folder(run.dir, run.linear_prior)
    out.mkdir(parents=True, exist_ok=True)
    iterations = sorted(run.models)
    if not iterations:  # a run still in its design: no network to report on yet
        print(f"{run.dir.name}: no trained network yet (the design is still being evaluated); nothing to report")
        return
    last = iterations[-1]
    boost_start = run.boost["start_iteration"] if run.boost else None
    run.refused_value = refused_value_of(run.dir, args.journal)
    report = dict(run=run.dir.name, parameters=run.names, temperature=run.temperature, best_true=run.best,
                  networks=iterations, boost=run.boost, linear_prior=list(run.linear_prior), iterations=[])
    print(f"{run.dir.name}: networks {iterations[0]} to {last}, {run.ndim} parameters, best true value {run.best:.2f}", flush=True)

    # ---- per iteration: sizes, held-out errors, audit, training, boost ---------------------
    for it in iterations:
        row = dict(iteration=it, training_points=int(len(run.data[it])) if it in run.data else None,
                   best_true_so_far=float(np.nanmax(run.data[it]["loglkl"])) if it in run.data else None,
                   held_out=held_out(run, it), training=training(run, it),
                   drift=(run.metric("convergence", it).get("metric_value")          # the chains' drift, when that is the
                          if run.metric("convergence", it).get("metric_name", "gaussian_posterior_drift")
                          == "gaussian_posterior_drift" else None),                 # run's metric (else it is not recorded)
                   sampling=run.metric("sampling", it), seconds=run.metric("iteration", it).get("total_iteration_time"))
        if it in run.audits:
            a = run.audits[it]
            first = a["rounds"][0] if a.get("rounds") else {}
            row["audit"] = dict(true_calls=a.get("true_calls"), added=a.get("added"), failed=a.get("failed"),
                                new_best=a.get("new_best"), clear=a.get("clear"), highest_before=first.get("highest_excess"),
                                highest_after=a.get("highest_excess"), maxima_above_before=first.get("distinct_above_tau"),
                                rounds=len(a.get("rounds", [])), seconds=a.get("seconds"))
        if it in run.boosts:
            b = run.boosts[it]
            # Points of the boost without a true value. Since 12 October 2026 the record counts those
            # the likelihood refused apart (`refused`, `zero_density`: zero weight in the corrected
            # mode weights) from those without any value (`failed`: the correction is withheld);
            # before, `failed` was all of them.
            row["boost"] = dict(asked=b.get("n"), failed=b.get("failed"),
                                refused=(b.get("refused") or 0) + (b.get("zero_density") or 0),
                                modes=len(b.get("modes", [])),
                                weights=[m.get("weight") for m in b.get("modes", [])],
                                points=[m.get("points") for m in b.get("modes", [])],
                                efficiency=[m.get("efficiency") for m in b.get("modes", [])],
                                selection=b.get("selection"), dropped_walkers=(b.get("search") or {}).get("dropped_walkers"))
        report["iterations"].append(row)
        ho = row["held_out"]
        main_set = ho.get("boost") or ho.get("batch") or {}
        print(f"  network {it:2d}: {row['training_points']} points; "
              + (f"held-out rms {main_set['rms']:.2f} on {main_set['n']} {'boost' if 'boost' in ho else 'batch'} points" if main_set.get("n") else "no acquisition yet")
              + (f"; audit {row['audit']['true_calls']} calls, clear {row['audit']['clear']}" if "audit" in row else ""), flush=True)

    # ---- the launcher's journal --------------------------------------------------------------
    batches = []
    for file in args.journal:
        batches += [json.loads(line) for line in Path(file).read_text().splitlines() if line.strip()]
    if batches:
        points = sum(b["points"] for b in batches)
        report["evaluations"] = dict(batches=len(batches), points=points, not_finite=sum(b["not_finite"] for b in batches),
                                     refused=sum(b.get("refused") or 0 for b in batches), refused_value=run.refused_value,
                                     seconds=sum(b["seconds"] for b in batches),
                                     core_seconds_per_point=float(np.median([b["seconds"] * b["ranks"] / b["points"] for b in batches if b["points"] >= b["ranks"]] or [np.nan])),
                                     journal=batches)

    # ---- posteriors at T = 1 -----------------------------------------------------------------
    posteriors = {}
    if not args.no_posteriors:
        import tensorflow as tf

        rng = np.random.default_rng(20261007)
        wanted = [it for it in (args.networks if args.networks is not None else iterations) if it in run.models and it in run.summaries]
        if args.last:
            wanted = wanted[-args.last:]
        for it in wanted:
            tf.keras.utils.set_random_seed(1000 + it)
            t0 = time.monotonic()
            chain, acceptance = sample_posterior(run, it, args.steps, args.walkers, rng)
            flat = chain.reshape(-1, run.ndim)
            posteriors[it] = dict(summarise(flat), chain=chain, acceptance=acceptance)
            from sampling.provenance import save_samples, sampler_identity
            sampling = dict(sampler_identity(run.surrogate(it, region=True).region), **run.mixtures.get(it, {}))
            save_samples(out / f"posterior_it_{it}.npz", flat, run.names, run.target_identity(it),
                         sampling=sampling, acceptance=acceptance)
            print(f"  network {it:2d} sampled at T = 1: {len(flat)} samples, acceptance {acceptance:.2f} ({time.monotonic() - t0:.0f} s)", flush=True)
        if posteriors:
            ref_it = max(posteriors)
            ref = posteriors[ref_it]
            half = ref["chain"].shape[1] // 2
            a, b = (summarise(ref["chain"][:, :half].reshape(-1, run.ndim)), summarise(ref["chain"][:, half:].reshape(-1, run.ndim)))
            report["posterior"] = dict(reference=ref_it, steps=args.steps, walkers=args.walkers,
                                       noise_floor=worst(compare(a, b), run.names), networks=[])
            previous = None
            for it in sorted(posteriors):
                p = posteriors[it]
                entry = dict(iteration=it, acceptance=p["acceptance"], mean=p["mean"].tolist(), sd=p["sd"].tolist(),
                             against_reference=worst(compare(p, ref), run.names),
                             shift=compare(p, ref)["shift"].tolist(), width=compare(p, ref)["width"].tolist())
                if previous is not None:
                    entry["against_previous"] = worst(compare(p, posteriors[previous]), run.names)
                if it in run.mixtures:                         # a region of several ellipsoids: the network's weight in each
                    entry["cells"] = [{key: cell[key] for key in ("weight", "log_evidence", "log_evidence_spread", "efficiency", "acceptance")}
                                      for cell in run.mixtures[it]["cells"]]
                    entry["unreliable_cells"] = run.mixtures[it]["unreliable_cells"]
                report["posterior"]["networks"].append(entry)
                previous = it
            reference_file = args.reference
            if reference_file is None:                         # the run folder's own file, when there is one
                name = re.sub(r"^\d{8}_\d{6}_", "", run.dir.name)
                candidate = Path(os.environ.get("CLIENT_HPC_DIR") or Path(__file__).resolve().parent.parent) / "runs" / name / "reference_values.json"
                reference_file = candidate if candidate.exists() else None
            if reference_file is not None:
                run.reference = load_reference(reference_file, run)
                report["posterior"]["against_published"] = dict(file=str(reference_file), network=ref_it,
                                                                rows=against_reference(run, ref, run.reference))

    # ---- files ---------------------------------------------------------------------------------
    report["seconds"] = time.monotonic() - started
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    write_markdown(report, run, out, boost_start)
    try:
        plots(report, run, posteriors, out, boost_start, args.corner, args.corner_parameters)
    except Exception as error:                                   # the numbers are written whatever the plots do
        print(f"plots not written: {error!r}", flush=True)
    print(f"written: {out}/report.md, report.json and figures ({report['seconds']:.0f} s)")


def linear_prior_line(names):
    """The report's statement of a prior changed with --linear-prior."""
    flat = " and ".join(f"10^{name}" for name in names)
    added = " + ".join(f"ln(10) {name}" for name in names)
    return (f"The posteriors have a prior flat in {flat} instead of the run's (--linear-prior): {added} is added to "
            "every network's log posterior where it is sampled. The held-out errors are those of the networks themselves.")


def write_markdown(report, run, out, boost_start):
    n, T = run.ndim, run.temperature
    lines = [f"# {report['run']}: convergence and errors", "",
             f"{len(report['networks'])} networks ({report['networks'][0]} to {report['networks'][-1]}), {n} parameters, "
             f"acquisition temperature {T:g}, best true value found {report['best_true']:.2f}."
             + (f" Boosts from iteration {boost_start}." if boost_start is not None else ""), "",
             *([linear_prior_line(report["linear_prior"]), ""] if report.get("linear_prior") else []),
             "All errors are network minus truth, in log-likelihood units.", "",
             "## Held-out errors: network N at the points it acquired", "",
             f"The tempered batch lies about T n / 2 = {0.5 * T * n:.0f} below the peak; boost points are posterior points (T = 1).", "",
             "For boost points the last two columns are the share of the points that carry the true posterior under the "
             "weights exp(truth minus network), and the Kullback-Leibler divergence from the network's posterior to the true one.", "",
             "| network | training points | acquired as | points | median depth below best | rms | bias | sd | largest | effective share | KL |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in report["iterations"]:
        for label in ("batch", "injected", "boost"):
            e = row["held_out"].get(label)
            if e and e.get("n"):
                posterior = (f"{e['effective_fraction']:.2f} | {e['kl']:.2f}" if label == "boost" and "kl" in e else "- | -")
                lines.append(f"| {row['iteration']} | {row['training_points']} | {label} | {e['n']} | {e['median_depth']:.1f} | "
                             f"{e['rms']:.2f} | {e['bias']:+.2f} | {e['sd']:.2f} | {e['largest']:.1f} | {posterior} |")
    lines += ["", f"The same for the tempered batch by depth below the best true value (bands of n = {n} and T n / 2, T n):", "",
              f"| network | 0 to {n} | {n} to {0.5 * T * n:.0f} | {0.5 * T * n:.0f} to {T * n:.0f} | beyond {T * n:.0f} |", "|---|---|---|---|---|"]
    for row in report["iterations"]:
        e = row["held_out"].get("batch")
        if e and e.get("n"):
            lines.append(f"| {row['iteration']} | " + " | ".join(
                (f"{b['rms']:.2f} ({b['n']})" if b.get("n") else "-") for b in e["bands"]) + " |")
    lines += ["", "Entries are rms (number of points)."]
    if getattr(run, "refused_value", None) is not None:
        lines += ["", "## Points the likelihood refused", "",
                  f"They carry the value {run.refused_value:g} in the training set and are left out of the errors above. "
                  "A falling share says the network has learned where they are.", "",
                  "| network | batch points | of them refused | injected, refused | audit points, refused |", "|---|---|---|---|---|"]
        for row in report["iterations"]:
            h = row["held_out"]
            if h:
                b = h.get("batch") or h.get("boost") or {}
                total = (b.get("n") or 0) + (b.get("refused") or 0)
                lines.append(f"| {row['iteration']} | {total} | {b.get('refused', 0)} | {(h.get('injected') or {}).get('refused', 0)} | "
                             f"{(h.get('audit') or {}).get('refused', 0)} |")
    lines += ["", "## Audit", "",
              "| network | true calls | points added | failed | highest maximum above best, before | after | maxima above threshold before | clear | better than best |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in report["iterations"]:
        a = row.get("audit")
        if a:
            lines.append(f"| {row['iteration']} | {a['true_calls']} | {a['added']} | {a['failed']} | {fmt(a['highest_before'], '+.1f')} | "
                         f"{fmt(a['highest_after'], '+.1f')} | {fmt(a['maxima_above_before'], 'd')} | {a['clear']} | {a['new_best']} |")
    lines += ["", "## Training and loop", "",
              "| network | epochs | lowest validation loss (epoch) | learning-rate reductions | training s | sampling acceptance | drift | iteration s |",
              "|---|---|---|---|---|---|---|---|"]
    for row in report["iterations"]:
        t = row["training"]
        if t:
            lines.append(f"| {row['iteration']} | {t['epochs']} | {t['val_loss_min']:.3g} ({t['epoch_of_min']}) | {t.get('learning_rate_reductions', '-')} | "
                         f"{fmt(t.get('seconds'), '.0f')} | {fmt(row['sampling'].get('acceptance_rate'), '.2f')} | {fmt(row.get('drift'), '.3g')} | {fmt(row.get('seconds'), '.0f')} |")
    if any("boost" in row for row in report["iterations"]):
        lines += ["", "## Boosts", "", "| iteration | modes | weights | points per mode | refused | without a value | proposal efficiency |",
                  "|---|---|---|---|---|---|---|"]
        for row in report["iterations"]:
            b = row.get("boost")
            if b:
                lines.append(f"| {row['iteration']} | {b['modes']} | {', '.join(fmt(100 * w, '.1f') + '%' for w in b['weights'])} | "
                             f"{', '.join(str(p) for p in b['points'])} | {b['refused']} | {b['failed']} | "
                             f"{', '.join(fmt(e, '.2f') for e in b['efficiency'])} |")
    if "posterior" in report:
        p = report["posterior"]
        f = p["noise_floor"]
        lines += ["", "## Posterior at T = 1, network by network", "",
                  f"Each network sampled with {p['walkers']} walkers and {p['steps']} steps. Distances are from network {p['reference']}: "
                  "the largest mean shift in units of its width, the largest width error, and the credible metric "
                  "(shift of both interval edges over its 68% width). Agreement between networks is convergence, not accuracy.", "",
                  "| network | largest mean shift (sigma) | parameter | largest width error | CM68 max | CM95 max | CM68 against the previous network |",
                  "|---|---|---|---|---|---|---|"]
        for e in p["networks"]:
            r = e["against_reference"]
            lines.append(f"| {e['iteration']} | {r['shift_max']:.3f} | {r['shift_worst']} | {r['width_max']:.3f} | {r['cm68_max']:.3f} | "
                         f"{r['cm95_max']:.3f} | {fmt(e.get('against_previous', {}).get('cm68_max'), '.3f')} |")
        mixed = [e for e in p["networks"] if e.get("cells")]
        if mixed:
            lines += ["", "The regions of these networks have several ellipsoids. Each cell (the points nearer to one ellipsoid than "
                          "to any other) was sampled by itself and weighted by the network's own evidence in it; the weights are the "
                          "network's, not corrected to the true likelihood (the boost's record has those).", "",
                      "| network | weight of each cell | efficiency of the estimate | not reliable |", "|---|---|---|---|"]
            lines += [f"| {e['iteration']} | {', '.join(fmt(100 * c['weight'], '.2f') + '%' for c in e['cells'])} | "
                      f"{', '.join(fmt(c['efficiency'], '.2f') for c in e['cells'])} | "
                      f"{', '.join(f'cell {k}' for k in e['unreliable_cells']) or '-'} |" for e in mixed]
        lines += ["", f"Noise floor (two halves of network {p['reference']}'s own chain): mean shift {f['shift_max']:.3f}, "
                      f"width {f['width_max']:.3f}, CM68 {f['cm68_max']:.3f}, CM95 {f['cm95_max']:.3f}.", "",
                  f"Posterior of network {p['reference']}:", "", "| parameter | mean | standard deviation |", "|---|---|---|"]
        final = p["networks"][-1]
        lines += [f"| {name} | {m:.6g} | {s:.3g} |" for name, m, s in zip(run.names, final["mean"], final["sd"])]
        if "against_published" in p:
            a = p["against_published"]
            flagged = [r["name"] for r in a["rows"] if r["flag"]]
            lines += ["", f"## Posterior of network {a['network']} against published values", "",
                      f"Published values from {a['file']}. Shift: the posterior mean minus the published centre, in published sigmas "
                      "of that side. Width: posterior standard deviation over the published one. Box edge: the distance of the mean "
                      "from the nearest edge of the training box that is not also a prior edge, in posterior standard deviations "
                      "(- when both edges are prior edges). Flagged: a shift beyond 3, or less than 3 from such an edge. "
                      "A network before its boosts and its true check is not a result; this is a check that nothing is far off.", "",
                      (f"Flagged: {', '.join(flagged)}." if flagged else "Nothing is flagged."), "",
                      "| parameter | published | posterior | shift | width | box edge | flag |", "|---|---|---|---|---|---|---|"]
            for r in a["rows"]:
                published = (f"{r['centre']:.6g} ± {r['plus']:.3g}" if r["minus"] == r["plus"] else
                             f"{r['centre']:.6g} +{r['plus']:.3g} −{r['minus']:.3g}")
                lines.append(f"| {r['name']} | {published} | {r['mean']:.6g} ± {r['sd']:.3g} | {fmt(r['shift'], '+.2f')} | "
                             f"{fmt(r['width'], '.2f')} | {fmt(r['edge'], '.1f')} | {r['flag']} |")
    if "evaluations" in report:
        e = report["evaluations"]
        lines += ["", "## True evaluations (the launcher's journal)", "",
                  f"{e['points']} points in {e['batches']} batches, {e['not_finite']} not finite"
                  + (f", {e['refused']} refused (value {e['refused_value']:g})" if e.get("refused_value") is not None else "")
                  + f"; {e['seconds'] / 3600:.2f} h of wall time in evaluation; "
                  f"{e['core_seconds_per_point']:.1f} core-seconds per point (median over batches). " + journal_note(e["journal"])]
    lines += ["", "Figures: errors.png, convergence.png, posterior.png, training.png, corner.png"
              + (", marginals.png." if "against_published" in report.get("posterior", {}) else ".")]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def plots(report, run, posteriors, out, boost_start, corner, corner_parameters=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.ticker import MaxNLocator

    rows = report["iterations"]

    def mark(ax):
        """Whole network numbers on the axis, and a line where the boosts begin."""
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if boost_start is not None:
            ax.axvline(boost_start + 0.5, color="0.6", ls=":", lw=1)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for label, colour in (("batch", "C0"), ("injected", "C2"), ("boost", "C3")):
        xy = [(r["iteration"], r["held_out"][label]["rms"]) for r in rows if r["held_out"].get(label, {}).get("n")]
        if xy:
            ax[0].plot(*zip(*xy), "o-", color=colour, label=f"{label} points")
    core = [(r["iteration"], r["held_out"]["batch"]["bands"][0]["rms"]) for r in rows
            if r["held_out"].get("batch", {}).get("n") and r["held_out"]["batch"]["bands"][0].get("n", 0) >= 5]
    if core:
        ax[0].plot(*zip(*core), "s--", color="C1", label=f"batch points within {run.ndim} of the best")
    ax[0].set(yscale="log", xlabel="network", ylabel="rms error on the points it acquired", title="Held-out error of every network")
    ax[0].legend(fontsize=8); mark(ax[0])
    depth = [(r["iteration"], r["held_out"]["batch"]["median_depth"]) for r in rows if r["held_out"].get("batch", {}).get("n")]
    if depth:
        ax[1].plot(*zip(*depth), "o-", label="tempered batch, median")
        ax[1].axhline(0.5 * run.temperature * run.ndim, color="0.5", ls="--", lw=1, label="T n / 2")
    best = [(r["iteration"], report["best_true"] - r["best_true_so_far"]) for r in rows if r["best_true_so_far"] is not None]
    ax[1].plot(*zip(*best), "s-", color="C1", label="best training point so far")
    ax[1].set(yscale="symlog", xlabel="network", ylabel="log-units below the best true value of the run", title="Where the points are")
    ax[1].legend(fontsize=8); mark(ax[1])
    fig.tight_layout(); fig.savefig(out / "errors.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot([r["iteration"] for r in rows if r["training"]], [r["training"]["epochs"] for r in rows if r["training"]], "o-")
    ax[0].set(xlabel="network", ylabel="epochs", title="Training length"); mark(ax[0])
    ax[1].plot([r["iteration"] for r in rows if r["training"]], [r["training"]["val_loss_min"] for r in rows if r["training"]], "o-")
    ax[1].set(yscale="log", xlabel="network", ylabel="lowest validation loss", title="Validation loss"); mark(ax[1])
    audits = [(r["iteration"], r["audit"]["highest_before"], r["audit"]["highest_after"], r["audit"]["true_calls"]) for r in rows if r.get("audit")]
    if audits:
        it, before, after, calls = zip(*audits)
        clean = lambda v: [np.nan if x is None else x for x in v]
        ax[2].plot(it, clean(before), "o-", label="highest maximum above best, before the audit")
        ax[2].plot(it, clean(after), "s-", label="after")
        ax[2].set(yscale="symlog", xlabel="network", ylabel="log-units above the best true value", title="Audit")
        twin = ax[2].twinx(); twin.bar(it, calls, alpha=0.2, color="0.3"); twin.set_ylabel("true calls")
        ax[2].legend(fontsize=8); mark(ax[2])
    fig.tight_layout(); fig.savefig(out / "training.png", dpi=140); plt.close(fig)

    if "posterior" not in report:
        return
    p = report["posterior"]
    its = [e["iteration"] for e in p["networks"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for key, label in (("shift_max", "largest mean shift (sigma)"), ("width_max", "largest width error"),
                       ("cm68_max", "credible metric, 68%"), ("cm95_max", "credible metric, 95%")):
        line, = ax[0].plot(its[:-1], [e["against_reference"][key] for e in p["networks"][:-1]], "o-", label=label)
        ax[0].axhline(p["noise_floor"][key], color=line.get_color(), ls=":", lw=1)
    ax[0].set(yscale="log", xlabel="network", ylabel=f"distance from network {p['reference']}",
              title=f"T = 1 posterior against network {p['reference']} (dotted: chain noise)")
    ax[0].legend(fontsize=8); mark(ax[0])
    drift = [(r["iteration"], r["drift"]) for r in rows if r.get("drift")]
    if drift:
        ax[1].plot(*zip(*drift), "o-", label="loop's drift metric (T = 7 chain, against the previous)")
    step = [(e["iteration"], e["against_previous"]["cm68_max"]) for e in p["networks"] if "against_previous" in e]
    if step:
        ax[1].plot(*zip(*step), "s-", label="T = 1 credible metric against the previous network")
    ax[1].set(yscale="log", xlabel="network", title="Change from one network to the next"); ax[1].legend(fontsize=8); mark(ax[1])
    fig.tight_layout(); fig.savefig(out / "convergence.png", dpi=140); plt.close(fig)

    shown = list(range(min(run.ndim, 16)))
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(len(shown) / cols)), cols, figsize=(4 * cols, 2.6 * int(np.ceil(len(shown) / cols))), squeeze=False)
    for k, a in zip(shown, axes.ravel()):
        shift = np.array([e["shift"][k] for e in p["networks"]]); width = np.array([e["width"][k] for e in p["networks"]])
        a.errorbar(its, shift, yerr=1.0 + width, fmt="o-", ms=3, lw=1, capsize=2)
        a.axhline(0, color="0.5", lw=0.8); a.set_title(run.names[k], fontsize=9); mark(a)
    for a in axes.ravel()[len(shown):]:
        a.axis("off")
    fig.supxlabel("network"); fig.supylabel(f"mean and width, in units of network {p['reference']}'s width, relative to its mean")
    fig.tight_layout(); fig.savefig(out / "posterior.png", dpi=130); plt.close(fig)

    from getdist import MCSamples, plots as gplots

    chosen = corner if corner else sorted({it for it in ([boost_start] if boost_start in posteriors else []) + [its[len(its) // 2], its[-1]]})
    chosen = [it for it in chosen if it in posteriors]
    cosmo = [k for k in range(run.ndim)][:corner_parameters]
    labels = run.metadata.param_labels if hasattr(run.metadata, "param_labels") else run.names
    samples = [MCSamples(samples=posteriors[it]["chain"].reshape(-1, run.ndim)[:, cosmo], names=[run.names[k] for k in cosmo],
                         labels=[str(labels[k]).strip("$") for k in cosmo], label=f"network {it}",
                         ranges={run.names[k]: (run.lower[k], run.upper[k]) for k in cosmo},
                         # as the axion paper's figures (31 October 2026): 0.4 standard deviations round off the peaks of
                         # a multimodal parameter, whose standard deviation is large; 1D automatic, 2D 0.2
                         settings=dict(smooth_scale_2D=0.2, smooth_scale_1D=-1)) for it in chosen]
    reference = getattr(run, "reference", {})
    g = gplots.get_subplot_plotter(width_inch=max(9, 1.5 * len(cosmo)))
    g.triangle_plot(samples, filled=True,                     # dashed lines: the published centres
                    markers={run.names[k]: reference[run.names[k]]["centre"] for k in cosmo if run.names[k] in reference})
    g.export(str(out / "corner.png"))

    published = p.get("against_published")
    if published:
        # Every parameter's marginal of the last sampled network, with the published 68% band,
        # the published centre and the edges of the training box (red: not a prior edge).
        rows_ = published["rows"]
        flat = posteriors[published["network"]]["chain"].reshape(-1, run.ndim)
        cols = 7
        fig, axes = plt.subplots(int(np.ceil(len(rows_) / cols)), cols, figsize=(2.6 * cols, 2.0 * int(np.ceil(len(rows_) / cols))), squeeze=False)
        for r, a in zip(rows_, axes.ravel()):
            k = run.names.index(r["name"])
            lo = min(r["mean"] - 4 * r["sd"], r["centre"] - 3 * max(r["minus"], r["plus"]))
            hi = max(r["mean"] + 4 * r["sd"], r["centre"] + 3 * max(r["minus"], r["plus"]))
            lo, hi = max(lo, r["lower"]), min(hi, r["upper"])
            pad = 0.04 * (hi - lo)
            a.axvspan(r["centre"] - r["minus"], r["centre"] + r["plus"], color="0.8", lw=0)
            a.axvline(r["centre"], color="0.4", lw=0.8, ls="--")
            a.hist(flat[:, k], bins=60, range=(lo, hi), histtype="step", color="C0", density=True, lw=1.2)
            for edge, prior_edge in ((r["lower"], r["prior"][0]), (r["upper"], r["prior"][1])):
                at_prior = prior_edge is not None and abs(edge - prior_edge) <= 1e-9 * max(1.0, abs(edge))
                if lo - pad <= edge <= hi + pad:
                    a.axvline(edge, color="0.3" if at_prior else "C3", lw=1.5)
            a.set_xlim(lo - pad, hi + pad); a.set_yticks([]); a.tick_params(labelsize=6)
            a.set_title(f"{r['name']}  {fmt(r['shift'], '+.1f')}σ" + ("  !" if r["flag"] else ""), fontsize=7,
                        color="C3" if r["flag"] else "k")
        for a in axes.ravel()[len(rows_):]:
            a.axis("off")
        fig.suptitle(f"network {published['network']} at T = 1 (blue), published 68% band and centre (grey), box edges "
                     "(red; dark grey where the prior ends too); title: shift of the mean in published sigmas", fontsize=9)
        fig.tight_layout(); fig.savefig(out / "marginals.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
