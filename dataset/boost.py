"""acquisition.boost: an iteration whose new training points are posterior samples of the network.

The loop's batches are drawn at the acquisition temperature T, some T ndim / 2 log-units under
the peak; the posterior itself lies ndim / 2 under it, and a batch puts a few points in a
thousand there. The network's level in the core of each mode is then set by extrapolation, and
the weights of separated modes, which depend on the difference of that level between them
exponentially, come out anywhere (SINU, 16 parameters: 12.7%, 35.7% and 38.1% for the strong
mode from three good networks; 24.8% measured).

A boost iteration replaces the tempered batch by `n` true evaluations at T = 1 samples of the
network: the modes are found (sampling/modes.py: groups of local maxima that a T = 1 chain
cannot travel between), each gets its own T = 1 chain, and each contributes the same number of
points, n / modes, whatever its weight. With one mode it is n posterior samples. The points
join the training set as a batch does; the next network is trained on them.

The true values also measure the network where it matters. With D = true minus network at a
mode's points, the true evidence of the mode is the network's times the mean of exp(D), so the
report gives the weights of the modes as the network has them and as corrected, and the
effective number of points behind each mean: a correction resting on a handful of points (the
network is off by more than a log-unit or so on its own samples) is not to be trusted, and the
boost was premature.

Opt-in `balance_floor` (mode_shares): a mode lighter than the floor gets its equal share times
its weight over the floor, so that a small group of walkers, which the posterior pilot cannot
check, does not take as many points as the posterior. Without it every mode has the same share.

The description above is the default unfiltered path. Opt-in density filtering
uses a balanced posterior reference and may accept fewer or unequal numbers per
mode; those selected rows do not support the unweighted evidence correction.
See docs/posterior_density_boost.md.

Files: RUN/boost/boost_it_N.csv (parameters, loglkl, the network's value, the mode) and
boost_it_N.json (the report).
"""

import json
import time
from pathlib import Path

import numpy as np

from run.storage import atomic_path
from sampling.modes import posterior_modes, split_evenly


def _write(path, record):
    """A record of the boosts, whole or not at all. start.json is written between two phases
    and read at the next start: torn by a job's end, it stopped every later one with a JSON
    error. (The text is what json.dumps gave before: a report may hold a NaN.)"""
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(record, indent=1))


def mode_shares(weights, floor):
    """The share of a boost each mode gets, the largest 1: min(weight / floor, 1), over the largest
    of those. Modes at or above the floor share equally, as without it; a lighter one gets the
    fraction weight / floor of an equal share. None (no floor): None, every mode the same."""
    if floor is None:
        return None
    units = np.minimum(np.asarray(weights, float) / float(floor), 1.0)
    if not len(units) or not np.all(np.isfinite(units)) or units.max() <= 0:
        raise ValueError("balance_floor needs finite mode weights with a positive largest")
    return units / units.max()


def split_by_shares(n, shares):
    """n as whole numbers in proportion to `shares` (largest remainders), at least one each."""
    shares = np.asarray(shares, float)
    exact = n * shares / shares.sum()
    counts = np.maximum(np.floor(exact).astype(int), 1)
    for k in np.argsort(-(exact - np.floor(exact)), kind="stable"):
        if counts.sum() >= n:
            break
        counts[k] += 1
    while counts.sum() > n and counts.max() > 1:
        counts[int(np.argmax(counts))] -= 1
    return counts.tolist()


def draw_boost(predict, lower, upper, tempered, temperature, config, seed=None, log=print, dataset=None):
    """Draw up to config.n points; optionally apply posterior density filtering.

    The default samples equally from each mode. Filtered counts can differ and
    underfill. Returns (points, draw); finish_boost receives the true values.
    """
    rng = np.random.default_rng(seed)
    modes, info = posterior_modes(
        predict, lower, upper, tempered, temperature, barrier=config.barrier,
        min_share=config.min_share, walkers=config.walkers, cool_steps=config.cool_steps,
        steps=config.steps, components=config.components, seed=seed, log=log,
        **({} if getattr(config, "mixing", None) is None else
           {"mixing": config.mixing, "candidate_barrier": config.candidate_barrier}),
        **({} if getattr(config, "mode_depth", None) is None else {"mode_depth": config.mode_depth}))
    points, predicted, label = [], [], []
    selection = None
    floor = getattr(config, "balance_floor", None)
    shares = None if floor is None else mode_shares([m["weight"] for m in modes], floor)
    if shares is not None and np.any(shares < 1):
        log(f"   balance floor {config.balance_floor:g}: shares of the modes "
            + ", ".join(f"{s:.3g}" for s in shares) + " (a mode lighter than the floor gets its weight over it)")
    if config.density_filter:
        if dataset is None:
            raise ValueError("filtered boost requires the current training dataset")
        from dataset.posterior_boost import select_posterior_boost
        selected, selection = select_posterior_boost(
            dataset, [m["x"] for m in modes], config.n,
            seed=seed, pool_factor=config.pool_factor,
            **({} if shares is None else {"shares": shares}))
        # Keep the mode membership of the sampled states, not a new classifier.
        keys = {row.tobytes() for row in selected}
        for k, mode in enumerate(modes):
            _, first = np.unique(mode["x"].astype(np.float32), axis=0, return_index=True)
            rows = []
            for i in first:
                key = mode["x"][i].astype(np.float32).tobytes()
                if key in keys:
                    rows.append(i)
                    keys.remove(key)
            points.append(mode["x"][rows])
            predicted.append(mode["f"][rows])
            label.append(np.full(len(rows), k))
        if not len(selected):
            raise RuntimeError("posterior density filter accepted no points; no unfiltered top-up")
        if keys:
            raise RuntimeError("selected boost coordinates lost their sampled mode membership")
        if "pool_factor_used" in selection:
            log(f"   density filter: {selection['distinct_states']} distinct states, too few for a pool of "
                f"{config.pool_factor} x {config.n} and a reference sample: pool factor {selection['pool_factor_used']}")
        log(f"   density filter: {len(selected)}/{config.n} accepted; no unweighted evidence correction")
    counts = split_evenly(config.n, len(modes)) if shares is None else split_by_shares(config.n, shares)
    for k, (mode, count) in enumerate(zip(modes, counts)):
        if config.density_filter:
            break
        # Sample chain rows, not distinct coordinates: rejected proposals carry mass.
        rows = rng.choice(len(mode["x"]), size=count, replace=count > len(mode["x"]))
        points.append(mode["x"][rows])
        predicted.append(mode["f"][rows])
        label.append(np.full(len(rows), k))
    summary = [dict(weight=m["weight"], log_evidence=m["log_evidence"],
                    log_evidence_spread=m["log_evidence_spread"], efficiency=m["efficiency"],
                    walkers=m["walkers"], peak=np.asarray(m["peak"], float).tolist(),
                    peak_value=m["peak_value"], acceptance=m["acceptance"],
                    mean=m["x"].mean(axis=0).tolist(), std=m["x"].std(axis=0).tolist()) for m in modes]
    if shares is not None:
        for entry, share in zip(summary, shares):
            entry["share"] = float(share)
    points = np.concatenate(points).astype(np.float32)
    log(f"   {len(points)} boost points: {', '.join(str(int(np.sum(np.concatenate(label) == k))) for k in range(len(modes)))} per mode")
    return points, dict(points=points, predicted=np.concatenate(predicted).astype(float),
                        label=np.concatenate(label), modes=summary, info=info,
                        selection=selection, evidence_correction_valid=not config.density_filter)


def boost_report(draw, truth, ignore=None, zero_density=None):
    """Per mode: the network's error on its own posterior samples and the weights before and
    after the correction by the true values (aligned with draw["points"]).

    ignore : the finite value a launcher gives a point the likelihood refused
        (hpc/refused.py); such points are left out of the pooled error
        (utils/held_out.py) and counted there.
    zero_density : the -inf returns that are refusals of the likelihood (finish_boost marks
        every one, as hpc/analysis/true_check.py counts them).

    A refused point, by either mark, is a point where the true posterior is zero: it has
    weight zero and stays in the denominator of its mode's mean of exp(truth - network), as
    in hpc/analysis/true_check.py, and is counted (`refused`, `zero_density`). A point without any
    value (NaN, +inf, a -inf that is not marked) is another matter: what the mean would have
    been is not known, and the corrected weights are withheld for every mode (`failed`)."""
    from scipy.special import logsumexp

    truth = np.asarray(truth, float).ravel()
    modes = []
    zero_density = np.zeros(len(truth), bool) if zero_density is None else np.asarray(zero_density, bool)
    if zero_density.shape != truth.shape or np.any(zero_density & ~np.isneginf(truth)):
        raise ValueError("zero_density must mark only explicitly confirmed -inf returns")
    refused = np.zeros(len(truth), bool) if ignore is None else truth == ignore
    unresolved = ~np.isfinite(truth) & ~zero_density
    missing_mode = any(not np.any(draw["label"] == k) for k in range(len(draw["modes"])))
    correction_valid = bool(not missing_mode and draw.get("evidence_correction_valid", True) and not unresolved.any())
    reason = ("selection is not a posterior draw" if not draw.get("evidence_correction_valid", True)
              else "objective returns without a value (NaN or +inf)" if unresolved.any() else
                "one or more modes have no draws" if missing_mode else None)
    for k, mode in enumerate(draw["modes"]):
        rows = draw["label"] == k
        ok = rows & np.isfinite(truth) & ~refused
        d = truth[ok] - draw["predicted"][ok]
        record = dict(points=int(rows.sum()), failed=int(np.count_nonzero(rows & unresolved)),
                      refused=int(np.count_nonzero(rows & refused)),
                      zero_density=int(np.count_nonzero(rows & zero_density)), weight=mode["weight"])
        if len(d):
            w = np.exp(d - logsumexp(d))
            record.update(error_mean=float(-d.mean()), error_sd=float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                          best_true=float(truth[ok].max()))
        if correction_valid and rows.any():
            # Refused draws (the refused value, a marked -inf) have weight zero but stay in the denominator.
            record.update(log_mean_exp=float(logsumexp(d) - np.log(rows.sum())),
                          effective_points=float(1.0 / np.sum(w ** 2)) if len(d) else 0.0)
        modes.append(record)
    known = [m for m in modes if "log_mean_exp" in m]
    if len(known) == len(modes):
        log_w = np.array([np.log(m["weight"]) + m["log_mean_exp"] for m in modes])
        if not np.isfinite(logsumexp(log_w)):
            correction_valid, reason = False, "all sampled importance weights are zero"
        for m, value in zip(modes if correction_valid else [], np.exp(log_w - logsumexp(log_w)) if correction_valid else []):
            m["corrected_weight"] = float(value)
    # The network against the true values over all the points, whatever the selection: an
    # error of the network where its posterior is, not a correction of anything.
    from utils.held_out import held_out_error

    return dict(modes=modes, n=int(len(truth)), failed=int(np.count_nonzero(unresolved)), refused=int(refused.sum()),
                zero_density=int(zero_density.sum()),
                evidence_correction_valid=correction_valid, correction_withheld_reason=reason, selection=draw.get("selection"),
                error=held_out_error(draw["predicted"], truth, ignore=ignore))


def mode_line(k, m, report):
    """A mode's line of the boost's log, from its record."""
    without = m["failed"] + m.get("refused", 0) + m["zero_density"]
    refused = m.get("refused", 0) + m["zero_density"]
    if not report["evidence_correction_valid"]:
        return (f"mode {k + 1}: {m['points'] - without} usable points; "
                f"posterior weight correction withheld ({report['correction_withheld_reason']})")
    if "log_mean_exp" not in m:
        return f"mode {k + 1}: no finite true value among its {m['points']} points"
    return (f"mode {k + 1}: {m['points'] - without} true values"
            + (f" ({refused} refused: zero weight, counted in the mean)" if refused else "")
            + f"; network minus truth {m.get('error_mean', float('nan')):+.2f} "
            f"(sd {m.get('error_sd', float('nan')):.2f}), {m['effective_points']:.0f} effective points; weight {100 * m['weight']:.1f}% "
            + (f"-> {100 * m['corrected_weight']:.1f}% with the true values" if "corrected_weight" in m else ""))


# A launcher may round a batch to single precision and keep it inside the box as single
# precision holds it (hpc/launch.py): a point within the last digit of an edge is then
# evaluated one step inside. Farther than this from every drawn point, in units in the last
# place of single precision, an evaluated point is not one of the boost's.
MOVED_BY_A_LAUNCHER = 2


def finish_boost(directory, iteration, names, draw, inputs, targets, log=print, ignore=None):
    """Report and files of a boost, from the evaluated (inputs, targets) in whatever order.

    An evaluated point is the drawn point with the same single-precision coordinates. One
    that a launcher moved by the last digit to keep it inside the box (MOVED_BY_A_LAUNCHER)
    is the nearest drawn point not yet matched, and the boost's file records it where it was
    evaluated, which is where the training set has it. Any other point stops the run."""
    import pandas as pd

    from collections import defaultdict, deque
    drawn = np.ascontiguousarray(draw["points"], np.float32)
    position = defaultdict(deque)
    for i, point in enumerate(drawn):
        position[point.tobytes()].append(i)
    truth = np.full(len(drawn), np.nan)
    recorded = np.array(draw["points"])              # a copy, of the draw's own precision: the file is as it was
    matched, moved = np.zeros(len(drawn), bool), []
    for point, value in zip(np.ascontiguousarray(inputs, np.float32), np.asarray(targets, float).ravel()):
        key = point.tobytes()
        if position[key]:
            row = position[key].popleft()
            truth[row], matched[row] = value, True
        else:
            moved.append((point, value))
    for point, value in moved:
        free = np.flatnonzero(~matched)
        if len(free):
            step = np.spacing(np.maximum(np.abs(drawn[free]), np.abs(point))).astype(float)
            apart = np.max(np.abs(drawn[free].astype(float) - point.astype(float)) / step, axis=1)
            nearest = int(np.argmin(apart))
        if not len(free) or apart[nearest] > MOVED_BY_A_LAUNCHER:
            raise ValueError("unexpected or extra boost evaluation coordinate: an evaluated point is not a drawn one, "
                             f"nor within {MOVED_BY_A_LAUNCHER} units in the last place of single precision of one")
        row = free[nearest]
        truth[row], matched[row], recorded[row] = value, True, point
    if moved:
        log(f"   {len(moved)} of the {len(drawn)} boost points were evaluated one step of single precision from where they "
            "were drawn, inside an edge of the box (the launcher's rounding), and are recorded as evaluated")
    # A -inf return is the likelihood's refusal of the point (a logged error of the Boltzmann
    # code, hpc/refused.py; the prior outside its support), as the true check counts it.
    report = boost_report(draw, truth, ignore=ignore, zero_density=np.isneginf(truth))
    for k, m in enumerate(report["modes"]):
        log("   " + mode_line(k, m, report))
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(recorded, columns=list(names))
    frame["loglkl"] = truth
    frame["predicted"] = draw["predicted"]
    frame["mode"] = draw["label"] + 1
    with atomic_path(directory / f"boost_it_{iteration}.csv") as temporary:
        frame.to_csv(temporary, index=False)
    report.update(iteration=int(iteration), search=draw["info"], created=time.strftime("%Y-%m-%d %H:%M:%S"))
    for m, extra in zip(report["modes"], draw["modes"]):
        m.update({key: extra[key] for key in extra if key != "weight"})
    _write(directory / f"boost_it_{iteration}.json", report)
    # What the user needs to decide whether another boost is worth its true calls: the error of
    # the network at this boost's points, which it had not seen, and the same for the boosts before.
    from utils.held_out import describe

    log(f"   network {iteration} at its boost points, which it had not seen: network minus truth " + describe(report["error"]))
    history = boost_errors(directory, iteration)
    if len(history) > 1:
        log("   scatter at the boost points, boost by boost: " + ", ".join(f"{value:.3g}" for _, value in history))
    return report


def record_start(directory, iteration, configured, verdict, temperature, threshold):
    """The ordinary loop converged at `iteration`, before acquisition.boost.start_iteration:
    the boosts begin there. RUN/boost/start.json; a continued run reads it (recorded_start)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _write(directory / "start.json",
           dict(start_iteration=int(iteration), configured=int(configured), reason="the ordinary loop converged",
                scatter_at_the_tempered_batches=verdict["used"], threshold=float(threshold), temperature=float(temperature),
                limit=verdict["limit"], created=time.strftime("%Y-%m-%d %H:%M:%S")))


def recorded_start(directory):
    """The iteration at which a run's boosts began because its ordinary loop had converged
    (RUN/boost/start.json), or None."""
    file = Path(directory) / "start.json"
    return int(json.loads(file.read_text())["start_iteration"]) if file.exists() else None


def record_fallback(directory, iteration, reason):
    """A boost iteration that could not draw its points and acquired as an ordinary iteration
    instead: RUN/boost/fallback_it_N.json. It has no boost_it_N files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _write(directory / f"fallback_it_{iteration}.json",
           dict(iteration=int(iteration), reason=str(reason), acquired="tempered batch",
                created=time.strftime("%Y-%m-%d %H:%M:%S")))


def boost_errors(directory, through=None):
    """[(iteration, scatter of network minus truth at that boost's points)] from the boost
    reports of a run, up to iteration `through`."""
    out = []
    for file in Path(directory).glob("boost_it_*.json"):
        saved = json.loads(file.read_text())
        error = saved.get("error")
        if error is None and len(saved.get("modes", [])) == 1:      # a report written before the pooled error
            error = dict(sd=saved["modes"][0].get("error_sd"))
        if error and error.get("sd") is not None and (through is None or saved["iteration"] <= through):
            out.append((int(saved["iteration"]), float(error["sd"])))
    return sorted(out)
