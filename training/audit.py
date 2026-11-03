"""Audit of a trained network's optima with true evaluations (`acquisition.audit`).

A network can predict a region far from every training point to be much better than the
best true value seen. The acquisition chain starts its walkers uniformly over the box, so
it can find such a region; the next batch is then drawn there, and its true values, all
poor, are what corrects the network. The audit does the same correction with a few true
evaluations instead of a batch:

    1. find_maxima: the network's highest maxima over the box (the network alone)
    2. flag those predicted more than `tau` log-units above the best true value
    3. evaluate the true likelihood at the `k` highest flagged maxima
    4. add them to the training set, fine-tune, and search again

The truth decides: a refuted maximum becomes an ordinary training point where the network
was wrong, a confirmed one is a better point than any seen. See docs/audit_optima.md for
the measurements behind the defaults and for what the audit cannot do.
"""
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from training.training import build_optimizer

# After the last round the fit near the best point is compared with the fit before the
# audit: fine-tuning on a few points the network had badly wrong can leave it worse for a
# while (measured: the error on the true profile ridge rose from 1.1 to 17.6 log-units in
# the middle of a seven-round audit and was back at 0.9 at its end). Up to SETTLE_ROUNDS
# further fine-tunes, without new points and at a third of the learning rate, run while
# the fit is more than SETTLE_FACTOR times its value before the audit.
SETTLE_FACTOR = 1.2
SETTLE_ROUNDS = 3


def _predict(model, points, batch=8192):
    points = np.asarray(points, dtype=np.float32)
    if not len(points):
        return np.empty(0, dtype=float)
    return np.concatenate([
        model(tf.constant(points[i:i + batch]), training=False).numpy().ravel()
        for i in range(0, len(points), batch)
    ]).astype(float)


def inside_float32(points, lower, upper):
    """`points` at single precision, inside the box: the coordinates the training set stores."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    lo = lower.astype(np.float32)
    hi = upper.astype(np.float32)
    lo = np.where(lo.astype(float) < lower, np.nextafter(lo, np.float32(np.inf)), lo)
    hi = np.where(hi.astype(float) > upper, np.nextafter(hi, np.float32(-np.inf)), hi)
    return np.clip(np.asarray(points, dtype=np.float32), lo, hi).astype(float)


STARTS_PER_PARAMETER = 128
# A round first searches with an eighth of the starts and evaluates what that flags:
# while false maxima are plentiful a small search finds them. The full search runs when
# the quick one flags nothing, and from then on in every round: it is what concludes that
# none is left. (At 58 parameters the full search has 14 976 starts and took 400 s on a
# loaded laptop against 40 s for the quick one.)
QUICK_DIVISOR = 8


def default_starts(ndim):
    """Starts of each kind when `starts` is 0: 128 per parameter, 2048 at 16 parameters,
    the only dimension measured so far. A provisional rule: the `hits` of each maximum
    are in the report so that the number needed can be measured on other likelihoods."""
    return STARTS_PER_PARAMETER * int(ndim)


def find_maxima(model, lower, upper, inputs, targets, starts=0, steps=400,
                learning_rate=0.02, n_top=128, merge=0.05, keep=50, seed=None, counts=None,
                project=None):
    """The network's highest maxima over the box, by projected gradient ascent.

    project : optional TensorFlow function on box-normalised points, applied to the starts
        and after every step. The ascent then stays on what it projects onto (a Region's
        project_unit: the region the boosts' posterior points reach, sampling/region.py),
        and the far starts become starts all over its surface.

    Starts: `starts` uniform points (0: default_starts), as many random box vertices, and
    the `n_top` best training points. The ascent runs in box-normalised coordinates (Adam,
    projected onto the box). End points closer than `merge` box widths (per root
    dimension) count as one maximum; the `keep` highest are returned, best first, as
    dicts with

        x          the point
        value      the network's prediction there
        excess     value minus the best training value
        distance   to the nearest training point, in box widths
        at_bound   number of coordinates at a bound
        hits       number of starts that ended at this maximum
        starts     number of starts in all (hits / starts: the share of the search
                   that found it, from which the starts needed to find it follow)

    `counts`, a dict, is filled with the whole search: `starts`, and for every distinct
    maximum (not only the `keep` highest) its `excess` and `hits`, highest first.

    The vertex starts matter: the most violent false maxima of a long-trained network sit
    at vertices, and uniform starts alone missed those of two networks out of six. The
    search is not exhaustive. Four times the starts and twice the steps found one or two
    further false maxima on two audited networks, with one seed and not with another.
    """
    rng = np.random.default_rng(seed)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    width = upper - lower
    ndim = len(lower)
    starts = starts or default_starts(ndim)
    inputs = np.asarray(inputs, float)
    targets = np.asarray(targets, float).ravel()
    blocks = [rng.uniform(size=(starts, ndim)),
              rng.integers(0, 2, size=(starts, ndim)).astype(float)]
    if n_top:
        top = inputs[np.argsort(targets)[-min(n_top, len(targets)):]]
        blocks.append((top - lower) / width)
    u = tf.Variable(np.clip(np.concatenate(blocks), 0.0, 1.0), dtype=tf.float32)
    if project is not None:
        u.assign(tf.clip_by_value(project(u), 0.0, 1.0))
    lo_t, width_t = tf.constant(lower, tf.float32), tf.constant(width, tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    @tf.function
    def ascend():
        with tf.GradientTape() as tape:
            loss = -tf.reduce_sum(model(lo_t + u * width_t, training=False))
        optimizer.apply_gradients([(tape.gradient(loss, u), u)])
        u.assign(tf.clip_by_value(u, 0.0, 1.0))
        if project is not None:
            u.assign(tf.clip_by_value(project(u), 0.0, 1.0))

    for _ in range(steps):
        ascend()
    end = u.numpy().astype(float)
    points = inside_float32(lower + end * width, lower, upper)
    end = (points - lower) / width
    values = _predict(model, points)
    # Every end point joins the highest maximum within the merge radius, or starts one.
    radius2 = (merge * np.sqrt(ndim)) ** 2
    centres = np.empty((0, ndim))
    members, hits = [], []
    for i in np.argsort(-values):
        if len(centres):
            d2 = ((centres - end[i]) ** 2).sum(axis=1)
            j = int(d2.argmin())
            if d2[j] < radius2:
                hits[j] += 1
                continue
        centres = np.vstack([centres, end[i]])
        members.append(i)
        hits.append(1)
    normalised = (inputs - lower) / width
    best = float(targets.max())
    if counts is not None:
        counts.update(starts=len(end), excess=[float(values[i] - best) for i in members],
                      hits=[int(h) for h in hits])
    maxima = []
    for i, n_hits in list(zip(members, hits))[:keep]:
        distance = np.sqrt(((normalised - end[i]) ** 2).sum(axis=1)).min()
        maxima.append(dict(
            x=points[i], value=float(values[i]), excess=float(values[i] - best),
            distance=float(distance),
            at_bound=int(np.sum((end[i] < 1e-4) | (end[i] > 1.0 - 1e-4))),
            hits=int(n_hits), starts=len(end),
        ))
    return maxima


def weights_digest(model):
    """A fingerprint of a network's weights: whether a chain was sampled from this very network."""
    import hashlib

    digest = hashlib.sha256()
    for weight in model.get_weights():
        digest.update(np.ascontiguousarray(weight).tobytes())
    return digest.hexdigest()


def summit_maxima(model, summits, lower, upper, inputs, targets, merge=0.05, keep=50, counts=None):
    """Points that another search climbed to, as find_maxima returns its maxima: the summits of
    a tempered chain's cooled walkers (sampling.modes.cooled_peaks). They are taken where they
    are. Climbed again by find_maxima, whose ascent takes steps of a fixed size and returns
    where it ends, a narrow summit was lost: handed a maximum 16 above the best true value,
    it came back with 9.6."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    width = upper - lower
    ndim = len(lower)
    points = inside_float32(np.asarray(summits, float).reshape(-1, ndim), lower, upper)
    end = (points - lower) / width
    values = _predict(model, points)
    radius2 = (merge * np.sqrt(ndim)) ** 2
    centres = np.empty((0, ndim))
    members, hits = [], []
    for i in np.argsort(-values):
        if len(centres):
            d2 = ((centres - end[i]) ** 2).sum(axis=1)
            j = int(d2.argmin())
            if d2[j] < radius2:
                hits[j] += 1
                continue
        centres = np.vstack([centres, end[i]])
        members.append(i)
        hits.append(1)
    normalised = (np.asarray(inputs, float) - lower) / width
    best = float(np.asarray(targets, float).max())
    if counts is not None:
        counts.update(starts=len(end), excess=[float(values[i] - best) for i in members],
                      hits=[int(h) for h in hits])
    return [dict(x=points[i], value=float(values[i]), excess=float(values[i] - best),
                 distance=float(np.sqrt(((normalised - end[i]) ** 2).sum(axis=1)).min()),
                 at_bound=int(np.sum((end[i] < 1e-4) | (end[i] > 1.0 - 1e-4))),
                 hits=int(n_hits), starts=len(end))
            for i, n_hits in list(zip(members, hits))[:keep]]


def fine_tune(model, inputs, targets, loss, validation, epochs, learning_rate, batch_size):
    """Continue training `model`; the final weights are kept, not the best-validation ones.

    Restoring the best validation loss would undo the correction: the validation points
    never include the audit points, so the loss on them is lowest before the network has
    learned those.

    Compiled without XLA: a fine-tune compiles a new step, on a training set of a new size, and
    every XLA compilation leaves memory mappings that the process never gives back (about 600
    per fine-tune of a 5 x 512 network, measured). With up to eight fine-tunes per iteration that
    is most of what a run accumulates, and runs of 21 iterations ended at Linux's limit of 65 530
    mappings per process ("LLVM compilation error: Cannot allocate memory"). Without XLA a
    fine-tune leaves about 20 and takes 10 to 25% longer.
    """
    model.compile(optimizer=build_optimizer(learning_rate), loss=loss, jit_compile=False)
    history = model.fit(
        np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32),
        epochs=epochs, batch_size=batch_size, validation_data=validation, verbose=0,
    )
    return float(history.history["loss"][-1])


def audit_optima(model, inputs, targets, validation_split, lower, upper, evaluate,
                 build_loss, config, batch_size, n_append, seed=None, log=print, penalties=(),
                 project=None, peaks=None, cloud=False):
    """Run the audit on `model`, in place. Returns (inputs, targets, derived, report).

    project : optional, passed to every search (find_maxima): with a Region's project_unit
        the searches stay inside the region of the boosts, so only maxima that a T = 1
        consumer could meet are looked for, evaluated and trained on.

    peaks : optional (acquisition.audit.chain, in a boost iteration). peaks(model) ->
        (summits, walkers): where the walkers of a tempered chain of the model end when
        cooled and climbed, and the cooled walkers themselves (sampling.modes.cooled_peaks),
        as the mode search of a boost finds its modes. A round whose own climbs flag nothing
        looks there last (the "chain" search): a maximum that only the chain reaches was
        otherwise met by the boost, at a third of its points. A chain per such round.

    cloud : with `peaks`, a flagged maximum of the chain search is evaluated together with
        cooled walkers that climbed to it, up to the round's k points: the truth where the
        network's posterior is, not at its summit alone, and a chain fewer on the one
        network this was measured on (docs/audit_optima.md).

    The posterior pilot (config.posterior, with `peaks`): after the search for maxima,
    config.posterior of the cooled T = 1 walkers of the network kept are evaluated. When the
    network's mean excess over the truth there is above config.posterior_tolerance, it is
    fine-tuned on them and the check is repeated with a new chain, at most
    config.posterior_rounds times; after a last fine-tune the new chain's summits are
    reported. Its points are trained on like the others. Without it the audit is as it was.

    inputs, targets : the arrays the network was just trained on, in the order Keras saw
        them: its validation set is their last `validation_split` fraction and stays
        held out here. The audit points are all trained on.
    evaluate(points) -> (inputs, targets, derived): true evaluations; a value that is not
        finite marks a point where the likelihood failed.
    build_loss(targets) : the training loss for a training set with these targets (its
        best value and the msre scale move when an evaluation beats the best).
    config : an AuditConfig.
    n_append : the acquisition's batch size; with config.k 0 it sets the evaluations per
        round, config.fraction * n_append / config.rounds.

    The returned arrays are the finite evaluations, to be added to the training set. A
    flagged maximum whose evaluation is not finite cannot become a training point (the
    run drops such points everywhere); it is set aside for the rest of this audit so the
    search does not return to it, and the report counts it. With the optional penalize
    policy, bounded failed probes can justify a separately stored synthetic constraint;
    it participates in fine-tuning only, never in the returned true-data arrays.
    """
    inputs = np.asarray(inputs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    width = upper - lower
    ndim = inputs.shape[1]
    k = config.per_round(n_append)
    split = int(len(inputs) * (1.0 - validation_split))
    held_out = (inputs[split:], targets[split:]) if split < len(inputs) else None
    check_x, check_y = held_out if held_out is not None else (inputs, targets)
    near = check_y.ravel() > float(targets.max()) - config.fit_window

    def fit():
        if not near.any():
            return float("nan")
        residual = check_y.ravel()[near] - _predict(model, check_x[near])
        return float(np.sqrt(np.mean(residual ** 2)))

    new_x = np.empty((0, ndim))
    new_y = np.empty(0)
    new_d = None
    failed = np.empty((0, ndim))
    from training.audit_failures import active_penalties, evaluate_with_recovery, penalty_training_rows

    constraints = active_penalties(penalties, inputs, lower, upper) if config.failure_policy == "penalize" else []
    inherited = len(constraints)
    added_penalties = []
    attempted = 0
    budget = k * config.rounds
    # With the chain search (`peaks`) its rounds have an allowance of their own, as large as
    # the climbs': it is the last look of a round, so it first sees a maximum when the climbs
    # have nothing left, which was the last round allowed (measured: found, and left standing).
    spent, used = dict(climb=0, chain=0), dict(climb=0, chain=0)
    rng = np.random.default_rng(seed)
    fit_before = fit()

    def tune(rate):
        all_true_x = np.concatenate([inputs, new_x.astype(np.float32)])
        active = active_penalties(constraints, all_true_x, lower, upper)
        px, py = penalty_training_rows(
            active, split + len(new_y), ndim, config.failure_training_fraction)
        fine_tune(model,
                  np.concatenate([inputs[:split], new_x.astype(np.float32), px]),
                  np.concatenate([targets[:split].ravel(), new_y.astype(np.float32), py]),
                  build_loss(np.concatenate([targets.ravel(), new_y.astype(np.float32)])),
                  held_out, config.epochs, rate, batch_size)

    if constraints:
        # Reapply after ordinary training, with real validation and loss calibration.
        tune(config.learning_rate)
        failed = np.asarray([p["x"] for p in constraints], float).reshape(-1, ndim)
    seen = {}  # the chain search's last look, and the network it looked at

    def look():
        found = peaks(model)
        seen.update(digest=weights_digest(model), summits=found[0], walkers=found[1])
        return found

    rounds = []
    states = []  # the weights of every network searched since the best true value last moved
    states_best = None
    full_only = False
    started = time.monotonic()
    for r in range((2 if peaks is not None else 1) * config.rounds + 1):
        all_x = np.concatenate([inputs, new_x.astype(np.float32)])
        all_y = np.concatenate([targets.ravel(), new_y.astype(np.float32)])
        best = float(all_y.max())
        round_start = time.monotonic()
        full = config.starts or default_starts(ndim)
        quick = max(64, full // QUICK_DIVISOR)
        stages = [("full", full)] if full_only or quick >= full else [("quick", quick), ("full", full)]

        def above(found):
            return [
                m for m in found
                if m["excess"] > config.tau and not any(
                    np.sqrt(np.sum(((m["x"] - f) / width) ** 2)) < 0.05 * np.sqrt(ndim)
                    for f in failed)
            ]

        walkers = None
        for stage, n_starts in stages:
            counts = {}
            maxima = find_maxima(
                model, lower, upper, all_x, all_y, starts=n_starts, steps=config.steps,
                keep=max(50, k), seed=None if seed is None else seed + r, counts=counts,
                project=project,
            )
            flagged = above(maxima)
            if flagged:
                break
        if not flagged and peaks is not None:
            # The climbs found nothing: look where a tempered chain's walkers end when cooled.
            climbed = maxima[0]
            stage, counts = "chain", {}
            summits, walkers = look()
            maxima = summit_maxima(model, summits, lower, upper, all_x, all_y, keep=max(50, k), counts=counts)
            if climbed["excess"] > maxima[0]["excess"]:
                maxima = [climbed] + maxima            # the round's highest maximum, whichever search found it
            flagged = above(maxima)
        full_only = full_only or stage in ("full", "chain")
        top = maxima[0]
        above = np.array(counts["excess"]) > config.tau
        distinct = dict(
            maxima=len(counts["excess"]), above_tau=int(above.sum()),
            starts_above_tau=int(np.array(counts["hits"])[above].sum()),
        )
        record = dict(
            round=r, search=stage, distinct_maxima=distinct["maxima"],
            distinct_above_tau=distinct["above_tau"],
            share_of_starts_above_tau=distinct["starts_above_tau"] / counts["starts"],
            highest_excess=top["excess"],
            highest_distance=top["distance"],
            highest_at_bound=top["at_bound"], flagged=len(flagged), fit=fit(),
            search_seconds=time.monotonic() - round_start, starts=top["starts"],
            highest_hits=top["hits"],
            flagged_hits=[m["hits"] for m in flagged],
            flagged_excess=[m["excess"] for m in flagged],
        )
        rounds.append(record)
        if best != states_best:
            states, states_best = [], best
        states.append((r, model.get_weights()))
        log(
            f"   audit round {r}: highest maximum {top['excess']:+.1f} above the best true "
            f"value ({top['distance']:.2f} box widths from the nearest training point, "
            f"{top['at_bound']} coordinates at a bound); {distinct['above_tau']} of "
            f"{distinct['maxima']} distinct maxima above +{config.tau:g}, reached by "
            f"{distinct['starts_above_tau']} of {counts['starts']} starts ({stage} search); "
            f"fit near the best {record['fit']:.2f}"
        )
        kind = "chain" if stage == "chain" else "climb"
        left = budget - spent[kind]
        if not flagged or used[kind] == config.rounds or left <= 0:
            break
        used[kind] += 1
        chosen = flagged[:min(k, left)]
        points = np.array([m["x"] for m in chosen])
        if stage == "chain" and cloud and len(points) < min(k, left):
            # cooled walkers whose own summit is one of the chosen maxima (the merge radius of the search)
            apart = np.sqrt((((np.asarray(summits, float)[:, None] - points[None]) / width) ** 2).sum(axis=2)).min(axis=1)
            at_them = np.flatnonzero(apart < 0.05 * np.sqrt(ndim))
            take = rng.permutation(at_them)[:min(k, left) - len(points)]
            record["cloud"] = int(len(take))
            if len(take):
                points = np.concatenate([points, np.asarray(walkers, float)[take]])
        ev_x, ev_y, ev_d, penalties_now, failures, primary_y = evaluate_with_recovery(
            points, evaluate, config, best, lower, upper, rng, left)
        attempted += len(ev_y)
        spent[kind] += len(ev_y)
        constraints.extend(penalties_now)
        added_penalties.extend(penalties_now)
        finite = np.isfinite(ev_y)
        record.update(
            evaluated=len(ev_y), primary_evaluated=len(points), failures=failures,
            predicted_excess=[m["excess"] for m in chosen],
            true_excess=[float(v - best) if np.isfinite(v) else None for v in primary_y],
        )
        log(
            f"      evaluated {len(chosen)} maxima"
            + (f", {len(points) - len(chosen)} cooled walkers at them" if len(points) > len(chosen) else "")
            + f" and {len(ev_y)-len(points)} nearby probes: predicted "
            f"{[int(round(m['excess'])) for m in chosen]} above the best, truth "
            f"{[int(round(v - best)) if np.isfinite(v) else 'failed' for v in primary_y[:len(chosen)]]}"
        )
        if not finite.all():
            failed = np.concatenate([failed, np.asarray(ev_x, float)[~finite]])
        if not finite.any() and not penalties_now:
            continue
        new_x = np.concatenate([new_x, np.asarray(ev_x, float)[finite]])
        new_y = np.concatenate([new_y, ev_y[finite]])
        if ev_d is not None:
            ev_d = np.asarray(ev_d, float)[finite]
            new_d = ev_d if new_d is None else np.concatenate([new_d, ev_d])
        tune(config.learning_rate)

    # The network that is kept. A fine-tune can leave the network worse than the one before
    # it, so the last round's is not kept as such: of the networks searched with the final
    # best true value, the one with the lowest highest maximum is. The points of every
    # round stay in the training set.
    last = rounds[-1]
    # Like with like: the chain search sees maxima that the climbs do not, so a round that it
    # looked at is compared with such rounds only. Against the climbs' rounds, whose highest
    # maximum is what the climbs found, the round with the deeper look lost and the
    # fine-tunes before it were undone (measured).
    deep = last["search"] == "chain"
    kept_round, kept_weights = min(
        (state for state in states if (rounds[state[0]]["search"] == "chain") == deep),
        key=lambda state: rounds[state[0]]["highest_excess"])
    kept = rounds[kept_round]
    if kept_round != last["round"]:
        model.set_weights(kept_weights)
        log(f"   audit: the network of round {kept_round} is kept (highest maximum "
            f"{kept['highest_excess']:+.1f}, fit near the best {kept['fit']:.2f}); the last "
            f"round's had {last['highest_excess']:+.1f} and {last['fit']:.2f}")
    del states

    settled = 0
    fit_after = fit()
    if len(new_y) or constraints:
        while (settled < SETTLE_ROUNDS and np.isfinite(fit_before)
               and fit_after > SETTLE_FACTOR * fit_before):
            before = model.get_weights()
            tune(config.learning_rate / 3.0)
            settled += 1
            fit_now = fit()
            if not fit_now < fit_after:  # no better: the network before this fine-tune stays
                model.set_weights(before)
                break
            fit_after = fit_now
        if settled:
            log(f"   audit: {settled} settling fine-tune(s); fit near the best "
                f"{fit_after:.2f} (before the audit {fit_before:.2f})")
    # Settling changes the weights after the best-round search. Report the final
    # network, using the full search even when a quick search ended that round.
    final_search = None
    if settled:
        counts = {}
        maxima = find_maxima(
            model, lower, upper, all_x, all_y,
            starts=config.starts or default_starts(ndim), steps=config.steps,
            keep=max(50, k), seed=None if seed is None else seed + config.rounds + 1,
            counts=counts, project=project,
        )
        final_search = dict(
            highest_excess=maxima[0]["excess"],
            distinct_above_tau=int(np.count_nonzero(np.asarray(counts["excess"]) > config.tau)),
            starts=counts["starts"],
        )
    final_status = final_search if final_search is not None else kept
    def add(x, y, d):
        """Points of the pilot, trained on from its next fine-tune on, as the rounds' are."""
        nonlocal new_x, new_y, new_d
        new_x, new_y = np.concatenate([new_x, x]), np.concatenate([new_y, y])
        if d is not None:
            new_d = d if new_d is None else np.concatenate([new_d, d])

    def current():
        return (np.concatenate([inputs, new_x.astype(np.float32)]),
                np.concatenate([targets.ravel(), new_y.astype(np.float32)]))

    pilot = posterior_pilot(model, config, look, seen, evaluate, add, current, tune, fit, lower, upper,
                            rng, log) if peaks is not None and config.posterior else None
    if pilot is not None:
        attempted += pilot.pop("calls")
        if pilot["tuned"]:
            fit_after = fit()
        if pilot["tuned"] or any(r["new_best"] for r in pilot["rounds"]):
            # The pre-pilot search cannot describe changed weights (or a new true
            # reference value). Recheck the delivered model without any true calls.
            pilot["pre_pilot_status"] = dict(final_status)
            all_x, all_y = current()
            counts = {}
            n_starts = config.starts or default_starts(ndim)
            maxima = find_maxima(
                model, lower, upper, all_x, all_y, starts=n_starts, steps=config.steps,
                keep=2 * n_starts + len(all_x), seed=None if seed is None else seed + config.rounds + 2,
                counts=counts, project=project)
            if seen.get("digest") != weights_digest(model):
                look()
            merged = {}
            top = summit_maxima(
                model, np.concatenate([np.asarray([m["x"] for m in maxima]), seen["summits"]]),
                lower, upper, all_x, all_y, keep=1, counts=merged)
            final_search = dict(
                highest_excess=top[0]["excess"],
                distinct_above_tau=int(np.count_nonzero(np.asarray(merged["excess"]) > config.tau)),
                starts=counts["starts"], chain_summits=len(seen["summits"]),
                weights_digest=weights_digest(model), stage="post_pilot")
            final_status = final_search
    report = dict(
        tau=config.tau, k=k, ndim=ndim, rounds=rounds, true_calls=attempted,
        call_budget=budget * (2 if peaks is not None else 1)
        + (config.posterior * config.posterior_rounds if pilot is not None else 0),
        failure_policy=config.failure_policy,
        failure_training_fraction=config.failure_training_fraction,
        penalized=len(added_penalties), penalties_added=added_penalties,
        penalties_reapplied=inherited,
        added=int(len(new_y)), failed=attempted - int(len(new_y)),
        new_best=int(np.sum(new_y > float(targets.max()))) if len(new_y) else 0,
        kept_round=int(kept_round),
        clear=bool(final_status["distinct_above_tau"] == 0),
        highest_excess=final_status["highest_excess"], final_search=final_search,
        fit_before=fit_before, fit_after=fit_after, settling_rounds=settled,
        seconds=time.monotonic() - started, restricted=project is not None,
        **({} if pilot is None else {"posterior": pilot}),
    )
    log(
        f"   audit: {report['true_calls']} true evaluations in {len(rounds) - 1} round(s)"
        + (f", {report['failed']} not finite" if report["failed"] else "")
        + f"; {report['added']} genuine points added, {report['penalized']} synthetic constraints"
        + (f", {report['new_best']} better than the best before" if report["new_best"] else "")
        + (f"; nothing above +{config.tau:g} is left in the search"
           if report["clear"] else
           f"; {final_status['distinct_above_tau']} maxima still above "
           f"+{config.tau:g} (highest {final_status['highest_excess']:+.1f})")
        + f"; fit near the best {fit_before:.2f} -> {fit_after:.2f}"
    )
    return new_x, new_y, new_d, report


def posterior_pilot(model, config, look, seen, evaluate, add, current, tune, fit, lower, upper, rng, log):
    """The posterior pilot of audit_optima (acquisition.audit.posterior). add(x, y, derived)
    puts finite evaluations among the audit's points; current() is (inputs, targets) with
    them. Returns a dict: the rounds, whether the network was fine-tuned, its summits'
    highest excess after the last fine-tune, and, for the caller, the calls made."""
    best = float(np.max(current()[1]))
    out = dict(k=int(config.posterior), tolerance=float(config.posterior_tolerance), rounds=[], tuned=False)
    calls = 0
    tuned = False
    for p in range(config.posterior_rounds):
        if seen.get("digest") != weights_digest(model):
            look()                                    # walkers of this very network
        walkers = np.asarray(seen["walkers"], float)
        pick = rng.choice(len(walkers), size=min(config.posterior, len(walkers)), replace=False)
        ev_x, ev_y, ev_d = evaluate(walkers[pick])
        ev_x, ev_y = np.asarray(ev_x, float), np.asarray(ev_y, float)
        calls += len(ev_y)
        finite = np.isfinite(ev_y)
        excess = _predict(model, ev_x[finite]) - ev_y[finite]
        record = dict(round=p, evaluated=int(len(ev_y)), finite=int(finite.sum()),
                      status="finite" if finite.any() else "no_finite_returns",
                      mean_excess=float(excess.mean()) if finite.any() else None,
                      rms=float(np.sqrt(np.mean(excess ** 2))) if finite.any() else None,
                      above_5=float(np.mean(excess > 5.0)) if finite.any() else None,
                      new_best=int(np.sum(ev_y[finite] > best)))
        if finite.any():
            add(ev_x[finite], ev_y[finite], None if ev_d is None else np.asarray(ev_d, float)[finite])
        tuned = bool(finite.any() and record["mean_excess"] > config.posterior_tolerance)
        record["fine_tuned"] = tuned
        out["rounds"].append(record)
        log(f"   audit, posterior pilot {p}: {len(ev_y)} cooled T = 1 walkers evaluated"
            + (f" ({len(ev_y) - int(finite.sum())} not finite)" if not finite.all() else "")
            + (f": network minus truth mean {record['mean_excess']:+.2f}, rms {record['rms']:.2f}, "
               f"{record['above_5']:.0%} more than 5 above"
               + (f"; above the tolerance {config.posterior_tolerance:g}: fine-tuned" if tuned else
                  f"; within the tolerance {config.posterior_tolerance:g}") if finite.any() else
               "; no finite returns: no fine-tune; posterior accuracy is unassessed"))
        if not tuned:
            break
        tune(config.learning_rate)
        out["tuned"] = True
    if out["tuned"]:
        # Reuse a look already made after an earlier tune; otherwise sample the final
        # model now. This also covers a tune followed by a no-tune pilot round.
        if seen.get("digest") != weights_digest(model):
            look()
        top = summit_maxima(model, seen["summits"], lower, upper, *current(), keep=1, counts={})
        out["highest_excess_after"] = float(top[0]["excess"])
        log(f"   audit, posterior pilot: after the last fine-tune the chain's highest summit is "
            f"{out['highest_excess_after']:+.1f} above the best true value; fit near the best {fit():.2f}")
    out["calls"] = calls
    return out


def save_audit(directory, iteration, names, inputs, targets, report, *,
               derived_names=(), derived=None):
    """audit_it_N.csv (the points added: the run's parameters, then loglkl, as a training
    data file) and audit_it_N.json (the report). The CSV is written also when it is empty:
    a continued run reads it to know what the saved network N was trained on."""
    import pandas as pd
    from run.storage import atomic_path, write_json

    directory = Path(directory)
    frame = pd.DataFrame(np.asarray(inputs, dtype=np.float32).reshape(-1, len(names)), columns=list(names))
    frame["loglkl"] = np.asarray(targets, dtype=np.float32).ravel()
    with atomic_path(directory / f"audit_it_{iteration}.csv") as temporary:
        frame.to_csv(temporary, index=False)
    write_json(directory / f"audit_it_{iteration}.json", report)

    # Synthetic constraints never enter the real-data CSV or derived sidecar.
    constraints = report.get("penalties_added", [])
    penalty_frame = pd.DataFrame([p["x"] for p in constraints], columns=list(names))
    penalty_frame["penalty_loglkl"] = [p["penalty_loglkl"] for p in constraints]
    with atomic_path(directory / f"audit_penalties_it_{iteration}.csv") as temporary:
        penalty_frame.to_csv(temporary, index=False)

    # Keep the training CSV schema unchanged; repeat inputs to validate row alignment.
    sidecar = frame[list(names)].copy()
    values = (np.full((len(frame), len(derived_names)), np.nan) if derived is None
              else np.asarray(derived, dtype=float).reshape(len(frame), len(derived_names)))
    for j, name in enumerate(derived_names):
        sidecar[name] = values[:, j]
    with atomic_path(directory / f"audit_derived_it_{iteration}.csv") as temporary:
        sidecar.to_csv(temporary, index=False)


def load_audit(directory, iteration, names, derived_names=()):
    """Read audit points and row-aligned derived values; old audits leave NaNs.

    Newly requested derived columns are also unknown. A present but misaligned
    sidecar is rejected rather than silently assigning another point's values.
    """
    import pandas as pd
    from dataset.augment import read_augmentation

    directory = Path(directory)
    inputs, targets = read_augmentation(directory / f"audit_it_{iteration}.csv", names)
    derived = np.full((len(inputs), len(derived_names)), np.nan)
    path = directory / f"audit_derived_it_{iteration}.csv"
    if path.exists() and derived_names:
        frame = pd.read_csv(path)
        if (any(n not in frame for n in names)
                or frame[list(names)].shape != inputs.shape
                or not np.array_equal(frame[list(names)].to_numpy(np.float32),
                                      inputs.astype(np.float32))):
            raise ValueError(f"{path.name}: derived rows do not match audit inputs")
        for j, name in enumerate(derived_names):
            if name in frame:
                derived[:, j] = frame[name].to_numpy(float)
    return inputs, targets, derived
