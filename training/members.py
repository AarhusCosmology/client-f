"""Several trainings of one network, the best kept (training.members; docs/members.md).

A network trained once can end in one of two states: it resolves the fine structure of the
likelihood, or it stays on a plateau where that structure is smoothed out. Which one depends on
the random start (docs/members.md: on the same training set one start in three or two in three
resolved it, a longer schedule did not help), and a warm start keeps
whatever state the previous network was in. So with training.members: K > 1 the network of an
iteration is trained K times on the same training set with the same settings: by rank 0 as
before (warm-started when training.warm_start is on), and K - 1 times from random weights,
each with its own seed, on ranks that would otherwise wait for the next evaluation. They are
compared on a selection set: a share of the points acquired since the previous training, held
out from every member's training this time (all of them see it from the next network on), so
unseen by all, the warm one included, whose own validation loss is partly in-sample. The warm
network is kept unless a member from random weights is better by more than
training.member_margin; without a warm one the best is kept.

members: 1 (the default) changes nothing: no point is held out and no rank is asked.
"""
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

MEMBER_TAG = 7311          # rank 0 -> a member rank: its task, or None
MEMBER_RESULT_TAG = 7312   # a member rank -> rank 0: its result


def trains_members(members, boost_start, iteration):
    """Whether network `iteration` is trained by the members: only in the boost phase, from the
    first training on boost points (the network after the boosts' start, acquisition.boost) on.
    Every network of the tempered phase is trained once, as without the key (Steen, 4 November
    2026: a feature of the boosts). Every rank asks it, with the same boost start."""
    return members > 1 and boost_start is not None and iteration > boost_start


def member_ranks(size, members):
    """The ranks that train members 1 .. members-1, spread evenly over the ranks (and so over
    the nodes), at most one per rank and never rank 0; fewer than members-1 when the run has
    fewer ranks. The members left over are trained on rank 0, one after another."""
    wanted = members - 1
    if size <= 1 or wanted <= 0:
        return []
    count = min(wanted, size - 1)
    step = size / (count + 1)
    ranks = sorted({min(size - 1, max(1, int(round(step * (k + 1))))) for k in range(count)})
    k = 1
    while len(ranks) < count:                   # rounding put two on one rank
        if k not in ranks:
            ranks.append(k)
        k += 1
    return sorted(ranks)


def oversubscription(size, members, threads, rank0_threads, cores):
    """The log's warning (None when there is nothing to warn of) when the members of one training
    cannot all train at once: members left over for rank 0, which trains them one after another,
    or more TensorFlow threads than the run has cores, taken as its ranks (one core per rank, as a
    cluster job usually starts them) or the cores of rank 0's node if those are more. Either makes the
    training, which lasts as long as its slowest member, take several times as long."""
    placed = len(member_ranks(size, members))
    left = members - 1 - placed
    demand = rank0_threads + placed * threads
    capacity = max(size, cores)
    notes = []
    if left > 0:
        notes.append(f"{left} member(s) are trained on rank 0 after its own network, one after another: "
                     f"the training takes about {left + 1} times as long")
    if demand > capacity:
        notes.append(f"the trainings ask for {demand} TensorFlow threads (rank 0's {rank0_threads} and {placed} "
                     f"members at {threads}) and the run has {capacity} cores (its {size} ranks, or the cores of "
                     f"rank 0's node if more): sharing them, the training takes about {demand / capacity:.1f} "
                     f"times as long")
    if not notes:
        return None
    fit = max(1, min(size, 1 + max(0, capacity - rank0_threads) // threads))
    return (f"   warning: {'; '.join(notes)}. At most {fit} {'member trains' if fit == 1 else 'members train'} "
            f"at once at {threads} threads here: set fewer members or a smaller training.member_threads")


def member_seed(base, iteration, member):
    """The seed of a member: fixed by the run's seed, the iteration and the member."""
    return int((0 if base is None else int(base)) + 104729 * member + 1299709 * (iteration + 1)) % (2 ** 31 - 1)


def previous_rows(run_dir, iteration):
    """How many rows of network `iteration`'s training set network `iteration - 1` trained on,
    including its pilot's and its audit's points (added to the training set after it was
    trained, so seen by the warm start): the training set of the previous network plus those."""
    if iteration <= 0:
        return 0
    run_dir = Path(run_dir)

    def rows(path):
        if not path.exists():
            return 0
        with path.open() as stream:
            return max(0, sum(1 for _ in stream) - 1)

    def pilot_rows(path):                    # those with a finite true value were added (load_pilot)
        if not path.exists():
            return 0
        import pandas as pd

        return int(np.isfinite(pd.read_csv(path, usecols=["loglkl"])["loglkl"].to_numpy(float)).sum())

    return (rows(run_dir / "training_data" / f"data_it_{iteration - 1}.csv")
            + pilot_rows(run_dir / "trained_models" / "members" / f"pilot_it_{iteration - 1}.csv")
            + rows(run_dir / "audit" / f"audit_it_{iteration - 1}.csv"))


def newest_rows(n_rows, n_previous):
    """Rows of a training set acquired since the previous training: those after its first
    n_previous (the training set of the previous network)."""
    newest = np.zeros(n_rows, bool)
    newest[max(0, min(int(n_previous), n_rows)):] = True
    return newest


def selection_rows(newest, share, rng):
    """The held-out selection set: a share of the newest rows (at least one when there are
    any), drawn with its own generator so that the run's other draws are unchanged."""
    candidates = np.flatnonzero(newest)
    held = np.zeros(len(newest), bool)
    if not len(candidates):
        return held
    count = max(1, int(round(share * len(candidates))))
    held[rng.choice(candidates, size=min(count, len(candidates)), replace=False)] = True
    return held


def selection_loss(model, loss, inputs, targets, batch=8192):
    """The training loss (msre) of a network on the selection set: the mean over its points."""
    import tensorflow as tf

    x = np.asarray(inputs, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
    if not len(x):
        return float("nan")
    total = 0.0
    for start in range(0, len(x), batch):
        stop = min(len(x), start + batch)
        prediction = model(tf.constant(x[start:stop]), training=False)
        total += float(loss(tf.constant(y[start:stop]), prediction)) * (stop - start)
    return total / len(x)


def choose(results, warm, margin):
    """Which member to keep: (index, reason). results[0] is rank 0's network; a result
    without a finite selection loss is never kept. With a warm network (warm=True) it is
    kept unless the best of the others is lower by more than `margin` (a fraction)."""
    usable = [r for r in results if r.get("error") is None and np.isfinite(r.get("selection_loss", np.nan))]
    if not usable:
        raise RuntimeError("no member of this training has a finite selection loss")
    first = results[0]
    first_ok = first.get("error") is None and np.isfinite(first.get("selection_loss", np.nan))
    best = min(usable, key=lambda r: r["selection_loss"])
    if warm and first_ok:
        others = [r for r in usable if r["member"] != first["member"]]
        if not others:
            return first["member"], "the warm network (no other member finished)"
        challenger = min(others, key=lambda r: r["selection_loss"])
        if challenger["selection_loss"] < (1.0 - margin) * first["selection_loss"]:
            return challenger["member"], (
                f"member {challenger['member']} (random weights), {challenger['selection_loss']:.3g} against the "
                f"warm network's {first['selection_loss']:.3g}: lower by more than {100 * margin:g}%")
        return first["member"], (
            f"the warm network, {first['selection_loss']:.3g}; the best from random weights "
            f"{challenger['selection_loss']:.3g} is not lower by more than {100 * margin:g}%")
    return best["member"], f"member {best['member']}, the lowest selection loss {best['selection_loss']:.3g}"


def train_member(task):
    """Train one member from random weights (on any rank) and save it: its result."""
    started = time.monotonic()
    result = dict(member=task["member"], seed=task["seed"], start="random weights", rank=task.get("rank", 0))
    try:
        threads = task.get("threads")
        import tensorflow as tf

        if threads:
            try:
                tf.config.threading.set_intra_op_parallelism_threads(int(threads))
                tf.config.threading.set_inter_op_parallelism_threads(1)
            except RuntimeError:
                pass                                  # TensorFlow already running on this rank
        from model.network import build_model
        from run.storage import save_model
        from training.losses import build_loss
        from training.training import save_history, train_model

        if task.get("clear", True):
            tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(task["seed"])
        inputs, targets = task["inputs"], task["targets"]
        m, s = task["model"], task["schedule"]
        model = build_model(inputs=inputs, targets=targets, n_layers=m["n_layers"], n_neurons=m["n_neurons"],
                            activation=m["activation"],
                            **({} if task["background"] is None else {"background": task["background"]}))
        loss = build_loss(**task["loss"], max_loglkl=float(np.max(targets)), targets=targets)
        history, metrics = train_model(model=model, inputs=inputs, targets=targets, loss=loss,
                                       restore_best=True, **s)
        result.update(epoch=int(metrics["epoch"]), loss=float(metrics["loss"]), val_loss=float(metrics["val_loss"]),
                      epochs=len(history.history["loss"]),
                      selection_loss=(float("nan") if task.get("selection_inputs") is None else
                                      selection_loss(model, loss, task["selection_inputs"], task["selection_targets"])))
        save_model(model, Path(task["model_path"]))
        save_history(history.history, Path(task["history_path"]))
        result.update(model_path=str(task["model_path"]), history_path=str(task["history_path"]))
        del model
        if task.get("clear", True):
            tf.keras.backend.clear_session()
    except Exception as error:                    # reported to rank 0, which keeps another member
        result["error"] = f"{type(error).__name__}: {error}"
    result["seconds"] = time.monotonic() - started
    return result


def train_on_master(task):
    """A member trained on rank 0 after its own network: the run's random streams as before."""
    state, python_state = np.random.get_state(), random.getstate()
    try:
        return train_member(dict(task, clear=False))   # rank 0's own network stays usable
    finally:
        np.random.set_state(state)
        random.setstate(python_state)


def serve(comm):
    """A worker rank's part of one training: wait (asleep) for rank 0's message; train the
    member it holds, if any, and send the result back."""
    task = receive(comm, 0, MEMBER_TAG)
    if task is None:
        return None
    task["rank"] = comm.Get_rank()
    if task.get("threads"):
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(task["threads"])
    result = train_member(task)
    comm.send(result, dest=0, tag=MEMBER_RESULT_TAG)
    return result


def dispatch(comm, tasks):
    """Rank 0: send every worker its task ({rank: task}) or None."""
    for rank in range(1, comm.Get_size()):
        comm.send(tasks.get(rank), dest=rank, tag=MEMBER_TAG)


def receive(comm, source, tag, poll=0.05, poll_max=0.5):
    """A message of any size, waited for without spinning: probe, sleep, then receive. (A
    non-blocking receive of a pickled message has a fixed buffer of 32 kB in mpi4py, and a
    member's task, its training set, is megabytes: MPI_ERR_TRUNCATE.)"""
    interval = poll
    while not comm.Iprobe(source=source, tag=tag):
        time.sleep(interval)
        interval = min(interval * 2.0, poll_max)
    return comm.recv(source=source, tag=tag)


def collect(comm, ranks, poll=0.5):
    """Rank 0: the result of each member rank, waiting without spinning."""
    return [receive(comm, rank, MEMBER_RESULT_TAG, poll=poll, poll_max=poll) for rank in ranks]


def write_summary(path, iteration, results, kept, reason, selection_points, share, margin, warm):
    from run.storage import atomic_path

    record = dict(iteration=iteration, kept=kept, reason=reason, warm=warm, selection_points=int(selection_points),
                  selection_share=share, margin=margin,
                  members=[{k: v for k, v in r.items() if k not in ("inputs", "targets")} for r in results])
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as temporary:
        Path(temporary).write_text(json.dumps(record, indent=1) + "\n")
    return record


def summary_line(record):
    """One line per training for the run's log."""
    parts = []
    for r in record["members"]:
        if r.get("error"):
            parts.append(f"{r['member']} failed ({r['error']})")
        else:
            parts.append(f"{r['member']}{' (warm)' if r.get('start') == 'warm start' else ''} "
                         f"{r['selection_loss']:.3g}, epoch {r['epoch']} of {r['epochs']}")
    return (f"Members ({record['selection_points']} held-out selection points): " + "; ".join(parts)
            + f". Kept: {record['reason']}")


def gather(cfg, run, iteration, model, history, metrics, loss, selection, tasks, warm):
    """Rank 0 after its own training: its own network saved as member 0, the members left for
    rank 0 trained here, the others' results collected. Returns every member's result."""
    from run.storage import save_model
    from training.training import save_history
    from utils.mpi_utils import _get_communicator, get_size

    members_dir = run.trained_models_dir / "members"
    own_model = members_dir / f"model_it_{iteration}_member0.keras"
    own_history = run.training_history_dir / f"history_it_{iteration}_member0.csv"
    own = dict(member=0, seed=None, start="warm start" if warm else "random weights", rank=0,
               epoch=int(metrics["epoch"]), loss=float(metrics["loss"]), val_loss=float(metrics["val_loss"]),
               epochs=len(history.history["loss"]),
               selection_loss=(float("nan") if selection is None else
                               selection_loss(model, loss, selection[0], selection[1])),
               model_path=str(own_model), history_path=str(own_history))
    save_model(model, own_model)
    save_history(history.history, own_history)
    results = [own]
    for task in tasks:
        if not task["rank"]:
            results.append(train_on_master(task))
    remote = [t["rank"] for t in tasks if t["rank"]]
    if remote and get_size() > 1:
        results += collect(_get_communicator(), remote)
    results.sort(key=lambda r: r["member"])
    return results


def kept_network(results, kept, model, history, metrics):
    """The kept member's network, history and metrics; rank 0's own when it is kept."""
    if kept == 0:
        return model, history, metrics
    from types import SimpleNamespace

    import pandas as pd
    import tensorflow as tf

    from model.network import load_model

    chosen = next(r for r in results if r["member"] == kept)
    tf.keras.backend.clear_session()
    return (load_model(chosen["model_path"]),
            SimpleNamespace(history=pd.read_csv(chosen["history_path"]).to_dict(orient="list")),
            dict(epoch=chosen["epoch"], loss=chosen["loss"], val_loss=chosen["val_loss"]))


def keep_member(cfg, run, iteration, model, history, metrics, loss, selection, tasks, warm):
    """The choice by the held-out selection set (training.member_pilot: 0): the members, the
    choice, their files and summary. Returns the kept network, its history and its metrics,
    which the loop then saves as network `iteration` as before."""
    from utils.mpi_utils import print_master

    results = gather(cfg, run, iteration, model, history, metrics, loss, selection, tasks, warm)
    kept, reason = choose(results, warm, cfg.training.member_margin)
    record = write_summary(run.trained_models_dir / "members" / f"members_it_{iteration}.json", iteration, results,
                           kept, reason, len(selection[0]), cfg.training.selection_share, cfg.training.member_margin,
                           warm)
    print_master(summary_line(record))
    return kept_network(results, kept, model, history, metrics)


# ---------------------------------------------------------------- the pilot (training.member_pilot)
#
# The held-out loss judges the members where the previous network put its posterior, and a loss
# averaged over points cannot see where a member will put its own: a network with a false
# plateau some 10-35 log-units above the truth had the lowest or a middling selection loss
# (network 8 of two axion monodromy runs with members: 12; docs/members.md).
# With member_pilot: n > 0 every member draws n points of its own posterior (T = 1, inside the
# network's region), the pool is evaluated with the true likelihood once, and the member with the
# smallest error over the pool's relevant points is kept. The points join the training set.

PILOT_TAG = 7313           # rank 0 -> a worker: {"task": a member to sample or None, "evaluate": True}, or None
PILOT_RESULT_TAG = 7314    # a member rank -> rank 0: its points


def msre_scale(sigma_level, ndim):
    """0.5 chi^2(sigma_level, ndim) by Wilson-Hilferty: the msre loss's distance below the best (training/losses.py)."""
    k = float(ndim)
    return 0.5 * k * (1.0 - 2.0 / (9.0 * k) + sigma_level * math.sqrt(2.0 / (9.0 * k))) ** 3


def pilot_starts(region, inputs, targets, walkers, rng):
    """Where a member's chain starts: inside the network's region, or, without one, at the best
    distinct training points."""
    if region is not None:
        return region.start(walkers, rng).astype(np.float32)
    order = np.argsort(-np.asarray(targets, float).ravel())
    _, first = np.unique(np.asarray(inputs, np.float32)[order], axis=0, return_index=True)
    rows = order[np.sort(first)[:walkers]]
    return np.asarray(inputs, np.float32)[rows]


def sample_member(task):
    """n points of one member's posterior at T = 1 (on any rank): its result."""
    started = time.monotonic()
    out = dict(member=task["member"], rank=task.get("rank", 0))
    try:
        from types import SimpleNamespace

        from likelihood.surrogate import SurrogateLikelihood, SurrogateMetadata
        from model.network import load_model
        from sampling.base import build_sampler
        from sampling.region import Region

        rng = np.random.default_rng(task["seed"])
        metadata = SurrogateMetadata.load(Path(task["metadata_path"]))
        region = None if task.get("region_path") is None else Region.load(Path(task["region_path"]))
        surrogate = SurrogateLikelihood(model=load_model(task["model_path"]), metadata=metadata,
                                        **({} if region is None else {"region": region}))
        names = metadata.param_names
        lower = np.array([metadata.bounds[n][0] for n in names], float)
        upper = np.array([metadata.bounds[n][1] for n in names], float)
        steps, walkers = task["steps"], max(int(task["walkers"]), 2 * len(names))   # AIES needs 2 ndim
        burn = steps // 4
        thin = max(1, (steps - burn) // 250)
        if region is not None and len(region) > 1:
            from sampling.mixture import sample_region_mixture

            chain, _ = sample_region_mixture(
                surrogate, np.column_stack([lower, upper]),
                SimpleNamespace(n_walkers=walkers, burn_in=burn, chunk_size=5000, progress=False),
                steps, thin, seed=int(rng.integers(2 ** 31)), diagnostics={})
            chain = np.asarray(chain, float).reshape(-1, len(names))
        else:
            sampler = build_sampler("aies", walkers, len(names), surrogate.logpost)
            starts = np.asarray(task["starts"], np.float32)
            if len(starts) < walkers:                 # more walkers than starts: repeat them
                starts = starts[np.arange(walkers) % len(starts)]
            sampler.run(n_steps=steps, initial_positions=starts[:walkers], burn_in=burn,
                        chunk_size=5000, thin=thin, progress=False)
            chain = np.asarray(sampler.chain(), float).reshape(-1, len(names))
            sampler.reset()
        rows = rng.choice(len(chain), size=min(task["n"], len(chain)), replace=False)
        out["points"] = chain[rows].astype(np.float32)
    except Exception as error:                    # this member is then not scored and not kept
        out["error"] = f"{type(error).__name__}: {error}"
    out["seconds"] = time.monotonic() - started
    return out


def sample_on_master(task):
    state, python_state = np.random.get_state(), random.getstate()
    try:
        return sample_member(task)
    finally:
        np.random.set_state(state)
        random.setstate(python_state)


def dispatch_pilot(comm, tasks, evaluate):
    """Rank 0: every worker gets {"task": its member to sample or None, "evaluate": True} when a
    pilot runs this iteration (all of them then join its true evaluation), or None."""
    for rank in range(1, comm.Get_size()):
        comm.send(dict(task=tasks.get(rank), evaluate=True) if evaluate else None, dest=rank, tag=PILOT_TAG)


def serve_pilot(comm, evaluate):
    """A worker's part of the pilot: sample its member if it has one, then join the true
    evaluation of the pool (`evaluate`, the loop's collective)."""
    message = receive(comm, 0, PILOT_TAG)
    if message is None:
        return
    if message.get("task") is not None:
        task = dict(message["task"], rank=comm.Get_rank())
        comm.send(sample_member(task), dest=0, tag=PILOT_RESULT_TAG)
    evaluate()


def pilot_scores(predictions, truth, best, depth, ignore=None):
    """Each member's rms error (prediction minus truth) over the pool's points that the truth or
    that member puts within `depth` of `best`: a false plateau and a missed peak both count.
    Points whose truth is not finite or is `ignore` (the fixed value a launcher may give a point
    the likelihood cannot evaluate) count for none."""
    truth = np.asarray(truth, float)
    ok = np.isfinite(truth)
    if ignore is not None:
        ok &= truth != ignore
    scores = {}
    for k, pred in predictions.items():
        pred = np.asarray(pred, float)
        relevant = ok & ((truth > best - depth) | (pred > best - depth))
        scores[k] = float(np.sqrt(np.mean((pred[relevant] - truth[relevant]) ** 2))) if relevant.any() else float("inf")
    return scores


def choose_by_pilot(scores, warm, margin):
    """(kept, reason): the warm network (member 0) unless another member's pilot error is lower
    by more than `margin`; without a warm one, the lowest."""
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not finite:
        raise RuntimeError("no member of this training has a pilot error")
    best = min(finite, key=finite.get)
    if warm and 0 in finite:
        others = {k: v for k, v in finite.items() if k != 0}
        if not others:
            return 0, "the warm network (no other member was scored)"
        challenger = min(others, key=others.get)
        if others[challenger] < (1.0 - margin) * finite[0]:
            return challenger, (f"member {challenger} (random weights), pilot error {others[challenger]:.3g} against "
                                f"the warm network's {finite[0]:.3g}: lower by more than {100 * margin:g}%")
        return 0, (f"the warm network, pilot error {finite[0]:.3g}; the best from random weights "
                   f"{others[challenger]:.3g} is not lower by more than {100 * margin:g}%")
    return best, f"member {best}, the lowest pilot error {finite[best]:.3g}"


def run_pilot(cfg, run, iteration, results, region_path, dataset_inputs, dataset_targets, names, evaluate, warm,
              derived_names=(), ignore=None):
    """Rank 0's pilot: the members sample their posteriors (on their ranks, and rank 0 for its
    own and any left over), the pool is evaluated once with the true likelihood (`evaluate`, all
    ranks), the members are scored and one is kept. Returns (kept, record, inputs, targets,
    derived) of the pool's evaluation, whose points the loop adds to the training set."""
    import pandas as pd

    from model.network import load_model
    from run.storage import atomic_path
    from training.audit import _predict
    from utils.mpi_utils import _get_communicator, get_size, print_master

    usable = [r for r in results if r.get("error") is None]
    rng = np.random.default_rng(member_seed(cfg.seed, iteration, 7))
    region = None
    if region_path is not None and Path(region_path).exists():
        from sampling.region import Region

        region = Region.load(Path(region_path))
    starts = pilot_starts(region, dataset_inputs, dataset_targets,
                          max(cfg.training.member_pilot_walkers, 2 * len(names)), rng)
    tasks = [dict(member=r["member"], rank=r["rank"], model_path=r["model_path"], n=cfg.training.member_pilot,
                  steps=cfg.training.member_pilot_steps, walkers=cfg.training.member_pilot_walkers,
                  seed=member_seed(cfg.seed, iteration, 1000 + r["member"]), starts=starts,
                  metadata_path=str(run.run_dir / "metadata.json"),
                  region_path=None if region is None else str(region_path)) for r in usable]
    remote = {t["rank"]: t for t in tasks if t["rank"]}
    comm = _get_communicator() if get_size() > 1 else None
    if comm is not None:
        dispatch_pilot(comm, remote, evaluate=True)
    drawn = [sample_on_master(t) for t in tasks if not t["rank"]]
    if comm is not None and remote:
        drawn += [receive(comm, rank, PILOT_RESULT_TAG) for rank in remote]
    failed = [d for d in drawn if d.get("error") is not None or not len(d.get("points", []))]
    drawn = sorted((d for d in drawn if d.get("error") is None and len(d.get("points", []))), key=lambda d: d["member"])
    for d in failed:
        print_master(f"   member {d['member']} could not be sampled for the pilot: {d.get('error') or 'no points'}")
    if not drawn:
        raise RuntimeError("no member of this training could be sampled for the pilot: "
                           + "; ".join(f"{d['member']}: {d.get('error') or 'no points'}" for d in failed))
    points = np.concatenate([d["points"] for d in drawn]).astype(np.float32)
    member_of = np.concatenate([np.full(len(d["points"]), d["member"]) for d in drawn])
    print_master(f"Members' pilot: {len(points)} points of {len(drawn)} members' posteriors "
                 f"({cfg.training.member_pilot} each) evaluated with the true likelihood")
    inputs, targets, derived = evaluate(points)
    truth = np.asarray(targets, float).ravel()
    models = {r["member"]: load_model(r["model_path"]) for r in usable if r["member"] in set(member_of.tolist())}
    predictions = {k: _predict(m, inputs) for k, m in models.items()}
    known = np.isfinite(truth) & (True if ignore is None else truth != ignore)
    best = max(float(np.max(dataset_targets)), float(truth[known].max()) if known.any() else -np.inf)
    depth = cfg.training.member_pilot_depth or msre_scale(cfg.training.sigma_level, len(names))
    scores = pilot_scores(predictions, truth, best, depth, ignore=ignore)
    kept, reason = choose_by_pilot(scores, warm, cfg.training.member_margin)
    own = {k: (float(np.mean(predictions[k][(member_of == k) & known] - truth[(member_of == k) & known]))
               if np.any((member_of == k) & known) else None) for k in predictions}
    for r in results:
        r["pilot_error"] = scores.get(r["member"])
        r["pilot_own_bias"] = own.get(r["member"])
    record = dict(iteration=iteration, kept=kept, reason=reason, warm=warm, choice="pilot",
                  pilot_points=int(len(points)), pilot_per_member=cfg.training.member_pilot, pilot_depth=depth,
                  best=best, margin=cfg.training.member_margin, failed=[d["member"] for d in failed],
                  members=[{k: v for k, v in r.items() if k not in ("inputs", "targets")} for r in results])
    members_dir = run.trained_models_dir / "members"
    with atomic_path(members_dir / f"members_it_{iteration}.json") as temporary:
        Path(temporary).write_text(json.dumps(record, indent=1) + "\n")
    frame = pd.DataFrame(np.asarray(inputs, float), columns=names)
    frame["member"] = member_of
    frame["loglkl"] = truth
    derived = np.asarray(derived, float).reshape(len(frame), -1)
    for j, name in enumerate(derived_names):
        frame[f"derived:{name}"] = derived[:, j]
    for k, pred in predictions.items():
        frame[f"pred_member{k}"] = pred
    with atomic_path(members_dir / f"pilot_it_{iteration}.csv") as temporary:
        frame.to_csv(temporary, index=False)
    print_master("Members' pilot errors: " + "; ".join(
        f"{k}{' (warm)' if (k == 0 and warm) else ''} {scores[k]:.3g}"
        + (f" (own points {own[k]:+.2f})" if own.get(k) is not None else "") for k in sorted(scores))
        + f". Kept: {reason}")
    return kept, record, inputs, targets, derived


def load_pilot(run_dir, iteration, names, derived_names=()):
    """(inputs, targets, derived) of pilot_it_N.csv: the pilot's points with a finite true value,
    which a continued run adds to the training set of a saved network N as the pilot did."""
    import pandas as pd

    frame = pd.read_csv(Path(run_dir) / "trained_models" / "members" / f"pilot_it_{iteration}.csv")
    if any(n not in frame for n in list(names) + ["loglkl"]):
        raise ValueError(f"pilot_it_{iteration}.csv: not the run's parameters and loglkl")
    derived = np.full((len(frame), len(derived_names)), np.nan)
    for j, name in enumerate(derived_names):
        if f"derived:{name}" in frame:
            derived[:, j] = frame[f"derived:{name}"].to_numpy(float)
    keep = np.isfinite(frame["loglkl"].to_numpy(float))       # as the acquisition adds its points
    return frame[list(names)].to_numpy(float)[keep], frame["loglkl"].to_numpy(float)[keep], derived[keep]
