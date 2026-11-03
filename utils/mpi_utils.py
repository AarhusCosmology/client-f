import os
import time

import numpy as np

_MPI_CHECKED = False
_COMM = None


def _get_communicator():
    global _MPI_CHECKED, _COMM
    if not _MPI_CHECKED:
        try:
            from mpi4py import MPI
        except ImportError:
            _COMM = None
        else:
            comm = MPI.COMM_WORLD
            _COMM = comm if comm.Get_size() > 1 else None
        _MPI_CHECKED = True
    return _COMM


def is_mpi_available():
    return _get_communicator() is not None


def _get_rank():
    comm = _get_communicator()
    return comm.Get_rank() if comm else 0


def get_size():
    comm = _get_communicator()
    return comm.Get_size() if comm else 1


def is_master():
    return _get_rank() == 0


def print_master(message, end="\n"):
    if is_master():
        print(message, end=end, flush=True)


def bcast(obj, root=0):
    comm = _get_communicator()
    return comm.bcast(obj, root=root) if comm else obj


# ---------------------------------------------------------------------------
# Idle waiting
#
# A blocking MPI call polls the network in a tight loop, so a rank waiting inside
# comm.bcast/scatter holds its core at 100%. That is invisible while every rank is
# evaluating, but the workers are idle for the WHOLE of training, sampling and
# acquisition -- the majority of an iteration once the likelihood is cheap -- and
# during that stretch seven spinning ranks compete with rank 0's multi-threaded
# Keras training for the same cores.
#
# Measured on the 8-core M3, rank 0 training a 5x512 network for 30 epochs while the
# other seven ranks waited for it:
#
#     1 rank, nobody waiting                      9.15 s   1.00x
#     8 ranks, blocking collective               20.03 s   2.19x   <- the bug
#     8 ranks, blocking + mpi_yield_when_idle=1  11.36 s   1.24x
#     8 ranks, idle_wait below                   10.30 s   1.13x
#     8 ranks, idle_wait + mpi_yield_when_idle=1  9.39 s   1.03x
#
# So the long waits are replaced by a non-blocking receive polled with sleep in
# between. Sleeping costs one poll interval of latency twice per iteration, against
# roughly half the training wall clock.
#
# The environment variable is complementary, not redundant -- it also helps the
# short per-batch waits inside the evaluation loop, which this cannot reach -- and
# the two together are essentially free. Open MPI only:
#
#     export OMPI_MCA_mpi_yield_when_idle=1
# ---------------------------------------------------------------------------

_WAKE_TAG = 7301
_IDLE_POLL_SECONDS = 0.05


def _idle_wait_for_master(comm, tag=_WAKE_TAG, root=0, poll=_IDLE_POLL_SECONDS,
                          poll_max=None):
    """Block until `root` releases us, sleeping rather than spinning.

    `poll` is the first sleep and `poll_max` the ceiling it doubles towards. The
    default is a flat 50 ms, which is right for the long between-phase wait where
    latency is irrelevant.
    """
    poll_max = poll if poll_max is None else poll_max
    request = comm.irecv(source=root, tag=tag)
    interval = poll
    while True:
        received, payload = request.test()
        if received:
            return payload
        time.sleep(interval)
        interval = min(interval * 2.0, poll_max)


def _release_idle_workers(comm, payload, tag=_WAKE_TAG):
    """Wake every worker parked in _idle_wait_for_master."""
    for destination in range(1, comm.Get_size()):
        comm.send(payload, dest=destination, tag=tag)


def bcast_idle(obj, root=0):
    """bcast for a value the workers may have to wait a long time for.

    Same contract as bcast, but the workers sleep instead of spinning. Use it
    wherever the master does substantial work before the broadcast; use plain
    bcast when the wait is short, since this one costs a poll interval.
    """
    comm = _get_communicator()
    if comm is None:
        return obj
    if comm.Get_rank() == root:
        _release_idle_workers(comm, obj)
        return obj
    return _idle_wait_for_master(comm, root=root)


def _format_progress(done, total, elapsed):
    percent = 100.0 * done / total if total else 100.0
    return f"Evaluated {done}/{total} samples ({percent:.1f}%) in {elapsed:.2f}s"


def _progress_batch_size(n_processes):
    return max(1, n_processes)


def _evaluate_local(samples, evaluator):
    return np.asarray([evaluator(sample) for sample in samples])


# ---------------------------------------------------------------------------
# Work distribution
#
# static (the default) splits each batch of n_ranks points one per rank and gathers. A
# gather is a barrier, so a batch costs the MAX of its points: about 13% on Planck, where
# CLASS varies about +/-10% from point to point (12.5% measured on the 2026-09-30 Planck +
# DESI run: each rank 87.5% busy).
#
# A dispatcher that hands out points one at a time was measured no faster than this and is not
# in the code (PERFORMANCE.md, section 5).
#
# CLIENT_MPI_SCHEDULE=chunked scatters the whole phase once, one contiguous slice per rank,
# and gathers once at the end. A chunk costs the SUM of its points, and with ranks
# homogeneous on average the only idle time left is the spread of the ranks' totals at the
# end. Progress is reported from rank 0's slice.
#
# A dispatcher on rank 0 handing out one point at a time was measured no faster than static
# on the real Planck likelihood (the ranks were homogeneous to +/-0.6%, so there was nothing
# for load balancing to recover, and the dispatcher cost an eighth of the machine) and far
# slower on microsecond likelihoods. It was removed.
#
# Results are identical either way: the point set is unchanged, loglkl is
# deterministic in x, and values are written back by index. Only WHICH rank
# evaluates a given point changes.
# ---------------------------------------------------------------------------

def _requested_schedule():
    requested = os.environ.get("CLIENT_MPI_SCHEDULE", "").strip().lower()
    return requested if requested in ("static", "chunked") else None


def _evaluate_chunked(comm, points, total, evaluator, start_time):
    """One contiguous slice per rank: a single scatter and gather, no per-point barrier."""
    size = comm.Get_size()
    chunks = np.array_split(points, size, axis=0) if is_master() else None
    local = comm.scatter(chunks, root=0)
    report_every = max(1, len(local) // 20)
    local_values = []
    for i, sample in enumerate(local):
        local_values.append(evaluator(sample))
        if is_master() and ((i + 1) % report_every == 0 or i + 1 == len(local)):
            # Rank 0's own share, scaled: the others run at the same average pace.
            done = min(total, round((i + 1) / len(local) * total))
            print_master(_format_progress(done, total, time.monotonic() - start_time) + " (rank 0's share)")
    local_values = np.asarray(local_values)
    gathered = comm.gather(local_values, root=0)
    if is_master():
        values = np.concatenate([v for v in gathered if len(v)])
        print_master(_format_progress(total, total, time.monotonic() - start_time))
        return values
    return np.array([])


def _evaluate_static(comm, points, total, evaluator, start_time):
    """Lock-step: scatter one slice per rank, gather, repeat. Barrier per batch."""
    size = comm.Get_size()
    gathered_values = []
    batch_size = _progress_batch_size(size)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        chunks = np.array_split(points[start:end], size, axis=0) if is_master() else None
        local_samples = comm.scatter(chunks, root=0)
        local_values = _evaluate_local(local_samples, evaluator)
        batch_values = comm.gather(local_values, root=0)
        if is_master():
            # A rank left without a point in the last batch returns a shape-(0,)
            # array, which will not concatenate with (n, k) rows from a vector
            # evaluator. Every batch holds at least one point, so one survives.
            gathered_values.append(np.concatenate([v for v in batch_values if len(v)]))
            print_master(_format_progress(end, total, time.monotonic() - start_time))

    if is_master() and gathered_values:
        return np.concatenate(gathered_values)
    return np.array([])


def _broadcast_and_evaluate(samples, evaluator):
    """Evaluate master-owned samples across MPI ranks.

    `evaluator` may return a scalar or a fixed-length 1-D row, e.g.
    BaseLikelihood.evaluate_with_derived; the master's values then have shape (n,)
    or (n, k) respectively.
    """
    comm = _get_communicator()
    if comm is None:
        points = np.asarray(samples)
        total = len(points)
        print_master(f"Evaluating {total} samples...")
        start_time = time.monotonic()
        values = _evaluate_local(points, evaluator)
        print_master(_format_progress(total, total, time.monotonic() - start_time))
        return points, values

    if is_master():
        points = np.asarray(samples)
        total = len(points)
        print_master(f"Evaluating {total} samples with {get_size()} MPI processes...")
    else:
        points = np.array([])
        total = None

    # The workers have been idle since the previous evaluation -- through training,
    # sampling and acquisition -- so this is the long wait, not a collective between
    # two busy ranks. Hand the count over without spinning; see bcast_idle above.
    total = bcast_idle(total, root=0)
    if total == 0:
        return points, np.array([])

    start_time = time.monotonic()

    # Static unless explicitly asked for otherwise.
    if _requested_schedule() == "chunked":
        values = _evaluate_chunked(comm, points, total, evaluator, start_time)
    else:
        values = _evaluate_static(comm, points, total, evaluator, start_time)

    if is_master():
        return points, values
    return np.array([]), np.array([])


def broadcast_and_evaluate_checked(samples, evaluator, width):
    """Audit-only transport of Python evaluation errors without stranding MPI peers.

    Every point returns a fixed-width numeric row. Error details are gathered after
    the evaluation phase, so the caller can collectively stop on fatal errors.
    Native process crashes/timeouts are outside this Python exception boundary.
    """
    failures = []

    def checked(x):
        try:
            value = np.asarray(evaluator(x), dtype=float).reshape(-1)
            if value.shape != (width,):
                raise ValueError(f"evaluation returned {len(value)} values, expected {width}")
            return value
        except Exception as error:
            # Only recognised numerical failures may enter the penalty policy.
            # Configuration/programming errors are transported, then raised on all ranks.
            recoverable = (isinstance(error, (FloatingPointError, OverflowError, ZeroDivisionError))
                           or any(c.__name__ == "CosmoComputationError"
                                  for c in type(error).__mro__))
            failures.append(dict(x=np.asarray(x, float).tolist(),
                                 exception=type(error).__name__, message=str(error)[:4000],
                                 recoverable=recoverable, rank=_get_rank()))
            return np.full(width, np.nan)

    points, values = _broadcast_and_evaluate(samples, checked)
    comm = _get_communicator()
    if comm is not None:
        gathered = comm.gather(failures, root=0)
        failures = [f for group in gathered for f in group] if is_master() else []
    return points, values, failures


_FAILURES_TOLD, _MESSAGE_TOLD = 5, 1000     # of a failed batch, to the ranks other than the master


class EvaluationBatchError(RuntimeError):
    """All ranks stop together; master retains successes for durable accounting.

    `failures` is every failure on the master and the first few on the other ranks;
    `total` is their number on every rank."""
    def __init__(self, failures, points=None, results=None, total=None):
        self.failures, self.points, self.results = failures, points, results
        self.total = len(failures) if total is None else int(total)
        super().__init__(f"objective batch failed at {self.total} point(s): " + "; ".join(
            f"rank {e['rank']} {e['exception']}: {e['message'][:_MESSAGE_TOLD]}" for e in failures[:_FAILURES_TOLD]))


def broadcast_and_evaluate(samples, evaluator):
    """Ordinary transport with collective Python-error propagation.

    Each scheduled evaluation returns an object envelope, so a failure never skips
    a gather/send. Native process crashes still require an external supervisor.
    """
    def protected(x):
        try:
            value = np.asarray(evaluator(x), dtype=float)
            if value.ndim > 1:
                raise ValueError("objective must return a scalar or one-dimensional row")
            record = dict(value=value, error=None)
        except Exception as exc:
            record = dict(value=None, error=dict(x=np.asarray(x, float).tolist(),
                rank=_get_rank(), exception=type(exc).__name__, message=str(exc)[:4000]))
        row = np.empty(1, dtype=object)
        row[0] = record
        return row

    points, envelopes = _broadcast_and_evaluate(samples, protected)
    values, errors = [], []
    if is_master():
        records = np.asarray(envelopes, object).reshape(-1).tolist()
        errors = [r['error'] for r in records if r['error'] is not None]
        values = [r['value'] for r in records]
        if not errors and values and any(v.shape != values[0].shape for v in values):
            errors = [dict(rank=0, exception='ValueError', message='inconsistent objective row shapes')]
    # The other ranks learn of a failed batch through bcast_idle, whose non-blocking receive
    # has mpi4py's default buffer of 32 KiB: a batch with a few dozen failures, each with its
    # point and a long message, did not fit and ended the job with MPI_ERR_TRUNCATE instead
    # of this error. They are told the number and the first few, cut short.
    total, told = bcast_idle((len(errors), [dict(rank=e['rank'], exception=e['exception'],
                                                 message=e['message'][:_MESSAGE_TOLD])
                                            for e in errors[:_FAILURES_TOLD]]))
    if total:
        raise EvaluationBatchError(errors if is_master() else told, points, values, total=total)
    return points, np.asarray(values) if is_master() else np.array([])
