"""Persistent backend processes (at most the host's CPUs); the parent owns durable call accounting."""

import importlib
import json
import multiprocessing
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from profiles.journal import records
from run.storage import digest
import numpy as np


class CallBudgetExceeded(RuntimeError):
    pass


class EvaluationLedger:
    """Count every request and backend invocation; cache only deterministic calls.

    The append-only journal records a start *before* calling the backend. Thus an
    interrupted invocation remains visible as an attempted call, even without a
    returned value. An imported value is explicitly historical, never a new call.
    Use a fresh ledger for each process; journals are not concurrent-writer safe.
    """
    def __init__(self, objective, bounds, *, journal, max_calls, deterministic=True):
        self.objective = objective
        self.bounds = np.asarray(bounds, float)
        if (self.bounds.ndim != 2 or self.bounds.shape[1] != 2
                or not np.isfinite(self.bounds).all()
                or np.any(self.bounds[:, 0] >= self.bounds[:, 1])):
            raise ValueError('finite ordered bounds required')
        if max_calls < 1: raise ValueError('positive call budget required')
        self.max_calls, self.deterministic = int(max_calls), bool(deterministic)
        self.journal = Path(journal)
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        if self.journal.exists() and self.journal.stat().st_size:
            raise ValueError('refuse to overwrite an existing evaluation journal')
        self.counts, self.by_stage, self.cache = Counter(), {}, {}
        self.stage = 'unspecified'

    def _event(self, **event):
        with self.journal.open('a') as f:
            f.write(json.dumps(dict(time=time.time(), stage=self.stage, **event), allow_nan=False)+'\n')
            f.flush()

    def _count(self, key):
        self.counts[key] += 1
        self.by_stage.setdefault(self.stage, Counter())[key] += 1

    def _key(self, x):
        return np.asarray(x, dtype='<f8').tobytes()

    def import_value(self, x, value, *, source):
        x = np.asarray(x, float)
        if not self.deterministic: raise ValueError('cannot cache a stochastic objective')
        if x.shape != (len(self.bounds),) or not np.isfinite(x).all() or not np.isfinite(value):
            raise ValueError('finite imported coordinate/value required')
        if np.any(x < self.bounds[:, 0]) or np.any(x > self.bounds[:, 1]):
            raise ValueError('imported coordinate outside bounds')
        key = self._key(x)
        if key in self.cache and self.cache[key] != float(value):
            raise ValueError('inconsistent imported values at identical coordinates')
        self.cache[key] = float(value)
        self._count('import_records')
        self._event(event='import', x=x.tolist(), value=float(value), source=str(source))

    def __call__(self, x):
        x = np.asarray(x, float)
        self._count('requests')
        if (x.shape != (len(self.bounds),) or not np.isfinite(x).all()
                or np.any(x < self.bounds[:, 0]) or np.any(x > self.bounds[:, 1])):
            self._count('invalid_requests')
            self._event(event='invalid_request')
            return -np.inf
        key = self._key(x)
        if self.deterministic and key in self.cache:
            self._count('cache_hits')
            self._event(event='cache_hit', x=x.tolist(), value=self.cache[key])
            return self.cache[key]
        if self.counts['backend_calls'] >= self.max_calls:
            self._count('budget_rejections')
            raise CallBudgetExceeded('total true-likelihood call budget reached')
        self._count('backend_calls')
        number = self.counts['backend_calls']
        self._event(event='start', call=number, x=x.tolist())
        start = time.monotonic()
        try:
            value = float(self.objective(x))
        except Exception as exc:
            self._count('exceptions')
            self._event(event='exception', call=number, error=repr(exc), seconds=time.monotonic()-start)
            raise
        finite = bool(np.isfinite(value))
        self._count('finite_returns' if finite else 'nonfinite_returns')
        self._event(event='return', call=number, value=value if finite else None,
                    seconds=time.monotonic()-start)
        if finite and self.deterministic: self.cache[key] = value
        return value if finite else -np.inf

    def summary(self):
        return dict(counts=dict(self.counts), by_stage={k:dict(v) for k,v in self.by_stage.items()},
                    max_calls=self.max_calls, cache_entries=len(self.cache),
                    deterministic=self.deterministic)


MAX_REFUSED_SHARE = 0.5  # of a batch; a judgement, not a measurement
_BACKEND = None
_NAMES = None


def clip_profile_starts(source, bounds):
    """A copy of a surrogate profile record with every start (xopt, best_x) inside the box.

    Surrogate scans can return a coordinate a few ulps outside its bound. The ledger rejects such a point as an invalid request
    (-inf), so true stages must start from the clipped point. Returns (source, changes);
    changes lists every coordinate moved, with the scan's value and the bound it was set to.
    """
    source = json.loads(json.dumps(source))
    bounds = np.asarray(bounds, float)
    names = source.get("full_parameter_names", [str(k) for k in range(len(bounds))])
    changes = []

    def clip(x, where):
        x = np.asarray(x, float)
        c = np.clip(x, bounds[:, 0], bounds[:, 1])
        for k in np.flatnonzero(c != x):
            changes.append(dict(where=where, coordinate=names[k], scan_value=float(x[k]),
                                clipped_to=float(c[k])))
        return c.tolist()

    if source.get("best_x") is not None:
        source["best_x"] = clip(source["best_x"], "best_x")
    for name, rec in source.get("profiles", {}).items():
        rec["xopt"] = [clip(x, f"{name}/{j}") for j, x in enumerate(rec["xopt"])]
    return source, changes


def initialize_backend(wrapper, input_path, names, bounds, module, expected_hash):
    global _BACKEND, _NAMES
    if module:
        imported = importlib.import_module(module)
        found = digest(Path(imported.__file__))
        if expected_hash and found != expected_hash:
            raise ValueError(f"{module} binary hash mismatch")
    from likelihood.base import build_likelihood

    _BACKEND = build_likelihood(wrapper, input_path)
    _NAMES = names
    if _BACKEND.param_names != names:
        raise ValueError("backend parameter order differs from metadata")
    # The persisted model domain is authoritative. Do not alter backend densities.
    _BACKEND._effective_bounds = dict(zip(names, map(tuple, bounds)))
    import atexit

    def close():
        if hasattr(_BACKEND, "cobaya_model"):
            _BACKEND.cobaya_model.close()
        elif hasattr(_BACKEND, "close"):
            _BACKEND.close()

    atexit.register(close)


def evaluate_backend(x):
    if hasattr(_BACKEND, "cobaya_model"):
        from likelihood.cobaya import native_value

        return native_value(
            _BACKEND.cobaya_model.logposterior(
                dict(zip(_NAMES, map(float, x))), return_derived=False, cached=False
            )
        )
    return float(_BACKEND.loglkl(x))


def record_refused(out, stage, points, values):
    """Which rows of a batch have a value (a boolean mask). The others, points the likelihood
    refused, are not training or validation data: they are listed in OUT/refused_points.json
    under `stage`, and left out by the caller. More than half of a batch refused stops the study:
    its design is not where the likelihood gives values."""
    from profiles.io import save

    points, values = np.asarray(points, float), np.asarray(values, float)
    keep = np.isfinite(values)
    refused = int(np.count_nonzero(~keep))
    if refused:
        path = Path(out) / "refused_points.json"
        saved = json.loads(path.read_text()) if path.exists() else {}
        listed = {tuple(x) for x in saved.get(stage, [])} | set(map(tuple, points[~keep].tolist()))
        saved[stage] = sorted(map(list, listed))
        save(path, saved)
        print(f"{stage}: the likelihood refused {refused} of {len(values)} points; they are left out", flush=True)
    if refused > MAX_REFUSED_SHARE * len(values):
        raise RuntimeError(f"{stage}: the likelihood refused {refused} of the batch's {len(values)} points (refused_points.json): "
                           "the design is not where the likelihood gives values; inspect the backend and the ranges")
    return keep


class BackendPool:
    def __init__(self, workers, **kwargs):
        self.executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_backend,
            initargs=tuple(
                kwargs[n]
                for n in [
                    "wrapper",
                    "input_path",
                    "names",
                    "bounds",
                    "module",
                    "expected_hash",
                ]
            ),
        )

    def submit(self, x):
        return self.executor.submit(evaluate_backend, np.asarray(x, float))

    def __call__(self, x):
        return self.submit(x).result()

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


class WorkflowLedger(EvaluationLedger):
    """Sequential optimizer calls and parallel acquisition share one exact cache.

    On resume import successful values only; count all prior attempts toward the
    lifetime budget, including interrupted starts. Imports are not new calls.
    """

    def __init__(self, pool, bounds, directory, budget):
        directory = Path(directory)
        directory.mkdir(exist_ok=True)
        old = sorted(directory.glob("attempt_*.jsonl"))
        historical = [(p, list(records(p, quarantine=True))) for p in old]
        self.prior_by_stage = Counter()
        self.prior_values_by_stage = {}
        for p, events in historical:
            starts = {}
            for e in events:
                if e['event'] == 'start':
                    self.prior_by_stage[e['stage']] += 1
                    starts[e['call']] = e
                elif e['event'] == 'truncated':
                    self.prior_by_stage['recovery/uncertain'] += 1
                elif e['event'] == 'return' and e.get('value') is not None:
                    start = starts[e['call']]
                    self.prior_values_by_stage.setdefault(start['stage'], []).append((start['x'], e['value']))
        self.prior_calls = sum(self.prior_by_stage.values())
        if self.prior_calls > budget:
            raise CallBudgetExceeded("lifetime call budget exhausted")
        super().__init__(
            pool,
            bounds,
            journal=directory / f"attempt_{len(old):04d}.jsonl",
            max_calls=max(1, budget - self.prior_calls),
        )
        self.max_calls = budget - self.prior_calls
        self.pool = pool
        self.directory = directory
        self.stage = "resume/import"
        for p, events in historical:
            starts = {}
            for e in events:
                if e["event"] == "start":
                    starts[e["call"]] = e["x"]
                if e["event"] == "return" and e.get("value") is not None:
                    self.import_value(starts[e["call"]], e["value"], source=p)

    def evaluate_many(self, points):
        # Schedule unique misses only. No threading touches the parent ledger.
        points = np.asarray(points, float)
        pending = {}
        values = {}
        failure = None
        if (
            points.ndim != 2
            or points.shape[1] != len(self.bounds)
            or not np.isfinite(points).all()
            or np.any(points < self.bounds[:, 0])
            or np.any(points > self.bounds[:, 1])
        ):
            raise ValueError("invalid batch coordinates")
        keys = [self._key(x) for x in points]
        misses = set(keys) - set(self.cache)
        if self.counts["backend_calls"] + len(misses) > self.max_calls:
            raise CallBudgetExceeded("batch would exceed lifetime call budget")
        for x, key in zip(points, keys):
            self._count("requests")
            if key in self.cache:
                self._count("cache_hits")
                self._event(event="cache_hit", x=x.tolist(), value=self.cache[key])
                values[key] = self.cache[key]
            elif key in pending:
                self._count("coalesced_requests")
                self._event(event="coalesced", x=x.tolist())
            else:
                self._count("backend_calls")
                number = self.counts["backend_calls"]
                self._event(event="start", call=number, x=x.tolist())
                start = time.monotonic()
                try:
                    future = self.pool.submit(x)
                except Exception as exc:
                    self._count("exceptions")
                    self._event(
                        event="exception",
                        call=number,
                        error=repr(exc),
                        seconds=time.monotonic() - start,
                    )
                    failure = exc
                    break
                pending[key] = (future, number, start, x.copy())
        for key, (future, number, start, x) in pending.items():
            try:
                value = float(future.result())
                finite = bool(np.isfinite(value))
                self._count("finite_returns" if finite else "nonfinite_returns")
                self._event(
                    event="return",
                    call=number,
                    value=value if finite else None,
                    seconds=time.monotonic() - start,
                )
                if finite:
                    self.cache[key] = value
                values[key] = value if finite else -np.inf     # as __call__: what a refused point is
            except Exception as exc:
                self._count("exceptions")
                self._event(
                    event="exception",
                    call=number,
                    error=repr(exc),
                    seconds=time.monotonic() - start,
                )
                failure = exc
        self.persist()
        if failure:
            raise failure
        return np.array([values[k] for k in keys])

    def persist(self):
        counts = dict(
            self.summary(),
            previous_attempts=self.prior_calls,
            lifetime_attempts=self.prior_calls + self.counts["backend_calls"],
        )
        from profiles.training import save

        save(self.directory / "current_counts.json", counts)

    def __call__(self, x):
        try:
            return super().__call__(x)
        finally:
            self.persist()


def aggregate_counts(directory):
    from collections import Counter

    counts = Counter()
    stages = {}
    unclosed = 0
    for p in sorted(Path(directory).glob("attempt_*.jsonl")):
        starts = set()
        closed = set()
        for e in records(p):
            event = e["event"]
            counts[event] += 1
            if event == "return":
                counts[
                    (
                        "finite_returns"
                        if e.get("value") is not None
                        else "nonfinite_returns"
                    )
                ] += 1
            stages.setdefault(e["stage"], Counter())[event] += 1
            if event == "start":
                starts.add(e["call"])
            elif event in ["return", "exception"]:
                closed.add(e["call"])
        unclosed += len(starts - closed)
    return dict(
        events=dict(counts),
        by_stage={k: dict(v) for k, v in stages.items()},
        backend_attempts=counts["start"],
        unclosed_attempts=unclosed,
        uncertain_tail_records=counts["truncated"],
        budget_reserved_attempts=counts["start"] + counts["truncated"],
        note="Objective backend attempts, not internal theory computations; imports/cache/coalescing are separate. No historical baseline costs inferred from row count.",
    )
