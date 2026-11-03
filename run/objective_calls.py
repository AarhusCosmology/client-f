"""Durable, stage-separated objective attempts; these are not internal solver counts."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time
import uuid
import numpy as np
from run.storage import write_json


# Seconds between fsyncs of a journal. Opening, syncing and closing for every record cost 3.8 ms
# per evaluation on a laptop's disk, thousands of times an analytic likelihood; a held file with
# this limit costs 19 us. A likelihood slower than this is synced before every call.
SYNC_SECONDS = 1.0


class ObjectiveCalls:
    def __init__(self, directory, rank):
        self.directory = Path(directory) / 'objective_calls'
        self.directory.mkdir(exist_ok=True)
        self.path = self.directory / f'rank{rank}-{uuid.uuid4().hex}.jsonl'
        self.stage = 'initial/design'
        self.counts = Counter()
        self._stream = None
        self._synced = -SYNC_SECONDS

    def append(self, record):
        """Every record is flushed, so a likelihood that kills its process leaves it behind. A
        node's loss can take what is not yet synced, which a start record is synced against:
        it carries the return before it, and the call after it may take long."""
        if self._stream is None:
            self._stream = self.path.open('a')
        self._stream.write(json.dumps(record, allow_nan=False) + '\n')
        self._stream.flush()
        now = time.monotonic()
        if record['event'] != 'return' and now - self._synced >= SYNC_SECONDS:
            os.fsync(self._stream.fileno())
            self._synced = now

    def evaluate(self, likelihood, point):
        call = uuid.uuid4().hex
        self.append(dict(event='start', id=call, stage=self.stage,
                         x=np.asarray(point, float).tolist(), time=time.time()))
        try:
            values = np.asarray(likelihood.evaluate_with_derived(point), float)
            native = float(values.ravel()[0])
            status = ('refused' if native == getattr(likelihood, 'refused_value', None)
                      else 'finite' if np.isfinite(native) else 'nonfinite')
        except Exception as exc:
            self.append(dict(event='exception', id=call, stage=self.stage,
                             error=f'{type(exc).__name__}: {exc}'))
            raise
        self.append(dict(event='return', id=call, stage=self.stage, status=status,
                         values=[float(v) if np.isfinite(v) else None for v in values.ravel()],
                         nonfinite=[str(v) if not np.isfinite(v) else None for v in values.ravel()]))
        return values


def inherited_values(root):
    """(record, count, note) of what a branch took over from its source: the record
    hpc/analysis/branch_run.py writes (branch_provenance/imports.json). This runs after every
    acquisition, so a record that cannot be read is reported, not raised: the count is then
    not known (None), as it is for a branch made before the tool kept the source's journals
    apart, whose stages hold the source's calls too."""
    file = root / 'branch_provenance' / 'imports.json'
    if file.exists():
        try:
            record = json.loads(file.read_text())
            count = record['imports']
            if type(count) is not int or count < 0:
                raise ValueError(f'imports is {count!r}')
            return record, count, None
        except (OSError, ValueError, KeyError, TypeError) as error:
            return None, None, (f'branch_provenance/imports.json cannot be read ({type(error).__name__}: {error}): '
                                'the values inherited from the source are not counted here.')
    if (root / 'BRANCHED_FROM.txt').exists():
        return None, None, ('A branch made before its source\'s call journals were kept apart (no branch_provenance/): '
                            'the stages include the source\'s calls, those after the branch point too, and the '
                            'values inherited from the source are not counted.')
    return None, 0, None


def summarize(directory):
    from profiles.journal import records
    root = Path(directory)
    totals = defaultdict(Counter)
    for path in (root / 'objective_calls').glob('rank*.jsonl'):
        for record in records(path, quarantine=True):
            count = totals[record['stage']]
            event = record['event']
            if event == 'start':
                count['attempts'] += 1
            elif event == 'return':
                count[record['status']] += 1
            elif event == 'exception':
                count['exceptions'] += 1
            elif event == 'truncated':
                count['uncertain_tail_records'] += 1
    for count in totals.values():
        count['unfinished'] = count['attempts'] - sum(count[k] for k in ('finite', 'nonfinite', 'refused', 'exceptions'))
    inherited, imports, branch_note = inherited_values(root)
    report = dict(stages={k: dict(v) for k, v in sorted(totals.items())},
                  imports=imports, cache_hits=0,
                  inherited=inherited, note='Actual ordinary-pipeline objective attempts. '
                  'No internal C++ work count. Historical pre-journal calls are not reconstructed.')
    if branch_note:
        report['branch_note'] = branch_note
    write_json(root / 'objective_calls' / 'summary.json', report)
    return report
