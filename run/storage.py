"""Atomic artifacts, exclusive run ownership, and explicit phase commits."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.stem}-{uuid.uuid4().hex}{path.suffix}')
    try:
        yield temporary
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')


def save_model(model, path):
    with atomic_path(path) as temporary:
        model.save(temporary)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def augmentation_training_iteration(path):
    """The completed first-training iteration, or None for a missing/torn marker."""
    try:
        record = json.loads(Path(path).read_text())
    except (ValueError, OSError):
        return None
    iteration = record.get('iteration') if isinstance(record, dict) else None
    return iteration if type(iteration) is int and iteration >= 0 else None


class RunLock:
    """Held for the master process lifetime; a dead process releases the OS lock."""
    def __init__(self, directory):
        import fcntl
        self.stream = (Path(directory) / '.client.lock').open('a+')
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise RuntimeError(f'Another process owns run {directory}') from None
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(str(os.getpid()) + '\n')
        self.stream.flush()

    def close(self):
        self.stream.close()


def phase_outputs(root, phase):
    """The files a phase writes before it is committed, as they are on disk now.

    A training phase (the training of network N, its region and its audit) writes the
    network, its history, the region, the training domain when N's training built it, the
    audit's files, the derived emulator of N and, with training.members, every member's network
    and history and their record, and marks the augmentation files it trained on first. An acquisition phase (the true
    evaluations chosen with network N, and their injection) writes the boost's record and the
    training set of N + 1. Chain summaries and metrics are written between phases and are
    written again when an iteration is redone.
    """
    root = Path(root)
    kind, _, number = phase.rpartition('_it_')
    if kind not in ('training', 'acquisition') or not number.isdigit():
        return []
    n = int(number)
    if kind == 'training':
        patterns = [f'trained_models/model_it_{n}.keras', f'training_history/history_it_{n}.csv',
                    f'boost/region_it_{n}.npz', f'boost/train_domain_it_{n}.npz', f'audit/*_it_{n}.*',
                    f'derived_data/derived_model_it_{n}.keras', f'derived_data/derived_history_it_{n}.csv',
                    f'derived_data/derived_report_it_{n}.csv',
                    # training.members (training/members.py): every member's network and history,
                    # and the record of which was kept
                    f'trained_models/members/model_it_{n}_member*.keras', f'trained_models/members/members_it_{n}.json',
                    f'trained_models/members/pilot_it_{n}.csv',
                    f'training_history/history_it_{n}_member*.csv']
    else:
        patterns = [f'training_data/data_it_{n + 1}.csv', f'derived_data/derived_it_{n + 1}.csv',
                    f'boost/boost_it_{n}.csv', f'boost/boost_it_{n}.json', f'boost/fallback_it_{n}.json']
    found = [path for pattern in patterns for path in sorted(root.glob(pattern))]
    if kind == 'training':
        # dataset/augment.py mark_trained: the marker names the training that used the points.
        for marker in sorted(root.glob('augmentation/augment_it_*.trained.json')):
            first = augmentation_training_iteration(marker)
            # A torn legacy marker cannot prove completion in any iteration. Preserve it
            # with this interrupted phase so the first-training schedule can run again.
            if first is None or first == n:
                found.append(marker)
    return found


# The iteration in a file's name: _it_N not followed by a digit, whatever else follows
# (evidence_it_25_nlive2000.json, training_it_25.json).
_ITERATION = re.compile(r'_it_(\d+)(?!\d)')
# Kept whole by a rewind: what earlier recoveries set aside, the ledger of true calls (every
# call made in this directory stays counted, those of the iterations set aside too), the
# identity's history, a branch's provenance, the saved inputs.
_NOT_REWOUND = ('interrupted', 'objective_calls', 'objective_identity_history', 'branch_provenance',
                'likelihood_input')


def later_files(root, iteration):
    """The files of a run that do not belong to a run stopped at network `iteration`, where
    `client.py RUN -s N` starts: [(path, keep a copy in place)].

    A restart there loads network N and its training set and does the acquisition of N
    again. Later than that are: every file named after an iteration above N, in any folder
    (training sets, networks, audits, regions, chain summaries, phase records, and the
    analyses of those networks: evidences, mode searches, chains, true checks); the
    acquisition of N itself (its boost or fallback record and its phase record; the training
    set of N + 1 is named after N + 1); the marker of an augmentation whose first training
    made a later network (named after the training set, it records the network: see
    phase_outputs); the record of a hand-over to the boosts after N; and the summaries in
    analysis/ that carry no iteration, which describe the run as it was. The metrics are
    kept in place as well as set aside: the loop reads the rows before N from them.
    """
    root = Path(root)
    found = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if path.is_dir() or relative.parts[0] in _NOT_REWOUND or relative.parts[:2] == ('phases', 'history'):
            continue
        name = relative.name
        if len(relative.parts) == 1:
            if name in ('metrics.json', 'metrics.md'):
                found.append((path, True))
            continue
        if relative.parts[0] == 'augmentation' and re.fullmatch(r'augment_it_\d+\.trained\.json', name):
            first = augmentation_training_iteration(path)
            if first is not None and first > iteration:
                found.append((path, False))
            continue
        if relative.parts == ('boost', 'start.json'):
            try:
                handed_over = int(json.loads(path.read_text())['start_iteration'])
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if handed_over > iteration:
                found.append((path, False))
            continue
        number = next((int(m.group(1)) for part in relative.parts if (m := _ITERATION.search(part))), None)
        if number is None:
            if relative.parts[0] == 'analysis':
                found.append((path, False))
            continue
        of_the_acquisition = number == iteration and (
            (relative.parts[0] == 'boost' and name.startswith(('boost_it_', 'fallback_it_')))
            or relative.parts == ('phases', f'acquisition_it_{iteration}.json'))
        if number > iteration or of_the_acquisition:
            found.append((path, False))
    return found


def set_later_aside(root, iteration, log=print):
    """Move what a run holds of the iterations after `iteration` (later_files) to
    RUN/interrupted/rewind_to_it_N-<time>/ under the paths it had, with a record of it
    (REWIND.json). Nothing is deleted. Call it holding the run lock, after
    PhaseStore.recover and before PhaseStore.check: the phase records of the later
    iterations go with the files they commit. Returns the folder, or None when the run
    holds nothing later."""
    root = Path(root)
    if type(iteration) is not int or iteration < 0 or not (root / f'training_data/data_it_{iteration}.csv').is_file():
        raise ValueError(f'Cannot rewind to iteration {iteration}: need a nonnegative iteration with saved training data')
    recover_rewind(root, log=log)
    files = later_files(root, iteration)
    if not any(not copy for _, copy in files):
        return None
    stamp = time.strftime('%Y%m%d_%H%M%S') + '-' + uuid.uuid4().hex[:8]
    aside = root / 'interrupted' / f'rewind_to_it_{iteration}-{stamp}'
    moved = [dict(file=str(path.relative_to(root)), kept_in_place=copy, sha256=digest(path))
             for path, copy in files]
    networks = sorted({int(m.group(1)) for entry in moved
                       if (m := re.fullmatch(r'trained_models/model_it_(\d+)\.keras', entry['file']))})
    record = dict(
        schema=1, status='pending', to_iteration=int(iteration), time=time.strftime('%Y-%m-%d %H:%M:%S'), networks_set_aside=networks,
        files=moved, note='The files of the iterations after the one the run was started again at, under the '
                          'paths they had. The true calls of those iterations stay in objective_calls/.')
    # Commit the entire move plan before moving anything, including phase records. A killed
    # rewind must never leave later models looking usable without their original commits.
    write_json(aside / 'REWIND.json', record)
    _finish_rewind(root, aside, record)
    log(f'--rewind: {sum(not e["kept_in_place"] for e in moved)} file(s) of the iterations after {iteration} '
        + (f'(networks {networks[0]} to {networks[-1]}) ' if networks else '')
        + f'are set aside in {aside.relative_to(root)}/, under the paths they had; nothing is deleted. '
        'Their true calls stay counted in objective_calls/.')
    return aside


# Written by the analysis commands, which do not hold the run lock: between a rewind that was
# killed and the start that finishes it, a report can have been made again or a stale file
# removed by hand.
_REWRITTEN_BY_ANALYSES = ('analysis', 'evidence', 'modes', 'posterior_chains')


def _written_without_the_lock(name):
    """Whether a file is one a command writes without the run lock: the analyses, and what
    `client.py augment` adds (never the marker of a first training, which the loop writes).
    One that appears after a rewind's plan was written joins the plan; any other file does not."""
    parts = Path(name).parts
    return parts[0] in _REWRITTEN_BY_ANALYSES or (parts[0] == 'augmentation' and not name.endswith('.trained.json'))


def _finish_rewind(root, aside, record):
    """Carry a recorded rewind to its end: every file of the plan is checked first, and
    nothing is moved while one of them stops it. Returns notes on what was not as planned.

    What the run committed (networks, training sets, their records) must be as the plan has
    it, in the run or in the folder; anything else stops the start, and the message says what
    can be done. What the analysis commands write is taken as it is found: a file written
    again since the plan is set aside as it is now, one that is gone is noted. A start that
    refused those could never be got past: such a file cannot be put back as it was.
    """
    # Refuse a checkpoint omitted from a manually edited plan before moving anything.
    # Otherwise automatic start selection can pick that later model after a "successful"
    # rewind. What the commands without the run lock wrote since the plan (an analysis, an
    # augmentation) joins it and is set aside with the rest; no phase commits such a file.
    planned = {entry['file'] for entry in record['files']}
    for path, copy in later_files(root, record['to_iteration']):
        name = str(path.relative_to(root))
        if not copy and name not in planned:
            if not _written_without_the_lock(name):
                raise RuntimeError(f'Interrupted rewind: {name} is not in the pending plan: it was written after the '
                                   'plan, or its entry was taken out. Move it to a separate folder under interrupted/ '
                                   'before retrying; a file of a later iteration must not remain active after a rewind.')
            record['files'].append(dict(file=name, kept_in_place=False, sha256=digest(path)))
    actions, notes = [], list(record.get('not_as_planned', []))
    for entry in record['files']:
        name = entry['file']
        source, saved = root / name, aside / name
        here = digest(source) if source.exists() else None
        there = digest(saved) if saved.exists() else None
        lenient = Path(name).parts[0] in _REWRITTEN_BY_ANALYSES
        # an analysis written again since the plan is expected as it was recorded before its move
        expected = entry.get('archived_sha256', entry['sha256']) if lenient else entry['sha256']

        def stop(what):
            return RuntimeError(
                f'Interrupted rewind: {name} {what}. The plan is {aside / "REWIND.json"} (nothing was moved by this '
                f'start). Put the file back as the plan has it (sha256 {expected[:12]}...), or, when it cannot be '
                'had again, first move every remaining copy of this file into a separate folder under interrupted/. '
                'Only then remove its entry from "files" in the plan and retry; never leave a later checkpoint active.')

        if there is not None and there != expected:
            raise stop('was set aside and its copy there has changed')
        if entry['kept_in_place']:
            if here != entry['sha256']:
                raise stop('is to stay in the run and is missing or changed')
            if there is None:
                actions.append(('copy', source, saved))
        elif there is not None:
            if here is None:
                continue
            if here == there:
                # a move between file systems that was cut after its copy: the copy is whole
                actions.append(('drop', source, saved))
            elif lenient:
                actions.append(('move', source, saved.with_name(f'{saved.name}.written-after-{uuid.uuid4().hex[:8]}')))
                notes.append(f'{name} was written again after it had been set aside; the later one is kept beside it')
            else:
                raise stop('is in the run again, and not as it was set aside')
        elif here == expected:
            actions.append(('move', source, saved))
        elif lenient and here is not None:
            # Keep the original hash, and durably record the replacement BEFORE its move.
            entry['archived_sha256'] = here
            actions.append(('move', source, saved))
            notes.append(f'{name} was written again since the plan; set aside as it is now')
        elif lenient:
            # Also when an earlier start recorded it as written again and was cut before the
            # move: that note and that hash are of a file nobody set aside.
            entry.pop('archived_sha256', None)
            notes = [note for note in notes if note != f'{name} was written again since the plan; set aside as it is now']
            notes.append(f'{name} was removed since the plan; nothing to set aside')
        else:
            raise stop('is missing' if here is None else 'has changed')
    notes = list(dict.fromkeys(notes))
    record = {key: value for key, value in record.items() if key != 'not_as_planned'}
    if notes:
        record['not_as_planned'] = notes
    write_json(aside / 'REWIND.json', record)  # write-ahead record, also on every recovery
    for kind, source, saved in actions:
        saved.parent.mkdir(parents=True, exist_ok=True)
        if kind == 'copy':
            with atomic_path(saved) as temporary:
                shutil.copy2(source, temporary)
        elif kind == 'drop':
            source.unlink()
        else:
            shutil.move(str(source), str(saved))
    write_json(aside / 'REWIND.json', dict(record, status='complete'))
    return notes


def recover_rewind(root, log=print):
    """Finish any recorded rewind before phase recovery or choosing the start iteration.

    Call holding the run lock. Original records without a status describe completed
    rewinds and are left alone. Every original artifact and true-call journal is preserved.
    """
    root = Path(root)
    recovered = []
    for path in sorted(root.glob('interrupted/rewind_to_it_*/REWIND.json')):
        record = json.loads(path.read_text())
        if record.get('status') != 'pending':
            continue
        notes = _finish_rewind(root, path.parent, record)
        recovered.append(record['to_iteration'])
        log(f'Finished interrupted rewind to iteration {record["to_iteration"]}; '
            f'original files remain in {path.parent.relative_to(root)}/.'
            + ''.join(f'\n   {note}' for note in notes))
    return recovered


class PhaseStore:
    """A phase is usable only after its last artifact is committed.

    A phase that did not reach its commit (a job at its time limit, a failed evaluation) is
    never taken as done and nothing of it is replayed: `recover` sets its files aside and the
    run does that phase again from what the phases before it committed. The true evaluations
    it had made stay in the objective_calls journals.
    """
    def __init__(self, directory):
        self.root = Path(directory)
        self.directory = self.root / 'phases'
        self.directory.mkdir(exist_ok=True)

    def check(self):
        for path in self.directory.glob('*.json'):
            state = json.loads(path.read_text())
            if state['status'] != 'complete':
                raise RuntimeError(f'Incomplete phase {state["phase"]}: {path}. '
                                   'A continued run sets such a phase aside and does it again '
                                   '(PhaseStore.recover); it is never taken as done.')
            for relative, expected in state['artifacts'].items():
                artifact = self.root / relative
                if not artifact.exists() or digest(artifact) != expected:
                    raise RuntimeError(f'Committed phase artifact changed or missing: {artifact}')

    def recover(self, log=print):
        """Set aside what interrupted phases left, so that the run redoes them.

        Call it holding the run lock: with the lock free, the process that began the phase
        is gone. For every training or acquisition phase without a commit, the files it may
        have written (phase_outputs) are moved to RUN/interrupted/<phase>-<time>/ under the
        paths they had, and the marker goes to phases/history/ with the list. Nothing is
        deleted. A network saved before its audit was committed is among those files, so it
        is trained and audited again instead of being taken as audited. Returns the phases
        recovered. An unfinished initial design, and a phase of a kind not known here, are
        refused: there is nothing committed to redo them from.
        """
        recovered = []
        for path in sorted(self.directory.glob('*.json')):
            state = json.loads(path.read_text())
            if state['status'] == 'complete':
                continue
            phase = state['phase']
            kind = phase.rpartition('_it_')[0]
            if kind not in ('training', 'acquisition'):
                raise RuntimeError(
                    f'Incomplete phase {phase}: {path}. '
                    + ('The initial design of this run did not finish, so there is no training set to '
                       'continue from: start a new run. ' if phase == 'initial' else '')
                    + 'The evaluations it made are in the objective_calls journals.')
            stamp = time.strftime('%Y%m%d_%H%M%S') + '-' + uuid.uuid4().hex[:8]
            aside = self.root / 'interrupted' / f'{phase}-{stamp}'
            moved = []
            for file in phase_outputs(self.root, phase):
                relative = file.relative_to(self.root)
                (aside / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file), str(aside / relative))
                moved.append(str(relative))
            history = self.directory / 'history'
            history.mkdir(exist_ok=True)
            write_json(history / f'{phase}-{stamp}.json',
                       dict(state, status='interrupted', set_aside=moved,
                            set_aside_in=str(aside.relative_to(self.root)) if moved else None))
            path.unlink()
            log(f'Phase {phase} did not finish in the job that began it: it is done again. '
                + (f'{len(moved)} file(s) it left are kept in {aside.relative_to(self.root)}/'
                   if moved else 'It left no file')
                + '; its evaluations are in objective_calls/.')
            recovered.append(phase)
        return recovered

    def begin(self, phase):
        path = self.directory / f'{phase}.json'
        if path.exists():
            # Intentional retraining may replace a completed phase, after check().
            history = self.directory / 'history'
            history.mkdir(exist_ok=True)
            shutil.copy2(path, history / f'{phase}-{uuid.uuid4().hex}.json')
        write_json(path, dict(phase=phase, status='incomplete', pid=os.getpid()))

    def complete(self, phase, artifacts):
        write_json(self.directory / f'{phase}.json', dict(
            phase=phase, status='complete', artifacts={
                str(Path(p).relative_to(self.root)): digest(p) for p in artifacts if Path(p).exists()}))
