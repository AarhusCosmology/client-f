"""Identity of the ordinary run's native objective and ordered domain.

What is recorded is what decides the value the backend returns at a point: the likelihood
input, the Boltzmann code that is loaded, the code of the likelihood components and of the
functions the input names, local modules of the objective, files the user lists, the wrapper's
declared convention, and the ordered domain. A continued run must match the record.

What is deliberately not recorded: the text of CLiENT's own wrapper files (a wrapper declares
its convention as a number, `BaseLikelihood.objective_convention`, raised when what it returns
changes; a comment edit is no change of the objective), unrelated launcher helpers, and
Python's own library. Imported local dependencies of an objective are recorded even when
they live beside the launcher.

A difference stops the continuation and is listed entry by entry. When it is a reviewed update
of code or environment that leaves the objective as it was, `--accept-identity-change`
continues once: the old record goes to objective_identity_history/ with the differences, and
the new one says that it was accepted and when.
"""
import ast
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time
import warnings
import site
import uuid
from dataclasses import asdict
from run.storage import digest, write_json

SCHEMA = 2
BACKENDS = ('classy', 'camb', 'clik')


def _source(thing):
    try:
        file = inspect.getsourcefile(thing)
    except TypeError:
        file = None
    return Path(file).resolve() if file and Path(file).is_file() else None


def _launcher_directory():
    """The directory of the script Python was started with, or None (interactive, -c, -m)."""
    main = sys.modules.get('__main__')
    file = getattr(main, '__file__', None)
    return Path(file).resolve().parent if file else None


def _objective_dependencies(roots, is_local):
    """Loaded local imports reachable from objective source modules.

    Inspect import statements as well as module/function references: ``from helper import
    constant`` leaves only a scalar in the namespace. Never import or execute a dependency
    to fingerprint it. Dynamically loaded files not reachable here still belong in
    likelihood.identity_files; this is not a hermetic dependency archive.
    """
    pending = [inspect.getmodule(root) for root in roots]
    seen, files = set(), {}
    while pending:
        module = pending.pop()
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        path = _source(module)
        if path is None or not is_local(path):
            continue
        files[f'module:{module.__name__}'] = digest(path)
        # References also catch a function imported or assigned dynamically from another
        # local module. Walking modules is bounded by `seen` and the local-file predicate.
        for value in vars(module).copy().values():
            if inspect.ismodule(value):
                pending.append(value)
            elif inspect.isfunction(value) or inspect.isclass(value):
                pending.append(inspect.getmodule(value))
        try:
            tree = ast.parse(path.read_bytes())
        except (SyntaxError, UnicodeError, OSError):
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = '.' * node.level + (node.module or '')
                if node.level:
                    try:
                        name = importlib.util.resolve_name(name, module.__package__)
                    except (ImportError, ValueError):
                        continue
                names = [name] + [f'{name}.{alias.name}' for alias in node.names]
            for name in names:
                # Include package __init__ files, which can contribute constants too.
                parts = name.split('.')
                pending.extend(sys.modules.get('.'.join(parts[:i])) for i in range(1, len(parts) + 1))
    return files


def identity(likelihood, config):
    from likelihood.surrogate import SurrogateMetadata
    source = Path(config.likelihood.input).resolve()
    files = {'input': digest(source)}
    # Backend extension binaries and imported local objective modules. Not CLiENT's own
    # modules, not the launcher and what lies beside it, not large external data directories.
    repo = Path(__file__).resolve().parents[1]
    installed = [Path(sys.base_prefix).resolve()] + [Path(p).resolve() for p in site.getsitepackages()]
    def local_objective(path):
        return (not any(path.is_relative_to(p) for p in installed)
                and (not path.is_relative_to(repo) or path.is_relative_to(repo / 'input')))
    launcher = _launcher_directory()
    for name, module in sorted(sys.modules.copy().items()):
        path = getattr(module, '__file__', None)
        if not path or name in ('__main__', '__mp_main__'):
            continue
        path = Path(path).resolve()
        if not path.is_file():
            continue
        backend = name.split('.')[0] in BACKENDS
        outside = local_objective(path)
        beside_launcher = launcher is not None and path.parent == launcher and not path.is_relative_to(repo / 'input')
        if backend or (outside and not beside_launcher):
            files[f'module:{name}'] = digest(path)
    # CAMB computes in a shared library loaded through ctypes, which is no module.
    library = getattr(getattr(sys.modules.get('camb.baseconfig'), 'camblib', None), '_name', None)
    if library and Path(library).is_file():
        files['library:camb'] = digest(library)
    for i, extra in enumerate(config.likelihood.identity_files or []):
        files[f'explicit:{i}:{Path(extra).name}'] = digest(extra)
    model = getattr(likelihood, 'cobaya_model', None)
    objective_roots = []
    for section in ('likelihood', 'theory'):
        for name, component in getattr(model, section, {}).items():
            objective_roots.append(type(component))
            file = _source(type(component))
            if file:
                files[f'{section}:{name}'] = digest(file)
            # A likelihood given as a function (`external:` in the input): the function's file,
            # wherever the launcher lies.
            function = getattr(component, 'external_function', None)
            file = _source(function) if function is not None else None
            if file:
                files[f'external:{name}'] = digest(file)
                objective_roots.append(function)
    files.update(_objective_dependencies(objective_roots, local_objective))
    metadata = SurrogateMetadata.from_likelihood(likelihood)
    parameters = [{k: v for k, v in asdict(p).items() if k != 'label'} for p in metadata.parameters]
    return dict(schema=SCHEMA, wrapper=config.likelihood.wrapper,
                convention=int(getattr(likelihood, 'objective_convention', 1)), parameters=parameters,
                objective='native backend return; priors follow backend convention', files=files)


def differences(saved, current):
    """What differs between two identities, as short strings (empty: the same)."""
    out = []
    if saved.get('schema', 1) != current.get('schema', 1):
        out.append(f"schema: {saved.get('schema', 1)} -> {current.get('schema', 1)}; migration requires review")
    for key in ('wrapper', 'convention', 'objective'):
        if saved.get(key) != current.get(key):
            out.append(f'{key}: {saved.get(key)!r} -> {current.get(key)!r}')
    if saved.get('parameters') != current.get('parameters'):
        old = {p.get('name'): p for p in saved.get('parameters') or []}
        new = {p.get('name'): p for p in current.get('parameters') or []}
        changed = [n for n in new if n in old and new[n] != old[n]]
        if list(old) != list(new):
            out.append('parameters: other names or another order')
        if changed:
            out.append('parameters: bounds, scale or prior of ' + ', '.join(changed[:8])
                       + (f' and {len(changed) - 8} more' if len(changed) > 8 else ''))
    old, new = saved.get('files', {}), current.get('files', {})
    for label, keys in (('changed', [k for k in new if k in old and new[k] != old[k]]),
                        ('no longer loaded', [k for k in old if k not in new]),
                        ('new', [k for k in new if k not in old])):
        if keys:
            out.append(f'{label}: ' + ', '.join(keys[:8]) + (f' and {len(keys) - 8} more' if len(keys) > 8 else ''))
    return out


def verify(directory, current, *, new=False, accept_change=False):
    directory = Path(directory)
    path = directory / 'objective_identity.json'
    if path.exists():
        record = json.loads(path.read_text())
        saved = record['identity']
        changed = differences(saved, current)
        if changed and not accept_change:
            raise ValueError('Native objective/domain identity changed; refusing to mix this '
                             'target with saved training values (' + '; '.join(changed) + '). '
                             f'Compare with {path}. If the objective itself is unchanged (a reviewed '
                             'update of code or environment), continue once with '
                             '--accept-identity-change; otherwise start a separate run.')
        metadata = directory / 'metadata.json'
        if metadata.exists():
            recorded = json.loads(metadata.read_text())['parameters']
            domain = lambda parameters: [{k: p.get(k, 1.0 if k == 'scale' else None)
                                          for k in ('name', 'lower', 'upper', 'scale')}
                                         for p in parameters]
            if domain(recorded) != domain(current['parameters']):
                raise ValueError('Saved model metadata differs from the objective identity')
        if changed or record['identity'] != current:
            # An accepted change, or a record of an earlier schema: the old record is kept.
            history = directory / 'objective_identity_history'
            stamp = time.strftime('%Y%m%d_%H%M%S')
            write_json(history / f'objective_identity_{stamp}-{uuid.uuid4().hex}.json',
                       dict(record, replaced=stamp, differences=changed))
            accepted = list(record.get('accepted_changes', []))
            if changed:
                accepted.append(dict(time=stamp, differences=changed))
                warnings.warn('Objective identity change accepted (' + '; '.join(changed) + '); the earlier '
                              f'record is in {history.name}/', RuntimeWarning)
            write_json(path, dict(record, identity=current, accepted_changes=accepted))
        return
    if not new:
        raise ValueError('This run has no objective identity (it was made before CLiENT recorded one), '
                         'so its training values cannot be checked against the current objective: '
                         'start a new run.')
    write_json(path, dict(identity=current, historical_values_verified=True))
