"""Identity of a sampled surrogate target; old caches without identity are not reusable.

Native objective values include the run's own prior convention. An additive conversion
of an evidence to another reporting convention is recorded separately by that consumer.
"""
import json
from pathlib import Path
from uuid import uuid4
from zipfile import BadZipFile

import numpy as np

from run.storage import digest


def target_identity(run, iteration, metadata, region, lower=None, upper=None):
    """Describe these files; consumers must use load_snapshot to bind them to a model.

    Hashing a live run after sampling cannot identify the in-memory model that was used.
    This function also remains useful when inspecting immutable saved checkpoints.
    """
    run = Path(run)
    names = list(metadata.param_names)
    bounds = np.asarray([metadata.bounds[n] for n in names], float)
    return dict(schema=1, model_sha256=digest(run / 'trained_models' / f'model_it_{iteration}.keras'),
        metadata_sha256=digest(run / 'metadata.json'),
        names=names, domain=bounds.tolist(),
        lower=(bounds[:, 0] if lower is None else np.asarray(lower, float)).tolist(),
        upper=(bounds[:, 1] if upper is None else np.asarray(upper, float)).tolist(),
        region=region, temperature=1.0, objective='native saved-network output')


def read_target(saved):
    """Return a versioned identity from an NPZ; no inference from legacy booleans."""
    if 'target_json' not in saved:
        return None
    try:
        target = json.loads(str(saved['target_json'].item()))
    except (ValueError, TypeError, AttributeError):
        return None
    required = {'schema', 'model_sha256', 'metadata_sha256', 'names', 'domain',
                'lower', 'upper', 'region', 'temperature', 'objective'}
    return target if isinstance(target, dict) and required <= target.keys() and target['schema'] == 1 else None


def sampler_identity(region):
    """Sampling algorithm identity, separate from the mathematical target/evidence identity."""
    return dict(schema=1, version=1, method=('region-cell bridge mixture'
                if region is not None and len(region) > 1 else 'single ensemble'))


def read_sampling(saved):
    """Read sampling metadata; absence is legacy, malformed metadata is an error."""
    if 'sampling_json' not in saved:
        return None
    value = json.loads(str(saved['sampling_json'].item()))
    if not isinstance(value, dict) or not {'schema', 'version', 'method'} <= value.keys():
        raise ValueError('Malformed sampling metadata')
    return value


def sampling_correction_problem(info):
    """Why a mixture cannot support population corrections; None for reliable/legacy draws."""
    if info is None:
        return None
    if info.get('schema') != 1 or info.get('version') != 1:
        return 'unsupported sampling metadata'
    if info.get('method') == 'single ensemble':
        return None
    if info.get('method') != 'region-cell bridge mixture':
        return 'unknown posterior sampling method'
    from sampling.mixture import MIN_RELIABLE_ESS

    cells = info.get('cells')
    if not isinstance(cells, list) or not cells or not isinstance(info.get('unreliable_cells'), list):
        return 'missing region-cell reliability diagnostics'
    try:
        bad = {int(k) for k in info['unreliable_cells']}
        for cell in cells:
            efficiency = float(cell['efficiency'])
            if not np.isfinite(efficiency) or efficiency < MIN_RELIABLE_ESS:
                bad.add(int(cell['cell']))
    except (KeyError, TypeError, ValueError, OverflowError):
        return 'invalid region-cell reliability diagnostics'
    return f'unreliable region-cell mixture weights (cells {sorted(bad)})' if bad else None


def cached_samples(path, target, *, sampler=None, diagnostics=None):
    """None for incompatible chains. Multi-cell consumers must supply the sampler identity.

    Old single-ensemble caches remain usable for single-cell targets. Old multi-cell caches
    are never evidence that the corrected cell-mixture sampler was used. Diagnostics travel
    with compatible caches, including their unreliability flags.
    """
    if not Path(path).exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as saved:
            if read_target(saved) != target or saved['names'].tolist() != target['names']:
                return None
            info = read_sampling(saved)
            if sampler is not None:
                if info is None:
                    if sampler['method'] != 'single ensemble':
                        return None
                elif any(info.get(k) != v for k, v in sampler.items()):
                    return None
            x = saved['x']
            if x.ndim != 2 or x.shape[1] != len(target['names']) or not len(x) or not np.isfinite(x).all():
                return None
            if diagnostics is not None:
                diagnostics.update(info or dict(sampler or {}, legacy_cache=True))
            return x
    except (OSError, ValueError, TypeError, AttributeError, KeyError, EOFError, BadZipFile):
        return None


def save_samples(path, x, names, target, *, sampling=None, **extra):
    """Publish atomically, preserving replaced caches of another target or sampling method.

    Repeated samples of the same target and method replace their cache.
    """
    path = Path(path)
    temp = path.with_name(f'.{path.stem}.{uuid4().hex}.npz')
    sampler = None if sampling is None else {k: sampling[k] for k in ('schema', 'version', 'method')}
    if sampling is not None:
        extra['sampling_json'] = json.dumps(sampling, sort_keys=True)
    np.savez_compressed(temp, x=np.asarray(x, np.float32), names=np.asarray(names),
                        target_json=json.dumps(target, sort_keys=True), **extra)
    if path.exists() and cached_samples(path, target, sampler=sampler) is None:
        path.rename(path.with_name(f'{path.stem}.previous-{uuid4().hex}.npz'))
    temp.replace(path)
