"""Read append-only journals without discarding intact calls after a torn final write."""
import hashlib
import json
import warnings
from pathlib import Path


def records(path, quarantine=False):
    path = Path(path)
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            item = json.loads(line)
            if not isinstance(item, dict) or 'event' not in item:
                raise ValueError('missing journal event')
        except (ValueError, UnicodeDecodeError) as exc:
            if i != len(lines) - 1 or line.endswith(b'\n'):
                raise ValueError(f'{path}: corrupt journal record {i + 1}') from exc
            digest = hashlib.sha256(line).hexdigest()
            if quarantine:
                sidecar = path.with_suffix('.truncated.json')
                sidecar.write_text(json.dumps(dict(line=i+1, sha256=digest, fragment_hex=line.hex()), indent=2))
            warnings.warn(f'{path}: preserved truncated final record; reserving one uncertain attempt', RuntimeWarning)
            yield dict(event='truncated', stage='recovery/uncertain', sha256=digest)
            return
        yield item
