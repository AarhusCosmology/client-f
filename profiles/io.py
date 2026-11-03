"""Atomic JSON artifacts, without importing a numerical backend."""

import json
import numpy as np

from run.storage import atomic_path


def save(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)

    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n")
