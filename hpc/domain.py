"""Explicit numerical domain of a run; Cobaya's prior densities remain unchanged.

The training box of CLiENT is a hard prior for the whole run. Here it is given parameter by
parameter in runs/<RUN>/domain_bounds.json, instead of as a number of
sigmas around a reference point.
"""
import json
import math
from pathlib import Path


def read_domain(path):
    bounds = json.loads(Path(path).read_text())
    return {name: (float(lo), float(hi)) for name, (lo, hi) in bounds.items()}


def apply_domain(truth, bounds):
    if set(bounds) != set(truth.param_names):
        raise ValueError(
            "Domain must specify every sampled parameter exactly once: "
            f"missing {sorted(set(truth.param_names) - set(bounds))}, unknown {sorted(set(bounds) - set(truth.param_names))}")
    original = truth.prior_bounds
    checked = {}
    for name in truth.param_names:
        lo, hi = map(float, bounds[name])
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError("Invalid finite domain: " + name)
        prior_lo, prior_hi = original[name]
        if (prior_lo is not None and lo < prior_lo) or (prior_hi is not None and hi > prior_hi):
            raise ValueError("Domain extends outside prior support: " + name)
        checked[name] = (lo, hi)
    truth._effective_bounds = checked
    return checked
