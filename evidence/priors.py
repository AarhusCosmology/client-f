"""The prior an evidence is quoted for, per backend, and the conversion to it.

BEST samples a uniform prior on the sampling box B and returns ln Z_box, with
Z_box = (1/V_B) INT_B exp(t) dx, where t is what the network learned. What t contains
depends on the backend, and so does what a "Bayesian evidence" for the run means:

  Cobaya        t = ln L + ln pi: Cobaya's log-posterior, its priors normalised (uniform
                over [min, max], Gaussian, ...). Z = INT L pi dx = V_B Z_box, provided B holds
                the posterior mass. This is the number Cobaya's own nested samplers report.
  MontePython   t = ln L as compute_lkl returns it: the likelihoods plus MontePython's
                Gaussian nuisance priors, unnormalised as MontePython writes them. Priors
                are uniform over each varying parameter's [min, max] in the .param file,
                so Z = V_B Z_box / PROD_i range_i. Where the .param leaves a side unbounded
                (None, usual for cosmological parameters), the run's domain is used.

A user range replaces the uniform range of a parameter in either backend (for Cobaya only
where Cobaya's prior is uniform; its density is renormalised). A range narrower than the
box shrinks the box to it. Nothing here imports MontePython, CLASS, CAMB or builds a Cobaya
model: the ranges come from the run's copy of the input file.
"""

import ast
import math
import re
from pathlib import Path

import yaml

_MP_PARAMETER = re.compile(r"^\s*data\.parameters\[(?P<key>[^\]]+)\]\s*=\s*(?P<value>\[.*\])")


def montepython_ranges(param_file, scales):
    """{name: (lower, upper)} from a MontePython .param file, in CLiENT's physical units
    (MontePython stores value/scale, as the wrapper undoes); None for an unbounded side.
    `scales` maps the run's parameter names to their scale."""
    ranges = {}
    for line in Path(param_file).read_text().splitlines():
        match = _MP_PARAMETER.match(line.split("#", 1)[0])
        if not match:
            continue
        name = ast.literal_eval(match.group("key").strip())
        if name not in scales:
            continue
        entry = ast.literal_eval(match.group("value"))
        scale = scales[name]
        lower, upper = (None if v is None else float(v) * scale for v in entry[1:3])
        ranges[name] = (lower, upper)
    missing = sorted(set(scales) - set(ranges))
    if missing:
        raise ValueError(f"{Path(param_file).name} has no entry for {', '.join(missing)}")
    return ranges


def cobaya_ranges(info_file, names):
    """{name: (lower, upper)} where Cobaya's prior on a sampled parameter is uniform, and
    {name: None} where it is not (Gaussian and other distributions)."""
    params = yaml.safe_load(Path(info_file).read_text()).get("params", {})
    out = {}
    for name in names:
        prior = (params.get(name) or {}).get("prior") if isinstance(params.get(name), dict) else None
        if isinstance(prior, (list, tuple)) and len(prior) == 2:
            out[name] = (float(prior[0]), float(prior[1]))
        elif isinstance(prior, dict) and "min" in prior and "max" in prior and prior.get("dist", "uniform") == "uniform":
            out[name] = (float(prior["min"]), float(prior["max"]))
        elif isinstance(prior, dict) and prior.get("dist") == "uniform" and "scale" in prior:
            loc = float(prior.get("loc", 0.0))
            out[name] = (loc, loc + float(prior["scale"]))
        else:
            out[name] = None
    return out


def parse_ranges(text=None, file=None):
    """User ranges: "a=1:2,b=0.1:0.3" and/or a YAML file {name: [low, high]} (the text wins)."""
    ranges = {}
    if file is not None:
        data = yaml.safe_load(Path(file).read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{file} must map parameter names to [low, high]")
        for name, value in data.items():
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"{file}: {name} must be [low, high]")
            ranges[str(name)] = (float(value[0]), float(value[1]))
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, bounds = item.rpartition("=")
        low, colon, high = bounds.partition(":")
        if not sep or not colon or not name:
            raise ValueError(f"cannot read prior range {item!r}; use name=low:high")
        ranges[name.strip()] = (float(low), float(high))
    for name, (low, high) in ranges.items():
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise ValueError(f"invalid prior range for {name}: [{low}, {high}]")
    return ranges


def resolve(backend, names, domain, box, backend_ranges, user=None):
    """The prior ranges and the box the evidence is computed on.

    Returns (box, table, ln_correction): the box (lower, upper lists) after shrinking to
    any narrower user range; per parameter, the prior range used and where it came from;
    and ln Z_backend - (ln Z_box + ln V_box).
    """
    user = dict(user or {})
    unknown = sorted(set(user) - set(names))
    if unknown:
        raise ValueError(f"prior ranges given for unknown parameters: {', '.join(unknown)}")
    lower, upper = [float(v) for v in box[0]], [float(v) for v in box[1]]
    table, correction = [], 0.0
    for i, name in enumerate(names):
        row = dict(name=name)
        if backend == "cobaya":
            cobaya = backend_ranges.get(name)
            if name in user:
                if cobaya is None:
                    raise ValueError(f"{name}: Cobaya's prior is not uniform, so a range cannot replace it; "
                                     "change it in the Cobaya input instead")
                lo, hi = user[name]
                correction += math.log(cobaya[1] - cobaya[0]) - math.log(hi - lo)
                row.update(lower=lo, upper=hi, source="user (replaces Cobaya's uniform prior)")
            elif cobaya is None:
                row.update(lower=None, upper=None, source="Cobaya prior (not uniform), normalised by Cobaya")
            else:
                row.update(lower=cobaya[0], upper=cobaya[1], source="Cobaya prior")
        elif backend == "montepython":
            if name in user:
                lo, hi = user[name]
                source = "user"
            else:
                param_lo, param_hi = backend_ranges[name]
                lo = domain[0][i] if param_lo is None else param_lo
                hi = domain[1][i] if param_hi is None else param_hi
                sides = [side for side, bound in (("lower", param_lo), ("upper", param_hi)) if bound is None]
                source = ".param" if not sides else ("run domain" if len(sides) == 2 else
                                                     f".param ({sides[0]}: run domain)")
            correction -= math.log(hi - lo)
            row.update(lower=lo, upper=hi, source=source)
        else:
            raise ValueError(f"evidence supports the cobaya and montepython wrappers, not {backend!r}")
        if name in user:
            lo, hi = user[name]
            if hi <= lower[i] or lo >= upper[i]:
                raise ValueError(f"{name}: user range [{lo}, {hi}] does not overlap the box "
                                 f"[{lower[i]}, {upper[i]}]")
            lower[i], upper[i] = max(lower[i], lo), min(upper[i], hi)
        table.append(row)
    return (lower, upper), table, correction


def changed_ranges(table):
    """One line per parameter whose prior range is not the backend's own as written. A Cobaya
    prior that is not uniform is the backend's own and has no range, so it is not listed."""
    return [f"{r['name']} [{r['lower']:.6g}, {r['upper']:.6g}] ({r['source']})" for r in table
            if r["lower"] is not None and r["source"] not in ("Cobaya prior", ".param")]
