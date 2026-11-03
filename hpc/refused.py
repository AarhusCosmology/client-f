"""A point that the Cobaya model refuses with a logged error is a refused point: -inf.

With `stop_at_error: false` Cobaya turns an error inside a theory code or a likelihood into
a log-posterior of -inf, with one exception: its own "logged" errors always stop. One of
them depends on the point, not on the set-up. The DES Y1 likelihood needs P(k) up to
k = (l + 1/2) / chi(z) for l = 50 000 at its lowest redshift, Cobaya's interpolator
extrapolates to k = 3000 / Mpc and no further, and for H0 above about 90 the distance is
short enough to pass that ("Not possible to extrapolate to k=..."). A uniform design over a
box that reaches such cosmologies met it at 3% of its points and CLiENT, which does not
expect an exception from an evaluation, stopped the run (job 6842216).

refuse_logged_errors(truth) makes the likelihood object return -inf (and no derived values)
for such a point, so that every path that evaluates the truth (design, injection, batches,
boosts, the audit) treats it as it treats a point CAMB refuses. Each rank prints the first
error of every kind it meets. Programming errors (TypeError, KeyError, ...) still stop.
A log-posterior of NaN is a refused point too (-inf, or the refused value below). Cobaya passes
one on when a theory code returns spectra that are not finite: CLASS++ did at 0.02-0.09% of the
points of the SIDR + DRMD runs (delta_Neff < 0.05 with f_idm 0.1-0.8), points the same commit
computes normally on another machine. Passed on as NaN, CLiENT counts such a
point as one without a value, and a boost or a true check that meets one withholds every corrected
mode weight; as a refused point it has weight zero there, and is left out of training when the
setting below is unset.

CLIENT_REFUSED_VALUE (a number, in a run's job.env) goes one step further: every refused
point, whether it came back as -inf or as a logged error, gets that finite value instead and
so becomes a training point. Without it CLiENT drops refused points, the network learns
nothing where the likelihood cannot be evaluated, and where that is a quarter of the box it
puts its highest maxima there: the tempered chain follows, 92% of the first batch of job
6842481 was refused, and the loop made no progress. The value should lie below every value
the likelihood really takes in the box, so that a refused region is the worst there is; it is
an assumption about the run's objective (refused means hopeless), recorded with the run.
Unset: refused points stay -inf and are dropped.

How low the value should be depends on the box, so a number chosen for one box is wrong for
the next: -250 000 was the median of one design and eight times the median of the same run in
a narrower box, where more than half of the first audit then went into refused regions.
CLIENT_REFUSED_VALUE=design takes it from the run's own design instead: the median of the
design's finite values, rounded down to three significant digits (design_value).
CLIENT_REFUSED_VALUE=design:Q takes the quantile Q of them (0 < Q < 1; smaller is lower). The
launcher sets it once, after the design, writes it to the run (refused_value.json) and a
continued run reads it there; until it is set a refused point is -inf, and the launcher gives
the design's refused points the value before CLiENT sees them (launch.py).
"""
import json
import math
import os
import re
from pathlib import Path

import numpy as np


def refused_setting():
    """CLIENT_REFUSED_VALUE: None (unset: refused points stay -inf), a finite number, or
    ("design", Q): the quantile Q of the design's finite values."""
    text = (os.environ.get("CLIENT_REFUSED_VALUE") or "").strip()
    if not text:
        return None
    if text.lower().startswith("design"):
        _, colon, quantile = text.partition(":")
        quantile = float(quantile) if colon else 0.5
        if not 0 < quantile < 1:
            raise ValueError(f"CLIENT_REFUSED_VALUE=design:Q needs 0 < Q < 1, not {text!r}")
        return ("design", quantile)
    value = float(text)
    if not np.isfinite(value):
        raise ValueError(f"CLIENT_REFUSED_VALUE must be a finite number or design[:Q], not {text!r}")
    return value


def refused_value():
    """The fixed finite value of a refused point, or None (unset, or still to be read off a design)."""
    setting = refused_setting()
    return setting if isinstance(setting, float) else None


def design_value(values, quantile=0.5):
    """The refused value read off a design's true values: their quantile (finite ones only),
    rounded down to three significant digits. A typical bad point of the box: low enough for
    the tempered chain to stay away, not a cliff below everything the likelihood really takes.
    The rounding keeps it apart from every real value, which the journal and the report use
    to count refused points."""
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("the design has no finite value to take the refused value from")
    raw = float(np.quantile(finite, quantile))
    unit = 10.0 ** (math.floor(math.log10(abs(raw))) - 2) if raw != 0 else 1.0
    value = math.floor(raw / unit) * unit
    while np.any(finite == value):
        value -= unit
    return float(value), raw


def read_record(path):
    """The refused value a run was given (refused_value.json), or None when there is no such file."""
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):      # absent, or another rank is still writing it
        return None


def write_record(path, record):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(record, indent=1) + "\n")
    temporary.replace(path)


def refuse_logged_errors(truth, rank=0, label="[launch]", value_file=None):
    """value_file: where, with CLIENT_REFUSED_VALUE=design, this rank finds the value once it
    has been set (rank 0 writes it after the design)."""
    from cobaya.log import LoggedError

    loglkl, loglkl_and_derived = truth.loglkl, truth.loglkl_and_derived
    kinds = {}
    setting = refused_setting()
    known = {"value": setting if isinstance(setting, float) else None}

    def set_value(value):
        known["value"] = truth.refused_value = float(value)

    def settle(value):
        if np.isfinite(value):
            return value
        if np.isnan(value):            # a theory code's output that is not finite: a refused point too
            kinds["NaN"] = kinds.get("NaN", 0) + 1
            if kinds["NaN"] == 1:
                print(f"{label} rank {rank}: the likelihood returns NaN at a point (a theory code's output not finite); "
                      "it and later ones count as refused points", flush=True)
            value = -np.inf
        if known["value"] is None and value_file is not None and isinstance(setting, tuple):
            record = read_record(value_file)
            if record is not None:
                set_value(record["value"])
        return value if known["value"] is None else known["value"]

    def note(error):
        kind = re.sub(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", "#", str(error))[:160]
        kinds[kind] = kinds.get(kind, 0) + 1
        if kinds[kind] == 1:
            print(f"{label} rank {rank}: the likelihood refuses a point with a logged error; it and later ones "
                  f"of this kind count as -inf: {error}", flush=True)

    def tolerant_loglkl(x):
        try:
            return settle(loglkl(x))
        except LoggedError as error:
            note(error)
            return settle(-np.inf)

    def tolerant_loglkl_and_derived(x):
        try:
            value, derived = loglkl_and_derived(x)
            return settle(value), derived
        except LoggedError as error:
            note(error)
            return settle(-np.inf), np.full(truth.n_derived, np.nan)

    truth.loglkl, truth.loglkl_and_derived = tolerant_loglkl, tolerant_loglkl_and_derived
    truth.refused_kinds = kinds
    truth.refused_setting = setting
    truth.refused_value = known["value"]
    truth.set_refused_value = set_value
    return truth
