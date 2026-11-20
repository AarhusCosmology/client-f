"""Data layer for the four-model note: exact, Galerkin m6, our RTA, Wong et al.

Sits ON TOP of `nb/dncdm.py` rather than replacing it. Everything about rounds,
column POSITION, round precedence, the pre-#413 table screen and the Richardson
error bar is inherited; this module adds exactly one thing, the `proxy_rta`
round, and the model bookkeeping that goes with it.

────────────────────────────────────────────────────────────────────────────────
THE ONE THING A READER MUST KNOW ABOUT PROVENANCE

The reference cells (exact, m6) were run on the CLUSTER on the campaign branch;
the proxy cells were run on the LAPTOP on master (85deb9fc). That is two
compilers, two machines and ten commits of drift, so a proxy-vs-exact residual
carries a floor that is NOT the closure error. `systematic_floor()` states what
is known about it and `SYSTEMATIC_NOTE` says how it was bounded. Any residual
quoted below ~0.1% should be read as consistent-with-zero, not as a measurement.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # paper/dncdm.py
import dncdm as D  # noqa: E402

HARNESS = D.HARNESS

# ── register the proxy round with the inherited machinery ────────────────────
# exact_ok/reduced_ok describe a round's DNCDM cells; a proxy round has neither
# kind, so both are False and nothing in dncdm.py will pick these up by accident.
D.ROUNDS["rta"] = D.Round(
    "rta", "proxy_rta", False, False,
    "the proxy (RTA) round for this note. One binary, master 85deb9fc, laptop.")

PROXY_DIR = HARNESS / "proxy_rta"
PX_RE = re.compile(r"^px_(?P<model>[a-z0-9]+)_G(?P<gl>[\d.]+)_m(?P<mass>[\d.]+)$")

# ── the models ───────────────────────────────────────────────────────────────
# Order is plot order. `ref` names the run family; `label` is the legend string.
MODELS = {
    "x":    dict(label=r"exact",              kind="ref"),
    "m6":   dict(label=r"Galerkin $m_6$",     kind="ref"),
    "ours":  dict(label=r"RTA (this work)",   kind="proxy"),
    "copw": dict(label=r"RTA, Chen et al.",         kind="proxy"),
    "ur1":  dict(label=r"RTA $+\,0.05\,N_{\rm ur}$", kind="proxy"),
    "ur2":  dict(label=r"RTA $+\,0.10\,N_{\rm ur}$", kind="proxy"),
    "c5a":  dict(label=r"RTA, $1.5\,C_5$",           kind="proxy"),
}
UR_DELTA = {"ur1": 0.05, "ur2": 0.10}

# The C5 probe, in the same units the kernel is differentiated in: the FRACTIONAL
# change in the NLO amplitude, so the fitted coefficient reads directly as the
# fractional recalibration C5 would need.
C5_BASE = 0.5283
C5_DELTA = {"c5a": 0.5}


def proxy_cell(model: str, gl: str, mass: str) -> D.Cell | None:
    """A dncdm.Cell for a proxy run, or None when that cell is not on disk.

    Built by hand rather than through dncdm._parse because TAG_RE knows nothing
    about `px_` tags -- and should not: teaching it would make proxy cells show
    up in `ladder()` and `cells()`, where a run with no daughter hierarchy at all
    has no business appearing next to exact ones.
    """
    tag = f"px_{model}_G{gl}_m{mass}"
    if D.find("rta", tag, "cl_lensed.dat", "cl.dat") is None:
        return None
    return D.Cell(round="rta", tag=tag, family="cmb", gl=gl,
                  gamma=10.0 ** float(gl), mass=mass, scheme="x", refine=1,
                  suffix="", table_ok=True, table_err=0.0)


def ref_cell(model: str, gl: str, mass: str) -> D.Cell | None:
    """The exact (`x`) or Galerkin (`m6`) cell at this rung, by round precedence."""
    return D.cell_at(mass, gl, scheme=model)


def cell(model: str, gl: str, mass: str) -> D.Cell | None:
    return (ref_cell if MODELS[model]["kind"] == "ref" else proxy_cell)(model, gl, mass)


def rungs(mass: str, model: str) -> list[str]:
    """Sorted gl strings where this model has output."""
    if MODELS[model]["kind"] == "proxy":
        out = set()
        for f in (PROXY_DIR / "out").glob(f"px_{model}_G*_m{mass}_cl_lensed.dat"):
            m = PX_RE.match(f.name.split("_cl_lensed")[0])
            if m:
                out.add(m["gl"])
        return sorted(out)
    return [c.gl for c in D.ladder(scheme=model, mass=mass)]


def common_rungs(mass: str, models: list[str]) -> list[str]:
    got = [set(rungs(mass, m)) for m in models]
    return sorted(set.intersection(*got)) if got else []


# ── the comparison metric ────────────────────────────────────────────────────
def resid(model: str, gl: str, mass: str, ref: str = "x", key="TT", lensed=True):
    """(ell, C_l[model]/C_l[ref] - 1). None when either cell is missing.

    TT and EE only -- dncdm.cl_ratio asserts on anything that crosses zero.
    """
    a, b = cell(ref, gl, mass), cell(model, gl, mass)
    if a is None or b is None:
        return None, None
    return D.cl_ratio(a, b, key=key, lensed=lensed)


def resid_pk(model: str, gl: str, mass: str, ref: str = "x", z=1, cb=False):
    a, b = cell(ref, gl, mass), cell(model, gl, mass)
    if a is None or b is None:
        return None, None
    return D.pk_ratio(a, b, z=z, cb=cb)


def rms(ell, y, lo=2, hi=2500) -> float:
    """Band rms of a residual. The note quotes rms, not max: a max over 2500
    multipoles is dominated by whichever ell happens to sit on a lensing wiggle."""
    m = (ell >= lo) & (ell <= hi) & np.isfinite(y)
    return float(np.sqrt(np.mean(y[m] ** 2))) if m.any() else float("nan")


# ── the systematic floor ─────────────────────────────────────────────────────
SYSTEMATIC_NOTE = """\
Proxy cells: laptop, master 85deb9fc, Apple clang, -ffast-math.
Reference cells: cluster, campaign branch (6bc5f66e / 46f05bee), GCC.
Bounded by (a) an l_max_dr doubling on the proxy side, which tests whether the
#420 closure change can matter at all at the l_max_dr = 17 these runs use, and
(b) the m6-vs-exact agreement at rungs where both exist, which carries the same
cross-machine floor and no closure difference."""


def systematic_floor(mass: str = "0.3", key="TT", lo=2, hi=2500) -> dict:
    """m6-vs-exact rms per rung: an upper bound on what cross-machine drift plus
    the Galerkin closure can produce. The proxy residual must be read against it."""
    out = {}
    for gl in common_rungs(mass, ["x", "m6"]):
        ell, r = resid("m6", gl, mass, ref="x", key=key)
        if ell is not None:
            out[gl] = rms(ell, r, lo, hi)
    return out


def provenance_rows(mass: str, models: list[str]) -> list[tuple]:
    """(model, gl, round, tag) for every cell a figure would draw. Print it."""
    rows = []
    for mdl in models:
        for gl in rungs(mass, mdl):
            c = cell(mdl, gl, mass)
            if c is not None:
                rows.append((mdl, gl, c.round, c.tag))
    return rows
