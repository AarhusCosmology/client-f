"""Shared plotting decisions for the note's figures.

Kept in one place so a colour or a band edge cannot drift between figures --
the four models must be the same four colours everywhere, or the reader has to
re-read the legend on every page.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pstyle as P
import rta

FIG = Path(__file__).resolve().parents[1] / "fig"      # paper/fig
FIG.mkdir(exist_ok=True)

# ── the four models, fixed colours ───────────────────────────────────────────
# exact is INK because it is the reference, not a series; the three approximations
# take categorical slots 1/2/6 (blue / orange / green), which the palette's own
# validator passes against each other and against ink.
STYLE = {
    "x":    dict(c=P.INK,     ls="-",  lw=1.5, label=r"full solution"),
    "m6":   dict(c=P.CAT[0],  ls="-",  lw=1.3, label=r"six moments"),
    "ours": dict(c=P.CAT[1],  ls="-",  lw=1.3, label=r"proxy (this work)"),
    "copw": dict(c=P.CAT[5],  ls="--", lw=1.3, label=r"proxy with the rate of Chen et al."),
    "ur1":  dict(c=P.CAT[4],  ls=":",  lw=1.2, label=r"RTA $+\,\Delta N_{\rm ur}$"),
    "ur2":  dict(c=P.CAT[6],  ls=":",  lw=1.2, label=r"RTA $+\,2\Delta N_{\rm ur}$"),
}

MASSES = ["0.3", "0.06"]
MASS_LABEL = {"0.3": r"$m_{\nu_H}=0.3$ eV", "0.06": r"$m_{\nu_H}=0.06$ eV"}

# Bands the note quotes rms in. Chosen on the physics, not on round numbers:
# 2-30 is where the reduced scheme's unexplained low-ell dip lives, 30-300 spans
# the first two acoustic peaks, 300-1500 is the damping tail the free-streaming
# daughters actually control.
BANDS = [(2, 30), (30, 300), (300, 1500)]
FULL = (2, 2500)


def smooth(y, n=9):
    """Boxcar over ell. Residuals carry the lensing wiggle at the 1e-4 level and
    the note is about a 1e-3-1e-2 effect; without this the reader sees texture
    that is not the signal. n is odd so the window is centred."""
    if n <= 1 or y.size < n:
        return y
    k = np.ones(n) / n
    pad = n // 2
    return np.convolve(np.pad(y, pad, mode="edge"), k, mode="valid")


def ref_for(gl: str, mass: str) -> str:
    """`x` while an exact cell exists at this rung, `m6` otherwise.

    This is the note's central bookkeeping decision and it is made in ONE place.

    ⚠ The exact cells sit at INTEGER rungs only (plus 10.5 at 0.3 eV), while the
    Galerkin ladder is on half decades. So the reference does not switch once at
    the ceiling -- it ALTERNATES all the way up the ladder, and because the RTA
    and m6 agree with each other several times better than either agrees with
    exact, a series plotted over every rung zigzags by a factor of three with the
    reference rather than with the physics. Any figure that draws one line
    through consecutive rungs must use `exact_rungs`/`above_ceiling` below, not
    this function alone.
    """
    return "x" if rta.cell("x", gl, mass) is not None else "m6"


def exact_rungs(mass: str, model: str) -> list[str]:
    """Rungs where `model` has run AND an exact reference exists.

    These are the only rungs at which a residual is a measurement against the
    exact solution; everything else is a measurement against m6.
    """
    return [gl for gl in rta.rungs(mass, model)
            if rta.cell("x", gl, mass) is not None]


def above_ceiling(mass: str, model: str) -> list[str]:
    """Rungs where `model` has run and lie ABOVE the top exact rung.

    Drawn separately and never joined to the exact-referenced series: they are
    bounds against m6, not measurements against exact.
    """
    xr = rta.rungs(mass, "x")
    if not xr:
        return []
    top = float(xr[-1])
    return [gl for gl in rta.rungs(mass, model) if float(gl) > top]


def err_band(gl: str, mass: str):
    """(ell, sigma) for the exact cell at this rung, or (None, None).

    dncdm.cl_error divides the refine pair by 2^p - 1 -- |f(2h) - f(h)| is the
    COARSE run's error, not the production cell's. order=2 is the conservative
    choice; the measured order is 3.3-4.7."""
    return rta.D.cl_error(mass, gl, order=2.0)
