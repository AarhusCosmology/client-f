"""Data access for the paper's figures.

Reads three sources on the same footing:

  * ``hpc_cmb``      -- the exact (psd) reference matrix
  * ``hpc_reduced``  -- the Galerkin-moment matrix
  * ``runs/out``     -- the proxy/RTA runs made for this paper

Two conventions are load-bearing and were each a bug first (they are inherited
from ``xreport/tools/xlib.py``):

  * column POSITION, never the printed number -- CLASS writes ``cl.dat`` with
    ``# 1:l  1:TT  2:EE ...``, so ``l`` and ``TT`` both claim column 1;
  * a SCALE-RELATIVE metric, never a pointwise ratio -- ``C_l^TE`` crosses zero
    and ``a^4 Pi`` oscillates ~1e3 times over a mode's history, so a ratio
    divides by a number that can sit 100x below the signal's own envelope.
"""
import gzip
import os
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

HARNESS = Path(os.environ.get("DNCDM_HARNESS") or Path(__file__).resolve().parents[1] / "campaigns")
HERE = Path(__file__).resolve().parent.parent
LOCAL = HERE / "runs" / "out"
CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)


def _open(p):
    return gzip.open(p, "rt") if str(p).endswith(".gz") else open(p)


def cols(path):
    """Column name -> 0-based POSITION (see module docstring)."""
    head = None
    with _open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if re.search(r"\d+:", line):
                head = line
    if head is None:
        raise ValueError(f"no header in {path}")
    names = re.findall(r"\d+:(.*?)(?=\s+\d+:|$)", head.rstrip())
    return {n.strip(): i for i, n in enumerate(names)}


def _dir(folder):
    return LOCAL if folder == "local" else HARNESS / folder / "out"


def find(folder, tag, *sufs):
    d = _dir(folder)
    for s in sufs:
        for c in (d / f"{tag}_{s}", d / f"{tag}_{s}.gz"):
            if c.exists() and c.stat().st_size:
                return c
    return None


# ── spectra ──────────────────────────────────────────────────────────────────
def cl(folder, tag, lensed=True):
    f = (find(folder, tag, "cl_lensed.dat", "cl.dat") if lensed
         else find(folder, tag, "cl.dat"))
    if f is None:
        return None
    idx = cols(f)
    d = np.loadtxt(_open(f))
    key = {k.strip().upper(): v for k, v in idx.items()}
    return d[:, 0], {k: d[:, v] for k, v in key.items() if k != "L"}


def pk(folder, tag, z=1):
    f = find(folder, tag, f"z{z}_pk.dat", "pk.dat")
    if f is None:
        return None
    d = np.loadtxt(_open(f))
    return d[:, 0], d[:, 1]


def scale_rel(x_ref, x, w=25):
    """|x - x_ref| / rolling-local-RMS(x_ref), floored at 1e-3 of global RMS."""
    from numpy.lib.stride_tricks import sliding_window_view
    p = np.pad(np.asarray(x_ref, float) ** 2, w, mode="edge")
    loc = np.sqrt(sliding_window_view(p, 2 * w + 1).mean(axis=-1))
    loc = np.maximum(loc, 1e-3 * np.sqrt(np.mean(np.asarray(x_ref, float) ** 2)))
    return np.abs(np.asarray(x, float) - np.asarray(x_ref, float)) / loc


def cl_ratio(ref, test, key="TT"):
    """test/ref - 1 on the common l range.  C_l^TT is positive, so a plain
    ratio is honest for TT specifically; never use this for TE."""
    lr, dr = ref
    lt, dt = test
    n = min(len(lr), len(lt))
    return lr[:n], dt[key][:n] / dr[key][:n] - 1.0


def pk_ratio(ref, test):
    kr, pr = ref
    kt, pt = test
    if len(kr) != len(kt) or not np.allclose(kr, kt, rtol=1e-9):
        f = PchipInterpolator(kt, pt, extrapolate=False)
        pt = f(kr)
        m = np.isfinite(pt)
        return kr[m], pt[m] / pr[m] - 1.0
    return kr, pt / pr - 1.0


# ── background ───────────────────────────────────────────────────────────────
def background(folder, tag):
    """-> (colname -> array).  Cached, because the exact matrix's background
    dumps are 20-80 MB each and there are thirteen of them."""
    f = find(folder, tag, "background.dat")
    if f is None:
        return None
    ck = CACHE / f"bg_{folder}_{tag}.npz"
    if ck.exists() and ck.stat().st_mtime > f.stat().st_mtime:
        z = np.load(ck)
        return {k: z[k] for k in z.files}
    idx = cols(f)
    d = np.loadtxt(_open(f))
    out = {k: d[:, i] for k, i in idx.items()}
    np.savez_compressed(ck, **out)
    return out


def bg_exact(G, m):
    """The background cells of the full solution's round (hpc_prod2): G = "1e5" reads bg_G05.0_m{m}."""
    return background("hpc_prod2", f"bg_G{int(G.split('e')[1]):02d}.0_m{m}")


# ── perturbations (k_output dumps) ───────────────────────────────────────────
PERT_WANT = ["tau [Mpc]", "a", "a4Pi_dncdm1", "a4Pi_dncdm1_H", "a4Pi_dncdm1_l",
             "a4Pi_dncdm1_phi", "delta_dncdm1", "psi", "phi", "delta_cdm",
             "shear_dncdm1"]


def pert(folder, tag, kidx, want=None):
    """-> (colname -> array) for one k_output mode.  Cached."""
    want = list(want or PERT_WANT)
    f = find(folder, tag, f"perturbations_k{kidx}_s.dat")
    if f is None:
        return None
    ck = CACHE / f"pt_{folder}_{tag}_k{kidx}.npz"
    if ck.exists() and ck.stat().st_mtime > f.stat().st_mtime:
        z = np.load(ck)
        return {k: z[k] for k in z.files}
    idx = cols(f)
    keep = {k: i for k, i in idx.items() if k in want}
    d = np.loadtxt(_open(f))
    out = {k: d[:, i] for k, i in keep.items()}
    np.savez_compressed(ck, **out)
    return out


def canonical_tau(k, lo, hi, per_period=128, dtau_max=3.0):
    """A tau grid fine enough to carry the oscillation, shared by every run.
    The rows of a k_output file are the evolver's own step endpoints, so no two
    runs share a grid."""
    dt = min(2 * np.pi / (k * per_period), dtau_max)
    n = int(np.ceil((hi - lo) / dt)) + 1
    return np.linspace(lo, hi, max(n, 64))


def resample(t, y, tg):
    """PCHIP, never a global cubic spline: CLASS restarts the evolver at every
    approximation-scheme boundary and a global spline rings across the
    step-size jump."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    o = np.argsort(t)
    t, y = t[o], y[o]
    keep = np.concatenate(([True], np.diff(t) > 0))
    return PchipInterpolator(t[keep], y[keep], extrapolate=False)(tg)


def envelope(t, y, n=48):
    """Running peak envelope of an oscillating series: phase-blind, so it
    isolates amplitude.  Returns (t_centres, |y|_max per window)."""
    m = len(t) // n
    if m < 2:
        return t, np.abs(y)
    tt = t[:m * n].reshape(n, m)
    yy = np.abs(y[:m * n]).reshape(n, m)
    return tt.mean(axis=1), yy.max(axis=1)


# ── run wallclocks ───────────────────────────────────────────────────────────
def local_wall(tag):
    p = LOCAL / f"{tag}.log"
    if not p.exists():
        return None
    return None
