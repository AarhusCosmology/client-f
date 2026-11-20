"""One data layer for the note: the measured Gamma_T bins, with the proxy
background's OWN eps_ne attached to each bin.

Everything here reads the campaign's derived/*.json through the nb/ data layer
(dncdm.py) and the proxy background columns through draft-paper/runs/out.  The
gate is the campaign's own: gamma in [2,15], where the refine pair agrees."""
import sys, re, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # paper/dncdm.py
import dncdm as D

OUT   = D.HARNESS / 'proxy_backgrounds' / 'out'
GLO, GHI = 2.0, 15.0
T0_EV    = 0.71611 * 2.7255 * 8.617333262e-5      # the campaign's dncdm1.T


def _bgtag(gl):
    """'05.0' -> '5', '10.5' -> '10.5'.  Derived, not tabulated.

    This was a hand-written dict of the SEVEN integer rungs, and every half-decade
    cell in hpc_gal/hpc_gal2 fell through it into the silent `continue` below -- half
    the measured ladder, absent from the calibration and invisible in the bin count."""
    return f"{float(gl):g}"


def _bg_eps(gl, mass):
    f = OUT / f"bg_G1e{gl}_m{mass}_background.dat"
    if not f.exists():
        return None
    hdr   = [l for l in f.read_text().splitlines() if l.startswith('#')][-1]
    names = [s.strip() for s in re.split(r'\d+:', hdr.lstrip('#'))][1:]
    d     = np.loadtxt(f)
    col   = {n: d[:, i] for i, n in enumerate(names)}
    a     = 1.0 / (1.0 + col['z'])
    o     = np.argsort(a)
    return dict(a=a[o], eps=col['proxy_eps_ne_dncdm1'][o])


# The half-decade rungs exist in hpc_gal/hpc_gal2 and were silently outside the
# calibration until 2026-09-14: a hand-written GLMAP of the seven integer rungs dropped
# them, because the proxy backgrounds carrying `proxy_eps_ne` only ever ran at integer
# decades.  They are ON now, with one documented exclusion.
#
# ⚠ Those backgrounds must be generated with the mantissa written out in full.
# `dncdm1.Gamma = 1e7.5` parses through strtod, whose exponent must be an INTEGER, so it
# reads 1e7 and yields a silent duplicate of the decade below -- no warning, and the key
# still counts as consumed.  The first attempt at this did exactly that and produced a
# confident, wrong conclusion (that the half-decade bins degrade the fit) that survived
# several diagnostics.  See draft-paper/runs/batch_bg_half.py.
HALF_DECADES = True

# The one group that does not go in.  Amplitudes fitted on the integer rungs alone and
# the half decades scored OUT OF SAMPLE against their own integer neighbours, per
# (mass, Gamma) region: half decades match or beat the integers in five of six regions
# (m=0.3 all: 0.244 vs 0.267; m=0.06 at Gamma >= 1e8: 0.180 vs 0.234).  The exception is
# m = 0.06 below 1e8, where the half decades give rms 1.013 against 0.336 for the
# integers in the same corner.  That is a quality criterion measured independently of the
# fit, not a fit-quality cut -- but WHY those three rungs fail is not known, and this
# exclusion should be revisited, not inherited.
EXCLUDE_HALF = lambda mass, gl: mass == '0.06' and float(gl) < 8.0


def cells(scheme='m6', mb=64, half_decades=None):
    """One record per (mass, Gamma, k), with per-bin arrays.

    Cells dropped for want of a proxy background are recorded in `cells.skipped`
    rather than vanishing: a calibration that silently shrinks is how the half-decade
    rungs went missing in the first place.  Callers should print it.
    """
    cells.skipped = []
    half = HALF_DECADES if half_decades is None else half_decades
    out = []
    for mass in ('0.3', '0.06'):
        # Candidates per (Gamma, k) in PREFER order, and the FIRST USABLE one wins --
        # not simply the first. Taking the preferred round's cell before checking it has
        # a twin discarded 56 of 172 cells outright, because a cell whose twin is missing
        # fails `sigma_rows` and the lower-precedence round never got asked. Falling back
        # costs nothing here: measured, gal and gal2's mb = 64 iso cells give the SAME
        # Gamma_T (median ratio 1.0000 over 1269 matched bins) -- the reduced table that
        # separates those rounds does not reach the a4Pi projection.
        cand = {}
        for c in D.iso(mass=mass, scheme=scheme, momenta_bins=mb):
            cand.setdefault((c.gl, c.k_index), []).append(c)
        seen = {}
        for key, cs in cand.items():
            pick = next((c for c in cs if len(D.sigma_rows(c, GLO, GHI)) >= 3), cs[0])
            seen[key] = pick
        for (gl, ki), c in seen.items():
            if not gl.endswith('.0'):
                if not half or EXCLUDE_HALF(mass, gl):
                    continue
            bg = _bg_eps(_bgtag(gl), mass)
            if bg is None:
                cells.skipped.append(
                    f"m={mass} gl={gl}: no proxy background "
                    f"bg_G1e{_bgtag(gl)}_m{mass}_background.dat")
                continue
            re_ = D.sigma_rows(c, GLO, GHI)
            rr  = [r for r, _ in re_]
            s   = np.array([x for _, x in re_])
            if len(rr) < 3:
                continue
            bgm = D.background(*D._bg_key(c))
            g   = np.array([r['gamma'] for r in rr])
            av  = np.array([r['a'] for r in rr])
            w   = np.interp(av, bgm['a'], D.rho_frac_sector(bgm))
            v   = np.array([r['GT'] for r in rr]) / (c.gamma / D.CL_KMS_MPC)
            m   = bg['eps'] > 0
            e   = np.exp(np.interp(np.log(av), np.log(bg['a'][m]), np.log(bg['eps'][m])))
            eH  = np.abs(np.array([r['GT_H'] for r in rr])) * g / (c.gamma / D.CL_KMS_MPC)
            ok  = (w > 0) & (v > 0) & np.isfinite(e) & (e > 0)
            if ok.sum() < 3:
                continue
            out.append(dict(mass=mass, gl=gl, k=ki, gamma=g[ok], y=v[ok] / w[ok],
                            eps=e[ok], epsH=eH[ok], sigma=s[ok],
                            X=av[ok] * float(mass) / T0_EV))
    return out


def flat(cs, *keys):
    return [np.concatenate([c[k] for c in cs]) for k in keys]


def stage1(c, n3, qpow=1.7995, shut=None):
    """(C3 eps^n3, C5) for ONE cell, the published two-stage recipe: the shut-off and
    the sector weight are divided out, then a weighted linear LSQ in the amplitudes."""
    g, y, s = c['gamma'], c['y'], c['sigma']
    sh = (1.0 - 1.0 / g**2)**qpow if shut is None else shut
    yy = y / sh
    X  = np.vstack([g**-3.0, g**-5.0]).T
    W  = 1.0 / (s * yy)**2
    A  = X.T @ (W[:, None] * X)
    cov = np.linalg.inv(A)
    b  = cov @ (X.T @ (W * yy))
    return b[0], b[1], np.sqrt(np.diag(cov))


def slope(x, y):
    """log-log slope with its standard error."""
    x, y = np.log(np.asarray(x)), np.log(np.asarray(y))
    M = np.vstack([np.ones_like(x), x]).T
    b, *_ = np.linalg.lstsq(M, y, rcond=None)
    r = y - M @ b
    s2 = r @ r / max(1, len(x) - 2)
    cov = s2 * np.linalg.inv(M.T @ M)
    return b[1], float(np.sqrt(cov[1, 1])), float(np.exp(b[0])), float(np.std(r))
