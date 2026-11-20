"""Data layer over the finished DNCDM HPC rounds.

Reads ONLY the post-merge rounds -- `hpc_prod2`, `hpc_gal`, `hpc_gal2`, `hpc_ratchet`.
The pre-merge matrices (`hpc_cmb`, `hpc_reduced`, `hpc_defl`, `hpc_prod`) are deliberately
NOT registered: merged master's DNCDM knob set is smaller than the campaign branches', and
CLASS ignores keys it does not recognise with no warning at any verbosity, so a pre-merge
ini runs and means something different. They are cross-checks, never co-plotted.

════════════════════════════════════════════════════════════════════════════════
THREE CONVENTIONS, EACH OF WHICH WAS A BUG FIRST

* **column POSITION, never the printed number.** CLASS writes `cl.dat` with
  `# 1:l  1:TT  2:EE ...`, so `l` and `TT` both claim index 1.
* **a scale-relative metric, never a pointwise ratio,** for anything that crosses zero.
  C_l^TT is positive so a plain ratio is honest there; C_l^TE and a^4 Pi are not.
* **PCHIP, never a global cubic spline,** when resampling. CLASS restarts the evolver at
  every approximation-scheme boundary and a global spline rings across the step jump.

════════════════════════════════════════════════════════════════════════════════
AND ONE THAT IS SPECIFIC TO THESE ROUNDS

A Galerkin (`m4`/`m6`/`m8`) cell run before PR #413 carries a reduced collision table
interpolated at a FIXED ln-a spacing, which is converged only at low Gamma. `table_error()`
gives the predicted C_2^TT error for any such cell, and `cells()` stamps every record with
`table_ok`. Nothing here silently drops those cells -- the divergence is a result in its
own right -- but no plot should mix them with post-#413 ones without saying so.
"""
from __future__ import annotations

import gzip
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

import os
# The campaign folders (hpc_prod2/, hpc_gal/, hpc_gal2/, proxy_rta/, ...) with their out/: DNCDM_HARNESS,
# or else ../campaigns of this branch.
HARNESS = Path(os.environ.get("DNCDM_HARNESS") or Path(__file__).resolve().parents[1] / "campaigns")
C2_TARGET = 1.481e-10   # C_2^TT is Gamma-INDEPENDENT in this model; departures are error
CL_KMS_MPC = 2.99792458e5   # Gamma is quoted in km/s/Mpc; CLASS stores Gamma/c in 1/Mpc


# ── which rounds exist, and what each is good for ────────────────────────────
@dataclass(frozen=True)
class Round:
    key: str
    dirname: str
    exact_ok: bool      # are its EXACT cells usable?
    reduced_ok: bool    # are its REDUCED cells on a converged collision table?
    note: str

    @property
    def path(self) -> Path:
        return HARNESS / self.dirname


ROUNDS = {
    r.key: r for r in [
        Round("prod2", "hpc_prod2", True, False,
              "the production round. Exact ladder at both masses; its Galerkin cells "
              "predate PR #413 and carry the fixed-spacing table."),
        Round("gal", "hpc_gal", True, False,
              "VOID as a Galerkin round -- it reproduced prod2 byte-identically because "
              "the table key was never emitted. Its EXACT cells and its momenta_bins = 64 "
              "iso arm are real."),
        Round("gal2", "hpc_gal2", True, True,
              "the redo, on the PR #413 binary with dr_table_max_dlna set per cell. This "
              "is the only round whose Galerkin cells are on a converged table."),
        Round("ratchet", "hpc_ratchet", True, True,
              "the balanced-gather A/B. Not a ladder: three gathers at matched cells. Its "
              "`lin` arm reproduces prod2 by construction."),
        Round("scenb", "hpc_scenb", True, False,
              "the scenario-B falsification round (PR #431, b1792c46). Its cells carry "
              "a SUFFIX: `_B1` is two daughters (N_ur = 0, dr_n_daughter = 2), `_A` is "
              "the one-free-streaming control re-run on the same binary. Exact only; no "
              "Galerkin arm. Deliberately NOT in PREFER -- `_B1` is a different sector, "
              "not a better measurement of the same one, so it must never win precedence "
              "over prod2 for a plain `iso_G*_m*_x` cell. Select it explicitly: "
              "dncdm.iso(rounds=['scenb'], suffix='B1')."),
    ]
}

# Precedence when the same cell exists in more than one round: newest binary wins.
# scenb is absent ON PURPOSE (see its note): it measures a different sector, and a
# round that answers a different question must not silently outrank one that answers
# this one.
PREFER = ["gal2", "gal", "prod2"]


# ── the pre-#413 table error, so the validity edge is drawn and not guessed ──
# err = A(m) (Gamma/1e11)^INDEX / n_sub^2, fitted to prod2's own n_sub = 1 ladder and
# validated against an independent n_sub sweep. See hpc_gal2/config.py, which is the
# source of these numbers, and species/dncdm_inv_species.h for the derivation.
KTABLE_INDEX = 1.103
KTABLE_A = {"0.3": 0.2947, "0.06": 0.8682}


def table_error(gamma: float, mass: str, n_sub: int = 1) -> float:
    """Predicted |C_2^TT| error from the reduced table's ln-a interpolation.

    Only meaningful while the error is small: the O(h^2) law does not describe the
    runaway (it predicts 105% at Gamma = 3.16e11 where the cell reads +506%)."""
    A = KTABLE_A.get(str(mass))
    if A is None:
        return float("nan")
    return A * (gamma / 1e11) ** KTABLE_INDEX / n_sub ** 2


# ── files ────────────────────────────────────────────────────────────────────
def _open(p: Path):
    return gzip.open(p, "rt") if str(p).endswith(".gz") else open(p)


def cols(path: Path) -> dict[str, int]:
    """Column name -> 0-based POSITION. Never the printed index (see module docstring)."""
    head = None
    with _open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if re.search(r"\d+:", line):
                head = line
    if head is None:
        raise ValueError(f"no column header in {path}")
    names = re.findall(r"\d+:(.*?)(?=\s+\d+:|$)", head.rstrip())
    return {n.strip(): i for i, n in enumerate(names)}


def find(rnd: str, tag: str, *suffixes) -> Path | None:
    d = ROUNDS[rnd].path / "out"
    for s in suffixes:
        for cand in (d / f"{tag}_{s}", d / f"{tag}_{s}.gz"):
            if cand.exists() and cand.stat().st_size:
                return cand
    return None


# ── the cell inventory ───────────────────────────────────────────────────────
TAG_RE = re.compile(
    r"^(?P<family>bg|cmb|iso|gal|ceil)_G(?P<gl>[\d.]+)_m(?P<mass>[\d.]+)"
    r"(?:_(?P<scheme>x|m4|m6|m8))?(?P<twin>c)?(?:_(?P<suffix>.+))?$")


@dataclass(frozen=True)
class Cell:
    round: str
    tag: str
    family: str
    gl: str            # "09.5"
    gamma: float       # km/s/Mpc
    mass: str          # "0.3"
    scheme: str        # "x" | "m4" | "m6" | "m8"
    refine: int        # 1 = production, 2 = the halved-perturbation-grid twin
    suffix: str        # "" | "q2x" | "mb64" | "t2x" | "lin"/"bal"/"chn" ...
    table_ok: bool     # is its reduced collision table converged? (True for exact)
    table_err: float   # predicted C_2 error from the table, 0 for exact / post-#413

    @property
    def reduced(self) -> bool:
        return self.scheme != "x"

    @property
    def label(self) -> str:
        s = {"x": "exact"}.get(self.scheme, self.scheme)
        return f"$\\Gamma=10^{{{float(self.gl):g}}}$, {s}"


def _parse(rnd: str, tag: str) -> Cell | None:
    m = TAG_RE.match(tag)
    if not m or m["scheme"] is None:
        return None
    gl, mass, scheme = m["gl"], m["mass"], m["scheme"]
    gamma = 10.0 ** float(gl)
    suffix, refine = m["suffix"] or "", 2 if m["twin"] else 1
    # hpc_ratchet appends the GATHER after the scheme, so its refine twin reads
    # `..._x_balc`, not `..._xc` -- the trailing `c` lands on the gather instead. Left
    # unhandled that silently files a convergence twin as a production cell under a
    # suffix of its own, which is the one mistake this parser must not make.
    for gather in ("lin", "bal", "chn"):
        if suffix == gather + "c":
            suffix, refine = gather, 2
        elif suffix.startswith(gather + "c_"):
            suffix, refine = gather + suffix[len(gather) + 1:], 2
    reduced = scheme != "x"
    if not reduced:
        ok, err = True, 0.0
    elif ROUNDS[rnd].reduced_ok:
        ok, err = True, 0.0
    else:
        err = table_error(gamma, mass, 1)
        # A conservative SCREEN on the predicted error against the sizing rule's own 0.1%
        # target -- not a verdict. At Gamma = 1e9 the law says 0.2% and the cell measures
        # 0.09%, so a flagged cell is one to check, not one to discard.
        ok = err < 1e-3
    return Cell(round=rnd, tag=tag, family=m["family"], gl=gl, gamma=gamma,
                mass=mass, scheme=scheme, refine=refine,
                suffix=suffix, table_ok=ok, table_err=err)


def cells(family: str | None = None, scheme: str | None = None,
          mass: str | None = None, refine: int | None = 1,
          suffix: str | None = "", rounds: list[str] | None = None,
          probe: str = "cl_lensed.dat") -> list[Cell]:
    """Every cell with output on disk, newest round first.

    `suffix=""` keeps only the plain ladder cells (no `_q2x`, `_mb64`, `_t2x`, no gather
    variant); pass None to keep everything, or a literal to select one rung.
    """
    out = []
    for rnd in (rounds or PREFER):
        d = ROUNDS[rnd].path / "out"
        if not d.is_dir():
            continue
        seen = set()
        for f in sorted(d.iterdir()):
            name = f.name
            for suf in (probe, probe + ".gz"):
                if name.endswith("_" + suf):
                    seen.add(name[: -len(suf) - 1])
        for tag in sorted(seen):
            c = _parse(rnd, tag)
            if c is None:
                continue
            if family and c.family != family:
                continue
            if scheme and c.scheme != scheme:
                continue
            if mass and c.mass != mass:
                continue
            if refine is not None and c.refine != refine:
                continue
            if suffix is not None and c.suffix != suffix:
                continue
            out.append(c)
    return out


def bg_rungs(mass: str, rounds=None) -> list[str]:
    """log10Gamma of every background table on disk for this mass, sorted.

    Backgrounds get their own lookup because a bg tag carries NO scheme -- `bg_G09.0_m0.3`
    -- so TAG_RE rejects it and `cells()` never sees one. The background solve is identical
    across schemes and refine, which is exactly why it is tagged without them."""
    out = set()
    for rnd in (rounds or PREFER):
        d = ROUNDS[rnd].path / "out"
        if not d.is_dir():
            continue
        for f in d.glob(f"bg_G*_m{mass}_bgslim.dat*"):
            m = re.match(rf"^bg_G([\d.]+)_m{re.escape(mass)}_bgslim\.dat", f.name)
            if m:
                out.add(m.group(1))
    return sorted(out)


def ladder(scheme="x", mass="0.3", family="cmb", rounds=None, **kw) -> list[Cell]:
    """One cell per Gamma, sorted, with round precedence applied."""
    got = cells(family=family, scheme=scheme, mass=mass, rounds=rounds, **kw)
    best: dict[str, Cell] = {}
    order = {r: i for i, r in enumerate(rounds or PREFER)}
    for c in got:
        cur = best.get(c.gl)
        if cur is None or order[c.round] < order[cur.round]:
            best[c.gl] = c
    return sorted(best.values(), key=lambda c: float(c.gl))


def provenance(cs: list[Cell]) -> str:
    """A table saying which round every plotted cell came from. Print it beside a figure;
    silently mixing rounds is the failure this whole layer exists to prevent."""
    w = max((len(c.tag) for c in cs), default=3)
    lines = [f"{'tag':<{w}}  {'round':<8} {'m':>5} {'scheme':>7} {'table':>18}"]
    for c in sorted(cs, key=lambda c: (c.mass, c.scheme, float(c.gl))):
        if not c.reduced:
            t = "n/a (exact)"
        elif c.table_ok and c.table_err == 0.0:
            t = "converged (#413)"
        else:
            t = f"PRE-#413 ~{100 * c.table_err:.1f}%"
        lines.append(f"{c.tag:<{w}}  {c.round:<8} {c.mass:>5} {c.scheme:>7} {t:>18}")
    return "\n".join(lines)


# ── spectra ──────────────────────────────────────────────────────────────────
def cl(c: Cell, lensed=True) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    f = find(c.round, c.tag, *(("cl_lensed.dat", "cl.dat") if lensed else ("cl.dat",)))
    if f is None:
        raise FileNotFoundError(f"no C_l for {c.round}/{c.tag}")
    idx = {k.upper(): v for k, v in cols(f).items()}
    with _open(f) as fh:
        d = np.loadtxt(fh)
    return d[:, 0], {k: d[:, v] for k, v in idx.items() if k != "L"}


def pk(c: Cell, z: int = 1, cb=False) -> tuple[np.ndarray, np.ndarray]:
    stem = f"z{z}_pk_cb.dat" if cb else f"z{z}_pk.dat"
    f = find(c.round, c.tag, stem)
    if f is None:
        raise FileNotFoundError(f"no P(k) for {c.round}/{c.tag}")
    with _open(f) as fh:
        d = np.loadtxt(fh)
    return d[:, 0], d[:, 1]


def cl_ratio(ref: Cell, test: Cell, key="TT", lensed=True):
    """test/ref - 1 on the common ell range. TT and EE are positive so a plain ratio is
    honest; NEVER use this for TE, which crosses zero."""
    assert key in ("TT", "EE", "PHIPHI"), f"{key} may cross zero -- use a scale-relative metric"
    lr, dr = cl(ref, lensed)
    lt, dt = cl(test, lensed)
    n = min(len(lr), len(lt))
    return lr[:n], dt[key][:n] / dr[key][:n] - 1.0


def pk_ratio(ref: Cell, test: Cell, z=1, cb=False):
    kr, pr = pk(ref, z, cb)
    kt, pt = pk(test, z, cb)
    if len(kr) == len(kt) and np.allclose(kr, kt, rtol=1e-9):
        return kr, pt / pr - 1.0
    y = PchipInterpolator(kt, pt, extrapolate=False)(kr)   # PCHIP, never a global spline
    m = np.isfinite(y)
    return kr[m], y[m] / pr[m] - 1.0


def c2(c: Cell, lensed=True) -> float:
    ell, d = cl(c, lensed)
    return float(d["TT"][int(np.searchsorted(ell, 2))])


# ── background ───────────────────────────────────────────────────────────────
_BG_CACHE: dict[tuple[str, str], dict[str, np.ndarray]] = {}


def background(rnd: str, gl: str, mass: str, suffix: str = "") -> dict[str, np.ndarray] | None:
    """The slim background table (per-q PSD block dropped), sorted by increasing a."""
    tag = f"bg_G{gl}_m{mass}" + (f"_{suffix}" if suffix else "")
    ck = (rnd, tag)
    if ck in _BG_CACHE:
        return _BG_CACHE[ck]
    f = find(rnd, tag, "bgslim.dat", "background.dat")
    if f is None:
        return None
    idx = cols(f)
    with _open(f) as fh:
        d = np.loadtxt(fh)
    out = {k: d[:, i] for k, i in idx.items()}
    out["a"] = 1.0 / (1.0 + out["z"])
    o = np.argsort(out["a"])
    out = {k: v[o] for k, v in out.items()}
    _BG_CACHE[ck] = out
    return out


def bg_for(c: Cell) -> dict[str, np.ndarray] | None:
    """The background belonging to a cell -- from its own round if it has one, else from
    whichever round does. The background solve is scheme- and refine-independent, so this
    is safe in a way that sharing a SPECTRUM across rounds would not be."""
    for rnd in [c.round] + [r for r in PREFER if r != c.round]:
        bg = background(rnd, c.gl, c.mass)
        if bg is not None:
            return bg
    return None


SECTOR = ["(.)rho_dncdm1", "(.)rho_dr_dncdm1_l", "(.)rho_dr_dncdm1_phi"]


def rho_frac_sector(bg) -> np.ndarray:
    """rho_H / (rho_H + rho_l + rho_phi): the weight the transport rate carries."""
    tot = sum(bg[k] for k in SECTOR)
    return np.divide(bg[SECTOR[0]], tot, out=np.zeros_like(tot), where=tot > 0)


def a_nr(mass: float) -> float:
    """Where the parent turns non-relativistic, <p>/m = 1. The decay epoch saturates at
    1.23 a_nr for any Gamma above H(a_nr)."""
    return 3.1515 * 0.71611 * 2.7255 * 8.617333e-5 / float(mass)


# ── the isotropisation projection (family B's derived/*.json) ────────────────
@dataclass
class Iso:
    round: str
    tag: str
    k: float
    k_index: int
    gl: str
    gamma: float          # km/s/Mpc
    mass: str
    scheme: str
    refine: int
    momenta_bins: int
    suffix: str
    rows: list[dict]
    gamma_max: float      # support.gamma_max -- past it GT is exactly zero, not measured
    table_ok: bool

    @property
    def reduced(self) -> bool:
        return self.scheme != "x"


_ISO_ALL: list[Iso] | None = None


def _iso_all() -> list[Iso]:
    """Every `derived/*.json` in every registered round, parsed ONCE.

    `twin()` is called inside every fit, and the original implementation re-globbed and
    re-parsed the whole derived tree on each call. With ~1300 records that is O(n^2) and
    a single weighted-fit table took minutes; the cache makes it seconds. Nothing here
    is written back, so the cache is safe for the life of a kernel -- restart it if a
    round syncs new cells in mid-session."""
    global _ISO_ALL
    if _ISO_ALL is not None:
        return _ISO_ALL
    out = []
    for rnd in ROUNDS:
        d = ROUNDS[rnd].path / "derived"
        if not d.is_dir():
            continue
        for f in sorted(d.glob("iso_*__k*.json")):
            try:
                r = json.loads(f.read_text())
            except Exception:
                continue          # a cell still being written
            c = _parse(rnd, r["tag"])
            if c is None:
                continue
            out.append(Iso(round=rnd, tag=r["tag"], k=r["k"], k_index=r["k_index"],
                           gl=r["log10Gamma"], gamma=r["Gamma"], mass=str(r["mass"]),
                           scheme=r["scheme"], refine=r["refine"],
                           momenta_bins=r["momenta_bins"], suffix=c.suffix,
                           rows=r["rows"],
                           gamma_max=r.get("support", {}).get("gamma_max", np.inf),
                           table_ok=c.table_ok))
    _ISO_ALL = out
    return out


def iso(mass=None, scheme=None, refine=1, k_index=None, rounds=None,
        suffix="", momenta_bins=None) -> list[Iso]:
    order = {r: i for i, r in enumerate(rounds or PREFER)}
    out = []
    for c in _iso_all():
        if c.round not in order:
            continue
        if mass and c.mass != str(mass):
            continue
        if scheme and c.scheme != scheme:
            continue
        if refine is not None and c.refine != refine:
            continue
        if k_index is not None and c.k_index not in np.atleast_1d(k_index):
            continue
        if suffix is not None and c.suffix != suffix:
            continue
        if momenta_bins is not None and c.momenta_bins != momenta_bins:
            continue
        out.append(c)
    out.sort(key=lambda c: (order[c.round], c.mass, c.gl, c.scheme, c.k_index))
    return out


# ── gates ────────────────────────────────────────────────────────────────────
# TWO of these are structural and always on; two are QUALITY FLAGS and are off by
# default, because the published estimator does not apply them and turning them on
# silently empties the upper gamma window -- which is the window the whole gamma^-5
# question lives in. `quality()` reports them instead of dropping on them.
GATE_GAMMA = 15.0    # above this the refine pair stops agreeing: not a measurement
MIN_CANCEL = 0.0     # a flag, not a cut: see quality()
MIN_R2 = 0.0         # a flag, not a cut: see quality()
FLAG_CANCEL = 0.05   # below this Gamma_T is a small residual of much larger legs
FLAG_R2 = 0.80       # below this "a relaxation rate" is not the right description


def gated(c: Iso, gmax=GATE_GAMMA, min_cancel=MIN_CANCEL, min_r2=MIN_R2, field="GT"):
    """(kept rows, drop tally). The `gsup` gate is the one that is easy to miss: a REDUCED
    cell applies the operator only inside its table window, so GT is exactly ZERO outside
    it -- and a bin of exact zeros fits Gamma_T = 0 and looks like a converged
    measurement."""
    keep, drop = [], dict(gsup=0, cancel=0, r2=0, nonpos=0, window=0)
    for r in c.rows:
        g = r.get("gamma", 0.0)
        if g > c.gamma_max:
            drop["gsup"] += 1
        elif g > gmax:
            drop["window"] += 1
        elif abs(r.get("cancel", 0.0)) < min_cancel:
            drop["cancel"] += 1
        elif r.get("R2", 0.0) < min_r2:
            drop["r2"] += 1
        elif not (r.get(field, 0.0) > 0.0):
            drop["nonpos"] += 1
        else:
            keep.append(r)
    return keep, drop


def gamma0_inv_mpc(c: "Iso") -> float:
    """Gamma_0 in 1/Mpc, which is what GT is measured in.

    ⚠ The tag and the manifest quote Gamma_0 in km/s/Mpc; CLASS stores and integrates
    Gamma_0/c. Dividing GT by the km/s/Mpc number is wrong by c = 2.998e5. It cancels out
    of a power-law SLOPE, so fit_p never noticed -- but C_3 and C_5 are amplitudes and it
    lands on them directly."""
    return c.gamma / CL_KMS_MPC


def quality(c: "Iso", glo=2.0, ghi=GATE_GAMMA) -> dict:
    """How trustworthy the surviving bins are, reported rather than acted on.

    `cancel` far below 1 means Gamma_T is a small residual of much larger parent/nu_l/phi
    legs, so its error bar is not its components'. Low R2 means the relaxation ansatz is
    not the right description of that bin at all."""
    keep, _ = gated(c, gmax=ghi)
    rows = [r for r in keep if r["gamma"] >= glo]
    if not rows:
        return dict(n=0, med_cancel=float("nan"), med_r2=float("nan"),
                    frac_low_cancel=float("nan"), frac_low_r2=float("nan"))
    ca = np.array([r.get("cancel", np.nan) for r in rows])
    r2 = np.array([r.get("R2", np.nan) for r in rows])
    return dict(n=len(rows), med_cancel=float(np.nanmedian(ca)),
                med_r2=float(np.nanmedian(r2)),
                frac_low_cancel=float(np.mean(np.abs(ca) < FLAG_CANCEL)),
                frac_low_r2=float(np.mean(r2 < FLAG_R2)))


def arr(rows, field="GT"):
    return (np.array([r["gamma"] for r in rows]),
            np.array([r[field] for r in rows]))


def fit_p(c: Iso, glo=2.0, ghi=GATE_GAMMA, field="GT"):
    """p in Gamma_T/Gamma_0 = C gamma^-p, by least squares in log-log.

    ⚠ A coarse perturbation grid does not SCATTER p, it fakes a STEEPER one: +0.96 for a
    4x coarser grid, +4.9 for 8x. Always read `dp_coarse()` beside a fitted p."""
    keep, _ = gated(c, gmax=ghi, field=field)
    g, v = arr([r for r in keep if r["gamma"] >= glo], field)
    if len(g) < 3:
        return None
    x, y = np.log(g), np.log(v / gamma0_inv_mpc(c))
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(1, len(x) - 2)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    return dict(p=-coef[1], sd=math.sqrt(cov[1, 1]), n=len(g), logC=coef[0],
                rms=math.sqrt(s2), gspan=(float(g.min()), float(g.max())))


def eps_H(c: Iso, glo=2.0, ghi=GATE_GAMMA) -> float:
    """The sector's departure from detailed balance: the parent's own l=2 collision term
    in units of its free-decay rate Gamma_0/gamma. 1 = the inverse decays compensate
    nothing; -> 0 = exact detailed balance.

    This is the variable the exponent actually tracks -- p collapses onto log10(eps_H)
    across both masses and four decades of Gamma_0."""
    keep, _ = gated(c, gmax=ghi, field="GT")
    g, v = arr([r for r in keep if r["gamma"] >= glo], "GT_H")
    q = np.abs(v) * g / (c.gamma / CL_KMS_MPC)
    q = q[q > 0]
    return float(np.exp(np.mean(np.log(q)))) if len(q) >= 3 else float("nan")


_TWIN_IX: dict | None = None


def _twin_key(c: "Iso"):
    return (c.round, c.gl, c.mass, c.scheme, c.momenta_bins, c.suffix, c.k_index)


def twin(c: Iso) -> Iso | None:
    """The refine = 2 cell of the same (round, Gamma, mass, scheme, momenta_bins, k)."""
    global _TWIN_IX
    if _TWIN_IX is None:
        _TWIN_IX = {_twin_key(x): x for x in _iso_all() if x.refine == 2}
    return _TWIN_IX.get(_twin_key(c))


def dp_coarse(c: Iso, **kw) -> float:
    """p(halved grid) - p(production). The convergence GATE, not a diagnostic: a coarse
    grid manufactures exactly the gamma^-5 this measurement exists to test, so a large
    |dp| means the fitted p is a discretisation artefact."""
    t = twin(c)
    if t is None:
        return float("nan")
    a, b = fit_p(c, **kw), fit_p(t, **kw)
    return float("nan") if not (a and b) else b["p"] - a["p"]


def median_p(cs: list[Iso], glo=2.0, ghi=GATE_GAMMA):
    """Median fitted p over the k modes of one cell, with the spread. The published
    number is a median over k, never a single mode."""
    ps = [f["p"] for f in (fit_p(c, glo, ghi) for c in cs) if f]
    if not ps:
        return None
    return dict(p=float(np.median(ps)), sd=float(np.std(ps)), n=len(ps))


# ── the two-power transport rate ─────────────────────────────────────────────
# Gamma_T/Gamma_0 = (rho_H/rho_sec) (1 - 1/gamma^2)^q [C3 gamma^-3 + C5 gamma^-5]
# at l = 2, where alpha_2 = beta_2 = 1 by the normalisation of the l-dependence.
def rate_design(c: Iso, rows, qpow: float):
    """(y, X) for the linear least squares in (C3, C5), with the background weight and
    the velocity shut-off divided out. Returns None if the background is unavailable."""
    bg = background(*_bg_key(c))
    if bg is None:
        return None
    g, v = arr(rows, "GT")
    a = np.array([r["a"] for r in rows])
    w = np.interp(a, bg["a"], rho_frac_sector(bg))
    shut = (1.0 - 1.0 / g ** 2) ** qpow
    denom = w * shut
    ok = denom > 0
    y = (v / gamma0_inv_mpc(c))[ok] / denom[ok]
    X = np.vstack([g[ok] ** -3.0, g[ok] ** -5.0]).T
    return y, X, g[ok]


def _bg_key(c: Iso) -> tuple[str, str, str]:
    for rnd in [c.round] + [r for r in PREFER if r != c.round]:
        if background(rnd, c.gl, c.mass) is not None:
            return rnd, c.gl, c.mass
    return c.round, c.gl, c.mass


def fit_two_power(c: Iso, qpow=1.7995, glo=2.0, ghi=GATE_GAMMA, powers=(3.0, 5.0)):
    """Fit C3, C5 (or any exponent pair) at fixed q by non-negative-free linear LSQ.

    ⚠ THE CONTROL MATTERS MORE THAN THE FIT. Over a 0.7-decade window with two free
    amplitudes, (3,5), (3,6), (3,7) and a single power all fit to a few per cent. Use
    `compare_forms()` and read the chi2 column before believing any of them."""
    keep, _ = gated(c, gmax=ghi)
    rows = [r for r in keep if r["gamma"] >= glo]
    if len(rows) < 3:
        return None
    d = rate_design(c, rows, qpow)
    if d is None:
        return None
    y, _X, g = d
    X = np.vstack([g ** -powers[0], g ** -powers[1]]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    # relative chi2: the rate spans decades, so an absolute residual is meaningless
    chi2 = float(np.mean((resid / y) ** 2))
    return dict(C3=float(coef[0]), C5=float(coef[1]), chi2=chi2, n=len(y),
                powers=powers, q=qpow)


def fit_one_power(c: Iso, qpow=1.7995, glo=2.0, ghi=GATE_GAMMA):
    """The control: ONE free amplitude and a free exponent, on the same weighted data."""
    keep, _ = gated(c, gmax=ghi)
    rows = [r for r in keep if r["gamma"] >= glo]
    if len(rows) < 3:
        return None
    d = rate_design(c, rows, qpow)
    if d is None:
        return None
    y, _X, g = d
    A = np.vstack([np.ones_like(g), np.log(g)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    pred = np.exp(A @ coef)
    return dict(C=float(np.exp(coef[0])), p=float(-coef[1]),
                chi2=float(np.mean((1.0 - pred / y) ** 2)), n=len(y))


# ════════════════════════════════════════════════════════════════════════════
# ADDED 2026-08-31, answering five questions the first pass raised.
# ════════════════════════════════════════════════════════════════════════════

# ── how big is a production cell's OWN error? ────────────────────────────────
# The first pass drew ONE grey band per panel: the envelope of |C_l(xc)/C_l(x) - 1|
# over the whole Gamma ladder, xc being the refine = 2 twin (the SAME cell on a
# perturbation grid with half the daughter-momentum points). Two things are wrong
# with that, and both matter:
#
#   1. It is not Gamma-independent. |xc/x - 1| at ell <= 20 runs 0.05% at Gamma = 1e5
#      to 3.8% at 1e10 -- a factor of 80 down one ladder. One envelope is the WORST
#      rung's number drawn under every rung.
#
#   2. |f(2h) - f(h)| is not the error of f(h); it is the error of f(2h). For a
#      scheme of order p, err(h) = |f(2h) - f(h)| / (2^p - 1).
#
# p is MEASURABLE here, because six rungs carry a `_q2x` cell -- the same cosmology
# at DOUBLE dr_N_q. With f(2h) = xc, f(h) = x and f(h/2) = q2x,
#
#       [f(2h) - f(h)] / [f(h) - f(h/2)] = 2^p,
#
# and `richardson_order()` reads it off. Measured, at the two rungs where the
# differences sit well above the round's reproducibility floor (Gamma = 1e7 and 1e8
# at m = 0.3, ell <= 200), it is p = 3.3-4.7. So the refine pair OVERSTATES the
# production cell's error by 10-25x, and the first pass's grey band was wide enough
# to swallow every closure residual in the campaign.
#
# ⚠ q2x doubles dr_N_q, which doubles the BACKGROUND grid as well as the
# perturbation one, while refine only subsamples the perturbation grid. So the
# triple is not a textbook-clean h-sequence and p from it is indicative. That is
# why `cl_error()` defaults to the CONSERVATIVE order = 2 (divide by 3) and why the
# direct |x - q2x| bracket is plotted beside it wherever it exists.
RICH_BANDS = [(2, 20), (20, 200), (200, 800), (800, 2000)]


def cell_at(mass: str, gl: str, scheme: str = "x", family: str = "cmb",
            suffix: str = "", refine: int = 1, rounds=None) -> Cell | None:
    """The one cell matching (mass, Gamma, scheme, ...), by round precedence."""
    for c in cells(family=family, scheme=scheme, mass=mass, refine=refine,
                   suffix=suffix, rounds=rounds):
        if c.gl == gl:
            return c
    return None


def _tt(c: Cell, lensed=True):
    ell, d = cl(c, lensed)
    return ell, d["TT"]


def cl_diff(a: Cell, b: Cell, key="TT", lensed=True):
    """b/a - 1 on the common ell range. Both are TT, which is positive."""
    la, da = cl(a, lensed)
    lb, db = cl(b, lensed)
    n = min(len(la), len(lb))
    return la[:n], db[key][:n] / da[key][:n] - 1.0


def richardson_order(mass: str, gl: str, bands=None) -> list[dict]:
    """Measured convergence order per ell band, from the (xc, x, q2x) triple.

    Returns [] when the rung has no q2x cell. `clean` flags the bands where the
    coarse-side difference is above 0.1% -- below that both differences are at the
    round's reproducibility floor and the ratio is noise over noise."""
    x = cell_at(mass, gl)
    xc = cell_at(mass, gl, refine=2)
    q2 = cell_at(mass, gl, suffix="q2x")
    if not (x and xc and q2):
        return []
    ell, T = _tt(x)
    _, Tc = _tt(xc)
    _, Tq = _tt(q2)
    n = min(len(T), len(Tc), len(Tq))
    ell, T, Tc, Tq = ell[:n], T[:n], Tc[:n], Tq[:n]
    out = []
    for lo, hi in (bands or RICH_BANDS):
        m = (ell >= lo) & (ell < hi)
        coarse = float(np.mean(Tc[m] / T[m] - 1.0))     # f(2h) - f(h), relative
        fine = float(np.mean(T[m] / Tq[m] - 1.0))       # f(h) - f(h/2), relative
        r = coarse / fine if fine else float("nan")
        out.append(dict(band=(lo, hi), coarse=coarse, fine=fine, ratio=r,
                        p=math.log2(r) if r > 0 else float("nan"),
                        clean=abs(coarse) > 1e-3))
    return out


def cl_error(mass: str, gl: str, order: float = 2.0, lensed=True):
    """(ell, sigma) -- the production exact cell's own C_l^TT error, per ell.

    sigma = |C_l(refine 2) / C_l(refine 1) - 1| / (2^order - 1), the Richardson
    estimate at the assumed order. order = 2 is the conservative default; the
    measured order is 3.3-4.7 (see `richardson_order`), which is 4-8x tighter.
    Returns (None, None) when the rung has no refine twin."""
    x = cell_at(mass, gl)
    xc = cell_at(mass, gl, refine=2)
    if not (x and xc):
        return None, None
    ell, r = cl_diff(x, xc, lensed=lensed)
    return ell, np.abs(r) / (2.0 ** order - 1.0)


def cl_error_direct(mass: str, gl: str, lensed=True):
    """(ell, |C_l(q2x)/C_l(x) - 1|) -- the error bracketed by DOUBLING the grid
    rather than halving it. No order assumption at all, but it exists at six rungs
    only, and it moves the background as well as the perturbations."""
    x = cell_at(mass, gl)
    q2 = cell_at(mass, gl, suffix="q2x")
    if not (x and q2):
        return None, None
    ell, r = cl_diff(x, q2, lensed=lensed)
    return ell, np.abs(r)


def band_mean(ell, y, lo, hi) -> float:
    m = (ell >= lo) & (ell < hi)
    return float(np.mean(np.abs(y[m]))) if m.any() else float("nan")


# ── the background's comoving energy ─────────────────────────────────────────
NU1 = 7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0)   # one massless nu per photon rho


def ecomov(bg) -> float:
    """a^4 rho_sector(today) / a^4 rho_sector(initial).

    The parent's rest mass converted to radiation: 1 means the sector never went
    non-relativistic, and the saturated value ~1.23 is what the campaign's universal
    high-Gamma limit predicts. Identical to the definition `collect.py --bg` uses,
    so numbers here and there are comparable."""
    a = bg["a"]
    sec = sum(a ** 4 * bg[k] for k in SECTOR)
    return float(sec[-1] / sec[0])


def neff_sector(bg) -> float:
    """The sector's contribution to N_eff today, in units of one massless neutrino."""
    a = bg["a"]
    sec = sum(a ** 4 * bg[k] for k in SECTOR)
    return float(sec[-1] / (a[-1] ** 4 * bg["(.)rho_g"][-1] * NU1))


def a_extinct(bg, thresh=1e-8) -> float:
    """Where the parent's share of rho_tot last exceeds `thresh` -- 'the parent is
    gone by here'. Quantised by the table's own a grid, so read it as a decade."""
    a = bg["a"]
    ok = np.where(bg["(.)rho_dncdm1"] / bg["(.)rho_tot"] >= thresh)[0]
    return float(a[ok[-1]]) if len(ok) and ok[-1] + 1 < len(a) else float("nan")


CEIL_RE = re.compile(r"^ceil_G(?P<gl>[\d.]+)_m(?P<mass>[\d.]+)_"
                     r"(?P<gather>lin|bal|chn)_b(?P<bpd>[\d.]+)$")


def ceil_backgrounds() -> list[dict]:
    """hpc_ratchet's `ceil` family: the SAME cell at three or four bins-per-decade.

    This is the only fixed-Gamma grid ladder in the campaign, and it is what puts an
    error bar on Ecomov instead of a hope. Every row is one background table."""
    out = []
    d = ROUNDS["ratchet"].path / "out"
    if not d.is_dir():
        return out
    for f in sorted(d.glob("ceil_*_background.dat*")) + \
            sorted(d.glob("ceil_*_bgslim.dat*")):
        tag = re.sub(r"_(bgslim|background)\.dat(\.gz)?$", "", f.name)
        m = CEIL_RE.match(tag)
        if not m:
            continue
        idx = cols(f)
        with _open(f) as fh:
            dat = np.loadtxt(fh)
        bg = {k: dat[:, i] for k, i in idx.items()}
        bg["a"] = 1.0 / (1.0 + bg["z"])
        o = np.argsort(bg["a"])
        bg = {k: v[o] for k, v in bg.items()}
        out.append(dict(tag=tag, gl=m["gl"], gamma=10.0 ** float(m["gl"]),
                        mass=m["mass"], gather=m["gather"], bpd=float(m["bpd"]),
                        E=ecomov(bg), neff=neff_sector(bg)))
    out.sort(key=lambda r: (r["mass"], float(r["gl"]), r["gather"], r["bpd"]))
    return out


def gather_backgrounds(gather: str) -> dict[tuple[str, str], float]:
    """{(mass, gl): Ecomov} for one of hpc_ratchet's gathers at PRODUCTION grid.

    `lin` reproduces hpc_prod2 exactly (verified: 1.2270 / 1.2293 / 1.2244 / 2.6256
    at the four cells they share), so the lin-vs-bal gap at matched cell and matched
    grid is a clean scheme-to-scheme systematic -- the only honest error bar on the
    production Ecomov ladder above Gamma = 1e9."""
    out = {}
    d = ROUNDS["ratchet"].path / "out"
    if not d.is_dir():
        return out
    for f in sorted(d.glob(f"bg_*_{gather}_bgslim.dat*")) + \
            sorted(d.glob(f"bg_*_{gather}_background.dat*")):
        m = re.match(rf"^bg_G([\d.]+)_m([\d.]+)_{gather}_", f.name)
        if not m:
            continue
        key = (m.group(2), m.group(1))
        if key in out:
            continue
        idx = cols(f)
        with _open(f) as fh:
            dat = np.loadtxt(fh)
        bg = {k: dat[:, i] for k, i in idx.items()}
        bg["a"] = 1.0 / (1.0 + bg["z"])
        o = np.argsort(bg["a"])
        bg = {k: v[o] for k, v in bg.items()}
        out[key] = ecomov(bg)
    return out


# ── a per-bin error bar on Gamma_T, and what it buys ────────────────────────
# The published estimator cuts at gamma = 15 and treats every surviving bin as
# equally good. Both halves of that are avoidable: the refine twin gives each bin
# its OWN error, and with those weights the fit can be told about the high-gamma
# bins instead of being blind to them.
#
# Measured, pooling all 623 refine pairs, the twin disagreement per gamma bin is
#
#     gamma      2     5    10    15    22    35    58   100   200   300
#     median   0.6%  0.7%  2.1%  4.1%  8.3%  9.8%  21%   41%   80%   93%
#     sign flip  2%    0%    0%    0%    3%    5%   16%   31%   49%   58%
#
# so the data does not fall off a cliff at 15 -- it degrades smoothly, and only past
# gamma ~ 100 does it stop being a measurement (half the bins change SIGN between the
# pair). The reason is visible in the same table: `cancel` falls from 1.0 at gamma = 2
# to 2e-5 at gamma = 100, i.e. Gamma_T becomes a five-digit cancellation between the
# parent, nu_l and phi legs.
SIGMA_FLOOR = 0.02      # no bin is claimed better than 2%; the round's own floor


def sigma_rows(c: Iso, glo=2.0, ghi=np.inf, floor=SIGMA_FLOOR):
    """[(row, sigma_ln)] for the bins of `c` that have a usable twin.

    sigma_ln = |ln(GT_coarse / GT_prod)| is a RELATIVE 1-sigma, and it is
    conservative twice over: it is the coarse cell's error, not the production
    cell's, and it takes no credit for the order of the scheme."""
    t = twin(c)
    tw = {(r["glo"], r["ghi"]): r for r in (t.rows if t else [])}
    out = []
    for r in c.rows:
        g = r.get("gamma", 0.0)
        if not (glo <= g <= ghi) or g > c.gamma_max or not (r.get("GT", 0.0) > 0):
            continue
        o = tw.get((r["glo"], r["ghi"]))
        if o is None or not (o.get("GT", 0.0) > 0):
            continue
        out.append((r, max(abs(math.log(o["GT"] / r["GT"])), floor)))
    return out


def fit_p_weighted(c: Iso, glo=2.0, ghi=np.inf):
    """p by chi2 least squares in log-log, each bin weighted by its own twin error.

    Because the weights fall as the data degrades, extending ghi from 15 to 100 moves
    the fitted p by at most ~0.3 and usually under 0.15 -- which is the answer to
    "are we throwing away the high-gamma data?". We are not throwing much away."""
    re_ = sigma_rows(c, glo, ghi)
    if len(re_) < 3:
        return None
    g = np.array([r["gamma"] for r, _ in re_])
    v = np.array([r["GT"] for r, _ in re_])
    s = np.array([s for _, s in re_])
    x, y, w = np.log(g), np.log(v / gamma0_inv_mpc(c)), 1.0 / s ** 2
    A = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(A.T @ (w[:, None] * A))
    coef = cov @ (A.T @ (w * y))
    chi2 = float(np.sum(w * (y - A @ coef) ** 2) / max(1, len(x) - 2))
    return dict(p=float(-coef[1]), sd=float(math.sqrt(cov[1, 1])), n=len(g),
                chi2=chi2, gmax=float(g.max()))


def median_p_weighted(cs: list[Iso], glo=2.0, ghi=np.inf):
    fs = [f for f in (fit_p_weighted(c, glo, ghi) for c in cs) if f]
    if not fs:
        return None
    return dict(p=float(np.median([f["p"] for f in fs])),
                sd=float(np.median([f["sd"] for f in fs])),
                spread=float(np.std([f["p"] for f in fs])), n=len(fs))


# ── the control, fitted in the metric it is scored in ───────────────────────
# ⚠ THE FIRST PASS'S HEADLINE CAVEAT WAS A METRIC ARTEFACT, and this is the fix.
#
# `fit_two_power` solves an ORDINARY least squares for (C3, C5) -- it minimises the
# ABSOLUTE residual -- and then reports chi2 as the mean squared RELATIVE residual.
# `fit_one_power` fits in log space, which IS the relative metric, and reports the
# same relative chi2. So the two forms were fitted in different metrics and compared
# in one of them. Gamma_T spans three decades over the window, so an absolute fit is
# pinned by the lowest-gamma point and its relative residual at high gamma is
# enormous by construction.
#
# Measured on the 323 m6 cells, window [2, 15], IDENTICAL bins:
#
#       (3,5) fitted absolutely  ->  median relative chi2 0.053, one power wins 80%
#       (3,5) fitted relatively  ->  median relative chi2 0.007, one power wins 32%
#
# Fit both in the same metric and the two-term form wins in about two cells out of
# three. That does NOT make the exponents measured -- (3,6) and (3,7) still fit a
# 0.7-decade window nearly as well, and `compare_forms` still says so -- but the
# specific published claim "a single free power fits BETTER" does not survive.
def fit_powers(c: Iso, powers=(3.0, 5.0), qpow=1.7995, glo=2.0, ghi=GATE_GAMMA,
               weighted=True, relative=True):
    """(C_a, C_b) at fixed exponents, fitted in the metric it is scored in.

    `relative=True` minimises (residual / y), matching the reported chi2.
    `weighted=True` additionally uses each bin's own twin error. Set both False to
    reproduce the first pass's `fit_two_power` exactly."""
    if weighted:
        re_ = sigma_rows(c, glo, ghi)
        rows = [r for r, _ in re_]
        s = np.array([x for _, x in re_])
    else:
        keep, _ = gated(c, gmax=ghi)
        rows = [r for r in keep if r["gamma"] >= glo]
        s = np.ones(len(rows))
    if len(rows) < 3:
        return None
    d = rate_design(c, rows, qpow)
    if d is None:
        return None
    y, _X, g = d
    if len(y) != len(s):
        return None
    X = np.vstack([g ** -powers[0], g ** -powers[1]]).T
    scale = (s * y) if relative else np.ones_like(y)
    coef, *_ = np.linalg.lstsq(X / scale[:, None], y / scale, rcond=None)
    resid = (y - X @ coef) / y
    return dict(C3=float(coef[0]), C5=float(coef[1]), n=len(y), powers=powers,
                chi2=float(np.mean(resid ** 2)),
                chi2_w=float(np.mean((resid / s) ** 2)))


def fit_free_power(c: Iso, qpow=1.7995, glo=2.0, ghi=GATE_GAMMA, weighted=True):
    """The control: one amplitude, one free exponent, on the same weighted bins."""
    if weighted:
        re_ = sigma_rows(c, glo, ghi)
        rows = [r for r, _ in re_]
        s = np.array([x for _, x in re_])
    else:
        keep, _ = gated(c, gmax=ghi)
        rows = [r for r in keep if r["gamma"] >= glo]
        s = np.ones(len(rows))
    if len(rows) < 3:
        return None
    d = rate_design(c, rows, qpow)
    if d is None:
        return None
    y, _X, g = d
    if len(y) != len(s):
        return None
    A = np.vstack([np.ones_like(g), np.log(g)]).T
    w = 1.0 / s ** 2
    coef, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], np.log(y) * np.sqrt(w),
                               rcond=None)
    resid = 1.0 - np.exp(A @ coef) / y
    return dict(C=float(np.exp(coef[0])), p=float(-coef[1]), n=len(y),
                chi2=float(np.mean(resid ** 2)),
                chi2_w=float(np.mean((resid / s) ** 2)))


# ── the gate is Gamma-dependent, and gamma = 15 is only right at low Gamma ──
# Pooling the refine pairs by (Gamma, gamma) rather than by gamma alone shows the
# fixed gate is throwing away the data it can least afford to lose. Median
# |ln(GT_coarse/GT_prod)| in per cent:
#
#     gamma ~     1e5  1e6  1e7  1e8  1e9  1e10  1e11  1e12
#        10        27   13   10    5    4     3     1     0
#        32        48   87   14    7   10    19     9     3
#       100       199  102  232   10   16     9     4    16
#
# At Gamma = 1e5-1e7 the pair stops agreeing at gamma ~ 20, exactly where the
# published gate is. At Gamma >= 1e8 it still agrees to ~10% at gamma = 100 -- five
# times further out. And the high-Gamma end is where the lever arm is worth most,
# because that is where p is climbing toward 5.
#
# The mechanism is not mysterious: gamma large means a small, i.e. EARLY. At low
# Gamma the collision term is weak then, so Gamma_T is a tiny residual of a tiny
# quantity. At high Gamma the collision is already strong at the same epoch.
def twin_scatter(cs: list[Iso], glo=2.0) -> list[tuple[float, float]]:
    """[(gamma, |ln(GT_coarse/GT_prod)|)] pooled over the k modes of one rung."""
    out = []
    for c in cs:
        t = twin(c)
        if t is None:
            continue
        tw = {(r["glo"], r["ghi"]): r for r in t.rows}
        for r in c.rows:
            g = r.get("gamma", 0.0)
            if g < glo or g > c.gamma_max or not (r.get("GT", 0.0) > 0):
                continue
            o = tw.get((r["glo"], r["ghi"]))
            if o is None or not (o.get("GT", 0.0) > 0):
                continue
            out.append((g, abs(math.log(o["GT"] / r["GT"]))))
    return sorted(out)


def bin_scatter(cs: list[Iso], glo=2.0) -> list[tuple[float, float, int]]:
    """[(gamma, median twin disagreement, n_k)] per SHARED gamma bin for one rung.

    `reduce_iso.py` uses one fixed set of bin edges for every cell, so the k modes of
    a rung line up exactly and the median over them is a real per-bin statistic
    rather than an average over different bins."""
    per: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for c in cs:
        t = twin(c)
        if t is None:
            continue
        tw = {(r["glo"], r["ghi"]): r for r in t.rows}
        for r in c.rows:
            g = r.get("gamma", 0.0)
            if g < glo or g > c.gamma_max or not (r.get("GT", 0.0) > 0):
                continue
            o = tw.get((r["glo"], r["ghi"]))
            if o is None or not (o.get("GT", 0.0) > 0):
                continue
            per.setdefault((r["glo"], r["ghi"]), []).append(
                (g, abs(math.log(o["GT"] / r["GT"]))))
    out = []
    for _k, v in sorted(per.items()):
        out.append((float(np.median([a for a, _ in v])),
                    float(np.median([b for _, b in v])), len(v)))
    return sorted(out)


def adaptive_window(cs: list[Iso], tol=0.25, glo=2.0, min_k=3) -> float:
    """How far out in gamma this RUNG is a measurement, from its own refine pairs.

    Walk outward one SHARED gamma bin at a time and stop at the first bin whose
    median disagreement over the rung's k modes exceeds `tol`. Two deliberate
    choices:

    * a **median over k**, not a per-cell walk -- one unlucky mode is not evidence
      that the rung stopped converging, and a per-cell walk-and-stop rule collapses
      whole rungs to `glo` on exactly one such bin;
    * a **local** test, not a cumulative one -- a cumulative median stays under any
      tolerance forever, because the many well-converged low-gamma bins outvote the
      few bad ones, and it happily opens the window into the region where
      gamma^3 Gamma_T turns back up.

    Falls back to GATE_GAMMA when there is no twin evidence at all: with nothing
    measured, the published gate is the honest default, not an open window."""
    sc = bin_scatter(cs, glo)
    if not sc:
        return GATE_GAMMA
    best = glo
    for g, d, n in sc:
        if n < min_k or d > tol:
            break
        best = g
    return max(best, glo)


# ── the fit, with its covariance, and the model it predicts ─────────────────
# The model is the RTA two-power form at l = 2, where alpha_2 = beta_2 = 1:
#
#   Gamma_T/Gamma_0 = (rho_H/rho_sec) (1 - 1/gamma^2)^q [ C3 eps^n3 gamma^-3
#                                                       + C5 gamma^-5 ]
#
# Nothing about it is fitted jointly across cells. It is TWO fits, and keeping them
# straight is the whole of reading the C3 panel:
#
#   STAGE 1, per cell.  Divide out the background weight rho_H/rho_sec (interpolated
#   from that cell's own background table at each bin's mean a) and the velocity
#   shut-off (1 - 1/gamma^2)^q at fixed q = 1.7995. What is left is LINEAR in the two
#   amplitudes, so (C3 eps^n3) and C5 come from one weighted linear least squares over
#   the cell's gamma bins. The two are fitted TOGETHER -- they are not separated first.
#   What stage 1 returns as "C3" is therefore the PRODUCT C3 eps^n3, not C3.
#
#   STAGE 2, across cells.  Regress ln(C3 eps^n3) on ln(eps_H) over the whole ladder.
#   The slope IS n3 -- measured, not imposed -- and the intercept is C3.
#
# So the C3 panel is a fit to the OUTPUT of another fit, and the published n3 = 0.871
# is being tested by the slope of the second one.
def fit_with_cov(c: Iso, powers=(3.0, 5.0), qpow=1.7995, glo=2.0, ghi=30.0,
                 weighted=True):
    """Stage 1, with everything needed to judge it: covariance, per-parameter sigma,
    the C3-C5 correlation, each term's share of the model at both ends of the window,
    and the normal matrix's condition number.

    `share3_lo` is the gamma^-3 term's share of the fitted model at the LOWEST gamma
    in the window, i.e. where that term is strongest. When it falls to a per cent, C3
    is being determined from a per-cent-level component of a rate whose own systematic
    (the parent grid) is 8-25%, and a small formal sigma says nothing about that."""
    if weighted:
        re_ = sigma_rows(c, glo, ghi)
        rows = [r for r, _ in re_]
        s = np.array([x for _, x in re_])
    else:
        keep, _ = gated(c, gmax=ghi)
        rows = [r for r in keep if r["gamma"] >= glo]
        s = np.full(len(rows), SIGMA_FLOOR)
    if len(rows) < 3:
        return None
    d = rate_design(c, rows, qpow)
    if d is None:
        return None
    y, _X, g = d
    if len(y) != len(s):
        return None
    X = np.vstack([g ** -powers[0], g ** -powers[1]]).T
    W = 1.0 / (s * y) ** 2
    A = X.T @ (W[:, None] * X)
    cov = np.linalg.inv(A)
    coef = cov @ (X.T @ (W * y))
    sd = np.sqrt(np.diag(cov))
    f3, f5 = coef[0] * g ** -powers[0], coef[1] * g ** -powers[1]
    tot = f3 + f5
    resid = (y - X @ coef) / y
    return dict(C3=float(coef[0]), C5=float(coef[1]),
                sd3=float(sd[0]), sd5=float(sd[1]),
                corr=float(cov[0, 1] / (sd[0] * sd[1])),
                n=len(y), cond=float(np.linalg.cond(A)),
                share3_lo=float(f3[0] / tot[0]), share3_hi=float(f3[-1] / tot[-1]),
                chi2=float(np.mean(resid ** 2)),
                gamma=g, y=y, sigma=s, model=tot, term3=f3, term5=f5,
                powers=powers, qpow=qpow)


def rate_reduced(c: Iso, glo=2.0, ghi=30.0, qpow=1.7995):
    """(gamma, y, sigma_ln) -- the measured rate with the background weight and the
    shut-off DIVIDED OUT, i.e. the quantity the two-power model actually predicts.

    Plot this, not Gamma_T/Gamma_0: the model is a statement about y, and comparing it
    to the raw rate hides the background weight inside the 'goodness of fit'."""
    re_ = sigma_rows(c, glo, ghi)
    rows = [r for r, _ in re_]
    s = np.array([x for _, x in re_])
    if len(rows) < 3:
        return None
    d = rate_design(c, rows, qpow)
    if d is None or len(d[0]) != len(s):
        return None
    y, _X, g = d
    return g, y, s
