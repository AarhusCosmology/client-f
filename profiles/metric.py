"""The profile metric: the credible metric's construction applied to profile crossings.

    python -m profiles.metric profiles_it10.json --reference exact_profiles.json
    python -m profiles.metric exact_profiles.json --floor

THE DEFINITION IS BORROWED, NOT INVENTED. The credible metric scores a posterior interval by its summed endpoint error in units of the reference interval's width,

    dCM = (|s_hi - r_hi| + |s_lo - r_lo|) / (r_hi - r_lo)

and lets the REFERENCE decide what kind of interval a parameter has, the surrogate then
being measured against that choice. This module applies the identical formula with the
endpoints taken to be the crossings of Delta chi-squared = 1 and 4, so that dPM and dCM
are directly comparable numbers.

INTERVAL KINDS, decided by the reference exactly as in the credible metric:

  two-sided      the reference crosses the level on both sides; both endpoints enter.
  one-sided      it crosses once; the open side is anchored at the grid edge for BOTH
                 curves, so that side contributes nothing and the metric measures the
                 crossing that exists.
  unconstrained  the entire sampled grid is below or at the level. Score zero only
                 when the emulator is also unconstrained; otherwise flag a spurious
                 constraint and score its endpoint displacement over the grid width.

Empty, point-like and disconnected confidence sets have no single-interval dPM.
They are flagged with NaN scores and reported as exclusions, not silently replaced
by their outer envelope. ``confidence_set`` exposes every connected component.

A crossing the EMULATOR fails to produce is anchored at the grid edge, which makes dPM
large rather than undefined. That is the correct direction: failing to constrain a
parameter the reference constrains is an error, not a missing measurement.

CHECK --floor WHEN INTERPRETING SMALL DIFFERENCES. The metric is computed by linearly
interpolating a locally parabolic curve between grid points, which systematically
underestimates the interval width. On a 25-point grid over a +/-10 sigma box that error is
2-8% at 1 sigma -- as large as the metric values themselves. ``--floor`` measures it
directly on a reference whose closed form is known: where the box does not bind, the true
profile is (theta-mu)^2/Sigma_ii and the crossing sits at sqrt(level)*sigma_marg,
estimated here by a quadratic fit allowing an off-grid centre. Non-quadratic regions
return no estimate. This is a reference interpolation error, not a hard lower bound
on dPM: common errors can cancel between the two interpolated curves. Refine crossing
brackets to validate small differences.

Both profiles must share a grid; the module checks this and refuses otherwise.
"""
import argparse
import json
import statistics as st
import warnings
from pathlib import Path

import numpy as np

LEVELS = (("1 sigma", 1.0), ("2 sigma", 4.0))


def _validated(grid, dchi2, level):
    g, d = np.asarray(grid, dtype=float), np.asarray(dchi2, dtype=float)
    if (g.ndim != 1 or d.shape != g.shape or len(g) < 2
            or not np.isfinite(g).all() or not np.isfinite(d).all()
            or not np.isfinite(level) or np.any(np.diff(g) <= 0)):
        raise ValueError("profile requires finite values on a strictly increasing grid")
    return g, d


def crossings(grid, dchi2, level):
    """Roots of the piecewise-linear profile, including both endpoints, without repeats."""
    g, d = _validated(grid, dchi2, level)
    out = list(g[d == level])
    for j, (a, b) in enumerate(zip(d[:-1], d[1:])):
        if (a < level < b) or (b < level < a):
            out.append(g[j] + (level - a) / (b - a) * (g[j + 1] - g[j]))
    return sorted(set(float(x) for x in out))


def confidence_set(grid, dchi2, level):
    """Closed components of {x: interpolated Delta chi-square(x) <= level}.

    Empty and disconnected sets are explicit; no outer-envelope approximation.
    This describes the sampled interpolant, not unresolved features between nodes.
    """
    g, d = _validated(grid, dchi2, level)
    components = []
    for j, (a, b) in enumerate(zip(d[:-1], d[1:])):
        left, right = float(g[j]), float(g[j + 1])
        if a > level and b > level:
            continue
        if a > level:
            left += (level - a) / (b - a) * (right - left)
        elif b > level:
            right = left + (level - a) / (b - a) * (right - left)
        if components and left <= components[-1][1]:
            components[-1] = (components[-1][0], right)
        else:
            components.append((left, right))
    return components


def interval(grid, dchi2, level, lo, hi):
    """Legacy endpoint interface; unsupported sets return NaNs and an explicit kind."""
    g, d = _validated(grid, dchi2, level)
    if lo != g[0] or hi != g[-1]:
        raise ValueError("interval bounds must match the sampled grid; cannot extrapolate")
    parts = confidence_set(g, d, level)
    if not parts:
        return float("nan"), float("nan"), "empty"
    if len(parts) > 1:
        return float("nan"), float("nan"), "disconnected"
    left, right = parts[0]
    if left == right:
        return left, right, "point"
    if left == lo and right == hi:
        return left, right, "unconstrained"
    if left == lo:
        return left, right, "upper limit"
    if right == hi:
        return left, right, "lower limit"
    return left, right, "two-sided"


def as_dchi2(prof, name):
    """Delta chi-squared on the grid, from either an emulator or an exact profile file."""
    rec = prof["profiles"][name]
    if "dchi2" in rec:                  # exact profile, already a Delta chi-squared
        return np.asarray(rec["dchi2"], dtype=float)
    return 2.0 * (prof["best_logl"] - np.asarray(rec["logl"], dtype=float))


def dpm(emulator, reference, names=None):
    """Per-level, per-parameter dPM: {level: {param: (value, kind)}}."""
    for label, profile in (("emulator", emulator), ("reference", reference)):
        quality = profile.get("optimization_quality")
        if quality is not None and not quality.get("converged", False):
            warnings.warn(f"{label} profile has incomplete optimization checks; scores are provisional",
                          RuntimeWarning, stacklevel=2)
    names = names or [n for n in emulator["names"] if n in reference["profiles"]]
    out = {lab: {} for lab, _ in LEVELS}
    for nm in names:
        gr = np.asarray(reference["profiles"][nm]["grid"], dtype=float)
        ge = np.asarray(emulator["profiles"][nm]["grid"], dtype=float)
        # Absolute tolerance scaled by the grid span, not by the parameter offset.
        tol = 1e-10 * np.ptp(gr) if gr.size else 0.0
        if gr.shape != ge.shape or not np.allclose(gr, ge, rtol=0, atol=tol):
            raise ValueError(
                f"{nm}: the emulator and reference profiles use different grids. dPM "
                "compares interpolated crossings and is only meaningful when both share "
                "a grid -- recompute one of them with matching --n-grid/--grid-frac.")
        dr, de = as_dchi2(reference, nm), as_dchi2(emulator, nm)
        lo, hi = float(gr[0]), float(gr[-1])
        for lab, level in LEVELS:
            r_lo, r_hi, kind = interval(gr, dr, level, lo, hi)
            s_lo, s_hi, s_kind = interval(ge, de, level, float(ge[0]), float(ge[-1]))
            invalid = {"empty", "disconnected", "point"}
            if kind in invalid or s_kind in invalid:
                flag = f"reference {kind}" if kind in invalid else f"emulator {s_kind}"
                warnings.warn(f"{nm}, {lab}: {flag}; single-interval dPM is undefined",
                              RuntimeWarning, stacklevel=2)
                out[lab][nm] = (float("nan"), flag)
                continue
            if kind == "unconstrained" and s_kind != "unconstrained":
                kind = "spurious constraint"
                warnings.warn(f"{nm}, {lab}: emulator constrains an unconstrained reference",
                              RuntimeWarning, stacklevel=2)
            if kind == "upper limit":
                s_lo, r_lo = lo, lo      # the anchored side contributes nothing
            elif kind == "lower limit":
                s_hi, r_hi = hi, hi
            width = r_hi - r_lo
            out[lab][nm] = ((abs(s_hi - r_hi) + abs(s_lo - r_lo)) / width
                            if width > 0 else float("nan"), kind)
    return out


def summarise(per_level, drop_unconstrained=True):
    """(median, max, n) over the parameters that carry information, per level."""
    summary = {}
    for lab, d in per_level.items():
        vals = [v for v, k in d.values()
                if np.isfinite(v) and not (drop_unconstrained and k == "unconstrained")]
        summary[lab] = (st.median(vals), max(vals), len(vals)) if vals \
            else (float("nan"), float("nan"), 0)
    return summary


def resolution_floor(reference, name, level):
    """Interpolation error on the reference itself, in the same units as dPM.

    Only meaningful where the reference profile is an unbinding Gaussian marginal, i.e.
    (theta-mu)^2/Sigma_ii. Returns None where the level is not crossed.
    """
    g, d = _validated(reference["profiles"][name]["grid"],
                      as_dchi2(reference, name), level)
    left, right, kind = interval(g, d, level, g[0], g[-1])
    if kind != "two-sided":
        return None
    # Fit the bracketed region, including one node beyond each crossing. This
    # estimates an off-grid centre and rejects visibly non-quadratic profiles.
    i = max(0, int(np.searchsorted(g, left)) - 1)
    j = min(len(g), int(np.searchsorted(g, right, side="right")) + 1)
    if j - i < 3:
        return None
    centre, scale = 0.5 * (g[i] + g[j-1]), 0.5 * (g[j-1] - g[i])
    u, vals = (g[i:j] - centre) / scale, d[i:j]
    a, b, c = np.polyfit(u, vals, 2)
    tolerance = 1e-7 * max(1.0, float(np.max(np.abs(vals))))
    if a <= 0 or np.max(np.abs(np.polyval([a, b, c], u) - vals)) > tolerance:
        return None
    minimum = c - b*b / (4*a)
    if abs(minimum) > tolerance or level <= minimum:
        return None
    exact = 2.0 * scale * np.sqrt((level - minimum) / a)
    return float(abs((right - left) - exact) / exact)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("profiles", nargs="+", help="emulator profile json file(s)")
    p.add_argument("-r", "--reference", help="reference profile json (exact or true-run)")
    p.add_argument("--floor", action="store_true",
                   help="instead of scoring, measure the grid's interpolation floor on "
                        "the first file given, which must be an exact profile")
    a = p.parse_args()

    if a.floor:
        ref = json.load(open(a.profiles[0]))
        print(f"grid-interpolation floor for {Path(a.profiles[0]).name} "
              f"({ref.get('n_grid', '?')} points, "
              f"grid_frac {ref.get('grid_frac', '?')})")
        print(f"{'parameter':>18}{'1 sigma':>10}{'2 sigma':>10}")
        cols = {lab: [] for lab, _ in LEVELS}
        for nm in ref["names"]:
            cells = []
            for lab, lev in LEVELS:
                f = resolution_floor(ref, nm, lev)
                cells.append("--" if f is None else f"{f:.4f}")
                if f is not None:
                    cols[lab].append(f)
            print(f"{nm:>18}{cells[0]:>10}{cells[1]:>10}")
        for lab, _ in LEVELS:
            if cols[lab]:
                print(f"  median {lab} floor: {st.median(cols[lab]):.4f}")
        print("\nThese are reference interpolation errors, not hard dPM floors: common "
              "errors can cancel. Refine crossing brackets to check small differences. "
              "-- means no valid unbound quadratic estimate.")
        return

    if not a.reference:
        p.error("--reference is required unless --floor is given")
    ref = json.load(open(a.reference))
    for f in a.profiles:
        emul = json.load(open(f))
        per = dpm(emul, ref)
        print(f"\n=== {Path(f).name} ===")
        print(f"{'parameter':>18}{'dPM 1s':>9}  {'kind 1s':<14}{'dPM 2s':>9}  kind 2s")
        for nm in per["1 sigma"]:
            v1, k1 = per["1 sigma"][nm]
            v2, k2 = per["2 sigma"][nm]
            print(f"{nm:>18}{v1:>9.4f}  {k1:<14}{v2:>9.4f}  {k2}")
        for lab, _ in LEVELS:
            med, mx, n = summarise(per)[lab]
            print(f"  {lab}: median {med:.4f}, max {mx:.4f}  "
                  f"(over {n} informative parameters)")
            excluded = sum(not np.isfinite(v) for v, _ in per[lab].values())
            if excluded:
                print(f"  WARNING: {excluded} undefined {lab} score(s) excluded; see flags above")


if __name__ == "__main__":
    main()
