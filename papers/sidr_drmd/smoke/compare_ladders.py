"""Two builds (or settings) at the same points: the spectra files smoke/kmax_ladder.py writes, compared
point by point and k_max by k_max.

    python smoke/compare_ladders.py logs/kmax_ladder_A.npz logs/kmax_ladder_B.npz

Prints how many points are finite in each, how many give identical spectra, and for those finite in both the
largest |C_l^B / C_l^A - 1| and a cosmic-variance-limited chi^2 (f_sky = 1): sum_l (2l+1)/2 (dC_l/C_l)^2 over
TT l <= 2500, EE l <= 2000, lensing potential 8 <= L <= 400. That is an upper bound on what any likelihood of
these spectra can see; the points that differ are listed with it. Made 21 October 2026 (the extrapolation guard).
"""
import sys

import numpy as np

RANGES = (("TT", 0, 2, 2500), ("EE", 1, 2, 2000), ("pp", 2, 8, 400))   # spectra rows: tt, ee, pp at l = 2..2500


def main(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    key = lambda d: {(int(r), float(k)): i for i, (r, k) in enumerate(zip(d["row"], d["kmax"]))}
    ka, kb = key(a), key(b)
    common = sorted(set(ka) & set(kb))
    ell = np.arange(2, 2501)
    print(f"{a_path} (A) against {b_path} (B): {len(common)} evaluations in common")
    finite_a = sum(bool(a["finite"][ka[c]]) for c in common)
    finite_b = sum(bool(b["finite"][kb[c]]) for c in common)
    identical, rows = 0, []
    for c in common:
        i, j = ka[c], kb[c]
        if not (a["finite"][i] and b["finite"][j]):
            continue
        x, y = a["spectra"][i], b["spectra"][j]
        if np.array_equal(x, y):
            identical += 1
            continue
        parts = []
        for name, s, lo, hi in RANGES:
            m = (ell >= lo) & (ell <= hi)
            r = y[s, m] / x[s, m] - 1
            parts.append((name, float(np.max(np.abs(r))), float(np.sum((2 * ell[m] + 1) / 2 * r ** 2))))
        rows.append((c, parts))
    print(f"finite: A {finite_a}, B {finite_b}; finite in both and identical: {identical}; finite in both and different: {len(rows)}")
    worst = {name: (0.0, 0.0) for name, *_ in RANGES}
    for (row, kmax), parts in rows:
        print(f"   row {row:6d} k_max {kmax:g}: " + "; ".join(f"{n} max |dC/C| {d:.1e}, CV chi2 {x2:.1e}" for n, d, x2 in parts))
        for n, d, x2 in parts:
            worst[n] = (max(worst[n][0], d), max(worst[n][1], x2))
    if rows:
        print("largest over the points that differ: " + "; ".join(f"{n} |dC/C| {d:.1e}, CV chi2 {x2:.1e}" for n, (d, x2) in worst.items()))


if __name__ == "__main__":
    main(*sys.argv[1:3])
