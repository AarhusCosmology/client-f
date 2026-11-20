#!/usr/bin/env python3
"""Reduce every family-B cell to its gamma-binned projection.  RUNS ON THE CLUSTER.

    python3 reduce_iso.py              # only cells whose dump is newer than the JSON
    python3 reduce_iso.py --clean      # redo everything
    python3 reduce_iso.py --tag T      # one cell

Writes derived/<tag>__k<i>.json (a few kB each) and derived/sweep.json.

════════════════════════════════════════════════════════════════════════════════
WHY THIS RUNS ON THE CLUSTER AND NOT AT HOME

The k_output dumps are one row per EVOLVER STEP: ~4M rows and ~40 columns per
cell, 0.7-1.4 GB gzipped at Gamma = 1e11, and the step count is exactly what
grows with Gamma.  Family B has ~120 cells x 8 modes.  Syncing that home would
dominate the campaign and buy nothing, because everything downstream -- the rate
curves, the two-window exponents, the epsilon_H collapse -- consumes only the
BINNED projection.  So each dump is read once, here, and never moved.

════════════════════════════════════════════════════════════════════════════════
THE ESTIMATOR

The claim under test is that the collision term acts as a relaxation on the
sector's total anisotropic stress,

    d(a^4 Pi)/dtau |_C  =  - a * Gamma_T * (a^4 Pi).

a^4 Pi oscillates through zero ~1e3 times over a mode's history, so a POINTWISE
ratio is 0/0 at every crossing.  Gamma_T is instead fitted by weighted least
squares inside a bin of gamma, with the conformal-time measure as the weight:

    Gamma_T = - sum_i w_i Pdot_i P_i / sum_i w_i a_i P_i^2

which is the amplitude-weighted average of the pointwise ratio -- i.e. weighted
by what actually damps the envelope -- so a crossing contributes nothing rather
than diverging.  Being LINEAR in the numerator, the parent/nu_l/phi decomposition
of the collision term carries through the fit unchanged, and
Gamma_T = Gamma_T^H + Gamma_T^l + Gamma_T^phi exactly.

R^2 of that one-parameter fit is emitted with it: it is what says whether "a
relaxation rate" is the right description at all, rather than the ansatz being
imposed on something that is in quadrature with a^4 Pi.

⚠ THE CONVERGENCE GATE IS THE POINT OF THIS ROUND.  Coarsening the perturbation
grid does not add scatter to the fitted exponent -- it FAKES A STEEPER ONE (+0.96
for a 4x coarser grid, +4.9 for 8x).  Every cell therefore has a refine = 2 twin,
and `collect.py --iso` compares them.  A gamma bin where the twins disagree is
not a measurement.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DERIVED = HERE / "derived"

# Log-spaced bins in the parent's number-weighted mean Lorentz factor.  The range
# is wide on purpose: the published gate is gamma <= 15, but this round exists to
# find out whether a converged plateau appears ABOVE it, so the bins must extend
# past the gate rather than assume it.
EDGES = np.geomspace(1.2, 400.0, 25)
MIN_PER_BIN = 20

COLS = ["a", "gamma_dncdm1",
        "a4Pi_dncdm1", "a4Pi_dncdm1_H", "a4Pi_dncdm1_l", "a4Pi_dncdm1_phi",
        "a4PiDot_coll_dncdm1", "a4PiDot_coll_dncdm1_H",
        "a4PiDot_coll_dncdm1_l", "a4PiDot_coll_dncdm1_phi",
        "delta_dncdm1_H", "delta_dncdm1_l", "delta_dncdm1_phi"]


def _open(p):
    return gzip.open(p, "rt") if str(p).endswith(".gz") else open(p)


def header(path):
    """Column name -> 0-based POSITION.

    Position, never the printed index: CLASS writes some tables with two columns
    both claiming index 1, and this reader is shared with those."""
    with _open(path) as f:
        f.readline()                       # "#scalar perturbations for mode k ="
        hdr = f.readline()
    return {n: int(i) - 1 for i, n in re.findall(r"(\d+):(\S+)", hdr)}


def tau_weights(tau):
    w = np.empty_like(tau)
    w[1:-1] = 0.5 * (tau[2:] - tau[:-2])
    w[0] = tau[1] - tau[0]
    w[-1] = tau[-1] - tau[-2]
    return w


def project(w, a, P, Pdot):
    """One-parameter relaxation fit.  Returns (Gamma_T, R2, corr)."""
    den = np.sum(w * a * P * P)
    if den <= 0:
        return float("nan"), float("nan"), float("nan")
    GT = -np.sum(w * Pdot * P) / den
    resid = Pdot + a * GT * P
    ss = np.sum(w * Pdot * Pdot)
    R2 = 1.0 - np.sum(w * resid * resid) / ss if ss > 0 else float("nan")
    x, y = -a * P, Pdot
    sw = np.sum(w)
    cx, cy = x - np.sum(w * x) / sw, y - np.sum(w * y) / sw
    corr = np.sum(w * cx * cy) / math.sqrt(np.sum(w * cx * cx) * np.sum(w * cy * cy))
    return GT, R2, corr


def read_cell(path):
    cols = header(path)
    with _open(path) as f:
        dat = np.loadtxt(f, comments="#")
    d = {"tau": dat[:, 0]}
    for c in COLS:
        if c in cols:
            d[c] = dat[:, cols[c]]
    return d


def derive(d, k):
    tau, a, g = d["tau"], d["a"], d["gamma_dncdm1"]
    P = d["a4Pi_dncdm1"]
    w_all = tau_weights(tau)
    comps = {c: d.get("a4PiDot_coll_dncdm1" + c) for c in ("", "_H", "_l", "_phi")}

    # THE ZERO-TAIL CHECK.  A REDUCED cell applies the collision operator only
    # inside the table's a-window, so a4PiDot_coll is exactly 0 outside it.  That
    # is harmless when the window covers the whole fit range -- measured at
    # Gamma = 1e5, the zeros sit entirely at gamma = 1.000, long after the decay
    # -- and fatal if it does not, because a bin of zeros fits Gamma_T = 0 and
    # looks like a converged measurement.  Record where the support ends so
    # collect.py can refuse to fit past it.
    nz = np.nonzero(comps[""] != 0.0)[0]
    support = {"gamma_max": float(g[nz].max()) if len(nz) else float("nan"),
               "gamma_min_nonzero": float(g[nz].min()) if len(nz) else float("nan"),
               "n_nonzero": int(len(nz)), "n_rows": int(len(g))}

    rows = []
    for i in range(len(EDGES) - 1):
        m = (g >= EDGES[i]) & (g < EDGES[i + 1]) & np.isfinite(P)
        if m.sum() < MIN_PER_BIN:
            continue
        ww, aa, PP = w_all[m], a[m], P[m]
        wt = ww * PP * PP                       # amplitude weight for gamma-bar
        r = {"glo": float(EDGES[i]), "ghi": float(EDGES[i + 1]), "n": int(m.sum()),
             "gamma": float(np.sum(wt * g[m]) / np.sum(wt)),
             "a": float(np.sum(wt * aa) / np.sum(wt)),
             "ktau": float(np.sum(wt * k * tau[m]) / np.sum(wt)),
             # Fraction of this bin that carries ANY collision term.  < 1 means
             # the bin straddles the edge of a reduced cell's table window.
             "frac_support": float(np.mean(comps[""][m] != 0.0))}
        for c, arr in comps.items():
            if arr is None:
                continue
            GT, R2, corr = project(ww, aa, PP, arr[m])
            r["GT" + c] = float(GT)
            if c == "":
                r["R2"], r["corr"] = float(R2), float(corr)
        if all(f"GT_{n}" in r for n in ("H", "l", "phi")):
            # How much of the total is cancellation between the three legs.  A
            # value far below 1 means Gamma_T is a small residual of much larger
            # terms and its own error bar is not its components'.
            denom = abs(r["GT_H"]) + abs(r["GT_l"]) + abs(r["GT_phi"])
            r["cancel"] = float(abs(r["GT"]) / denom) if denom > 0 else float("nan")
        rows.append(r)
    return {"rows": rows, "support": support}


def cells(manifest):
    import csv
    with open(manifest) as fh:
        return [r for r in csv.DictReader(fh) if r["family"] == "iso"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--tag")
    a = ap.parse_args()

    DERIVED.mkdir(exist_ok=True)
    man = HERE / "manifest.csv"
    if not man.exists():
        print("no manifest.csv -- run gen.py first", file=sys.stderr)
        return 1
    import config as C
    kvals = [float(x) for x in C.K_OUTPUT_ISO]

    n_new = n_skip = n_missing = 0
    for meta in cells(man):
        tag = meta["tag"]
        if a.tag and tag != a.tag:
            continue
        for i, kv in enumerate(kvals):
            src = OUT / f"{tag}_perturbations_k{i}_s.dat.gz"
            if not src.exists():
                src = OUT / f"{tag}_perturbations_k{i}_s.dat"
            if not src.exists():
                n_missing += 1
                continue
            dst = DERIVED / f"{tag}__k{i}.json"
            if (not a.clean and dst.exists()
                    and dst.stat().st_mtime >= src.stat().st_mtime):
                n_skip += 1
                continue
            try:
                rec = derive(read_cell(src), kv)
            except Exception as e:                       # noqa: BLE001
                print(f"  ! {tag} k{i}: {e}", file=sys.stderr)
                continue
            rec.update({"tag": tag, "k": kv, "k_index": i,
                        "log10Gamma": meta["log10Gamma"],
                        "Gamma": float(meta["gamma"]), "mass": float(meta["mass"]),
                        "scheme": meta["scheme"], "moments": int(meta["moments"]),
                        "refine": int(meta["refine"]), "dr_N_q": int(meta["dr_N_q"]),
                        "n_pt": int(meta["n_pt"]),
                        "momenta_bins": int(meta["momenta_bins"])})
            dst.write_text(json.dumps(rec))
            n_new += 1
            print(f"  + {tag} k{i}  ({len(rec['rows'])} bins, "
                  f"support to gamma={rec['support']['gamma_max']:.1f})", flush=True)

    allc = {}
    for f in sorted(DERIVED.glob("*__k*.json")):
        r = json.loads(f.read_text())
        allc[f"{r['tag']}|{r['k_index']}"] = r
    (DERIVED / "sweep.json").write_text(json.dumps(allc))
    size = sum(f.stat().st_size for f in DERIVED.glob("*.json")) / 1024
    print(f"\nderived: {n_new} rebuilt, {n_skip} up to date, {n_missing} dumps absent")
    print(f"derived/: {len(allc)} (cell, k) records, {size:.0f} kB total")
    print("sync_from_hpc.sh brings derived/ home; the dumps stay here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
