#!/usr/bin/env python3
"""Reduce every family-B cell to its gamma-binned projection.  RUNS ON THE CLUSTER.

    python3 reduce_iso.py              # only cells whose dump is newer than the JSON
    python3 reduce_iso.py --clean      # redo everything
    python3 reduce_iso.py --tag T      # one cell
    python3 reduce_iso.py -j 16        # 16 dumps at once ($SLURM_CPUS_PER_TASK, else 4)
    python3 reduce_iso.py --probe F    # where the time goes on THIS filesystem

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

════════════════════════════════════════════════════════════════════════════════
WHY THE READER IS HAND-WRITTEN AND NOT numpy.loadtxt

This script's whole cost is turning ~2-6 GB of ASCII per cell into 14 columns of
float64, and gzip.open + numpy.loadtxt was doing it at 0.19 Mrow/s.  Three things
fix that.  All numbers below are measured on one real dump -- 1.93 Mrow, 1.84 GB
inflated, 482 MB gzipped -- on an M3 Pro with the file in page cache, so they are
the CPU floor with the filesystem taken out of the picture:

  1. THE DUMPS ARE A FIXED-WIDTH GRID.  CLASS writes every field as %25.12e, so
     field c of row r is ALWAYS at byte 952*r + 25*c, and the sign, the 13
     digits and the two exponent digits are at fixed offsets inside the field.
     Parsing is then pure numpy arithmetic on a uint8 view -- no tokenising, no
     per-line Python -- and only the 14 of 38 columns the estimator reads are
     ever touched.  `_layout_ok` re-checks the assumption on every dump and falls
     back to loadtxt, loudly, if one is ever written some other way.

  2. THE INFLATE AND THE PARSE OVERLAP.  zlib releases the GIL, so a reader
     thread inflates block n+1 while the main thread parses block n: read+inflate
     is 2.89 s, parse is ~1.4 s, and the pair costs 2.99 s rather than 4.27 s.

  3. READS ARE 32 MB, UNBUFFERED, SEQUENTIAL-ADVISED.  gzip.open() pulls the
     compressed file through an 8 KiB buffer, which is free on an SSD and is the
     classic way to get single-digit MB/s out of Lustre: 8 KiB of a striped file
     is one RPC round trip and there are 130000 of them in a 1 GB dump.  One
     32 MB read is one burst.

     ⚠ AND WATCH FOR THE COPY.  `rem + block` to re-join a row split across two
     blocks, and `list(blocks)` to iterate them, are both invisible in a profile
     and cost more than everything above: the first memcpy'd 122 MB per block
     (1.3 s of the 4.3), the second held the whole inflated 6 GB dump in RAM.
     Blocks are passed as memoryviews now.  Only the straddling row is copied.

Net per dump: 11.13 s -> 3.14 s, and derive() is 4.4x on top of that (one
np.digitize instead of three full sweeps of a 6M-row column inside each of 24
bins).  The parser is exact against loadtxt to 1 ULP: 2.2e-16 per column, 1.5e-13
in the fitted Gamma_T, verified end to end against the previous round's
derived/*.json.  That is ten orders below what this estimator resolves.

WHETHER ANY OF IT MATTERS ON A GIVEN FILESYSTEM IS A MEASUREMENT, NOT A GUESS:
`--probe out/<dump>.gz` prints read / inflate / parse separately.  If the read
line is most of the total, the CPU work above is irrelevant and the answer is to
stripe the dumps wider or stage them to node-local scratch; if it is not, add
cores with -j.  Reduction is one process per dump, biggest first.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
import os
import queue
import re
import sys
import threading
import time
import zlib
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

READ_CHUNK = 32 << 20        # one big read, not 4096 small ones -- see the docstring
FIELD_W = 25                 # CLASS writes %25.12e


# ══════════════════════════════════════════════════════════════════════════════
# READER

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


def _read_inflate(path, chunk):
    """Read `path` in `chunk`-sized bursts, inflating on the fly if it is .gz."""
    gz = str(path).endswith(".gz")
    dec = zlib.decompressobj(31) if gz else None
    with open(path, "rb", buffering=0) as f:
        if hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            except OSError:
                pass
        while True:
            comp = f.read(chunk)
            if not comp:
                break
            out = dec.decompress(comp) if gz else comp
            if out:
                yield out
    if gz:
        tail = dec.flush()
        if tail:
            yield tail


def _blocks(path, chunk=READ_CHUNK, overlap=True):
    """Yield the uncompressed bytes of `path` in large blocks.

    With overlap=True the read+inflate runs on its own thread; zlib drops the GIL
    for the inflate, so it proceeds while the caller is parsing the last block."""
    if not overlap:
        yield from _read_inflate(path, chunk)
        return

    q: queue.Queue = queue.Queue(maxsize=2)
    err = []

    def run():
        try:
            for blk in _read_inflate(path, chunk):
                q.put(blk)
        except BaseException as e:                       # noqa: BLE001
            err.append(e)
        finally:
            q.put(None)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    while True:
        blk = q.get()
        if blk is None:
            break
        yield blk
    th.join()
    if err:
        raise err[0]


with np.errstate(over="ignore"):                 # |exp| > 308 is a real overflow,
    _POW10 = 10.0 ** np.arange(-340, 341)        # not a parser bug -- index: exp + 340
_MULT = (10 ** np.arange(12, -1, -1)).astype(np.int64)


def _parse(block, rb, want, out_list):
    """block: a whole number of rb-byte rows.  Appends one (len(want), n) array.

    Layout of a %25.12e field, counting back from its last byte e:
        e-19 sign(' '/'-')  e-18 digit  e-17 '.'  e-16..e-5 12 digits
        e-4 'e'  e-3 expsign  e-2,e-1 two exponent digits
    """
    b = np.frombuffer(block, np.uint8).reshape(-1, rb)
    n = b.shape[0]
    out = np.empty((len(want), n), np.float64)
    dig = np.empty((n, 13), np.int64)
    for j, c in enumerate(want):
        f = b[:, FIELD_W * (c + 1) - 19:FIELD_W * (c + 1)]
        dig[:, 0] = f[:, 1]
        dig[:, 1:] = f[:, 3:15]
        dig -= 48
        ex = (f[:, 17].astype(np.int64) - 48) * 10 + (f[:, 18] - 48)
        np.negative(ex, out=ex, where=(f[:, 16] == 45))          # '-'
        v = (dig @ _MULT) * _POW10[ex - 12 + 340]
        np.negative(v, out=v, where=(f[:, 0] == 45))
        out[j] = v
    out_list.append(out)
    return n


def _layout_ok(row, rb, ncol):
    """Is this dump the fixed-width %25.12e grid _parse assumes?"""
    if rb != FIELD_W * ncol + 2 or row[-1:] != b"\n":
        return False
    for c in range(ncol):
        f = row[FIELD_W * (c + 1) - 19:FIELD_W * (c + 1)]
        if (f[0:1] not in b" -" or f[1:2] not in b"0123456789" or f[2:3] != b"."
                or not f[3:15].isdigit() or f[15:16] != b"e"
                or f[16:17] not in b"+-" or not f[17:19].isdigit()):
            return False
    return True


def read_cell(path, progress=None):
    """{name: float64 array} for 'tau' plus every COLS entry the dump carries."""
    blocks = _blocks(path)
    head = b""
    for blk in blocks:                       # block 0 always holds both header lines
        head += blk
        if head.count(b"\n") >= 3:
            break
    else:
        raise ValueError(f"{path}: fewer than three lines")

    nl1 = head.index(b"\n")
    nl2 = head.index(b"\n", nl1 + 1)
    nl3 = head.index(b"\n", nl2 + 1)
    cols = {n: int(i) - 1 for i, n in re.findall(r"(\d+):(\S+)", head[nl1 + 1:nl2].decode())}
    rb = nl3 - nl2                                     # bytes per data row, '\n' included
    names = ["tau"] + [c for c in COLS if c in cols and c != "tau"]
    want = [cols[n] for n in names]

    if not _layout_ok(head[nl2 + 1:nl3 + 1], rb, len(cols)):
        print(f"  ! {Path(path).name}: not the fixed-width %25.12e grid -- "
              f"falling back to numpy.loadtxt (slow)", file=sys.stderr, flush=True)
        with _open(path) as f:
            dat = np.loadtxt(f, comments="#", usecols=want)
        return dict(zip(names, dat.T.copy()))

    # Blocks are handed to _parse as memoryviews.  `rem + blk` would look
    # innocent and copy the whole 122 MB block on every iteration, and
    # `list(blocks)` would hold the entire inflated 6 GB dump at once; both
    # showed up as unexplained seconds and gigabytes before they were removed.
    parts, rem, done = [], b"", 0
    for mv in itertools.chain([memoryview(head)[nl2 + 1:]],
                              (memoryview(b) for b in blocks)):
        if rem:                                        # row straddling a block edge
            need = rb - len(rem)
            if len(mv) < need:
                rem += bytes(mv)
                continue
            done += _parse(rem + bytes(mv[:need]), rb, want, parts)
            mv, rem = mv[need:], b""
        n = len(mv) // rb
        if n:
            done += _parse(mv[:n * rb], rb, want, parts)
            if progress:
                progress(done)
        rem = bytes(mv[n * rb:])
    if rem.strip():
        raise ValueError(f"{path}: {len(rem)} trailing bytes are not a whole row")
    dat = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
    return dict(zip(names, dat))


# ══════════════════════════════════════════════════════════════════════════════
# ESTIMATOR  (unchanged; agrees with the loadtxt reader to 1.5e-13)

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

    # One pass for the bin index and one for finiteness, instead of three full
    # sweeps of a 6M-row column inside each of 24 bins.  Membership, and so the
    # order of every sum below, is identical to the mask it replaces.
    ibin = np.digitize(g, EDGES) - 1
    finite = np.isfinite(P)
    supported = comps[""] != 0.0

    rows = []
    for i in range(len(EDGES) - 1):
        m = (ibin == i) & finite
        nm = int(m.sum())
        if nm < MIN_PER_BIN:
            continue
        ww, aa, PP = w_all[m], a[m], P[m]
        wt = ww * PP * PP                       # amplitude weight for gamma-bar
        swt = np.sum(wt)
        r = {"glo": float(EDGES[i]), "ghi": float(EDGES[i + 1]), "n": nm,
             "gamma": float(np.sum(wt * g[m]) / swt),
             "a": float(np.sum(wt * aa) / swt),
             "ktau": float(np.sum(wt * k * tau[m]) / swt),
             # Fraction of this bin that carries ANY collision term.  < 1 means
             # the bin straddles the edge of a reduced cell's table window.
             "frac_support": float(np.mean(supported[m]))}
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


# ══════════════════════════════════════════════════════════════════════════════
# DRIVER

def cells(manifest):
    with open(manifest) as fh:
        return [r for r in csv.DictReader(fh) if r["family"] == "iso"]


def _fmt(sec):
    sec = int(sec)
    return f"{sec//3600:d}h{sec//60%60:02d}m" if sec >= 3600 else f"{sec//60:d}m{sec%60:02d}s"


def _ticker(src):
    """Intra-file progress.  A terminal gets a repainted line twice a second; a
    SLURM .out gets one whole line a minute, because the alternative is 174
    carriage returns per dump in a file nobody can then read."""
    tty = sys.stderr.isatty()
    every = 0.5 if tty else 60.0
    name, nrow = Path(src).name, os.path.getsize(src) / 952.0   # rows if uncompressed
    if src.endswith(".gz"):
        nrow *= 4.0                                             # measured gzip ratio
    state = [time.perf_counter(), time.perf_counter()]

    def tick(n):
        now = time.perf_counter()
        if now - state[0] < every:
            return
        state[0] = now
        rate = n / max(now - state[1], 1e-9)
        line = (f"    {name}: {n/1e6:.2f} Mrow (~{100*n/max(nrow,1):.0f}%), "
                f"{rate/1e3:.0f} krow/s")
        print(f"\r{line:<78s}" if tty else line, end="" if tty else "\n",
              file=sys.stderr, flush=True)
    return tick


def one(task):
    """Reduce a single (cell, k).  Runs in a worker process; writes its own JSON."""
    src, dst, kv, meta, verbose = task
    t0 = time.perf_counter()
    try:
        rec = derive(read_cell(src, _ticker(src) if verbose else None), kv)
        if verbose and sys.stderr.isatty():
            print("\r" + " " * 80 + "\r", end="", file=sys.stderr, flush=True)
    except Exception as e:                                   # noqa: BLE001
        return (dst, None, f"{type(e).__name__}: {e}", 0.0)
    rec.update({"tag": meta["tag"], "k": kv, "k_index": meta["k_index"],
                "log10Gamma": meta["log10Gamma"],
                "Gamma": float(meta["gamma"]), "mass": float(meta["mass"]),
                "scheme": meta["scheme"], "moments": int(meta["moments"]),
                "refine": int(meta["refine"]), "dr_N_q": int(meta["dr_N_q"]),
                "n_pt": int(meta["n_pt"]),
                "momenta_bins": int(meta["momenta_bins"])})
    Path(dst).write_text(json.dumps(rec))
    return (dst, rec, None, time.perf_counter() - t0)


def probe(path):
    """Where does the time go ON THIS FILESYSTEM?  Answers IO-bound vs CPU-bound."""
    sz = os.path.getsize(path)
    print(f"probe: {path}  ({sz/1e6:.0f} MB on disk)\n")

    t0 = time.perf_counter(); n = 0
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(READ_CHUNK)
            if not b:
                break
            n += len(b)
    t_read = time.perf_counter() - t0
    print(f"  1. read only, {READ_CHUNK>>20} MB blocks  {t_read:8.2f} s "
          f"{n/t_read/1e6:8.1f} MB/s   <-- the filesystem, nothing else")

    t0 = time.perf_counter(); raw = 0
    for blk in _blocks(path, overlap=False):
        raw += len(blk)
    t_inf = time.perf_counter() - t0
    print(f"  2. + inflate ({raw/1e9:5.2f} GB out)    {t_inf:8.2f} s "
          f"{raw/t_inf/1e6:8.1f} MB/s   (inflate alone {t_inf-t_read:.2f} s)")

    t0 = time.perf_counter(); d = read_cell(path)
    t_all = time.perf_counter() - t0
    nrow = len(d["tau"])
    print(f"  3. + parse to {len(d):2d} columns      {t_all:8.2f} s "
          f"{nrow/t_all/1e6:8.2f} Mrow/s  ({nrow/1e6:.2f} Mrow end to end, read thread overlapped)")

    t0 = time.perf_counter(); derive(d, 1e-2)
    print(f"  4. derive()                    {time.perf_counter()-t0:8.2f} s\n")

    t0 = time.perf_counter()
    with _open(path) as f:
        np.loadtxt(f, comments="#")
    t_old = time.perf_counter() - t0
    print(f"  reference: old gzip.open + loadtxt(all cols)  {t_old:7.2f} s  "
          f"-> speedup {t_old/t_all:.1f}x on the read")
    print(f"\n  READ is {100*t_read/t_all:.0f}% of the new per-dump cost.  "
          f"Above ~50% the filesystem is the wall (stripe the dumps wider, or\n"
          f"  stage them to node-local scratch); below it, add cores with -j.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--tag")
    ap.add_argument("-j", "--jobs", type=int, default=0,
                    help="dumps in flight (default: $SLURM_CPUS_PER_TASK, else 4)")
    ap.add_argument("--probe", metavar="DUMP",
                    help="time read / inflate / parse on one dump and exit")
    a = ap.parse_args()

    if a.probe:
        return probe(a.probe)

    DERIVED.mkdir(exist_ok=True)
    man = HERE / "manifest.csv"
    if not man.exists():
        print("no manifest.csv -- run gen.py first", file=sys.stderr)
        return 1
    import config as C
    kvals = [float(x) for x in C.K_OUTPUT_ISO]

    try:
        ncpu = len(os.sched_getaffinity(0))          # respects the cgroup, unlike cpu_count
    except AttributeError:
        ncpu = os.cpu_count() or 1
    # Inside a job take the allocation; on a login node take four, because the
    # default must not be "open sixteen GB-scale streams on the shared node".
    # A whole-round --clean belongs in a batch job with -j $SLURM_CPUS_PER_TASK.
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    jobs = a.jobs if a.jobs > 0 else (int(slurm) if slurm else min(ncpu, 4))

    tasks, n_skip, n_missing = [], 0, 0
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
            m = dict(meta, k_index=i)
            tasks.append((str(src), str(dst), kv, m, False))

    # Biggest dumps first: at Gamma = 1e12 one cell is 100x another, so a
    # size-ordered queue is the difference between all cores busy to the end and
    # fifteen of them idling behind the last 6 GB file.
    sizes = {t[0]: os.path.getsize(t[0]) for t in tasks}
    tasks.sort(key=lambda t: -sizes[t[0]])
    total = max(sum(sizes.values()), 1)
    jobs = max(1, min(jobs, len(tasks)))
    if jobs == 1:                       # nothing else is writing, so show row progress
        tasks = [t[:4] + (True,) for t in tasks]

    # A worker holds one dump's 14 columns, and doubles that while concatenating
    # the blocks.  At Gamma = 1e12 that is ~1.4 GB, so say so before the OOM does.
    biggest = max(sizes.values()) if tasks else 0
    ratio = 4.0 if any(t[0].endswith(".gz") for t in tasks) else 1.0
    peak = 2 * (biggest * ratio / (FIELD_W * 38 + 2)) * 14 * 8 / 1e9
    print(f"reduce_iso: {len(tasks)} dumps to reduce, {total/1e9:.1f} GB on disk, "
          f"{n_skip} already current, {n_missing} absent")
    if tasks:
        print(f"            {jobs} in parallel on {ncpu} visible cores"
              f"  (largest dump {biggest/1e9:.1f} GB -> ~{peak:.1f} GB peak per worker,"
              f" ~{jobs*peak:.0f} GB total)\n", flush=True)

    t_start = time.perf_counter()
    done_b = n_new = n_err = 0
    results = []

    def report(dst, rec, err, dt):
        nonlocal done_b, n_new, n_err
        done_b += sizes[str(src_of[dst])]
        el = time.perf_counter() - t_start
        eta = el * (total - done_b) / done_b if done_b else 0.0
        name = Path(dst).stem
        if err:
            n_err += 1
            head = f"  ! {name}: {err}"
        else:
            n_new += 1
            head = (f"  + {name}  ({len(rec['rows'])} bins, support to "
                    f"gamma={rec['support']['gamma_max']:.1f}, {dt:.1f} s)")
        print(f"{head:<78s} [{done_b/total*100:5.1f}%  {_fmt(el)} elapsed, "
              f"{_fmt(eta)} left]", flush=True)

    src_of = {t[1]: t[0] for t in tasks}

    if jobs == 1 or len(tasks) <= 1:
        for t in tasks:
            report(*one(t))
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
        with ctx.Pool(jobs) as pool:
            for res in pool.imap_unordered(one, tasks):
                report(*res)

    allc = {}
    for f in sorted(DERIVED.glob("*__k*.json")):
        r = json.loads(f.read_text())
        allc[f"{r['tag']}|{r['k_index']}"] = r
    (DERIVED / "sweep.json").write_text(json.dumps(allc))
    size = sum(f.stat().st_size for f in DERIVED.glob("*.json")) / 1024
    el = time.perf_counter() - t_start
    print(f"\nderived: {n_new} rebuilt, {n_skip} up to date, {n_missing} dumps absent"
          + (f", {n_err} FAILED" if n_err else ""))
    print(f"         {total/1e9:.1f} GB in {_fmt(el)}"
          + (f"  ({total/el/1e6:.0f} MB/s of input off disk)" if el > 0 else ""))
    print(f"derived/: {len(allc)} (cell, k) records, {size:.0f} kB total")
    print("sync_from_hpc.sh brings derived/ home; the dumps stay here.")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
