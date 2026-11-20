#!/usr/bin/env python3
"""Read the campaign back: completion, cost, convergence, verdicts.

    ./collect.py                 --status (the default)
    ./collect.py --status        what finished, what it cost, what to paste into
                                 config.py's cost tables
    ./collect.py --bg            background pre-flight: Ecomov, Neff, a_ext, and
                                 the dr_q_max adequacy check
    ./collect.py --cl            C_l / P(k) convergence down each refine pair
    ./collect.py --gal           Galerkin closure error: reduced against exact
    ./collect.py --iso           isotropisation rate + its convergence gate

--status needs only the logs and works without numpy, so it runs anywhere.  The
others read data tables and need numpy.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
LOGS = HERE / "logs"

# One massless neutrino species in units of the photon energy density today.
NU1 = 7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0)


def manifest():
    p = HERE / "manifest.csv"
    if not p.exists():
        sys.exit("no manifest.csv -- run gen.py first")
    with p.open() as fh:
        return list(csv.DictReader(fh))


def _open(p):
    return gzip.open(p, "rt") if str(p).endswith(".gz") else open(p)


def find(tag, *sufs):
    for s in sufs:
        for c in (OUT / f"{tag}_{s}", OUT / f"{tag}_{s}.gz"):
            if c.exists() and c.stat().st_size:
                return c
    return None


def cols(path):
    """Column name -> 0-based POSITION.

    ⚠ POSITION, never the printed number.  CLASS writes cl.dat as
    `# 1:l  1:TT  2:EE ...`, so `l` and `TT` both claim column 1 and a
    name->printed-index map silently returns the wrong column."""
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


# ── status ───────────────────────────────────────────────────────────────────
def parse_logs():
    """tag -> what its log says.  Later logs win: a requeued cell reruns."""
    out = {}
    if not LOGS.exists():
        return out
    for f in sorted(LOGS.glob("*.out"), key=lambda p: p.stat().st_mtime):
        rec, ini = {}, None
        for line in f.read_text(errors="replace").splitlines():
            if line.startswith("ini  "):
                ini = line.split()[1]
            elif line.startswith("ELAPSED_S "):
                rec["elapsed"] = float(line.split()[1])
            elif line.startswith("CPUS "):
                rec["cpus"] = int(line.split()[1])
            elif line.startswith("N_K "):
                rec["n_k"] = line.split()[1]
            elif line.startswith("class exit "):
                rec["rc"] = int(line.split()[2])
            elif line.startswith("QMAX_WARNING "):
                rec["qmax_warn"] = line.split()[1]
            elif line.startswith("PARTITION "):
                rec["partition"] = line.split()[1]
            elif "input line not recognized" in line:
                rec["unread"] = rec.get("unread", 0) + 1
        if ini:
            rec["log"] = f.name
            out[Path(ini).stem] = rec
    return out


# ── live Slurm state ─────────────────────────────────────────────────────────
def _hms(t):
    """Slurm's %M: 'MM:SS', 'HH:MM:SS' or 'D-HH:MM:SS' -> hours.  nan for
    'INVALID', 'UNLIMITED' and anything else squeue decides to print."""
    t = (t or "").strip()
    days = 0
    if "-" in t:
        d, _, t = t.partition("-")
        try:
            days = int(d)
        except ValueError:
            return float("nan")
    try:
        v = [float(x) for x in t.split(":")]
    except ValueError:
        return float("nan")
    if len(v) == 2:
        v = [0.0] + v
    if len(v) != 3:
        return float("nan")
    return days * 24 + v[0] + v[1] / 60 + v[2] / 3600


def _expand_array(spec):
    """'6039504_[24-57%8]' -> 24..57;  '6039504_[3,7,9-12]' -> 3,7,9..12;
    '6039504_24' -> [24].  The %N throttle is not a task id."""
    if "_" not in spec:
        return []
    tail = spec.partition("_")[2].strip()
    if not tail.startswith("["):
        return [int(tail)] if tail.isdigit() else []
    out = []
    for piece in tail.strip("[]").split("%")[0].split(","):
        if "-" in piece:
            a, _, b = piece.partition("-")
            if a.isdigit() and b.isdigit():
                out.extend(range(int(a), int(b) + 1))
        elif piece.isdigit():
            out.append(int(piece))
    return out


def _tag_for(fam, task):
    """(family, array task) -> tag.

    THE LOG IS AUTHORITATIVE for a task that has started, because gen.py
    rewrites runfile_<family> and a running array still refers to the lines it
    was submitted against.  A PENDING task has no log, so the runfile is all
    there is -- and if gen.py has been re-run since submission that mapping is a
    guess, which is why it is only the fallback."""
    # NEWEST first: an array killed at the wall and requeued has several logs for
    # the same task, and the live job is the one that wrote last.  (parse_logs
    # resolves the same ambiguity the same way, by letting later mtimes win.)
    for f in sorted(LOGS.glob(f"{fam}_*_{task}.out"),
                    key=lambda q: q.stat().st_mtime, reverse=True):
        for line in f.read_text(errors="replace").splitlines():
            if line.startswith("ini  "):
                return Path(line.split()[1]).stem
    rf = HERE / f"runfile_{fam}"
    if rf.exists():
        lines = rf.read_text().splitlines()
        if 1 <= task <= len(lines):
            return Path(lines[task - 1].split()[0]).stem
    return None


def squeue_state():
    """tag -> {state, elapsed_h, jobid, partition, reason}, or (None, why)."""
    import shutil
    import subprocess
    if not shutil.which("squeue"):
        return None, ("no `squeue` on PATH -- run this on the cluster to see "
                      "running and queued state")
    # Filter by user AND by job name.  The name filter alone is not enough on a
    # shared cluster -- these submit scripts are not the only ones that could
    # ever be named dncdm_* -- and -u alone would sweep in unrelated jobs.
    cmd = ["squeue", "-h", "-o", "%i|%j|%T|%M|%P|%R"]
    import getpass
    try:
        user = os.environ.get("USER") or getpass.getuser()
    except Exception:                                            # noqa: BLE001
        user = ""
    if user:
        cmd += ["-u", user]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                             check=False)
    except Exception as e:                                       # noqa: BLE001
        return None, f"squeue failed: {e}"
    if out.returncode != 0:
        return None, f"squeue exit {out.returncode}: {out.stderr.strip()[:120]}"
    by_tag = {}
    for line in out.stdout.splitlines():
        f = [x.strip() for x in line.split("|")]
        if len(f) < 6 or not f[1].startswith("dncdm_"):
            continue
        jid, fam, state, used, part, reason = (f[0], f[1][6:], f[2], f[3],
                                               f[4], f[5])
        for task in _expand_array(jid):
            tag = _tag_for(fam, task)
            if tag is not None:
                by_tag[tag] = dict(state=state, elapsed_h=_hms(used),
                                   jobid=f"{jid.split('_')[0]}_{task}",
                                   partition=part, reason=reason)
    return by_tag, None


def _wall_limit_h(fam):
    import config as C
    return _hms(C.FAMILIES.get(fam, {}).get("time", ""))


def live_board(rows, logs):
    """done / running / queued / failed / not-submitted, and how long the running
    ones have been going.  Printed first because it is the only part of --status
    that can still be acted on."""
    live, why = squeue_state()
    print("── live")
    if live is None:
        print(f"   {why}.")
        print("   Completion below comes from the logs, which is what syncs "
              "home, and is\n   still authoritative for what FINISHED.\n")
        return
    if not live:
        print("   nothing from this round is in the queue right now.")
    running = []
    header = (f"   {'family':<8} {'done':>5} {'running':>8} {'queued':>7} "
              f"{'failed':>7} {'not-sub':>8} {'of':>5}")
    body = []
    for fam in sorted({r["family"] for r in rows}):
        sub = [r for r in rows if r["family"] == fam and r["queued"] == "1"]
        n = dict(done=0, run=0, pend=0, fail=0, nosub=0)
        for r in sub:
            L, S = logs.get(r["tag"]), live.get(r["tag"])
            if S and S["state"] == "RUNNING":
                n["run"] += 1
                running.append((fam, r, S))
            elif S:
                n["pend"] += 1
            elif L and L.get("rc") == 0:
                n["done"] += 1
            elif L and L.get("rc") is not None:
                n["fail"] += 1
            else:
                n["nosub"] += 1
        body.append(f"   {fam:<8} {n['done']:>5} {n['run']:>8} {n['pend']:>7} "
                    f"{n['fail']:>7} {n['nosub']:>8} {len(sub):>5}")
    if body:
        print(header)
        print("\n".join(body))
    if running:
        print(f"\n   {'RUNNING':<28} {'elapsed':>9} {'est':>8} {'of est':>7} "
              f"{'of wall':>8} {'part':>6}")
        alerts = []
        for fam, r, S in sorted(running,
                                key=lambda x: -(x[2]["elapsed_h"] or 0.0)):
            est, wall, el = float(r["est_wall_h"]), _wall_limit_h(fam), S["elapsed_h"]
            fe = f"{100 * el / est:.0f}%" if est > 0 else "?"
            fw = f"{100 * el / wall:.0f}%" if wall == wall and wall > 0 else "?"
            print(f"   {r['tag']:<28} {el:8.1f}h {est:7.1f}h {fe:>7} {fw:>8} "
                  f"{S['partition']:>6}")
            # The one genuinely actionable case: a cell killed at the wall has
            # produced NOTHING, and its charged core-hours are gone with it.
            if wall == wall and wall > 0 and el > 0.85 * wall:
                alerts.append((r["tag"], el, wall, est))
        for tag, el, wall, est in alerts:
            print(f"   ⚠ {tag}: {el:.0f} h into a {wall:.0f} h wall. It will be "
                  f"KILLED with no output at all.")
            print(f"     The estimate was {est:.0f} h. Decide now: let the rung "
                  f"go, or scancel and\n     requeue it coarser -- waiting costs "
                  f"the full allocation either way.")
    pend = {t: S for t, S in live.items() if S["state"] != "RUNNING"}
    if pend:
        shown = ", ".join(sorted(pend)[:4])
        more = f" (+{len(pend) - 4} more)" if len(pend) > 4 else ""
        reasons = sorted({S["reason"] for S in pend.values()})
        print(f"\n   QUEUED ({len(pend)}): {shown}{more}")
        print(f"   waiting on: {', '.join(reasons[:3])}")
    print()


def status():
    rows, logs = manifest(), parse_logs()
    live_board(rows, logs)
    print("Completion and cost.  `est` is gen.py's ordering heuristic, not a "
          "prediction;\n`meas` is what the cell actually took.\n")
    fam_tot = {}
    bad = []
    for fam in sorted({r["family"] for r in rows}):
        sub = [r for r in rows if r["family"] == fam and r["queued"] == "1"]
        done = [r for r in sub if logs.get(r["tag"], {}).get("rc") == 0]
        print(f"── family {fam}: {len(done)}/{len(sub)} complete")
        if done:
            print(f"   {'tag':<28} {'n_k':>5} {'est_h':>7} {'meas_h':>7} "
                  f"{'ratio':>6} {'core-h':>9} {'part':>6}")
        for r in sorted(done, key=lambda r: (r["mass"], r["log10Gamma"], r["tag"])):
            L = logs[r["tag"]]
            meas_h = L["elapsed"] / 3600.0
            ch = meas_h * L.get("cpus", 1)
            est = float(r["est_wall_h"])
            fam_tot[fam] = fam_tot.get(fam, 0.0) + ch
            print(f"   {r['tag']:<28} {L.get('n_k','?'):>5} {est:7.1f} {meas_h:7.1f} "
                  f"{(meas_h / est if est > 0 else float('nan')):6.2f} {ch:9.1f} "
                  f"{L.get('partition','?'):>6}")
        for r in sub:
            L = logs.get(r["tag"])
            if L is None:
                continue
            if L.get("unread"):
                bad.append((r["tag"], f"{L['unread']} UNREAD KEY(S) -- guard 3 "
                                      f"rejected it; fix config.py"))
            elif L.get("rc") not in (0, None):
                bad.append((r["tag"], f"class exit {L['rc']}"))
            if L.get("qmax_warn") == "yes":
                bad.append((r["tag"], "dr_q_max below the emission band (see --bg)"))
            nk = L.get("n_k")
            if nk and nk.isdigit():
                import config as C
                want = C.N_K_ISO if r["family"] == "iso" else C.N_K_CMB
                if abs(int(nk) - want) > 0.15 * want:
                    bad.append((r["tag"],
                                f"solved {nk} k modes, config says ~{want}"))
        print()
    for fam, ch in sorted(fam_tot.items()):
        print(f"  family {fam}: {ch:.0f} charged core-hours so far")
    if bad:
        print("\n⚠ NEEDS ATTENTION")
        for t, why in bad:
            print(f"   {t:<30} {why}")
    else:
        print("\nno rejected cells, no q_max warnings, no mode-count surprises.")

    # The cost tables get better every time a cell finishes rather than needing a
    # refit.  Print the rows to paste, per scheme and mass, normalised back to
    # N_K_REF modes at 32 cores so they are comparable with what is already there.
    print("\n── measured cost, normalised to config.N_K_REF modes (paste into config.py)")
    import config as C
    acc = {}
    for r in rows:
        L = logs.get(r["tag"])
        if not L or L.get("rc") != 0 or r["refine"] != "1":
            continue
        nk = int(L["n_k"]) if str(L.get("n_k", "")).isdigit() else int(r["n_k"])
        ch = L["elapsed"] / 3600.0 * L.get("cpus", 1)
        # Undo the mode count and the momenta_bins factor; leave the wave padding
        # in, because that is part of what a cell really costs.
        norm = (ch * (C.N_K_REF / nk)
                / C.MOMENTA_COST_FACTOR.get(int(r["momenta_bins"]), 1.0))
        acc.setdefault((r["scheme"], r["mass"]), []).append(
            (float(r["gamma"]), norm))
    for (scheme, mass), pts in sorted(acc.items()):
        pts.sort()
        body = ", ".join(f"{g:.0e}: {c:.2f}" for g, c in pts)
        print(f'   {scheme:>3} "{mass}": {{{body}}}')
    if not acc:
        print("   (nothing finished yet)")


# ── background pre-flight ────────────────────────────────────────────────────
def bg_table():
    import numpy as np
    import config as C
    print("Background pre-flight.\n")
    print("Ecomov = a^4 rho_sector(today) / a^4 rho_sector(initial): the parent's")
    print("rest mass converted to radiation, i.e. what the sector adds to N_eff.")
    print("An under-resolved daughter grid shows up HERE and nowhere else -- P(k)")
    print("and C_l do not see it.\n")
    print("⚠ The Gamma trend is PHYSICS, not grid error.  Ecomov saturates only")
    print("  above Gamma ~ H(a_nr); below that the parent survives longer, is more")
    print("  non-relativistic when it decays, and converts more rest mass, so low")
    print("  Gamma SHOULD read high.  a_ext saturates where Ecomov does.\n")
    print(f"{'m':>6} {'log10G':>7} {'dr_N_q':>7} {'b/dec':>7} {'Ecomov':>9} "
          f"{'Neff_sec':>9} {'a_ext':>10} {'q*(a_ext)':>10} {'q_max ok':>9}")
    seen, last = set(), None
    for r in sorted(manifest(), key=lambda r: (float(r["mass"]),
                                               float(r["log10Gamma"]))):
        key = (r["mass"], r["log10Gamma"])
        if key in seen:
            continue
        f = find(f"bg_G{r['log10Gamma']}_m{r['mass']}", "bgslim.dat", "background.dat")
        if f is None:
            continue
        seen.add(key)
        if last is not None and r["mass"] != last:
            print()
        last = r["mass"]
        idx = cols(f)
        with _open(f) as fh:
            d = np.loadtxt(fh)
        a = 1.0 / (1.0 + d[:, idx["z"]])
        o = np.argsort(a)
        a = a[o]
        g = lambda c: d[o, idx[c]]                                   # noqa: E731
        h = lambda c: a ** 4 * g(c)                                  # noqa: E731
        sec = (h("(.)rho_dncdm1") + h("(.)rho_dr_dncdm1_l")
               + h("(.)rho_dr_dncdm1_phi"))
        frac = g("(.)rho_dncdm1") / g("(.)rho_tot")
        ext = np.where(frac >= 1e-8)[0]
        a_ext = a[ext[-1]] if len(ext) and ext[-1] + 1 < len(a) else float("nan")
        neff = sec[-1] / (h("(.)rho_g")[-1] * NU1)

        # THE dr_q_max ADEQUACY CHECK, done properly.  CLASS's own warning
        # evaluates q* at a = 1 unconditionally, which is the wrong epoch: the
        # energy leak needs the parent to still be DECAYING when a*M/2 outgrows
        # the grid.  The right test uses a_ext, where the parent is extinct.
        #   q* = 0.5 * (sqrt(q1max^2 + (a M)^2) + q1max),  M = m / (T_ncdm * T_cmb)
        # in the kernel's dimensionless units.  q1max is the parent's top
        # trapezoid node.
        # k_B T_cmb in eV, then the parent's own temperature, then its mass in
        # the kernel's dimensionless q = p/T units.  Checked against the number
        # CLASS itself prints: m = 0.3 eV at T_ncdm = 0.71611 gives M = 1783.7.
        kT_cmb = 8.617333262e-5 * float(C.COSMOLOGY["T_cmb"])
        M = float(r["mass"]) / (float(C.PARENT_T) * kT_cmb)
        # The parent's top trapezoid node.  CLASS reports q_max = 15 for
        # momenta_bins = 16, so it is one less than the bin count.
        q1max = float(int(r["momenta_bins"]) - 1)
        am = a_ext * M if a_ext == a_ext else M
        qstar = 0.5 * (math.sqrt(q1max ** 2 + am ** 2) + q1max)
        ok = "yes" if qstar <= float(C.Q_MAX) else "** NO **"
        print(f"{r['mass']:>6} {r['log10Gamma']:>7} {r['dr_N_q']:>7} "
              f"{float(r['dr_N_q']) / C.DECADES:7.1f} {sec[-1] / sec[0]:9.4f} "
              f"{neff:9.4f} {a_ext:10.3e} {qstar:10.1f} {ok:>9}")
    print(f"\nq*(a_ext) is the highest daughter momentum the decay can actually emit")
    print(f"while the parent still exists.  dr_q_max = {C.Q_MAX}.  A '** NO **' means")
    print("the top-bin clamp is losing energy in proportion to q_max/q* -- silently,")
    print("because the identities the kernel asserts are NUMBER identities.")


# ── C_l / P(k) convergence down the refine pairs ─────────────────────────────
def scale_rel(ref, x, w=25):
    """|dx| against the LOCAL rms of the reference, not a pointwise ratio.

    C_l^TE crosses zero, so a ratio divides by a number that can sit 100x below
    the signal's own envelope.  A rolling local rms, floored, is the metric the
    earlier campaigns settled on for exactly that reason."""
    import numpy as np
    p = np.pad(ref ** 2, w, mode="edge")
    loc = np.sqrt(np.convolve(p, np.ones(2 * w + 1) / (2 * w + 1), "valid"))
    loc = np.maximum(loc, 1e-3 * np.sqrt(np.mean(ref ** 2)))
    return np.abs(x - ref) / loc


def cl_table():
    import numpy as np
    import config as C
    print("Convergence down each refine pair: refine = 2 is the HALVED")
    print("perturbation grid at the SAME background, so the pair brackets the")
    print("answer.  Small numbers mean the production cell (refine = 1) is")
    print("converged; large ones mean raise BPD_PT_AT_1E7, not add rungs.\n")
    print(f"{'tag':<26} {'TT max':>9} {'TT med':>9} {'EE max':>9} "
          f"{'TE max':>9} {'P(k) max':>9}")
    rows = manifest()
    by = {r["tag"]: r for r in rows}
    n = 0
    for r in sorted(rows, key=lambda r: r["tag"]):
        if r["refine"] != "1":
            continue
        twin = by.get(r["tag"] + "c")
        if twin is None:
            continue
        a = find(r["tag"], "cl_lensed.dat", "cl.dat")
        b = find(twin["tag"], "cl_lensed.dat", "cl.dat")
        if a is None or b is None:
            continue
        ca, cb = cols(a), cols(b)
        with _open(a) as fh:
            da = np.loadtxt(fh)
        with _open(b) as fh:
            db = np.loadtxt(fh)
        m = min(len(da), len(db))
        lmax = min(C.L_PUBLISH, int(da[m - 1, 0]))
        sel = da[:m, 0] <= lmax
        out = []
        for nm in ("TT", "EE", "TE"):
            if nm not in ca or nm not in cb:
                out.append(float("nan"))
                continue
            x, y = da[:m, ca[nm]][sel], db[:m, cb[nm]][sel]
            e = scale_rel(x, y) if nm == "TE" else np.abs(y / x - 1.0)
            out.append(e)
        pa, pb = find(r["tag"], "z1_pk.dat", "pk.dat"), find(twin["tag"], "z1_pk.dat", "pk.dat")
        pk = float("nan")
        if pa and pb:
            with _open(pa) as fh:
                A = np.loadtxt(fh)
            with _open(pb) as fh:
                B = np.loadtxt(fh)
            k = min(len(A), len(B))
            pk = np.max(np.abs(B[:k, 1] / A[:k, 1] - 1.0))
        tt, ee, te = out
        print(f"{r['tag']:<26} {np.max(tt):9.2e} {np.median(tt):9.2e} "
              f"{np.max(ee):9.2e} {np.max(te):9.2e} {pk:9.2e}")
        n += 1
    if not n:
        print("(no completed refine pairs yet)")


# ── the doubled daughter grid ────────────────────────────────────────────────
C2_TARGET = 1.481e-10   # C_2^TT is Gamma-independent in this model


def grid_table():
    """The GRID_RUNG pairs: `_q2x` against the ladder's own `x` cell.

    The whole point is low ell, so this reports C_2 and C_10 explicitly rather
    than a max over the published range -- above ell ~ 30 the pair agrees to
    0.16% and a max would drown the signal in the noise floor."""
    import numpy as np
    print("The doubled daughter grid against the production one, SAME (Gamma,")
    print("mass, scheme, refine) and only dr_N_q changed.  The rule is a power")
    print("law in Gamma; the measured requirement is not -- it PEAKS mid-ladder.")
    print("See GRID_RUNG in config.py.\n")
    print("C_2^TT is Gamma-INDEPENDENT in this model, so `C_2 prod` is readable")
    print("on its own: departures from ~1.481e-10 ARE the grid error.  `d C_2`")
    print("is the correction the doubled cell applies.\n")
    print(f"{'tag':<24} {'bpd':>6} {'->':>3} {'bpd2':>6} {'C_2 prod':>11} "
          f"{'C_2 x2':>11} {'d C_2':>8} {'d C_10':>8} {'d C_l>30':>9}")
    rows = manifest()
    by = {r["tag"]: r for r in rows}
    n = 0
    for r in sorted(rows, key=lambda r: r["tag"]):
        if not r["tag"].endswith("_q2x"):
            continue
        base = by.get(r["tag"][:-4])
        if base is None:
            continue
        a = find(base["tag"], "cl_lensed.dat", "cl.dat")
        b = find(r["tag"], "cl_lensed.dat", "cl.dat")
        if a is None or b is None:
            continue
        ca, cb = cols(a), cols(b)
        with _open(a) as fh:
            da = np.loadtxt(fh)
        with _open(b) as fh:
            db = np.loadtxt(fh)
        m = min(len(da), len(db))
        x, y = da[:m, ca["TT"]], db[:m, cb["TT"]]
        ell = da[:m, 0]

        def at(l):
            i = np.searchsorted(ell, l)
            return (x[i], y[i]) if i < len(ell) else (np.nan, np.nan)

        c2a, c2b = at(2)
        c10a, c10b = at(10)
        hi = ell > 30
        dhi = np.max(np.abs(y[hi] / x[hi] - 1.0)) if hi.any() else np.nan
        print(f"{base['tag']:<24} {base['bins_per_decade']:>6} {'->':>3} "
              f"{r['bins_per_decade']:>6} {c2a:11.4e} {c2b:11.4e} "
              f"{100 * (c2b / c2a - 1):+7.1f}% {100 * (c10b / c10a - 1):+7.1f}% "
              f"{100 * dhi:8.2f}%")
        n += 1
    if not n:
        print("(no completed GRID_RUNG pairs yet)")
    else:
        print("\nRead it as a CURVE, not six patches.  The two shoulders "
              "(Gamma = 1e5 and\n1e8 at m = 0.3) are already clean at production "
              "resolution, so `d C_2`\nthere MUST be ~0.  If it is not, the "
              "production rule has no margin\nanywhere and the whole ladder is "
              "in question, not four rungs of it.")


# ── the reduced table's ln-a spacing ─────────────────────────────────────────
def table_table():
    """The TABLE_RUNG pairs: `_t2x` at half the spacing against the rule's own cell.

    This round's premise is that the table spacing was wrong at every Gamma and
    nothing noticed, so this is the gate on everything else here: read it FIRST.

    C_2^TT is Gamma-INDEPENDENT in this model, so the target is absolute --
    1.481e-10 -- and BOTH members of a pair have to reach it. A pair that agrees
    somewhere else has converged to the wrong place, and a pair that only lands on
    target at 2x is the rule being wrong with the cost paid anyway."""
    import numpy as np

    import config as C
    eps = C.KTABLE_EPS
    print("The reduced table at HALF the spacing the rule asks for, against the")
    print("rule's own cell.  Same (Gamma, mass, scheme, dr_N_q); only")
    print("dr_table_max_dlna changed.  See TABLE_RUNG and the KTABLE block in")
    print("config.py.\n")
    print(f"PASS needs BOTH: |d C_2| <= {100 * eps:.1f}% between the pair, AND both")
    print(f"within {100 * eps:.1f}% of {C2_TARGET:.4g}.\n")
    print(f"{'tag':<26} {'n_sub':>6} {'->':>3} {'n_sub2':>7} {'C_2 rule':>11} "
          f"{'C_2 2x':>11} {'d C_2':>8} {'vs target':>10} {'d C_l>30':>9}  verdict")
    rows = manifest()
    by = {r["tag"]: r for r in rows}
    n = 0
    for r in sorted(rows, key=lambda r: (float(r["log10Gamma"]), r["tag"])):
        if not r["tag"].endswith("_t2x"):
            continue
        base = by.get(r["tag"][:-4])
        if base is None:
            continue
        a = find(base["tag"], "cl_lensed.dat", "cl.dat")
        b = find(r["tag"], "cl_lensed.dat", "cl.dat")
        if a is None or b is None:
            continue
        ca, cb = cols(a), cols(b)
        with _open(a) as fh:
            da = np.loadtxt(fh)
        with _open(b) as fh:
            db = np.loadtxt(fh)
        m = min(len(da), len(db))
        x, y, ell = da[:m, ca["TT"]], db[:m, cb["TT"]], da[:m, 0]
        i = int(np.searchsorted(ell, 2))
        c2a, c2b = x[i], y[i]
        hi = ell > 30
        dhi = np.max(np.abs(y[hi] / x[hi] - 1.0)) if hi.any() else np.nan
        d_pair = c2b / c2a - 1.0
        d_tgt = max(abs(c2a / C2_TARGET - 1.0), abs(c2b / C2_TARGET - 1.0))
        ns = C.n_sub(float(base["gamma"]), base["mass"])
        ok = abs(d_pair) <= eps and d_tgt <= eps
        # A shoulder is a cell the rule asks NO sub-division for; doubling it must
        # change nothing, and that is a different statement from "converged".
        note = "PASS" if ok else ("FAIL pair" if abs(d_pair) > eps else "FAIL target")
        if ns == 1:
            note += "  [SHOULDER: any move here voids the whole rule]"
        print(f"{base['tag']:<26} {ns:>6} {'->':>3} {ns * 2:>7} {c2a:11.4e} "
              f"{c2b:11.4e} {100 * d_pair:+7.2f}% {100 * d_tgt:9.2f}% "
              f"{100 * dhi:8.2f}%  {note}")
        n += 1
    if not n:
        print("(no completed TABLE_RUNG pairs yet)")
    else:
        print("\nA FAIL here means the KTABLE fit in config.py is wrong, and every")
        print("other number this round produces is resting on it.  Do not read --cl,")
        print("--gal or --iso until this passes.")


# ── the Galerkin closure error ───────────────────────────────────────────────
def gal_table():
    import numpy as np
    import config as C
    print("Galerkin closure error: the reduced scheme against the EXACT one at")
    print("matched (Gamma, mass, grid).  This is the number the paper has to")
    print("quote before the reduction is allowed to carry the ladder past where")
    print("the exact scheme stops.\n")
    print(f"{'m':>6} {'log10G':>7} {'scheme':>7} {'TT max':>9} {'TT med':>9} "
          f"{'EE max':>9} {'P(k) max':>9}")
    rows = manifest()
    idx = {(r["family"], r["mass"], r["log10Gamma"], r["scheme"]): r
           for r in rows if r["refine"] == "1"}
    last = None
    for (fam, mass, gl, scheme), r in sorted(idx.items()):
        if scheme == "x":
            continue
        ex = idx.get(("cmb", mass, gl, "x")) or idx.get((fam, mass, gl, "x"))
        if ex is None:
            continue
        a = find(ex["tag"], "cl_lensed.dat", "cl.dat")
        b = find(r["tag"], "cl_lensed.dat", "cl.dat")
        if a is None or b is None:
            continue
        if last is not None and (mass, gl) != last:
            print()
        last = (mass, gl)
        ca, cb = cols(a), cols(b)
        with _open(a) as fh:
            da = np.loadtxt(fh)
        with _open(b) as fh:
            db = np.loadtxt(fh)
        m = min(len(da), len(db))
        sel = da[:m, 0] <= C.L_PUBLISH
        tt = np.abs(db[:m, cb["TT"]][sel] / da[:m, ca["TT"]][sel] - 1.0)
        ee = np.abs(db[:m, cb["EE"]][sel] / da[:m, ca["EE"]][sel] - 1.0)
        pa, pb = find(ex["tag"], "z1_pk.dat", "pk.dat"), find(r["tag"], "z1_pk.dat", "pk.dat")
        pk = float("nan")
        if pa and pb:
            with _open(pa) as fh:
                A = np.loadtxt(fh)
            with _open(pb) as fh:
                B = np.loadtxt(fh)
            k = min(len(A), len(B))
            pk = np.max(np.abs(B[:k, 1] / A[:k, 1] - 1.0))
        print(f"{mass:>6} {gl:>7} {scheme:>7} {np.max(tt):9.2e} "
              f"{np.median(tt):9.2e} {np.max(ee):9.2e} {pk:9.2e}")
    print("\nRead the MEDIAN as well as the max: a max driven by a handful of")
    print("multipoles is a different statement from a uniform offset.")


# ── the isotropisation rate ──────────────────────────────────────────────────
def iso_table(gate):
    import json
    import numpy as np
    d = HERE / "derived"
    files = sorted(d.glob("*__k*.json")) if d.exists() else []
    if not files:
        sys.exit("no derived/ -- run reduce_iso.py on the cluster first")
    recs = [json.loads(f.read_text()) for f in files]
    print(f"Isotropisation rate from {len(recs)} (cell, k) records.\n")
    print("p is the exponent in Gamma_T ~ gamma^-p, fitted over the gamma window")
    print(f"below.  ⚠ A COARSE GRID DOES NOT ADD SCATTER TO p -- IT FAKES A")
    print("STEEPER ONE.  `dp(coarse)` is the production cell against its refine=2")
    print("twin: where that is large, p is a discretisation artefact, not a rate.\n")
    print(f"{'m':>6} {'log10G':>7} {'scheme':>7} {'k':>9} {'p':>7} {'+-':>6} "
          f"{'R2':>6} {'gsup':>7} {'dp(coarse)':>11}")

    def fit(rows, lo, hi):
        # ⚠ SIGN.  reduce_iso.py stores GT with the RELAXATION convention, i.e.
        # GT > 0 means isotropisation.  This used to negate it and then keep only
        # y > 0, which discarded every physically sensible record and printed nan
        # for every p in the hpc_prod audit (637 records came back on dropping the
        # minus).  Do not "restore" it: check reduce_iso.py's convention instead.
        x = np.array([r["gamma"] for r in rows])
        y = np.array([r["GT"] for r in rows])
        m = (x >= lo) & (x <= hi) & np.isfinite(y) & (y > 0)
        if m.sum() < 3:
            return float("nan"), float("nan")
        lx, ly = np.log(x[m]), np.log(y[m])
        A = np.vstack([lx, np.ones_like(lx)]).T
        coef, res, *_ = np.linalg.lstsq(A, ly, rcond=None)
        sd = float("nan")
        if len(res) and m.sum() > 2:
            s2 = res[0] / (m.sum() - 2)
            sd = math.sqrt(s2 / np.sum((lx - lx.mean()) ** 2))
        return -coef[0], sd

    by = {(r["tag"], r["k_index"]): r for r in recs}
    for r in sorted(recs, key=lambda r: (r["mass"], r["log10Gamma"],
                                         r["scheme"], r["k_index"])):
        if r["refine"] != 1:
            continue
        p, sd = fit(r["rows"], 2.0, gate)
        twin = by.get((r["tag"] + "c", r["k_index"]))
        dp = float("nan")
        if twin:
            pc, _ = fit(twin["rows"], 2.0, gate)
            dp = pc - p
        gsup = r["support"]["gamma_max"]
        flag = "" if gsup >= gate else "  <- support ends INSIDE the window"
        print(f"{r['mass']:>6} {r['log10Gamma']:>7} {r['scheme']:>7} "
              f"{r['k']:9.1e} {p:7.2f} {sd:6.2f} "
              f"{np.median([q['R2'] for q in r['rows']]):6.2f} {gsup:7.1f} "
              f"{dp:11.2f}{flag}")
    print(f"\nwindow gamma in [2, {gate}].  gsup is where the collision term's")
    print("support ends: a REDUCED cell applies the operator only inside its table")
    print("window, and a bin of exact zeros fits Gamma_T = 0, which looks like a")
    print("converged measurement.  Any row whose gsup is below the window top is")
    print("not a measurement.")


def main() -> int:
    ap = argparse.ArgumentParser()
    for f in ("status", "bg", "cl", "grid", "table", "gal", "iso"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--gate", type=float, default=15.0,
                    help="top of the gamma window for --iso (default 15)")
    a = ap.parse_args()
    if not any((a.status, a.bg, a.cl, a.grid, a.table, a.gal, a.iso)):
        a.status = True
    if a.status:
        status()
    if a.bg:
        bg_table()
    if a.cl:
        cl_table()
    if a.grid:
        grid_table()
    if a.table:
        table_table()
    if a.gal:
        gal_table()
    if a.iso:
        iso_table(a.gate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
