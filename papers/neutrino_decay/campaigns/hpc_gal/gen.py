#!/usr/bin/env python3
"""Generate the ini files, the Slurm runfiles and manifest.csv from config.py.

    ./gen.py              write everything, print the matrix and the cost report
    ./gen.py --dry-run    print only, write nothing
    ./gen.py --family cmb restrict to one family

Stdlib only: this runs on a login node with no modules loaded.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import config as C

HERE = Path(__file__).resolve().parent
INI = HERE / "ini"

SCHEMES = {"x": 0, "m4": 4, "m6": 6, "m8": 8}   # tag -> dr_reduced_moments


# ── the daughter grid ────────────────────────────────────────────────────────
def round_up_grid(n: int) -> int:
    """Smallest N >= n with (N - 1) % GRID_STEP == 0, so refine = 2 is legal."""
    s = C.GRID_STEP
    return max(1 + s, 1 + s * math.ceil(max(n - 1, 0) / s))


def bpd(gamma: float, mass: str, mult: float = 1.0) -> float:
    """Bins per decade this cell needs: the MAX of the two requirements.

    The limiter cut the background's requirement by ~5x, so it is now usually the
    PERTURBATION one that binds -- which is why dr_bg_refine is a convergence
    rung here rather than a production knob.  Taking the max (and running
    refine = 1 in production) means N_bg == N_pt: one grid, no subsampling, and
    nothing is being refined away from."""
    bg = C.BPD_BG_AT_1E8 * (gamma / 1e8) ** C.BPD_BG_INDEX
    pt = C.BPD_PT_AT_1E7 * (gamma / 1e7) ** C.BPD_PT_INDEX
    # An unlisted mass takes the LARGEST factor: over-resolving costs CPU, while
    # under-resolving produces a healthy-looking run with a several-per-cent error
    # that nothing downstream can see.
    f = C.BPD_MASS_FACTOR.get(str(mass), max(C.BPD_MASS_FACTOR.values()))
    # `mult` is GRID_RUNG's doubled-grid probe.  It multiplies the REQUIREMENT,
    # not dr_N_q, so the ini's own "requirement at this (Gamma, mass)" comment
    # stays honest about what the cell was sized for.
    return max(bg, pt) * f * mult


def n_bg_for(gamma: float, mass: str, mult: float = 1.0) -> tuple[int, bool]:
    want = round_up_grid(int(math.ceil(bpd(gamma, mass, mult) * C.DECADES)))
    n = max(C.N_BG_FLOOR, want)
    capped = n > C.N_BG_CAP
    return (round_up_grid(C.N_BG_CAP) if capped else n), capped


# ── cost ─────────────────────────────────────────────────────────────────────
def _interp(tab: dict, gamma: float) -> float:
    """Log-log interpolation of a {Gamma: core-hours} table, clamped below the
    lowest measured point and extrapolated above the highest at the LOCAL slope
    of the top two.  Anything above the measured range is an ordering hint."""
    pts = sorted(tab.items())
    g = max(gamma, pts[0][0])
    if g <= pts[0][0]:
        return pts[0][1]
    if g >= pts[-1][0]:
        (g0, c0), (g1, c1) = pts[-2], pts[-1]
        idx = math.log(c1 / c0) / math.log(g1 / g0)
        return pts[-1][1] * (g / pts[-1][0]) ** idx
    for (g0, c0), (g1, c1) in zip(pts, pts[1:]):
        if g0 <= g <= g1:
            f = math.log(g / g0) / math.log(g1 / g0)
            return math.exp(math.log(c0) + f * (math.log(c1) - math.log(c0)))
    return pts[-1][1]


def _table(store: dict, mass: str) -> dict:
    """Per-mass table, falling back to the most expensive one for an unknown
    mass -- under-estimating is what queues a job that cannot finish."""
    t = store.get(str(mass))
    if t is None:
        t = max(store.values(), key=lambda d: max(d.values()))
    return t


def cost(gamma: float, mass: str, scheme: str, refine: int,
         n_k: int, cpus: int, momenta: int) -> tuple[float, float, int]:
    """(work core-hours, estimated wall hours, waves).

    TWO TERMS, and hpc_prod only had the first.

    WAVES ARE THE POINT for the k loop.  perturbations_module.cpp fans it out
    over a task pool, so a cell costs ceil(n_k / cpus) waves, and Slurm charges
    for the whole allocation whether or not the last wave fills it.  126 modes on
    64 cores is two waves at 98% occupancy; 129 modes would be three at 67%.

    THE BACKGROUND IS SERIAL and every scheme pays it in full.  It does not
    divide by cpus, does not scale with n_k, and does not care about refine -- so
    it is added to the WALL directly, and multiplied by cpus to charge for the
    cores it leaves idle.  Leaving it out is what made hpc_prod under-estimate
    every Galerkin cell by 5-15x: at high Gamma those cells ARE the background.
    It is also the ceiling: see GAMMA_CEILING in config.py."""
    if scheme == "x":
        base = _interp(_table(C.COST_EXACT, mass), gamma)
        # An exact cell carries the full daughter q-grid in its ODE state, so its
        # cost is proportional to n_pt and a halved perturbation grid halves it.
        base /= refine
    else:
        base = _interp(_table(C.COST_M6, mass), gamma) * C.MOMENT_COST_FACTOR[scheme]
        # A TABULATED reduced cell's k-loop does NOT scale with the quadrature
        # grid: the operator table is S x S with S = momenta_bins + 2*n_moments,
        # so refining dr_N_q costs only the background solve and the table
        # assembly.  refine therefore buys almost nothing here.
    mom = C.MOMENTA_COST_FACTOR.get(momenta, max(C.MOMENTA_COST_FACTOR.values()))
    work = base * (n_k / C.N_K_REF) * mom
    w = max(1, math.ceil(n_k / cpus))
    skew = C.SKEW_ONE_WAVE if w == 1 else C.SKEW_MULTI_WAVE
    pad = (w * cpus / n_k) * skew
    wall = work / cpus * pad
    # The background: serial, so straight onto the wall, and charged at cpus.
    # momenta_bins is in the background ODE state too, hence the same factor.
    bg_wall = _interp(_table(C.COST_BG, mass), gamma) * mom
    return work + bg_wall * cpus, wall + bg_wall, w


# ── the ini ──────────────────────────────────────────────────────────────────
def ini_text(fam: str, gs: str, mass: str, scheme: str, refine: int,
             n_bg: int, momenta: int, tag: str) -> str:
    gamma = float(gs)
    n_pt = (n_bg - 1) // refine + 1
    moments = SCHEMES[scheme]
    L = [
        "# Generated by gen.py -- edit config.py, not this file.",
        f"# family {fam}  Gamma = {gamma:.6g}  m = {mass} eV  scheme = {scheme}",
        f"# dr_N_q = {n_bg}, dr_bg_refine = {refine} -> perturbation grid {n_pt} "
        f"points over {C.DECADES:.3f} decades = {n_pt / C.DECADES:.1f} bins/decade",
        f"# requirement at this (Gamma, mass): {bpd(gamma, mass):.1f} bins/decade",
    ]
    _mult = (n_pt / C.DECADES) / bpd(gamma, mass)
    if _mult > 1.5:
        L.append(
            f"# GRID RUNG: {_mult:.1f}x the rule's requirement, on purpose. The rule is a"
            f" power law;\n# the measured requirement is not (it PEAKS mid-ladder). See"
            f" GRID_RUNG in config.py.")
    L += [
        (f"# REDUCED: {moments} moments per daughter"
         if moments else "# EXACT: full daughter q-grid in the ODE state"),
    ]
    for k, v in C.COSMOLOGY.items():
        L.append(f"{k} = {v}")
    L += [
        f"N_ur = {C.N_UR}",
        "input_verbose = 1",
        "background_verbose = 1",
        # GUARD 3's source.  Without this CLASS silently ignores unknown keys and
        # runcell.sh has nothing to fail on.  Do not remove.
        "write warnings = yes",
        # Records every parameter actually used, defaults included, next to the
        # output.  Provenance for a production round, and it costs nothing.
        "write parameters = yes",
        "dncdm1.type = ncdm_decay_dr",
        f"dncdm1.m = {mass}",
        f"dncdm1.T = {C.PARENT_T}",
        "dncdm1.deg = 1",
        "dncdm1.inverse_decays = yes",
        "dncdm1.quantum_statistics = yes",
        "dncdm1.dr_f_ini_l = 1",
        "dncdm1.dr_f_ini_phi = 0",
        f"dncdm1.quadrature_strategy = {C.QUAD_STRATEGY}",
        f"dncdm1.momenta_bins = {momenta}",
        f"dncdm1.dr_q_min = {C.Q_MIN}",
        f"dncdm1.dr_q_max = {C.Q_MAX}",
        f"dncdm1.dr_N_q = {n_bg}",
        f"dncdm1.dr_bg_refine = {refine}",
    ]
    if moments:
        L.append(f"dncdm1.dr_reduced_moments = {moments}")
    L += [
        f"dncdm1.Gamma = {gamma:.6g}",
        f"evolver_background = {C.EVOLVER_BACKGROUND}",
        f"evolver_thermodynamics = {C.EVOLVER_THERMO}",
        f"evolver_perturbations = {C.EVOLVER_PERTURBATIONS}",
        f"tol_perturb_integration = {C.TOL_PERTURB}",
        f"l_max_ncdm = {C.L_MAX_NCDM}",
    ]
    if fam == "bg":
        # Background pre-flight: no perturbations at all, and the full PSD table.
        L += ["output = ", "write background = yes"]
    elif fam == "iso":
        # ONLY the k_output modes.  `output = mTk` leaves has_cls false, so
        # k_max_cmb collapses to k_min; P_k_max below k_min leaves k_max there
        # too; the k list is then a single mode into which k_output_values are
        # inserted.  MEASURED: 3 requested values give exactly 4 modes.
        L += [f"{k} = {v}" for k, v in C.PRECISION.items()]
        L += [
            "output = mTk",
            f"P_k_max_1/Mpc = {C.P_K_MAX_ISO}",
            "z_pk = 0",
            f"k_output_values = {','.join(C.K_OUTPUT_ISO)}",
            "write background = no",
        ]
    else:                                    # cmb, gal
        L += [f"{k} = {v}" for k, v in C.PRECISION.items()]
        L += [
            "output = tCl,pCl,lCl,mPk,mTk",
            f"lensing = {C.LENSING}",
            f"l_max_scalars = {C.L_MAX_SCALARS}",
            f"delta_l_max = {C.DELTA_L_MAX}",
            f"z_pk = {C.Z_PK}",
            # The background table carries f_dr_*[j] for EVERY daughter bin, so it
            # is ~2*dr_N_q columns wide and identical across a refine pair by
            # construction.  submit_bg.sh writes it once per (Gamma, mass).
            "write background = no",
        ]
    # Explicit root.  Without it CLASS derives one from the ini PATH and lands
    # everything in output/ini/, which does not exist -- the run then dies after
    # the background with "could not open backfile".
    L.append(f"root = out/{tag}_")
    return "\n".join(L) + "\n"


# ── the matrix ───────────────────────────────────────────────────────────────
def build(families):
    rows, per_family = [], {f: {"queued": [], "skipped": []} for f in families}
    bg_cells = {}
    for fam in families:
        spec = C.FAMILIES[fam]
        cpus = spec["cpus"]
        momenta = C.MOMENTA_BINS_ISO if fam == "iso" else C.MOMENTA_BINS_CMB
        n_k = C.N_K_ISO if fam == "iso" else C.N_K_CMB
        for scheme, by_mass in C.LADDER[fam].items():
            for mass, ladder in by_mass.items():
                for gl in ladder:
                    gamma = 10.0 ** float(gl)
                    n_bg, capped = n_bg_for(gamma, mass)
                    bg_cells[(gl, mass)] = n_bg
                    for refine in C.REFINES[fam]:
                        if refine > 1 and scheme != "x" and fam != "iso":
                            # A refine twin on a tabulated reduced cell measures
                            # almost nothing: its k-loop does not see the grid.
                            continue
                        tag = (f"{fam}_G{gl}_m{mass}_{scheme}"
                               + ("" if refine == 1 else "c"))
                        work, wall, waves = cost(gamma, mass, scheme, refine,
                                                 n_k, cpus, momenta)
                        over = wall > C.TARGET_WALL_H[fam] * C.WALL_MARGIN
                        rows.append(dict(
                            family=fam, tag=tag, log10Gamma=gl, gamma=f"{gamma:.6g}",
                            mass=mass, scheme=scheme, moments=SCHEMES[scheme],
                            refine=refine, dr_N_q=n_bg,
                            n_pt=(n_bg - 1) // refine + 1,
                            bins_per_decade=round(((n_bg - 1) // refine + 1) / C.DECADES, 1),
                            momenta_bins=momenta, n_k=n_k, cpus=cpus, waves=waves,
                            est_work_core_h=round(work, 1),
                            est_wall_h=round(wall, 1),
                            est_charged_core_h=round(wall * cpus, 1),
                            bg_unconverged=int(capped), queued=int(not over),
                            ini=f"ini/{tag}.ini"))
                        key = "skipped" if over else "queued"
                        per_family[fam][key].append((tag, f"ini/{tag}.ini", wall))

    # The momenta_bins rung: the same family-B cell at 16 and 64 parent bins, so
    # that the production choice of 32 is justified against the FITTED EXPONENT
    # rather than against the a4Pi wiggle amplitude it was originally picked on.
    if "iso" in families:
        spec = C.FAMILIES["iso"]
        for gl, mass, bins_list in getattr(C, "MOMENTA_RUNG", []):
            n_bg, capped = n_bg_for(10.0 ** float(gl), mass)
            for mb in bins_list:
                tag = f"iso_G{gl}_m{mass}_x_mb{mb}"
                work, wall, waves = cost(10.0 ** float(gl), mass, "x", 1,
                                         C.N_K_ISO, spec["cpus"], mb)
                over = wall > C.TARGET_WALL_H["iso"] * C.WALL_MARGIN
                rows.append(dict(
                    family="iso", tag=tag, log10Gamma=gl,
                    gamma=f"{10.0 ** float(gl):.6g}", mass=mass, scheme="x",
                    moments=0, refine=1, dr_N_q=n_bg, n_pt=n_bg,
                    bins_per_decade=round(n_bg / C.DECADES, 1),
                    momenta_bins=mb, n_k=C.N_K_ISO, cpus=spec["cpus"],
                    waves=waves, est_work_core_h=round(work, 1),
                    est_wall_h=round(wall, 1),
                    est_charged_core_h=round(wall * spec["cpus"], 1),
                    bg_unconverged=int(capped), queued=int(not over),
                    ini=f"ini/{tag}.ini"))
                key = "skipped" if over else "queued"
                per_family["iso"][key].append((tag, f"ini/{tag}.ini", wall))

    # The doubled daughter grid.  Same (Gamma, mass) as a ladder cell, same
    # scheme and refine, ONLY dr_N_q multiplied -- so the pair with the ladder's
    # own `x` cell isolates the grid and nothing else.  See GRID_RUNG in
    # config.py for why these six and not a global increase.
    if "cmb" in families:
        spec = C.FAMILIES["cmb"]
        for gl, mass, mult in getattr(C, "GRID_RUNG", []):
            gamma = 10.0 ** float(gl)
            n_bg, capped = n_bg_for(gamma, mass, mult)
            tag = f"cmb_G{gl}_m{mass}_x_q{mult:g}x"
            work, wall, waves = cost(gamma, mass, "x", 1, C.N_K_CMB,
                                     spec["cpus"], C.MOMENTA_BINS_CMB)
            # An exact cell's k-loop cost is proportional to the grid it carries
            # in its ODE state, and COST_EXACT is tabulated at mult = 1.  The
            # BACKGROUND term inside cost() scales too, but it is measured at
            # mult = 1 as well, so scale the whole thing and accept that this
            # over-charges the background slightly.  Over-estimating is the safe
            # direction: it moves a cell to `skipped`, not into a wall.
            work, wall = work * mult, wall * mult
            over = wall > C.TARGET_WALL_H["cmb"] * C.WALL_MARGIN
            rows.append(dict(
                family="cmb", tag=tag, log10Gamma=gl, gamma=f"{gamma:.6g}",
                mass=mass, scheme="x", moments=0, refine=1, dr_N_q=n_bg,
                n_pt=n_bg, bins_per_decade=round(n_bg / C.DECADES, 1),
                momenta_bins=C.MOMENTA_BINS_CMB, n_k=C.N_K_CMB,
                cpus=spec["cpus"], waves=waves,
                est_work_core_h=round(work, 1), est_wall_h=round(wall, 1),
                est_charged_core_h=round(wall * spec["cpus"], 1),
                bg_unconverged=int(capped), queued=int(not over),
                ini=f"ini/{tag}.ini"))
            per_family["cmb"]["skipped" if over else "queued"].append(
                (tag, f"ini/{tag}.ini", wall))

    # A convergence twin without its production cell measures nothing -- the pair
    # is the measurement.  If the refine = 1 cell was over budget, drop the
    # refine = 2 twin too rather than spending cores on half a bracket.
    for fam, buckets in per_family.items():
        orphaned = {t[:-1] for t, _, _ in buckets["skipped"] if not t.endswith("c")}
        keep, moved = [], []
        for entry in buckets["queued"]:
            tag = entry[0]
            if tag.endswith("c") and tag[:-1] in orphaned:
                moved.append(entry)
            else:
                keep.append(entry)
        buckets["queued"], buckets["skipped"] = keep, buckets["skipped"] + moved
        for r in rows:
            if r["family"] == fam and r["tag"].endswith("c") \
                    and r["tag"][:-1] in orphaned:
                r["queued"] = 0
    return rows, per_family, bg_cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--family", action="append",
                    help="restrict to this family (repeatable)")
    a = ap.parse_args()

    families = a.family or ["cmb", "iso", "gal"]
    rows, per_family, bg_cells = build(families)

    if not a.dry_run:
        INI.mkdir(exist_ok=True)
        for f in INI.glob("*.ini"):
            f.unlink()
        (HERE / "EXPECTED_COMMIT").write_text(C.EXPECTED_COMMIT + "\n")
        for r in rows:
            fam, tag = r["family"], r["tag"]
            # Per-cell, not per-family: the momenta_bins rung overrides it.
            momenta = int(r["momenta_bins"])
            (INI / f"{tag}.ini").write_text(
                ini_text(fam, r["gamma"], r["mass"], r["scheme"], r["refine"],
                         r["dr_N_q"], momenta, tag))

    # Background pre-flight: one per (Gamma, mass), independent of scheme and
    # refine, because the background does not depend on either.
    bg = []
    for (gl, mass), n_bg in sorted(bg_cells.items()):
        tag = f"bg_G{gl}_m{mass}"
        if not a.dry_run:
            (INI / f"{tag}.ini").write_text(
                ini_text("bg", f"{10.0 ** float(gl):.6g}", mass, "x", 1,
                         n_bg, C.MOMENTA_BINS_CMB, tag))
        bg.append((tag, f"ini/{tag}.ini"))

    if not a.dry_run:
        for fam in families:
            q, s = per_family[fam]["queued"], per_family[fam]["skipped"]
            (HERE / f"runfile_{fam}").write_text("".join(f"{p}\n" for _, p, _ in q))
            (HERE / f"runfile_{fam}_skipped").write_text("".join(
                f"{p}   # est wall {w:.0f} h > TARGET_WALL_H[{fam}]="
                f"{C.TARGET_WALL_H[fam]:.0f}"
                + ("  (twin of a skipped production cell)\n"
                   if p.endswith("c.ini") else "\n")
                for _, p, w in s))
        (HERE / "runfile_bg").write_text("".join(f"{p}\n" for _, p in bg))
        # Family D is generated but never submitted: the RTA rate has to be
        # refitted from family B before a proxy cell means anything.
        (HERE / "runfile_proxy").write_text(
            "# DEFERRED -- see README section 'Family D'.  Refit the transport\n"
            "# rate from family B first, then add a proxy ladder to config.py.\n")
        with (HERE / "manifest.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    report(rows, per_family, bg, families, a.dry_run)
    return 0


def report(rows, per_family, bg, families, dry):
    print(f"{len(rows)} cells over {len(families)} families, "
          f"{len(bg)} background pre-flight tasks\n")
    print("Estimated WALL HOURS per cell -- an ordering heuristic, NOT a prediction.")
    print(f"A cell is skipped if it exceeds its family's TARGET_WALL_H "
          f"(the Slurm limit is {C.TIME_LIMIT} for all of them).\n")
    for fam in families:
        spec = C.FAMILIES[fam]
        n_k = C.N_K_ISO if fam == "iso" else C.N_K_CMB
        print(f"── family {fam}: {spec['cpus']} cores on {spec['partition']}, "
              f"{n_k} k modes, {max(1, math.ceil(n_k / spec['cpus']))} wave(s), "
              f"budget {C.TARGET_WALL_H[fam]:.0f} h")
        sub = [r for r in rows if r["family"] == fam]
        schemes = sorted({r["scheme"] for r in sub})
        masses = sorted({r["mass"] for r in sub}, key=float, reverse=True)
        for mass in masses:
            print(f"   m = {mass} eV")
            print(f"   {'log10G':>7} {'dr_N_q':>7} " +
                  " ".join(f"{s + ('/c' if any(r['refine'] > 1 for r in sub if r['scheme'] == s) else ''):>12}"
                           for s in schemes))
            gls = sorted({r["log10Gamma"] for r in sub if r["mass"] == mass})
            for gl in gls:
                cells, nq = [], ""
                for s in schemes:
                    got = [r for r in sub if r["mass"] == mass
                           and r["log10Gamma"] == gl and r["scheme"] == s]
                    if not got:
                        cells.append(f"{'-':>12}")
                        continue
                    nq = got[0]["dr_N_q"]
                    txt = "+".join(f"{r['est_wall_h']:.0f}{'' if r['queued'] else '*'}"
                                   for r in sorted(got, key=lambda r: r["refine"]))
                    cells.append(f"{txt:>12}")
                print(f"   {gl:>7} {nq:>7} " + " ".join(cells))
        q, s = per_family[fam]["queued"], per_family[fam]["skipped"]
        tot = sum(w * spec["cpus"] for _, _, w in q)
        print(f"   {len(q)} queued, {len(s)} over budget; "
              f"~{tot:.0f} charged core-hours "
              f"(~{tot / spec['cpus']:.0f} node-hours at {spec['cpus']} cores)\n")
    print("  * = over the family budget, written to runfile_<family>_skipped")
    print("  a+b in a cell = production (refine 1) + convergence twin (refine 2)")
    print("  CAP rows (bg_unconverged=1 in manifest.csv) mean dr_N_q hit N_BG_CAP,")
    print("     so the BACKGROUND is not converged there.  Run submit_bg.sh first.\n")

    if dry:
        return
    # Sync EVERY #SBATCH directive config.py owns, not just --array.  A knob that
    # moves the printed report but not the job is the worst kind: PARTITION, cpus,
    # mem and time were all dead config in an earlier harness for exactly that
    # reason.
    for fam in families + ["bg"]:
        spec = C.FAMILIES[fam]
        n = len(bg) if fam == "bg" else len(per_family[fam]["queued"])
        f = HERE / f"submit_{fam}.sh"
        if not f.exists():
            continue
        want = {
            "--array=": f"1-{max(n, 1)}%{spec['conc']}",
            "--partition=": spec["partition"],
            "--cpus-per-task=": str(spec["cpus"]),
            "--mem=": spec["mem"],
            "--time=": spec["time"],
        }
        seen, out = set(), []
        for line in f.read_text().splitlines():
            for key, val in want.items():
                if line.startswith(f"#SBATCH {key}") and key not in seen:
                    line = f"#SBATCH {key}{val}"
                    seen.add(key)
                    break
            out.append(line)
        f.write_text("\n".join(out) + "\n")
        os.chmod(f, 0o755)
        missing = [k for k in want if k not in seen]
        print(f"  submit_{fam}.sh: partition={spec['partition']} "
              f"cpus={spec['cpus']} mem={spec['mem']} time={spec['time']} "
              f"array=1-{max(n, 1)}%{spec['conc']}")
        if missing:
            print(f"    ! no #SBATCH line to patch for {missing} -- add one")

    # Family B's threads MUST match its mode count, or cores sit idle for the
    # whole run (or, worse, modes queue behind each other and the wall doubles).
    if "iso" in families and C.FAMILIES["iso"]["cpus"] != C.N_K_ISO:
        print(f"\n  ! family iso has {C.FAMILIES['iso']['cpus']} cores for "
              f"{C.N_K_ISO} k modes -- set cpus = N_K_ISO in config.py")
    print(f"\nwrote {len(list(INI.glob('*.ini')))} inis, manifest.csv, "
          f"runfile_* and EXPECTED_COMMIT ({C.EXPECTED_COMMIT})")


if __name__ == "__main__":
    sys.exit(main())
