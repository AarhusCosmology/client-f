"""Where to put the phi window of an axion run, whether CLASS++'s sampling resolves the oscillation, and the cost.

    sbatch smoke/submit_axion_probe.sh axion_pp_high

The primordial spectrum of the run is A_s (k/k*)^(n_s-1) [1 + A_osc sin(omega ln(k/k*) + phi)],
periodic in phi. CLiENT's training box is not periodic, so phi gets a window
[phi0 - pi, phi0 + pi], and a mode whose phase lies at the window's edge would be cut in two.
This program chooses phi0 before the run, with true evaluations only.

1. The phase probe. With every other parameter at the run's reference point (the centre of
   each parameter's `ref` in its cobaya.yaml), the true log-posterior on a grid of omega
   (--omega-step) over Cobaya's prior range (--omega-range: the parts of a domain split in omega
   share the cobaya.yaml, so the prior and the window; the box does not restrict a true
   evaluation; --omega-prior-max widens the prior in a copy of the file, to see what lies beyond
   it) and --phases phases over one period, at two amplitudes (--amplitudes), and
   once at A_osc = 0. The C_l are linear in A_osc cos(phi) and A_osc sin(phi) (up to lensing,
   a per-cent effect) and Plik and CamSpec are Gaussian in the C_l, so at each omega the log-posterior is a
   quadratic in A_osc whose slope is a first harmonic in phi and whose curvature a constant
   plus a second harmonic: the three amplitudes fix it, and give the best gain over A_osc = 0
   profiled over A_osc and phi (`profile_quadratic`). The amplitudes need not be near the best
   ones: on Plik TT the first probe, at 0.05 and 0.1, had every gain negative, and the profile
   from those evaluations put the best amplitudes at 0.006 to 0.018.
   The peaks that place the edge are the local maxima in omega with a chi2 gain of at least
   --min-chi2: their phase width 1/sqrt(2 gain) is at most about 1 rad, where below that a
   "peak" is the A_osc = 0 plateau and its phase means nothing. Each is weighted by its mass,
   exp(gain) A* / (2 gain) (height times its widths in A_osc and phi), and the edge goes where
   the least of that mass lies beyond it (`choose_phi0_mass`); phi0 is opposite the edge.
2. The sampling check (unless --no-check). The strongest peaks at their best phase, the
   highest omega, and the frequencies of --check-omegas at A_osc = --check-amplitude (each at
   the profile's best phase there), are evaluated again with CLASS++'s sampling changed: one
   setting, or a ladder of settings separated by ';' (--dense). CLASS++'s defaults are l_logstep
   1.12 and l_linstep 50 (40 before 0a57bbc6) for the multipoles at which the transfer functions
   are computed, and
   q_linstep_trans 0.45 and q_logstep_trans 400 for the wavenumbers over which the C_l integrate
   the primordial spectrum: at omega = 50 the latter gives 2 to 6 wavenumbers per period of
   the oscillation below l ~ 140. The gain over A_osc = 0 is compared with the last setting of
   the ladder, taken as converged: a difference of more than a few tenths says the run's
   likelihood at those frequencies is set by the sampling, not by the data.
3. The cost: every evaluation is timed, and the probe reports the median and the 90th percentile
   in seconds per evaluation (one rank, one thread), for the run's settings and for each
   setting of the ladder.

Output: logs/axion_probe_<job>.out and logs/axion_probe_<job>.json. Nothing in runs/ is
written (the next sync_to_hpc.sh would remove it): phi0 goes into the run's cobaya.yaml and
domain_bounds.json on the laptop.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent.parent
TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------- the analysis (tested on the laptop)
def profile_quadratic(phases, c1, c2, a1, a2, a_max=0.5):
    """At one omega, from the gains over A_osc = 0 at the given phases, c1 at amplitude a1 and c2 at
    a2: the log-posterior is gain(A, phi) = A g(phi) - A^2 f(phi) / 2, with g a first harmonic in
    phi and f a constant plus a second harmonic. Returns the phase, amplitude (in [0, a_max]) and
    gain of its maximum, and the rms residuals of the two harmonic fits (how well that holds)."""
    phases, c1, c2 = (np.asarray(v, float) for v in (phases, c1, c2))
    good = np.isfinite(c1) & np.isfinite(c2)
    if good.sum() < 3:
        return dict(phi=float("nan"), A=float("nan"), gain=float("nan"), slope_rms=float("nan"), curvature_rms=float("nan"))
    p, c1, c2 = phases[good], c1[good], c2[good]
    f = 2.0 * (a2 * c1 - a1 * c2) / (a1 * a2 * (a2 - a1))
    g = (c1 + 0.5 * a1 ** 2 * f) / a1
    first = lambda q: np.stack([np.cos(q), np.sin(q)], axis=-1)
    second = lambda q: np.stack([np.ones_like(q), np.cos(2 * q), np.sin(2 * q)], axis=-1)
    cg, *_ = np.linalg.lstsq(first(p), g, rcond=None)
    cf, *_ = np.linalg.lstsq(second(p), f, rcond=None)
    fine = np.linspace(-math.pi, math.pi, 7200, endpoint=False)
    G, F = first(fine) @ cg, second(fine) @ cf
    A = np.where((G > 0) & (F > 0), np.clip(G / np.where(F > 0, F, 1.0), 0.0, a_max), 0.0)
    A = np.where((G > 0) & (F <= 0), a_max, A)
    gain = G * A - 0.5 * F * A ** 2
    k = int(np.argmax(gain))
    return dict(phi=float(fine[k]), A=float(A[k]), gain=float(gain[k]),
                slope_rms=float(np.sqrt(np.mean((first(p) @ cg - g) ** 2))),
                curvature_rms=float(np.sqrt(np.mean((second(p) @ cf - f) ** 2))))


def peaks_with_phase(profile, min_chi2):
    """The strict local maxima in omega of the profiled gain whose chi2 gain is at least min_chi2."""
    gains = np.array([o["gain"] if np.isfinite(o["gain"]) else -np.inf for o in profile])
    out = []
    for i in range(len(gains)):
        left = gains[i - 1] if i > 0 else -np.inf
        right = gains[i + 1] if i + 1 < len(gains) else -np.inf
        if gains[i] > left and gains[i] > right and 2.0 * gains[i] >= min_chi2:
            out.append(profile[i])
    return out


def _circ(a, b):
    return np.abs(np.angle(np.exp(1j * (np.asarray(a) - b))))


def split_mass(peaks, edge):
    """The share of the peaks' mass that lies beyond a window edge at `edge`: each peak a Gaussian
    in phi of width 1/sqrt(2 gain), weighted by exp(gain) A* / (2 gain)."""
    if not peaks:
        return 0.0
    mass = np.array([math.exp(p["gain"]) * p["A"] / (2.0 * p["gain"]) for p in peaks])
    width = np.array([1.0 / math.sqrt(2.0 * p["gain"]) for p in peaks])
    beyond = np.array([0.5 * math.erfc(d / math.sqrt(2.0)) for d in _circ([p["phi"] for p in peaks], edge) / width])
    return float((mass * beyond).sum() / mass.sum())


def choose_phi0_mass(peaks):
    """The window centre phi0 whose edge (phi0 + pi) splits the least of the peaks' mass, and that
    share. (0, 0) without peaks: the window stays where it is."""
    if not peaks:
        return 0.0, 0.0
    edges = np.linspace(-math.pi, math.pi, 7200, endpoint=False)
    shares = np.array([split_mass(peaks, e) for e in edges])
    e = float(edges[int(np.argmin(shares))])
    return float(np.angle(np.exp(1j * (e + math.pi)))), float(shares.min())


def analyse(names, rows, values, omegas, amplitudes, min_chi2, a_max=0.5):
    """The profile over A_osc and phi at every omega, the peaks with a phase, and phi0, from the
    probe's evaluations (rows in `names` order; the first two amplitudes are used)."""
    ia, io, ip = names.index("A_osc"), names.index("omega"), names.index("phi")
    base = values[rows[:, ia] == 0.0]
    base = float(np.nanmax(base)) if np.isfinite(base).any() else float("nan")
    a1, a2 = amplitudes[:2]
    profile = []
    for w in omegas:
        s1 = (rows[:, ia] == a1) & np.isclose(rows[:, io], w)
        s2 = (rows[:, ia] == a2) & np.isclose(rows[:, io], w)
        o1, o2 = np.argsort(rows[s1, ip]), np.argsort(rows[s2, ip])
        if not np.allclose(rows[s1, ip][o1], rows[s2, ip][o2]):
            raise ValueError(f"omega {w}: the two amplitudes were not evaluated at the same phases")
        one = profile_quadratic(rows[s1, ip][o1], values[s1][o1] - base, values[s2][o2] - base, a1, a2, a_max)
        profile.append(dict(omega=float(w), **one))
    peaks = peaks_with_phase(profile, min_chi2)
    phi0, split = choose_phi0_mass(peaks)
    return dict(baseline=base, profile=profile, peaks=peaks, phi0=phi0, split=split)


def report(result, window_lo, min_chi2, print_=print):
    """The probe's account of its profile and its choice of phi0."""
    prof = result["profile"]
    print_(f"log-posterior at A_osc = 0: {result['baseline']:.3f}; quadratic model in A_osc: largest residuals "
           f"{max(o['slope_rms'] for o in prof):.3g} (slope) and {max(o['curvature_rms'] for o in prof):.3g} (curvature)")
    print_("the ten largest gains over A_osc = 0, profiled over A_osc and phi (every other parameter at the reference point):")
    for o in sorted(prof, key=lambda o: -o["gain"] if np.isfinite(o["gain"]) else 1e300)[:10]:
        print_(f"   omega {o['omega']:6.2f}  A_osc {o['A']:.4f}  phi {o['phi']:+.3f}  gain {o['gain']:+7.3f}  (chi2 {2 * o['gain']:+.2f})")
    peaks = result["peaks"]
    current_edge = window_lo                                   # the edge of the present window
    print_(f"peaks with a chi2 gain of at least {min_chi2} (they place the window): {len(peaks)}")
    if peaks:
        mass = np.array([math.exp(p["gain"]) * p["A"] / (2.0 * p["gain"]) for p in peaks])
        new_edge = result["phi0"] + math.pi
        for p, m in sorted(zip(peaks, mass), key=lambda t: -t[1]):
            width = 1.0 / math.sqrt(2.0 * p["gain"])
            print_(f"   omega {p['omega']:6.2f}  phi {p['phi']:+.3f}  A_osc {p['A']:.4f}  chi2 gain {2 * p['gain']:5.2f}  "
                   f"{100 * m / mass.sum():5.1f}% of the mass;  {float(_circ(p['phi'], new_edge)) / width:4.1f} phase widths from "
                   f"the new edge, {float(_circ(p['phi'], current_edge)) / width:4.1f} from the present one")
        print_(f"\nphi0 = {result['phi0']:+.4f}: window [{result['phi0'] - math.pi:+.4f}, {result['phi0'] + math.pi:+.4f}] splits "
               f"{100 * result['split']:.1f}% of their mass; the present window [{window_lo:+.4f}, {window_lo + TWO_PI:+.4f}] "
               f"{100 * split_mass(peaks, current_edge):.1f}%")
    else:
        print_("\nno peak with a phase: the window stays where it is")


def dense_input(info, settings):
    """The run's Cobaya input with CLASS++'s extra arguments updated, or None without classy."""
    theory = (info.get("theory") or {}).get("classy")
    if theory is None:
        return None
    out = json.loads(json.dumps(info))
    out["theory"]["classy"].setdefault("extra_args", {}).update(settings)
    return out


def parse_settings(text):
    out = {}
    for item in filter(None, (s.strip() for s in text.split(","))):
        key, value = item.split("=", 1)
        out[key.strip()] = float(value) if "." in value or "e" in value.lower() else int(value)
    return out


def parse_ladder(text):
    """--dense: one setting, or several separated by ';' (the last taken as converged)."""
    return [s.strip() for s in text.split(";") if s.strip()]


def seconds(times):
    """Median and 90th percentile of the timed evaluations, in seconds."""
    t = np.asarray(times, float)
    t = t[np.isfinite(t)]
    return (float(np.median(t)), float(np.percentile(t, 90))) if len(t) else (float("nan"), float("nan"))


def print_check(check, ladder, print_=print):
    """The gains of the check's points under the run's settings and each setting of the ladder,
    and their differences from the last (converged) setting."""
    last = ladder[-1]
    for c in check[1:]:
        gains = c.get("gains") or {last: c["gain_dense"]}
        print_(f"   omega {c['omega']:6.2f}  phi {c['phi']:+.3f}  A_osc {c['A_osc']:.3g}:  gain {c['gain_run']:+8.3f} (run)  "
               + "  ".join(f"{gains[s]:+8.3f} ({k + 1})" for k, s in enumerate(ladder))
               + f"  run - converged {c['gain_run'] - gains[last]:+.3f}")
    for k, s in enumerate(ladder):
        print_(f"   ({k + 1}) {s}")


def reference_point(info, names, box):
    """Each sampled parameter at the centre of its `ref` (loc, or the middle of min/max, or a
    number); the box's centre where there is none."""
    point = []
    for name in names:
        ref = info["params"][name].get("ref") if isinstance(info["params"][name], dict) else None
        if isinstance(ref, (int, float)):
            value = float(ref)
        elif isinstance(ref, dict) and "loc" in ref:
            value = float(ref["loc"])
        elif isinstance(ref, dict) and "min" in ref and "max" in ref:
            value = 0.5 * (float(ref["min"]) + float(ref["max"]))
        else:
            value = 0.5 * (box[name][0] + box[name][1])
        point.append(min(max(value, box[name][0]), box[name][1]))
    return np.array(point)


# ---------------------------------------------------------------- the job
def offline(args):
    """--from-json: the analysis of a probe already made, from its JSON, on the laptop."""
    d = json.loads(Path(args.from_json).read_text())
    names, rows = d["names"], np.array(d["rows"], float)
    values = np.array([np.nan if v is None else v for v in d["values"]], float)
    result = analyse(names, rows, values, np.array(d["omegas"]), d["amplitudes"], args.min_chi2,
                     a_max=d.get("a_max", 0.5))
    report(result, d["phases"][0], args.min_chi2)
    if d.get("check"):
        print("\nits sampling check:")
        print_check(d["check"], d.get("ladder") or [d.get("dense") or "dense"])
    for name, cost in (d.get("cost") or {}).items():
        print(f"cost, {name}: median {cost[0]:.1f} s, 90th percentile {cost[1]:.1f} s per evaluation")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?")
    p.add_argument("--amplitudes", default="0.02,0.05", help="the two A_osc values of the grid (any two; see 1. above)")
    p.add_argument("--omega-step", type=float, default=0.25)
    p.add_argument("--omega-range", default="prior",
                   help="omega range of the phase probe: 'prior' (Cobaya's, shared by the parts of a domain split in "
                        "omega, as is the phi window), 'box' (this run's training box) or LO:HI")
    p.add_argument("--omega-prior-max", type=float, default=None,
                   help="evaluate with a copy of the run's cobaya.yaml whose omega prior ends here instead "
                        "(logs/axion_probe_<job>_wide.yaml): to see what lies beyond the prior before choosing it")
    p.add_argument("--phases", type=int, default=8)
    p.add_argument("--min-chi2", type=float, default=1.0, help="peaks that place the window: chi2 gain at least this")
    p.add_argument("--check-peaks", type=int, default=4)
    p.add_argument("--check-omegas", default="10,20,30,40,50",
                   help="frequencies also checked, at --check-amplitude and the profile's best phase ('' for none)")
    p.add_argument("--check-amplitude", type=float, default=0.02)
    p.add_argument("--dense", default="l_logstep=1.12,l_linstep=50;l_logstep=1.05,l_linstep=16;"
                                      "l_logstep=1.01,l_linstep=3,q_linstep_trans=0.05,q_logstep_trans=45",
                   help="CLASS++ settings of the sampling check: one, or a ladder separated by ';' (the last converged). "
                        "The default: CLASS++'s defaults (0a57bbc6), a cheaper denser setting, and the converged one")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--from-json", help="analyse a probe already made (its logs/axion_probe_<job>.json); no evaluation")
    args = p.parse_args()
    if args.from_json:
        offline(args)
        return
    if not args.run:
        p.error("a run, or --from-json")

    from mpi4py import MPI

    sys.path.insert(0, str(HERE))
    import client_paths  # noqa: E402,F401  CLiENT and its hpc/ tools (domain, loading)
    from domain import apply_domain, read_domain  # noqa: E402
    from loading import build_in_turn  # noqa: E402
    from likelihood.cobaya import CobayaLikelihood  # noqa: E402
    from utils.mpi_utils import broadcast_and_evaluate  # noqa: E402

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    job = os.environ.get("SLURM_JOB_ID", "local")
    run = HERE / "runs" / args.run
    cfg = yaml.safe_load((run / "client.yaml").read_text())
    source = HERE / cfg["likelihood"]["input"]
    info = yaml.safe_load(source.read_text())
    if args.omega_prior_max is not None:                    # a probe-only copy: the run's file is not touched
        info["params"]["omega"]["prior"]["max"] = float(args.omega_prior_max)
        source = HERE / "logs" / f"axion_probe_{job}_wide.yaml"
        if rank == 0:
            source.write_text(yaml.safe_dump(info, sort_keys=False))
            print(f"omega's prior widened to [{info['params']['omega']['prior']['min']}, {args.omega_prior_max}] "
                  f"for this probe only ({source.name})", flush=True)
        comm.Barrier()

    def build(path):
        lik = build_in_turn(comm, lambda: CobayaLikelihood(str(path)), label="[axion probe]")
        apply_domain(lik, read_domain(run / "domain_bounds.json"))
        return lik

    lik = build(source)
    names = list(lik.param_names)
    box = {n: tuple(lik.prior_bounds[n]) for n in names}
    for needed in ("A_osc", "omega", "phi"):
        if needed not in names:
            raise SystemExit(f"{args.run} samples no {needed}")
    amplitudes = [float(a) for a in args.amplitudes.split(",")]
    if len(amplitudes) != 2 or not 0 < amplitudes[0] < amplitudes[1] <= box["A_osc"][1]:
        raise SystemExit(f"--amplitudes: two increasing values in (0, {box['A_osc'][1]}]")
    if args.omega_range == "box":
        lo_w, hi_w = box["omega"]
    elif args.omega_range == "prior":                      # Cobaya's: the parts of a split share it and the window
        lo_w, hi_w = (float(info["params"]["omega"]["prior"][k]) for k in ("min", "max"))
    else:
        lo_w, hi_w = (float(v) for v in args.omega_range.split(":"))
    omegas = np.round(np.arange(lo_w, hi_w + 1e-9, args.omega_step), 10)
    lo_p = box["phi"][0]
    phases = lo_p + TWO_PI * np.arange(args.phases) / args.phases      # one period inside the window
    ref = reference_point(info, names, box)
    ia, io, ip = names.index("A_osc"), names.index("omega"), names.index("phi")

    rows = None
    if rank == 0:
        grid = [ref.copy()]
        grid[0][ia] = 0.0
        for a in amplitudes:
            for w in omegas:
                for ph in phases:
                    x = ref.copy()
                    x[ia], x[io], x[ip] = a, w, ph
                    grid.append(x)
        rows = np.array(grid)
        print(f"== {args.run}: {len(rows)} evaluations ({len(omegas)} omegas x {len(phases)} phases x "
              f"{len(amplitudes)} amplitudes, and A_osc = 0) on {comm.Get_size()} ranks", flush=True)
        print("reference point: " + "  ".join(f"{n}={v:.6g}" for n, v in zip(names, ref)), flush=True)

    def protected(target):
        def f(x):
            t = time.perf_counter()
            try:
                value = float(target.loglkl(x))
            except Exception:
                value = float("nan")
            return [value, time.perf_counter() - t]
        return f

    def evaluate(points, target):
        """The log-posteriors and the seconds of each evaluation (on rank 0; empty elsewhere)."""
        pts, out = broadcast_and_evaluate(points, protected(target))
        if rank != 0:
            return pts, None, None
        out = np.asarray(out, float).reshape(-1, 2)
        return np.asarray(pts, float), out[:, 0], out[:, 1]

    cost = {}
    t0 = time.perf_counter()
    rows, values, times = evaluate(rows, lik)
    result = None
    if rank == 0:
        result = analyse(names, rows, values, omegas, amplitudes, args.min_chi2, a_max=box["A_osc"][1])
        median, p90 = cost["run's settings, phase probe"] = seconds(times)
        print(f"\nphase probe: {time.perf_counter() - t0:.0f} s; {int(np.isfinite(values).sum())} of {len(values)} finite; "
              f"median {median:.1f} s, 90th percentile {p90:.1f} s per evaluation", flush=True)
        report(result, lo_p, args.min_chi2)

    check, ladder = None, parse_ladder(args.dense)
    if not args.no_check and dense_input(info, {}) is None:
        if rank == 0:
            print("\nsampling check: the run's input has no classy theory; skipped")
    elif not args.no_check:
        points = None
        if rank == 0:
            picks = sorted(result["peaks"], key=lambda f: -f["gain"])[: args.check_peaks]
            picks.append(result["profile"][-1])                           # the highest frequency, at its best
            picks = [dict(A=f["A"], omega=f["omega"], phi=f["phi"]) for f in picks if f["A"] > 0]
            for w in (float(v) for v in args.check_omegas.split(",") if v.strip()):
                at = min(result["profile"], key=lambda o: abs(o["omega"] - w))
                if np.isfinite(at["phi"]):
                    picks.append(dict(A=args.check_amplitude, omega=at["omega"], phi=at["phi"]))
            base = ref.copy()
            base[ia] = 0.0
            points = [base]
            for f in picks:
                x = ref.copy()
                x[ia], x[io], x[ip] = f["A"], f["omega"], lo_p + np.mod(f["phi"] - lo_p, TWO_PI)   # into the window
                points.append(x)
            points = np.array(points)
            print(f"\nsampling check: {len(points)} points with the run's settings and with each of {len(ladder)}: "
                  + " ; ".join(ladder), flush=True)
        t1 = time.perf_counter()
        pts, plain, times = evaluate(points, lik)
        if rank == 0:
            cost["run's settings, check"] = seconds(times)
        ladder_values = []
        for k, setting in enumerate(ladder):
            dense_path = HERE / "logs" / f"axion_probe_{job}_dense{k + 1}.yaml"
            if rank == 0:
                dense_path.write_text(yaml.safe_dump(dense_input(info, parse_settings(setting)), sort_keys=False))
            comm.Barrier()
            heavy = build(dense_path)                     # one model at a time: CamSpec's memory per rank
            _, fine, times = evaluate(points, heavy)
            del heavy
            if rank == 0:
                ladder_values.append(fine)
                cost[f"({k + 1}) {setting}"] = seconds(times)
        if rank == 0:
            check = []
            for i, x in enumerate(pts):
                gains = {s: float(v[i] - v[0]) for s, v in zip(ladder, ladder_values)}
                check.append(dict(A_osc=float(x[ia]), omega=float(x[io]), phi=float(x[ip]), run=float(plain[i]),
                                  gain_run=float(plain[i] - plain[0]), gains=gains, gain_dense=gains[ladder[-1]],
                                  values={s: float(v[i]) for s, v in zip(ladder, ladder_values)}))
            print(f"   ({time.perf_counter() - t1:.0f} s)  log-posterior at A_osc = 0: run {plain[0]:.3f}, "
                  + ", ".join(f"({k + 1}) {v[0]:.3f}" for k, v in enumerate(ladder_values)))
            print_check(check, ladder)
            worst = max(abs(c["gain_dense"] - c["gain_run"]) for c in check[1:])
            print(f"   largest difference in gain, run - converged: {worst:.3f}"
                  + ("  -- MORE THAN 0.3: the sampling matters at these frequencies" if worst > 0.3 else ""), flush=True)

    if rank == 0:
        for name, c in cost.items():
            print(f"cost, {name}: median {c[0]:.1f} s, 90th percentile {c[1]:.1f} s per evaluation")
        out = HERE / "logs" / f"axion_probe_{job}.json"
        out.write_text(json.dumps(dict(run=args.run, names=names, reference=ref.tolist(), amplitudes=amplitudes,
                                       omegas=omegas.tolist(), phases=phases.tolist(), rows=rows.tolist(),
                                       values=[None if not np.isfinite(v) else float(v) for v in values],
                                       a_max=box["A_osc"][1], baseline=result["baseline"], profile=result["profile"], peaks=result["peaks"],
                                       phi0=result["phi0"], split=result["split"], check=check,
                                       dense=ladder[-1] if check is not None else None,
                                       ladder=ladder if check is not None else None, cost=cost,
                                       omega_prior_max=args.omega_prior_max), indent=1) + "\n")
        print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
