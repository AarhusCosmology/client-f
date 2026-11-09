"""Seconds per evaluation of a run's likelihood on a full node, measured BEFORE its long job.

    sbatch smoke/submit_timing.sh [RUN]            # default lcdm_planck_desi

Draws points uniformly in the run's training box (where the initial design lands), evaluates
them through CLiENT's CobayaLikelihood and CLiENT's own MPI dispatcher, with every rank busy
as in the real run, and reports seconds per evaluation, failures, peak memory per rank, and
the wall-clock time that implies for the run's evaluations. The laptop cannot stand in for
this: the harness this script comes from measured a LambdaCDM evaluation at 1.9 s on a
laptop and 6 to 8 s on a loaded q48 node.
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from mpi4py import MPI

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import client_paths  # noqa: E402,F401  CLiENT's hpc/ tools (domain, loading, refused)
from domain import apply_domain, read_domain  # noqa: E402
from loading import build_in_turn, memory_gb  # noqa: E402
from likelihood.cobaya import CobayaLikelihood  # noqa: E402
from utils.mpi_utils import broadcast_and_evaluate  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("run", nargs="?", default="lcdm_planck_desi")
p.add_argument("--per-rank", type=int, default=4, help="evaluations per evaluating rank")
p.add_argument("--iterations", type=int, default=16, help="networks of the planned run, for the projection")
args = p.parse_args()

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
run = HERE / "runs" / args.run
cfg = yaml.safe_load((run / "client.yaml").read_text())
lik = build_in_turn(comm, lambda: CobayaLikelihood(str(HERE / cfg["likelihood"]["input"])), label="[timing]")   # CLIENT_LOAD_GROUP
if (run / "domain_bounds.json").exists():
    apply_domain(lik, read_domain(run / "domain_bounds.json"))
else:
    lik.restrict_prior_bounds(cfg["prior"]["n_sigma"])
names = lik.param_names
box = np.array([lik.prior_bounds[n] for n in names], dtype=float)

schedule = os.environ.get("CLIENT_MPI_SCHEDULE", "").strip().lower() or "default"
evaluators = size - 1 if schedule == "dynamic" and size >= 4 else size   # the dynamic dispatcher does not evaluate
n = args.per_rank * evaluators
points = None
if rank == 0:
    rng = np.random.default_rng(1509)
    points = box[:, 0] + rng.random((n, len(names))) * (box[:, 1] - box[:, 0])
    print(f"{args.run}: {n} points uniform in the training box, {size} ranks "
          f"({schedule} schedule, {evaluators} evaluating)", flush=True)

log = []
raised = {}                                # kind of exception -> count, on this rank


def timed(x):
    t = time.perf_counter()
    try:
        value = float(lik.loglkl(x))
    except Exception as error:             # NOT what the loop does: it stops on an exception. Reported below.
        value = float("nan")
        kind = f"{type(error).__name__}: " + re.sub(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", "#", str(error))[:140]
        raised[kind] = raised.get(kind, 0) + 1
    log.append((time.perf_counter() - t, value, *x))
    return value


t0 = time.perf_counter()
broadcast_and_evaluate(points, timed)
wall = time.perf_counter() - t0
now_gb, peak_gb = memory_gb()
gathered = comm.gather((MPI.Get_processor_name(), (now_gb, peak_gb), log), root=0)
all_raised = comm.gather(raised, root=0)
if rank != 0:
    sys.exit(0)

rows = np.array([r for _, _, lg in gathered for r in lg])
value = rows[:, 1]
bad = ~np.isfinite(value)
sec = rows[~bad, 0]                      # a refused point costs almost nothing: timed apart
print(f"\n{len(rows)} evaluations in {wall:.0f} s wall; {int((~bad).sum())} finite, {int(bad.sum())} not finite")
if len(sec):
    print(f"seconds per finite evaluation: median {np.median(sec):.1f}  mean {sec.mean():.1f}  "
          f"p90 {np.percentile(sec, 90):.1f}  max {sec.max():.1f}  min {sec.min():.1f}")
if (~bad).any():
    from refused import design_value

    suggested, _ = design_value(value)
    print(f"finite log-posterior: best {value[~bad].max():.6g}  median {np.median(value[~bad]):.6g}  worst {value[~bad].min():.6g}; "
          f"CLIENT_REFUSED_VALUE=design would give about {suggested:g} (a run takes it from its own design)")
if bad.any():
    print(f"NON-FINITE log-posterior at {bad.sum()} of {len(rows)} points ({100 * bad.mean():.0f}%; the loop drops such "
          f"points), {np.median(rows[bad, 0]):.2f} s each (median). The first of them:")
    for r in rows[bad][:5]:
        print("   " + "  ".join(f"{n}={v:.5g}" for n, v in zip(names, r[2:])))
else:
    print("every log-posterior finite")
kinds = {}
for one in all_raised:
    for kind, count in one.items():
        kinds[kind] = kinds.get(kind, 0) + count
if kinds:
    print(f"\nEXCEPTIONS at {sum(kinds.values())} of the {len(rows)} points. A refusal that comes back as -inf is harmless; an "
          "exception stops CLiENT's loop unless the launcher turns it into -inf (refused.py does that for Cobaya's "
          "logged errors, and for those only):")
    for kind, count in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"   {count:4d}  {kind}")
else:
    print("no evaluation raised an exception")
per_node = {}
for host, gb, _ in gathered:
    per_node.setdefault(host, []).append(gb)
for host, gbs in per_node.items():
    peaks = [p for _, p in gbs]
    print(f"peak RSS on {host}: max {max(peaks):.2f} GB per rank (reached while the data are read)")
    if all(n is not None for n, _ in gbs):
        nows = [n for n, _ in gbs]
        print(f"RSS now on {host}: max {max(nows):.2f} GB per rank, {sum(nows):.0f} GB for its {len(gbs)} ranks")

# What that means for the run: CLiENT evaluates n_samples, then n_append per iteration
# (plus the audit's tenth of a batch when acquisition.audit is set).
design, append = cfg["prior"]["n_samples"], cfg["acquisition"]["n_append"]
audit = 0.1 * append if "audit" in cfg["acquisition"] else 0.0
total = design + (append + audit) * (args.iterations - 1) + audit
mean = sec.mean() if len(sec) else float("nan")
print(f"\nrun of {args.iterations} networks: about {total:.0f} evaluations, "
      f"{total * mean / 3600:.0f} core-hours of evaluation at {mean:.1f} s each")
print(f"  on this allocation ({evaluators} evaluating ranks): {total * mean / evaluators / 3600:5.1f} h of evaluation, "
      f"plus training and sampling on rank 0 (see README, 'How long')")
