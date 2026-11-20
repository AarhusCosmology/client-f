"""The cost of one true evaluation in the two CLiENT runs, from their logs (10 November 2026): table of section 5.

    python3 -I eval_costs.py [KIT_DIR]        # KIT_DIR: CLIENT_PAPER_KIT, or else this branch's papers/neutrino_decay/

CLiENT writes a "start" record (time, parameters, stage) for every call of the true likelihood into
results/<run>/objective_calls/rank*.jsonl; the "return" record has no time. A rank evaluates its points of a batch one
after another, so the time between two starts of the same rank in the same stage is the duration of the first call
(plus negligible overhead); the last call of a rank in a stage, whose gap holds the wait for the next stage, is left
out, as are gaps above 3000 s. The batch totals (evaluations, core-hours) come from logs/evaluations_<job>.jsonl.
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

KIT = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CLIENT_PAPER_KIT") or Path(__file__).resolve().parents[1])
RUNS = {"Planck+BAO+SN": ("20261106_121906_dncdm_proxy_pp", "7047807"),
        "SPA+BAO+SN": ("20261106_122443_dncdm_proxy_wc", "7047808")}
BINS = ((-2, 2), (2, 6), (6, 9), (9, 11), (11, 12), (12, 12.5), (12.5, 13), (13, 13.5), (13.5, 14))

for name, (run, job) in RUNS.items():
    rows = []
    for f in glob.glob(str(KIT / "results" / run / "objective_calls" / "*.jsonl")):
        starts = []
        for line in open(f):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("event") == "start":
                starts.append((e["time"], e["x"][0], e["stage"]))
        starts.sort()
        rows += [(g, t1 - t0) for (t0, g, s0), (t1, _, s1) in zip(starts, starts[1:]) if s0 == s1 and t1 - t0 < 3000]
    a = np.array(rows)
    batches = [json.loads(line) for line in open(KIT / "logs" / f"evaluations_{job}.jsonl")]
    points = sum(b["points"] for b in batches)
    core_h = sum(b["seconds"] * b["ranks"] for b in batches) / 3600
    wall_h = sum(b["seconds"] for b in batches) / 3600
    print(f"{name}: {points} evaluations in {len(batches)} batches on {batches[0]['ranks']} cores, {core_h:.0f} core-hours, "
          f"{wall_h:.1f} h of evaluation; one evaluation median {np.median(a[:, 1]):.0f} s ({len(a)} timed)")
    for lo, hi in BINS:
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if m.sum():
            print(f"   log10Gamma [{lo:5}, {hi:5}): {m.sum():6d} calls, median {np.median(a[m, 1]):5.0f} s")
