"""Where a profile study stands, while it runs or afterwards.

    python true_profiles/status.py results_profiles/<name>

Reads only the study's own files (no likelihood, no network): the stage, the nodes finished
with how they ended and what they cost, the true calls so far against the budget, and an
estimate of the time left from the rate of the last calls.

How a node ended: `converged` (a measured gradient predicts less than the tolerance),
`budget` (it used its `maxfun` calls), `stalled` (no step from a measured gradient improved
the value), `unverified_bound` (a probe inside a bound did not evaluate). A study in which
most nodes end on `budget` or `stalled` is fighting noise or a poor start: stop it and look
(README, "True profiles").
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])
    manifest = json.loads((out / "manifest.json").read_text())
    config = manifest["configuration"]
    parameters = config["parameters"]
    planned = len(parameters) * config["profile_points"]
    stage = (out / "status.txt").read_text().strip() if (out / "status.txt").exists() else "?"
    print(f"{out.name}: {len(parameters)} parameters x {config['profile_points']} points = {planned} nodes, from network "
          f"{manifest['iteration']} of {Path(manifest['run']).name}")
    print(f"stage: {stage}" + ("   (COMPLETE)" if (out / "complete.json").exists() else ""))
    if (out / "failure.json").exists() and not (out / "complete.json").exists():
        print("last failure:", json.loads((out / "failure.json").read_text()).get("error", "")[:300])

    # true calls, from the journals
    starts, times, seconds = 0, [], []
    for journal in sorted((out / "calls").glob("attempt_*.jsonl")):
        for line in journal.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:                       # a line being written
                continue
            if event["event"] == "start":
                starts += 1
            elif event["event"] == "return":
                times.append(event["time"])
                seconds.append(event.get("seconds", np.nan))
    if not starts and (out / "counts.json").exists():          # the journals were left on the cluster
        starts = json.loads((out / "counts.json").read_text()).get("backend_attempts", 0)
    print(f"true calls: {starts} of the budget {config['total_call_budget']}"
          + (f"; a call takes {np.nanmedian(seconds):.1f} s (median)" if seconds else ""))

    nodes = []
    for path in sorted((out / "true").glob("node_*.json")):
        try:
            node = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        p, j = (int(x) for x in path.stem.split("_")[1:3])
        d = node.get("diagnostics", {})
        nodes.append(dict(parameter=parameters[p], index=j, reason=d.get("reason", "?"), calls=d.get("backend_calls", 0),
                          rounds=d.get("gradient_rounds", 0), gain=(node["logl"] - node["direct_logl"])
                          if node.get("direct_logl") is not None and node.get("logl") is not None else np.nan))
    if not nodes:
        print("no node of the true stage has finished yet")
        return
    calls = np.array([n["calls"] for n in nodes], float)
    gains = np.array([n["gain"] for n in nodes], float)
    reasons = Counter(n["reason"] for n in nodes)
    print(f"nodes finished: {len(nodes)} of {planned}; ended as " + ", ".join(f"{k} {v}" for k, v in reasons.most_common()))
    print(f"true calls per node: median {np.median(calls):.0f}, 90% under {np.quantile(calls, 0.9):.0f}, most {calls.max():.0f} "
          f"(limit {config['maxfun']}); gradient rounds per node: median {np.median([n['rounds'] for n in nodes]):.0f}")
    print(f"gain of the true stage over the true value at the network's optimum (log-objective): median {np.nanmedian(gains):.3f}, "
          f"90% under {np.nanquantile(gains, 0.9):.3f}, most {np.nanmax(gains):.3f}")
    if np.nanmedian(gains) > 5:
        print("NOTE: the true stage gained more than 5 at half the nodes: the network's optima are not near the truth's. "
              "A boost network searched outside its region does this (`region` in the profile YAML); the profiles of such a study mean nothing.")
    if len(times) > 200 and not (out / "complete.json").exists():
        recent = np.sort(times)[-min(len(times), 2000):]
        rate = (len(recent) - 1) / max(recent[-1] - recent[0], 1e-9)             # calls per second, lately
        left = (planned - len(nodes)) * np.median(calls)
        print(f"rate: {3600 * rate:.0f} true calls per hour lately; about {left:.0f} calls left at the median cost per node: "
              f"{left / rate / 3600:.1f} h" + (f" (last call {time.time() - recent[-1]:.0f} s ago)" if time.time() - recent[-1] > 600 else ""))
        if starts + left > config["total_call_budget"]:
            print("NOTE: at this cost the study would pass its total_call_budget before it is done")
    print(f"\n{'parameter':14s} {'nodes':>5s} {'converged':>9s} {'calls':>7s} {'median':>7s} {'largest gain':>13s}")
    for name in parameters:
        mine = [n for n in nodes if n["parameter"] == name]
        if mine:
            print(f"{name:14s} {len(mine):5d} {sum(n['reason'] == 'converged' for n in mine):9d} {sum(n['calls'] for n in mine):7.0f} "
                  f"{np.median([n['calls'] for n in mine]):7.0f} {np.nanmax([n['gain'] for n in mine]):13.3f}")
    if (out / "profiles.csv").exists():
        print(f"\nresults: {out}/profiles.pdf, profiles.csv (surrogate, true at the surrogate's optimum, true optimised), intervals_grid.json")


if __name__ == "__main__":
    main()
