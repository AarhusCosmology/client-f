"""Wall time of the residual-Newton true stage on many workers, from a call journal.

The dependency structure of a finished 1D or 2D residual-Newton true stage is fixed by the
algorithm and recorded in its journal (calls/attempt_*.jsonl): each node is a sequence of
rounds (one call, or a batch of calls whose starts were journaled together), nodes form
chains, and chains wait for their seeds. This replays that structure on W backend workers
with a discrete-event simulation, so its scaling can be studied without the hardware.

Structure (profiles/residual_newton.py, profiles/hybrid_surface.py):
  seeds    one node per profile (1D) or per row (2D), all concurrently;
  chains   two per profile/row, outward from the seed, each node started from its
           predecessor's state;
  anchor   the global best fit, after every chain.

Schedules:
  current     historical global seed barrier (name retained for JSON compatibility);
  dependency  production schedule: a chain waits only for its own seed;
  priority    dependency, and a free worker takes the waiting call whose chain has the
              most work left (critical path first).

Call durations come from the journal itself (`--durations journal`), a constant
(`--durations 1.9`), or are drawn from another journal's (`--durations resample:PATH`);
the master's time between a round's last return and the next round's first start is
taken from the journal (`--think journal`) or neglected (`--think 0`).

python benchmarking/schedule_newton.py STUDY_DIR [--workers 4 16 64 256] \
    [--durations journal|SECONDS|resample:JOURNAL] [--think journal|0] [--json OUT]
"""

import argparse
import heapq
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

NODE = re.compile(r"corrected/(direct|node)/(.+)/(\d+)$")


def read_journal(study):
    events = []
    for path in sorted(Path(study).glob("calls/attempt_*.jsonl")):
        events += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted(events, key=lambda e: e["time"])


def node_rounds(events):
    """{key: [round]}, key (profile, index) or 'anchor'; a round is dict(start, end, seconds)."""
    rounds, open_calls, current = defaultdict(list), {}, {}
    starts = {}
    for e in events:
        stage = e["stage"]
        m = NODE.match(stage)
        if m:
            key = (m.group(2), int(m.group(3)))
        elif stage.startswith("corrected/anchor"):
            key = "anchor"
        else:
            continue  # audit and earlier stages are not part of the true stage
        if e["event"] == "start":
            if not open_calls.get(key):
                rounds[key].append(dict(start=e["time"], end=e["time"], seconds=[]))
                open_calls[key] = 0
            open_calls[key] += 1
            starts[e["call"]] = (key, len(rounds[key]) - 1)
        elif e["event"] in ("return", "exception") and e.get("call") in starts:
            k, r = starts.pop(e["call"])
            rounds[k][r]["seconds"].append(float(e.get("seconds", 0.0)))
            rounds[k][r]["end"] = max(rounds[k][r]["end"], e["time"])
            open_calls[k] -= 1
    return dict(rounds)


def structure(events, chain_length=None):
    """Seeds, chains and the anchor, as lists of nodes; a node is a list of rounds.
    `chain_length` must be the study's own (profiles.residual_newton.chain_plan)."""
    from profiles.residual_newton import chain_plan

    rounds = node_rounds(events)
    first = {k: v[0]["start"] for k, v in rounds.items()}
    profiles = sorted({k[0] for k in rounds if k != "anchor"})
    seeds, chains = [], []
    for p in profiles:
        nodes = sorted(k[1] for k in rounds if k != "anchor" and k[0] == p)
        j0 = min(nodes, key=lambda j: first[(p, j)])  # the profile's first seed starts first
        seeds.append(dict(profile=(p, j0), nodes=[(p, j0)]))
        for side in ([j for j in nodes if j > j0], [j for j in nodes if j < j0][::-1]):
            own, further = chain_plan(side, chain_length)
            if own:
                chains.append(dict(profile=(p, j0), nodes=[(p, j) for j in own]))
            for seed, chain in further:
                seeds.append(dict(profile=(p, seed), nodes=[(p, seed)]))
                if chain:
                    chains.append(dict(profile=(p, seed), nodes=[(p, j) for j in chain]))
    anchor = [dict(profile=None, nodes=["anchor"])] if "anchor" in rounds else []

    def think_times(sequence):
        """Master time before each round: from the previous round's end (in the same chain)."""
        out, last_end = [], None
        for key in sequence:
            for r in rounds[key]:
                out.append(0.0 if last_end is None else max(0.0, r["start"] - last_end))
                last_end = r["end"]
        return out

    for group in (seeds, chains, anchor):
        for c in group:
            c["rounds"] = [len(r["seconds"]) for key in c["nodes"] for r in rounds[key]]
            c["seconds"] = [r["seconds"] for key in c["nodes"] for r in rounds[key]]
            c["think"] = think_times(c["nodes"])
    return dict(seeds=seeds, chains=chains, anchor=anchor, profiles=profiles)


def simulate(plan, workers, schedule="current", think_scale=1.0):
    """Makespan of the true stage on `workers` backend workers (seconds)."""
    tasks = [dict(c, kind=kind) for kind in ("seeds", "chains", "anchor") for c in plan[kind]]
    n_seeds = len(plan["seeds"])
    by_profile = defaultdict(list)
    for i, t in enumerate(tasks):
        if t["kind"] == "chains":
            by_profile[t["profile"]].append(i)
    remaining = [sum(max(s, default=0.0) + think_scale * th for s, th in zip(t["seconds"], t["think"]))
                 for t in tasks]
    position = [0] * len(tasks)      # next round of each task
    outstanding = [0] * len(tasks)   # calls of its current round not yet finished
    done = [False] * len(tasks)
    events, queue, free, now, order = [], [], workers, 0.0, 0
    # Seeds and chains run on a thread pool of `workers` threads (pool.map, in order); the
    # anchor runs on the main thread.
    threads, waiting = workers, []

    def issue(i, t0):
        heapq.heappush(events, (t0 + think_scale * tasks[i]["think"][position[i]], 0, "issue", i))

    def start(i, t0):
        nonlocal threads
        if tasks[i]["kind"] != "anchor":
            if threads == 0:
                waiting.append(i)
                return
            threads -= 1
        if tasks[i]["rounds"]:
            issue(i, t0)
        else:
            finish(i, t0)

    def finish(i, t0):
        nonlocal threads
        done[i] = True
        kind = tasks[i]["kind"]
        if kind != "anchor":
            threads += 1
            if waiting:
                start(waiting.pop(0), t0)
        if kind == "seeds":
            if schedule == "current":
                if all(done[:n_seeds]):
                    for j in (j for j, t in enumerate(tasks) if t["kind"] == "chains"):
                        start(j, t0)
            else:
                for j in by_profile[tasks[i]["profile"]]:
                    start(j, t0)
        if all(done[j] for j, t in enumerate(tasks) if t["kind"] != "anchor"):
            for j, t in enumerate(tasks):
                if t["kind"] == "anchor" and not done[j] and position[j] == 0 and outstanding[j] == 0:
                    start(j, t0)

    for i in range(n_seeds):
        start(i, 0.0)
    if not tasks[:n_seeds]:  # nothing but an anchor
        for j, t in enumerate(tasks):
            start(j, 0.0)
    busy = 0.0
    while events or queue:
        while queue and free:
            _, _, i, seconds = heapq.heappop(queue)
            free -= 1
            busy += seconds
            heapq.heappush(events, (now + seconds, 1, "return", i))
        if not events:
            break
        now, _, kind, i = heapq.heappop(events)
        if kind == "issue":
            calls = tasks[i]["seconds"][position[i]]
            outstanding[i] = len(calls)
            for seconds in calls:
                order += 1
                key = -remaining[i] if schedule == "priority" else 0.0
                heapq.heappush(queue, (key, order, i, seconds))
            if not calls:
                heapq.heappush(events, (now, 1, "return", i))
                outstanding[i] = 1
        else:
            if kind == "return" and tasks[i]["seconds"][position[i]]:
                free += 1
            outstanding[i] -= 1
            if outstanding[i] == 0:
                t = tasks[i]
                remaining[i] -= max(t["seconds"][position[i]], default=0.0) + think_scale * t["think"][position[i]]
                position[i] += 1
                if position[i] < len(t["rounds"]):
                    issue(i, now)
                else:
                    finish(i, now)
    return now, busy


def with_durations(plan, mode, seed=0):
    if mode == "journal":
        return plan
    rng = np.random.default_rng(seed)
    if mode.startswith("resample:"):
        pool = np.array([e["seconds"] for e in read_journal(Path(mode.split(":", 1)[1]))
                         if e["event"] == "return"])
        draw = lambda n: list(rng.choice(pool, n))
    else:
        draw = lambda n: [float(mode)] * n
    out = {k: v for k, v in plan.items()}
    for kind in ("seeds", "chains", "anchor"):
        out[kind] = [dict(c, seconds=[draw(len(s)) for s in c["seconds"]]) for c in plan[kind]]
    return out


def report(study, workers, durations="journal", think="journal", seed=0, chain_length=None):
    study = Path(study)
    events = read_journal(study)
    plan = with_durations(structure(events, chain_length), durations, seed)
    think_scale = 1.0 if think == "journal" else float(think)
    all_rounds = [s for kind in ("seeds", "chains", "anchor") for c in plan[kind] for s in c["seconds"]]
    work = sum(sum(s) for s in all_rounds)
    critical = {name: simulate(plan, 10**9, name, think_scale)[0] for name in ("current", "dependency")}
    rows = []
    for w in workers:
        row = dict(workers=w)
        for name in ("current", "dependency", "priority"):
            t, _ = simulate(plan, w, name, think_scale)
            row[name] = t
            row[name + "_efficiency"] = work / (w * t)
        rows.append(row)
    summary = dict(study=str(study), durations=durations, think=think,
                   profiles=len(plan["profiles"]), seeds=len(plan["seeds"]), chains=len(plan["chains"]),
                   calls=sum(len(s) for s in all_rounds), rounds=len(all_rounds),
                   work_seconds=work, critical_path=critical, rows=rows)
    return summary


def study_chain_length(study):
    """The study's own chain_length, from its manifest: the plan it ran depends on it."""
    path = Path(study) / "manifest.json"
    if not path.exists():
        print(f"WARNING: no {path}: replayed with one chain per side; give --chain-length if the study had one")
        return None
    return json.loads(path.read_text())["configuration"].get("chain_length")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("study", type=Path)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024])
    parser.add_argument("--durations", default="journal")
    parser.add_argument("--think", default="journal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--chain-length", type=int, help="the study's chain_length; default: its manifest's")
    args = parser.parse_args(argv)
    chain_length = study_chain_length(args.study) if args.chain_length is None else args.chain_length
    s = report(args.study, args.workers, args.durations, args.think, args.seed, chain_length)
    print(f"{s['study']}\n  {s['profiles']} profiles/rows, {s['chains']} chains, {s['calls']} calls in "
          f"{s['rounds']} rounds, {s['work_seconds']:.0f} s of backend work "
          f"(durations {s['durations']}, think {s['think']}; {s['seeds']} seeds)")
    print(f"  critical path: {s['critical_path']['current']:.0f} s with historical seed barrier, "
          f"{s['critical_path']['dependency']:.0f} s with production dependencies")
    print(f"  {'workers':>7}  {'old seed barrier':>18}  {'production':>18}  {'priority model':>18}")
    for r in s["rows"]:
        cells = [f"{r[n]:8.0f} s {100 * r[n + '_efficiency']:5.1f}%" for n in ("current", "dependency", "priority")]
        print(f"  {r['workers']:>7}  " + "  ".join(f"{c:>18}" for c in cells))
    if args.json:
        args.json.write_text(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
