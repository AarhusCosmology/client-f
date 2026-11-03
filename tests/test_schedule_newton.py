"""benchmarking/schedule_newton.py: journal parsing, chain structure and the simulation."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarking.schedule_newton import read_journal, simulate, structure, study_chain_length


class ChainLength(unittest.TestCase):
    def test_it_comes_from_the_study_s_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(study_chain_length(d))
            for value in (2, None):
                (Path(d) / "manifest.json").write_text(json.dumps(dict(configuration=dict(chain_length=value))))
                self.assertEqual(study_chain_length(d), value)


def journal(directory, nodes, anchor_rounds=((1.0,),)):
    """A residual-Newton journal for one profile "p": `nodes` maps a grid index to its
    rounds (each a tuple of call seconds), written in the order given, each round
    starting as the previous one returns (no master time)."""
    lines, t, call = [], 0.0, 0

    def rounds(stage, spec):
        nonlocal t, call
        for seconds in spec:
            numbers = []
            for s in seconds:
                call += 1
                numbers.append((call, s))
                lines.append(dict(time=t, stage=stage, event="start", call=call, x=[0.0]))
            for number, s in numbers:
                lines.append(dict(time=t + s, stage=stage, event="return", call=number, seconds=s, value=0.0))
            t += max(seconds)

    for j, spec in nodes.items():
        rounds(f"corrected/direct/p/{j}", spec[:1])
        rounds(f"corrected/node/p/{j}", spec[1:])
    rounds("corrected/anchor", anchor_rounds)
    path = Path(directory) / "calls"
    path.mkdir()
    (path / "attempt_0000.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines))
    return read_journal(directory)


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # Seed j0 = 2 is written first; then its chains 3, 4 and 1, 0. Each node: a direct
        # call, a batch of three probes and one step.
        spec = [(1.0,), (1.0, 1.0, 2.0), (1.0,)]
        self.events = journal(self.tmp.name, {2: spec, 3: spec, 4: spec, 1: spec, 0: spec})

    def tearDown(self):
        self.tmp.cleanup()

    def test_structure(self):
        plan = structure(self.events)
        self.assertEqual([s["nodes"] for s in plan["seeds"]], [[("p", 2)]])
        self.assertEqual([c["nodes"] for c in plan["chains"]], [[("p", 3), ("p", 4)], [("p", 1), ("p", 0)]])
        self.assertEqual(plan["seeds"][0]["rounds"], [1, 3, 1])
        plan = structure(self.events, chain_length=0)
        self.assertEqual(sorted(s["nodes"][0][1] for s in plan["seeds"]), [0, 1, 2, 3, 4])
        self.assertEqual(plan["chains"], [])

    def test_simulation_bounds(self):
        plan = structure(self.events)
        work = 5 * (1 + 4 + 1) + 1  # every call, and the anchor's
        one, busy = simulate(plan, 1, "current", 0.0)
        self.assertAlmostEqual(one, work)
        self.assertAlmostEqual(busy, work)
        # With a worker per call: the seed, then the longer of its chains, then the anchor;
        # a node takes 1 + 2 + 1 seconds.
        self.assertAlmostEqual(simulate(plan, 100, "current", 0.0)[0], 4 + 2 * 4 + 1)
        # Every node its own seed: one node, then the anchor.
        self.assertAlmostEqual(simulate(structure(self.events, 0), 100, "dependency", 0.0)[0], 4 + 1)
        times = [simulate(plan, w, "dependency", 0.0)[0] for w in (1, 2, 3, 4, 8)]
        self.assertEqual(times, sorted(times, reverse=True))


if __name__ == "__main__":
    unittest.main()
