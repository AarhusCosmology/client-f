"""The members' pilot of a run, scored again from its files: which member each score keeps.

    python hpc/analysis/member_pilot_scores.py results/<timestamp>_<RUN>

For every network with `trained_models/members/members_it_N.json` and `pilot_it_N.csv` (a run with
training.member_pilot), the members' errors at the pilot's points as `training.member_pilot_score`
computes them (`training.members.pilot_errors`: rms, offset, the rms without the offset) and the
member each score keeps with the run's margin and warm start. It reads the files only, so that a run
made with one score can be judged with the other (19 November 2026: the offset-corrected score
became the default; docs/members.md). Prints a table and writes nothing.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.members import PILOT_SCORES, choose_by_pilot, pilot_errors  # noqa: E402


def replay(record, frame, refused=None):
    """{"errors": pilot_errors, "kept": {score: member}, "run_kept": the run's} for one network,
    from its members_it_N.json (`record`) and pilot_it_N.csv (`frame`, a DataFrame); `refused` is
    the run's refused value (RUN/refused_value.json), which counts for no member, as in the run."""
    predictions = {int(c[len("pred_member"):]): frame[c].to_numpy(float)
                   for c in frame.columns if c.startswith("pred_member")}
    errors = pilot_errors(predictions, frame["loglkl"].to_numpy(float), record["best"], record["pilot_depth"],
                          ignore=refused)
    kept = {score: choose_by_pilot({k: v[score] for k, v in errors.items()}, record["warm"], record["margin"])[0]
            for score in PILOT_SCORES}
    return dict(errors=errors, kept=kept, run_kept=record["kept"], score=record.get("pilot_score", "rms"))


def lines(network, result):
    """The table's lines for one network."""
    errors, kept = result["errors"], result["kept"]
    head = (f"network {network}: the run ({result['score']}) kept {result['run_kept']}; rms keeps {kept['rms']}, "
            f"corrected keeps {kept['corrected']}" + ("" if kept["rms"] == kept["corrected"] else "  <- differ"))
    rows = [f"   member {k:2d}: rms {e['rms']:8.3g}  offset {e['offset']:+8.3g}  corrected {e['corrected']:8.3g}"
            for k, e in sorted(errors.items())]
    return [head] + rows


def main(argv=None):
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    args = ap.parse_args(argv)
    members = args.run / "trained_models" / "members"
    from hpc.refused import read_record

    refused = (read_record(args.run / "refused_value.json") or {}).get("value")
    found = sorted((int(re.search(r"_it_(\d+)\.json$", p.name).group(1)), p)
                   for p in members.glob("members_it_*.json"))
    done = 0
    for network, path in found:
        record = json.loads(path.read_text())
        pilot = members / f"pilot_it_{network}.csv"
        if record.get("choice") != "pilot" or not pilot.exists():
            continue
        print("\n".join(lines(network, replay(record, pd.read_csv(pilot), refused))))
        done += 1
    if not done:
        raise SystemExit(f"{args.run}: no network with a members' pilot (members_it_N.json and pilot_it_N.csv)")


if __name__ == "__main__":
    main()
