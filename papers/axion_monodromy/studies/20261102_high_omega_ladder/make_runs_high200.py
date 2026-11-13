"""The whole high-omega parts again, to omega 200, with several trainings per network (4 November 2026; Steen: redo
the entire high part with the new code, not only from 55, and combine with the low parts at the end).

From runs/axion_{pp,wc}_omega200 (the converged CLASS++ sampling of this ladder; PDP+ from axion_pp_high, H0WC from the
fast likelihood of axion_wc_combined):
  - the omega prior [1, 200] (the old parts' was [1, 60]); training boxes omega [2.5, 200] (PDP+) and [5.5, 200] (H0WC),
    the lower edges of the old high parts, so the old low parts' boxes [1, 3.1] and [1, 6.1] still overlap them;
  - the phi windows of the old parts (phi0 +0.4224 PDP+, +0.4782 H0WC), not the omega-200 probes' re-centred ones, so
    that the files differ from what a low part needs only in the training box;
  - client.yaml: the old high parts' training settings, plus training.members 12 and training.member_pilot 100.

    python make_runs_high200.py KIT

Afterwards, by hand: client.yaml's lines on the partner part and the command were rewritten for the new
boxes (the files in the kit are the record).
"""
import json
import re
import sys
from pathlib import Path

kit = Path(sys.argv[1])
RUNS = {  # new run: (source, old high part, lower omega edge of the training box)
    "axion_pp_high200": ("axion_pp_omega200", "axion_pp_high", 2.5),
    "axion_wc_high200": ("axion_wc_omega200", "axion_wc_high", 5.5),
}
for run, (src, old, lower) in RUNS.items():
    s, o, d = kit / "runs" / src, kit / "runs" / old, kit / "runs" / run
    d.mkdir(exist_ok=True)
    old_box = json.loads((o / "domain_bounds.json").read_text())
    lo, hi = old_box["phi"]
    old_text = (o / "cobaya.yaml").read_text()
    phi_ref = re.search(r"\n  phi:\n    prior:\n      min: \S+\n      max: \S+\n    ref: (\S+)", old_text).group(1)

    text = (s / "cobaya.yaml").read_text()
    note = re.search(r"# The training run of omega \[55, 200\] \(3 November 2026; make_runs200\.py\).*?"
                     r"evidences weighted by their prior ranges\.\n", text, re.S)
    assert note, run
    text = text.replace(note.group(0),
        f"# The high part of omega [{lower:g}, 200] (4 November 2026; make_runs_high200.py of the same folder): the omega prior\n"
        f"# [1, 200], the training box omega [{lower:g}, 200] (domain_bounds.json), the phi window of the earlier parts\n"
        f"# ({lo:.5f} to {hi:.5f}, phi0 {float(phi_ref):+.4f}). A low part made from this file, with only its training box\n"
        f"# changed, is combined with it by client.py combine. The earlier low part (runs/{old.replace('high', 'low')}) has\n"
        "# the older CLASS++ sampling and the omega prior [1, 60], and client.py combine refuses it as it is.\n")
    prior_line = "#   - the omega prior [55, 200], which is also the training box (domain_bounds.json);\n"
    assert text.count(prior_line) == 1, run
    text = text.replace(prior_line, f"#   - the omega prior [1, 200] and the training box omega [{lower:g}, 200] (domain_bounds.json);\n")
    text, n = re.subn(r"(\n  omega:\n    prior:\n      min: )55\.0(\n      max: 200\.0\n    ref: )127\.5", r"\g<1>1.0\g<2>25.0", text)
    assert n == 1, run
    text, n = re.subn(r"(\n  phi:\n    prior:\n      min: )\S+(\n      max: )\S+(\n    ref: )\S+",
                      lambda m: f"{m.group(1)}{lo:.11f}{m.group(2)}{hi:.11f}{m.group(3)}{phi_ref}", text)
    assert n == 1, run
    assert text.count(f"runs/{src}/generate_Pk.py") == 1, run
    text = text.replace(f"runs/{src}/generate_Pk.py", f"runs/{run}/generate_Pk.py")
    (d / "cobaya.yaml").write_text(text)

    box = json.loads((s / "domain_bounds.json").read_text())
    assert box["omega"] == [55.0, 200.0], run
    box["omega"] = [lower, 200.0]
    box["phi"] = [lo, hi]
    (d / "domain_bounds.json").write_text(json.dumps(box, indent=1) + "\n")
    (d / "generate_Pk.py").write_text((s / "generate_Pk.py").read_text())
    (d / "job.env").write_text((s / "job.env").read_text())

    cfg = (o / "client.yaml").read_text()
    for name in ("cobaya.yaml", "generate_Pk.py"):
        cfg = cfg.replace(f"runs/{old}/{name}", f"runs/{run}/{name}")
    assert f"input: runs/{run}/cobaya.yaml" in cfg and f"runs/{run}/generate_Pk.py" in cfg, run
    cfg = cfg.replace("prior omega U[1, 60]", "prior omega U[1, 200]")
    training = "\ntraining:\n"
    assert cfg.count(training) == 1, run
    cfg = cfg.replace(training, training +
        "  members: 12              # in the boost phase, several trainings per network (CLiENT docs/members.md): rank 0 warm,\n"
        "  member_pilot: 100        # eleven from random weights, kept by 1200 true evaluations at their own posterior points\n")
    header = (f"# runs/{old}/client.yaml (the training settings of the earlier high part) for the high part of omega [{lower:g}, 200]\n"
              f"# (4 November 2026, make_runs_high200.py): this folder's inputs, and training.members 12 with member_pilot 100, which act on the networks trained on\n"
              "# boost points only (CLiENT dev-sth1509 from the boost-only fix of 4 November 2026; the tempered phase trains once).\n"
              "# On four nodes: sbatch --nodes=4 --ntasks=192 --time=5-00:00:00 submit_client.sh " + run + "\n")
    (d / "client.yaml").write_text(header + cfg)
    print(run, "from", src, "and", old, ": omega box", box["omega"], "phi", [round(lo, 5), round(hi, 5)], "ref", phi_ref)
