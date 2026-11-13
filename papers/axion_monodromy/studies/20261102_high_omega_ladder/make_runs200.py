"""Turn the omega <= 200 probe inputs (make_omega200.py) into training runs (3 November 2026): the omega prior equal to the
training box [55, 200], and the phi window [phi0 - pi, phi0 + pi] with the phi0 of the probes over omega 55-200
(jobs 7010679 PDP+ +1.0009, 7010680 H0WC +1.9007). Run after make_omega200.py.

    python make_runs200.py KIT
"""
import json
import math
import re
import sys
from pathlib import Path

kit = Path(sys.argv[1])
PHI0 = {"axion_pp_omega200": (1.0009, "7010679"), "axion_wc_omega200": (1.9007, "7010680")}
for run, (phi0, job) in PHI0.items():
    d = kit / "runs" / run
    lo, hi = phi0 - math.pi, phi0 + math.pi
    text = (d / "cobaya.yaml").read_text()
    old_note = "# For the probe only so far (smoke/axion_probe.py): no network is trained on it.\n"
    assert text.count(old_note) == 1
    text = text.replace(old_note,
        "# The training run of omega [55, 200] (3 November 2026; make_runs200.py): the omega prior is the training box,\n"
        f"# [55, 200], and the phi window [phi0 - pi, phi0 + pi] has the phi0 = {phi0:+.4f} of the probe over omega 55-200 (job {job}).\n"
        "# Its results are joined with those of the existing combined emulator of omega [1, 60] (whose likelihood has the\n"
        "# runs' older CLASS++ settings, so client.py combine does not apply): evidences weighted by their prior ranges.\n")
    text, n = re.subn(r"(\n  omega:\n    prior:\n      min: )1\.0(\n      max: 200\.0\n    ref: )25\.0", r"\g<1>55.0\g<2>127.5", text)
    assert n == 1
    text, n = re.subn(r"(\n  phi:\n    prior:\n      min: )\S+(\n      max: )\S+(\n    ref: )\S+",
                      lambda m: f"{m.group(1)}{lo:.11f}{m.group(2)}{hi:.11f}{m.group(3)}{phi0:.4f}", text)
    assert n == 1
    old_prior = "#   - the omega prior widened to [1, 200] and a training box omega [55, 200] (domain_bounds.json);\n"
    assert text.count(old_prior) == 1
    text = text.replace(old_prior, "#   - the omega prior [55, 200], which is also the training box (domain_bounds.json);\n")
    (d / "cobaya.yaml").write_text(text)
    box = json.loads((d / "domain_bounds.json").read_text())
    assert box["omega"] == [55.0, 200.0]
    box["phi"] = [round(lo, 11), round(hi, 11)]
    (d / "domain_bounds.json").write_text(json.dumps(box, indent=1) + "\n")
    src = "axion_pp_high" if run.startswith("axion_pp") else "axion_wc_high"     # the training settings of the high part
    cfg = (kit / "runs" / src / "client.yaml").read_text()
    for name in ("cobaya.yaml", "generate_Pk.py"):
        cfg = cfg.replace(f"runs/{src}/{name}", f"runs/{run}/{name}")
    assert f"input: runs/{run}/cobaya.yaml" in cfg and f"runs/{run}/generate_Pk.py" in cfg
    (d / "client.yaml").write_text(f"# runs/{src}/client.yaml (the training settings of the high part) for the training run of "
                                   f"omega [55, 200] (3 November 2026):\n# only the input paths changed.\n" + cfg)
    print(run, "phi window", round(lo, 5), round(hi, 5))
