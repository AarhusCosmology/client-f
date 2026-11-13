"""Make the kit's probe inputs for omega up to 200 (2 November 2026): runs/axion_pp_omega200 from runs/axion_pp_high and
runs/axion_wc_omega200 from runs/axion_wc_combined (the H0WC final, fast likelihood), with the converged CLASS++
settings of this folder's ladder, the omega prior widened to [1, 200] and a training box omega [55, 200].

    python make_omega200.py KIT          (KIT: hpc_client; writes runs/axion_{pp,wc}_omega200, overwriting)
"""
import json
import re
import shutil
import sys
from pathlib import Path

kit = Path(sys.argv[1])
here = Path(__file__).resolve().parent
LADDER = "results/20261102_high_omega_ladder of the laptop work folder"
HEADER = """# Probe input for omega up to 200 (2 November 2026; made by make_omega200.py of {ladder}) from runs/{src}:
# the same data, model and parameters, with
#   - the omega prior widened to [1, 200] and a training box omega [55, 200] (domain_bounds.json);
#   - CLASS++'s sampling of the high-omega ladder, converged to omega 200 (oscillation signal at A_osc 0.02 within a
#     noise-weighted chi^2 of {err} of l_logstep 1.005, l_linstep 2, q 0.025/22, 2000 points per decade):
#     l_logstep 1.013, l_linstep 8, q_linstep_trans 0.15, q_logstep_trans 90, accurate lensing, and a primordial table
#     of 1000 points per decade (custom7, read by this folder's generate_Pk.py). The runs' settings are off by
#     {off} at omega 200.
# For the probe only so far (smoke/axion_probe.py): no network is trained on it.
#
"""


def edit(text, old, new, count=1):
    if text.count(old) != count:
        raise SystemExit(f"expected {count} of {old!r}")
    return text.replace(old, new)


def make(src, dst, err, off, swaps):
    s, d = kit / "runs" / src, kit / "runs" / dst
    d.mkdir(exist_ok=True)
    text = (s / "cobaya.yaml").read_text()
    for old, new in swaps:
        text = edit(text, old, new)
    text = re.sub(r"(\n  omega:\n    prior:\n      min: 1\.0\n      max: )60\.0", r"\g<1>200.0", text)
    assert "\n  omega:\n    prior:\n      min: 1.0\n      max: 200.0" in text
    (d / "cobaya.yaml").write_text(HEADER.format(ladder=LADDER, src=src, err=err, off=off) + text)
    box = json.loads((s / "domain_bounds.json").read_text())
    box["omega"] = [55.0, 200.0]
    (d / "domain_bounds.json").write_text(json.dumps(box, indent=1) + "\n")
    cfg = (s / "client.yaml").read_text()
    assert f"input: runs/{src}/cobaya.yaml" in cfg
    cfg = cfg.replace(f"runs/{src}/cobaya.yaml", f"runs/{dst}/cobaya.yaml")      # also in its comments
    cfg = re.sub(r"runs/axion_(pp|wc)_high/generate_Pk\.py", f"runs/{dst}/generate_Pk.py", cfg)
    (d / "client.yaml").write_text(f"# runs/{src}/client.yaml with this folder's inputs: for smoke/axion_probe.py "
                                   f"(2 November 2026), no network yet.\n" + cfg)
    shutil.copy(s / "job.env", d / "job.env")
    shutil.copy(here / "generate_Pk_kpd.py", d / "generate_Pk.py")
    print("written", d)


Q = ("      q_linstep_trans: 0.15      # high-omega ladder (2 November 2026): CLASS++'s 0.45, three times denser\n"
     "      q_logstep_trans: 90        # CLASS++'s 400\n")
make("axion_pp_high", "axion_pp_omega200", "0.010", "0.35 (a signal of 1.5, so by about half)", [
    ("      command: python3 runs/axion_pp_high/generate_Pk.py\n",
     "      command: python3 runs/axion_pp_omega200/generate_Pk.py\n"
     "      custom7: 1000              # points per decade of the primordial table (generate_Pk.py; 200 in the runs)\n"),
    ("      l_logstep: 1.026           # CLASS++'s default 1.12: the C_l's multipoles, denser for the oscillation (see the header)\n",
     "      l_logstep: 1.013           # high-omega ladder: the runs' 1.026, CLASS++'s default 1.12\n"),
    ("      l_linstep: 8               # CLASS++'s default 50 (40 before 0a57bbc6)\n",
     "      l_linstep: 8               # CLASS++'s default 50 (40 before 0a57bbc6)\n" + Q
     + "      accurate_lensing: 1        # as the H0WC runs: the fast mode is off by 0.01-0.02 at omega 160-200\n"),
])
make("axion_wc_combined", "axion_wc_omega200", "0.006", "2.0 (a signal of 2.0, so wholly)", [
    ("      command: python3 runs/axion_wc_high/generate_Pk.py\n",
     "      command: python3 runs/axion_wc_omega200/generate_Pk.py\n"
     "      custom7: 1000              # points per decade of the primordial table (generate_Pk.py; 200 in the runs)\n"),
    ("      l_logstep: 1.026           # logarithmic l sampling to l ~ 770 (4 points per period at omega 60), then every 20:\n",
     "      l_logstep: 1.013           # high-omega ladder: the runs' 1.026 with l_linstep 20, off at omega >~ 100\n"),
    ("      l_linstep: 20              # the preset's step; as accurate for the oscillation as the runs' 1.026/8\n",
     "      l_linstep: 8               # the runs' 20 (the ACT preset's)\n" + Q),
])
