"""The H0WC high part in omega zones (5 November 2026; Steen: axion_wc_high200 is stuck as axion_pp_high200 was, so
split it as PDP+). The PDP+ cuts (make_runs_zones.py, make_runs_zone2_split.py) were chosen quiet in both data sets;
on H0WC's probe scan (7010680) the largest gain in their bands is 1.04 (60.5-62), 0.81 (87.25-88.75), 0.68 (110.5-112)
and 0.66 (153.5-155). One cut more than PDP+: H0WC's top zone [152.5, 200] holds nine peaks of 3.4 to 6.2 (157.75 (6.2),
the comb 165.25, 168, 170.5, 172 (4.4-4.9), 178 (3.4), 193.5, 195.5, 197.5 (3.4-5.0)), the crowded case that kept PDP+'s
zone 2 at a scatter of about 1. omega = 174.5 parts the comb from the group at 193-198: the largest gain in the band
[173.75, 175.25] is 0.75 (PDP+ is quiet there too, 173.25-175).

For runs/axion_wc_high200 (make_runs_high200.py, make_runs_low200.py: balance_floor 0.1), runs/axion_wc_zone1, 2a, 2b,
3, 4a, 4b: cobaya.yaml and job.env byte-identical (prior omega U[1, 200]; every part runs the high part's
generate_Pk.py), client.yaml's settings unchanged, domain_bounds.json with the zone's omega box, each box 1 beyond its
bands; the low part axion_wc_low200 ([1, 6.1], band 6.0:6.1) overlaps zone 1 as it overlapped the high part.

    python make_runs_wc_zones.py KIT
"""
import json
import sys
from pathlib import Path

kit = Path(sys.argv[1])
HIGH, LOW = "axion_wc_high200", "axion_wc_low200"
ZONES = {  # run: (training box in omega, the probe's largest H0WC peaks inside it)
    "axion_wc_zone1": ([5.5, 63.0], "below 55 no probe; the omega < 60 runs' true-profile best fit at omega ~ 31; 55.5 (4.1), 57.5 (3.9)"),
    "axion_wc_zone2a": ([59.5, 89.75], "81.5 (chi^2 gain 5.0 at the reference point), 73 (3.7)"),
    "axion_wc_zone2b": ([86.25, 113.0], "none of 3 or more"),
    "axion_wc_zone3": ([109.5, 156.0], "141.5 (5.6), 146.25 (4.8), 140 (3.9), 116.25 (3.3), 117.75 (3.1)"),
    "axion_wc_zone4a": ([152.5, 176.25], "157.75 (6.2), 168 (4.9), 172 (4.9), 165.25 (4.5), 170.5 (4.4)"),
    "axion_wc_zone4b": ([172.75, 200.0], "195.5 (5.0), 193.5 (4.3), 178 (3.4), 197.5 (3.4)"),
}
BANDS = ["6.0:6.1", "60.5:62.0", "87.25:88.75", "110.5:112.0", "153.5:155.0", "173.75:175.25"]
PARTS = [LOW] + list(ZONES)
h = kit / "runs" / HIGH
cfg = (h / "client.yaml").read_text()
settings = cfg[cfg.index("# Settings:"):]
assert settings.count(f"input: runs/{HIGH}/cobaya.yaml") == 1
assert f"- runs/{HIGH}/generate_Pk.py" in settings and "balance_floor: 0.1" in settings
first = cfg[cfg.index("# Axion monodromy with running"):cfg.index("# The HIGH part")]
assert first.count("identical in both parts") == 1
first = first.replace("identical in both parts", f"identical in all {len(PARTS)} parts")
cut6 = cfg[cfg.index("# Cut at omega ~ 6"):cfg.index("# A low part made from")]
parts = " \\\n#       ".join(f"--part results/<time>_{p}" for p in PARTS)
bands = " ".join(f"--band {b}" for b in BANDS)
for run, (omega, peaks) in ZONES.items():
    d = kit / "runs" / run
    d.mkdir(exist_ok=True)
    (d / "cobaya.yaml").write_bytes((h / "cobaya.yaml").read_bytes())
    (d / "job.env").write_bytes((h / "job.env").read_bytes())
    box = json.loads((h / "domain_bounds.json").read_text())
    assert box["omega"] == [5.5, 200.0]
    box["omega"] = omega
    (d / "domain_bounds.json").write_text(json.dumps(box, indent=1) + "\n")
    name = run.rsplit("zone", 1)[1]
    head = (f"# runs/{HIGH}/client.yaml (its settings unchanged) for zone {name} of the H0WC high part: omega [{omega[0]:g}, {omega[1]:g}]\n"
            "# (5 November 2026, make_runs_wc_zones.py in results/20261102_high_omega_ladder): this folder's inputs;\n"
            "# training.members 12 with member_pilot 100 act on the networks trained on boost points only.\n"
            f"# sbatch --nodes=2 --ntasks=96 --time=5-00:00:00 submit_client.sh {run}   (or --nodes=4 --ntasks=192)\n")
    block = (f"# Zone {name} of a domain split in omega: training box omega [{omega[0]:g}, {omega[1]:g}] (domain_bounds.json), prior\n"
             f"# omega U[1, 200]; cobaya.yaml is a byte-identical copy of runs/{HIGH}'s. The probe's largest peaks in it:\n"
             f"# {peaks}.\n" + (cut6 if run.endswith("zone1") else "") +
             "# With the low part and the other zones it is combined by\n"
             f"#   python client.py combine OUTPUT --parameter omega {parts} \\\n"
             f"#       {bands}\n"
             "#\n"
             f"#   sbatch --nodes=2 --ntasks=96 --time=5-00:00:00 submit_client.sh {run}\n"
             "#\n")
    (d / "client.yaml").write_text(head + first + block + settings.replace(f"input: runs/{HIGH}/cobaya.yaml", f"input: runs/{run}/cobaya.yaml"))
    print(run, "omega box", omega)
boxes = [json.loads((kit / "runs" / p / "domain_bounds.json").read_text())["omega"] for p in PARTS]
for (lo_box, hi_box), b in zip(zip(boxes, boxes[1:]), BANDS):
    lo, hi = map(float, b.split(":"))
    assert lo_box[0] < lo < hi <= lo_box[1] and hi_box[0] <= lo < hi < hi_box[1], (b, lo_box, hi_box)
print("bands inside both adjacent boxes:", BANDS)
