"""Zone 2 of the PDP+ zones split again at omega = 88 (5 November 2026; Steen). After six boosts axion_pp_zone2
(omega [59.5, 113]) was stuck at a scatter of about 1.0 at its boost points, with warm networks 26 to 47 too high at
their own pilot points, as axion_pp_high200 was; zones 1, 3 and 4 had fallen to 0.29, 0.15 and 0.14. Zone 2 holds PDP+'s
pair of comparable peaks at 100.75 (chi^2 gain 6.5 at the reference point) and 104.75 (5.8) and smaller ones from 64 to
98. omega = 88 is the cleanest cut of the probe scans (omega_cuts.py): the largest gain in the band [87.25, 88.75] is
0.55 (PDP+) and 0.81 (H0WC), and in the overlap [86.25, 89.75] 0.80 and 0.82.

For runs/axion_pp_high200, as make_runs_zones.py (whose four runs are left as they are), runs/axion_pp_zone2a
(omega [59.5, 89.75]: peaks 64.75 (3.9), 67.5 (3.3)) and runs/axion_pp_zone2b (omega [86.25, 113]: 100.75 (6.5),
104.75 (5.8), 97.75 (3.2)): cobaya.yaml and job.env byte-identical, client.yaml's settings unchanged, each box 1 beyond
its bands. The six parts with the low part are combined by
    python client.py combine OUTPUT --parameter omega --part results/<time>_axion_pp_low200 \
        --part results/<time>_axion_pp_zone1 --part results/<time>_axion_pp_zone2a --part results/<time>_axion_pp_zone2b \
        --part results/<time>_axion_pp_zone3 --part results/<time>_axion_pp_zone4 \
        --band 2.6:3.0 --band 60.5:62.0 --band 87.25:88.75 --band 110.5:112.0 --band 153.5:155.0

    python make_runs_zone2_split.py KIT
"""
import json
import sys
from pathlib import Path

kit = Path(sys.argv[1])
HIGH = "axion_pp_high200"
PARTS = ["axion_pp_low200", "axion_pp_zone1", "axion_pp_zone2a", "axion_pp_zone2b", "axion_pp_zone3", "axion_pp_zone4"]
BANDS = ["2.6:3.0", "60.5:62.0", "87.25:88.75", "110.5:112.0", "153.5:155.0"]
ZONES = {  # run: (training box in omega, the probe's largest PDP+ peaks inside it)
    "axion_pp_zone2a": ([59.5, 89.75], "64.75 (chi^2 gain 3.9 at the reference point), 67.5 (3.3)"),
    "axion_pp_zone2b": ([86.25, 113.0], "100.75 (6.5), 104.75 (5.8), 97.75 (3.2)"),
}
h = kit / "runs" / HIGH
cfg = (h / "client.yaml").read_text()
settings = cfg[cfg.index("# Settings:"):]
assert settings.count(f"input: runs/{HIGH}/cobaya.yaml") == 1
assert f"- runs/{HIGH}/generate_Pk.py" in settings
first = cfg[cfg.index("# Axion monodromy with running"):cfg.index("# The HIGH part")]
assert first.count("identical in both parts") == 1
first = first.replace("identical in both parts", "identical in all six parts")
parts = " \\\n#       ".join(f"--part results/<time>_{p}" for p in PARTS)
bands = " ".join(f"--band {b}" for b in BANDS)
for run, (omega, peaks) in ZONES.items():
    d = kit / "runs" / run
    d.mkdir(exist_ok=True)
    (d / "cobaya.yaml").write_bytes((h / "cobaya.yaml").read_bytes())
    (d / "job.env").write_bytes((h / "job.env").read_bytes())
    box = json.loads((h / "domain_bounds.json").read_text())
    assert box["omega"] == [2.5, 200.0]
    box["omega"] = omega
    (d / "domain_bounds.json").write_text(json.dumps(box, indent=1) + "\n")
    half = run[-2:]
    head = (f"# runs/{HIGH}/client.yaml (its settings unchanged) for zone {half} of the PDP+ high part: omega [{omega[0]:g}, {omega[1]:g}],\n"
            "# half of zone 2 split again at omega = 88 (5 November 2026, make_runs_zone2_split.py in\n"
            "# results/20261102_high_omega_ladder): this folder's inputs; training.members 12 with member_pilot 100 act on the\n"
            "# networks trained on boost points only.\n"
            f"# On two nodes: sbatch --nodes=2 --ntasks=96 --time=5-00:00:00 submit_client.sh {run}\n")
    block = (f"# Zone {half} of a domain split in omega: training box omega [{omega[0]:g}, {omega[1]:g}] (domain_bounds.json), prior\n"
             f"# omega U[1, 200]; cobaya.yaml is a byte-identical copy of runs/{HIGH}'s. The probe's largest peaks in it:\n"
             f"# {peaks}. With the low part and the other zones it is combined by\n"
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
print("bands inside both adjacent boxes:", list(zip(PARTS, boxes)), BANDS)
