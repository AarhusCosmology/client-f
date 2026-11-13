"""The PDP+ high part split into four omega zones (4 November 2026; Steen: pp_high200 has not localised any omega peak
after six boosts, so run the range in zones, PDP+ first, H0WC later if needed).

For runs/axion_pp_high200 (make_runs_high200.py), runs/axion_pp_zone1 .. axion_pp_zone4:
  - cobaya.yaml and job.env: byte-identical copies of the high part's (prior omega U[1, 200], the converged CLASS++
    sampling, the earlier phi window; every part runs the high part's generate_Pk.py), as client.py combine requires;
  - domain_bounds.json: the high part's, with omega cut to the zone's training box (ZONES below);
  - client.yaml: the high part's settings unchanged (members 12, member_pilot 100, ten boosts), its own input.
The cuts lie where the probe scans over omega (omega_cuts.py) show no peak: within +-1 of each band PDP+'s gain over
A_osc = 0 is at most 0.8 (61), 1.6 (111) and 1.1 (154), and so is H0WC's (1.7, 0.7, 0.6), for the same cuts later.
Each box reaches 1 beyond its bands (the existing low part, axion_pp_low200, overlaps zone 1 as it overlapped the high
part). All five parts are combined with
    python client.py combine OUTPUT --parameter omega --part results/<time>_axion_pp_low200 \
        --part results/<time>_axion_pp_zone1 --part results/<time>_axion_pp_zone2 \
        --part results/<time>_axion_pp_zone3 --part results/<time>_axion_pp_zone4 \
        --band 2.6:3.0 --band 60.5:62.0 --band 110.5:112.0 --band 153.5:155.0

    python make_runs_zones.py KIT
"""
import json
import sys
from pathlib import Path

kit = Path(sys.argv[1])
HIGH = "axion_pp_high200"
BANDS = ["2.6:3.0", "60.5:62.0", "110.5:112.0", "153.5:155.0"]        # low part | zone 1 | zone 2 | zone 3 | zone 4
ZONES = {  # run: (training box in omega, the probe's largest PDP+ peaks inside it)
    "axion_pp_zone1": ([2.5, 63.0], "51.5 (chi^2 gain 10.2 at the reference point), 21.5 (4.8), 18.25 (4.6), 30.5 (4.5)"),
    "axion_pp_zone2": ([59.5, 113.0], "100.75 (6.5), 104.75 (5.8), 64.75 (3.9), 67.5 (3.3), 97.75 (3.2)"),
    "axion_pp_zone3": ([109.5, 156.0], "128.25 (4.2), 137.25 (4.1), 131.25 (3.4), 150.5 (3.4); none stands out"),
    "axion_pp_zone4": ([152.5, 200.0], "196.0 (8.6), 179.0 (8.0), 182.25 (4.0)"),
}
h = kit / "runs" / HIGH
cfg = (h / "client.yaml").read_text()
settings = cfg[cfg.index("# Settings:"):]
assert settings.count(f"input: runs/{HIGH}/cobaya.yaml") == 1
assert f"- runs/{HIGH}/generate_Pk.py" in settings          # the high part's script, as the low part
first = cfg[cfg.index("# Axion monodromy with running"):cfg.index("# The HIGH part")]
assert first.count("identical in both parts") == 1
first = first.replace("identical in both parts", "identical in all five parts")
parts = " \\\n#       ".join(["--part results/<time>_axion_pp_low200"] + [f"--part results/<time>_{z}" for z in ZONES])
bands = " ".join(f"--band {b}" for b in BANDS)
for k, (run, (omega, peaks)) in enumerate(ZONES.items(), start=1):
    d = kit / "runs" / run
    d.mkdir(exist_ok=True)
    (d / "cobaya.yaml").write_bytes((h / "cobaya.yaml").read_bytes())
    (d / "job.env").write_bytes((h / "job.env").read_bytes())
    box = json.loads((h / "domain_bounds.json").read_text())
    assert box["omega"] == [2.5, 200.0]
    box["omega"] = omega
    (d / "domain_bounds.json").write_text(json.dumps(box, indent=1) + "\n")
    head = (f"# runs/{HIGH}/client.yaml (its settings unchanged) for zone {k} of four of the PDP+ high part: omega [{omega[0]:g}, {omega[1]:g}]\n"
            "# (4 November 2026, make_runs_zones.py in results/20261102_high_omega_ladder): this folder's inputs; training.members 12\n"
            "# with member_pilot 100 act on the networks trained on boost points only.\n"
            f"# On two nodes: sbatch --nodes=2 --ntasks=96 --time=5-00:00:00 submit_client.sh {run}\n")
    block = (f"# Zone {k} of a domain split in omega: training box omega [{omega[0]:g}, {omega[1]:g}] (domain_bounds.json), prior\n"
             f"# omega U[1, 200]; cobaya.yaml is a byte-identical copy of runs/{HIGH}'s. The probe's largest peaks in it:\n"
             f"# {peaks}. With the low part and the other zones it is combined by\n"
             f"#   python client.py combine OUTPUT --parameter omega {parts} \\\n"
             f"#       {bands}\n"
             "#\n"
             f"#   sbatch --nodes=2 --ntasks=96 --time=5-00:00:00 submit_client.sh {run}\n"
             "#\n")
    text = head + first + block + settings.replace(f"input: runs/{HIGH}/cobaya.yaml", f"input: runs/{run}/cobaya.yaml")
    (d / "client.yaml").write_text(text)
    print(run, "omega box", omega)
# the bands lie inside both adjacent boxes, and the parts cover the range between them
boxes = [json.loads((kit / "runs/axion_pp_low200/domain_bounds.json").read_text())["omega"]] + [z[0] for z in ZONES.values()]
for (lo_box, hi_box), b in zip(zip(boxes, boxes[1:]), BANDS):
    lo, hi = map(float, b.split(":"))
    assert lo_box[0] < lo < hi <= lo_box[1] and hi_box[0] <= lo < hi < hi_box[1], (b, lo_box, hi_box)
print("bands inside both adjacent boxes:", BANDS)
