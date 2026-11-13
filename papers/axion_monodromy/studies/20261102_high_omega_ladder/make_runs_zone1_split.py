"""Zone 1 of both data sets split at omega = 37.5 (6 November 2026; Steen), as PDP+'s zone 2 was split at 88. PDP+'s
axion_pp_zone1 (omega [2.5, 63]) ended at a true-check rms of 0.25 while its validation loss kept falling (2e-3 to 2e-5),
flat at 0.21-0.29 over its last five boosts; the other PDP+ parts ended at 0.04-0.09, and the final error grows with the
zone's width (26.8: 0.04, 30.2: 0.05, 46.5: 0.07, 47.5: 0.09, 60.5: 0.25). H0WC's axion_wc_zone1 (omega [5.5, 63]) was on
the plateau of the old H0WC high part after six boosts (validation loss 1.4-1.6e-3 in all twelve members, true check
about 1.3); the old part broke through only after eight boosts and needed fifteen to reach 0.10. Zone 2's halves ended
at 0.049 and 0.043.

omega = 37.5 is quiet in both: on the PDP+ probe scan the largest chi^2 gain within +-1.75 of it is 0.9 (its peaks at 30.5
and 51.5 lie well inside the halves); H0WC has no probe scan below 55, and the best H0WC zone-1 training point within +-1.75
of it lies 3.4 below that zone's best (omega 31.2). The band is [36.75, 38.25], each box 1 beyond its bands as before:
  axion_pp_zone1a [2.5, 39.25], axion_pp_zone1b [35.75, 63]; axion_wc_zone1a [5.5, 39.25], axion_wc_zone1b [35.75, 63].
For each: cobaya.yaml and job.env byte-identical copies of the high part's (runs/axion_{pp,wc}_high200), client.yaml's
settings unchanged, domain_bounds.json with the zone's omega box. axion_pp_zone1 and axion_wc_zone1 are left as they are.

    python make_runs_zone1_split.py KIT
"""
import json
import sys
from pathlib import Path

kit = Path(sys.argv[1])
DATA = {
    "pp": dict(high="axion_pp_high200", box=[2.5, 200.0], low="axion_pp_low200", label="PDP+",
               zones={"axion_pp_zone1a": ([2.5, 39.25], "21.5 (chi^2 gain 4.8 at the reference point), 18.25 (4.6), 30.5 (4.5)"),
                      "axion_pp_zone1b": ([35.75, 63.0], "51.5 (chi^2 gain 10.2 at the reference point)")},
               rest=["axion_pp_zone2a", "axion_pp_zone2b", "axion_pp_zone3", "axion_pp_zone4"],
               bands=["2.6:3.0", "36.75:38.25", "60.5:62.0", "87.25:88.75", "110.5:112.0", "153.5:155.0"]),
    "wc": dict(high="axion_wc_high200", box=[5.5, 200.0], low="axion_wc_low200", label="H0WC",
               zones={"axion_wc_zone1a": ([5.5, 39.25], "below 55 no probe; the best training value of axion_wc_zone1 at omega 31.2"),
                      "axion_wc_zone1b": ([35.75, 63.0], "55.5 (chi^2 gain 4.1 at the reference point), 57.5 (3.9); below 55 no probe")},
               rest=["axion_wc_zone2a", "axion_wc_zone2b", "axion_wc_zone3", "axion_wc_zone4a", "axion_wc_zone4b"],
               bands=["6.0:6.1", "36.75:38.25", "60.5:62.0", "87.25:88.75", "110.5:112.0", "153.5:155.0", "173.75:175.25"]),
}
for key, D in DATA.items():
    high = D["high"]
    h = kit / "runs" / high
    cfg = (h / "client.yaml").read_text()
    settings = cfg[cfg.index("# Settings:"):]
    assert settings.count(f"input: runs/{high}/cobaya.yaml") == 1
    assert f"- runs/{high}/generate_Pk.py" in settings
    # the settings of the zone being replaced are the high part's (only its input differs)
    old = (kit / "runs" / f"axion_{key}_zone1" / "client.yaml").read_text()
    assert old[old.index("# Settings:"):] == settings.replace(f"input: runs/{high}/cobaya.yaml", f"input: runs/axion_{key}_zone1/cobaya.yaml")
    parts = [D["low"]] + list(D["zones"]) + D["rest"]
    first = cfg[cfg.index("# Axion monodromy with running"):cfg.index("# The HIGH part")]
    assert first.count("identical in both parts") == 1
    first = first.replace("identical in both parts", f"identical in all {len(parts)} parts")
    cut6 = cfg[cfg.index("# Cut at omega ~ 6"):cfg.index("# A low part made from")] if key == "wc" else ""
    part_args = " \\\n#       ".join(f"--part results/<time>_{p}" for p in parts)
    band_args = " ".join(f"--band {b}" for b in D["bands"])
    for run, (omega, peaks) in D["zones"].items():
        d = kit / "runs" / run
        d.mkdir(exist_ok=True)
        (d / "cobaya.yaml").write_bytes((h / "cobaya.yaml").read_bytes())
        (d / "job.env").write_bytes((h / "job.env").read_bytes())
        box = json.loads((h / "domain_bounds.json").read_text())
        assert box["omega"] == D["box"]
        box["omega"] = omega
        (d / "domain_bounds.json").write_text(json.dumps(box, indent=1) + "\n")
        name = run.rsplit("zone", 1)[1]
        head = (f"# runs/{high}/client.yaml (its settings unchanged) for zone {name} of the {D['label']} high part: omega [{omega[0]:g}, {omega[1]:g}],\n"
                "# half of zone 1 split at omega = 37.5 (6 November 2026, make_runs_zone1_split.py in\n"
                "# results/20261102_high_omega_ladder): this folder's inputs; training.members 12 with member_pilot 100 act on the\n"
                "# networks trained on boost points only.\n"
                f"# On two nodes: sbatch --nodes=2 --ntasks=96 --time=5-00:00:00 submit_client.sh {run}\n")
        block = (f"# Zone {name} of a domain split in omega: training box omega [{omega[0]:g}, {omega[1]:g}] (domain_bounds.json), prior\n"
                 f"# omega U[1, 200]; cobaya.yaml is a byte-identical copy of runs/{high}'s. The largest peaks in it:\n"
                 f"# {peaks}.\n" + (cut6 if run.endswith("zone1a") else "") +
                 "# With the low part and the other zones it is combined by\n"
                 f"#   python client.py combine OUTPUT --parameter omega {part_args} \\\n"
                 f"#       {band_args}\n"
                 "#\n"
                 f"#   sbatch --nodes=2 --ntasks=96 --time=5-00:00:00 submit_client.sh {run}\n"
                 "#\n")
        (d / "client.yaml").write_text(head + first + block + settings.replace(f"input: runs/{high}/cobaya.yaml", f"input: runs/{run}/cobaya.yaml"))
        print(run, "omega box", omega)
    boxes = [json.loads((kit / "runs" / p / "domain_bounds.json").read_text())["omega"] for p in parts]
    for (lo_box, hi_box), b in zip(zip(boxes, boxes[1:]), D["bands"]):
        lo, hi = map(float, b.split(":"))
        assert lo_box[0] < lo < hi <= lo_box[1] and hi_box[0] <= lo < hi < hi_box[1], (b, lo_box, hi_box)
    print(D["label"], "bands inside both adjacent boxes:", list(zip(parts, boxes)), D["bands"])
