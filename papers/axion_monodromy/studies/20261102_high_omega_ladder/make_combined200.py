"""The run folders of the two combined emulators for omega <= 200 (6 November 2026): runs/axion_pp_combined200 (the low
part and the six PDP+ zones) and runs/axion_wc_combined200 (the low part and the seven H0WC zones). They are what the
kit's analysis jobs read for a combined emulator (analysis/queue_axion_combined200.sh): job.env and cobaya.yaml,
byte-identical copies of the parts' (checked to be the same in every part), domain_bounds.json (the union of the parts'
boxes, omega [1, 200]) and client.yaml, the low part's settings with a header, kept for reference only (the loop refuses
a combined run). As runs/axion_pp_combined and runs/axion_wc_combined of the omega <= 60 study.

    python make_combined200.py KIT
"""
import json
import sys
from pathlib import Path

kit = Path(sys.argv[1])
DATA = {  # name: (dataset, parts in increasing omega, bands)
    # zone 1 of both split at omega = 37.5 on 6 November (make_runs_zone1_split.py): PDP+ 7 parts, H0WC 8. The PDP+
    # combine of 6 November (results/20261106_090834_axion_pp_combined200) had the six parts with zone 1 whole.
    "axion_pp_combined200": ("PDP+", ["axion_pp_low200", "axion_pp_zone1a", "axion_pp_zone1b", "axion_pp_zone2a",
                                      "axion_pp_zone2b", "axion_pp_zone3", "axion_pp_zone4"],
                             ["2.6:3.0", "36.75:38.25", "60.5:62.0", "87.25:88.75", "110.5:112.0", "153.5:155.0"]),
    "axion_wc_combined200": ("H0WC", ["axion_wc_low200", "axion_wc_zone1a", "axion_wc_zone1b", "axion_wc_zone2a",
                                      "axion_wc_zone2b", "axion_wc_zone3", "axion_wc_zone4a", "axion_wc_zone4b"],
                             ["6.0:6.1", "36.75:38.25", "60.5:62.0", "87.25:88.75", "110.5:112.0", "153.5:155.0",
                              "173.75:175.25"]),
}
for name, (label, parts, bands) in DATA.items():
    src = [kit / "runs" / p for p in parts]
    for f in ("cobaya.yaml", "job.env"):
        assert all((s / f).read_bytes() == (src[0] / f).read_bytes() for s in src), (name, f)
    boxes = [json.loads((s / "domain_bounds.json").read_text()) for s in src]
    for b in boxes[1:]:
        assert {k: v for k, v in b.items() if k != "omega"} == {k: v for k, v in boxes[0].items() if k != "omega"}, name
    omega = [b["omega"] for b in boxes]
    for (below, above), band in zip(zip(omega, omega[1:]), bands):     # each band inside both neighbouring boxes
        lo, hi = map(float, band.split(":"))
        assert above[0] <= lo < hi <= below[1], (name, band, below, above)
    union = dict(boxes[0], omega=[min(o[0] for o in omega), max(o[1] for o in omega)])
    assert union["omega"] == [1.0, 200.0], union["omega"]
    d = kit / "runs" / name
    d.mkdir(exist_ok=True)
    for f in ("cobaya.yaml", "job.env"):
        (d / f).write_bytes((src[0] / f).read_bytes())
    (d / "domain_bounds.json").write_text(json.dumps(union, indent=1) + "\n")
    cfg = (src[0] / "client.yaml").read_text()
    first = cfg[cfg.index("# Axion monodromy with running"):]
    part_args = " \\\n#       ".join(f"--part results/<time>_{p}:S+10" for p in parts)
    head = (f"# runs/{name} (6 November 2026, make_combined200.py in results/20261102_high_omega_ladder): the folder the\n"
            f"# kit's analysis jobs read for the combined emulator of {label} for omega <= 200, of the {len(parts)} parts\n"
            f"# {', '.join(parts)} at their final networks, S + 10 with each part's own S\n"
            f"# (analysis/queue_axion_combined200.sh): its job.env for the true check, cobaya.yaml (a byte-identical copy of the\n"
            "# parts') and domain_bounds.json (the union of the parts' boxes, omega [1, 200]). The parts are combined with\n"
            f"#   python client.py combine OUTPUT --parameter omega {part_args} \\\n"
            f"#       {' '.join(f'--band {b}' for b in bands)}\n"
            f"# The settings below are runs/{parts[0]}'s, kept for reference: the loop refuses a combined run.\n"
            "#\n")
    # likelihood.input must be this folder's own cobaya.yaml (install/check.py, which submit_true_check.sh runs), as in
    # runs/axion_pp_combined; the copy is byte-identical. Fixed 6 November after true check 7047446 stopped on it.
    src_input = f"  input: runs/{parts[0]}/cobaya.yaml\n"
    assert first.count(src_input) == 1, name
    first = first.replace(src_input, f"  input: runs/{name}/cobaya.yaml\n")
    (d / "client.yaml").write_text(head + first)
    print(name, ":", len(parts), "parts, omega", union["omega"], ", bands", " ".join(bands))
