"""The low-omega parts again, as partners of the new high parts (4 November 2026; Steen: redo the low parts too).

For runs/axion_{pp,wc}_high200 (make_runs_high200.py), runs/axion_{pp,wc}_low200:
  - cobaya.yaml: a byte-identical copy of the high part's (prior omega U[1, 200], the converged CLASS++ sampling, the
    earlier phi window; it runs the high part's generate_Pk.py), as client.py combine requires;
  - the training box of the earlier low parts, omega [1, 3.1] (PDP+) and [1, 6.1] (H0WC);
  - client.yaml: the high part's settings (those of the earlier low parts too), with training.members 12 and
    member_pilot 100 (boost phase only), and for H0WC acquisition.boost.balance_floor 0.1, which both H0WC parts of the
    final combine had (results/20261030_100704_axion_wc_low, results/20261030_163913_axion_wc_high).
Also gives runs/axion_wc_high200 the same balance_floor 0.1 (missing from make_runs_high200.py, which copied
runs/axion_wc_high's settings).

    python make_runs_low200.py KIT
"""
import json
import sys
from pathlib import Path

kit = Path(sys.argv[1])
FLOOR = "    balance_floor: 0.1      # as both H0WC parts of the final combine (two modes, 90.6/9.4% in the low part)\n"
MIN_SHARE = "    min_share: 0.005\n"


def add_floor(text):
    if "balance_floor:" in text:
        return text
    assert text.count(MIN_SHARE) == 1
    return text.replace(MIN_SHARE, MIN_SHARE + FLOOR)


RUNS = {  # low part: (high part, omega box, band, key, extra header lines)
    "axion_pp_low200": ("axion_pp_high200", [1.0, 3.1], "2.6:3.0", "pp", ""),
    "axion_wc_low200": ("axion_wc_high200", [1.0, 6.1], "6.0:6.1", "wc",
                        "# Cut at omega ~ 6, not 3 as PDP+ (Steen, 29 October 2026). acquisition.boost.balance_floor 0.1 as both\n"
                        "# H0WC parts of the final combine.\n"),
}
for run, (high, omega, band, key, extra) in RUNS.items():
    h, d = kit / "runs" / high, kit / "runs" / run
    d.mkdir(exist_ok=True)
    (d / "cobaya.yaml").write_bytes((h / "cobaya.yaml").read_bytes())
    (d / "job.env").write_bytes((h / "job.env").read_bytes())
    box = json.loads((h / "domain_bounds.json").read_text())
    box["omega"] = omega
    (d / "domain_bounds.json").write_text(json.dumps(box, indent=1) + "\n")

    cfg = (h / "client.yaml").read_text()
    if key == "wc":
        cfg = add_floor(cfg)
        (h / "client.yaml").write_text(cfg)            # the high part gets the floor too
    settings = cfg[cfg.index("# Settings:"):]
    assert settings.count(f"input: runs/{high}/cobaya.yaml") == 1
    settings = settings.replace(f"input: runs/{high}/cobaya.yaml", f"input: runs/{run}/cobaya.yaml")
    assert f"- runs/{high}/generate_Pk.py" in settings        # the high part's script, as the earlier low parts
    head = cfg[:cfg.index("# The HIGH part")]
    head = head.replace(f"runs/axion_{key}_high/client.yaml (the training settings of the earlier high part) for the high "
                        f"part of omega [{'2.5' if key == 'pp' else '5.5'}, 200]",
                        f"runs/axion_{key}_low/client.yaml (the training settings of the earlier low part, the high part's)"
                        f" for the low part\n# of omega [{omega[0]:g}, {omega[1]:g}]")
    head = head.replace(f"submit_client.sh {high}", f"submit_client.sh {run}").replace("--nodes=4 --ntasks=192", "--nodes=2 --ntasks=96")
    head = head.replace("make_runs_high200.py", "make_runs_low200.py").replace("# On four nodes:", "# On two nodes:")
    head = head.replace(f"runs/{high}/cobaya.yaml, identical", f"runs/{run}/cobaya.yaml, identical")
    block = (f"# The LOW part of a domain split in omega: training box omega [{omega[0]:g}, {omega[1]:g}] (domain_bounds.json), prior\n"
             f"# omega U[1, 200]; cobaya.yaml is a byte-identical copy of runs/{high}'s. The two are combined with\n"
             f"#   python client.py combine OUTPUT --parameter omega --part results/<time>_{run} \\\n"
             f"#       --part results/<time>_{high} --band {band}\n" + extra +
             "#\n"
             f"#   sbatch --nodes=2 --ntasks=96 --time=5-00:00:00 submit_client.sh {run}\n"
             "#\n")
    (d / "client.yaml").write_text(head + block + settings)
    print(run, "box", omega, "partner", high)
