"""Make every figure of the paper from the files in data/: python make_all.py [--series wc]

--series wc makes the figures of Sec. 5 (the runs on the H0 World Cup data) from data/wc/, written by ../build_wc.py.
A figure whose data are not there yet is left out there, with a message (the section is drafted while runs finish);
in the default series every figure's data must be there.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# the data each script needs, in the World Cup series (data/wc/)
NEEDS = {"plot_profile_breakdown.py": ["profile_breakdown.json"], "plot_profile_logGeff.py": ["profiles.json"],
         "plot_profiles.py": ["profiles.json"], "plot_triangle.py": ["best_fits.json"]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", choices=("drmd2", "wc"), default="drmd2")
    args = parser.parse_args()
    env = dict(os.environ, PAPER_SERIES=args.series)
    for script in sorted(HERE.glob("plot_*.py")):
        if args.series == "wc":
            if script.name not in NEEDS:
                continue
            missing = [f for f in NEEDS[script.name] if not (HERE / "data" / "wc" / f).exists()]
            if missing:
                print(f"== {script.name}: left out, no data/wc/{', '.join(missing)} yet", flush=True)
                continue
        print(f"== {script.name}", flush=True)
        subprocess.run([sys.executable, script.name], cwd=HERE, env=env, check=True)
