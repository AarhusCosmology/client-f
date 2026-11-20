"""Table 1 of the paper: the cost of one lensed CMB and matter power spectrum, in core-hours of a q64 core.

    python3 -I cost_table.py

The full solution and the six moments ran on Grendel's q64 nodes (64 cores of an Intel Xeon Platinum 8358, CLASS++
37f95e77), and their cost is 64 times the wall-clock time each cell's log records (ELAPSED_S in
$DNCDM_HARNESS/hpc_prod2/logs/*.out). The proxy was timed on a laptop (Apple M3 Pro, one thread, CLASS++ cb2255f4) on
the inputs of figure 2 (campaigns/proxy_rta/px_ours_*.ini), and is converted with the factor 1.6 of a loaded q48 core
over the laptop (an evaluation of SPA+BAO+SN at log10Gamma 12.5-13: 498 s in CLiENT's logs on q48, 306 s on the
laptop); a q64 core is as fast as a q48 core or up to about 20% faster, hence the caption's 20%. The laptop's
seconds are in proxy_timings_laptop.txt; time_proxy.sh measures them again.
"""
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = Path(os.environ.get("DNCDM_HARNESS") or HERE.parents[0] / "campaigns")
LAPTOP_TO_CORE = 1.6
ROWS = {"0.3": ("05.0", "07.0", "09.0", "10.0", "11.0", "12.0"), "0.06": ("09.0", "10.0")}

walls = {}
for log in (HARNESS / "hpc_prod2" / "logs").glob("cmb_*.out"):
    text = log.read_text()
    ini, elapsed, cpus = (re.search(p, text, re.M) for p in (r"^ini\s+\S*/(\S+)\.ini", r"^ELAPSED_S\s+(\d+)", r"^CPUS\s+(\d+)"))
    if ini and elapsed and cpus:
        walls[ini.group(1)] = int(elapsed.group(1)) * int(cpus.group(1)) / 3600

laptop = {}
for line in (HERE / "proxy_timings_laptop.txt").read_text().splitlines():
    if line.strip() and not line.startswith("#"):
        mass, gl, seconds = line.split()
        laptop[(mass, gl)] = float(seconds)


def fmt(x):
    if x is None:
        return "--"
    return f"{x:.2g}" if x < 1 else f"{x:.3g}" if x < 1000 else f"{round(x, -2):,.0f}"


print(f"{'m [eV]':>6} {'log10G':>6} {'full':>8} {'moments':>8} {'proxy':>8}   core-hours")
for mass, gls in ROWS.items():
    for gl in gls:
        full = walls.get(f"cmb_G{gl}_m{mass}_x")
        m6 = walls.get(f"cmb_G{gl}_m{mass}_m6")
        px = laptop.get((mass, gl))
        px = None if px is None else px * LAPTOP_TO_CORE / 3600
        print(f"{mass:>6} {float(gl):6.0f} {fmt(full):>8} {fmt(m6):>8} {fmt(px):>8}")
