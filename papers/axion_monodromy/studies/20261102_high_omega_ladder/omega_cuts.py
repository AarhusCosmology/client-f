"""Where to cut omega into zones (4 November 2026): the probe scans' chi^2 gain over A_osc = 0 (profiled over A_osc and
phi, every other parameter at the reference point) near candidate cuts, for PDP+ (probes 6973447, 6973710, 7010679,
joined as plot_scan.py does) and H0WC (7010680, omega 55-200 only), and the peaks of at least 3 inside each zone.

    python omega_cuts.py KIT_LOGS
"""
import json
import sys

import numpy as np

logs = sys.argv[1]


def profile(job):
    d = json.load(open(f"{logs}/axion_probe_{job}.json"))
    return np.array([p["omega"] for p in d["profile"]]), 2 * np.array([p["gain"] for p in d["profile"]])


pieces = [(profile("6973447"), (1, 40)), (profile("6973710"), (40, 55)), (profile("7010679"), (55, 200.01))]
SCANS = {"PDP+": np.concatenate([np.column_stack([o, c])[(o >= lo) & (o < hi)] for (o, c), (lo, hi) in pieces]),
         "H0WC": np.column_stack(profile("7010680"))}


def largest(d, lo, hi):
    m = (d[:, 0] >= lo) & (d[:, 0] <= hi)
    return np.max(np.maximum(d[m, 1], 0)) if m.any() else float("nan")


def peaks(d, lo, hi, at_least=3.0):
    m = (d[:, 0] >= lo) & (d[:, 0] < hi)
    om, g = d[m, 0], np.maximum(d[m, 1], 0)
    return [(om[i], g[i]) for i in range(1, len(g) - 1) if g[i] >= at_least and g[i] >= g[i - 1] and g[i] >= g[i + 1]]


print("cut     largest gain within +-1 / +-2 / +-3 of it")
for cut in (60.5, 61, 62, 88, 111, 112, 123, 154, 155):
    print(f"{cut:6.1f}  " + "   ".join(f"{name} " + " / ".join(f"{largest(d, cut - w, cut + w):3.1f}" for w in (1, 2, 3))
                                    for name, d in SCANS.items()))
for name, d in SCANS.items():
    for lo, hi in ((3, 61), (61, 111), (111, 155), (155, 200.01)):
        p = peaks(d, lo, hi)
        print(f"{name} [{lo}, {hi:g}): {len(p)} peaks >= 3: " + ", ".join(f"{o:.2f} ({g:.1f})" for o, g in p))
