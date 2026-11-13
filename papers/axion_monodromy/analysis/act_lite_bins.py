"""The bins of ACT-lite (ACTDR6CMBonly) on Grendel, to check the binning assumed in analysis/binning200.py (9 Nov 2026).

    cd ~/codes/hpc_client && source env.sh && python analysis/act_lite_bins.py

Prints, for each spectrum's bandpower windows in the ACT-lite data file, the multipole range of every bin; the laptop's
estimate took the bins of the multi-frequency DR6 data (Delta l = 50 up to l ~ 2000, then 100, 200 and 400).
"""
import glob
import os

import numpy as np
from astropy.io import fits

root = os.environ.get("COBAYA_PACKAGES_PATH", os.path.expanduser("~/codes/cobaya_packages"))
files = sorted(f for f in glob.glob(os.path.join(root, "**", "*.fits"), recursive=True) if "cmbonly" in f.lower() or "lite" in f.lower())
print("files:", files)
for f in files:
    h = fits.open(f)
    for x in h:
        if x.name.startswith("window:Bandpower"):
            l, W = np.array(x.data["values"]), np.array(x.data["weight"])
            edges = [(int(l[W[:, b] > 0].min()), int(l[W[:, b] > 0].max())) for b in range(W.shape[1]) if (W[:, b] > 0).any()]
            print(os.path.basename(f), x.name, len(edges), "bins:", edges)
