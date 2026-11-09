"""Prints every node of a profile_chain.py log of the toy against the exact profile of its own branch and the other's.

    python check_toy.py T LOG
"""
import json, sys
from pathlib import Path
import numpy as np
T = Path(sys.argv[1]); logf = Path(sys.argv[2])
ex = json.loads((T / "toy" / "exact.json").read_text())
const = -4.526126978647639
r = json.loads(logf.read_text())
worst = 0
for label, ch in r["chains"].items():
    p = ch["profile"]; e = ex[p]
    for nd in ch["nodes"]:
        x = nd["coordinate"]
        exact = max(-0.5 * (x - e["mu1"]) ** 2 / e["var1"], -0.5 * (x - e["mu2"]) ** 2 / e["var2"] - e["off2"]) + e["nuis"] + const
        got = nd["best"]["logpost"] if nd["best"] else np.nan
        worst = max(worst, abs(2 * (got - exact)))
        print(f"{label:12s} {p}={x:+.2f}  -2lnP got {-2*got:8.4f} exact {-2*exact:8.4f}  diff chi2 {2*(exact-got):+.5f}  evals {nd['evaluations']:3d}  {nd.get('ended','')[:30]}")
    if "error" in ch: print(label, "ERROR", ch["error"][-300:])
print("worst |chi2 diff|", worst)
