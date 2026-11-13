"""The probe scans to omega 200 (2 November 2026): the chi^2 gain over A_osc = 0, profiled over A_osc and phi with every
other parameter at the reference point. PDP+: probes 6973447 (omega 1-50), 6973710 (40-100, the runs' settings) and
7010679 (55-200, the converged settings), each used where it is the newest; H0WC: 7010680 (55-200).

    python plot_scan.py KIT_LOGS OUT.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logs, out = sys.argv[1], sys.argv[2]


def profile(job):
    d = json.load(open(f"{logs}/axion_probe_{job}.json"))
    return np.array([p["omega"] for p in d["profile"]]), 2 * np.array([p["gain"] for p in d["profile"]])


pieces = [(profile("6973447"), (1, 40)), (profile("6973710"), (40, 55)), (profile("7010679"), (55, 200.01))]
pp = np.concatenate([np.column_stack([o, c])[(o >= lo) & (o < hi)] for (o, c), (lo, hi) in pieces])
wc = np.column_stack(profile("7010680"))
INK, MUTED, LINE = "#1f3b73", "#6b7280", "#c0c4cc"
fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.6), sharex=True, constrained_layout=True)
for ax, data, title, best in ((axes[0], pp, "PDP+", None), (axes[1], wc, "H$_0$WC", 7.48)):
    ax.plot(data[:, 0], np.maximum(data[:, 1], 0), color=INK, lw=1.0)
    ax.axvline(60, color=MUTED, ls="--", lw=0.8)
    ax.text(61.5, 0.93, "prior of the paper ends", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=8, va="top")
    if best is not None:
        ax.axhline(best, color=MUTED, ls=":", lw=0.8)
        ax.text(199, best + 0.2, f"true-profile best fit at $\\omega \\simeq 31$: {best}", color=MUTED, fontsize=8, ha="right", va="bottom")
    hi = data[data[:, 0] > 60]
    for k in np.argsort(-hi[:, 1])[:1]:
        pass
    tops = []
    for w, c in sorted(hi.tolist(), key=lambda t: -t[1]):
        if all(abs(w - t) > 8 for t, _ in tops):
            tops.append((w, c))
        if len(tops) == 3:
            break
    for w, c in tops:
        ax.annotate(f"{w:g}", (w, c), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=INK)
    if title == "PDP+":
        w, c = data[np.argmax(data[:, 1])]
        ax.annotate(f"{w:g}", (w, c), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_ylabel(r"$\Delta\chi^2$ gain")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_ylim(0, 12)
    ax.grid(axis="y", color=LINE, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[1].set_xlabel(r"$\omega$")
axes[1].set_xlim(0, 201)
fig.suptitle(r"Gain over $A_{\rm osc}=0$, profiled over $A_{\rm osc}$ and $\phi$, other parameters at the reference point",
             fontsize=9, color=MUTED)
fig.savefig(out, dpi=160)
print("written", out, "PDP+ max above 60:", pp[pp[:, 0] > 60][np.argmax(pp[pp[:, 0] > 60][:, 1])], "H0WC max:", wc[np.argmax(wc[:, 1])])
