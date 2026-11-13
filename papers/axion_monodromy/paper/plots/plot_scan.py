"""The scans of the likelihood over the frequency, up to omega = 200 (Appendix B).

    python plot_scan.py

data/scan_omega.json: per dataset, the improvement in chi^2 over the model without oscillation at the reference point of
the other parameters, maximised over A_osc and phi on a grid in omega, as the probes of the likelihood computed it (for
PDP+ joined from three probes: omega < 55 with the settings of the runs, above with the settings for omega <= 200; for
H0WC omega >= 55 only). Writes ../Figures/scan_omega.pdf: one panel per dataset, with the frequencies of the four highest
maxima. (The draft for omega <= 60 also marks omega = 60, the upper bound of its prior.)
"""
import numpy as np

from common import COLOURS, DATA, TEXTWIDTH, load_json, save_text_width, setup

LABELLED, SEPARATION = 4, 8.0          # the highest maxima labelled, at least this far apart in omega


def maxima(om, dchi):
    """The LABELLED highest local maxima, at least SEPARATION apart."""
    peaks = [k for k in range(1, len(om) - 1) if dchi[k] >= dchi[k - 1] and dchi[k] >= dchi[k + 1]]
    chosen = []
    for k in sorted(peaks, key=lambda k: -dchi[k]):
        if all(abs(om[k] - om[j]) > SEPARATION for j in chosen):
            chosen.append(k)
        if len(chosen) == LABELLED:
            break
    return chosen


def main():
    setup()
    import matplotlib.pyplot as plt

    scan = load_json("scan_omega.json")
    fig, ax = plt.subplots(2, 1, figsize=(TEXTWIDTH, 3.6), sharex=True, sharey=True)
    for a, d in zip(ax, ("pp", "wc")):
        om, dchi = np.array(scan[d]["omega"]), np.maximum(np.array(scan[d]["dchi2"]), 0.0)
        a.plot(om, dchi, "-", color=COLOURS[d], lw=0.8)
        for k in maxima(om, dchi):
            a.text(om[k], dchi[k] + 0.3, f"${om[k]:g}$", ha="center", va="bottom", fontsize=6.5, color="0.2")
        a.text(0.995, 0.93, DATA[d], transform=a.transAxes, ha="right", va="top", fontsize=8)
        a.set_ylabel(r"$-\Delta\chi^2$", fontsize=9)
        a.tick_params(labelsize=7)
    ax[0].set_ylim(0, 12)
    ax[1].set_xlim(0, 200)
    ax[1].set_xlabel(r"$\omega$", fontsize=9)
    fig.subplots_adjust(hspace=0.08)
    save_text_width(fig, "scan_omega")


if __name__ == "__main__":
    main()
