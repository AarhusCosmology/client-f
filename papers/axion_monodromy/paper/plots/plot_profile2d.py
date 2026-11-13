"""The true profile likelihood of PDP+ in the plane of the frequency and the phase of the oscillation.

    python plot_profile2d.py

data/profile2d_pp_omega_phi.json (written by ../build.py when the study is complete): the 50 x 50 nodes, at each of which
A_osc and all other parameters are maximised with the true likelihood, and the best fit (the study's optimised anchor).
Writes ../Figures/profile2d_omega_phi.pdf: Delta chi^2 relative to the best fit, with the Gaussian 68%, 95% and 99.7%
levels for two parameters (Delta chi^2 = 2.30, 6.18 and 11.83, where they occur) and the best fit as a diamond.
"""
import numpy as np

from common import COLOURS, DIAMOND_EDGE, DIAMOND_SIZE, TEXTWIDTH, load_json, save_text_width, setup

VMAX = 12.0          # colour scale up to this Delta chi^2; the A_osc = 0 level lies at about 10 (Sec. 4.2)


def main():
    setup()
    import matplotlib.pyplot as plt

    d = load_json("profile2d_pp_omega_phi.json")
    assert d["parameters"] == ["omega", "phi"]
    om, ph = np.array(d["axes"][0]), np.array(d["axes"][1])
    logl = np.array([[np.nan if v is None else v for v in row] for row in d["logl"]], float).T     # [phi, omega]
    dchi = 2 * (d["best"] - logl)
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 0.42 * TEXTWIDTH))
    im = ax.pcolormesh(om, ph, np.minimum(dchi, VMAX), cmap="Blues_r", vmin=0, vmax=VMAX, shading="nearest",
                       rasterized=True)
    # the Gaussian 68%, 95% and 99.7% levels for two parameters; in (omega, phi) the last lies above the A_osc = 0 level
    # (Delta chi^2 = 10.1), so it encloses the whole plane and no line of it is drawn
    levels = [lv for lv in (2.30, 6.18, 11.83) if lv < dchi.max()]
    ax.contour(om, ph, dchi, levels=levels, colors="k", linewidths=[0.9, 0.6, 0.6][:len(levels)],
               linestyles=["-", "--", ":"][:len(levels)])
    ax.plot(d["best_x"]["omega"], d["best_x"]["phi"], "D", ms=DIAMOND_SIZE, mfc=COLOURS["pp"], mec="k", mew=DIAMOND_EDGE)
    ax.set_xlabel(r"$\omega$", fontsize=9)
    ax.set_ylabel(r"$\phi$", fontsize=9)
    ax.tick_params(labelsize=7)
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.04)
    cb.set_label(r"$\Delta\chi^2$", fontsize=9)
    cb.ax.tick_params(labelsize=7)
    save_text_width(fig, "profile2d_omega_phi")


if __name__ == "__main__":
    main()
