"""The frequency of the oscillation: the true profile likelihood and the marginalised posterior of omega, per dataset.

    python plot_omega.py

data/profile_<d>_omega_phi.json: the nodes of the true (omega, phi) profile done so far, each maximised over A_osc and all
other parameters with the true likelihood; per omega the best node over phi. data/profile_<d>_aosc_omega.json: its row
A_osc = 0 gives the best fit without oscillation. data/posterior_<d>.npz: the combined emulator's posterior.
data/best_fit_<d>.json: the best fit with and without oscillation.
Writes ../Figures/omega_pp.pdf and ../Figures/omega_wc.pdf: top, the improvement in chi^2 over the model without
oscillation, with dashed lines at Delta chi^2 = 1 and 4 relative to the best fit of the dataset, the best fit as a diamond
(its omega from data/profile2d_<d>_omega_phi.json), and shaded where the value is still the emulator's; bottom, the posterior density of omega on a logarithmic scale, with the band in which the two emulators are
blended. In the H0WC figure PDP+ is repeated as a thin line for comparison. The top panel of H0WC stays without its
own line until the study has nodes.
"""
import numpy as np

from common import COLOURS, DATA, DATA_DIR, DIAMOND_EDGE, DIAMOND_SIZE, TEXTWIDTH, load_json, save_text_width, setup, shade_cells

BAND = {"pp": (2.6, 3.0), "wc": (6.0, 6.1)}
NETWORK = "#e08214"        # the network's profile (Figs. omega, omega_wc)
LEVELS_1D = [1, 4]


def profile_gain(d):
    """omega, the improvement in chi^2 of the best node per omega and whether that node is still the emulator's; None before
    the study has nodes."""
    if not (DATA_DIR / f"profile_{d}_omega_phi.json").exists():
        return None
    op = load_json(f"profile_{d}_omega_phi.json")["nodes"]
    nodes = load_json(f"profile_{d}_aosc_omega.json")["nodes"]
    row = ([n["logl"] for n in nodes if n["i"] == 0 and n["reason"] == "converged"]
           or [n["logl"] for n in nodes if n["i"] == 0 and n["reason"] == "emulator"])
    if not op or not row:
        return None
    best = {}
    for n in op:
        if n["i"] not in best or n["logl"] > best[n["i"]]["logl"]:
            best[n["i"]] = n
    cols = [best[i] for i in sorted(best)]
    return (np.array([c["omega"] for c in cols]), np.array([2 * (c["logl"] - max(row)) for c in cols]),
            np.array([c["reason"] == "emulator" for c in cols]))


def density(d):
    z = np.load(DATA_DIR / f"posterior_{d}.npz")
    names = [str(n) for n in z["names"]]
    h, e = np.histogram(z["samples"][:, names.index("omega")], bins=np.arange(1, 60.001, 0.25), density=True)
    return 0.5 * (e[1:] + e[:-1]), np.where(h > 0, h, np.nan)


def figure_200(d="pp"):
    """omega <= 200 (data/omega200_<d>.json and data/posterior200_<d>.npz, written by ../build200.py): the true profile from
    the (omega, phi) surface joined from its windows and the network's own profile (the optima of the studies' scans of the
    emulator), both against the true best fit without oscillation; the posterior of the newest combined emulator, and its
    bands. One data set per figure (8 November: H0WC without PDP+)."""
    import matplotlib.pyplot as plt

    e = load_json(f"omega200_{d}.json")
    om = np.array(e["omega"])
    gain = np.array([np.nan if g is None else g for g in e["gain"]], float)
    ok = np.isfinite(gain)
    fig, ax = plt.subplots(2, 1, figsize=(TEXTWIDTH, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[1.1, 1]))
    ax[0].axhline(0, color="0.6", lw=0.6)
    for band in e["bands"]:
        for a in ax:
            a.axvspan(*band, color="0.85", lw=0)
    shade_cells(ax[0], om, np.zeros(len(om), bool) if e.get("closed") else np.array(e["emulator"], bool))
    # the network's own profile (the scans' optima), against the same best fit without oscillation (8 November, Steen)
    if "network_gain" in e:
        net = np.array([np.nan if g is None else g for g in e["network_gain"]], float)
        # wider, underneath: where the two agree it shows as an edge of the true curve
        ax[0].plot(om, net, "-", color=NETWORK, lw=2.2, alpha=0.85, label="network", zorder=2, solid_capstyle="round")
    ax[0].plot(om[ok], gain[ok], "-", color="k", lw=0.8, label="true likelihood", zorder=3)
    ax[0].plot(om[ok], gain[ok], "o", color="k", ms=1.4, zorder=3)
    # the best fit (10 November, Steen): the best true node of the three planes, found in (A_osc, phi) with omega free, so
    # that it lies between two nodes of the grid in omega and above the profile drawn on the grid
    if (DATA_DIR / f"profile2d_{d}_omega_phi.json").exists():
        best_omega = load_json(f"profile2d_{d}_omega_phi.json")["best_x"]["omega"]
        ax[0].plot(best_omega, 2 * (e["lnp"] - e["lnp_lcdm"]), "D", ms=DIAMOND_SIZE, mfc=COLOURS[d], mec="k",
                   mew=DIAMOND_EDGE, zorder=4, label="best fit")
    for lv in LEVELS_1D:
        y = 2 * (e["lnp"] - e["lnp_lcdm"]) - lv
        ax[0].axhline(y, color="0.45", ls="--", lw=0.6)
        ax[0].text(3.5, y, rf"$\Delta\chi^2 = {lv}$", ha="left", va="bottom", fontsize=6.5, color="0.3")
    ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, 0.99), fontsize=7, frameon=False, ncol=3, borderaxespad=0.1)
    z = np.load(DATA_DIR / f"posterior200_{d}.npz")
    names = [str(n) for n in z["names"]]
    h, edges = np.histogram(z["samples"][:, names.index("omega")], bins=np.arange(1, 200.001, 0.25), density=True)
    ax[1].step(0.5 * (edges[1:] + edges[:-1]), np.where(h > 0, h, np.nan), where="mid", color=COLOURS[d], lw=0.6)
    ax[0].set_ylabel(r"$-\Delta\chi^2$", fontsize=9)
    ax[1].set_yscale("log")
    ax[1].set_ylabel(r"$P(\omega)$", fontsize=9)
    ax[1].set_xlabel(r"$\omega$", fontsize=9)
    ax[1].set_xlim(1, 200)
    for a in ax:
        a.tick_params(labelsize=7)
    fig.subplots_adjust(hspace=0.06)
    save_text_width(fig, f"omega_{d}")


def figure(d):
    import matplotlib.pyplot as plt

    if (DATA_DIR / f"omega200_{d}.json").exists() and (DATA_DIR / f"posterior200_{d}.npz").exists():
        return figure_200(d)
    fig, ax = plt.subplots(2, 1, figsize=(TEXTWIDTH, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[1.1, 1]))
    ax[0].axhline(0, color="0.6", lw=0.6)
    ax[1].axvspan(*BAND[d], color="0.85", lw=0)
    if d != "pp":
        ax[0].plot(*profile_gain("pp")[:2], "-", color=COLOURS["pp"], lw=0.6, label=DATA["pp"])
        ax[1].step(*density("pp"), where="mid", color=COLOURS["pp"], lw=0.5, label=DATA["pp"])
    gain = profile_gain(d)
    if gain is not None:
        om, dchi, from_emulator = gain
        shade_cells(ax[0], om, from_emulator)
        ax[0].plot(om, dchi, "-", color="k", lw=0.9, label=None if d == "pp" else DATA[d])
        ax[0].plot(om, dchi, "o", color="k", ms=2.2)
        fit = load_json(f"best_fit_{d}.json")
        for lv in LEVELS_1D:
            y = 2 * (fit["lnp"] - fit["lnp_lcdm"]) - lv
            ax[0].axhline(y, color="0.45", ls="--", lw=0.6)
            ax[0].text(2.6, y, rf"$\Delta\chi^2 = {lv}$", ha="left", va="bottom", fontsize=6.5, color="0.3")
    ax[1].step(*density(d), where="mid", color=COLOURS[d], lw=0.9, label=DATA[d] if d != "pp" else None)
    ax[0].set_ylabel(r"$-\Delta\chi^2$", fontsize=9)
    ax[1].set_yscale("log")
    ax[1].set_ylabel(r"$P(\omega)$", fontsize=9)
    ax[1].set_xlabel(r"$\omega$", fontsize=9)
    ax[1].set_xlim(1, 60)
    for a in ax:
        a.tick_params(labelsize=7)
        if d != "pp":
            a.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    fig.subplots_adjust(hspace=0.06)
    save_text_width(fig, f"omega_{d}")


def main():
    setup()
    for d in ("pp", "wc"):
        figure(d)


if __name__ == "__main__":
    main()
