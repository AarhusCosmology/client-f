"""Fig. 5 (model comparison): the true profile likelihood of log10 Geff in the full model, and the true chi^2 at the
emulator's best fits on the boundary Delta N_eff = f_idm = 0 (LCDM, SInu at log10 Geff = -4, SInu's best fit in the strongly interacting
mode).

    python plot_profile_logGeff.py

data/profiles.json: the log10 Geff nodes (x, t) and, in "face", the three points on that boundary (log10 Geff, Delta chi^2 to the
full model's global best fit, label). Writes ../Figures/profile_logGeff.pdf.
"""
from common import COLOURS, DATA, DIAMOND_EDGE, DIAMOND_SIZE, TEXTWIDTH, load_json, save_text_width, setup

FACE_NAMES = {"LCDM": r"$\Lambda$CDM", "SInu_weak": r"SI$\nu$, moderate", "SInu_strong": r"SI$\nu$, strong",
              "SInu_best": r"SI$\nu$, best fit"}
# Drawn at the width of the text (common.save_text_width; until 28 October 2026 11 inches wide and shrunk to 0.57, with
# labels of 4-6 pt). The caption gives the reference of Delta chi^2.
HEIGHT = 0.43                    # the height of the figure relative to its width


def main():
    setup()
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple

    profs = load_json("profiles.json")
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, HEIGHT * TEXTWIDTH))
    lines, face_marker = [], None
    for ax, d in zip(axes, DATA):
        if d not in profs:                      # no profile of this data set yet (a draft of the World Cup section)
            ax.set_title(DATA[d] + " (pending)", fontsize=9, pad=3)
            continue
        rows = profs[d]["P"].get("logGeff", [])
        lines += ax.plot([r["x"] for r in rows], [r["t"] for r in rows], "-o", color=COLOURS[d], ms=2, lw=1.1)
        face = [(p["logGeff"], p["dchi2"]) for p in profs[d]["face"]]
        face_marker = ax.plot(*zip(*face), "D", color="0.2", mfc="white", ms=DIAMOND_SIZE, mew=DIAMOND_EDGE)[0]
        top = max(12, max(y for _, y in face) + 5)
        for p in profs[d]["face"]:
            right = p["logGeff"] > -2.0          # near the right edge the name goes to the left of the point
            # a name to the left must clear the profile under it (28 October 2026: with PDP+ SInu's point fell to 3.7
            # and its name lay on the bump of the profile at -2.5); else above the point as before
            under = [r["t"] for r in rows if p["logGeff"] - 1.0 <= r["x"] <= p["logGeff"]]
            if right and under and max(under) > p["dchi2"]:
                ax.annotate(FACE_NAMES[p["label"]], (p["logGeff"], p["dchi2"]), xytext=(p["logGeff"] - 0.03, max(under) + 0.04 * top),
                            textcoords="data", ha="right", fontsize=7)
                continue
            ax.annotate(FACE_NAMES[p["label"]], (p["logGeff"], p["dchi2"]), xytext=(-3, 7) if right else (4, 3),
                        textcoords="offset points", ha="right" if right else "left", fontsize=7)
        ax.axhline(1, color="0.6", lw=0.5)
        ax.axhline(4, color="0.6", lw=0.5, ls=":")
        ax.set_ylim(-0.5, top)
        ax.set_xlim(-5.6, -0.8)
        ax.set_xlabel(r"$\log_{10}(G_\mathrm{eff}\,\mathrm{MeV}^2)$", fontsize=8.5, labelpad=1.5)
        ax.set_title(DATA[d], fontsize=9, pad=3)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7, pad=1.5, length=2.5, width=0.6)
        for side in ax.spines.values():
            side.set_linewidth(0.6)
    axes[0].set_ylabel(r"$\Delta\chi^2$", fontsize=8.5, labelpad=1.5)
    # one legend above the panels (inside them, at print size, it covered SInu's points and their names): the profile
    # line in the colours of both data sets
    legend = fig.legend([tuple(lines), face_marker], [r"SI$\nu$+DRMD: profile likelihood",
                                                      r"SI$\nu$ ($\Delta N_\mathrm{eff} = f_\mathrm{idm} = 0$)"],
                        handler_map={tuple: HandlerTuple(ndivide=None)}, loc="lower center", ncol=2, fontsize=7.5,
                        frameon=False, handlelength=3.0, columnspacing=2.5, borderaxespad=0, borderpad=0)

    def layout():
        top = 1 - (10.5 + 10) / 72.27 / fig.get_size_inches()[1]    # room for the legend's row of 7.5 pt
        fig.tight_layout(pad=0.05, w_pad=0.8, rect=(0, 0, 1, top))
        left, right = axes[0].get_position().x0, axes[-1].get_position().x1
        legend.set_bbox_to_anchor(((left + right) / 2, top + 0.006), transform=fig.transFigure)
    save_text_width(fig, "profile_logGeff", layout)
    plt.close(fig)


if __name__ == "__main__":
    main()
