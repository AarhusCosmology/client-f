#!/usr/bin/env python3
"""The calibrated model drawn on the measured isotropisation rate, BOTH masses.

Was a 1x4 row at m = 0.3 only, which showed half the calibration set and none of the
mass dependence. Now 2x4: one row per mass, four couplings spanning each ladder, with
three of the four columns matched in Gamma so the mass comparison is read across.
"""
import numpy as np, matplotlib.pyplot as plt
import pstyle as ps, isodata as I
from fit_structured import design, fit, PHI2, PHI4, A3, A5, N3 as N
ps.use()
FIG = __file__.rsplit('/', 2)[0] + '/fig'      # paper/fig

# Amplitudes come from fit_structured, which is the single source of truth.

I.GHI = 15.
CS = I.cells()

# Four couplings per mass, spanning each ladder. 1e5/1e7/1e9 are matched across the two
# rows; the last column is each ladder's own top rung, which is not the same Gamma.
SHOW = {'0.3':  ['05.0', '07.0', '09.0', '11.0'],
        '0.06': ['05.0', '07.0', '09.0', '10.0']}
MLAB = {'0.3': r'$m_{\nu_H}=0.3$ eV', '0.06': r'$m_{\nu_H}=0.06$ eV'}

fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.6), sharey=True)
for row, mass in enumerate(('0.3', '0.06')):
    for col, gl in enumerate(SHOW[mass]):
        ax = axes[row, col]
        sub = [c for c in CS if c['mass'] == mass and c['gl'] == gl]
        if not sub:
            ax.axis('off')
            continue
        G, Y, E, S, X, a = design(sub)
        o = np.argsort(X)
        ax.errorbar(X[o], Y[o], yerr=(S * Y)[o], color=ps.INK, marker='o', ms=2.6, lw=0,
                    mew=0, elinewidth=0.7, capsize=1.3, zorder=5, label='measured')
        xf = np.geomspace(X.min() * 0.9, X.max() * 1.1, 200)
        ef = np.exp(np.interp(np.log(xf), np.log(X[o]), np.log(E[o])))
        t3 = A3 * ef ** N * xf ** 3 * PHI2(xf) / 12.
        t5 = A5 * xf ** 5 * PHI4(xf) / 12.
        # The total goes UNDERNEATH and wide; the components on top. At high coupling the
        # leading term is negligible and the total lies exactly on top of the NLO one, so
        # a thin total drawn last would simply hide it.
        ax.plot(xf, t3 + t5, color=ps.CAT[0], lw=3.0, alpha=0.30, zorder=2,
                solid_capstyle='round', label='model')
        ax.plot(xf, t5, color=ps.CAT[2], lw=1.2, ls=(0, (5, 1.6)), zorder=3,
                label=r'$A_5X^5\Phi_4$')
        ax.plot(xf, t3, color=ps.CAT[3], lw=1.2, ls=(0, (1.6, 1.6)), zorder=4,
                label=r'$A_3\varepsilon^{1/2}X^3\Phi_2$')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.text(0.96, 0.06, rf'$\Gamma=10^{{{float(gl):g}}}$', transform=ax.transAxes, ha='right', va='bottom',
                fontsize=7.0)
        if row == 1:
            ax.set_xlabel(r'$X = a\,m_{\nu_H}/T_0$')
    # The mass belongs to the row, not to any one panel.
    axes[row, 0].set_ylabel(r'$\Gamma_T\,/\,(\Gamma\,\bar\rho_{\nu_H}/\bar\rho_{\rm sec})$')
    axes[row, 0].annotate(MLAB[mass], xy=(0.05, 0.92), xycoords='axes fraction', va='top',
                          fontsize=7.0, color=ps.INK)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.995), handlelength=2.4)
fig.subplots_adjust(left=0.105, right=0.985, bottom=0.11, top=0.9, wspace=0.08, hspace=0.12)
ps.finish(fig, FIG + '/fig_ratefit.pdf', crop=False)

G, Y, E, S, X, a = design(CS)
ns = np.linspace(0.0, 1.6, 81)
rr = [fit(G, Y, E, S, X, a, n)[1] for n in ns]
nb = ns[int(np.argmin(rr))]
b, r = fit(G, Y, E, S, X, a, nb)
print(f"{len(Y)} bins, {len(CS)} cells")
print(f"  best n = {nb:.3f}  A3 = {b[0]:.4f}  A5 = {b[1]:.4f}  rel-rms {r:.4f}")
b2, r2 = fit(G, Y, E, S, X, a, 0.5)
print(f"  n pinned 1/2: A3 = {b2[0]:.4f}  A5 = {b2[1]:.4f}  rel-rms {r2:.4f}")
