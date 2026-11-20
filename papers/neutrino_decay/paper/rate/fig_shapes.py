#!/usr/bin/env python3
"""The two shape functions and what they do to the effective exponent."""
import numpy as np, matplotlib.pyplot as plt
from scipy.special import exp1
import pstyle as ps
ps.use()
FIG = __file__.rsplit('/', 2)[0] + '/fig'      # paper/fig

P2 = lambda X: (1 + X) * np.exp(-X) - X**2 * exp1(X)
P4 = lambda X: 0.5 * np.exp(-X) * (X - 1) + (1 - 0.5 * X**2) * exp1(X)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.1, 2.7))
X = np.geomspace(1e-3, 30, 600)

# ── (a) the two rate shapes, and where they die ─────────────────────────────
axA.plot(X, X**3 * P2(X), color=ps.CAT[0], lw=1.5, label=r'$X^3\Phi_2(X)$   leading')
axA.plot(X, X**5 * P4(X), color=ps.CAT[1], lw=1.5, label=r'$X^5\Phi_4(X)$   next-to-leading')
axA.plot(X, X**3, color=ps.CAT[0], lw=0.9, ls=(0, (2, 2)),
         label=r'$X^3\ \ (\propto\gamma^{-3})$')
axA.plot(X, X**5, color=ps.CAT[1], lw=0.9, ls=(0, (2, 2)),
         label=r'$X^5\ \ (\propto\gamma^{-5})$')
axA.axvspan(0.187, 1.764, color=ps.GRID, zorder=0)
axA.annotate('measured\nwindow', xy=(0.58, 3e-6), fontsize=6.2, color=ps.INK2,
             ha='center', linespacing=1.3)
axA.set_xscale('log'); axA.set_yscale('log')
axA.set_xlim(1e-3, 30); axA.set_ylim(1e-9, 3e2)
axA.set_xlabel(r'$X = a\,m_{\nu_H}/T_0$')
axA.set_ylabel('rate shape')
axA.legend(loc='upper left', fontsize=6.0)
ps.panel(axA, '(a)', loc=(-0.03, 1.02))

# ── (b) the local exponent ──────────────────────────────────────────────────
def slope(f, X, h=1e-4):
    return (np.log(f(X * (1 + h))) - np.log(f(X))) / np.log(1 + h)
axB.plot(X, slope(lambda x: x**5 * P4(x), X), color=ps.CAT[1], lw=1.5,
         label=r'$X^5\Phi_4$')
axB.plot(X, slope(lambda x: x**3 * P2(x), X), color=ps.CAT[0], lw=1.5,
         label=r'$X^3\Phi_2$')
for lev, c in ((5., ps.CAT[1]), (3., ps.CAT[0])):
    axB.axhline(lev, color=c, lw=0.8, ls=(0, (2, 2)))
axB.axvspan(0.187, 1.764, color=ps.GRID, zorder=0)
axB.annotate('the asymptotes are reached\nonly for $X\\ll1$', xy=(2e-3, 1.2),
             fontsize=6.4, color=ps.INK2, linespacing=1.4)
axB.set_xscale('log'); axB.set_xlim(1e-3, 30); axB.set_ylim(-1, 5.6)
axB.set_xlabel(r'$X = a\,m_{\nu_H}/T_0$')
axB.set_ylabel(r'local exponent  $\mathrm{d}\ln(\cdot)/\mathrm{d}\ln X$')
axB.legend(loc='lower left', fontsize=6.6)
ps.panel(axB, '(b)', loc=(-0.03, 1.02))
fig.subplots_adjust(wspace=0.30)
ps.finish(fig, FIG + '/fig_shapes.pdf')

lo, hi = 0.187, 1.764
r = (hi**5 * P4(hi)) / (lo**5 * P4(lo))
print(f"over the measured window X = {lo}-{hi}:")
print(f"  X^5 Phi_4 changes by {r:.4g}x  -> effective power {np.log(r)/np.log(hi/lo):.2f}")
print(f"  local slope {slope(lambda x: x**5*P4(x), np.array([lo]))[0]:.2f} -> "
      f"{slope(lambda x: x**5*P4(x), np.array([hi]))[0]:.2f}")
