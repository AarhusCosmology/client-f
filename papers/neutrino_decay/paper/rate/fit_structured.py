#!/usr/bin/env python3
"""The structured two-component model, fitted against the same 539 measured bins.

Both components are the SAME momentum integral, one order apart:

    Phi_n(X) = int_0^inf du u^3 exp(-sqrt(u^2+X^2)) / (u^2+X^2)^(n/2)

    LO  (Hannestad-Raffelt, properly structured):  X^3 Phi_2(X),  Phi_2(0) = 1
    NLO (Chen/Barenboim, as published):            X^5 Phi_4(X),  Phi_4 = F exactly

so the model is

    Gamma_T / (Gamma_0 rho_H/rho_sec) = (1/12) [ A3 eps^n X^3 Phi_2(X)
                                               + A5      X^5 Phi_4(X) ]

NOTE THE ABSENT `a`.  The campaign's Gamma_T is defined by d(a^4 Pi)/dtau =
-a Gamma_T (a^4 Pi), i.e. with the a DIVIDED OUT, while COPW eq. (13) and the code's
TransportRate both carry it.  Putting it in as well costs a factor 1/a ~ 1e3 and
turns A5 into a few hundred -- which is how this was caught.

with A5 = 1 if COPW's normalisation is right.  As X -> 0 the two terms go as
gamma^-3 and gamma^-5, but that is an asymptote, not the fitted shape: over the
campaign's own window X in [0.19, 1.76] the NLO term's effective power is 3.44,
not 5, and BOTH terms shut off as X grows -- no invented factor needed.
"""
import numpy as np
from scipy.integrate import quad
import isodata as I

T0_EV = I.T0_EV
_XG = np.geomspace(1e-6, 60., 400)


def _phi(n):
    v = np.array([quad(lambda u: u**3*np.exp(-np.sqrt(u*u+X*X))/(u*u+X*X)**(n/2),
                       0, np.inf, limit=300)[0] for X in _XG])
    return lambda X: np.exp(np.interp(np.log(np.clip(X, _XG[0], _XG[-1])),
                                      np.log(_XG), np.log(v)))


PHI2, PHI4 = _phi(2), _phi(4)

# ── the calibration, in ONE place ────────────────────────────────────────────
# Refit 2026-09-14 on 1214 bins / 151 cells (was 483/60): the half-decade rungs entered
# once their proxy backgrounds existed, and the usable-cell fallback recovered cells that
# round precedence had discarded before the quality gate. rel-rms 0.3012 -> 0.2603.
# n3 stays pinned at 1/2 -- best fit is 0.580 and pinning costs 0.5%.
# These are the CLASS++ defaults too (species/dncdm_proxy_species.cpp, PR #432); if you
# refit, change them there as well or the note and the code drift apart.
A3, A5, N3 = 0.0743, 0.5532, 0.5
# The gamma-power ("two-power") form's amplitudes, same fit, same bins. alphafit's
# share5 -- and therefore fig_alphameas and the S5.1 table -- runs on THESE, not on the
# structured ones above.
P_C3, P_C5, P_N3 = 0.2335, 1.4945, 0.5


def design(cs):
    G, Y, E, S, X = I.flat(cs, 'gamma', 'y', 'eps', 'sigma', 'X')
    mass = np.concatenate([np.full(len(c['gamma']), float(c['mass'])) for c in cs])
    a = X * T0_EV / mass
    return G, Y, E, S, X, a


def fit(G, Y, E, S, X, a, n):
    """linear in (A3, A5) at fixed n; weighted by the per-bin sigma."""
    M = np.vstack([E**n*X**3*PHI2(X)/12., X**5*PHI4(X)/12.]).T
    W = 1.0/(S*Y)**2
    b = np.linalg.solve(M.T@(W[:, None]*M), M.T@(W*Y))
    r = (Y - M@b)/Y
    return b, float(np.sqrt(np.mean(r**2)))


def fit_powers(G, Y, E, S, n, q=0.0):
    """the incumbent: C3 eps^n gamma^-3 + C5 gamma^-5, optional shut-off."""
    sh = (1.-1./G**2)**q
    yy = Y/sh
    M = np.vstack([E**n*G**-3., G**-5.]).T
    W = 1.0/(S*yy)**2
    b = np.linalg.solve(M.T@(W[:, None]*M), M.T@(W*yy))
    r = (yy - M@b)/yy
    return b, float(np.sqrt(np.mean(r**2)))


if __name__ == "__main__":
    I.GHI = 15.
    cs = I.cells()
    G, Y, E, S, X, a = design(cs)
    print(f"{len(Y)} bins, {len(cs)} cells;  X in [{X.min():.3f}, {X.max():.3f}]")
    # A calibration that quietly shrinks is the failure this reports on.
    for line in getattr(I.cells, "skipped", []):
        print(f"  SKIPPED  {line}")
    print()
    ns = np.linspace(0.0, 1.6, 65)
    rr = [fit(G, Y, E, S, X, a, n)[1] for n in ns]
    nb = ns[int(np.argmin(rr))]
    bb, rb = fit(G, Y, E, S, X, a, nb)
    print("STRUCTURED MODEL   (1/12)[A3 eps^n X^3 Phi_2 + A5 X^5 Phi_4]")
    print(f"{'n':>7} {'A3':>10} {'A5':>10} {'rel-rms':>9}")
    for n in (0.0, 0.25, 0.5, 0.75, 1.0, nb):
        b, r = fit(G, Y, E, S, X, a, n)
        tag = '   <- best' if abs(n-nb) < 1e-9 else ''
        print(f"{n:>7.3f} {b[0]:>10.4f} {b[1]:>10.4f} {r:>9.4f}{tag}")
    print(f"\n  A5 = {bb[1]:.4f}   (COPW's own normalisation would be 1)")
    print("\nINCUMBENT for comparison, same bins, same weights:")
    for n, q, nm in ((0.5, 0.0, 'two-power, no shut-off (shipped)'),
                     (0.5, 1.7995, 'two-power + fitted shut-off'),
                     (0.8714, 1.7995, 'two-power, published n3 + shut-off')):
        b, r = fit_powers(G, Y, E, S, n, q)
        print(f"  {nm:<36} C3={b[0]:.4f} C5={b[1]:.4f}  rel-rms {r:.4f}")
