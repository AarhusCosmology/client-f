#!/usr/bin/env python3
"""The primordial spectrum of the axion monodromy runs (runs/axion_pp_high/generate_Pk.py), with the points per decade
in k as custom7 (default 200, the runs'): the high-omega sampling ladder of 2 November 2026.

    P(k) = A_s x^(n_s - 1 + (1/2) alpha_s ln x) [1 + A_osc sin(omega ln x + phi)],   x = k / k*,

with k* = 0.05/Mpc: R. Lumbye's template (runs/e2e_axion_tt15/generate_Pk.py) with the running of the
spectral index alpha_s = dn_s/dlnk added, in CLASS's convention (its analytic_Pk, primordial module).

CLASS++ runs this script at every evaluation (P_k_ini type external_Pk) from hpc_client/, and passes
custom1 to custom10 as arguments with %g (six significant digits); the last four are not used:

    custom1 = A_s, custom2 = n_s, custom3 = alpha_s, custom4 = A_osc, custom5 = omega, custom6 = phi

The k grid, 200 points per decade from 1e-6 to 1e7/Mpc, is R. Lumbye's: at omega = 50 one period of the
oscillation spans 0.055 decades, 11 points, and CLASS's spline in ln k is then good to about 1e-3 of the
oscillation's amplitude.
"""
import sys
from math import log, sin

try:
    A_s, n_s, alpha_s, A_osc, omega, phi = (float(v) for v in sys.argv[1:7])
    K_PER_DECADE = int(float(sys.argv[7])) if len(sys.argv) > 7 and float(sys.argv[7]) > 0 else 200
except ValueError:
    raise SystemExit("expected six numbers: A_s n_s alpha_s A_osc omega phi (custom1 to custom6)")

K_PIVOT = 0.05
K_MIN, K_MAX = 1.0e-6, 1.0e7

step = 10.0 ** (1.0 / K_PER_DECADE)
k = K_MIN
lines = []
while k <= K_MAX * step:
    lnx = log(k / K_PIVOT)
    power = A_s * (k / K_PIVOT) ** (n_s - 1.0 + 0.5 * alpha_s * lnx)
    lines.append("%.18g %.18g" % (k, power * (1.0 + A_osc * sin(omega * lnx + phi))))
    k *= step
sys.stdout.write("\n".join(lines) + "\n")
