#!/usr/bin/env python3
"""The proxy backgrounds at the HALF-DECADE rungs.

`isodata._bg_eps` reads `proxy_eps_ne_dncdm1` from these files, and `cells()` drops
any iso cell whose rung has none. The iso ladders in hpc_gal/hpc_gal2 run on half
decades; these backgrounds only ever ran on integer ones, so half the measured rate
bins have been silently outside the calibration. Same recipe as batch_bg.py -- only
the rung list differs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mk import run
PROXY = ("dncdm1.inverse_decays = yes\ndncdm1.dr_representation = proxy\n"
         "dncdm1.dr_N_q = 192\ndncdm1.dr_q_min = 1e-2\n")
# Rungs where an iso cell exists but no proxy background does.
RUNGS = {"0.3":  ["5.5", "6.5", "7.5", "8.5", "9.5", "10.5", "11.5"],
         "0.06": ["5.5", "6.5", "7.5", "8.5", "9.5"]}
for m, gs in RUNGS.items():
    for e in gs:
        # ⚠ NOT f"1e{e}".  CLASS parses the value with strtod, whose exponent must be an
        # INTEGER: "1e7.5" reads as 1e7 and the run is a silent duplicate of the decade
        # below -- identical eps_ne, no warning, no unused-parameter entry. Write the
        # mantissa out in full.
        G = repr(10.0 ** float(e))
        tag = f"bg_G1e{e}_m{m}"
        if os.path.exists(f"out/{tag}_background.dat"):
            print("skip", tag); continue
        run(tag, f"dncdm1.m = {m}\ndncdm1.Gamma = {G}\n" + PROXY +
                 "output = \nwrite background = yes\n")
