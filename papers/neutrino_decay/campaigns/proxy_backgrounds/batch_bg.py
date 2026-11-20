#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mk import run
PROXY = ("dncdm1.inverse_decays = yes\ndncdm1.dr_representation = proxy\n"
         "dncdm1.dr_N_q = 192\ndncdm1.dr_q_min = 1e-2\n")
for m in ["0.3", "0.06"]:
    for e in range(5, 12):
        G = f"1e{e}"
        tag = f"bg_G{G}_m{m}"
        if os.path.exists(f"out/{tag}_background.dat"):
            print("skip", tag); continue
        run(tag, f"dncdm1.m = {m}\ndncdm1.Gamma = {G}\n" + PROXY +
                 "output = \nwrite background = yes\n")
