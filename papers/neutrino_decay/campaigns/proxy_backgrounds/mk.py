#!/usr/bin/env python3
"""Generate + run CLASS inis for the paper's figures."""
import os, subprocess, sys, time, textwrap

CLASS = os.environ.get("CLASS", "class")   # a class binary; the README says which commit
HERE  = os.path.dirname(os.path.abspath(__file__))
OUT   = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

BASE = """h = 0.67
omega_b = 0.022032
omega_cdm = 0.12
N_ur = 1.0176
N_ncdm_standard = 0
T_cmb = 2.7255
Omega_k = 0.
tau_reio = 0.0925
n_s = 0.9619
A_s = 2.215e-9
dncdm1.type = ncdm_decay_dr
dncdm1.T = 0.71611
dncdm1.deg = 1
dncdm1.quadrature_strategy = 3
dncdm1.momenta_bins = 16
dncdm1.dr_f_ini_l = 1
dncdm1.dr_f_ini_phi = 0
evolver_background = 3
evolver_thermodynamics = 1

l_max_ncdm = 17
background_verbose = 1
"""

def run(tag, extra, timeout=100000):
    ini = os.path.join(HERE, f"{tag}.ini")
    with open(ini, "w") as f:
        f.write(BASE)
        f.write(f"root = {OUT}/{tag}_\noverwrite_root = yes\n")
        f.write(extra)
    t0 = time.time()
    log = os.path.join(OUT, f"{tag}.log")
    with open(log, "w") as lf:
        p = subprocess.run([CLASS, ini], stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
    dt = time.time() - t0
    ok = (p.returncode == 0)
    print(f"{'OK ' if ok else 'FAIL'} {tag:34s} {dt:8.1f}s", flush=True)
    if not ok:
        print(open(log).read()[-1500:])
    return ok, dt

if __name__ == "__main__":
    pass
