"""High-omega sampling ladder (2 November 2026): can CLASS++ 0a57bbc6 resolve the oscillation at 60 <= omega <= 200?
One setting per process, ONE thread (as a Grendel rank), so that times and peak memory are the setting's own.

    python -I ladder3.py BUILD_LIB SCRIPT {pp,wc} SETTING OUT.npz

pp: the PDP+ reference point of sampling_test/ladder2.py (28 October), lensed TT, TE, EE, phiphi to l = 2500, the runs'
    settings (l_logstep 1.026, l_linstep 8). A_osc 0 and A_osc 0.02 at omega 60, 80, 100, 130, 160, 200, phases 0, pi/2.
wc: the H0WC reference point of sampling_test/ladder_wc.py, the H0WC runs' settings (pre/act_dr6.pre lines,
    accurate_lensing, l_logstep 1.026, l_linstep 20), lensed spectra to l = 6400. A_osc 0 and 0.02 at omega 60, 100,
    150, 200, phase 0.
Each setting changes one thing at a time against the runs': the primordial table (points per decade in k, custom7 of
generate_Pk_kpd.py), the l sampling, the transfer wavenumbers q; REF and REFP are dense in all three (REFP twice as
dense as REF, to show that REF has converged). compare3.py compares them.
"""
import resource
import sys
import time

sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402
from classy import Class  # noqa: E402

SCRIPT, DATA, SETTING, OUT = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
PY = sys.executable
Q1 = {"q_linstep_trans": 0.05, "q_logstep_trans": 45.0}          # ladder2's REF transfer sampling
Q2 = {"q_linstep_trans": 0.025, "q_logstep_trans": 22.0}         # twice as dense
QM = {"q_linstep_trans": 0.15, "q_logstep_trans": 90.0}          # in between, a candidate for production
if DATA == "pp":
    ref = dict(omega_b=0.022298362999824215, omega_cdm=0.1177745466173623, logA=3.0413202231576277,
               n_s=0.9671203855355183, tau_reio=0.05655829026689818, theta100=1.0419870798777824)
    LMAX = 2500
    base = {"output": "tCl pCl lCl", "lensing": "yes", "l_max_scalars": LMAX, "non_linear": "halofit"}
    run_l = {"l_logstep": 1.026, "l_linstep": 8}
    settings = {
        "P": dict(run_l, kpd=200),                                   # the runs
        "PK": dict(run_l, kpd=1000),                                 # denser primordial table
        "PQ": dict(run_l, kpd=1000, **Q1),                           # + dense transfer q
        "L4K": dict(l_logstep=1.013, l_linstep=4, kpd=1000),         # denser l
        "L4Q": dict(l_logstep=1.013, l_linstep=4, kpd=1000, **QM),   # candidate
        "L4Q200": dict(l_logstep=1.013, l_linstep=4, kpd=200, **QM),  # the candidate with the runs' primordial table
        "L8Q": dict(l_logstep=1.013, l_linstep=8, kpd=1000, **QM),   # the candidate with the runs' linear l step
        "L8Q200": dict(l_logstep=1.013, l_linstep=8, kpd=200, **QM),
        "REF": dict(l_logstep=1.01, l_linstep=3, kpd=1000, **Q1),
        "REFP": dict(l_logstep=1.005, l_linstep=2, kpd=2000, **Q2),
        "PL": dict(run_l, kpd=200, accurate_lensing=1),             # the runs with the H0WC runs' accurate lensing
        "REFL": dict(l_logstep=1.01, l_linstep=3, kpd=1000, accurate_lensing=1, **Q1),
    }
    cases = [(0.0, 60.0, 0.0)] + [(0.02, w, p) for w in (60.0, 80.0, 100.0, 130.0, 160.0, 200.0) for p in (0.0, np.pi / 2)]
else:
    ref = dict(omega_b=0.02243026, omega_cdm=0.1175959, logA=3.051726, n_s=0.9723719, tau_reio=0.06208804,
               theta100=1.041807)
    LMAX = 6400
    ACT = dict(radiation_streaming_trigger_tau_over_tau_k=240, perturb_sampling_stepsize=0.05, delta_l_max=1800,
               accurate_lensing=1, q_logstep_limber=1.025)
    base = dict({"output": "tCl pCl lCl", "lensing": "yes", "l_max_scalars": LMAX, "non_linear": "halofit"}, **ACT)
    run_l = {"l_logstep": 1.026, "l_linstep": 20}
    settings = {
        "W": dict(run_l, kpd=200),                                   # the runs
        "WK": dict(run_l, kpd=1000),
        "WL": dict(l_logstep=1.013, l_linstep=8, kpd=1000),
        "WQ": dict(run_l, kpd=1000, **Q1),
        "WLQ": dict(l_logstep=1.013, l_linstep=8, kpd=1000, **QM),   # candidate (the PDP+ one with l_linstep 8)
        "WL200": dict(l_logstep=1.013, l_linstep=8, kpd=200),
        "WREF": dict(l_logstep=1.01, l_linstep=3, kpd=1000, **Q1),
    }
    cases = [(0.0, 60.0, 0.0)] + [(0.02, w, 0.0) for w in (60.0, 100.0, 150.0, 200.0)]
base.update({"omega_b": ref["omega_b"], "omega_cdm": ref["omega_cdm"], "tau_reio": ref["tau_reio"],
             "100*theta_s": ref["theta100"], "N_ur": 3.044, "N_ncdm": 0})
extra = dict(settings[SETTING])
kpd = extra.pop("kpd")


def ext(A, w, p):
    return {"P_k_ini type": "external_Pk", "command": f"{PY} {SCRIPT}",
            "custom1": 1e-10 * np.exp(ref["logA"]), "custom2": ref["n_s"], "custom3": 0.0,
            "custom4": A, "custom5": w, "custom6": p, "custom7": kpd}


out, times = {}, []
c = Class()                          # one object, as a sampler (CLASS++'s kept Bessel table)
for k, (A, w, p) in enumerate(cases):
    c.set(dict(base, **ext(A, w, p), **extra))
    t = time.perf_counter()
    c.compute()
    times.append(time.perf_counter() - t)
    cl = c.lensed_cl(LMAX)
    for s in ("tt", "te", "ee", "pp"):
        out[f"{k}_{s}"] = cl[s]
    c.struct_cleanup()
    c.empty()
    print(f"{DATA} {SETTING} case {k} (A {A}, omega {w:g}, phi {p:.2f}): {times[-1]:.1f} s", flush=True)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30        # bytes on macOS
out.update(times=np.array(times), rss_gb=rss, cases=np.array(cases), setting=SETTING, data=DATA)
np.savez(OUT, **out)
print(f"{DATA} {SETTING}: median {np.median(times):.1f} s per call, peak memory {rss:.2f} GB; written {OUT}", flush=True)
