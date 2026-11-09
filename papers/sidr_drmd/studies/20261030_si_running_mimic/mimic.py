"""Can the neutrino self-interaction be mimicked by the primordial spectrum? (30 October 2026, Martin Sloth's comment)

At the H0WC+SH0ES best fit of the SInu+DRMD model (paper Sec. 5.5), compute the lensed CMB spectra with CLASS++
bfe6729e and the World Cup run's settings, once with the self-interaction (log10 G_eff of the best fit) and once
without it (log10 G_eff = -5.5, the prior boundary used for no self-interaction), everything else fixed. Then fit the
primordial spectrum of the case without self-interaction,

    ln P(k) = ln A_s + (n_s - 1) L + alpha_s L^2 / 2 + beta_s L^3 / 6,   L = ln(k / 0.05 Mpc^-1),

to the spectra with self-interaction, by Gauss-Newton on a Gaussian chi^2 of TT, TE and EE, and report what is left.

    OMP_NUM_THREADS=4 python -I mimic.py [--case best|strong] [--lmax 5000]

Weights (approximate, for orders of magnitude only):
  CV:   cosmic variance only, f_sky = 0.4, 30 <= ell <= 4000 (an idealised upper bound on distinguishability)
  P+A:  Planck-like for 30 <= ell < 1000 (f_sky 0.6, 33 / 60 muK-arcmin in T / P, beam 7 arcmin) and ACT-like for
        1000 <= ell <= 6500 (f_sky 0.4, 15 / 21 muK-arcmin, beam 1.4 arcmin)
Fits: "As,ns" / "As,ns,as" / "As,ns,as,bs" / "As,ns,as,bs,theta" (the last also shifts 100 theta_s).
"""
import argparse, json, os, sys, tempfile, time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.environ["CLASSPP_INSTALL"])  # the _install of a CLASS++ bfe6729e build
from classy import Class  # noqa: E402

HERE = Path(__file__).resolve().parent
KIT = Path(os.environ.get("CLIENT_PAPER_KIT", HERE.parents[1]))   # the cluster folder (by default this one)
BEST = HERE.parents[1] / "paper" / "plots" / "data" / "wc" / "best_fits.json"   # written by paper/build_wc.py
KPIV = 0.05
GEFF_OFF = -5.5

cfg = yaml.safe_load((KIT / "runs/sidr_drmd_wcmb_high/cobaya.yaml").read_text())
EXTRA = dict(cfg["theory"]["classy"]["extra_args"])
TMP = Path(tempfile.mkdtemp(prefix="pk_", dir=os.environ.get("MIMIC_TMP", tempfile.gettempdir())))
NCALL = [0]


def write_pk(p):
    k = np.logspace(-7, 3, 3000)
    L = np.log(k / KPIV)
    lnp = np.log(1e-10) + p["ln10A"] + (p["n_s"] - 1) * L + p["alpha_s"] * L**2 / 2 + p["beta_s"] * L**3 / 6
    f = TMP / f"pk_{NCALL[0]}.dat"
    np.savetxt(f, np.c_[k, np.exp(lnp)], fmt="%.10e")
    return f


def spectra(p, lmax):
    """Lensed TT, TE, EE and phi-phi (muK^2, ell 0..lmax), P(k, z=0) on a grid, 100 theta_s and H0."""
    NCALL[0] += 1
    d = dict(EXTRA)
    d.update({"omega_b": p["omega_b"], "omega_cdm": p["omega_cdm"], "tau_reio": p["tau_reio"],
              "nui1.m": p["m_int"] / 3, "nui1.log10G_eff": p["logGeff"], "delta_Neff_drmd": p["delta_Neff_drmd"],
              "f_idm_drmd": p["f_idm_drmd"], "z_stop": 10 ** p["log10_zstop"],
              "G_over_aH_drmd_ini": 10 ** p["log10_G_over_aH"],
              "P_k_ini type": "external_Pk", "command": f"cat {write_pk(p)}",
              "output": "tCl,pCl,lCl,mPk", "lensing": "yes", "l_max_scalars": lmax, "P_k_max_1/Mpc": 10.0,
              "z_pk": 0.0})
    if "theta_s_100" in p:
        d["100*theta_s"] = p["theta_s_100"]
    else:
        d["H0"] = p["H0"]
    M = Class()
    M.set(d)
    t = time.time()
    M.compute()
    cl = M.lensed_cl(lmax)
    T0 = M.T_cmb() * 1e6
    kg = np.logspace(-3, 0.5, 50)
    out = dict(tt=cl["tt"] * T0**2, te=cl["te"] * T0**2, ee=cl["ee"] * T0**2, pp=cl["pp"].copy(),
               pk=np.array([M.pk_lin(k, 0.0) for k in kg]), k=kg,
               theta=M.get_current_derived_parameters(["100*theta_s"])["100*theta_s"], H0=100 * M.h(),
               seconds=time.time() - t)
    M.struct_cleanup()
    M.empty()
    return out


def noise(ell, delta_uK_arcmin, beam_arcmin):
    s = np.radians(beam_arcmin / 60) / np.sqrt(8 * np.log(2))
    return (np.radians(delta_uK_arcmin / 60)) ** 2 * np.exp(ell * (ell + 1) * s**2)


def weights(ref, name, lmax):
    """Per-ell 3x3 inverse covariances of (TT, TE, EE) and the ells used."""
    ell = np.arange(lmax + 1)
    if name == "CV":
        use = (ell >= 30) & (ell <= min(4000, lmax))
        fsky, NT, NP = 0.4 * np.ones_like(ell, float), 0 * ell, 0 * ell
    else:
        use = (ell >= 30) & (ell <= min(6500, lmax))
        planck = ell < 1000
        fsky = np.where(planck, 0.6, 0.4)
        NT = np.where(planck, noise(ell, 33, 7.0), noise(ell, 15, 1.4))
        NP = np.where(planck, noise(ell, 60, 7.0), noise(ell, 21, 1.4))
    T, E, X = ref["tt"] + NT, ref["ee"] + NP, ref["te"]
    W = np.zeros((lmax + 1, 3, 3))
    for l in np.where(use)[0]:
        C = 2.0 / ((2 * l + 1) * fsky[l]) * np.array([
            [T[l] ** 2, T[l] * X[l], X[l] ** 2],
            [T[l] * X[l], (T[l] * E[l] + X[l] ** 2) / 2, E[l] * X[l]],
            [X[l] ** 2, E[l] * X[l], E[l] ** 2]])
        W[l] = np.linalg.inv(C)
    return W, use


def vec(s):
    return np.stack([s["tt"], s["te"], s["ee"]], axis=1)


def chi2(r, W, use):
    return float(np.einsum("li,lij,lj->", r[use], W[use], r[use]))


def per_ell(r, W):
    return np.einsum("li,lij,lj->l", r, W, r)


STEPS = dict(ln10A=0.005, n_s=0.004, alpha_s=0.004, beta_s=0.004, theta_s_100=0.0004)
FITS = {"As,ns": ["ln10A", "n_s"], "As,ns,as": ["ln10A", "n_s", "alpha_s"],
        "As,ns,as,bs": ["ln10A", "n_s", "alpha_s", "beta_s"],
        "As,ns,as,bs,theta": ["ln10A", "n_s", "alpha_s", "beta_s", "theta_s_100"]}


def jacobian(p0, s0, names, lmax):
    """d(TT, TE, EE)/d(parameter) at p0 by one-sided differences; the response to the primordial spectrum is smooth."""
    J = {}
    for n in names:
        q = dict(p0)
        q[n] += STEPS[n]
        J[n] = (vec(spectra(q, lmax)) - vec(s0)) / STEPS[n]
    return J


def fit(target, p0, s0, names, J, W, use, lmax, iters=4):
    """Gauss-Newton with the Jacobian of p0, the residual recomputed with CLASS++ at every step."""
    p, s = dict(p0), s0
    Jm = np.stack([J[n] for n in names], axis=2)               # (ell, 3, n)
    A = np.einsum("lia,lij,ljb->ab", Jm[use], W[use], Jm[use])
    hist = [chi2(vec(target) - vec(s), W, use)]
    for _ in range(iters):
        r = vec(target) - vec(s)
        dp = np.linalg.solve(A, np.einsum("lia,lij,lj->a", Jm[use], W[use], r[use]))
        q = dict(p)
        for n, v in zip(names, dp):
            q[n] += v
        sq = spectra(q, lmax)
        c = chi2(vec(target) - vec(sq), W, use)
        if c > hist[-1]:
            break
        p, s = q, sq
        hist.append(c)
        if hist[-2] - hist[-1] < 1e-3 * max(hist[-1], 1):
            break
    return p, s, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="best", choices=["best", "strong"])
    ap.add_argument("--lmax", type=int, default=5000)
    a = ap.parse_args()
    b = json.loads(BEST.read_text())["wcmb"]["values"]
    base = {n: b[n] for n in ("omega_b", "omega_cdm", "H0", "n_s", "tau_reio", "m_int", "logGeff", "delta_Neff_drmd",
                              "f_idm_drmd", "log10_zstop", "log10_G_over_aH")}
    base.update(ln10A=b["logA"], alpha_s=0.0, beta_s=0.0)
    if a.case == "strong":
        base["logGeff"] = -1.7
    on = spectra(base, a.lmax)
    print(f"[on]  log10 G_eff {base['logGeff']:.2f}: {on['seconds']:.1f} s, 100 theta_s {on['theta']:.6f}", flush=True)
    off0 = dict(base, logGeff=GEFF_OFF)
    s_off = spectra(off0, a.lmax)
    off_t = dict(off0, theta_s_100=s_off["theta"])
    off_t.pop("H0")
    prim = ["ln10A", "n_s", "alpha_s", "beta_s"]
    J = jacobian(off0, s_off, prim, a.lmax)
    J.update(jacobian(off_t, s_off, ["theta_s_100"], a.lmax))
    print(f"[off] no self-interaction: 100 theta_s {s_off['theta']:.6f} (with it {on['theta']:.6f}); "
          f"{NCALL[0]} calls so far", flush=True)
    res = {"case": a.case, "base": base, "fits": {}}
    bands = [(30, 1000), (1000, 2000), (2000, 3000), (3000, 6501)]
    for wname in ("CV", "P+A"):
        W, use = weights(on, wname, a.lmax)
        before = chi2(vec(on) - vec(s_off), W, use)
        pe0 = per_ell(vec(on) - vec(s_off), W)
        res["fits"][wname] = {"before": before, "before_bands": {f"{lo}-{hi - 1}": float(pe0[lo:hi][use[lo:hi]].sum())
                                                                 for lo, hi in bands if lo <= a.lmax}}
        print(f"[{wname}] no self-interaction, same parameters: chi2 = {before:.1f}", flush=True)
        for fname, names in FITS.items():
            p0 = off_t if "theta_s_100" in names else off0
            p, s, hist = fit(on, p0, s_off, names, J, W, use, a.lmax)
            pe = per_ell(vec(on) - vec(s), W)
            dpp = s["pp"][100:2001] / on["pp"][100:2001] - 1
            dpk = s["pk"] / on["pk"] - 1
            out = dict(chi2=hist[-1], hist=hist, shifts={n: p[n] - p0[n] for n in names},
                       values={n: p[n] for n in names}, H0=s["H0"], theta=s["theta"],
                       bands={f"{lo}-{hi - 1}": float(pe[lo:hi][use[lo:hi]].sum()) for lo, hi in bands if lo <= a.lmax},
                       pp_maxdev=float(np.max(np.abs(dpp))), pk_dev={f"{k:.3g}": float(v) for k, v in zip(s["k"], dpk)})
            res["fits"][wname][fname] = out
            print(f"[{wname}] fit {fname:18s}: chi2 {hist[0]:.1f} -> {hist[-1]:.2f} ({len(hist) - 1} steps); "
                  + ", ".join(f"d{n} {v:+.4f}" for n, v in out["shifts"].items())
                  + f"; max |dC_pp/C_pp| (100-2000) {out['pp_maxdev']:.3f}", flush=True)
            np.savez(HERE / f"spectra_{a.case}_{wname.replace('+', '')}_{fname.replace(',', '_')}.npz",
                     on_tt=on["tt"], on_te=on["te"], on_ee=on["ee"], on_pp=on["pp"], fit_tt=s["tt"], fit_te=s["te"],
                     fit_ee=s["ee"], fit_pp=s["pp"], off_tt=s_off["tt"], off_te=s_off["te"], off_ee=s_off["ee"],
                     off_pp=s_off["pp"], k=on["k"], on_pk=on["pk"], fit_pk=s["pk"], off_pk=s_off["pk"])
    res["calls"] = NCALL[0]
    (HERE / f"result_{a.case}.json").write_text(json.dumps(res, indent=1))
    print(f"{NCALL[0]} CLASS++ calls")


if __name__ == "__main__":
    main()
