#!/usr/bin/env python3
"""Proxy (RTA) cells for the four-model note, all on ONE binary.

Why this directory exists rather than reusing `proxy_prod2`:

  * `proxy_prod2` never ran `dr_rta_form = copw`, which IS the Wong et al. model
    (Chen, Oldengott, Pierobon & Wong, arXiv:2203.09075 eq. 13). Its `mC`/`mD`
    cells are that paper's rate re-expressed through the `structured` machinery,
    which is close but not the paper: `copw` evaluates the published two-branch
    CurlyF, `structured` evaluates the exact Gamma(0,X) form.
  * `proxy_prod2` ran on a Sep-1 binary. Master has since taken #420 (the DR
    hierarchy closes at the Limber argument) and #424 (s_l sized by asking the
    species). Both touch the daughter hierarchy this note plots. Mixing a Sep-1
    proxy with a Sep-9 one is the round-mixing bug nb/README warns about, so
    every cell plotted in the note is regenerated here.

Base ini precedence, per rung:
  * the EXACT cell's own `_x_used.ini` where one exists (13 rungs) -- this is what
    makes "isolates the closure and nothing else" literally true;
  * else hpc_gal2's `_m6_used.ini`, which differs from the exact ini only in the
    two reduced-moment keys (asserted in check_bases()).

Only the root, the representation and the rate model are ever changed.
"""
import os, re, glob, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.environ.get("DNCDM_HARNESS") or os.path.dirname(HERE)   # the campaigns folder
EXACT = f"{HARNESS}/hpc_prod2/out"
M6 = f"{HARNESS}/hpc_gal2/out"
OUT = os.path.join(HERE, "out")
CLASS = os.environ.get("CLASS", "class")   # a class binary built from CLASS++ 85deb9fc
os.makedirs(OUT, exist_ok=True)

# ── the four-model comparison needs two proxy models; the rest is the UR test ──
# `ours`  the calibrated form the proxynote ships: structured shapes, alpha_l^eff,
#         beta_l = l(l+1)/2 - 2. C3/C5/n3 are written out rather than left to the
#         defaults so the ini is a complete record of the model.
# `copw`  the paper. ONE term, amplitude 1/12 with no free normalisation, analytic
#         quartic alpha_l. dr_rta_C3/n3 are ignored by this branch by construction.
OURS = ("dncdm1.dr_rta_form = structured\n"
        "dncdm1.dr_rta_alpha = integrated\n"
        "dncdm1.dr_rta_beta = legs\n"
        "dncdm1.dr_rta_C3 = 0.0743\n"
        "dncdm1.dr_rta_C5 = 0.5532\n"
        "dncdm1.dr_rta_n3 = 0.5\n")
COPW = ("dncdm1.dr_rta_form = copw\n"
        "dncdm1.dr_rta_C5 = 1\n"
        "dncdm1.dr_rta_alpha = quartic\n")

MODELS = {"ours": OURS, "copw": COPW}

# ── the UR fudge ─────────────────────────────────────────────────────────────
# Two probe amplitudes on `ours`, run only where an EXACT reference exists. The
# note fits the amplitude from the response kernel rather than scanning blindly;
# two probes are what it takes to check the response is linear over the range the
# fit lands in (dNur1 is the kernel, dNur2 is the linearity control).
UR_PROBES = {"ur1": 0.05, "ur2": 0.10}

# The companion probe, and the reason the UR test is not run alone. If the RTA
# residual is a rate-normalisation error rather than a missing free-streaming
# component, it should align with the response to C5 -- the NLO amplitude, the
# one number the calibration actually fits -- and not with the response to N_ur.
# Running both makes the UR question decidable instead of merely answerable:
# "the N_ur kernel does not fit" means little on its own; "the N_ur kernel does
# not fit and the C5 kernel does" is a result. Scaled by 1.5 for the same reason
# the UR probe is 0.05: large enough to sit above the run-to-run floor, small
# enough that the response is still linear.
C5_BASE = 0.5283
C5_PROBES = {"c5a": 1.5}

DROP = re.compile(r"^\s*(root|overwrite_root|dncdm1\.dr_reduced_moments"
                  r"|dncdm1\.dr_table_max_dlna|dncdm1\.dr_representation"
                  r"|dncdm1\.dr_rta_\w+)\s*=")


def rungs():
    """(G, mass, base_ini, has_exact) for every CMB rung on the ladder."""
    out = {}
    for f in sorted(glob.glob(f"{M6}/cmb_G*_m*_m6_used.ini")):
        m = re.match(r"cmb_(G[\d.]+)_m([\d.]+)_m6_used\.ini$", os.path.basename(f))
        if m:
            out[(m.group(1), m.group(2))] = [f, False]
    for f in sorted(glob.glob(f"{EXACT}/cmb_G*_m*_x_used.ini")):
        m = re.match(r"cmb_(G[\d.]+)_m([\d.]+)_x_used\.ini$", os.path.basename(f))
        if m:
            out.setdefault((m.group(1), m.group(2)), [None, False])
            out[(m.group(1), m.group(2))] = [f, True]
    return [(G, mass, base, ex) for (G, mass), (base, ex) in sorted(out.items())]


def body_of(path):
    return "".join(l for l in open(path)
                   if not DROP.match(l) and not l.startswith("#"))


def check_bases():
    """The exact and m6 inis must differ ONLY in the keys DROP removes.

    If they ever stop agreeing, basing some rungs on one and some on the other
    silently changes the cosmology halfway up the ladder. Fail loudly instead.
    """
    bad = []
    for f in sorted(glob.glob(f"{EXACT}/cmb_G*_m*_x_used.ini")):
        b = os.path.basename(f).replace("_x_used.ini", "_m6_used.ini")
        g = os.path.join(M6, b)
        if os.path.exists(g) and body_of(f) != body_of(g):
            bad.append(b)
    if bad:
        sys.exit("exact and m6 base inis disagree beyond the reduced keys: "
                 + ", ".join(bad))
    return True


def n_ur_of(body):
    m = re.search(r"^\s*N_ur\s*=\s*([\deE.+-]+)\s*$", body, re.M)
    if not m:
        sys.exit("no N_ur in the base ini -- the UR probe has nothing to shift")
    return float(m.group(1))


def build():
    check_bases()
    jobs = []
    for G, mass, base, has_exact in rungs():
        body = body_of(base)
        variants = dict(MODELS)
        if has_exact:
            base_nur = n_ur_of(body)
            for tag, d in UR_PROBES.items():
                variants[tag] = OURS + f"__NUR__{base_nur + d:.6f}\n"
            for tag, f in C5_PROBES.items():
                variants[tag] = OURS.replace(
                    f"dncdm1.dr_rta_C5 = {C5_BASE}",
                    f"dncdm1.dr_rta_C5 = {C5_BASE * f:.6f}")
        for v, extra in variants.items():
            tag = f"px_{v}_{G}_m{mass}"
            txt = body
            if "__NUR__" in extra:
                new = extra.split("__NUR__")[1].strip()
                txt = re.sub(r"^\s*N_ur\s*=.*$", f"N_ur = {new}", txt, flags=re.M)
                extra = extra.split("__NUR__")[0]
            ini = os.path.join(HERE, tag + ".ini")
            with open(ini, "w") as fh:
                fh.write(txt)
                fh.write("dncdm1.dr_representation = proxy\n")
                fh.write(extra)
                fh.write(f"root = {OUT}/{tag}_\noverwrite_root = yes\n")
            jobs.append((tag, ini))
    return jobs


if __name__ == "__main__":
    js = build()
    rs = rungs()
    print(f"{len(js)} jobs over {len(rs)} rungs "
          f"({sum(1 for r in rs if r[3])} with an exact reference)")
