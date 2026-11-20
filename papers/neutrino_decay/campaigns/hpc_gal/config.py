#!/usr/bin/env python3
"""THE single source of truth for the DNCDM PRODUCTION round.

Edit this file, re-run ./gen.py, resubmit.  Nothing else needs touching.
Stdlib only, so it runs on a login node with no modules loaded.

════════════════════════════════════════════════════════════════════════════════
WHAT THIS ROUND IS

The final production run for the inverse-decay paper, on MERGED MASTER, after
PR #387.  Three families, one config:

  A  cmb   C_l and P(k) ladders, exact and Galerkin.  The published ratio is
           C_l(Gamma) / C_l(Gamma = 1e5), so the lowest rung is the denominator
           of every figure and is not optional.
  B  iso   a4PiDot -> the isotropisation rate Gamma_T, for fig 5 and the fit.
           Solves ONLY the k_output modes, so it is ~1/15 of a CMB cell and can
           be pushed two decades further in Gamma.
  C  gal   The Galerkin closure error: a moment ladder (4/6/8) against the exact
           scheme, in the Gamma band where both are affordable.

  D  proxy DEFERRED.  gen.py writes runfile_proxy and nothing submits it.  The
           RTA transport rate has to be refitted from family B FIRST, because
           the published fit is what the proxy cells would be testing.

════════════════════════════════════════════════════════════════════════════════
⚠ THE OLD hpc_* OUTPUTS ARE NOT MIXABLE WITH THESE

Merged master's DNCDM knob set is SMALLER than the campaign branches':
`lump_limiter`, `deflate_collision`, `dr_reduced_weight`, `dr_reduced_table` and
`dr_reduced_table_stride` no longer exist -- the behaviours they selected are now
hardwired (limiter on, no deflation, table always used).  A pre-merge ini runs on
master with rc = 0 and means something different, because CLASS SILENTLY IGNORES
KEYS IT DOES NOT RECOGNISE: no warning, no error, not even at input_verbose = 2.

Two things follow, and both are enforced rather than documented:
  * every ini carries `write warnings = yes`, and runcell.sh FAILS the cell if
    CLASS reports an unread key (guard 3).  That catches typos too, which a
    hand-maintained blacklist never would.
  * runcell.sh refuses a binary that still contains the removed keys (guard 1)
    or that was not built from EXPECTED_COMMIT (guard 2).

The old cells stay useful as cross-checks.  They must not appear in a figure
beside these ones.
"""

# ── provenance ───────────────────────────────────────────────────────────────
# The commit build.sh must have built from.  runcell.sh compares BUILD_COMMIT
# against this and refuses to run on a mismatch.  Bump deliberately, never to
# silence the guard.
EXPECTED_COMMIT = "FILL_ME_AFTER_THE_KTABLE_FIX"   # paper branch: the logs of this round record no commit (see the README of papers/neutrino_decay)
# ⚠ DELIBERATE PLACEHOLDER.  This round exists BECAUSE of a code change that is not
# committed yet (the Gamma-aware kTableMaxDlna, see README §1).  Guard 1 in runcell.sh
# compares BUILD_COMMIT against this and refuses to start, which is exactly what should
# happen: every cell here is meaningless on a binary with the fixed 7e-3 constant.
# ⚠ 6bc5f66e is the HEAD of `fix-reduced-table-lna-density` (PR #411), not a
# master commit -- #411 was not merged when this round was built.  Two things
# depend on it and neither is optional:
#   * the reduced collision table no longer truncates its own damping.  Without
#     it EVERY Galerkin cell diverges: hpc_prod came back with 153 of 335
#     transfer outputs blown, |phi| to 1e75, and family C was void.
#   * back_integration_stepsize is back at 7e-3.  #397 had moved it to 0.07,
#     which is what triggered the above.
# BUMP THIS to the squash-merge hash the moment #411 lands, and rebuild.  A
# binary from plain master will pass guards 2-4 and produce garbage in family C.

# ── cosmology ────────────────────────────────────────────────────────────────
COSMOLOGY = {
    "T_cmb": "2.7255", "omega_b": "0.022032", "omega_cdm": "0.12", "h": "0.67",
}
# CLASS defaults N_ur = 3.046 and does NOT subtract a species when an NCDM is
# added, so the naive fiducial runs at N_eff = 5.07 -> 5.54 (three massless
# neutrinos PLUS the parent PLUS its daughter).  1.0176 keeps the three-species
# budget with the decaying sector counted once.
N_UR = "1.0176"

# ── the decaying species ─────────────────────────────────────────────────────
# ONLY keys that exist on merged master.  Anything else is caught by guard 3.
PARENT_T = "0.71611"
Q_MIN, Q_MAX = "1e-2", "50"
DECADES = 3.699              # log10(Q_MAX / Q_MIN)
QUAD_STRATEGY = 3            # trapezoid; Gauss-Laguerre is 2x for nothing
L_MAX_NCDM = 17              # psi/phi/delta_cdm at 17 are within 1e-4 of 35

# dr_q_max = 50 is kept from the earlier campaigns, and it is defensible ONLY
# because this ladder starts at Gamma = 1e5.  CLASS warns whenever dr_q_max sits
# below q*(a = 1), which it does here -- but that check is unconditional in a,
# while the leak needs the parent to still be DECAYING when a*M/2 outgrows the
# grid.  At Gamma = 1e5 the parent is extinct by a ~ 2.5e-3, where q* ~ 19, so
# nothing is clamped.  collect.py --bg re-checks it per cell against the measured
# a_ext instead of trusting this paragraph, and runcell.sh records the warning.
EVOLVER_BACKGROUND = 3       # ndf15 CANNOT integrate this background above ~1e6
EVOLVER_THERMO = 1
EVOLVER_PERTURBATIONS = 3    # ETD ahead at every Gamma tested; step count is
                             # Gamma-independent where rkdp45's is not

# ⚠ 3e-6, NOT the 3e-8 the hpc_reduced / hpc_nodefl / hpc_lna / hpc_freeze
# diagnostics used.  Those matrices were resolving 2.5e-5 differences between
# variants; a production cell is not.  The difference is a measured factor of
# 4.2-5.2x in cost (exact, m = 0.3: 24.2 core-h at 3e-6 against 100.6 at 3e-8 for
# Gamma = 1e8; 212.5 against 967.8 at 1e9), which is why the cost tables below
# are all normalised to 3e-6.
TOL_PERTURB = "3e-6"

# ── the k sampling: measured locally, see tune/ ──────────────────────────────
# 113-121 k modes against the CLASS++ defaults' 297, at <= 1% absolute error in
# the lensed spectra over 2 <= l <= 2000 and in P(k), verified at BOTH the
# N_eff = 5.1 fiducial and N_ur = 1.0176.  tune/ktune.py has the scan and the
# reasoning; the three things worth repeating here:
#
#  * WAVES, NOT MODES, ARE THE COST.  perturbations_module.cpp fans the k loop
#    out over a task pool, so a cell costs ceil(n_k / cpus) waves.  At 64 cores
#    129 modes and 192 modes cost the SAME wall time.  This preset is the most
#    accurate one that still fits TWO waves -- which is why the goal was 128 and
#    not "as few as possible".
#  * k_step_super is TIGHTER than the default, not looser.  precision_fast.pre's
#    0.02 puts C_l^EE at l < 30 out by 43%; even 0.012 gives 10%.  It buys margin
#    at low k so that k_step_transition can spend it.
#  * the k TAIL above the acoustic grid is nearly free to cut.  Requesting lCl
#    switches on the full-Limber lensing grid, which extends the SOURCE k range
#    to k_max_limber_over_l_max_scalars * l_max_scalars = 1.25/Mpc by default --
#    5x past k_max_cmb, and the most expensive modes in the run.  Cutting it also
#    stops P(k) at ~0.37 h/Mpc, i.e. exactly the range the CMB needs anyway.
PRECISION = {
    # acoustic grid
    "k_min_tau0": "0.6",
    "k_max_tau0_over_l_max": "1.3",
    "k_step_super": "0.002",
    "k_step_sub": "0.13",
    "k_step_transition": "0.6",
    # the tail above k_max_cmb -- measured free (see tune/)
    "k_per_decade_for_pk": "3",
    "k_per_decade_for_bao": "10",
    "k_max_limber_over_l_max_scalars": "0.0001",
}
# MEASURED (tune/ktune.py --stage robust, and tune/dncdm_spotcheck.py):
#
#                        n_k     TT      EE      TE   EE(l<30)   P(k)
#   LambdaCDM N_eff 5.1   113   0.26%   0.48%   0.43%   0.48%    0.39%
#   LambdaCDM N_ur 1.0176 121   0.31%   0.29%   0.60%   0.29%    0.29%
#   DNCDM Gamma=1e5*      126   0.20%   0.58%   0.95%   0.58%    0.34%
#   DNCDM Gamma=1e6*      126   0.20%   0.59%   0.96%   0.59%    0.34%
#
#   * the DNCDM rows were measured with the pre-tail-cut acoustic preset, which
#     was 0.80% worst-case in LambdaCDM against this one's 0.60%.  They are the
#     conservative reading, and their agreement with the LambdaCDM rows is the
#     evidence that LambdaCDM was the right place to tune the k grid at all.
#
# AND THE NUMBER THAT ACTUALLY MATTERS.  The published quantity is a RATIO, in
# which the common-mode k error cancels:
#
#   max error in C_l(1e6)/C_l(1e5), l = 2..2000:   TT 0.06%,  EE 0.02%
#
# i.e. 10-30x below the absolute error the 1% gate was written against.
# l_max_scalars and delta_l_max keep their DEFAULTS.  They are worth 13 modes out
# of 297, and cutting them alongside k_max_tau0_over_l_max is what put an
# irreducible 1.2% floor under TT in the first combined scan: k_max is the
# PRODUCT k_max_tau0_over_l_max * l_max_scalars, so the two cuts compound.
L_MAX_SCALARS = 2500         # publishes clean lensed C_l to l = 2000
DELTA_L_MAX = 500            # CLASS++ default; the lensing headroom
LENSING = "yes"
L_PUBLISH = 2000             # what the figures actually use

# n_k the cost model assumes.  MEASURED: 123 at N_eff = 5.1, 128 at N_ur =
# 1.0176, 126 for an actual DNCDM cell at Gamma = 1e5.  Whatever a cell really
# solves is recorded as N_K in its log, so collect.py can check this.
N_K_CMB = 121

# ── the daughter momentum grid ───────────────────────────────────────────────
# dr_q_min, dr_q_max and dr_N_q are ONE knob: only bins per decade matters, to
# four digits, across a 100x change in the window.  TWO requirements, and the
# binding one is the PERTURBATION grid.
#
# BACKGROUND, under the (now hardwired) lump limiter, against Ecomov's converged
# limit: Gamma = 1e7 -> ~19 bins/decade for 1%, 1e8 -> ~28, 1e9 -> ~50, i.e.
# Gamma^0.25.  Without the limiter it was 45 / 135 / 430 -- a discretisation
# ratchet, not a resolution requirement.
BPD_BG_AT_1E8, BPD_BG_INDEX = 28.0, 0.25
# PERTURBATIONS: genuine resolution of delta-f(q), which the limiter never
# touched.  ⚠ THE WEAKEST NUMBER IN THIS FILE -- one convergence ladder at one
# Gamma, extrapolated by an index measured on a different scheme.  The refine = 2
# twin in every family is what checks it.
BPD_PT_AT_1E7, BPD_PT_INDEX = 27.0, 0.17
# A lighter parent goes non-relativistic later, decays later, banks more
# collisions and carries a bigger residual ratchet at the same bins/decade.
# MEASURED at one mass pair, so a lookup and not a law: an unlisted mass takes
# the LARGEST factor, because under-resolving is the failure that looks fine.
BPD_MASS_FACTOR = {"0.3": 1.0, "0.06": 2.0}
N_BG_FLOOR = 41
N_BG_CAP = 6401              # rows that hit it are flagged bg_unconverged
GRID_STEP = 2                # (dr_N_q - 1) divisible by this, so refine = 2 is legal

# ── momenta_bins: DIFFERENT for family B, deliberately ───────────────────────
# momenta_bins scales the NODE count, hence ~linear in runtime.
#   16 for C_l and P(k): psi 4.8e-4 and delta_cdm 1.0e-5 against a finer
#      reference, and it is a 1.85x per-cell speed-up.
#   32 for a4Pi: the a4Pi wiggle is O(h^2) in the PARENT grid -- 1.449 / 0.374 /
#      0.080 at 16 / 32 / 64 -- and family B's whole deliverable is a projection
#      onto a4Pi.  Family B carries a 16/32/64 rung at one Gamma to show the
#      fitted exponent does not move.
# This makes A and B non-comparable cell-for-cell.  That is accepted: they answer
# different questions, and the Galerkin validation lives entirely inside A/C.
MOMENTA_BINS_CMB = 16
MOMENTA_BINS_ISO = 64

# ⚠ 64, NOT 32.  hpc_prod2 ran the rung it prescribed -- iso_G09.0_m0.3_x at
# momenta_bins 16 / 32 / 64, exact scheme -- and p MOVED, which its own comment said
# would be "the finding, not a footnote".  Measured (collect.py's estimator, median
# over 7 k, gamma in [2,gate]):
#
#     gate    mb=16   mb=32   mb=64   64-16   64-32
#       15     3.93    3.65    3.55   -0.38   -0.10
#       20     3.90    3.60    3.48   -0.43   -0.13
#
# 32 -> 64 sits AT or above the round's own dp > 0.10 "moved" threshold and the
# sequence is still monotonically decreasing (successive differences 0.28 -> 0.10,
# ratio 0.36, so the residual at 64 is ~0.06 and p_inf ~ 3.49).  A p published from a
# mb = 32 ladder is biased HIGH by ~0.10-0.15, in the same direction a coarse grid
# always fakes.  32 was chosen from the a4Pi WIGGLE amplitude, never from p.
#
# Cost: MOMENTA_COST_FACTOR puts 64 at 3.4 against 32's 1.85, i.e. 1.84x the family-B
# ladder.  That is the price of the deliverable being right.
#
# The rung that now justifies 64: run one cell at 128 and show p has stopped moving.
# Entries are (log10Gamma, mass, [momenta_bins ...]).
MOMENTA_RUNG = [("09.0", "0.3", [128])]

# ── the doubled daughter grid: a RESONANCE, not a rising requirement ─────────
# BPD_PT_AT_1E7 = 27 with index 0.17 assumes the requirement is a power law in
# Gamma.  hpc_prod's finished exact ladder says it is not.  C_2^TT (lensed) is
# Gamma-INDEPENDENT in this model -- measured, m = 0.06 reads 1.4860 / 1.4860 /
# 1.4859 e-10 at Gamma = 1e7 / 1e8 / 1e9 and m = 0.3 reads 1.4812 / 1.4834 /
# 1.4811 at 1e5 / 1e9 / 1e10 -- so any departure from 1.481e-10 is grid error,
# and the ladder measures the requirement directly:
#
#   m = 0.3      log10G   6.5   12.7   18.7   27.3   40.3   59.2   88.9 bins/dec
#                   5.0  +0.1%  +0.0%
#                   6.0        -10.5%  +5.1%
#                   7.0               -41.1% (x)         <- 13.8 bpd is +666%
#                   8.0                      -3.0%  +1.5%
#                   9.0                             -2.4%  +0.2%
#                  10.0                                    -4.8%  +0.0%
#
#   m = 0.06     log10G  12.7   18.7   25.1   27.3   37.0   54.3   80.3 bins/dec
#                   5.0 +34.8%        -8.6%
#                   6.0        -12.3%              +14.9%
#                   7.0                      +0.2%        +0.3%
#                   8.0                                   +0.1%  +0.3%
#
# ⚠ THE REQUIREMENT IS NON-MONOTONIC IN GAMMA, so it is NOT a power law of any
# index or prefactor.  The decisive pair needs no modelling: at m = 0.3,
# 27.3 bins/decade is -41.1% at Gamma = 1e7 while 20.3 bins/decade is only -3.0%
# at Gamma = 1e8.  FEWER bins at HIGHER Gamma is better.  A power law cannot do
# that.  Same at m = 0.06: 37.0 bpd is +14.9% at 1e6 while 27.3 bpd is +0.2% at
# 1e7.
#
# So the requirement has a PEAK -- near Gamma ~ 1e7 at m = 0.3 and ~1e6 at
# m = 0.06, i.e. it moves DOWN by ~10x for a 5x lighter parent -- and away from
# it the production rule has roughly 2x of margin (every `x` cell above the peak
# is within 0.3%, every `xc` twin at half the grid is 2-5% off).
#
# ⚠ AND THAT IS WHY THE LOWEST RUNG IS FINE.  Gamma = 1e5, m = 0.3 is clean at
# SIX AND A HALF bins/decade (+0.1%).  It is not that a slow decay needs few
# bins; it is that the lowest rung sits BELOW the resonance.  m = 0.06 has no
# such rung because its peak is at or below where its ladder starts -- which is
# also why its lowest two rungs are the contaminated ones there.
#
# THE MECHANISM IS NOT ESTABLISHED.  The standing hypothesis is the spurious
# ell = 1 eigenvalue of the lumped loss, lambda ~ Gamma * N_q^-3.6, integrated
# over the collision window -- lambda rises with Gamma while the window shrinks,
# which does produce a peak, and the peak should then sit near Gamma ~ H(a_nr),
# which moves the right way with mass since a_nr ~ 1/m.  But H(a_nr) comes out
# ~20x below the observed peak at both masses, so this is a story that gets the
# SHAPE and the mass ORDERING right and the location wrong.  Treat it as a
# hypothesis with a cheap test, which is what GRID_RUNG now is.
#
# Not raised globally: dr_N_q doubles the ODE state, so every exact cell doubles
# and the reach goes with it -- and above the peak it would buy nothing at all.
#
# Above ell ~ 30 none of this matters: the Gamma = 1e7 pair agrees to 0.16% over
# ell = 200-800.  The damage is confined to ell <~ 20 and k <~ 1e-3.
#
# Entries are (log10Gamma, mass, bins-per-decade multiplier).  Family cmb,
# scheme x, refine 1; tagged `_q<mult>x` so they never collide with the ladder.
GRID_RUNG = []
# ⚠ EMPTIED ON PURPOSE.  hpc_prod2 carried six `_q2x` cells to chase a non-monotonic
# "daughter-grid resonance" peaking near Gamma ~ 1e7.  That resonance was ROOT-CAUSED on
# 2026-08-28 and it was never the daughter grid: it was the GLOBAL
# back_integration_stepsize at #397's 0.07, which PR #411 restored to 7e-3.  Isolated on
# ONE binary, one ini, that one key changed, at Gamma=1e7 / m=0.3 / 27.3 bins/decade --
# the exact cell hpc_prod published -41.1% for:
#
#     back_integration_stepsize = 0.07   C_2^TT = 8.5406e-11   -42.3%
#     back_integration_stepsize = 7e-3   C_2^TT = 1.4844e-10    +0.2%
#
# hpc_prod2 and hpc_ratchet both already run at 7e-3 and agree with each other to the
# digit, and their `c` twins (HALF the production perturbation grid) are on target on all
# 18 rungs.  There is no resonance left to probe, and the six cells would only re-measure
# a fixed bug.
#
# ⚠ The same twins say BPD_PT_AT_1E7 = 27 is ~2x CONSERVATIVE -- worth a median 48% of
# wall on an exact cell (n=21).  NOT taken here: this round has exactly one exact cell and
# it exists to anchor the hpc_prod2 ladder, so changing its grid rule would make it
# non-comparable with the cells it is meant to anchor.  It is the obvious saving for the
# round after this one.
# ⚠ NOT INCLUDED, deliberately: a doubled cell at Gamma = 1e11, m = 0.3.  It was
# in the plan on the reasoning that the paper's low-ell claim rests on
# cmb_G11.0_m0.3_x, and the finished ladder has since answered it for free -- the
# `x` cells at 1e8, 1e9 and 1e10 read +1.5%, +0.2% and +0.0%, so the requirement
# tracks the production rule with margin everywhere above the peak.  The -5.1%
# at 1e11 is its `xc` twin, and every xc twin on that ladder reads -2.4 to -5.1%.
# The cell would have cost of order 10 000 core-hours -- more than doubling the
# round -- to re-measure something three rungs already agree on.

# ── the Gamma ceiling: it is the BACKGROUND, and it is serial ────────────────
# hpc_prod measured the background pre-flight, one cell per (Gamma, mass), and
# it is the thing that ends the ladder -- not the perturbations, and not the
# budget:
#
#     m = 0.3     1e9 0.04 h   1e10 0.25 h   1e11 1.71 h   1e11.5 4.89 h
#                 1e12 13.89 h
#     m = 0.06    1e9 0.42 h   1e9.5 1.02 h  1e10 2.54 h   1e10.5 KILLED at the
#                 4 h bg wall (extrapolates to ~6.3 h), and nothing above it ran
#
# ~2.8x per half decade at both masses.  THE BACKGROUND IS SERIAL (measured: not
# stiffness-limited but accuracy-limited, and threading it does not pay), so on a
# 64-core family-A allocation those 13.89 h are 889 CHARGED core-hours of one
# busy core -- and EVERY scheme pays them, exact and Galerkin alike, because the
# background solve is identical across schemes and across a refine pair.
#
# That is also why hpc_prod's cost model was wrong in the direction it was: it
# modelled only the k-loop, so every Galerkin cell came in 5-15x over estimate.
# Checked against the bg table, those cells are ~100% BACKGROUND -- e.g.
# cmb_G11.0_m0.3_m6 measured 1.5 h against bg_G11.0_m0.3's 1.71 h.  The reduction
# does what it claims; there is simply nothing left to reduce.  COST_BG below
# fixes the model.
GAMMA_CEILING = {"0.3": 12.0, "0.06": 10.0}

# ── the Gamma ladders ────────────────────────────────────────────────────────
# Labelled by log10(Gamma) so the tags sort numerically as strings.
def _dec(lo, hi):            # full decades
    return [f"{g:04.1f}" for g in [lo + i for i in range(int(hi - lo) + 1)]]


def _half(lo, hi):           # half decades
    n = int(round((hi - lo) * 2)) + 1
    return [f"{lo + 0.5 * i:04.1f}" for i in range(n)]


def _cap(mass, rungs):
    """Clip a ladder at GAMMA_CEILING[mass].  Applied to every ladder rather than
    hand-editing the ranges, so the ceiling is stated once and cannot drift."""
    top = GAMMA_CEILING[mass]
    return [g for g in rungs if float(g) <= top + 1e-9]


# Family A.  EXACT on full decades -- those are the expensive anchors -- and
# Galerkin on half decades, so the dense published ladder comes from the
# reduction with an exact anchor every decade.  The reach is set by the cost
# model below and re-checked by gen.py against TARGET_WALL_H.
LADDER = {
    # ── WHAT THIS ROUND REDOES, AND WHAT IT KEEPS ───────────────────────────────
    # KEPT from hpc_prod2, not rerun: every EXACT (`x`) cell.  The exact scheme never
    # touches the reduced collision table, so the defect this round exists to fix
    # cannot reach it -- verified, at Gamma = 1e10 / m = 0.3 the exact cell reads +0.0%
    # against the Gamma-independent C_2 target while its m6 twin reads -2.4%.
    #
    # REDONE: every Galerkin cell, at every Gamma.  Not only the visibly broken ones:
    # the table spacing was wrong everywhere, it just did not SHOW below ~1e9.5, and a
    # ladder half of whose rungs came off a different discretisation is not a ladder.
    # They are cheap -- an m6 cell is ~100% its serial background, which is why
    # FAMILIES gives family A 16 cores here and not 64.
    "cmb": {
        # ONE new exact anchor, and it is the scientifically load-bearing cell of the
        # round.  Above Gamma = 1e10 (m=0.3) the Galerkin ladder has NO exact twin, so
        # nothing there is validated against anything -- and that is exactly the range
        # where the old table failed (-7.9% at 1e10.5, -30% at 1e11, runaway by 1e12).
        # Refitting the table and re-running m6 alone would replace an unvalidated
        # number with a different unvalidated number.
        "x":  {"0.3": ["10.5"], "0.06": []},
        "m6": {"0.3": _cap("0.3", _half(5, 13)), "0.06": _cap("0.06", _half(5, 12))},
    },
    # Family B.  Same reasoning, and it carries the momenta_bins = 64 correction as
    # well, so every one of these cells changes for TWO independent reasons.
    "iso": {
        # "09.0" is the PARTNER for the momenta_bins = 128 rung: MOMENTA_RUNG runs the
        # SAME cell at 128, and a convergence rung with nothing to converge against is
        # not a rung.  hpc_prod2 got this for free because its iso x ladder was full.
        "x":  {"0.3": ["09.0", "10.5"], "0.06": []},
        "m6": {"0.3": _cap("0.3", _half(5, 13)), "0.06": _cap("0.06", _half(5, 13))},
    },
    # Family C.  The closure arm reads the tabulated operator too, so it is as exposed
    # as family A and is redone unchanged in shape.
    "gal": {
        "m4": {"0.3": _cap("0.3", _dec(8, 11)), "0.06": _cap("0.06", _dec(8, 10))},
        "m8": {"0.3": _cap("0.3", _dec(8, 11)), "0.06": _cap("0.06", _dec(8, 10))},
    },
}

# ── the Gamma-aware table spacing: the code change this round is built on ────
# `kTableMaxDlna` was a fixed 7e-3 -- identical to back_integration_stepsize, so
# n_sub = ceil(dlna_row/kTableMaxDlna - slack) = 1 and PR #411's sub-division was
# INERT at the shipped default.  The reduced table's ln a spacing was therefore just
# the background table's, at EVERY Gamma, while everything else in the chain scales.
#
# MEASURED (see README §1), cells cmb_G10.5_m0.3_m6 and cmb_G11.0_m0.3_m6:
#
#     Gamma      n_sub=1     n_sub=10    n_sub=20
#     3.16e10     -7.9%        -0.1%         --
#     1e11       -30.1%        -0.3%       -0.1%
#
# Holding 0.1% needs n_sub = 10 at 3.16e10 and 20 at 1e11, i.e. Gamma^0.60.  Cross-check:
# at FIXED spacing the error grows 7.9% -> 30.1% over Gamma x 3.16, i.e. ~Gamma^1.16, and
# linear interpolation has error ~ (h/L)^2, giving L ~ Gamma^-0.58.  The integrator's own
# fixed-grid step-count exponent is 0.662 (11 ladders, hpc_ratchet `ceil`), so it is a
# safe upper bound on this one.
KTABLE_AT_1E9, KTABLE_INDEX = 7e-3, 0.60

# ⚠ WHAT THE CEILING COSTS THE ROUND, stated plainly so it is a decision and not
# an accident.  The Galerkin ladders lose their top: cmb/m6 m=0.3 stops at 1e12
# instead of 1e13 and m=0.06 at 1e10 instead of 1e12; iso/m6 the same.  Those
# rungs were the reduction's whole reach argument -- "the overlap earns it the
# right to carry the fit past where the exact scheme stops".  It still does, from
# 1e11 (exact, m=0.3) to 1e12, but the lever arm is one decade, not two.
#
# They are not dropped for budget.  They are dropped because their BACKGROUND
# does not finish: nothing above 1e12 (m=0.3) or 1e10 (m=0.06) has ever produced
# a background table on this cluster, and a Galerkin cell cannot start without
# one.  Raising the ceiling means making the background cheaper first, which is
# a code question (dr_N_q is the lever; see the notes on dr_bg_refine), not a
# scheduling one.

# refine = 1 is the production cell; refine = 2 runs the SAME background under a
# HALVED perturbation grid, so the pair brackets the answer.  It is the paper's
# own convergence gate, and it is CHEAPER than production, not dearer.
#
# ⚠ NON-NEGOTIABLE IN FAMILY B.  Coarsening the perturbation grid does not add
# scatter to the isotropisation exponent -- it FAKES A STEEPER ONE.  Going to a
# 4x coarser grid moves the fitted p by +0.96 on average and an 8x coarser one by
# +4.9, so any under-resolved run looks as though it has found gamma^-5, which is
# precisely the result this round exists to test.
REFINES = {"cmb": [1, 2], "iso": [1, 2], "gal": [1]}

# ── k_output ─────────────────────────────────────────────────────────────────
# Family B solves ONLY these plus k_min.  `output = mTk` with P_k_max below k_min
# collapses the k list to a single mode, into which k_output_values are then
# inserted -- MEASURED: 3 requested values give exactly 4 modes.
#
# Seven values, because wallclock is set by the SLOWEST SINGLE MODE when there is
# one mode per core, so every mode up to the core count is free in wall time and
# buys another gamma-bin for the fit.  8 modes, 8 cores.
K_OUTPUT_ISO = ["1e-3", "2.154e-3", "4.642e-3", "1e-2",
                "2.154e-2", "4.642e-2", "1e-1"]
N_K_ISO = len(K_OUTPUT_ISO) + 1          # + k_min; MUST equal cpus for family B
P_K_MAX_ISO = "1e-6"                     # below k_min: collapses the base grid

# Family A carries NO k_output.  The dumps are 0.7-1.4 GB gzipped per cell at
# high Gamma and family B owns that data; adding them here would also add 3 modes
# to every CMB cell for nothing.
Z_PK = "0,49,99"

# ── Slurm ────────────────────────────────────────────────────────────────────
# ⚠ q192 DOES NOT EXIST.  Every earlier hpc_* config carried a q192 row; it was a
# typo for q144.  Removed here and in build.sh.
PARTITION_MARCH = {
    "q20":  "ivybridge",     "q24":  "haswell",         "q28":  "broadwell",
    "q36":  "skylake-avx512", "q40": "cascadelake",     "q48":  "cascadelake",
    "q64":  "icelake-server", "q128": "znver4",         "q144": "znver5",
}

# Wall limit is 15 days = 360 h on every partition; 359 leaves a margin against
# being killed at the boundary.  TARGET_WALL_H is what the MATRIX is sized to, so
# there are ~5 days of headroom on the largest cell.
TIME_LIMIT = "359:00:00"
# What the MATRIX is sized to, per family.  Not one number, because the families
# have different risk profiles:
#   cmb  240 h leaves ~5 days of headroom on a cell that costs 64 cores for the
#        whole time.  Being wrong about the frontier cell here is expensive.
#   iso  330 h.  These cells hold 8 cores, so even the largest is ~2600 charged
#        core-hours -- 5% of one CMB frontier cell -- and they are what
#        establishes the exact/Galerkin OVERLAP.  Without the overlap the
#        reduction has no warrant to carry the fit past Gamma = 1e11, which is
#        the whole point of the round.  Cheap to risk, expensive to lose.
#   gal  240 h; nothing in it comes close anyway.
#   bg   24 h.  hpc_prod ran the bg family at 4 h and bg_G10.5_m0.06 was KILLED
#        at the wall (it extrapolates to ~6.3 h) -- which is why nothing above
#        it has a background and why GAMMA_CEILING is where it is.  The ceiling's
#        own top cell, m = 0.3 at 1e12, measured 13.89 h.
TARGET_WALL_H = {"cmb": 240.0, "iso": 330.0, "gal": 240.0, "bg": 24.0}

FAMILIES = {
    # partition, cpus, mem, time, concurrency
    # ⚠ 16 CORES, NOT 64.  hpc_prod2's family A was k-loop bound and a whole q64 node
    # was the right shape for it.  THIS round's family A is almost entirely Galerkin,
    # and an m6 cell is ~100% its SERIAL background -- hpc_prod2 measured
    # cmb_G11.0_m0.3_m6 at 1.5 h against bg_G11.0_m0.3's 1.71 h.  A serial solve on 64
    # cores is charged 64x for cores that are asleep; hpc_ratchet made exactly this
    # correction and it was the single largest line item it removed.  The one exact
    # anchor (x, 1e10.5) is genuinely k-loop bound and DOES want the cores, so this is a
    # split difference at 32: the anchor drops from ~138 h of wall to ~69 h, and the
    # Galerkin cells waste ~400 charged core-hours of sleeping core for it.  Worth it --
    # 138 h is most of a week of nothing else happening on that reservation.
    "cmb": dict(partition="q64", cpus=32, mem="64G", time=TIME_LIMIT, conc=12),
    # A partition LIST: these cells are small and numerous, so they should land
    # wherever there is a gap.  runcell.sh resolves class_$SLURM_JOB_PARTITION at
    # RUN time, so every partition named here needs a binary from build.sh.
    "iso": dict(partition="q48,q40,q64,q128,q144", cpus=N_K_ISO, mem="32G",
                time=TIME_LIMIT, conc=24),
    "gal": dict(partition="q40,q48", cpus=32, mem="64G", time=TIME_LIMIT, conc=12),
    # 24 h, not the 4 h hpc_prod used: that wall is what killed bg_G10.5_m0.06.
    # cpus = 4 is not parallelism, it is memory bandwidth and the writer -- the
    # background solve itself is serial.
    "bg":  dict(partition="q48,q40", cpus=4, mem="64G", time="24:00:00", conc=12),
}

# ── cost model ───────────────────────────────────────────────────────────────
# MEASURED core-hours, from the completed cells of the earlier campaigns, all
# NORMALISED TO tol_perturb_integration = 3e-6 and to N_K_REF k modes at 32
# cores, refine = 1, momenta_bins = 16.
#
# It is an ORDERING HEURISTIC, not a prediction.  collect.py --status prints the
# numbers to paste back in as cells finish.
N_K_REF = 300

# EXACT (dr_reduced_moments = 0).  RE-MEASURED from hpc_prod's own finished
# cells, family cmb, refine 1, with the BACKGROUND SUBTRACTED (it is now its own
# term) -- so these are the k-loop and nothing else.  hpc_prod inherited these
# from hpc_cmb/hpc_defl and they were ~2x LOW at the bottom of the ladder and
# ~1.4x low at the top.
#
# The bottom of the ladder is not DNCDM at all: at Gamma = 1e5 a family-A cell
# spends most of its k-loop on thermodynamics, the transfer functions and
# lensing, which is why cmb reads 8.0 where iso -- same physics, no C_l -- reads
# 1.9.  This table is the cmb one, because family A is what the budget is for;
# family B runs 1.1-1.3x above it at high Gamma, which SKEW_ONE_WAVE covers.
COST_EXACT = {
    "0.3":  {1e5: 8.0, 1e6: 11.5, 1e7: 16.2, 1e8: 33.8, 1e9: 154.2,
             1e10: 1610.0, 1e11: 12406.0},
    "0.06": {1e5: 27.4, 1e6: 36.1, 1e7: 76.3, 1e8: 400.8, 1e9: 3866.8},
}
# GALERKIN, 6 moments.  ⚠ THIS TABLE IS ALMOST ENTIRELY ZERO, AND THAT IS THE
# MEASUREMENT.  Same extraction as above over hpc_prod's m6 cells: the k-loop
# costs ~0.5 core-hours at m = 0.3 and ~1.0 at m = 0.06, FLAT in Gamma, and from
# Gamma ~ 1e8 up the measured wall is BELOW the background time -- i.e. the whole
# cell is its background solve and the reduction has nothing left to reduce.
#
# The number is conservative twice over: those cells were DIVERGING (PR #411),
# and a diverging solve takes smaller steps, not larger; and the residue is
# dominated by the same Gamma-independent CMB machinery as the bottom of
# COST_EXACT.  A fixed cell cannot cost more than this.
#
# hpc_prod's version of this table was extrapolated from hpc_nodefl at
# tol = 3e-8 and divided by a TOL_SCALE measured on the exact scheme.  It is
# gone: it under-estimated every Galerkin cell by 5-15x, and the reason was not
# the tolerance scaling but that the model had no background term at all.
COST_M6 = {
    "0.3":  {1e5: 0.9, 1e7: 0.5, 1e9: 0.5, 1e12: 0.5},
    "0.06": {1e5: 1.1, 1e7: 0.8, 1e10: 0.8},
}
# 4 and 8 moments against 6, measured at Gamma = 1e8..1e11 (hpc_reduced m4t/m6t/
# m8t): 0.85x and 1.13x, near enough flat in Gamma.
MOMENT_COST_FACTOR = {"m4": 0.85, "m6": 1.0, "m8": 1.13}

# momenta_bins scales the node count, hence runtime, near-linearly.
MOMENTA_COST_FACTOR = {16: 1.0, 32: 1.85, 64: 3.4}

# ── the background, which hpc_prod's model did not have at all ───────────────
# MEASURED WALL HOURS from hpc_prod's bg family, at momenta_bins = 16 and the
# dr_N_q this config's grid rule gives -- i.e. exactly the background a
# production cell at the same (Gamma, mass) solves, since dr_N_q is identical
# across schemes AND across a refine pair (dr_bg_refine subsamples the
# PERTURBATION grid; the background is the same solve either way).
#
# THREE PROPERTIES, none of which the k-loop model shares:
#   * SERIAL.  It does not divide by cpus.  On the 64-core family-A allocation
#     one hour of background is 64 charged core-hours.
#   * INDEPENDENT OF n_k and of refine.  Do not scale it by either.
#   * scales with momenta_bins, because the parent grid is part of the
#     background ODE state.  Family B's 32 therefore pays ~1.85x here too.
#     ASSUMED from the perturbation factor, not separately measured.
#
# These are an UPPER BOUND: the bg cells also write and slim a GB-scale PSD
# table, which production cells (write background = no) do not.  Measured
# cross-check: cmb_G11.0_m0.3_m6 came in at 1.5 h against this table's 1.71 h,
# and cmb_G10.0_m0.06_m6 at 2.2 h against 2.54 h.
COST_BG = {
    "0.3": {1e5: 0.0006, 3.162e5: 0.0006, 1e6: 0.0006, 3.162e6: 0.0006,
            1e7: 0.0008, 3.162e7: 0.0025, 1e8: 0.0044, 3.162e8: 0.0122,
            1e9: 0.0375, 3.162e9: 0.1019, 1e10: 0.2492, 3.162e10: 0.6567,
            1e11: 1.7083, 3.162e11: 4.8950, 1e12: 13.8872},
    "0.06": {1e5: 0.0006, 3.162e5: 0.0008, 1e6: 0.0014, 3.162e6: 0.0028,
             1e7: 0.0075, 3.162e7: 0.0242, 1e8: 0.0769, 3.162e8: 0.1758,
             1e9: 0.4172, 3.162e9: 1.0242, 1e10: 2.5372},
}

# One mode per core (family B) means the wall is the SLOWEST mode, not the mean.
# With several waves the pool smooths that out.  Booked, not measured.
SKEW_ONE_WAVE = 1.8
SKEW_MULTI_WAVE = 1.15

# Cells whose ESTIMATED wall exceeds this fraction of TARGET_WALL_H are written to
# runfile_<family>_skipped with the reason instead of being queued.
WALL_MARGIN = 1.0
