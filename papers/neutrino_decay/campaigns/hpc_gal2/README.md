# hpc_gal2 — redo the Galerkin ladder, this time with the key actually set

**~7 900 charged core-hours, 103 cells.** Replaces `hpc_gal`, which produced no new
information at all.

> ## Why `hpc_gal` is void
>
> That round existed to measure a Γ-aware reduced-table spacing. The spacing was never
> set, and nothing stopped the round:
>
> * **`gen.py` never emitted a key.** `config.py` held `KTABLE_AT_1E9, KTABLE_INDEX =
>   7e-3, 0.60` as a *specification*; no code read those names and no ini in `hpc_gal/ini`
>   carries any table key. Its README §6 — "What the code change must expose" — was a
>   requirement, never a report.
> * **The code change did not exist either.** On master `kTableMaxDlna` was still
>   `static constexpr double = 7e-3`, with no input key. PR #411 had given the table its
>   own named constant but left the number equal to `back_integration_stepsize`, so
>   `n_sub = ceil(7.0021e-3/7e-3 − 0.01) = 1` at every Γ.
> * **Guard 1 did not fire.** `EXPECTED_COMMIT` was the literal
>   `FILL_ME_AFTER_THE_KTABLE_FIX`, which should have refused every cell — but `BUILD_COMMIT`
>   was absent, and guard 1 answered a missing file with `note: ... commit check skipped`
>   and carried on. All 125 logs say so.
>
> The result is a repeat of the previous round: `hpc_gal/out/cmb_*_used.ini` is
> **byte-identical** to `hpc_prod2`'s, and the spectra agree to 1e-6 (Γ = 10⁹) to 1.7e-4
> (Γ = 10¹¹·⁵) — thread-order round-off, not a code difference. Every C₂ error is
> reproduced, +506% and +4×10¹⁰% included.
>
> ⚠ **Do not `scancel` `hpc_gal`'s four in-flight cells.** As of 2026-08-30 it still has
> `cmb_G10.5_m0.3_x` (2 214 core-h, started 29 Aug, ~69 h wall), `iso_G10.5_m0.3_x`, its
> `xc` twin, and `iso_G12.0_m0.3_m6` running. Three of the four are EXACT cells and are
> valid on arrival for the reason below; only the last is void. Of its ~7 100 core-hour
> estimate the void part is the ~1 200 already spent on finished Galerkin cells, not the
> round.
>
> **What survives from it, and is NOT rerun here:** the exact (`x`) cells. The exact
> scheme never touches the reduced table, so `iso_G09.0_m0.3_x`, its `xc` twin and the
> `mb128` rung are valid measurements and are kept. Its `iso` m6 ladder at
> `momenta_bins = 64` is a real change from `hpc_prod2`'s 32 and is re-run here only
> because it reads the table.

---

## 1 · The fix: `dncdm1.dr_table_max_dlna` (PR #413)

The reduced table's ln a spacing is now an **input**, defaulting to the 7e-3 that
reproduces old behaviour exactly (verified bit-identical, same build directory).

It is an input and not a law inside CLASS on purpose, and that decision is the round's
premise. Several knobs in this model have requirements that grow with Γ — `dr_N_q` above
all — a production cell costs hundreds of core-hours, and a caller who has not been made
to think about one of them has not been made to think about the others. An explicit key
also gives the convergence test a handle that a hidden law does not have: **halve it, and
the pair brackets the answer.** That is what `TABLE_RUNG` below is.

### What the old spacing cost

C₂^TT is Γ-independent in this model, so any departure from **1.481e-10 is grid error**.
`hpc_prod2`'s six-moment ladder at n_sub = 1, m = 0.3:

| Γ | 10⁹ | 10⁹·⁵ | 10¹⁰ | 10¹⁰·⁵ | 10¹¹ | 10¹¹·⁵ | 10¹² |
|---|---|---|---|---|---|---|---|
| C₂ vs target | −0.1% | −0.7% | −2.4% | −7.9% | −30.1% | **+506%** | **+4×10¹⁰%** |

m = 0.06 does the same about half a decade lower. It is the reduction and not the physics:
at Γ = 10¹⁰ the exact cell reads +0.0% where its m6 twin reads −2.4%, on the same
background. And nothing in the run says so — the identities the kernel asserts are
*number* identities, which a mis-interpolated operator still satisfies.

### The sizing rule, corrected in three places

`hpc_gal`'s rule was wrong beyond not being emitted. Linear interpolation is O(h²) in the
realised spacing h = dlna_row/n_sub, so

```
err(Gamma, m, n_sub) = A(m) (Gamma/1e11)^1.103 / n_sub^2
A(0.30 eV) = 0.2947,  A(0.06 eV) = 0.8682
```

fitted to the n_sub = 1 ladder above. Three corrections against the old block:

1. **The index is 1.103, so n_sub ~ Γ^0.551 — not Γ^0.60.** The old index came from
   d ln err/d ln Γ read across 7.9% → 30.1%, and the 30.1% point is already leaving the
   regime where the h² law holds. Fit only the perturbative rungs and it is 1.103.
2. **The normalisation was low.** Anchoring n_sub = 1 at Γ = 10⁹ asks for n_sub = 8 at
   Γ = 3.16×10¹⁰, where **10 was measured to be needed**.
3. **There was no mass dependence at all**, while m = 0.06 needs ~1.7× the n_sub of
   m = 0.3 at the same Γ (A ratio 2.95, √2.95 = 1.72).

Γ = 10⁹ is excluded from the fit at both masses: the residual there is the six-moment
**closure floor** (−0.11 to −0.56%, no Γ trend), not table error, and including it reads a
floor as a slope.

**Validated against the independent n_sub sweep the fit never saw:**

| Γ | n_sub | predicted | observed |
|---|---|---|---|
| 3.16×10¹⁰ | 10 | 0.083% | 0.1% |
| 10¹¹ | 10 | 0.295% | 0.3% |
| 10¹¹ | 20 | 0.074% | 0.1% |
| 3.16×10¹¹ | 20 | 0.262% | 0.4% |

and end to end on the PR #413 binary at Γ = 10¹⁰ / m = 0.3, where the rule asks for
n_sub = 5: C₂^TT reads 1.18295e-10 / 1.21801e-10 / 1.21910e-10 at n_sub = 1 / 5 / 10, so
the rule's own value sits **0.090%** from ten times the table. Refinement is nearly free —
164 s → 197 s for that tenfold table, because `Apply` is O(S²) per call whatever the row
count.

⚠ **`KTABLE_A` is two masses.** Between them it is an interpolation; outside them it is
nothing. A third mass needs its own measurement, and `config.n_sub` raises rather than
extrapolating.

### It is not a memory problem

A table row is `(l_max+1)·size²·8` = 110 KB with `size = momenta_bins + 2·moments`, and is
**independent of `dr_N_q`**. The active window *shrinks* as the parent decays earlier —
measured 0.307 GB/n_sub at Γ = 3.16×10¹⁰ (~2918 rows) against 0.145 GB/n_sub at 10¹¹
(~1378 rows), i.e. Γ^−0.65 — which nearly cancels the +0.55, so `rows = window × n_sub`
is close to flat in Γ and the table stays a few GB at every rung. Two points only, and the
window exponent is weakly determined; but an earlier estimate of 627 GB at Γ = 10¹⁴ assumed
a fixed window and was simply wrong. No big-memory node is needed.

## 2 · Why the isotropisation exponent moves too — `momenta_bins` 32 → 64

`hpc_prod2` ran the rung it prescribed (`MOMENTA_RUNG`, exact scheme, Γ = 10⁹, m = 0.3)
and then did not read it. Fitted p, median over 7 k:

| gate | mb = 16 | mb = 32 | mb = 64 | 64−16 | 64−32 |
|---|---|---|---|---|---|
| 15 | 3.93 | 3.65 | 3.55 | −0.38 | **−0.10** |
| 20 | 3.90 | 3.60 | 3.48 | −0.43 | **−0.13** |

32 → 64 sits **at or above** the Δp > 0.10 threshold and the sequence is still descending
(successive differences 0.28 → 0.10, ratio 0.36 ⇒ residual at 64 ≈ 0.06, p_∞ ≈ 3.49). **A
p published from an mb = 32 ladder is biased high by ~0.10–0.15**, in the direction a
coarse grid always fakes. So family B runs at **64**, and `MOMENTA_RUNG` carries a single
**128** cell to show p has stopped moving.

`hpc_gal` already ran that arm at 64 and those cells are valid *as far as the parent grid
goes* — but its m6 cells read the reduced table, so they carry the wrong spacing above
Γ ≈ 10⁹·⁵ like everything else, and are re-run here.

## 3 · What is kept, and what is redone

| | |
|---|---|
| **KEPT** — not rerun | every **exact** (`x`) cell of `hpc_prod2`. The exact scheme never touches the reduced table, verified: Γ = 10¹⁰ / m = 0.3 reads +0.0 % where its m6 twin reads −2.4 %. |
| **REDONE** | every **Galerkin** cell, at every Γ — cmb `m6`, iso `m6`, and the `gal` closure arm, which reads the tabulated operator too. |
| **KEPT** — not rerun | `hpc_gal`'s **exact** cells, for the same reason: `iso_G09.0_m0.3_x`, its `xc` twin, and `iso_G09.0_m0.3_x_mb128`. Those are the `momenta_bins` convergence rung of §2 and they are valid; the round they came in is not. |
| **NEW** | one exact anchor at Γ = 10¹⁰·⁵ (cmb and iso), and — the round's own gate — the `TABLE_RUNG` twins at 2× the rule's spacing. |

Not only the visibly broken rungs. The table spacing was wrong *everywhere*; it just did
not **show** below ~10⁹·⁵, and a ladder half of whose rungs came off a different
discretisation is not a ladder.

### The exact anchor is the load-bearing cell of the round

Above Γ = 10¹⁰ (m = 0.3) the Galerkin ladder has **no exact twin**, so nothing there is
validated against anything — and that is precisely the range where the old table failed.
Refitting the table and re-running m6 alone would replace an unvalidated number with a
*different* unvalidated number. `cmb_G10.5_m0.3_x` + its refine twin is ~3 300 core-hours,
47 % of the round, and it is what makes the rest of it evidence.

⚠ Family B's own resolution diagnostic has **gone blind** exactly there:
`dp(coarse)` reads 0.00 at every Γ ≥ 10¹⁰·⁵, because the reduced perturbation state is six
moments and no longer depends on the daughter grid. p plateaus at 4.12–4.14, which may be
the physical saturation or may be the reduction failing in a way p cannot see. The iso
exact anchor at 10.5 is the only thing that can tell those apart.

---

## 4 · Budget

| family | cells | cores | charged core-h | what it is |
|---|---|---|---|---|
| `cmb` | 32 | 32 | ~4 920 | the Galerkin C_l ladder + the exact anchor + 4 `TABLE_RUNG` twins |
| `iso` | 57 | 8 | ~2 610 | family B at mb = 64, + exact anchor, + the 128 rung |
| `gal` | 14 | 32 | ~330 | the closure arm (m4/m8) |
| `bg` | 26 | 4 | small | pre-flight |

**103 cells, ~7 900 charged core-hours.** The four `TABLE_RUNG` cells add ~750 of that and
they are the cheapest insurance in the round: without them the only evidence the table is
converged at Γ = 10¹² is an extrapolation of a fitted exponent, which is exactly the kind
of claim this round exists to stop making.

**cmb runs on 32 cores, not 64.** An m6 cell is ~100 % its **serial** background —
`hpc_prod2` measured `cmb_G11.0_m0.3_m6` at 1.5 h against `bg_G11.0_m0.3`'s 1.71 h — so a
64-core node is charged 64× for sleeping cores. `hpc_ratchet` made this correction and it
was the single largest line item it removed. 32 is a split difference: the one k-loop-bound
exact anchor drops from ~138 h of wall to ~69 h, and the Galerkin cells waste ~400 charged
core-hours of sleeping core for it.

---

## 5 · What this round deliberately does not do

* **No `GRID_RUNG`.** `hpc_prod2` carried six `_q2x` cells to chase a "daughter-grid
  resonance" peaking near Γ ~ 10⁷. That was **root-caused 2026-08-28 and it was never the
  daughter grid**: it was the global `back_integration_stepsize` at #397's 0.07, which
  #411 restored to 7e-3. Isolated on one binary, one ini, that key alone, at the exact
  cell `hpc_prod` published −41.1 % for — 0.07 gives **−42.3 %**, 7e-3 gives **+0.2 %**.
  There is no resonance left to probe.
* **No `BPD_PT` reduction**, though the `c` twins say the rule is ~2× conservative and it
  is worth a median 48 % of wall on an exact cell (n = 21). This round has exactly one
  exact cell and it exists to *anchor* the `hpc_prod2` ladder, so changing its grid rule
  would make it non-comparable with the cells it anchors. Obvious saving for the round
  after.
* **No gather axis.** PR #412 has since merged, so the binary *has* `balanced_gather` —
  but it defaults off and no ini here sets it, so this round runs the shipped scheme. That
  is deliberate: `hpc_ratchet` measured the gather at ≤ 0.35 % on C₂ at low Γ and, decisively,
  it does **not** cure the high-Γ divergence (−30.10 % → −29.57 % at Γ = 10¹¹). That
  divergence is the table spacing, which is what this round fixes. Varying both axes would
  make every cell answer neither. #412 may still move `GAMMA_CEILING` — its `ceil` family
  measures a flat 14.3 bins/decade to Γ = 10¹¹ where the shipped scheme is 54 % apart at
  114.4 — and that would *extend* this round's ladders. It is a reason for a later round,
  not for mixing them into this one.

---

## 6 · The round's own convergence gate: `TABLE_RUNG`

Every other axis here carries a twin — `refine = 2` halves the perturbation grid, the `mb`
rung doubles the parent grid — and until PR #413 the table spacing had **none**. That is
precisely how a wrong spacing shipped at every Γ without anyone noticing. A rule with no
way to test it is the same failure one exponent along.

`TABLE_RUNG` runs four cells at **half** the spacing the rule asks for:

| cell | rule's n_sub | twin | what it tests |
|---|---|---|---|
| `cmb_G11.0_m0.3_m6` | 18 | 36 | the top of the m = 0.3 ladder that still has an exact anchor below it |
| `cmb_G12.0_m0.3_m6` | 62 | 124 | the ladder's top rung — the +4×10¹⁰% runaway under n_sub = 1 |
| `cmb_G10.0_m0.06_m6` | 9 | 18 | the top of the m = 0.06 ladder, and the mass the old rule ignored |
| `cmb_G07.0_m0.3_m6` | **1** | 2 | a **shoulder** — the rule asks for no sub-division at all here |

**What counts as converged:** the pair agrees to `KTABLE_EPS` (1e-3) *and* both sit within
`KTABLE_EPS` of 1.481e-10. Agreeing somewhere else is converging to the wrong place;
landing on target only at double resolution is the rule being wrong and the cost being
paid anyway.

⚠ **The shoulder is not padding.** At Γ = 10⁷ the rule asks for n_sub = 1, so doubling
must change **nothing**. If it moves, the rule has no margin anywhere and the whole ladder
is in question rather than its top. `hpc_prod2`'s daughter-grid rung learned this the
expensive way: a rung that only tests where you expect a change cannot distinguish "the
fix works" from "the knob does something everywhere".

---

## 7 · Running it

```
./sync_to_hpc.sh
ssh YOUR_USER@YOUR_CLUSTER
cd ~/codes/CLASSpp/hpc_gal2
ml load gcc
git -C ~/codes/CLASSpp checkout dncdm-table-dlna-input   # PR #413, until it merges
./build.sh ~/codes/CLASSpp q40 q48 q64 q128 q144     # writes BUILD_COMMIT
python3 gen.py

sbatch submit_bg.sh                                  # pre-flight, always first
sbatch submit_cmb.sh
sbatch submit_gal.sh
sbatch submit_iso.sh
python3 reduce_iso.py
./sync_from_hpc.sh
python3 collect.py --status
```

### The two guards that failed last time, and what changed

Both were fixed here, in the scripts rather than in a note asking someone to check:

* **`build.sh` now FAILS if it cannot record the commit.** It used to write `BUILD_COMMIT`
  inside an `if git rev-parse ...; then` with no `else`, so a repo it could not read left
  the file simply absent. It also warns loudly if what it built is not `EXPECTED_COMMIT`.
* **`runcell.sh` guard 1 now FAILS on a missing `BUILD_COMMIT`.** It used to print
  `note: ... commit check skipped` and run. A guard that can be skipped by the *absence*
  of a file is not a guard — that one line cost `hpc_gal` ~7 000 core-hours and
  `hpc_ratchet` its provenance (all 181 of its logs say it too).
* **Guard 2's functional probe now feeds CLASS `dr_table_max_dlna`.** Without that, a
  binary from before #413 passes every other guard, silently ignores the key this round
  exists to set, and reproduces `hpc_prod2` — which is the exact failure being repaired.
  Same lesson `hpc_ratchet` applied to `balanced_gather`.

⚠ **Bump `EXPECTED_COMMIT` to the squash-merge hash the moment #413 lands, and rebuild.**
Not for the guard's sake — for reproducibility. `master` is the only branch pushed to
`CLASSpp_public`, so a branch head is not a commit anyone outside this machine can check
out, and each round has to name a public SHA.

## 8 · Reading the result

`collect.py --status` / `--cl` / `--iso` as in `hpc_prod2`. The three questions:

0. **Do the `TABLE_RUNG` pairs agree?** Read this FIRST — it is what says the other three
   answers mean anything. Criterion in §6, shoulder included.
1. **Does the Galerkin ladder land on 1.481e-10 at every Γ?** It must, to 1 %, all the way
   to the ceiling. Anything else means the exponent in §1 is wrong.
2. **Does m6 agree with exact at Γ = 10¹⁰·⁵?** This is the anchor. `|Δp| ≤ 0.10` for iso
   and ≤ 1 % on C₂ for cmb, the same thresholds the overlap already meets at Γ ≤ 10¹⁰.
3. **Has p stopped moving?** `iso_G09.0_m0.3_x` (64) against `iso_G09.0_m0.3_x_mb128`.
   Δp ≤ 0.05 says 64 is enough; anything larger says the ladder needs 128 and this round's
   family B is itself under-resolved.
