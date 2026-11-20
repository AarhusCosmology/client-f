# hpc_gal — redo everything that reads the reduced collision table

**~7 100 charged core-hours, 99 cells, ~21 % of `hpc_prod2`.**

`hpc_prod2`'s **exact** cells are keepers. Its **Galerkin** cells are not, and neither
is its isotropisation exponent. This round redoes exactly those, and nothing else.

> ## ⛔ THIS ROUND CANNOT RUN YET
>
> `EXPECTED_COMMIT` is the literal string `FILL_ME_AFTER_THE_KTABLE_FIX`, so guard 1 in
> `runcell.sh` refuses to start any cell. That is deliberate. The code change this round
> exists to measure — a **Γ-aware `kTableMaxDlna`** — is diagnosed and measured but **not
> implemented**. §1 says what to implement and §6 says what else it must expose.
>
> Every cell here is meaningless on a binary that still has the fixed 7e-3 constant: it
> would reproduce `hpc_prod2` and cost 7 100 core-hours to do it.

---

## 1 · Why: the reduced table's ln a spacing never scaled with Γ

`hpc_prod2`'s Galerkin ladder degrades monotonically above Γ = 10⁹ and diverges outright
by 10¹¹·⁵ — C₂^TT against the Γ-independent target 1.481e-10, m = 0.3:

| Γ | 10⁹ | 10⁹·⁵ | 10¹⁰ | 10¹⁰·⁵ | 10¹¹ | 10¹¹·⁵ | 10¹² |
|---|---|---|---|---|---|---|---|
| C₂ vs target | −0.1 % | −0.7 % | −2.4 % | −7.9 % | −30.1 % | **+506 %** | **+4×10¹⁰ %** |

m = 0.06 does the same one decade lower. At Γ = 10¹² `max|C_l| = 1.135`, i.e. O(1) — a
full runaway. **It is the reduction, not the physics:** at Γ = 10¹⁰ the exact cell reads
+0.0 % while its m6 twin reads −2.4 %, on the same background.

### The mechanism

`species/dncdm_inv_species.h` had `static constexpr double kTableMaxDlna = 7e-3` —
*identical* to the restored `back_integration_stepsize`. So

```
n_sub = ceil(dlna_row / kTableMaxDlna − kTableDlnaSlack)
      = ceil(7.0018e-3 / 7e-3 − 0.01) = 1
```

**PR #411's sub-division machinery is inert at the shipped default.** The reduced table's
ln a spacing was simply the background table's, at every Γ, while the collision rate aΓ
— and therefore everything the table has to resolve — grew. #411 validated its fix on
*"all ten cells of the Γ = 1e5..1e9 ladder"*, and the ladder is clean to **exactly** 10⁹.

### The fix, and the exponent it needs

Make `kTableMaxDlna` Γ-aware. Anchored at 7e-3 for Γ ≤ 10⁹, index from `config.KTABLE_INDEX`:

```
kTableMaxDlna(Γ) = KTABLE_AT_1E9 · (Γ / 1e9)^(−KTABLE_INDEX)      # 7e-3, 0.60
```

Measured, one binary per constant, same ini, nothing else changed:

| cell | n_sub = 1 | 10 | 20 |
|---|---|---|---|
| `cmb_G10.5_m0.3_m6` (Γ = 3.16e10) | −7.9 % | **−0.1 %** | — |
| `cmb_G11.0_m0.3_m6` (Γ = 1e11) | −30.1 % | −0.3 % | **−0.1 %** |
| `cmb_G11.5_m0.3_m6` (Γ = 3.16e11) | **+506.1 %** | — | −0.4 % |

Holding 0.1 % needs n_sub = 10 at 3.16e10 and 20 at 1e11 ⇒ **Γ^0.60**. Confirmed a full
decade up: at Γ = 3.16e11 the law asks for n_sub = 40, and n_sub = 20 — half of it —
lands at 0.4 %, which is exactly the 4× that linear interpolation's h² scaling predicts.
Three Γ, self-consistent, and it takes a +506 % runaway to −0.4 %. Cross-checked
independently: at fixed spacing the error grows 7.9 % → 30.1 % over Γ × 3.16, i.e.
∝ Γ^1.16, and linear interpolation has error ∝ (h/L)², giving L ∝ Γ^−0.58. Two routes to
~0.6. The integrator's own fixed-grid step-count exponent is **0.662** (11 ladders,
`hpc_ratchet`'s `ceil` family), which is a safe upper bound on this one.

⚠ **`n_sub` currently depends only on the constant, not on Γ at all** — `dlna_row` is
just the background spacing. That is the whole bug and the whole fix.

### It is not a memory problem

A table row is `(l_max+1)·size²·8` = 110 KB with `size = momenta_bins + 2·moments`, and is
**independent of `dr_N_q`**. The active window *shrinks* as the parent decays earlier —
measured from the per-`n_sub` RSS slope, 0.307 GB/n_sub at Γ = 3.16e10 (~2918 rows) against
0.145 GB/n_sub at 1e11 (~1378 rows), i.e. **Γ^−0.65**. That nearly cancels the +0.60, so

```
rows = window × n_sub ∝ Γ^−0.05          table stays 2–3 GB at EVERY Gamma
```

⚠ Two points only; the window exponent is weakly determined and worth one more
measurement. But the direction is not in doubt, and an earlier estimate of 627 GB at
Γ = 10¹⁴ assumed a fixed window and was **wrong**. No big-memory node is needed. Cubic
spline instead of linear stays in reserve — it buys ~10× in h at ε = 1e-4 (~33× at 1e-6),
a constant factor only, since h/L = const regardless of interpolation order.

Time cost is small: **+9.4 % wall for a 10× table** (1305 → 1428 s), because `Apply` is
O(size²) per call regardless of row count. The reduction keeps its ~50× advantage over
exact.

---

## 2 · Why the isotropisation exponent moves too — `momenta_bins` 32 → 64

`hpc_prod2` ran the rung it prescribed (`MOMENTA_RUNG = [("09.0","0.3",[16,64])]`,
exact scheme, Γ = 10⁹, m = 0.3) and then did not read it. Fitted p, `collect.py`'s own
estimator, median over 7 k:

| gate | mb = 16 | mb = 32 | mb = 64 | 64−16 | 64−32 |
|---|---|---|---|---|---|
| 15 | 3.93 | 3.65 | 3.55 | −0.38 | **−0.10** |
| 20 | 3.90 | 3.60 | 3.48 | −0.43 | **−0.13** |

32 → 64 sits **at or above** the round's own Δp > 0.10 threshold and the sequence is still
descending (successive differences 0.28 → 0.10, ratio 0.36 ⇒ residual at 64 ≈ 0.06,
p_∞ ≈ 3.49). **A p published from a mb = 32 ladder is biased high by ~0.10–0.15**, in the
direction a coarse grid always fakes. 32 was chosen from the a4Pi *wiggle amplitude*,
never from p — and `config.py` itself said "if it does [move], that rung is the finding,
not a footnote."

So family B runs at **64**, at 1.84× the cost (`MOMENTA_COST_FACTOR` 3.4 against 1.85),
and `MOMENTA_RUNG` now carries a single **128** cell to show p has stopped moving. Its
partner at 64 (`iso_G09.0_m0.3_x`) is in the ladder on purpose: a convergence rung with
nothing to converge against is not a rung.

---

## 3 · What is kept, and what is redone

| | |
|---|---|
| **KEPT** — not rerun | every **exact** (`x`) cell of `hpc_prod2`. The exact scheme never touches the reduced table, verified: Γ = 10¹⁰ / m = 0.3 reads +0.0 % where its m6 twin reads −2.4 %. |
| **REDONE** | every **Galerkin** cell, at every Γ — cmb `m6`, iso `m6`, and the `gal` closure arm, which reads the tabulated operator too. |
| **NEW** | one exact anchor at Γ = 10¹⁰·⁵ (cmb and iso), and the `momenta_bins` = 128 rung. |

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
| `cmb` | 28 | 32 | ~4 170 | the Galerkin C_l ladder + the exact anchor |
| `iso` | 57 | 8 | ~2 610 | family B at mb = 64, + exact anchor, + the 128 rung |
| `gal` | 14 | 32 | ~330 | the closure arm (m4/m8) |
| `bg` | 26 | 4 | small | pre-flight |

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
* **No gather axis.** `balanced_gather` is PR #412's question, not this one. If #412
  merges first, `GAMMA_CEILING` may move (its `ceil` family measures a flat 14.3
  bins/decade to Γ = 10¹¹ where the shipped scheme is 54 % apart at 114.4) — that would
  *extend* this round's ladders, and is a reason to sequence #412 first, not to mix them.

---

## 6 · What the code change must expose, beyond the Γ-aware constant

**One `gen.py` addition is still needed**, and it is the round's own convergence gate.

Every other axis here carries a twin — `refine = 2` halves the perturbation grid, the
`mb` rung doubles the parent grid — but the table spacing would have **none**, and this
round's whole premise is that the spacing was wrong and unnoticed. A Γ-aware constant with
no way to test it is the same failure mode one exponent along.

So the fix should expose the spacing as an **ini key** (a multiplier on the Γ-aware
default is enough, e.g. `dncdm1.dr_table_refine = 2`) and `gen.py` should emit a twin at
2× for a handful of cells — the same shape as `REFINES`, in the same place. **What counts
as converged:** the pair agrees to 1 % *and* both sit within 1 % of 1.481e-10, which is
the criterion `decide.py` already uses for C₂.

Without that key, the only evidence the table is converged at Γ = 10¹² is an extrapolation
of a two-point exponent — which is exactly the kind of claim this round exists to stop
making.

---

## 7 · Running it

```
./sync_to_hpc.sh
ssh YOUR_USER@YOUR_CLUSTER
cd ~/codes/CLASSpp/hpc_gal
ml load gcc
git -C ~/codes/CLASSpp checkout <the kTableMaxDlna fix>
./build.sh ~/codes/CLASSpp q40 q48 q64 q128 q144     # writes BUILD_COMMIT
python3 gen.py                                       # after EXPECTED_COMMIT is real

sbatch submit_bg.sh                                  # pre-flight, always first
sbatch submit_cmb.sh
sbatch submit_gal.sh
sbatch submit_iso.sh
python3 reduce_iso.py
./sync_from_hpc.sh
python3 collect.py --status
```

⚠ **`build.sh` must write `BUILD_COMMIT`.** Every one of `hpc_ratchet`'s 181 logs says
`note: no BUILD_COMMIT/EXPECTED_COMMIT pair -- commit check skipped`, so guard 1 — the one
the harness calls "authoritative and precise" — never ran for that entire round. Check for
that line in the first log this round produces and stop if it is there.

## 8 · Reading the result

`collect.py --status` / `--cl` / `--iso` as in `hpc_prod2`. The three questions:

1. **Does the Galerkin ladder land on 1.481e-10 at every Γ?** It must, to 1 %, all the way
   to the ceiling. Anything else means the exponent in §1 is wrong.
2. **Does m6 agree with exact at Γ = 10¹⁰·⁵?** This is the anchor. `|Δp| ≤ 0.10` for iso
   and ≤ 1 % on C₂ for cmb, the same thresholds the overlap already meets at Γ ≤ 10¹⁰.
3. **Has p stopped moving?** `iso_G09.0_m0.3_x` (64) against `iso_G09.0_m0.3_x_mb128`.
   Δp ≤ 0.05 says 64 is enough; anything larger says the ladder needs 128 and this round's
   family B is itself under-resolved.
