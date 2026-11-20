# hpc_prod2 — the production round

The final run for the inverse-decay paper. Replaces `hpc_prod`, which was built
on merged master (PR #387) and **could not be used as it stood**:

* Every reduced-moment (Galerkin) cell diverged — 153 of 335 transfer outputs
  blown, `max|phi|` to 1e75, nine cells to NaN. Family C was void and family A's
  half-decade ladder with it. Root cause and fix: **PR #411**.
* The trigger was `back_integration_stepsize` 7e-3 → 0.07 in #397. That default
  is **back at 7e-3** in the same PR, because the speed it was moved for is not
  there: measured, it is worth 0.3% of a lensed ΛCDM run, and the whole of
  background + thermodynamics is 3% of one.
* `used_in_output` was sized by output rows rather than equations, silently
  corrupting the background table for any cell with `dr_N_q >= 220` (PR #409).
  That is most of this ladder above Γ = 10⁹.

⚠ **`hpc_prod`'s outputs must not appear in a figure beside these.** Not the
old-knob argument that applies to the pre-merge matrices (below) — a simpler
one: different binary, and on the Galerkin cells a different answer.

⚠ **Do not reuse `hpc_prod/out`.** `runcell.sh` resumes on a byte-identical
ini, and nothing in the ini changed — so dropping this config into that
directory would silently keep all 151 stale cells.

Everything is driven by `config.py`. Edit it, re-run `gen.py`, resubmit.

> **Here to analyse a finished round rather than run one?** Skip to
> **If you are here to ANALYSE, not to run** — the four questions this round
> exists to settle, what each cell leaves behind, and what counts as a result.
> Start with `collect.py --grid`: it decides whether the low-ℓ end is usable.

```
./sync_to_hpc.sh
ssh YOUR_USER@YOUR_CLUSTER
cd ~/codes/CLASSpp/hpc_prod2
ml load gcc
git -C ~/codes/CLASSpp checkout fix-reduced-table-lna-density   # until #411 lands
./build.sh ~/codes/CLASSpp q64 q48 q40 q128 q144
python3 gen.py

sbatch submit_bg.sh          # FIRST — minutes, and it is what says whether the
python3 collect.py --bg      # high-Gamma cells are worth submitting at all

sbatch submit_cmb.sh         # family A   q64,  64 cores
sbatch submit_iso.sh         # family B   list,  8 cores
sbatch submit_gal.sh         # family C   q40/q48, 32 cores

python3 collect.py --status
python3 collect.py --grid    # the doubled daughter grid -- read this early
python3 reduce_iso.py        # family B -> derived/, a few kB per cell
python3 reduce_iso.py -j 16  # ...one process per dump, biggest first
python3 reduce_iso.py --probe out/iso_G12.0_m0.3_m6_perturbations_k5_s.dat.gz
                             # read / inflate / parse, to see which one is the wall
./sync_from_hpc.sh           # the GB-scale dumps stay on the cluster
```

⚠ **`EXPECTED_COMMIT` is `6bc5f66e`, a PR-#411 branch head, not a master
commit.** Bump it to the squash-merge hash the moment #411 lands, and rebuild.
A binary from plain master passes guards 2–4 and produces garbage in family C.

**198 cells, ~33 000 charged core-hours**: family A 57 / 25 200, family B 101 /
7 600, family C 14 / 330, plus 26 background pre-flights at ~100 core-h. Two
cells are over budget and go to `runfile_*_skipped`, both the m = 0.06 exact
frontier at Γ = 10¹⁰ — 289 h against family A's 240 h target, 419 h against
family B's 330 h and the 359 h Slurm wall. Its exact anchor at Γ = 10⁹ survives,
so the Galerkin ladder still has an overlap at that mass.

That is 33 000 against `hpc_prod`'s estimated 57 000 for a *larger* matrix, and
the difference is not optimism: the cost model was rebuilt on `hpc_prod`'s own
measured walls. See **The cost model was wrong in two places**.

---

## The three families

| | what it delivers | output | cores | partition | budget |
|---|---|---|---|---|---|
| **A · cmb** | C_l and P(k) ladders, exact + Galerkin | `tCl,pCl,lCl,mPk,mTk` | 64 | q64 | 240 h |
| **B · iso** | a4PiDot → the isotropisation rate | `mTk` + k_output only | 8 | q48,q40,q64,q128,q144 | 330 h |
| **C · gal** | Galerkin closure error (4/6/8 moments) | as A | 32 | q40,q48 | 240 h |
| **D · proxy** | **deferred** — see below | — | — | — | — |

### Reach, and the ceiling that sets it

Exact on full decades, Galerkin on half decades. Estimated wall hours, production
cell + convergence twin:

| | m = 0.3 exact | m = 0.3 Galerkin | m = 0.06 exact | m = 0.06 Galerkin |
|---|---|---|---|---|
| **A · cmb** | Γ ≤ 10¹¹ (97+49 h) | 10¹² (14 h) | Γ ≤ 10⁹ (30+15 h) | 10¹⁰ (2 h) |
| **B · iso** | Γ ≤ 10¹¹ (267+135 h) | 10¹² (27+27 h) | Γ ≤ 10⁹·⁵ (120+61 h) | 10¹⁰ (5+5 h) |

Two different things stop the ladder, and only one of them is money.

**The exact frontier is cost.** Cost is ≈ Γ¹·⁰ for the exact scheme, so the
m = 0.06 cell at 10¹⁰ lands at 289 h (family A) and 419 h (family B) and is
skipped in both.

**The Galerkin frontier is the BACKGROUND, and it is a hard stop.**
`GAMMA_CEILING` is Γ ≤ 10¹² at m = 0.3 and 10¹⁰ at m = 0.06, from `hpc_prod`'s
measured background pre-flights:

| | 10⁹ | 10⁹·⁵ | 10¹⁰ | 10¹⁰·⁵ | 10¹¹ | 10¹¹·⁵ | 10¹² |
|---|---|---|---|---|---|---|---|
| m = 0.3 | 0.04 h | 0.10 h | 0.25 h | 0.66 h | 1.71 h | 4.89 h | **13.89 h** |
| m = 0.06 | 0.42 h | 1.02 h | **2.54 h** | *killed at the 4 h wall* | — | — | — |

~2.8× per half decade at both masses. Nothing above those two entries has ever
produced a background table on this cluster, and **a Galerkin cell cannot start
without one** — the background solve is identical across schemes and across a
refine pair, so the reduction does not help with it at all. `hpc_prod` ran the bg
family at a 4 h wall, which is what killed `bg_G10.5_m0.06`; this round runs it
at 24 h, which is enough for its own top cell and not enough to go further.

That costs the round the top of both Galerkin ladders — m = 0.3 stops at 10¹²
instead of 10¹³, m = 0.06 at 10¹⁰ instead of 10¹². The reduction's reach argument
survives with a one-decade lever arm (exact to 10¹¹, Galerkin to 10¹²) rather
than two. **Raising the ceiling is a code question, not a scheduling one**: the
background is serial, accuracy-limited rather than stiff, and `dr_N_q` is the
lever.

### The cost model was wrong in two places

`hpc_prod`'s model had **no background term at all**, and its Galerkin cells came
in 5–15× over estimate because of it. Checked against the bg table, those cells
are ~100% background: `cmb_G11.0_m0.3_m6` measured 1.5 h against
`bg_G11.0_m0.3`'s 1.71 h, and `cmb_G10.0_m0.06_m6` 2.2 h against 2.54 h.

Re-extracting every finished cell's wall with the background subtracted says the
reduction's k-loop costs **~0.5 core-hours flat in Γ** — `COST_M6` in `config.py`
is now almost entirely a constant, and that is the measurement, not a placeholder.
It is conservative twice over: those cells were *diverging*, which makes a solve
dearer and not cheaper, and the residue is the same Γ-independent thermodynamics
/ transfer / lensing floor that puts `COST_EXACT` at 8 core-hours for a Γ = 10⁵
cell where family B, same physics without C_l, reads 1.9.

`COST_EXACT` was ~2× low at the bottom of the ladder and ~1.4× low at the top;
it is now `hpc_prod`'s measured family-A walls directly.

The background enters `cost()` as its own term: **serial**, so it is added to the
wall rather than divided by `cpus`, and multiplied by `cpus` to charge for the
cores it leaves idle. It does not scale with `n_k` or with `refine`.

### The doubled daughter grid: a resonance, not a rising requirement

> ⚠ **SUPERSEDED 2026-09-14. This whole section rests on `hpc_prod`, the round
> PR #411 voided, and this round's own `q2x` cells refute it.** The "resonance"
> was `back_integration_stepsize` 0.07, restored to 7e-3 by #411 — not the
> daughter grid. Measured here, post-#411, the production cell's own low-ℓ error
> is **monotone and smooth** in Γ: +0.19% / +0.00% / −0.03% / −0.09% / −0.17% /
> −0.33% at Γ = 10⁵…10¹⁰ (C₂ᵀᵀ). In particular the −41.1% quoted below at
> Γ = 10⁷ now reads **+0.036%** off `cmb_G07.0_m0.3_x_q2x`. The requirement does
> **not** peak mid-ladder, `BPD_PT_AT_1E7`'s power-law form is not refuted, and
> `GRID_RUNG`'s "four contaminated rungs / two shoulders" framing is void.
> Kept below only so the argument that drove `GRID_RUNG` can be traced.
> See *What the refine pair actually measures*.

`BPD_PT_AT_1E7 = 27` with index 0.17 assumes the daughter-grid requirement is a
power law in Γ. **`hpc_prod`'s finished exact ladder says it is not.**

C_2^TT is Γ-independent in this model — measured, m = 0.06 reads 1.4860 / 1.4860
/ 1.4859 ×10⁻¹⁰ at Γ = 10⁷ / 10⁸ / 10⁹, and m = 0.3 reads 1.4812 / 1.4834 /
1.4811 at 10⁵ / 10⁹ / 10¹⁰ — so any departure from 1.481×10⁻¹⁰ *is* grid error,
and the ladder measures the requirement directly:

| m = 0.3, bins/decade → | 6.5 | 12.7 | 18.7 | 27.3 | 40.3 | 59.2 | 88.9 |
|---|---|---|---|---|---|---|---|
| Γ = 10⁵ | +0.1% | +0.0% | | | | | |
| 10⁶ | | −10.5% | **+5.1%** | | | | |
| 10⁷ | | | | **−41.1%** | | | |
| 10⁸ | | | | −3.0% | +1.5% | | |
| 10⁹ | | | | | −2.4% | +0.2% | |
| 10¹⁰ | | | | | | −4.8% | +0.0% |

| m = 0.06, bins/decade → | 12.7 | 18.7 | 25.1 | 27.3 | 37.0 | 54.3 | 80.3 |
|---|---|---|---|---|---|---|---|
| Γ = 10⁵ | +34.8% | | **−8.6%** | | | | |
| 10⁶ | | −12.3% | | | **+14.9%** | | |
| 10⁷ | | | | +0.2% | | +0.3% | |
| 10⁸ | | | | | | +0.1% | +0.3% |

⚠ **The requirement is non-monotonic in Γ, so it is not a power law of any index
or prefactor.** The decisive pair needs no modelling: at m = 0.3, 27.3
bins/decade is −41.1% at Γ = 10⁷ while **20.3 bins/decade is only −3.0% at
Γ = 10⁸**. Fewer bins at higher Γ is *better*. Same at m = 0.06: 37.0 bpd is
+14.9% at 10⁶ while 27.3 bpd is +0.2% at 10⁷.

So the requirement has a **peak** — near Γ ~ 10⁷ at m = 0.3 and ~10⁶ at
m = 0.06, i.e. it moves *down* by ~10× for a 5× lighter parent — and away from it
the production rule has roughly 2× of margin: every `x` cell above the peak is
within 0.3%, every `xc` twin at half the grid is 2–5% off.

⚠ **And that is why the lowest rung was not affected.** Γ = 10⁵, m = 0.3 is clean
at **six and a half** bins/decade. It is not that a slow decay needs few bins; it
is that the lowest rung sits *below* the resonance. m = 0.06 has no such rung
because its peak is at or below where its ladder starts — which is also why its
lowest two rungs are the contaminated ones there.

**The mechanism is not established.** The standing hypothesis is the spurious
ℓ = 1 eigenvalue of the lumped loss, λ ~ Γ·N_q^−³·⁶, integrated over the collision
window: λ rises with Γ while the window shrinks, which does produce a peak, and
the peak should then sit near Γ ~ H(a_nr), which moves the right way with mass
since a_nr ~ 1/m. But H(a_nr) comes out ~20× below the observed peak at both
masses. It gets the *shape* and the mass *ordering* right and the location wrong.

`GRID_RUNG` is the cheap test: six `_q2x` cells at double `dr_N_q` and nothing
else changed. Four sit on the measured-contaminated rungs and are the
corrections; two are **shoulders** (Γ = 10⁵ and 10⁸ at m = 0.3) that are already
clean at production resolution, so doubling must change nothing there. If it
does, the production rule has no margin anywhere and the whole ladder is in
question rather than four rungs of it. `collect.py --grid` reads the pairs.

*Already measured locally, Γ = 10⁵ / m = 0.3, 12.7 → 25.1 bins/decade: C_2^TT
moves −0.15%. The lower shoulder holds.*

⚠ The refine twin does **not** bracket this — both members of the pair are
contaminated where the resonance bites, so `xc` agreeing with `x` is not evidence
of anything there. Above ℓ ~ 30 none of it matters: the Γ = 10⁷ pair agrees to
0.16% over ℓ = 200–800. The damage is confined to ℓ ≲ 20 and k ≲ 10⁻³.

A doubled cell at Γ = 10¹¹, m = 0.3 was planned and is **deliberately not
included**: the paper's low-ℓ claim rests on `cmb_G11.0_m0.3_x`, and the finished
ladder has since answered it for free — the `x` cells at 10⁸, 10⁹ and 10¹⁰ read
+1.5%, +0.2% and +0.0%, so the rule tracks the requirement with margin everywhere
above the peak. The −5.1% at 10¹¹ is its `xc` twin, and every `xc` twin on that
ladder reads −2.4 to −5.1%. The cell would have cost of order 10 000 core-hours —
more than doubling the round — to re-measure what three rungs already agree on.

### What the refine pair actually measures

*Added 2026-09-14 from `hpc_refine` (12/12 cells: matched `h2`/`h1`/`q2` triples
at Γ = 10⁸, 10⁹, 10¹⁰, plus this round's `xc` twin reproduced on the same
binary). Binary drift between the two rounds, across #420 and #424, is **0.000%**
at every rung and band — the two `out/` trees are directly comparable.*

**The order is p = 4, measured.** Not assumed, and not the conservative 2 the
notes carry:

| Γ | p at ℓ<20 | spread | refine pair | ÷3 (p=2) | ÷15 (p=4, measured) |
|---|---|---|---|---|---|
| 10⁸ | 3.93 | ±0.12 | 0.826% | 0.275% | **0.071%** |
| 10⁹ | 3.98 | ±0.01 | 2.113% | 0.704% | **0.141%** |
| 10¹⁰ | 3.99 | ±0.03 | 3.947% | 1.316% | **0.266%** |

So the exact reference **is** converged at low ℓ, everywhere this round runs.
`rtanote`'s *"the low-ℓ ladder is not converged at all"* is the coarse cell's
number and should be deleted; `RICH_P = 3.3` in `rtanote/tools/mknumbers.py` and
the "order 2, which is conservative" captions make every grey band **5× too
wide**.

**The 10¹⁰/10¹¹ plateau is not an anomaly.** Those two refine pairs agree to
0.01% absolute despite `dr_N_q` 329 vs 583, which looked impossible at p = 4. The
missing term is `C` in `C·h^p`, which is constant in h but **not in Γ**. Held at
half the requirement at every rung, C measures 1.70e3 / 1.64e4 / 1.58e5 at
Γ = 10⁸/10⁹/10¹⁰ — **×9.6 per decade**, dead steady. From 10¹⁰ to 10¹¹ the
bins/decade rule refines h by 1.77×, worth 1.77⁴ = 9.88×, and the two cancel:
predicted E(10¹¹) = 3.91%, measured 3.94%. A coincidence of two power laws.

> Consequence for `BPD_PT_AT_1E7`: the rule is very nearly the **break-even**
> rule. It refines just fast enough to hold the low-ℓ error flat as Γ climbs, not
> fast enough to reduce it. Holding a fixed *absolute* error would need an index
> of log₂(9.6)/4 = **0.82** per decade, against the 0.17 in `config.py`.

**But the grid error does not cancel in the published ratio.** `plot_cl.py`'s
ratio to the lowest rung divides out the k-grid error — that part is sound, 0.2–
1.0% absolute down to 0.06%. The *daughter*-grid error is not common-mode: it
**changes sign** across the ladder (Γ = 10⁵ sits **above** the continuum limit,
10⁷ and up **below** it), so numerator and denominator errors **add**. At
Γ = 10¹⁰ vs 10⁵ that is 0.33% + 0.19% = **0.51% at ℓ = 2**:

| ℓ | 10⁹ raw → corr | 10¹⁰ raw → corr | 10¹¹ raw → corr |
|---|---|---|---|
| 2 | +0.120% → **+0.479%** | −0.037% → **+0.476%** | −0.383% → **+0.122%** |
| 4 | +0.303% → +0.617% | +0.155% → +0.637% | −0.152% → **+0.321%** |
| 10 | +0.951% → +1.072% | +0.933% → +1.186% | +0.876% → +1.123% |
| 20 | +1.644% → +1.696% | +1.909% → +2.006% | +2.122% → +2.217% |
| 30 | +1.793% → +1.780% | +2.285% → +2.264% | +2.684% → +2.663% |
| 50 | +1.261% → +1.174% | +1.975% → +1.841% | +2.608% → +2.475% |

**The ℓ ≈ 15–60 bump is solid** — the correction moves it by ≤ 0.10%, and the
acoustic-peak feature and damping tail are untouched. **ℓ ≲ 6 is not**: there the
correction exceeds the signal and flips its sign. Quote the low-ℓ ladder from
ℓ ≥ 15, or quote it corrected.

**An independent check, and the one thing it fails.** C₂ᵀᵀ is Γ-independent in
this model, so any spread across the ladder is grid error and a correct
correction must shrink it. For Γ = 10⁷–10¹⁰ it does: 0.225% → **0.075%**, from
data that knew nothing about C₂. That is independent confirmation of p = 4.

⚠ **Γ = 10⁵ and 10⁶ do not come into line.** They sit 0.5–0.6% below the others
and their corrections barely move them — at 10⁶ the `q2x` pair reports 0.001%,
i.e. "perfectly converged", while C₂ says 0.6% off. Either they are
**pre-asymptotic** at 12.7 and 18.7 bins/decade, in which case a 2-point pair at
an assumed p is invalid and *cannot detect this about itself*; or C₂ is not
exactly Γ-independent down there (it was only ever checked at 10⁵, 10⁹, 10¹⁰ —
never 10⁶). Nothing in this round separates them. **This is the ratio's
denominator.**

**The fix is three cells.** At Γ = 10⁵, 10⁶, 10⁷ the `x` and `x_q2x` cells are
already here, and `dr_N_q` = 93, 137, 201 are exactly `2N - 1` — so `q2` exists.
Only the half-grid `h2` leg is missing, at `dr_N_q` = **24, 35, 51** (each halves
the spacing exactly: 46, 68, 100 are even). On this round's own costing that is
~2, ~3 and ~4 charged core-hours. It is the cheapest decisive cell in the
campaign and it is small enough to run on a laptop — see *Completing the low
rungs locally*.

### Completing the low rungs locally

Run the **whole triple** locally, not just the missing `h2`. Three reasons:

1. `h2` alone would be compared against `x` and `x_q2x` built at
   `fix-reduced-table-lna-density`; master has since taken #411, #413, #420,
   #422, #424, #427, #429 and #431, and **#429 changed `l_max_dr_col`'s
   default** — which these inis do not set. One binary for all three legs
   removes the question instead of bounding it.
2. Richardson reads *differences* between the three. A build offset in one leg
   lands directly in the numerator of p.
3. It costs almost nothing extra. Per rung: `h2` + `x` + `q2x` ≈ 12 / 17 / 23
   work core-hours at 10⁵ / 10⁶ / 10⁷ — call it 100 core-hours for all three
   rungs with the ×2 the iso cells showed. On 11 cores that is one overnight.

The inis are self-contained; nothing in them needs the cluster. Point `root` at a
fresh directory — **not** `out/` — so the existing cells cannot be resumed over,
and keep the cluster `x` as a drift check: local `x` vs cluster `x` should be at
the 0.0016% laptop/cluster floor. If it is not, that is a finding in itself.

### Family D is deferred on purpose

`gen.py` writes `runfile_proxy` and nothing submits it. The proxy/RTA cells test
the fitted transport rate Γ_T(γ), and that fit is exactly what family B exists to
redo. Running them now would test the old fit. Sequence: family B → refit → add a
proxy ladder to `config.py` → regenerate.

---

## Family B solves only the k_output modes

`output = mTk` leaves `has_cls` false, so `k_max_cmb` collapses to `k_min`;
`P_k_max_1/Mpc = 1e-6` sits below `k_min` so `k_max` stays there too; the k list
is then a **single** mode, into which `k_output_values` are inserted. Measured:
7 requested values give exactly **8 modes** (the 7 plus `k_min`).

Two consequences the config depends on:

* **`cpus` must equal `N_K_ISO`.** One mode per core, so the wall time is the
  *slowest single mode* — and every mode up to the core count is then free in
  wall time. That is why there are 7 k values and not 3: more γ-bins for the fit
  at no cost. `gen.py` warns if `cpus != N_K_ISO`.
* **`momenta_bins = 32` here, 16 in family A.** The a4Pi wiggle is O(h²) in the
  *parent* grid (1.449 / 0.374 / 0.080 at 16 / 32 / 64) and family B's whole
  deliverable is a projection onto a4Pi. This makes A and B non-comparable
  cell-for-cell; that is accepted, because the Galerkin validation lives entirely
  inside A and C.

### The convergence twin is not optional here

Every cell has a `refine = 2` twin: the same background under a halved
perturbation grid. It is *cheaper* than production, not dearer.

⚠ **The pair does not bracket the answer** — this file said it did, and that was
wrong. Both members sit on the *same side* of the continuum limit, and
|f(2h) − f(h)| is the error of the **coarse** member alone. The production cell's
is that divided by `2^p - 1`, measured at **15**, not 3. See *What the refine
pair actually measures* above.

> **A coarse grid does not add scatter to the fitted exponent — it fakes a
> steeper one.** A 4× coarser grid moves p by **+0.96** on average and an 8×
> coarser one by **+4.9**. Any under-resolved run looks as though it has found
> γ⁻⁵, which is precisely the result this round exists to test.

`collect.py --iso` prints `dp(coarse)` — the production cell against its twin —
next to every fitted p. Where that is large, p is a discretisation artefact.

### The dumps stay on the cluster

k_output dumps are one row per *evolver step*: 0.7–1.4 GB gzipped per cell at
Γ = 10¹¹, and the step count is what grows with Γ. `reduce_iso.py` runs **on the
cluster** and reduces each cell to a few kB of γ-binned projection in `derived/`,
which is all any figure reads. `sync_from_hpc.sh` excludes the dumps by default;
`--with-perturbations` fetches them when a cell needs checking by hand.

---

## Precision: 121 k modes instead of 297

Tuned locally on ΛCDM in `tune/` (`ktune.py`), because the k sampler in
`perturbations_module.cpp` never looks at the species list, and with DNCDM the
perturbation k-loop is >99% of runtime — so the *only* precision knobs that
change what a cell costs are the ones that change how many k modes are solved.

Gate: **≤ 1% absolute** in the lensed spectra over 2 ≤ ℓ ≤ 2000 and in P(k),
verified at both N_eff = 5.1 and N_ur = 1.0176.

| | n_k | TT | EE | TE | EE(ℓ<30) | P(k) |
|---|---|---|---|---|---|---|
| ΛCDM, N_eff 5.1 | 113 | 0.26% | 0.48% | 0.43% | 0.48% | 0.39% |
| ΛCDM, N_ur 1.0176 | 121 | 0.31% | 0.29% | 0.60% | 0.29% | 0.29% |
| DNCDM Γ=10⁵ † | 126 | 0.20% | 0.58% | 0.95% | 0.58% | 0.34% |
| DNCDM Γ=10⁶ † | 126 | 0.20% | 0.59% | 0.96% | 0.59% | 0.34% |

† measured with the pre-tail-cut preset, which was 0.80% worst-case in ΛCDM
against the final one's 0.60% — the conservative reading. That the DNCDM rows
land on top of the ΛCDM rows is the evidence that ΛCDM was the right place to
tune the k grid at all.

**And the number that actually matters.** The published quantity is a ratio, in
which the common-mode k error cancels:

> max error in C_l(10⁶)/C_l(10⁵) over ℓ = 2…2000: **TT 0.06%, EE 0.02%** —
> 10–30× below the absolute error the gate was written against.

### Four things the scan settled that are not obvious

* **Waves, not modes, are the cost.** The k loop fans out over a task pool, so a
  cell costs `ceil(n_k / cpus)` waves. At 64 cores, 129 modes and 192 modes cost
  the *same* wall time. The target was therefore "the most accurate preset that
  still fits two waves", not "as few modes as possible".
* **`k_step_super` stays at its default.** `precision_fast.pre`'s 0.02 puts
  C_l^EE at ℓ < 30 out by **43%**, and even 0.012 gives 10%. The entire saving
  comes from `k_step_transition`, `k_step_sub`, `k_max_tau0_over_l_max` and
  `k_min_tau0`.
* **k_max is the *product* `k_max_tau0_over_l_max × l_max_scalars`.** Cutting
  both compounds: 1.3 × 2250 put an irreducible **1.2%** floor under TT that no
  k-step choice could get below, while either cut alone reads 0.14–0.18%.
  `l_max_scalars` and `delta_l_max` therefore keep their defaults — they are
  worth 13 modes out of 297.
* **The k tail above the acoustic grid is nearly free to cut.** Requesting `lCl`
  switches on the full-Limber lensing grid, which extends the *source* k range to
  `k_max_limber_over_l_max_scalars × l_max_scalars` = 1.25/Mpc — five times past
  `k_max_cmb`, and the most expensive modes in the run. Measured one at a time:
  `k_per_decade_for_bao` 70→10 and `k_per_decade_for_pk` 10→3 cost **nothing at
  all** (metrics identical to four decimals); truncating the Limber grid saves 18
  modes for 0.04%. Cutting it also stops P(k) at ~0.37 h/Mpc — exactly the range
  the CMB needs anyway, rather than carrying modes out to 2.29 h/Mpc that nothing
  published uses.

`tune/` is **not** pushed to the cluster (`sync_to_hpc.sh` excludes it). Its only
output that matters is the preset, and that lives in `config.py`.

---

## ⚠ The old `hpc_*` outputs are not mixable with these

Merged master's DNCDM knob set is **smaller** than the campaign branches':
`lump_limiter`, `deflate_collision`, `dr_reduced_weight`, `dr_reduced_table` and
`dr_reduced_table_stride` no longer exist — the behaviours they selected are now
hardwired (limiter on, no deflation, table always used).

A pre-merge ini runs on master with `rc = 0` and means something different,
because **CLASS silently ignores keys it does not recognise** — no warning, no
error, not even at `input_verbose = 2`. Verified directly.

The old cells stay useful as cross-checks. They must not appear in a figure
beside these ones.

## The guards

All in `runcell.sh`, shared by every family so they cannot drift apart.

1. **The binary is `EXPECTED_COMMIT`.** `build.sh` records what it built from;
   a mismatch stops the cell. Authoritative, so it runs first.
2. **The binary understands this campaign's DNCDM key set** — a *functional
   probe*, not a `strings` scan. It feeds CLASS one background-only cell carrying
   the keys the campaign depends on and fails if any comes back unread. Costs
   well under a second and fails fast, before a 15-day job starts.
   *Verified both ways: passes on master, and rejects a real pre-merge binary
   (`class_campaign_0428`) naming `dncdm1.dr_reduced_moments`.*
3. **Every key in the ini was read.** Every ini carries `write warnings = yes`;
   if CLASS reports an unread key the cell **fails** and its output is moved
   aside as `REJECTED`. This catches typos too, which a hand-maintained blacklist
   never would. *Verified: all five ini flavours read every key on master.*
4. **The daughter grid held the emission band.** `dr_q_max` below q\* is an energy
   *leak*, not a resolution trade-off — the two-bin deposit clamps into the top
   bin, conserving number and filing the energy at `q_max` instead of at q\*, and
   it is silent because the identities the kernel asserts are number identities.

Plus resume-safety: a finished cell is skipped only if the ini that produced it is
**byte-identical**. The tag encodes (family, Γ, mass, scheme, rung) but *not* the
grid, so changing the bins-per-decade rule leaves every tag unchanged and a bare
"output exists" test would keep the stale result.

### Why guard 2 is a probe and not a `strings` scan

The first version grepped the binary for `dr_reduced_moments` and refused to run
if the literal was absent. **That rejected good binaries.** Whether a key literal
survives into `.rodata` is a *compiler* decision, not a property of the code: in
the very same binary `dr_q_min` and `dr_q_max` do not appear as standalone strings
at all — only inside longer diagnostic messages, their own literals merged away —
while `dr_reduced_moments` happens to survive under clang and need not under
GCC 14.2 with `-march=cascadelake`. Asking the parser is the only check that
tests the property that actually matters.

### About `dr_q_max = 50`

CLASS warns that it sits below q\*(a = 1) — for m = 0.3 that is q\* = 899. **That
warning is a false alarm on this ladder**, and it is worth knowing why rather than
suppressing it: the check evaluates q\* at a = 1 unconditionally, but the leak
needs the parent to still be *decaying* when a·M/2 outgrows the grid. At Γ = 10⁵
the parent is extinct by a ≈ 2.5×10⁻³, where q\* ≈ 15. `collect.py --bg` re-does
the check properly, per cell, against the *measured* `a_ext` — it does not trust
this paragraph.

---

## If you are here to ANALYSE, not to run

Everything above this line is how the round was built and submitted. This
section is the entry point once the jobs have finished. It is deliberately in
this file rather than a separate hand-over note: two documents drift, and the
traps that matter are already written down above.

### The four questions this round exists to settle

1. **The C_l and P(k) ladders** (family A). The published quantity is a **ratio**,
   `C_l(Γ) / C_l(Γ = 10⁵)` — the lowest rung is the denominator, not a data point,
   which is why it exists at both masses. The ratio is what the precision preset
   was justified against: 121 k modes give 0.20–0.96% absolute but **TT 0.06%,
   EE 0.02% in the ratio**, because the k error is common-mode. Never quote the
   absolute spectra at that precision.
2. **The Galerkin closure error** (families A + C). `dr_reduced_moments` 4 / 6 / 8
   against the exact scheme at matched (Γ, mass, grid). This is what earns the
   reduction the right to carry the ladder past where the exact scheme stops.
   ⚠ Every Galerkin cell in `hpc_prod` was divergent; this is the first round in
   which this question has a valid answer at all.
3. **The isotropisation rate** (family B). `a4PiDot → Γ_T(γ)`, and specifically
   the exponent `p` in `Γ_T ∝ γ^-p`. The question is whether `p` reaches ~5 or
   stalls near 3. ⚠ **A coarse perturbation grid does not scatter `p`, it fakes a
   steeper one** (+0.96 for 4× coarser, +4.9 for 8×) — i.e. it manufactures
   exactly the γ⁻⁵ this exists to test. `dp(coarse)` is the gate.
4. **The daughter-grid requirement** (`GRID_RUNG`). Whether the resonance picture
   above holds, and how big the correction is on the four contaminated rungs.
   Read this **first**: it decides whether the low-ℓ end of question 1 is usable.

### What each cell leaves behind

`out/<tag>_*`, gzipped where large. `<tag>` is
`<family>_G<log10Γ>_m<mass>_<scheme>[c][_suffix]` — trailing `c` is the
`refine = 2` convergence twin, `_q2x` is a `GRID_RUNG` cell, `_mb16`/`_mb64` the
momenta rung.

| file | what | families |
|---|---|---|
| `…_cl.dat`, `…_cl_lensed.dat` | C_l. **Use lensed.** Columns TT EE TE BB φφ Tφ Eφ | A, C |
| `…_z1_pk.dat` / `_z2_` / `_z3_` | P(k) at **z = 0 / 49 / 99** (`Z_PK` order) | A, C |
| `…_z1_pk_cb.dat` | P(k) for baryons+CDM only — the P_m vs P_cb convention matters here | A, C |
| `…_z1_tk.dat` | transfer functions; `phi` is the divergence canary | A, C |
| `…_tk.dat` | the k_output dump's companion | B |
| `…_bgslim.dat` | background table with the per-q PSD block dropped | bg |
| `…_parameters.ini` | **every** parameter actually used, defaults included | all |
| `…_unused_parameters` | guard 3's evidence. Should contain only `write warnings` | all |
| `…_class.out` | the run's stdout | all |
| `derived/<tag>__k<i>.json` | family B reduced to a few kB — **the only family-B data that comes home** | B |

`derived/*.json` per (cell, k): cell metadata (`tag`, `k`, `Gamma`, `mass`,
`scheme`, `moments`, `refine`, `dr_N_q`, `momenta_bins`) plus `rows`, one per γ
bin, each carrying `gamma`, `a`, `ktau`, `n`, the fitted `GT` and its per-leg
`GT_H` / `GT_l` / `GT_phi`, `R2`, `corr`, `cancel` and `frac_support`.
`derived/sweep.json` is all of them concatenated.

⚠ **`GT > 0` means relaxation.** Do not negate it — that convention error is what
made `hpc_prod`'s `--iso` print `nan` for every `p`.

### What counts as a result

| check | command | acceptable | what a failure means |
|---|---|---|---|
| daughter grid | `--grid` | shoulders ~0%; contaminated rungs move | the whole ladder is in question, not four rungs |
| refine pair | `--cl` | **divide by 15 first** (p = 4 measured); then below the k preset's own 0.20–0.31% TT, over ℓ ≤ `L_PUBLISH` = 2000 | raise `BPD_PT_AT_1E7`, do not add rungs |
| Galerkin closure | `--gal` | 0.05–0.11% over ℓ = 200–800 was the local measurement | the reduction cannot carry the ladder |
| background | `--bg` | `Ecomov` → 1.23, smooth in Γ; `a_ext` clears `dr_q_max` | energy leak; see *About `dr_q_max = 50`* |
| isotropisation | `--iso` | `dp(coarse)` ≲ 0.14; `frac_support` = 1 | `p` is a discretisation artefact, not a rate |

Then read *Two things `--iso` reports that are easy to miss* below, before
trusting a single fitted `p`.

### The plotting scripts are rudimentary on purpose

They exist to eyeball a round, not to make figures. `plot_bg.py` (ρ_i/ρ_tot vs z
per bg cell), `plot_cl.py` and `plot_pk.py` (absolute spectra **and ratios to the
lowest Γ rung** — the published quantity), `plot_iso.py` (Γ_T(γ) from
`derived/*.json`). `CPU.py` is the cost bookkeeping.

⚠ **No paper figure reads this round.** `tools/plib.py` still hard-codes
`hpc_cmb` and `hpc_reduced`, which are pre-merge matrices whose knobs no longer
exist. Pointing it here is a deliberate act, not a path edit — see *The old
`hpc_*` outputs are not mixable with these*.

### Where to be suspicious

Read *Known weak points* at the end of this file first. The short version: the
daughter-grid rule's functional form is measured wrong and only patched at six
rungs; `COST_M6` is derived from divergent cells; `GAMMA_CEILING` is the largest
Γ that has ever produced a background, not a physical limit; and family D
(proxy/RTA) is **absent by design** until family B refits Γ_T — running the old
fit's cells would test the old fit.

## Reading the results

```
python3 collect.py --status   # LIVE state first, then completion, measured vs
                              # estimated cost, and the cost-table rows to paste
                              # back into config.py
python3 collect.py --bg       # Ecomov, N_eff, a_ext, dr_q_max adequacy
python3 collect.py --cl       # C_l / P(k) convergence down each refine pair
python3 collect.py --grid     # the doubled daughter grid against production
python3 collect.py --gal      # Galerkin closure error vs the exact scheme
python3 collect.py --iso      # fitted p, R^2, and dp(coarse) — the gate
```

`--status` needs only the logs and works without numpy; the others read data
tables and need it.

### `--status` opens with what is still actionable

```
── live
   family    done  running  queued  failed  not-sub    of
   cmb         28        1      25       0        4    58
   iso         91        1       0       6       20   118

   RUNNING                        elapsed      est  of est  of wall   part
   cmb_G11.0_m0.3_x                 65.2h    97.0h     67%      18%    q64

   QUEUED (25): cmb_G05.0_m0.06_m6, ... (+21 more)
   waiting on: (QOSMaxJobsPerUserLimit)
```

It reads `squeue` directly, so it only shows anything **on the cluster** — run
locally it says so and falls through to the log-derived completion, which is
still authoritative for what finished (the logs are what `sync_from_hpc.sh`
brings home).

* **`not-sub`** is the useful column after a partial submission or a `scancel`:
  cells with no log and nothing in the queue. That is what is left to submit.
* **`of wall`** is the one that costs money. A cell killed at the Slurm wall
  produces *nothing* — no output, no resume point — and its charged core-hours
  are gone. Past 85% of the wall `--status` says so explicitly, with the estimate
  beside it, because the decision (let the rung go, or `scancel` and requeue it
  coarser) has to be made before the wall and not after.
* Array tasks map to tags through the **log**, not `runfile_<family>` — `gen.py`
  rewrites the runfile and a running array still refers to the lines it was
  submitted against. A *pending* task has no log yet, so it falls back to the
  runfile; if you re-ran `gen.py` after submitting, those names are a guess and
  the running ones are not.

⚠ **`--iso`'s sign bug is fixed here.** In `hpc_prod` it printed `nan` for every
`p`: `reduce_iso.py` stores `GT > 0` for relaxation and `fit()` negated it before
masking on `y > 0`, discarding every physical record. 637 records came back on
dropping the minus. If it ever prints all-`nan` again, check `reduce_iso.py`'s
convention rather than re-adding the negation.

### Two things `--iso` reports that are easy to miss

* **`gsup`** — where the collision term's support ends. A *reduced* cell applies
  the operator only inside its table window, so `a4PiDot_coll` is exactly zero
  outside it. Measured at Γ = 10⁵ the zeros sit entirely at γ = 1.000, long after
  the decay, so the fit window is unaffected — but a bin of exact zeros fits
  Γ_T = 0 and *looks like a converged measurement*. Any row whose `gsup` is below
  the window top is not a measurement.
* **`cancel`** (in `derived/`) — how much of Γ_T is cancellation between the
  parent/ν_l/φ legs. Far below 1 means Γ_T is a small residual of much larger
  terms and its error bar is not its components'.

---

## Files

| file | what |
|---|---|
| `config.py` | **the single source of truth** — matrix, precision, cost model, Slurm |
| `gen.py` | config.py → `ini/`, `runfile_*`, `manifest.csv`, and every `#SBATCH` line |
| `build.sh` | one binary per partition, with that partition's `-march`; records the commit |
| `runcell.sh` | runs one cell: binary selection, the four guards, resume, post-processing |
| `submit_{bg,cmb,iso,gal}.sh` | `#SBATCH` headers only; they all `exec ./runcell.sh` |
| `reduce_iso.py` | **cluster-side** — family B dumps → `derived/*.json`, a few kB each. Fixed-width reader + `-j` (3.5× per dump × cores); `--probe` says whether the filesystem or the CPU is the wall |
| `collect.py` | live state, cost, convergence tables, verdicts — the analysis entry point |
| `slim_background.awk` | drops the per-q PSD block from the background table (~GB → ~MB) |
| `plot_{bg,cl,pk,iso}.py` | rudimentary eyeball plots; `cl`/`pk` also plot the **ratio to the lowest Γ rung**, which is the published quantity |
| `CPU.py` | cost bookkeeping |
| `sync_{to,from}_hpc.sh` | push the harness / pull the results. The GB-scale dumps stay on the cluster unless `--with-perturbations` |
| `tune/` | the local ΛCDM k-grid study. Not pushed to the cluster. |

`q192` does not exist on this cluster — every earlier `hpc_*` config carried a
`q192` row, which was a typo for `q144`. Fixed in `config.py` and `build.sh`.

## Known weak points

Stated because they are the things most likely to be wrong, not because they are
expected to be:

* **The ratio's denominator, Γ = 10⁵, is the least-verified cell on the ladder.**
  It runs at 12.7 bins/decade — the coarsest grid in the campaign — it divides
  every published curve, and it is one of the two rungs whose convergence order
  has never been measured. `C₂ᵀᵀ` says it and 10⁶ are 0.5–0.6% off while their
  own `q2x` pairs call them converged. Three `h2` cells settle it; see *What the
  refine pair actually measures*.
* **`BPD_PT_AT_1E7`'s index is too shallow, though its functional form now
  stands.** The mid-ladder "resonance" that made it look *wrong* was
  `back_integration_stepsize`, not the grid, and it is gone (see the SUPERSEDED
  note above). What remains is that the error constant grows ×9.6 per decade of Γ
  while index 0.17 refines h by only ~1.5×: the rule holds the low-ℓ error flat
  rather than reducing it, and a fixed absolute target needs ~0.82.
* **m = 0.06 eV has no daughter-grid control above Γ = 10⁶.** Its `q2x` cells stop
  there and its requirement carries a 2× mass factor, so its low-ℓ ratio is
  uncontrolled at exactly the rungs where m = 0.30 eV needed the correction.
  Nothing in `hpc_refine` transfers to it.
* **The Galerkin cost table is now a measurement, and it is a surprising one.**
  `COST_M6` is ~0.5 core-hours flat in Γ, because a reduced cell *is* its
  background solve. That is derived from cells that were diverging, so it is an
  upper bound; the first clean Galerkin cell will confirm or shrink it, not
  raise it.
* **`GAMMA_CEILING` is measured, not modelled.** It is the largest Γ at which a
  background pre-flight has actually finished on this cluster. The extrapolation
  above it (~2.8× per half decade) is an ordering hint.
* **Everything above the measured Γ range is an ordering hint, not a budget.**
  The cost model extrapolates at the local slope of its top two entries.
  `collect.py --status` prints replacement rows as cells land.
* **The `momenta_bins = 16` vs `32` seam** between families A and B is deliberate
  but real. `MOMENTA_RUNG` in `config.py` runs the same family-B cell at 16 and
  64 parent bins (Γ = 10⁹, m = 0.3) so that the production choice of 32 is
  justified against the *fitted exponent* rather than against the a4Pi wiggle
  amplitude it was originally picked on. If p moves, that rung is the finding.
