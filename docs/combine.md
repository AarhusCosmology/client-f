# Domain splitting: `client.py combine`

A run on the whole box can miss a mode for good. Its chains follow the network, and a network
that rates a region far too low never sends a point there. The axion monodromy run on Planck TT
is the case in point. A long MCMC without an emulator puts 1.8% of the posterior on a ridge at
ω < 2 and A_osc > 0.015, where n_s, ω_cdm and A_s trade off against the oscillation. Every network
of the full-box run was 120 to 280 log-units too low there. Neither a hotter boost, a true
profile, nor that profile's points added to the training set brought it back
([boost.md](boost.md), "A region the networks miss").

Domain splitting runs CLiENT more than once on the same objective, each time with the training
box cut down in one parameter, and then combines the networks. In a box cut down to such a mode,
the mode is the main feature of the posterior, and the ordinary loop finds it by itself. A run
on the axion likelihood with ω in [1, 3] did exactly that:

- At its eighth network its best training point lay in the corner, 3.8 below the global best.
- After one boost its network matched the true values at the long MCMC's corner points to 0.15
  to 0.18 rms.
- Its region held all of the long MCMC's corner points.

## The runs

Each part is an ordinary run with the same likelihood and the same prior. Only its training box
differs, and only in the one parameter.

With the launcher in [`hpc/`](../hpc/README.md), give each part a byte-identical copy of the
full run's Cobaya file. The prior on the split parameter must stay the full one: a narrower prior would shift the objective
by the logarithm of the ratio of the prior widths. Give each part its own `domain_bounds.json`.

Choose the cut by the likelihood, not by where a network ran short of points. For the axion run,
nothing fits well between the corner at ω ≈ 1 to 2 and the oscillation peaks from ω ≈ 10 up. The
cut is therefore at ω = 3, and the full-box run supplies everything above it.

## The command

```bash
python client.py combine OUTPUT --parameter omega \
    --part results/<timestamp>_low_omega --part results/<timestamp>_full_box:18 \
    --band 2.6:3.0
```

**Arguments.**
- `--part` takes `RUN[:N]` (default N: the latest network). List the parts in increasing order of
  the parameter.
- `--band LO:HI` marks where two adjacent parts are blended. There is one band fewer than parts.
- `--no-walls` lets each network be used outside its own region; not recommended.

**What the command checks**, refusing with a message when a check fails:
- The parts record the same objective. Their `objective_identity.json` must agree in everything
  except the split parameter's bounds.
- Their boxes differ in that parameter only.
- Every band lies inside both adjacent boxes.
- Every part covers the whole range in which it will be used.
- Either every part has a saved region, or none does. Mixing restricted and unrestricted
  parts is rejected, including with `--no-walls`: an outer union of only the restricted
  parts would silently truncate the others in evidence, sampling and profile consumers.
- All saved regions have the same inflation edge.

The command first copies the consumed files into temporary snapshots and verifies their
hashes and the selected network/region before and after copying. A change during copying
is rejected; later changes to a source cannot affect the combination. This includes the
metadata, objective identity, settings, refusal value, chain summary and recursive
`likelihood_input/` files, not just the network and training data. `combine.json` records
the original source paths and the hashes of those consumed bytes (`parts[].artifacts`).
The output is staged beside its destination and published by one directory rename only
after it is complete, including its `combine.json` marker. Failure leaves no partial run;
a nonempty destination is never overwritten.

## The combined network

OUTPUT is a new directory that reads as a finished run with one network, 0.

**How the value is assembled.**
- Below the first band, part 1's network is used. Above the last band, the last part's is used.
- In a band the value is (1 − s) A + s B. Here s is the quintic smootherstep
  6t⁵ − 15t⁴ + 10t³ of the parameter's position t in the band (`model.network.SmoothStep`).
- The smootherstep's first and second derivatives vanish at the band's ends, so the profile code
  can take curvatures across it.
- Outside the band both weights are exactly 0 or 1, so the value there is exactly one network's.
  The form A + s(B − A) rounded it in float32 by up to 0.12 next to a wall.

**Walls.** Each part's network is seen only inside its own boost region (`RegionWall`, inside the
combined model). A network has maxima outside its region. One part's region may also reach into
the range where another part's network is used, and with the walls that part is cut off there
too.

**Files.**

| file | content |
|---|---|
| `trained_models/model_it_0.keras` | the combined network: one flat Keras graph, with every part's layers copied under `part<k>_` and the same weights (the profile code copies a model layer by layer at double precision) |
| `boost/region_it_0.npz` | the union of the parts' region cells, re-expressed in the combined box's units, with identical ellipsoids represented once |
| `training_data/data_it_0.csv` | every part's training points that have a value; refused ones are left out |
| `metadata.json` | the combined box: the split parameter's widest range, every other range as the parts' |
| the YAML, `likelihood_input/`, `refused_value.json` | from the part with the widest box |
| `objective_identity.json` | that part's, with the combined bounds |
| `convergence_stats/chain_summary_it_0.npz` | pooled from the parts' (`hpc/analysis/report.py` samples only networks that have one) |
| `combine.json` | the parts (run, network, box, hashes of model, data and region), the bands, the walls |

**What reads it.** The evidence, `hpc/analysis/report.py` and `true_check.py`, the profiles and
`client.py modes` read the directory as any run. It is not a run to continue: the loop refuses a
directory with `combine.json`. Continue its parts and combine again. To use it outside CLiENT,
export it (`client.py export`, [export.md](export.md)): the `.keras` file needs CLiENT's layer
classes, the exported SavedModel only TensorFlow.

## Sampling it, and the weights of the parts

Both networks emulate the objective itself, not a normalised posterior. The parts' relative
weights therefore need no evidence ratio: they follow from the values, as far as each network is
right near its own posterior.

- **Chains.** Distinct union cells use CLiENT's cell-by-cell path
  (`sampling/mixture.py`). Each cell is sampled by itself and weighted by its own evidence,
  and no chain has to cross between the parts. Identical ellipsoids from different parts
  are merged in this outer region: otherwise nearest-ellipsoid tie-breaking leaves an
  empty second cell that cannot be sampled. Equality compares centres and normalised
  quadratic forms after reboxing, allowing only floating-point roundoff (32 machine
  epsilons); general overlaps remain separate. `region_cell_sources` records the source
  cells represented by each union cell; their diagnostic row counts are summed. The
  individual network walls remain unchanged. A one-cell union uses ordinary sampling.
- **The evidence.** BEST integrates the combined network over the whole combined box, which for
  the axion run is the full run's box. Its ln Z is therefore comparable with the full run's.

**First trial** (19 October 2026), with the ω < 3 run's network 12 (after one boost) and the full
run's network 18, blended over ω in [2.6, 3.0]:
- **Consistency checks.** Below the band the combined network equals the ω < 3 network exactly,
  and above it the full run's network exactly. In the band it equals the blend to 3 × 10⁻⁵. The
  union region reproduces each part's distances to 2 × 10⁻¹⁶.
- **Against the long MCMC.** The median of network minus truth is +0.01 in the corner (rms 0.15
  and 0.18) and −0.13 at ω ≥ 5.
- **Sampling.** It gave the ω < 3 cell 3.9% of the posterior, against the 4.5% that
  the long MCMC puts at ω < 3. The full run's network alone had 0.5% there.

## The axion run, combined, against an independent MCMC

On the cluster (19 October 2026) the ω < 3 run's network 13, after two of its boosts, was
combined with the full-box run's network 18, blended over ω in [2.6, 3.0].
`hpc/analysis/report.py` then sampled the combined network: 85 504 samples at T = 1, cell by cell. The reference is a long MCMC
of the same likelihood (30 000 samples, true likelihood, no emulator), run independently of CLiENT.

| share of the posterior | combined | independent MCMC | full-box network 18 alone |
|---|---|---|---|
| ω < 2 | 3.69% | 3.57% | 0.25% |
| ω < 3 | 4.36% | 4.52% | 0.53% |
| the corner: ω < 2, A_osc > 0.015 | 2.14% | 1.76% | 0.00% |
| ω 3 to 8 (the full-box network's range) | 1.89% | 2.56% | 1.96% |
| main peak, ω 30.5 to 33 | 28.6% | 29.2% | 30.8% |
| A_osc, 95% upper limit | 0.0296 | 0.0309 | 0.0290 |
| n_s, mean ± sd | 0.9714 ± 0.0064 | 0.9712 ± 0.0057 | 0.9708 ± 0.0049 |

- **The mass below ω = 3 is back.** It agrees with the MCMC to about 0.2 percentage points; the
  full-box network had an eighth of it.
- **n_s widens** as the MCMC's does, because in the corner it trades off against the
  oscillation.
- **The A_osc limit and the main peak's share move towards the MCMC.**
- **Remaining differences:**
  - The corner is slightly overweighted and n_s slightly too wide. That is with a corner network
    after two boosts.
  - ω 3 to 8 is a little low. That range comes from the full-box network, which this split
    leaves in charge there.
- **The cell weights** were 4.1% for the ω < 3 run's cells and 95.9% for the full-box run's. The
  corner cell's bridge efficiency was 0.090, below the 0.1 that marks a cell's weight "not
  reliable" (Limits, below).

## Limits

- **The user must know where to split.** Nothing in CLiENT detects a mode that every network
  misses. Here an independent chain showed it.
- **Cell weights may be marked "not reliable".** The bridge's Gaussian-mixture proposal fits a
  ridge badly: the axion corner cell had an efficiency of 0.063, below the 0.1 that marks a cell.
  Where that happens `hpc/analysis/true_check.py` withholds its corrected moments, by the multi-cell rule
  of [AGENTS.md](../AGENTS.md). BEST's evidence is not affected.
- **An outside region can cut a part's posterior.** A point inside the combined region but outside
  its own part's region gets that part's wall. In the trial, about 1% of the long MCMC's points at
  ω < 3 lay outside the ω < 3 run's region after its first boost. The region grows with the boosts.
- **No derived parameters.** The derived-parameter emulators are not combined.
- **Tests:** `tests/test_combine.py`.
