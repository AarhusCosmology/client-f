# Boost: deciding the modes by measured mixing

A boost divides the posterior into modes "that a T = 1 chain cannot travel between" and
gives each its own chain, its own evidence and the same number of points
([boost.md](boost.md)). By default two maxima are one mode when the straight path between
them dips less than `barrier` (4 log-units). Two things are wrong with a fixed number:

* how high a barrier a chain crosses depends on the number of parameters, and
* a straight path between two maxima of a curved mode crosses ground the chain never visits.

`mixing` replaces the number by the thing it stands for: a chain is run and asked whether it
travels. It is opt-in; without the key the modes are found by the fixed barrier.

```yaml
acquisition:
  boost:
    n: 2000
    start_iteration: 15
    mixing: 0.1            # two basins are one mode when a T = 1 chain keeps at most this
                           # much memory of which of them a walker started in
    # candidate_barrier: 1.0   # log-units; maxima closer than this are one basin without a test
```

The same analysis of any saved network, outside the loop and without the likelihood:

```bash
python client.py modes RUN_DIR [-i N] [--mixing M] [--barrier-only] [--tempered-steps S] [--seed S] [--tag T]
```

## What is done

The first step of a boost is the same: an ensemble started on the tempered chain is cooled to
T = 1 and every walker climbs to its local maximum. Then, in place of the grouping by
`barrier`:

1. **Candidate basins.** The maxima are grouped as by the fixed rule, but with the low
   `candidate_barrier` (1 log-unit: dips under it are the network's own wiggles, and no chain
   is needed to know that they are crossed). Nothing is dropped at this stage. A climb that
   stalled on a slope goes to the basin of the highest maximum it is joined to. A basin with
   less than `min_share` of the walkers is too small to stand for itself: it joins the large
   basin with the smallest dip to it when that dip is under `barrier`, and is otherwise left
   out and counted.
2. **One basin: one mode.** No chain is run, and the test costs nothing.
3. **A free chain.** With several basins, one T = 1 chain of the boost's own `walkers` and
   `steps` is started on the cooled walkers and left free in the whole box.
4. **Where the walkers are.** At eight times in the first half of the chain, and half a chain
   later, every walker's position is given a basin by letting it climb: it is in the basin of
   the maximum it reaches (the basin to one of whose maxima the straight path from there dips
   least; up to 48 maxima stand for a basin).
5. **Memory.** For two basins A and B, among the walkers found in A or B at the later time,

       memory = P(in A later | in A before) - P(in A later | in B before).

   It is 1 when no walker changes basin and 0 when where a walker is no longer depends on where
   it was. It is measured half a chain apart and squared, which is what is left after the whole
   chain if the memory decays exponentially.
6. **Decision.** Basins whose memory after the chain is at most `mixing` are one mode; this is
   applied to every pair, so basins that mix through a third are one mode too. The modes are
   numbered by size.

Then the boost goes on as with the fixed barrier: one confined chain and one bridge-sampled evidence per mode,
the same number of points from each. A mode made of several basins has as its region the union
of theirs. The Gaussians that mark the regions are fitted to the cooled walkers and to the
labelled positions of step 4 together.

The threshold has a direct meaning. For two basins the memory is also the factor by which an
error in the starting split between them has shrunk by the end of the chain. With
`mixing: 0.1` two basins share a chain when that chain forgets at least nine tenths of its
starting split; otherwise their weights are taken from the two evidences, which do not depend
on the split at all.

The test is about the chain the boost samples with: the same sampler, number of walkers and
number of steps. A longer chain mixes across more, and the answer changes with it. That is
intended.

## Measurements

Every number below is from the whole of `posterior_modes` run as a boost runs it (512 walkers
cooled from a sample at T = 7, 10 000 steps), once with the fixed barrier of 4 and once with
`mixing: 0.1`.

### Two unit Gaussians a known barrier apart

The barrier is the dip of the straight path between the two maxima, which here is the true
one. Number of modes found, fixed barrier → mixing, and in brackets the memory left after
the chain:

| barrier | 16 parameters | 58 parameters |
|---|---|---|
| 0.5 | 1 → 1 (one basin) | 1 → 1 (one basin) |
| 2 | 1 → 1 (0.00) | 1 → 1 (0.00) |
| 3 | 1 → 1 (0.00) | 1 → 1 (0.03) |
| 4 | 1 → 1 (0.00) | 1 → **2** (0.15) |
| 6 | 2 → **1** (0.01) | 2 → 2 (0.51) |
| 9 | 2 → 2 (0.34) | 2 → 2 (0.91) |

With 16 parameters the chain still forgets its start completely across 6 log-units, and the
fixed rule divides a mode that one chain samples well. With 58 parameters it no longer mixes
across 4, and the fixed rule leaves two modes to one chain: in that case the single chain put
48.5% on one side where the exact share is 50%, and the two evidences give 50.2%.

The same with the second mode holding 10% and 2% of the posterior:

| second mode | parameters | barrier 3 | barrier 6 | barrier 9 |
|---|---|---|---|---|
| 10% | 16 | 1 → 1 (0.00) | 2 → **1** (0.00) | 2 → 2 (0.16) |
| 10% | 58 | 1 → 1 (0.06) | 2 → 2 (0.62) | 2 → 2 (0.91) |
| 2% | 16 | 1 → 1 (0.00) | 2 → **1** (0.00) | 2 → 2 (0.15) |
| 2% | 58 | 1 → 1 (0.06) | 2 → 2 (0.48) | 2 → 2 (0.85) |

The decision does not depend on the weight of the smaller mode down to 2%. Where both rules
find two modes, the weights with `mixing` are at least as close to the exact ones, because the
regions are then fitted to the labelled chain positions as well as to the cooled walkers (a
mode holding 10% of 512 walkers has fewer walkers than parameters at 58):

| case (58 parameters) | exact | fixed barrier | mixing |
|---|---|---|---|
| 10%, barrier 6 | 10.0% | 6.3% | 9.6% |
| 10%, barrier 9 | 10.0% | 8.8% | 10.1% |
| 2%, barrier 6 | 2.0% | 2.1% | 2.0% |
| 2%, barrier 9 | 2.0% | 2.2% | 2.0% |
| 50%, barrier 6 | 50.0% | 49.2% | 50.0% |
| 50%, barrier 9 | 50.0% | 49.6% | 50.0% |

### Curved modes

| target | 16 parameters | 58 parameters |
|---|---|---|
| banana: one maximum in a curved valley | 1 → 1 | 1 → 1 |
| ring: a flat ridge on a circle | 1 → 1 | 1 → 1 |
| ring with three maxima on it, 2 log-units above the saddles between them | 3 → **1** | 3 → 3 |
| the same and a separate Gaussian holding 20% | 4 → **2** | 4 → 4 |

On the ring with three maxima the straight path from one maximum to the next crosses the hole
and dips 23 log-units, while the way round the ridge dips 2. The fixed rule reports three modes
at any number of parameters. With 16 parameters the chain goes round (memory 0.00 to 0.01) and
the ring is one mode; with the separate Gaussian added there are two, with weights 79.9% and
20.1% (exact 80% and 20%), where the memory between the Gaussian and the ring is 0.71 to 0.77.
With 58 parameters the chain does not get round the ring in 10 000 steps (memory 0.26 to
0.29), so the three maxima are three modes: that is the measured answer, and their weights
come out at 33.5%, 33.3% and 33.2% (fixed rule: 34.0%, 33.8%, 32.2%).

### Networks

`python client.py modes` on three saved networks. On the real SINU network it was run with
both rules, from the same tempered chain and seed; for the two analytic networks the fixed
rule's answer is the one of the ten-boost study they come from
([boost_background_region.md](boost_background_region.md)):

| network | fixed barrier | mixing | memory | straight path dips |
|---|---|---|---|---|
| 58-parameter Gaussian after ten boosts | 1 mode (in the ten-boost study) | 1 mode (one basin, no chain) | | |
| analytic twin of SINU after ten boosts (16 parameters) | 2 modes (in the ten-boost study) | 2 modes, 80.1% and 19.9% | 0.99 (7 of 4096 positions changed basin) | 60.6 |
| real SINU run, network 15 (16 parameters) | 2 modes, 86.6% and 13.4%; 4 walkers left out | 2 modes, 85.3% and 14.7%; none left out | 0.99 (9 of 4096) | 57.8 |
| the same, from a tempered chain of 20 000 steps | 2 modes, 86.1% and 13.9%; 6 left out | 2 modes, 86.3% and 13.7%; none left out | 0.89 (62 of 4096) | 56.2 |

The twin's exact weights inside its box are 80.9% and 19.1%. Its two modes are joined by a
curved valley that rises 6.5 log-units; the chain does not get through it, although a straight
barrier of 6 with 16 parameters is crossed freely (first table). What a chain crosses depends
on the width and the shape of the way as well as on its height, which is the reason for
measuring it.

On the real SINU network both rules find the two modes, in both runs.

- **Weights.** The strong mode gets 14.7% and 13.7% with `mixing`, 13.4% and 13.9% with the
  fixed rule; three earlier evaluations of this network with the fixed rule gave 13.2%, 13.5%
  and 13.9%. The largest difference, in the first pair, is 0.11 in the log evidence of the
  weak mode, where the spreads of the two evidences are 0.05 and 0.07. No difference between
  the rules shows beyond that scatter.
- **Walkers left out.** The fixed rule leaves four and six cooled walkers out (climbs that
  stalled, or groups under `min_share`). With `mixing` a stalled climb stays with the hill it
  is on, and a small group goes with the large basin it is nearest to (six walkers in the
  second run).
- **The shoulder on the barrier.** The strong mode has a shoulder at log10 G_eff between
  -2.8 and -2.0, presumably where these walkers sit (their positions were not looked at). With
  `mixing` it is sampled as part of the strong mode and holds 0.4% and 0.3% of the posterior in
  the two runs. With the fixed rule it holds under
  0.1% and 0.3%: there it depends on where the boundary between the two regions happens to
  fall. Whether the true likelihood has the shoulder is not known from this
  ([boost.md](boost.md) notes an island at the same place in network 16 that was gone in
  network 17). An unfiltered boost with `mixing` would put 2 to 3% of the strong mode's points
  on it, which is where true values would settle it.
- **Memory.** 0.99 in one run and 0.89 in the other, 9 and 62 positions of 4096 having
  changed basin. Probably these are walkers on the shoulder, which can go either way (not
  checked). Both values are far from the threshold.

## Reading the log

```
   mixing: basins 1 and 2 (386 and 126 walkers; the straight path between their maxima dips 57.8 log-units): memory 0.99 after 10000 steps (9 of 4096 walker positions half a chain apart had changed basin): separate modes
   cooled 512 walkers from T = 7 through 4.3, 2.65, 1.63, 1 and climbed: 2 modes of 2 candidate basins (measured mixing, memory at most 0.1)
   mode 1: weight 85.3% (log evidence -5340.66 +- 0.05, proposal efficiency 0.39); 386 walkers; highest value -5293.78
   mode 2: weight 14.7% (log evidence -5342.42 +- 0.01, proposal efficiency 0.43); 126 walkers; highest value -5294.54
```

- **Memory** near 0: the chain has forgotten where its walkers started, and the basins are
  sampled by one chain. Near 1, as here: hardly a walker changed basin. Values between say that
  walkers do cross but too rarely for the chain's split to be trusted.
- **Walker positions half a chain apart** are the pairs behind the estimate (at most eight
  times the number of walkers). A basin with few of them has a noisy memory; near the
  threshold the decision can then go either way. Neither answer is wrong there: a chain that
  half mixes gives a usable split, and so do two evidences.
- **The straight path between their maxima** is printed for comparison with the fixed rule. It
  can be far above what the chain has to cross (the ring: 23 against 2).
- A further line gives the time the test took.
- The report (`RUN/boost/boost_it_N.json` under `search.mixing`, or
  `RUN/modes/modes_it_N.json`) has the candidate basins, every pair's memory at half the chain
  and after it, the number of exchanges, the walkers of small groups that were joined to a
  basin and those left out, the seconds, and `on_no_known_hill`: positions of the free chain
  that climbed to a hill none of the known maxima is on (they are given the nearest basin).

## The `modes` command

`python client.py modes RUN_DIR` runs the tempered chain of the run's `sampling` block on a
saved network (the latest, or `-i N`), then exactly what a boost does first, and stops before
any true evaluation. It reads the run's `acquisition.boost` settings when there are any, and
uses the region that belongs to that network (the latest saved at or before its iteration) when
the run has one. It decides by mixing
(`--mixing`; by default the run's own `mixing`, or 0.1 when the run has none) unless
`--barrier-only` is given. It writes

- `RUN_DIR/modes/modes_it_N[_TAG].json`: for every mode the weight, the evidence and its
  spread, the peak, the mean and standard deviation of every parameter; the mixing test pair
  by pair; and `target`, the identity of what was searched (the hashes of the network and the
  metadata, the domain, the region with its hash and edge), as the chains and evidences
  record it;
- `RUN_DIR/modes/modes_it_N[_TAG].npz`: the T = 1 samples of every mode (`x0`, `f0`, `x1`, ...),
  `weights`, `names` and the same identity (`target_json`). Together they are the network's
  posterior: each mode's samples carry its weight.

The network, the metadata and the region are loaded from one verified private copy
(`sampling.targets.load_snapshot`): a network written again while the search runs, by a
training that ends, is not the one the result is recorded for.

This is also the way to get the weights of a finished run's modes, which the loop itself does
not report after its last iteration. It needs no MPI and changes no file of the run outside
`RUN_DIR/modes/`.

## Cost

Nothing when there is one candidate basin. Otherwise one more chain of `steps` and the climbs
of up to 16 times `walkers` positions: 8192 with the defaults.

| | fixed barrier | mixing | of which the test |
|---|---|---|---|
| two Gaussians, 16 parameters, two candidate basins | 5 s | 5 to 9 s | |
| two Gaussians, 58 parameters, two candidate basins | 34 to 46 s | 40 to 58 s | |
| real SINU network 15 (5 x 512, laptop CPU, tempered chain of 20 000 steps) | 259 s | 550 s | 284 s |

On a trained network most of it is probably the climbs (each of the 8192 positions climbs for
600 steps and is then compared with up to 48 maxima of every basin), judging from what the
other chains of that length cost; that part has not been tuned. The time is printed in the
log and stored in the report (`seconds`). It is spent on the master while the MPI workers
sleep, as the rest of the mode search.

## What is not established

- The candidate basins still come from straight paths between maxima, with a low threshold. A
  basin none of the cooled walkers climbed to is not a candidate; positions of the free chain
  found on such a hill are counted (`on_no_known_hill`) and not acted on.
- The memory is squared from half a chain to the whole on the assumption that it decays
  exponentially. For two basins with one crossing rate it does; for a chain that first crosses
  quickly from the barrier's edge and then slowly from the cores it underestimates the memory
  that is left.
- One threshold, 0.1, was used throughout. The decisions above change between memories of
  0.03 and 0.15, so anything from about 0.05 to 0.15 gives the same table.
- The sampler of the chain is the loop's ensemble sampler with the stretch move. A network
  sampled afterwards with another sampler mixes differently.
- More modes at more parameters means fewer points for each from a boost of `n`. With the
  three-maximum ring at 58 parameters each of three modes gets a third.
- On its own in the loop it has been run for two boosts only, on the analytic twin with short
  settings (the boost, its report, the region built from the modes, the next training and audit):
  a check that the pieces fit, not a study. It has not been studied on a real likelihood: the real
  branches of [boost.md](boost.md#two-recommended-settings) that use it changed it together with
  other settings.
- With `region` the surrogate is a wall outside the region. When the straight path between two
  maxima leaves the region, the dip printed is the wall's and says nothing about the chain: in
  the second boost of that short run two basins with 4161 log-units between them by the
  straight path had a memory of 0.03 and were merged.
- A boost with `mixing` can change the number of modes from one boost to the next (basins
  that begin to mix are merged). The region of `acquisition.boost.region` copes with that by
  construction (an earlier cloud joins the later boost's mode that it overlaps and whose
  reach its mean lies in, several clouds into one where a later boost finds one mode; see
  [boost_background_region.md](boost_background_region.md)), but on its own that combination has
  only been exercised in the short run above.

## Files

- `sampling/modes.py`: `find_basins`, `assign_basins`, `mixing_memory`, `modes_by_mixing`,
  `Regions(..., groups=)`, `posterior_modes(..., mixing=, candidate_barrier=)`.
- `sampling/detect.py`: the `modes` command. `config/config.py`: `BoostConfig.mixing`,
  `.candidate_barrier`. `dataset/boost.py` passes them on.
- `tests/test_modes_mixing.py`: basins, memory, grouped regions, the three decisions (far
  apart, a barrier the chain crosses, a barrier it does not), configuration, the command.