# Several trainings per network (`training.members`)

`training.members: K` trains every network of the boost phase K times on the same training set
with the same settings and keeps one. Rank 0 trains it as before (warm-started from the previous
network when `training.warm_start` is on, the default). K - 1 members are trained from random
weights, each with its own seed, on ranks that would otherwise wait for the next evaluation: during
a training only rank 0 works, and on a cluster the other ranks' cores are idle for the whole of it.
Members left over when the run has fewer ranks than members are trained on rank 0 after its own
network.

It is a feature of the boosts. The members act from the first training on boost points (the
network after the boosts' start, `acquisition.boost.start_iteration` or the hand-over of
`tempered_batch_scatter`) on; every network of the tempered phase, the hand-over's included, is
trained once, as without the key, and a run without `acquisition.boost` that sets it is refused.
The log says so at the start (`training.members: K trainings of every network trained on boost
points`).

```yaml
training:
  members: 4              # 1 (the default): one training per network, as before
  selection_share: 0.1    # of the points acquired since the previous training, held out to choose
  member_margin: 0.2      # the warm network is kept unless another is better by more than this
  member_threads: 8       # TensorFlow threads of a member (default TF_NUM_INTRAOP_THREADS, else 4)
```

## How one is chosen

- **The selection set.** A share of the points acquired since the previous training (its boost or
  batch and its injection; the previous network's audit points are excluded, since the warm start
  has seen them) is held out from every member's training this time. Every member, the warm one
  included, is judged on points it has not seen; a warm network's own validation loss is partly
  in-sample. The points are trained on from the next network on. The draw has its own generator,
  so the run's other draws are unchanged.
- **The choice.** The selection loss is the training loss (msre) on the selection set. The warm
  network is kept unless a member from random weights has a selection loss lower by more than
  `member_margin`; without a warm network (warm start off, or the first training after a background
  fit) the lowest is kept. A member that fails is never kept and is named in the log.
- **What is kept.** The chosen network becomes `trained_models/model_it_N.keras` and its history
  `training_history/history_it_N.csv`; the audit, the region, the boost and every later use see only
  it, and the next warm start starts from it. Every member is saved as
  `trained_models/members/model_it_N_memberK.keras` with `training_history/history_it_N_memberK.csv`
  (member 0 is rank 0's), and `trained_models/members/members_it_N.json` records the seeds, the
  starts, the epochs, the losses and which was kept and why. These are files of the training phase
  (`run.storage.phase_outputs`). The run's log has one line per training (`Members (...)`).

## The choice by true evaluations (`training.member_pilot`)

```yaml
training:
  members: 12
  member_pilot: 100         # points of each member's own posterior, evaluated with the true likelihood
  member_pilot_steps: 20000 # of its chain (T = 1, inside the network's region)
  member_pilot_walkers: 256 # at least 2 ndim (raised to that when lower)
  member_pilot_depth: null  # default: the msre scale 0.5 chi^2(sigma_level, ndim) below the best
```

With `member_pilot: n > 0` the members are chosen after the region is built, by true evaluations,
and nothing is held out:

- **The points.** Every member samples its own posterior at T = 1 inside the region of rank 0's
  network (`acquisition.boost.region`; without a region, from the best distinct training points),
  each on the rank that trained it, and n of its points are drawn. The pool (n times the members
  that could be sampled) is evaluated once with the true likelihood on all ranks; the calls are
  journalled under the stage `members/itN`.
- **The score.** A member's pilot error is the rms of its prediction minus the truth over the pool's
  points that the truth or that member puts within `member_pilot_depth` of the best value known
  (the training set and the pool). Points whose true value is not finite, or is the fixed value a
  launcher gives a point the likelihood cannot evaluate (`refused_value`), count for none. A member
  that puts a false plateau above the truth is judged where it puts its posterior; a member that
  misses a peak is judged where the truth is high. Points far below the best in both count for none.
- **The choice.** As by the selection loss: the warm network is kept unless another member's pilot
  error is lower by more than `member_margin`; without a warm network the lowest is kept.
- **The points afterwards.** Those with a finite true value join the training set before the
  audit, as the acquisition's do (they are true evaluations at points where some network puts its
  posterior). They are saved in `trained_models/members/pilot_it_N.csv` with the member that drew
  each, the truth, the derived parameters (`derived:NAME`) and every member's prediction; a run
  continued from the saved network N adds them again. `members_it_N.json` has `choice: pilot`,
  every member's `pilot_error` and its mean error at its own points (`pilot_own_bias`).

Why: the selection loss judges the members where the previous network put its posterior, and a
loss averaged over points does not see where a member will put its own. On network 8 of branches
with `members: 12` of the two axion monodromy runs below, the member kept by the selection loss was
35.6 (H0WC) and 3.2 (PDP+) too high on average at the points of its own pilot chain (a chain held on
a false plateau weighs it more than a boost does: at the two networks' boost points the means were
+1.4 and +0.2); the H0WC one then repeated the setback in its boost (scatter 4.0 at the boost points
against 2.1-2.2 for every member from random weights at the same points). The pilot chooses member 5
(error 1.8) and member 7 (1.5, against 4.3 for the kept one); on PDP+ the rank correlation between
the selection loss and the pilot error was 0.19.

Cost: n times K true evaluations per network (1200 for n = 100 and K = 12), on all ranks, and one
chain per member, run in parallel on the members' ranks.

## Measured on two axion monodromy runs

The two axion monodromy runs on which random weights were compared with warm starts
([boost.md](boost.md), "Warm start is the default; random weights are an option": H0WC with omega
in [5.5, 60], PDP+ with omega in [2.5, 60]) were continued after network 7 with `members: 12` and
`member_pilot: 100` on 96 ranks, their settings otherwise unchanged, and compared with three other
continuations of the same runs: one warm-started training per network (the default), one training
from random weights (`warm_start: false`), and `members: 12` chosen by the selection loss. The
table has the scatter of network minus truth at each network's boost points, which no network had
seen.

| network | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
|---|---|---|---|---|---|---|---|---|---|---|
| H0WC, member pilot | 1.26 | 1.26 | 1.27 | 1.37 | 0.67 | 0.45 | 0.27 | 0.19 | 0.17 | 0.14 |
| H0WC, warm start | 4.74 | 3.96 | 2.10 | 1.82 | 1.51 | 1.45 | 1.41 | 1.33 | 1.32 | 1.31 |
| H0WC, random weights | 1.36 | 1.14 | 1.22 | 1.11 | 2.99 | 1.52 | 0.41 | 0.27 | 0.21 | 0.15 |
| H0WC, selection loss | 3.97 | | | | | | | | | |
| PDP+, member pilot | 0.92 | 0.54 | 0.32 | 0.33 | 0.24 | 0.27 | 0.28* | 0.22 | | |
| PDP+, warm start | 1.11 | 0.50 | 0.31 | 0.33 | 0.29 | 0.25 | 0.20 | 0.22 | | |
| PDP+, random weights | 1.48 | 0.50 | 0.55 | 0.57 | 0.28 | 0.25 | 0.38 | 0.35 | | |
| PDP+, selection loss | 0.95 | 0.47 | | | | | | | | |

\* The main mode. The boost of that network found a second mode (9 walkers at omega near 26, weight
4e-4), and the density filter, which aims at the same training density in every mode, gave it 9261
of the 10 000 points (scatter 0.44 there, 0.43 over all of them). This is the boost's doing, not the
members'.

- **False maxima.** On H0WC the warm-started member had a false maximum at every network from 8 to
  12, at A_osc 0.02-0.04 and omega 25-45, where the true likelihood lies 30-70 below the best: it
  was 46-80 too high on average at the points of its own pilot chain, and its pilot error was 15-25
  against 1.2-2.3 for the member kept. Members from random weights had false maxima of their own at
  networks 12 and 13 (two each, 32-115 too high). At the other runs' boost points these networks are
  hardly worse than the rest (at network 10, 1.30-1.60 against 1.22-1.48): the boost points do not
  see the false maxima, the pilot does. Every choice from 8 to 12 rejected the warm network, and the
  run avoided the setbacks of the warm-started run (8 and 9) and of the run from random weights (12).
- **The fine structure.** At H0WC network 12 two members from random weights resolved the structure
  (validation loss 5-6e-4 against 0.8-1.8e-3 for the others) and the pilot kept one of them. From
  network 13 on the warm start carried the resolved state on: the warm network was kept at 13 and
  15-17, and at 14 member 7 replaced it (pilot error 0.23 against 0.38; 0.21-0.27 against 0.28-0.42
  at the boost points of every run). The run had at network 17 the error the run from random weights
  reached at 18 (0.14); the warm-started one stayed at 1.3.
- **Ranking among sound members.** Where no member has a false structure, the pilot ranks the members
  roughly as the boost points do (rank correlation 0.75-0.83 at H0WC 14, 0.55-0.73 at PDP+ 15).
  Where one has a false maximum, its pilot points can decide the ranking of the others: at H0WC
  network 11 three members put the rejected warm member's points 13-23 too high, inside the depth,
  which alone gave them pilot errors of 1.8-2.0 (1.08-1.12 without those points, two of them the
  lowest). The member kept had a scatter of 1.37 at its boost points, against 1.19 for the best of
  the three. The score prefers a network without a raised region about 20 below the best, which a
  profile would show, to one slightly better at the posterior's points. Scoring the others without
  the points of a member already rejected has not been tried.
- **The margin.** On PDP+ the warm network was kept at 10-14 and 16 and replaced at 8 (pilot error
  3.55 against 0.90), 9 (0.60 against 0.46) and 15 (0.24 against 0.19). At 9 the warm member was
  slightly better at the boost points of every run, at 15 the two were equal: for networks this
  close a margin of 0.2 is inside the pilot's noise. A margin above 0.39 would have kept the warm
  network at H0WC 14, which was worse at the boost points of every run. The run ended level with
  the warm-started one (0.22 at network 15) and below the one from random weights (0.35).

## Why several trainings

A training ends in one of two states: it resolves the fine structure of the likelihood or it stays
on a plateau where that structure is smoothed out, and the random start decides which. Retrained on
the saved training sets of the H0WC run above, every training sat on the same plateau at epoch 100;
on the training set of network 13 one random start in three left it, on that of network 14 two in
three, with an error at posterior points none had seen of 0.6-1.0 against 3.9-4.0 for the others. A
longer or more patient learning-rate schedule did not help: it memorised the training set. The
validation loss separated the two states (2-3e-4 against 1.2-1.3e-3). A warm start keeps the state
it inherits, so a network that smoothed the structure passes it on; warm starts are nevertheless
better on almost every likelihood tested, which is why rank 0 keeps its warm start and the extra
members start from random weights. With a chance p of resolving per start, K starts miss with
probability (1 - p)^K: 0.20 for p = 1/3 and K = 4.

When the training set has too few points where the true posterior lies, no start resolves the
structure (none of 8 on that run's network 11); selection cannot help there, and more data can (with
10 000 points from the final posterior added, both starts did).

## Cost

The members train while rank 0 trains its own, so an iteration's training takes as long as the
slowest of them (the schedules end at different epochs). Each member uses `member_threads` cores
and its own memory (the training set, about 10 MB for 100 000 points of 23 parameters, and its
network). On the two axion runs above (96 ranks, 12 members at 8 threads) an epoch took as long per
training point as in the continuation with one training from random weights (H0WC) or 5% longer
(PDP+): the members hardly slowed each other down. The training took 26% (H0WC, networks 9-17) and
35% (PDP+, 9-15) longer than there, and the iteration (training, pilot, audit and a boost of
10 000) 15% and 20% longer. The training sets were 2-8% larger (the pilot's points), and the slowest
member trained on average 14% (H0WC) and 15% (PDP+) more epochs than the median member of the same
training, 57% more at H0WC network 12, where a member that resolved the structure trained longest.
The pilot's 1200 true evaluations took 250 s (H0WC) and 100 s (PDP+), against 1950 s and 790 s for a
boost of 10 000. The warm-started continuations are no reference for the time: their steps were
faster per point than in either of the other two.

This holds while every member has cores of its own: at most 1 + (cores - rank 0's threads) /
`member_threads` members, 12 on two 48-core nodes at 8 threads and 6 on one. With more, the
members share cores, or the ones left over train on rank 0 one after another, and the training
takes several times as long. The log then has a warning after the members' line
(`members.oversubscription`), which takes the run's cores as its ranks (one core per rank, as a
cluster job usually starts them) or the cores of rank 0's node if those are more. Replayed on the
first five choices of the two runs, 6 members (the warm one and every set of 5 of the 11 others)
made the same decisions as 12, and the warm network with a false peak was rejected with 2: more
members help only where few trainings from random weights resolve the structure. With 6 the
slowest member would have trained 10-11% more epochs than the median on average and with 4 8%,
against 14-15% with 12: fewer members save little time while each has cores of its own.

With `members: 1` nothing is held out, no rank is asked and the run is as before, bit for bit
(tests/mpi_members_smoke.py).

## On GPUs

```yaml
training:
  member_device: auto     # a GPU when the member's process sees one, else the CPU; cpu; gpu (stop without one)
```

Only rank 0 and the ranks that train members run TensorFlow (`utils/devices.py`). Each of them takes
one GPU of its node, the processes of a node in turn over its GPUs, with memory growth on. Without
it, the first TensorFlow process on a GPU reserves nearly all of its memory, and the members started
after it there fail. Every other rank hides the GPUs from itself before the likelihood is built
(`CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` empty), so that a likelihood code importing
TensorFlow or JAX takes none; the launcher (`hpc/submit_client.sh`) no longer does it, and a run
started with a plain `mpirun` behaves the same.

`training.member_device` decides for the members. With `auto`, the default, a member trains on a GPU
when its process sees one and on the CPU otherwise; `cpu` keeps the members on the CPU; `gpu` stops
a member that sees no GPU (it fails and is never kept). Rank 0 uses a GPU whenever it sees one.
On a GPU, `member_threads` are the CPU threads of what TensorFlow keeps on the CPU, and the
oversubscription warning counts only the members left over for rank 0.

The log says where rank 0 trains (`TensorFlow on rank 0: GPU 0 (...)` or `CPU`) and, after each
training, where every member did (`members trained on: 0-11 GPU 0 (...)`); `members_it_N.json`
records each member's `device`.

`tests/gpu_members_smoke.py` checks on a GPU node that rank 0 and the members train on the GPUs (both
GPUs of a node with two), that none fails, and that `member_device: cpu` keeps the members on the
CPU. `benchmarking/gpu_training.py` trains one network of a run again on a GPU with TF32
(TensorFlow's default on Ampere and newer GPUs, for every training there, rank 0's included), on a
GPU in full float32 and on the CPU, from the same random weights, and compares the times and the
errors at that network's boost points.

Measured on one NVIDIA L40S (TensorFlow 2.19 with its CUDA wheels):

- **The members on one GPU.** Rank 0 and three members trained on the same GPU at once, and none
  failed; the pilot's chains ran on it too, and `member_device: cpu` kept the members on the CPU.
- **A network of the PDP+ axion run** (network 10, 61 610 training points, the run's full schedule,
  one seed). On the GPU in full float32 the training followed the CPU's exactly: the same 506 epochs,
  the same validation loss and the same errors at the 10 000 boost points (scatter 0.372), in 259 s
  against 1284 s on 8 CPU threads. With TF32 it took 242 s and ended elsewhere, as another seed would
  (scatter 0.345, inside the 0.29 to 0.59 of that network's twelve members). TF32 therefore stays as
  TensorFlow has it.

Open: a node with two GPUs, and how many members share one GPU before they slow each other down
(three shared one here, on a small network).

## Tests

`tests/test_members.py` (the ranks, the selection set, the choice, the settings, the phase's files,
one member trained; the pilot's score, choice, depth, starts, settings and saved points),
`tests/test_devices.py` (which ranks keep the GPUs, which GPU a process takes, TensorFlow's set-up
without a GPU) and `tests/mpi_members_smoke.py` (the loop on three ranks and on one, `members: 1`
against no key; the pilot on three ranks and on one, and a pilot run stopped after training and
continued). On a GPU node: `tests/gpu_members_smoke.py`.
