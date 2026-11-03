# Contributor guide

Why the code is as it is, module by module, for anyone changing it, human or agent. Each section lists what must stay
true and the measurement behind it. New behaviour goes behind a config key unless there is evidence to make it a default.

## Repository

* Keep saved objectives, physical bounds and historical outputs intact. Never edit a source checkout or data directory merely
  referenced by a run manifest.
* Report objective call counts separately from wall time.
* Push or publish only when the user has authorised it. Verify the remote and target branch first, and do not force-push over
  others' work.
* What is specific to the Grendel cluster (install, sync, Slurm headers, run definitions) is in the private repository
  `AarhusCosmology/ClientCluster`. `hpc/` here holds the generic tools.

## Layout

| path | role |
|---|---|
| `client.py` | the loop and the commands |
| `config/` | the run YAML's schema, with the evidence for each default in its comments |
| `dataset/` | designs, injection, density selection, boosts, augmentation |
| `training/`, `model/` | loss, training, the audit, the training domain; network, background, walls |
| `sampling/` | AIES, modes, the region, chains over several cells, provenance of a sampled target |
| `convergence/` | drift and the tempered verdict |
| `run/` | phase records, objective identity, call journal, combine, export |
| `evidence/`, `derived/` | BEST on a network; derived parameters and their views |
| `profiles/hybrid.py` | batched surrogate evaluator, projected BFGS `ascend`, `certify`, basin-hopping `hop`, `scan_hybrid`, the ledger views the true stage uses |
| `profiles/residual_newton.py` | the corrected true stage: `NewtonConfig`, `NewtonRefiner`, `refine_profiles_newton` |
| `profiles/config.py` | the profile YAML's schema |
| `profiles/workflow.py`, `surface_workflow.py` | the study: scan (`scan_checked`), true stage, resume; a manifest that lacks a current key is refused |
| `profiles/region.py` | the setting `region`: which region belongs to a network, nodes outside it |
| `profiles/surrogate.py`, `surface.py` | L-BFGS-B at one node, used by augmentation designs and display grids, and the scans of `surrogate_optimizer: legacy`. Real 2D studies use that scan on rugged networks: keep it unchanged in behaviour |
| `profiles/evaluation.py`, `metric.py`, `training.py` | the ledger and its budget; the profile metric; the retraining preset |
| `hpc/` | real likelihoods on a cluster: launcher, refused points, true check, corrected evidence, branching |
| `benchmarking/` | posterior comparison against reference chains; the 31D exact-reference benchmark; `schedule_newton.py`, a load-independent wall-time model from a call journal |

## Before changing `dataset/augment.py`

`client.py augment` adds a profile study's true evaluations to a run's training set through
the acquisition's density filter; read [docs/augment_training.md](docs/augment_training.md).
Keep these:

* Both densities in the filter come from the same k-NN estimator and whitening (the
  training set's density at the candidate, and the law read off the training set at the
  candidate's true log L). An analytic target density on one side reintroduces the
  estimator's high-dimensional bias.
* One pass with retention [1 − ρ/ρ(L)]₊. Drawing until no candidate is short admits nearly
  every candidate that k-NN noise makes look short: when that was tried it took 44% of the
  candidates of a training set already at its target. `tests/test_augment.py` guards this.
* It writes `RUN/augmentation/` and changes no existing file. Coordinates are compared at
  single precision, the training set's.
* The first training with the points uses `training.augment_schedule_factor`; the marker
  `augment_it_N.trained.json` is what makes it happen once, also across restarts.
* A lower msre `c` and a cold start were both measured as alternatives to the longer
  schedule and rejected (table in the doc). Do not swap them in without that measurement.

## Before changing `dataset/peak_samples.py`

The injected points (`prior.n_inject`, `acquisition.n_inject`) and their window
(`INJECT_WINDOW`, 16); read [docs/injection_window.md](docs/injection_window.md). Keep these:

* `gaussian_peak_samples(..., window=None)` fits every point; the loop always passes the window.
* `_fit_quadratic` is shared with the profile code (`profiles/`,
  `dataset/profile_augmentation.py`). The window selects points before the fit
  and caps widths after it; the fit itself is not touched.
* On a real likelihood in a wide box the fit to every point is not a poor approximation, it
  is meaningless (residual of tens of thousands of log-units). Read the log line: a "vertex
  value" far above the best training value, or a "fit rms" of thousands, says the injected
  points are not near the peak.
* The window is in units of T n / 2 below the best, with a floor of three points per
  coefficient. Narrower windows and small fixed counts were erratic on the non-Gaussian
  likelihood; a hold-out choice of the window and drawing in rounds were tried and not adopted.
* The document has two kinds of depths: true values (a pair of cluster runs, and the runs'
  own old injections) and the runs' last networks standing in for the truth (everything
  else). Do not quote the second kind as true values.

## Before changing `dataset/boost.py` or `sampling/modes.py`

`acquisition.boost` replaces an iteration's tempered batch by posterior samples of the
network, the same number from every mode; read [docs/boost.md](docs/boost.md). Keep these:

* Opt-in. Without the key the loop is exactly as it was, under MPI too: the one extra
  `bcast_idle` (the workers sleep while the master samples, and learn whether it found a
  mode to boost) is sent only in a boost iteration.
* A boost iteration's true calls are the `n` asked for: no tempered batch, no injection.
  The one exception is a network without a mode (`sampling.modes.NoModeFound`): that
  iteration is an ordinary one on every rank (`client.draw_boost_points`, which also tells
  the workers; `tests/mpi_boost_fallback_smoke.py`), with its tempered batch and injection
  and a record in `RUN/boost/fallback_it_N.json`. Catch that error only: another failure
  of a boost must still stop the run.
  By default points bypass the density filter. Opt-in `density_filter: true` uses a
  balanced T=1 density target and may underfill; see [docs/posterior_density_boost.md](docs/posterior_density_boost.md).
  Filtered rows must never receive the unweighted evidence correction.
* For the default unfiltered boost, the same number from every mode, whatever the weights. Each mode's mean of exp(true minus
  network) must be known equally well, and the mode the network under-rates is the one a
  proportional split would starve.
* Modes are groups of *maxima*, not of T = 1 walkers. A walker sits ndim / 2 log-units under
  its peak, which hides a barrier of that size, and a walker stranded on the barrier joins the
  two sides (the first version did both on SINU). Climbs that stall more than `barrier` under
  a maximum they are joined to are left out for the same reason.
* Mode weights come from the per-mode evidences (bridge sampling), never from the tempered
  chain's split between modes: that was off by 0.2 in log-odds per network, measured. An evidence
  that is not finite stops the boost before any true call (`mode_log_evidence`): the weights are
  fractions of one sum, so it would leave none of them known.
* The corrected weights are only as good as the effective number of points behind them. Keep
  that number in the log and the report. A point the likelihood refused (a -inf return, or
  the run's refused value) has weight zero and stays in the denominator of its mode's mean,
  as in the true check (`hpc/analysis/true_check.py`), and is counted
  (`zero_density`, `refused`); the two programs must keep the same rule. A point without any value (NaN, +inf) still withholds
  the corrected weights of every mode (`boost_report`, `correction_withheld_reason`,
  `failed`): do not give it weight zero, its mode's mean is not known. Never drop refused
  points from the denominator: that over-rates the mode they were drawn in.
* `mode_depth` (opt-in) leaves out a mode whose highest maximum lies more
  than that below the highest mode's, before its chain is run; the highest mode always stays.
  It is decided by the modes' maxima, never by their weights (the same-number rule above is
  about weights the network may get wrong; a mode hundreds of log-units down is not one of
  those). Without the key the modes and the search's info are as before. `auto` (`posterior_modes`
  resolves it) is `temperature * ndim / 2` with the tempered chain's temperature; the info's
  `mode_depth` holds the number.
* `balance_floor` (opt-in) gives a mode lighter than the floor its equal share
  times its weight over the floor (`dataset/boost.py` `mode_shares`; the filtered path cuts the
  clouds in `select_posterior_boost`). Without the key `draw_boost` and `select_posterior_boost`
  are the equal shares (`tests/test_boost.py`, `tests/test_posterior_boost.py`): keep that.
  It is for small groups of walkers that `min_share` 0.005 lets through, which the posterior
  pilot cannot check; a mode at or above the floor keeps its equal share.
* Two recommended boost settings (docs/boost.md, "Two recommended settings"):
  near-Gaussian (the reference runs' block) and known non-Gaussian (the axion pilot branch's).
  Only documentation: no key selects either, and no default changed. The measured split is the
  training domain: it cost the 49-parameter and HiLLiK branches accuracy (tails too high at the
  first domain network; 2.5x and 6.5x the reference's error at the last boosts) and is what the
  axion run needed. Do not make either set a default without the confirmation runs' full results.
  What decides it is mostly the output scale the far points set ("What the training domain
  changes": retrainings of five runs' first boost networks); a choice of the
  three candidates by their held-out error near the best picked the better one in all five, but
  it is not built (one seed each).
* A true value that is not finite is not trained on (the loop's own filter) and is counted
  in the report.
* `RUN/boost/boost_it_N.csv` is a record; the points themselves are in `data_it_<N+1>.csv`.
* `finish_boost` matches each evaluated point to a drawn one at single precision. The launcher rounds a batch to single precision and keeps it inside the box as single
  precision holds it (`hpc/launch.py`, `evaluate`), so a point drawn in the last
  digit before an edge comes back one step inside. Such a point is the nearest unmatched
  drawn one within `MOVED_BY_A_LAUNCHER` units in the last place and is recorded where it was
  evaluated; anything farther still stops the run. Matched bit for bit it stopped every run under the launcher whose boost drew one: rare on a good network,
  certain on one pressed against its box. The unit suite and the MPI smokes run `client.py`
  directly and did not see it. After a change to this match, to the launcher's rounding, or
  to anything else that compares evaluated coordinates with chosen ones, run a small
  analytic case through `hpc/launch.py` to its boosts.
* The pooled error of a boost (`error` in its report, the two log lines after the modes) is a
  measurement of the network at points it had not seen, printed for every boost, filtered or
  not. It decides nothing: the number of boosts is the user's, and the loop's convergence
  test must not be applied to boost iterations. It is not the evidence correction, which
  stays withheld for filtered rows.

## Before changing `run/combine.py` (domain splitting)

`python client.py combine` puts runs of one objective on adjacent boxes in one parameter into one
emulator; read [docs/combine.md](docs/combine.md). Keep these:

* The combined network is one flat Keras graph: every part's layers copied under `part<k>_` with
  the same weights (`rebuild`). The profile code copies a model with `clone_model` layer by layer
  at double precision (`copy_model_dtype`); a nested sub-model is not what it was tested on.
* The blend is (1 - s) A + s B with two `SmoothStep` weights (`model/network.py`, registered so
  that `load_model` finds them), each exactly 0 or 1 outside the band: there the value is exactly
  one network's. A + s (B - A) rounded it by up to 0.12 in float32 next to a wall. The
  smootherstep's first and second derivatives vanish at the band's ends (the profiles' curvature).
* Each part's network is used inside its own region only (`RegionWall` in the graph) unless
  `--no-walls`: a network has maxima beyond its region, and another part's region can reach into
  its range.
* The parts must record the same objective: `run.identity.differences` with only the split
  parameter's bounds removed. Do not widen that: a narrower Cobaya prior on the split parameter
  changes the objective by a constant, and the networks would no longer share one scale.
* The combined region is every part's cells re-expressed in the combined box's units (`rebox`:
  distances unchanged, tested to 1e-12). A combined emulator is sampled cell by cell
  (`sampling/mixture.py`), each cell weighted by its own evidence.
  Identical normalized ellipsoids are merged only in this outer union, with source-cell
  provenance; individual network walls stay intact. Mixed region/no-region inputs are
  rejected. Consume verified immutable copies of all inputs, record their hashes, and
  publish the complete output by directory rename; never expose a partial run without
  its `combine.json` marker.
* The training points are the parts' points with a value; refused ones are left out, since the
  parts' refused values differ.
* The loop refuses a directory with `combine.json` (`client.py main`): it is not a run to continue.
* `tests/test_combine.py` covers the values in and around the bands, the walls, the files, the
  consumers (`load_snapshot`, `restrict_model`), the refusals and the loop's refusal.

## Before changing `run/export.py`

`python client.py export` writes a network as a TensorFlow SavedModel that loads without CLiENT;
read [docs/export.md](docs/export.md). Keep these:

* The default export is the network inside the region of its boosts (`restrict_model`), as
  CLiENT's own consumers use it; `--no-region` is the bare network. Do not make the bare network
  the default: a boost network has false maxima outside its region.
* No export is accepted until it has been reloaded in a process that cannot import CLiENT
  (`python -I`, another working directory) and compared with the source. The tolerance is
  absolute plus relative to the value: traced and eager TensorFlow order operations differently,
  and at walls of 1e5 and more they differ by a few float32 steps.
* The network comes from `load_snapshot`, so that the hashes in `emulator.json` are of the bytes exported, and OUTPUT is built
  beside itself and renamed last, so that a refused or failed export leaves nothing.
* A relative OUTPUT must work: the reload runs in the staging directory, so paths are resolved first (`client.py export RUN exported` failed
  while every test used absolute ones).
* A loaded SavedModel must be kept in a variable while it is called; a one-expression load-and-call
  frees its variables (the test's first version failed that way).

## Before changing the convergence of the ordinary loop

`convergence.metric: tempered_batch_scatter` (`convergence/convergence.py::tempered_verdict`,
`client.py::tempered_convergence`); read [docs/tempered_convergence.md](docs/tempered_convergence.md).
Keep these:

* Opt-in. With `gaussian_posterior_drift` the loop, its broadcasts and its files are as they
  were. With the new metric the hand-over travels in the broadcast that already carried the
  stopping decision, and one more broadcast at the start tells the workers a recorded start.
* The threshold is in units of the sampling temperature, on the scatter (not the rms), for two
  networks running. 1.2 is measured (an analytic suite at four
  temperatures, and a real 49-parameter run branched at the network where it is met). Do not
  change the default, the measure or the unit without repeating that.
* It belongs to the ordinary loop alone. Once the boosts have begun it is not evaluated: a
  boost reports its error at posterior points against true values and the user decides when
  to stop (`-i`, `max_iterations`, `acquisition.boost.boosts`). Do not add an automatic stop
  to the boost phase.
* With `-i` nothing follows from the verdict: it is printed and recorded only.
* A hand-over is recorded in `RUN/boost/start.json`, and every later use of the boosts' start
  (the background, the region, the acquisition, a continued run, the report in `hpc/`)
  reads the earlier of that and `acquisition.boost.start_iteration`. Never rewrite the run's
  copy of its settings.

## Before changing the mixing test of the modes

`acquisition.boost.mixing` and `python client.py modes` (`sampling/modes.py`,
`sampling/detect.py`); read [docs/boost_modes_mixing.md](docs/boost_modes_mixing.md). Keep these:

* Opt-in. With `mixing` absent `posterior_modes` takes the fixed-barrier path unchanged
  (`find_modes`), and its info has no `mixing` entry; `tests/test_modes_mixing.py` guards it.
* The test chain is the chain the modes are then sampled with: same sampler, `walkers` and
  `steps`. The question is whether *that* chain mixes; do not shorten it to save time.
* A walker's basin is the one it climbs to (`assign_basins`), not the nearest fitted Gaussian.
  A Gaussian fitted to the few walkers of a small basin in 58 dimensions gave the basin away
  (memory 0.00 on basins that exchanged almost nothing, measured).
* Memory is P(A later | A before) - P(A later | B before) among walkers in A or B later,
  half a chain apart, squared. Counting crossings or the share of walkers that moved depends
  on the basins' weights; the memory does not (2% to 50% tested).
* `find_basins` drops nothing. A group under `min_share` then goes with the large basin it
  has the smallest dip to when that dip is under `barrier`, and is otherwise left out and
  counted; it is never tested as a basin of its own (too few walkers for a memory).
* Crossings where the straight path dips 10 to 20 log-units are real with the stretch move at
  moderate dimension (it throws walkers from the inner tail). Do not treat them as a fault of
  the labelling.
* `client.py modes` evaluates the network only and writes only `RUN/modes/`.

## Before changing the boost's background or region

`acquisition.boost.background` and `.region` (`model/network.py`, `sampling/region.py`); read
[docs/boost_background_region.md](docs/boost_background_region.md). Keep these:

* Both opt-in. Without them nothing changes, under MPI too: everything they add runs on the
  master inside steps that exist already.
* The quadratic is fitted once, at the first training that holds boost points (the first after
  the boosts' start; also when that first boost fell back to an ordinary batch), and that
  training starts from random weights. Afterwards it is read from the previous model
  (`model_background`) and never refitted: a network cannot absorb a change of it, so a refit
  is a cold start.
* The background alone is not safe on a likelihood that is not Gaussian: where the truth is
  flatter than the quadratic the network must add a rising correction, and it carries it into
  the empty corners (the twin: maxima 2000 above the best). Do not recommend it without the
  region.
* The region is built from boost rows only, and only from those whose true value confirms
  them. No network enters it. Outside it the surrogate does not evaluate the network.
* The audit's searches are projected onto the region (`Region.project_unit`), not walled: a
  wall throws a 58-parameter search about (14 792 end points for 14 976 starts, against one).
* Current boost groups never share historical rows. Match clouds by verified-row overlap,
  not the mode number (its weight rank); multiple older clouds can join one current group.
  An overlap is not enough to join two sets: the mean of the one with fewer rows must lie
  inside the reach of the other (`_one_mode`). On any overlap, one stray row merged two
  modes and tripled a cloud's reach.
* A sparse cloud (at most `SPARSE` x ndim distinct rows) takes the shape of the well-sampled
  cloud with the most rows (`_donor`), its own centre, and its own reach capped at the
  donor's. Never give it a covariance shrunk towards a sphere in box units, or any shape
  estimated from its own handful of rows: a real cloud is hundreds of times narrower in some
  directions, and the shrunk ellipsoid admitted the network at false maxima of the twin
  (`docs/boost_background_region.md` has the table). A cloud without a shape (one distinct
  row; sparse while nothing is well sampled) is left out by `boost_region`, counted
  (`left_out`) and logged. It must not raise in the loop: the error would come from the
  saved boost files, at every restart. Well-sampled clouds stay bit-identical to the saved
  regions of earlier runs: after a change, build the regions of a long run's boost files
  with the old and the new code and compare them.
* `Region.start(component=k)` must find points in any cell, also one that is a small part
  of its ellipsoid: it closes in on the ellipsoid's centre. Without a component the draw is
  the one it always was, number for number, whenever it finds its points in 200 draws (every
  saved region checked, 224); after that it closes in on the
  centres too. Until then it raised, and a region reaching far beyond the box (the axion
  run's network 18, 0.46% of the draws inside) stopped the loop's last chain for good.
* `region_quantile` and `train_domain` (17 October 2026; "The training domain and the
  quantile reach" in the doc) are opt-in and need `region`. Without them every region is
  bit-identical to the saved ones (218 regions of 41 runs rebuilt with the code before and
  after) and the loop trains on every point; keep both true. The quantile changes the
  region's reach only, never which rows are one mode (`_matching`, `_one_mode`, `shaped`
  keep the farthest).
* The training domain is built once, at the first training after the boosts' start, from the
  rows `fit_background` uses (within T * ndim of the best), and is never rebuilt: a domain
  that grew with the boosts would bring back the far points and the jumps of the
  farthest-point reach. Its file is named after the network that built it, so that a
  branch at the start and a rewind before it leave it behind, and it is a training phase's
  output (`run/storage.py phase_outputs`, the loop's artifact list). The cut is applied to the
  training arrays only: the dataset, its files, the acquisition's density and the derived
  emulator keep every point. Keep it opt-in; see the measurements and limitations in the doc.
  Its coverage diagnostic estimates box-clipped union volume with volume/overlap weights
  and reports effective sample counts. Never use `Region.start` as uniform volume draws:
  its equal-cell selection and centre-shrinking fallback serve initialization instead.
* The region keeps the chains to the known modes. It is for the boost phase and for consumers
  of a finished network (the evidence and profiles, both by default), never for the
  tempered acquisition of the ordinary loop.

## Before changing the region in the profile code

The profile setting `region` (`profiles/region.py`, `model/network.py::restrict_model` and
`RegionWall`, `profiles/workflow.py`); read "Networks trained on boosts" in
[docs/create_profiles.md](docs/create_profiles.md). Keep these:

* The default is `run`. A network trained on boosts searched over the whole box gave false
  maxima 40 posterior sigma from the data on a real 49-parameter run and no node converged;
  do not make the bare network the default again. A network without a region file is
  searched over the whole box whatever the setting, and that path is bit for bit the one
  before the setting.
* What the searches get is a real Keras model at double precision, so every optimiser takes
  it (the node optimiser copies it with `copy_model_dtype`). Profile
  training gets the network itself, never the confined one.
* The manifest holds the resolved region (file, hash, edge; None for the bare network) and a
  resume compares that, not `configuration.region`: a study made before the setting resumes
  with the default exactly when its network has no region.
* A derived-coordinate view (`derived/coordinates.py`) copies the region of its network into
  `VIEW/region/` and records it in `view.json` (`region`); never into `VIEW/boost/`, where the
  run's chains, evidence and `modes` would read it in the view's coordinates. The region stays
  in the run's coordinates: `ViewRegion` measures a point z of the view at x(z), the pivot
  solve's output, and `restrict_view` puts the wall between the network and the feasibility
  penalty. The manifest's entry carries the pivot map (`coordinates`, file and hash). A view
  made before 14 October 2026 from a network with a region holds the bare network: the
  default refuses it (remake the view), `region: null` uses it as before.
* A node whose optimum is not inside the region is cut from the end of its range and the
  scan repeated, before any true call of a node; the cut is recorded in the scan's file
  (`region_cuts`) and in `ranges.json`. Do not let such a node through to the true stage:
  its start is on the wall or pressed against it, and the residual Newton stage reads the
  surrogate's curvature there.
* `INSIDE` (0.98 of the edge) is where a search pressed against the edge ends (rho = 1.0000
  in the tests). It is not tuned on anything else.
* 2D surfaces are searched inside the region, and nodes outside are not cut there: the study
  stops after the scan with the list (`profiles/surface_workflow.py`). Do not let them
  through either; the scan certifies a point of the wall.
* Measured: one real 49-parameter study with the edge at 1 (735 nodes, 677 converged), and
  the two-mode twin with the edge at 2 and at 1. The cut has only been exercised on analytic
  networks. `tests/test_profile_region.py` covers the wall, the setting, the cut, resumes
  and both dimensions.

## Before changing `run/storage.py` or `run/identity.py`

The run-state layer: phase commits, the run lock, the objective identity;
read [docs/run_recovery.md](docs/run_recovery.md). Keep these:

* A phase without a commit is never taken as done, and a continued run can always get past
  it: `PhaseStore.recover` moves the phase's files to `RUN/interrupted/` and the phase is done
  again. Do not turn that into a refusal: a job at its time limit lands inside a training or
  an acquisition nearly every time, and a refusal there ends the run. Do not replay journal
  values into the training set either.
* `phase_outputs` must list every file a phase writes before its commit. A file missing from
  it survives an interruption and is read by the redone phase or by the start-iteration
  detection (`run/run.py`), which takes the latest network and training set on disk.
* Nothing in a run directory is deleted by this layer.
* A committed file is never rewritten outside the phase that commits it. The region of a
  loaded network is read from its file for that reason (`client.py`, "Region of the boosts");
  a rebuilt one changed the file's hash and the next start was refused. The derived emulator's
  files are phase outputs that are not committed for the same reason: `client.py derived --retrain`
  replaces them on a finished network, as documented, and a hash would refuse the next start.
* The identity records what decides the value at a point, not the text of CLiENT's wrapper
  files or unrelated launcher helpers. Dependencies of the objective must be recorded even
  beside the launcher; schema migration must not silently trust added or renamed entries.
  A wrapper change that alters returned values is
  declared by raising `BaseLikelihood.objective_convention` in that wrapper. Do not go back to
  hashing wrapper sources: a comment edit then refuses every run.
* `--accept-identity-change` keeps the earlier record and never waives a domain change
  against `metadata.json`.
* `hpc/analysis/branch_run.py` copies phase records only when every file they commit
  is copied, and rejects incomplete/damaged retained checkpoints before creating a branch;
  `tests/test_hpc_branch_run.py` checks a branch with the real `PhaseStore`. It reads a
  file's iteration as `_it_N` not followed by a digit (with a dot required, the evidences
  and the phase history had none and went into every branch), leaves `evidence/`, `modes/`,
  `analysis/`, `interrupted/`, `phases/history/` and `boost/start.json` behind, and takes
  the source's boost start as the earlier of its settings and `boost/start.json`.
  `--train-from N` is the other kind of branch, inside the boosts: it keeps the source's start and
  `boost/start.json`, takes networks 0 to N - 1 with their committed training and acquisition and the
  training set of N, and not N's audit (`imports.json` says `trained_again_from`); network N is trained
  again by the branch.
* Branches keep complete parent call journals under `branch_provenance/source/`, outside
  active `objective_calls/rank*.jsonl`. `imports.json` counts inherited finite native
  coordinate/value pairs from the checkpoint, audit and optional augmentation separately from new objective calls.
  Never count earlier cumulative training sets again or describe these imports as cache hits.
  `run.objective_calls.summarize` runs after every acquisition: a record it cannot read, or a
  branch from before `293887a` (no `branch_provenance/`), is named in `branch_note`, never raised.
* The branch is filled under `<name>.partial` and renamed last, so a directory that reads as
  a run is a whole branch (the threshold suite takes the one directory it finds). A file's
  iteration is its name's, except `augmentation/augment_it_K.trained.json`: that one belongs
  to the network its content names (K + 1 or later) and stays behind when the branch does not
  hold it, or the branch's first training on those points runs the ordinary schedule.
* Records read at the next start are written whole or not at all (`run.storage.atomic_path`):
  `boost/start.json`, the boost and fallback records, the chain summaries, the derived
  metadata, the evidence's output.
* `-s N` in a run that has gone further than network N is refused (`client.rewind_to_start`):
  done in place it left every phase record complete and the later files where the next
  start without `-s` took them, the earlier line's, without the redone iteration's points.
  `--rewind` first moves what belongs to the later iterations to
  `RUN/interrupted/rewind_to_it_N-<time>/` (`run.storage.later_files`, `set_later_aside`):
  what stays is what `branch_run.py --at N` would copy, and the two must agree on what
  "later" is (files named after an iteration above N anywhere in the run, the acquisition
  of N, an augmentation marker by the network it records, a hand-over after N), plus the
  summaries in `analysis/`. Nothing is deleted, the call journal is not touched, and the
  metrics are copied, since the loop reads the rows before N from them. Do not make `-s N`
  silent again, and do not let the rewind delete.
  The complete move plan and hashes are committed before the first move. A pending
  `REWIND.json` is finished under the run lock before phase recovery and automatic start
  selection; otherwise a killed rewind can leave later models with their phase records
  already moved away. Test interruption at every move and a killed CLI process.
  Hashes of regenerated analysis files and their notes must be recorded before the move,
  not only on completion: test a second interruption before/after that move and after a
  copy without removal. Reject later checkpoints omitted from a manually edited plan;
  quarantine their files under `interrupted/` before omitting them, never leave them active.
  What a command without the run lock wrote since the plan joins it and is set aside with
  the rest (`_written_without_the_lock`: the four analysis folders, and `augmentation/`
  except the marker of a first training, which the loop writes). Nothing a phase commits
  may join that way, `derived_data/` included.
  Finishing it is strict about what the run committed (a changed or missing network,
  training set or record stops the start, and the message says what can be done) and takes
  what the analysis commands write (`analysis/`, `evidence/`, `modes/`,
  `posterior_chains/`) as it finds it, with a note in the record: those are rewritten
  without the run lock, cannot be put back as they were, and a start that refused them
  could never be got past.
* The call journal (`run/objective_calls.py`) flushes every record, so that a likelihood that kills its process leaves them
  behind, and syncs at most once a second, before a call. Syncing every record through its own open and close cost 3.8 ms per
  evaluation on a laptop's disk (19 us now), which is what an analytic design of many points notices, and a slow likelihood
  is still synced before every call.
* `tests/test_run_recovery.py` and `tests/pipeline_recovery_smoke.py` (five runs through the
  command line, each once refused) cover this, and `tests/mpi_loop_recovery_smoke.py` runs
  the loop itself on two ranks, killed inside an acquisition and continued twice.

## Before changing `training/audit.py`

`acquisition.audit` evaluates the true likelihood at a new network's own maxima before the
acquisition chain runs; read [docs/audit_optima.md](docs/audit_optima.md). Keep these:

* Opt-in. Without the key the loop is exactly as it was, under MPI too: the master sends
  one `bcast_idle(False)` only when the key is set.
* Collective under MPI. The master announces every batch with `bcast_idle(True)` and the
  end with `bcast_idle(False)`; the workers loop on it. `k` is a setting, never the rank
  count, so results do not depend on the number of ranks.
* The budget is `fraction` × `n_append` per iteration in `rounds` rounds. Few rounds of
  many evaluations is deliberate: a round costs a search and a fine-tune on the master, its
  evaluations are parallel. Do not replace it with many small rounds without the
  measurement in the doc.
* The audit points are all trained on; the network's own validation set stays held out.
  A fine-tune keeps its final weights (the validation points never include audit points).
* The network kept is the one of the round with the lowest highest maximum, among those
  searched since the best true value last moved, not the last round's. A settling
  fine-tune that does not improve the fit near the best is undone.
* A true value that is not finite is never added to genuine data. The opt-in
  `failure_policy: penalize` uses 1–4 nearby probes within the same total call budget,
  then a separate synthetic constraint if all fail. Never mix penalties into real
  data, derived training, validation, best-fit estimates or loss calibration. See the
  failure-policy section of the audit documentation.
* `RUN/audit/audit_it_N.csv` plus `data_it_N.csv` supplies genuine training data;
  optional `audit_penalties_it_*.csv` files separately record synthetic constraints.
  A continued run restores genuine audit additions when it keeps that network; a new
  training reapplies earlier constraints only under the enabled penalize policy.
* The fine-tunes are compiled without XLA. Every XLA compilation
  leaves memory mappings for the life of the process, a fine-tune compiles every time, and
  runs of 21 iterations ended at Linux's limit of 65 530 mappings. Do not turn XLA back on
  there, or add another compile-per-call training step, without counting mappings over a
  full run (the doc has the numbers).
* It corrects optimism only. `tests/test_audit.py` covers the search, the loop, the
  budget, the best-round rule, failed evaluations and the files.
* `chain: true` (opt-in; "The chain search" in the doc) adds a last look to a round whose
  climbs flag nothing, in a boost iteration: the network's own acquisition chain, cooled and
  climbed as the boost's mode search begins (`sampling.modes.cooled_peaks`), run on the
  master inside the audit's collective. Without the key `audit_optima` and the loop are what
  they were (`client.run_chain` is the sampling step's code, moved into a function). Keep
  these: the summits are taken where they are (`summit_maxima`), never climbed again by
  `find_maxima`, which loses narrow ones; the chain rounds have an allowance of their own;
  the best round is chosen among rounds of the same kind of search (compared with the
  climbs' rounds the deeper look lost and the fine-tunes were undone); the sampling step
  reuses the chain only when `weights_digest` says the network is the one it sampled.
  `tests/mpi_audit_chain_smoke.py` runs it in the loop on two ranks. Measured on one real
  two-mode run after the fact and on small analytic ones; do not make it a default on that.
* `posterior` (opt-in, with `chain`; "The posterior pilot" in the doc) evaluates cooled T = 1
  walkers after the rounds for maxima and the choice of the kept network, and fine-tunes when
  the network's mean excess over the truth there is above `posterior_tolerance`. Keep it after
  those rounds: they, their budget, the generator's draws and the kept round must stay what
  they were without it. A fine-tuned network gets a new chain before any walker of it is
  evaluated, and after the last fine-tune a final one, which the sampling step reuses; never
  evaluate walkers of a network that is no longer the one they were cooled on.
  All-nonfinite pilot batches are counted and explicitly unassessed, with no fine-tune.
  After pilot tuning (or an improved best true value), `clear`/`highest_excess` must
  describe full climbs plus chain summits of the final weights, not the pre-pilot model.

## Before changing `profiles/residual_newton.py`

Read the "residual Newton" section and the version history in
[docs/hybrid_optimisers.md](docs/hybrid_optimisers.md). The method was developed against
the **real** SINU TT likelihood (CLASS++), not only the analytic 31D target, because the
two behave differently: the real residual carries ~1e-3 - 2.5e-3 of value noise at
probe scale, and the network's flattest direction (the tau_reio - A_s degeneracy) runs
into the tau prior edge within about one network sigma, where the truth drops by tens of
log L. Hence the small initial trust radius, the long central-difference baseline along
the flattest directions and a tolerance (1e-2) a few times the gradient-noise floor.
Judge a change on a network standing in for the truth with added noise of 0.0025 (calls and the worst local gap)
and on real CLASS++ nodes, never on the 31D target alone. Results must not depend on `workers`
(`tests/test_residual_newton.py` checks serial = parallel and resume).

A node whose linear algebra fails (`LinAlgError`: numpy's solve raises for a singular system
and for one with a NaN in it) ends as `singular` with its best true point, and passes the
state it was given on along its chain; a transfer step that fails is skipped (the axion run's two A_osc studies each lost their last node, and with it the whole study,
in `ascend` of the corrected model at large amplitude). A node that does not fail is
unchanged (`tests/test_newton_singular.py`). Where the NaN came from was not established.

A node whose start the likelihood refuses (a prior edge, a Boltzmann failure, which the cluster backend turns into -inf) has
no value (`refused_start`: `logl` and `x` None, `reason` "refused start", flag 1) and passes on the state it was given in the
same way, also in 2D; the study goes on, and it stops only if every node is refused. The reports read over the nodes
that have a value (`profiles/metric.py` refuses a NaN; `reporting.crossings_with_values`) and warn. A
checkpoint keeps this on a resume, since the objective is deterministic. Within a study a refused point is also
not called again (`ConcurrentLedger.refused`); a resume asks once more, as it imports finite values only.

A gradient round that ends exactly at the node's call budget still gets its verdict (the
test costs no call; the node was reported out of budget). Both proxy testbeds are unchanged
by it, node by node.

Two rules, each with a test there: the curvature bound certifies
convergence only after a step inside the trust radius whose point is the incumbent (after a
step at the radius, or when a probe is better, a new gradient round follows); and a step
without a finite true value is a failed step that does not enter the curvature model. As an
excess of -inf it was written in as the stiffest curvature allowed, and the node and those
started from its state ran out of budget.

## Before changing `evidence/`

`python client.py evidence` (BEST nested sampling of a network); read
[docs/evidence.md](docs/evidence.md), "Single precision, and when BEST stops". Keep these:

* BEST works in single precision and adds ln Z up in it. It is given the network minus a
  reference near the peak (the best training value, itself a single-precision number, so the
  difference is exact), and the reference is added back in double precision to ln Z, ln Z_box
  and the chain's column. Do not hand BEST raw values of a real likelihood again: at -15656,
  taken to the end, the sum was 0.46 low with a quoted error of 0.03 (measured;
  `tests/test_evidence_precision.py`).
* `--tolerance` is the fraction of the evidence left in the live points, and BEST gets its
  logarithm. BEST's own tolerance is compared with ln(Z_live / Z_dead): a positive value
  stops every run at one half.
* The default is 2048 live points, a power of two: BEST adds the prior volume up in single
  precision, `n_live_updates / n_live` at every iteration, and that sum is exact only then
  (the command says so for another number). Do not go back to a round decimal default.
* `--n-live-updates` is a sixteenth of the live points by default (128 of 2048; 10 until 14
  October 2026). BEST's time is its network calls in sequence, and a call of 128 points costs
  about twice one of 10: 2.5 to 4.3 times faster, the same ln Z on Gaussians with exact
  evidences, SINU and a 49-parameter network, and the same (BEST's own, about −0.6) bias on the
  exact 31-parameter ridge (`docs/evidence.md`, "Live points replaced per iteration"). A
  single-thread benchmark of a small network says otherwise; do not judge it on one.
* The region of the boosts is the default, by the shared resolver (`sampling/targets.py`);
  `--region none` is the bare network. A bare `--region` must keep meaning "the region, or
  stop": the cluster's job script passes it. `--region E` is the same region with its edge
  at E reaches (another target: its output gets `_edgeE` and stands beside the default one,
  and the job script leaves its own `--region` out when the arguments carry one).
* The summary in `hpc/analysis/evidence_summary.py` withholds the correction of an evidence whose own posterior (its chain
  beside the JSON) lies where its true check has no sample: above `UNSAMPLED` (5%) of it at
  network values above the highest sample, or farther from the samples' mean than the
  farthest, where about 1 / (n + 1) is expected. Without the chain (or the check's `x` and
  `names`) the correction is made as before. The samples' own distances are leave-one-out:
  measured from a mean and covariance they entered, the farthest of 500 in 49 parameters left
  2 to 5% of sound networks outside and withheld a correction . Measured:
  at most 0.5% (higher) and 0.75% (farther) on twelve real networks of 16 to 49 parameters,
  99.9% on the SINU branch's network 21, whose evidence lived on a false corner maximum.
* The output's `region` is the resolved entry or None. Analysis caches record full target
  provenance (`sampling/provenance.py`). In that summary an evidence and a true check
  that both record a target must record the same one, or there is no correction. A result
  from before targets were recorded is joined by the network's number and the summary names it: withholding
  those corrections emptied the table of existing runs, whose evidences take hours each.
* The summary compares the networks with each other (the difference to the last one, the
  spread of the corrected evidences) whatever each was integrated over, with a note when
  that differs: every boost network has its own region, so a rule that compares only equal
  targets compares nothing, and the comparison is what the table is for. Different prior
  conventions are not compared. The convention is `convention_of`: the convention's name,
  each prior as defined, the constant that follows from them. Never the whole `priors` table
  of an output: its rows also hold the box and the mass sampled at its faces, which differs
  in any two runs (compared, it blanked every comparison of real files while every test,
  whose fixtures had no such table, passed). Test the summary on files shaped as
  `client.py evidence` writes them (`Summary.realistic` in `tests/test_hpc_evidence.py`).
* Seed-scatter notes need the same target, convention, stopping rule and tolerance, and no
  capped run. A recorded target that cannot be used is not a legacy result: no correction.
* Analysis identities belong to verified private snapshots (`sampling.targets.load_snapshot`),
  not whatever model is on disk when sampling ends. Keep the loaded network, metadata,
  resolved region and hash together; a replacement during sampling must never relabel results.
  Tests exercise real evidence/derived-chain producers and the evidence summary.
  `client.py modes` (`target` in its JSON, `target_json` in its samples) and the derived view
  (`source_target` and `source_files` in `view.json`) follow the same rule; a new consumer of
  a saved network takes it from `load_snapshot` too.
* A chain replaced by one of the same target is not archived (the report samples every
  network again each run); `corner.py` and `make_config.py` read `posterior_it_N.npz` only,
  never an archived `posterior_it_N.previous-<id>.npz`.
* BEST's sum of ln X is in single precision too, exact only for a power of two of live
  points. That is BEST's; it is documented, not corrected.

## Before changing `hpc/refused.py` or `hpc/launch.py`

The launcher of real runs; `hpc/README.md` has the use. Points the likelihood cannot evaluate ("refused") and
`CLIENT_REFUSED_VALUE` are described in `refused.py`. Keep these:

* Opt-in. Unset, a refused point is -inf and CLiENT drops it, as without the launcher. The one thing the launcher always does is
  turn a Cobaya logged error at a point into -inf, and a log-posterior of NaN (a theory code's output not finite) as well.
  Passed on as NaN, CLiENT counts it as a point without a value and every corrected mode weight of its boost or true check is
  withheld; this rule is for the launcher only, and the core's own NaN rule (above) stays as it is.
* The value belongs to the run: it is in the training set of every network. It is written to `RUN/refused_value.json`, and a
  continued run given another value, or none, is stopped.
* The value is applied in the likelihood object of every rank, not in the launcher's batch hook: the audit's evaluations do
  not pass through `client.broadcast_and_evaluate`. The launcher wraps their transport as well
  (`utils.mpi_utils.broadcast_and_evaluate_checked`), for the same rounding and the journal, with `kind` "audit" in its lines.
* With `design` the value does not exist until the design has been evaluated. Rank 0 sets it from the first evaluation of a new
  run, gives it to that batch's refused points before CLiENT sees the result, and writes `logs/refused_value_<job>.json`, which the
  other ranks read at their next refused point. Keep that order.
* The journal, the report and the true check count refused points as the values equal to the refused value, which they read
  from `RUN/refused_value.json` whatever the job's environment says. A refused point is never a finite true value: finite error
  moments exclude it, corrected moments give it zero weight against all surrogate draws, and the evidence correction keeps it in the
  denominator. Unresolved failures withhold corrections (`tests/test_hpc_true_check.py`, `tests/test_hpc_evidence.py`).
  `design_value` rounds so that no real value equals it; a value chosen by hand must not be one the likelihood can return.
* This lives outside `client.py` on purpose.
* The build of classy and camb a job evaluates with is decided by one rule, `boltzmann_build.location_problem`, used by `launch.py`
  and `true_profiles/cluster_backend.py`. Without `CLASSPP_GUARD_DIR` or `CLASSPP_BUILD_DIR` it is the environment's build; with
  the first, classy must be the patched build in that folder's `lib/`, with the second the unpatched build of another commit in its
  `lib/`; both set is refused. Never patch the environment's classy itself (`tests/test_hpc_boltzmann_build.py`).
* Nothing in `hpc/` names a machine. What one site needs (install, partitions, its runs and queues) lives in that site's own
  repository beside CLiENT, which sets `CLIENT_HPC_DIR` so that the launcher's `logs/` and the analysis programs' `runs/` are
  its own (for Grendel: the private `AarhusCosmology/ClientCluster`).
* The patch (`hpc/patches/classpp_extrap_guard.patch`) changes CLASS++'s source extrapolation only where the two-point fit is ill posed
  (dark acoustic oscillations); it is a regularisation of uncalibrated halofit and HMcode output, not a model. At every point where the fit is
  well posed the expression is the original, and the spectra there were bit for bit those of the unpatched build at 40 random points
  of the SIDR + DRMD box and its best fit, with halofit and HMcode. Judge a change on that comparison.

## Before changing `hpc/true_profiles/`

The launcher of true-profile studies; `hpc/README.md` has the use. Keep these:

* It changes no file of CLiENT. `cluster_backend.install()` replaces `profiles.evaluation.initialize_backend` / `evaluate_backend` in the
  launcher's process. The two backend functions stay module-level: a spawned backend process finds them by name.
* The region of a boost network is the profile setting `region` (section above), which `make_config.py --region` writes into the YAML.
* A refused point is -inf in a profile, never the run's refused value: the job script unsets `CLIENT_REFUSED_VALUE`.
* The settings beyond the profile YAML (`CLIENT_PROFILE_NEWTON`, `CLIENT_CAMB_ACCURACY_TARGET`) are written beside the study and a
  resumed study must be given the same.
* `CLIENT_PROFILE_NEWTON` values are outside what the residual Newton stage was validated with; do not make them defaults on that evidence.
* `tests/test_hpc_profiles.py` covers the backend wrapper, the settings, the ranges and the status report.

## Invariants

* A reported true profile value is always an evaluated native objective; when that
  objective includes priors, do not label it a likelihood-only chi-square.
* A study's manifest must hold every current key: one that lacks a key is refused, and a new
  key therefore ends the resume of older studies.
* Stage names stay `corrected/anchor` and `corrected/{direct,node}/NAME/J`. The stage reloads
  finished nodes (each checkpoint holds the state its successor needs) and re-runs an
  interrupted node from the ledger's cache.
* Everything a `corrected` node sees must be fixed before it starts (its fixed predecessor's
  state); anything timing-dependent breaks serial = parallel.
* Do not run the profile command under `mpirun`. The code allows `workers` up to the
  host's CPU count.
* `chain_length` (default 2; null: one seed per profile) decides which nodes start cold. Any
  setting must give results that
  depend on it alone, never on `workers` or completion order: seeds run together, a chain
  starts when its own seed is done, and ties between nodes are broken in plan order.
  `benchmarking/schedule_newton.py` replays a journal on any number of workers; use it,
  not wall times on a loaded machine, to judge parallel efficiency.
* `profiles/hybrid_surface.py` implements joint 2D surfaces with fixed-coordinate
  masks and deterministic independent-row Newton chains. Keep both physical values
  exact on backend requests; never transfer state between rows.
* `continuation` (2D, opt-in; docs/create_profiles_2d.md, "Continued rows")
  starts every node after a row's seed from its predecessor's true optimum, read from that
  node's checkpoint, and the seed from the best training point (or the scan's optimum). Without
  it there is no `continuation` in `method.json`.
  It needs `chain_length: null`: a node's start comes from its own chain, never from another
  row or from anything that depends on completion order.
* A study's manifest is compared on resume by what the run's files hold (their hashes) and
  by their place inside the run, never by absolute paths: a run directory that was moved or
  synced elsewhere with the same files resumes. The stored manifest is not rewritten.
* A resumed 2D study reads the scan it finished and checked (`surrogate_surface.json` with
  the names, for the same axes) instead of scanning again: the node checkpoints' signature
  holds every start, so a scan done again had to be identical bit for bit.
* Augmentation/validation nuisance designs use the L-BFGS-B node optimiser
  (`profiles/surrogate.py`).
* A study whose manifest lacks a key is not resumed: there is no filling of keys for
  studies made before them.

## Before changing how a profile study treats a refused point

`WorkflowLedger.evaluate_many` returns -inf for a point without a value, as `EvaluationLedger.__call__` and the hybrid ledger do; a
raised error is still raised. Its callers decide, because they use the value differently. Keep these:

* The label audit stops: its rows are the best the run has, so a refusal there means another backend, and -inf must not reach its JSON.
* A design or validation point is left out and listed (`record_refused`, `refused_points.json`), never redrawn, which would make the
  number of calls depend on what is refused, and never given a penalty value. Never let -inf into `augmented_data.npy` or `validation.npz`:
  a -inf target makes the msre loss NaN.
* Validation pairs index ridge rows: `without_refused` drops those touching a refused row and renumbers the rest. A parameter's profile
  pairs share its central node, so losing that row loses them, and with no pair of a kind left the study stops: the contrast score needs them.
* More than half of a batch refused stops the study (`MAX_REFUSED_SHARE`, a judgement). A study with nothing refused is unchanged.
* A refused point is not remembered across a resume: a batch finishes and is saved, so only one interrupted mid-batch asks again, and a
  repaired backend must get its chance. `tests/test_profiles_workflow.py` runs a real retraining study with part of the box refused.
* A report never fills in a node without a value. It is unknown, not excluded (an analytic case: 25 written in the spline at a node whose
  true value is 0.5, a false hole in the contours). In 2D the true curves become linear contours on the nodes with values; in 1D the
  crossings are read over them and the report warns.

## Before changing a program in `hpc/` that needs MPI and a likelihood

`hpc/analysis/true_check.py` cannot run in the unit suite (MPI and a Cobaya
likelihood), and a line of it that no test ran once called a string: the job
would have stopped after its true evaluations. Keep these:

* Anything such a program prints or computes from its results is a function that a test
  calls (`report_line`, `chain_line`, `network_entry`, `narrower_than_the_run`), not text
  inside `main`.
* After changing it, run it: `mpirun -n 2 python hpc/analysis/true_check.py RUN
  --cobaya input/cobaya/banana_lcdm16_wide.yaml --networks N --points 200 --steps 2000
  --discard-chains` on a scratch copy of a run of the analytic 16-parameter target (a
  small run) takes seconds.
* `tests/test_source_compiles.py` compiles every file with warnings as errors. A compiler
  warning is a defect waiting for its line to run; do not silence it.
* A consumer is checked on files its real producer wrote (`WhatTheCommandsWrite` in
  `tests/test_hpc_evidence.py`), not on files rebuilt for the test.

## Environment and tests

From the repository root in the environment declared by `environment.yaml`:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_*.py'
```

The suite includes serial/parallel equivalence, interrupted recovery, boundary-release and
failed-probe checks. It uses analytic
targets and does not need CLASS++ or Planck likelihood data.

`python tests/release_smoke.py` runs the whole workflow through the command line, in about three minutes, and
`python tests/run_mpi_smokes.py` the MPI paths. After changing a thing that compares evaluated coordinates with chosen ones, or
the launcher, run `tests/mpi_train_domain_smoke.py` as well, which goes through `hpc/launch.py`.

## Known limits

* **Networks with several modes.** A network whose region has several ellipsoids is sampled cell by cell and weighted by its own evidence in each
  cell (`sampling/mixture.py`): by the derived chain, and by the report and true check in `hpc/`. Do not substitute starting proportions or an
  older boost's corrected weights; keep the per-cell diagnostics, and do not use weights the sampler marks as not reliable. Chain caches carry the
  sampler method and version separately from `target_json`. Unreliable weights withhold moment, ESS and evidence corrections; retain the true
  evaluations and pointwise errors. Every reader of a chain asks how it was sampled (`hpc/analysis/chain_sampling.py`, 
  `derived.chain.weights_problem` for `derived --exact`, and the record `benchmarking/benchmark.py` keeps in its cached chain). The mark is the efficiency of the bridge's Gaussian mixture as a proposal for the cell, not
  the chain's length: do not write "sample longer" as the remedy. Before a real run with several modes, check the cell weights against the boost's own
  (`boost_it_N.json`) on its first networks.
* **The objective identity keeps its scope** (`run/identity.py`: user-site packages and the likelihood's component files are recorded). A package
  upgrade therefore stops each continued run once, and `--accept-identity-change` after a look at the listed differences is the answer; do not
  narrow the record to avoid that.
* **Profiles.** Nodes whose optimum lies at rho 0.9 to 0.98 of the region stall, since their probes cross the wall (`profiles/region.py` INSIDE);
  second-order certification is not required for `converged` in the hybrid scan; the retraining path (`retrain: true`) raises on designs
  outside the region.
* **An uncleared audit does not stop the loop.** With `clear: false` the native loop goes on; see `docs/posterior_density_boost.md`.
* **The validation set under a warm start** is drawn afresh at every training. Measured on an analytic target at four seeds, the printed validation loss is
  optimistic (12% above the training loss where a held-out set is 42% above), and the networks are no worse for it; a persistent set was, if anything, slightly
  worse. Do not read `val_loss` as a network's error. A warm-started training of the boost phase keeps its last weights: with
  the best ones, most boost-phase trainings kept their first of about 400 epochs. Branches of three runs with `last` were 0.48 to 0.86 of the originals' error at true
  values where those had kept an early epoch, and the same where they had not (client.py, `restore_best`). Not measured in the ordinary loop, whose networks decide the
  hand-over (the 1.2 T threshold was measured with `best`): do not extend it there without that measurement.

## Running a study

Same command as the original code (`python client.py profiles …` or `python -m profiles …`,
see [docs/create_profiles.md](docs/create_profiles.md)); the optimiser keys go in the
profile YAML ([input/profiles_example.yaml](input/profiles_example.yaml)).
