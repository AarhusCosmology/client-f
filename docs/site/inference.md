# Inference workflows

## True-objective profiles

A profile maximises over nuisance coordinates at fixed values of the chosen parameter or pair. It does not integrate them out. The standard workflow starts from a saved model, finds surrogate optima and refines against actual backend returns:

```sh
python client.py profiles RUN_DIR --params PARAMETER \
  --points 15 --workers 4 --output PROFILE_DIR
```

By default 1D and 2D studies have **`retrain: false` and `added_points: 0`**, so augmentation, validation acquisition and training before profiling are skipped. They remain explicit options; requesting a positive added-point budget enables retraining when its flag is omitted. Saved studies resume their recorded policy.

The surrogate stage is the hybrid scan by default (`surrogate_optimizer: hybrid`), and `legacy` selects L-BFGS-B with continuation passes, which can do better on a rugged 2D network. The true stage is residual Newton. `chain_length: 2` exposes concurrent work by adding cold starts. Worker count changes scheduling, while chain settings can change results and cost. Choose workers for the host; do not launch profiles under MPI.

A network trained on boosts is searched inside its saved boost region by default (`region: run`). See the precise 1D/2D behaviour and resume rules below. The objective-label audit recomputes the saved labels with the highest values.

- [1D command and settings](_generated/library/docs/create_profiles.md)
- [Joint 2D surfaces](_generated/library/docs/create_profiles_2d.md)
- [Hybrid and residual-Newton mathematics](_generated/library/docs/hybrid_optimisers.md)

## Feeding profile evaluations back into training

This is a **separate opt-in operation**, not automatic augmentation before profiling:

```sh
python client.py augment RUN_DIR --profiles PROFILE_DIR -n 300
```

It selects genuine profile evaluations through a density filter and writes augmentation files for a subsequent continued training run. The first training that incorporates them gets a longer schedule. See [profile-to-training augmentation](_generated/library/docs/augment_training.md).

## Modes and posterior samples

```sh
python client.py modes RUN_DIR
```

This analyses a saved network and writes to its `modes/` directory. It does not call the true likelihood. Mode identification, bridge estimates and mixing diagnostics must be interpreted together; a tempered chain's occupancy alone is not a reliable separated-mode weight. See [mode analysis](_generated/library/docs/boost_modes_mixing.md).

## Bayesian evidence

```sh
python client.py evidence RUN_DIR
```

The command uses the BEST nested sampler on the network. Install BEST as described in the [evidence guide](_generated/library/docs/evidence.md). The reported prior convention matters: `ln Z` and `ln Z_box` are not interchangeable, and surrogate accuracy affects the answer. User-specified prior ranges must be recorded with any model comparison. A network trained on boosts is integrated inside its region by default; `--region none` is the bare network.

## Derived parameters

Derived values can be recorded with true evaluations and emulated for later chains. Profiles in derived coordinates require their coordinate constraint and true validation. Read the [derived-parameter guide](_generated/library/docs/derived_parameters.md) for `derived`, `derived-view`, backend conventions and supported workflows.

## Joining runs on adjacent boxes

```sh
python client.py combine OUTPUT --parameter PARAMETER \
  --part RUN_LOW --part RUN_HIGH --band LOW:HIGH
```

This puts runs of one objective on adjacent boxes in one parameter into one emulator, blended over the band, for a likelihood whose structure in that parameter one network does not capture. The evidence, profile and `modes` commands read the output as any run, but it is not a run to continue. See [domain splitting](_generated/library/docs/combine.md).

## Exporting a network

```sh
python client.py export RUN_DIR OUTPUT
```

This writes a network as a TensorFlow SavedModel that loads without CLiENT, by default inside the region of its boosts; `--no-region` is the bare network. See [export](_generated/library/docs/export.md).

## Profiles inside a saved boost region

The selected network is paired with the latest `boost/region_it_M.npz` whose M is no later than its iteration. Profile `region: run` uses that file's edge, `region: null` uses the bare network, and `region: 1.0` (or another positive number) overrides its extent. Without a saved region file, each setting uses the bare network. A double-precision differentiable wall confines the surrogate optimisation; it does not change native physical priors or turn refused true evaluations into finite values.

For 1D scans, an optimum beyond 98% of the region edge is marked outside. The workflow retains the consecutive inside segment around the best node, shortens the parameter range and redistributes the same number of nodes. It records `region_cuts` and `ranges.json`, allows at most four scans, and fails if fewer than two adjacent inside nodes remain or the retry limit is exhausted. This precedes the true slice refinements; setup and objective-audit calls can already have occurred.

**2D searches use the same confinement, and a rectangle is not trimmed:** a surface with nodes whose optimum lies at the region's edge or beyond stops after its scan, before any true call of a node, and names them; give ranges inside the region. These results concern the explored region; they do not establish a global profile over undiscovered modes.

The resolved region file, hash and edge enter the manifest. Resume checks the resolved choice, not merely whether the YAML says `run`. For the bare network on a boosted run, select `region: null`; a run with no region file is always searched over the whole box.

For true profiles and evidence corrections on real likelihoods, see [real likelihoods on a cluster](cluster.md).
