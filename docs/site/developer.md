# Developer guide

## Architecture map

| Area | Entry points | Responsibility |
|---|---|---|
| Run orchestration | `client.py`, `run/run.py` | New/resumed runs, stages, persistent outputs |
| Run state | `run/storage.py`, `run/identity.py`, `run/objective_calls.py` | Phase records, recovery, the objective's identity, call journals |
| Finished networks | `run/combine.py`, `run/export.py` | Domain splitting, export as a SavedModel |
| Configuration | `config/config.py`, `profiles/config.py` | YAML parsing, defaults and validation |
| Native objective | `likelihood/base.py`, backend adapters | Parameter order, native priors, derived outputs |
| Acquisition | `dataset/`, `sampling/` | Designs, density selection, boosts and modes |
| Model and training | `model/`, `training/` | Network/scalers, optimisation, audits |
| Profiles | `profiles/` | Surrogate scans and counted true refinement |
| Diagnostics and handover | `utils/held_out.py`, `run/metrics.py`, `convergence/convergence.py` | Pre-training batch errors, persisted metrics and ordinary-loop verdict |
| Profile region | `profiles/region.py` | Resolve/hash boost geometry, confine search and trim 1D grids |
| Cluster tools | `hpc/` | Explicit training box, refusal policy, journals and analysis jobs |
| Other inference | `evidence/`, `derived/` | Evidence conventions and derived-coordinate tools |
| Benchmarks | `benchmarking/` | Comparison with reference chains, the 31D exact-reference benchmark, wall-time model of a call journal |

The [Python reference](api/index) is deliberately selected, not a promise that every internal helper is a stable public API. AutoAPI parses source files without importing scientific dependencies.

## Backend contracts

Start with `BaseLikelihood` and `ParameterInfo`. Keep the parameter order consistent across coordinates, labels, bounds and returned values. The method called `loglkl` returns the backend's native objective; its name alone does not mean likelihood-only. Derived values have a separate ordered array and must not be confused with extra sampled coordinates.

A custom backend must preserve failures and nonfinite values honestly. Synthetic audit penalties belong to the separate training-only path, not to the genuine-data ledger or a best-fit result. See the API contract and [audit policy](_generated/library/docs/audit_optima.md).

## Reproducibility and concurrency

Checkpoint compatibility includes the objective, model, domain, method and continuation settings. Scheduling should not change the planned numerical work. Imported values and new true attempts remain separate. `chain_length` changes the profile plan; `workers` changes its execution. Test deterministic scheduling and resume before altering either implementation.

Read the [contributor guide](_generated/library/AGENTS.md) before method changes: it lists, module by module, what must stay true and the measurement behind it. Source-level contracts should explain units, bounds, shapes, ownership of files and numerical assumptions. Avoid comments that merely repeat a signature.

## Contributing documentation

Update the maintained guide with a feature change; the website consumes that original file. Update a docstring when its contract changes. Keep measurements dated and scoped, and link their data/configuration provenance. New features must say whether they are default or opt-in, and go behind a configuration key unless there is evidence to make them a default.

See [building and maintaining the manual](maintenance.md) for build commands, validation and publication boundaries.

## Persistence contracts

The selecting network's error must be measured before incorporating its batch; do not mix injection or refused sentinels into the ordinary convergence statistic. An early boost transition is persisted in `boost/start.json`; no-mode fallback is separately recorded and must not masquerade as a successful posterior batch.

A profile manifest records the resolved region file/hash/edge. Comparing only the literal YAML value `run` is insufficient, since a later network can have another region. Confinement applies to the search wrapper, not to the original network's stored weights or the backend's prior. Keep the 1D trimming rule separate from 2D behaviour: a surface with nodes outside the region stops after its scan with the list.

Cluster finite-refusal substitution is an explicit training policy. Preserve both native-return accounting and substituted labels, and disable substitution for true profiles. Site-specific environment controls should not silently become library defaults.
