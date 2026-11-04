# CLiENT documentation

The [searchable manual](site/README.md) brings these guides, the mathematics, worked examples and a generated
configuration and API reference together. These links work directly on GitHub.

## Start here

- [Installation and commands](../README.md#getting-started)
- [What this version adds to the first CLiENT](../CHANGES.md) and [how to move from it](../UPGRADING.md)
- [Performance measurements](../PERFORMANCE.md)
- Settings to start from: `input/recommended_near_gaussian.yaml` and `input/recommended_non_gaussian.yaml`

## Training, checking and improving a network

| Task | Guide |
|---|---|
| Decide when the ordinary loop can stop or hand over to boosts | [True-evaluation batch scatter](tempered_convergence.md) |
| Check false surrogate optima before acquisition | [Optimum audit](audit_optima.md) |
| Train on true evaluations of posterior samples | [Boosts, errors and no-mode fallback](boost.md) |
| Select density-deficit posterior points | [Density-filtered boosts](posterior_density_boost.md) |
| Identify modes by measured mixing | [Mode analysis](boost_modes_mixing.md) |
| Use a quadratic background and a known-mode region | [Background and region](boost_background_region.md) |
| Train each network several times and keep the best | [Several trainings per network](members.md) |
| Understand peak injection | [Injection window](injection_window.md) |
| Reuse true profile evaluations in later training | [Explicit augmentation](augment_training.md) |
| Continue or recover a run | [Continuing and recovering a run](run_recovery.md) |

## Inference and worked examples

- [1D true profiles](create_profiles.md), [2D surfaces](create_profiles_2d.md) and [hybrid and residual-Newton methods](hybrid_optimisers.md)
- [Derived parameters and their profiles](derived_parameters.md)
- [Bayesian evidence and prior conventions](evidence.md), and [profiles and evidence on an exact reference](PROFILES_AND_EVIDENCE.md)
- [Domain splitting](combine.md) and [export of a network](export.md)
- [The whole loop on Planck + DESI](boost_real_lcdm.md)

## Real likelihoods on a cluster

[`hpc/`](../hpc/README.md) has the launcher with an explicit training box and refused points, an example Slurm job,
the true check, the corrected evidence, branching and true profiles.
