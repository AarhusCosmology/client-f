# CLiENT

**Learn an expensive cosmological objective. Check what the network learned. Use it for inference.**

CLiENT builds a differentiable neural-network approximation to a native log-probability supplied by a likelihood backend. It alternates true evaluations, training and sampling. Its tools then support posterior exploration, true-objective profiles, evidence estimates and derived parameters.

The source revision appears in the banner. This manual brings the guides, mathematical explanations and selected Python interfaces into one searchable reference.

```{important}
A well-mixed surrogate chain is not a validation of the true posterior. A stable acquisition distribution does not prove that every mode has been found. Read the [validation guide](validation.md) before interpreting a result.
```

## Choose a starting point

| You want to… | Start here |
|---|---|
| Try CLiENT on an inexpensive likelihood | [First run](getting-started.md) |
| Understand what is enabled | [Configuration and defaults](configuration.md) |
| Improve an existing network with boosts | [Training and boosting](workflow.md) |
| Compute profiles or evidence, join runs or export a network | [Inference workflows](inference.md) |
| Understand the mathematics | [Scientific methods](methods.md) |
| Extend a backend or inspect a function | [Developer guide](developer.md) and [Python reference](api/index) |
| See what this version adds | [Release overview](release-notes.md) |
| Run on a cluster | [Real likelihoods on a cluster](cluster.md) |
| Read a guide or a measurement in full | [Guide and report library](library.md) |

## The workflow

1. **Define the objective and domain.** Keep the backend's priors and parameter conventions explicit.
2. **Evaluate and train.** Start with a design, then add true evaluations chosen using the current network.
3. **Audit and improve.** Optional audits target false maxima; optional boosts target posterior accuracy and mode weights.
4. **Infer and validate.** Inspect sampling diagnostics, check against the true objective, and preserve call ledgers and provenance.

```{toctree}
:maxdepth: 1
:caption: Using CLiENT

getting-started
workflow
inference
configuration
methods
examples/index
validation
cluster
release-notes
```

```{toctree}
:maxdepth: 1
:caption: Reference and development

_generated/reference/configuration
developer
api/index
library
maintenance
```
