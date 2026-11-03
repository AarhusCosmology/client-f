# Real cosmological likelihoods

## A whole-loop example

The [ΛCDM, Planck + DESI study](../_generated/library/docs/boost_real_lcdm.md) is a documented whole-loop real-likelihood example. It compares two runs with ten ordinary tempered iterations followed by five boosts, with auditing, density filtering, quadratic background and a region restriction explicitly configured.

The report includes true checks at independent draws from each final surrogate posterior, the comparison between the runs, intermediate errors and call accounting. Those checks measure the networks in the regions sampled. They do not prove global mode completeness, and two runs differing in an injection setting are not an exhaustive robustness study.

All cosmological conclusions belong to the likelihood, theory version, priors, domain and accuracy settings recorded in that report. Its 48-core timings are measurements for that setup, not a prediction for every machine.

## Separated modes

The [boost guide](../_generated/library/docs/boost.md) explains why separated self-interacting-neutrino modes motivated posterior-focused acquisition: a good fit in broad tempered training data did not fix the relative normalisation of the posterior modes. The [profile solver guide](../_generated/library/docs/hybrid_optimisers.md) records true CLASS++ tests and their remaining local stopping flags.

These studies use different likelihood combinations and parameter conventions. Compare like objectives, and retain the original physical bounds.

## Reproduction boundary

A real-likelihood rerun requires the matching data installations, CLASS++ build, backend versions, configuration and sufficient compute. This documentation build does not download those data or execute the experiment. `hpc/runs/lcdm_planck_desi/` holds the configuration of the Planck + DESI example; see [real likelihoods on a cluster](../cluster.md). Use a new output directory, and audit the objective convention before comparing values.

## Handover tests

The [tempered-convergence study](../_generated/library/docs/tempered_convergence.md) adds a controlled earlier-boost handover comparison on a 49-parameter real likelihood. It tests a scheduling decision against true posterior evaluations; it is not an all-model precision certificate.
