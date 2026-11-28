# Validation and limitations

## What each diagnostic establishes

| Diagnostic | Useful evidence | Does not establish |
|---|---|---|
| Held-out training loss | Approximation error on that held-out design | Accuracy in every posterior mode |
| Acquisition batch scatter | Error against true returns at selected new points; ordinary-loop handover signal | Posterior accuracy, a global error bound or mode completeness |
| Acquisition drift | Change between successive acquisition distributions | Posterior convergence or mode completeness |
| Acceptance and autocorrelation | Behaviour of the sampled chain | Agreement between the network and truth |
| Kish importance ESS | Concentration of the reported weights | Independence of correlated samples |
| Error at a boost's points | Error against true returns at posterior samples of the network | Accuracy where the network's posterior did not go. It decides nothing: the number of boosts is the user's |
| Cleared optimum audit | No remaining detected optimistic maxima under that search and budget | Absence of underestimated or unseen modes |
| Profile stop flag | The local solver's stopping condition | A global optimum or frequentist coverage |
| True checks on posterior draws | Local surrogate error where that chain went | Accuracy in missed regions |

## Call accounting

Record these separately: requested candidates, actual objective attempts, finite returns, nonfinite returns, exceptions, imported values and cache hits. Surrogate value/gradient/Hessian work is another category. An objective attempt is not an independently measured count of internal CLASS++ computations.

Reuse of saved training points does not turn them into new calls. A reported best fit must identify the objective convention, source coordinates and whether the value is a true evaluation or a surrogate prediction. Keep backend/data/configuration hashes with the results.

## Reading the existing studies

The [library](library.md) contains reports from several dates. Their settings are evidence about those experiments, and they are not automatically defaults. The [31D worked example](examples/ridge.md) explains the exact-reference checks; the [real-likelihood example](examples/cosmology.md) distinguishes posterior validation from global coverage.

No scientific likelihood run is triggered by building this site. The small [analytic identities example](examples/gaussian.md) can be executed in the documentation-only environment. The full CLiENT smoke example needs the scientific environment and is explicitly a separate command.

## Refused points and region-restricted inference

The launcher in `hpc/` optionally assigns a fixed finite sentinel to points the backend refuses. These are **synthetic training labels**, not successful finite objective returns. The policy is opt-in, stored in `refused_value.json`, and checked on continuation. Error summaries exclude the sentinel; counts must continue to distinguish refusals from real values. Cluster profile jobs turn this substitution off. This mechanism is separate from the audit's training-only penalty constraints.

A saved boost region describes coverage of known evaluated modes. Profiles or evidence restricted to it must name that restriction; cutting a 1D range to its interior is not evidence that the omitted physical domain contains no better solution. Inspect the recorded range cuts, region hash and edge. A 2D search does not cut its rectangle: it stops after the scan and lists the nodes outside the region.
