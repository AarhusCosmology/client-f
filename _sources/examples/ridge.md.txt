# The 31D ridge likelihood

This worked example combines a 27-dimensional correlated Gaussian with four additional coordinates. Up to a constant, its native log-objective is

$$\ell(z,a,b,u,v)=C-\tfrac12(z-\mu)^T\Sigma^{-1}(z-\mu)
-\tfrac12(ab)^2-\tfrac12(v-u^2)^2,$$

with $(a,b)\in[0,5]^2$ and $(u,v)\in[-4,4]\times[0,16]$.

## Why it is informative

Every extra coordinate has a flat 1D profile: choose $b=0$, $a=0$, $v=u^2$ or an allowed square root for $u$. Yet its joint 2D surfaces have strong structure:

$$q(a,b)=(ab)^2,\qquad q(u,v)=(v-u^2)^2.$$

For a Gaussian parameter pair, the exact profile is

$$q(z_I)=(z_I-\mu_I)^T\Sigma_{II}^{-1}(z_I-\mu_I),$$

provided the conditional nuisance optimum lies inside the saved box. Check that qualification before using the unrestricted Gaussian formula as a reference.

These surfaces expose errors hidden by a 1D plot: a misplaced ridge, an inaccurate nuisance optimum, or interpolation error between grid nodes.

## Read the comparisons

The [hybrid optimiser guide](../_generated/library/docs/hybrid_optimisers.md) records separate surrogate and true-stage comparisons on this target. Its tables identify algorithms and budgets. The default also uses short continuation chains; do not transfer a measured wall time to another worker/chain configuration without measuring it.

## What to record in a reproduction

1. Saved network, training-data and backend identities; saved rows are not new calls.
2. Fixed coordinates and exact grids, with all nuisance bounds unchanged.
3. Surrogate optimum, true value at that optimum, and true-refined value.
4. Node error relative to the exact solution, plus unrestricted-anchor error.
5. Central-region and full-grid errors separately; remote tails can dominate the maximum.
6. Actual attempts, finite returns, failures, caches/imports and solver flags.
7. Contour interpolation error separately from the error at evaluated nodes.

A small local error or a successful solver flag does not prove global optimisation or interval coverage. The analytic reference is used to score the result, not to seed the optimiser with the answer.
