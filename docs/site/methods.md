# Scientific methods

Let $\ell(x)$ denote the **native log-objective** actually returned by the backend, and $s_\theta(x)$ its network approximation inside the saved domain. The objective may include priors. Preserve that convention when training, comparing fits or computing evidence.

## Tempered acquisition and the posterior

A chain sampling the tempered surrogate targets

$$p_T(x) \propto \exp[s_\theta(x)/T].$$

For samples of that distribution, the unnormalised importance weight to its own T=1 surrogate target is

$$w(x) \propto \exp[(1-1/T)s_\theta(x)].$$

If the chain stores $\log p_T=s_\theta/T$ up to a constant, this is $\exp[(T-1)\log p_T]$. Subtract the largest log-weight before exponentiation. These weights change the temperature; they do not correct network error.

The Kish effective sample size is

$$N_{\mathrm{eff}}=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.$$

It measures weight concentration and is **not correlation-adjusted**. Inspect autocorrelation and mode exploration separately. The identically named loss-weight ESS (`training.msre_ess_floor`) is computed over training weights and answers another question.

## Profiles

For coordinates of interest $q$ and nuisance coordinates $\eta$,

$$\ell_{\mathrm{prof}}(q)=\max_{\eta:(q,\eta)\in\mathcal D}\ell(q,\eta).$$

The surrogate stage solves the analogous problem for $s_\theta$. The true stage starts from useful surrogate solutions and constructs a local residual model,

$$m(x)=s_\theta(x)+r_b+g_r^T(x-x_b)
       +\tfrac12(x-x_b)^TQ_r(x-x_b),\qquad r=\ell-s_\theta.$$

True probes measure the residual; surrogate curvature provides geometry for local steps. Accepted reported values come from the true backend, not from $m$. Boundary probes, trust regions, continuation and stopping are described in the [solver guide](_generated/library/docs/hybrid_optimisers.md).

A plot of $2[\ell_{\max}-\ell_{\mathrm{prof}}(q)]$ is a native-objective difference. Call it likelihood-only $\Delta\chi^2$ only if the objective convention justifies that label. A local stop flag does not establish a global maximum or confidence coverage.

## Boosts and separated modes

For mode region $R_k$, define surrogate and true integrals with the same measure and domain:

$$\widehat Z_k=\int_{R_k}e^{s_\theta(x)}\,dx,\qquad
 Z_k=\int_{R_k}e^{\ell(x)}\,dx.$$

If $x$ is sampled from the normalised surrogate restricted to that mode, then

$$Z_k=\widehat Z_k\,
\mathbb E_{x\sim\widehat p_k}\!\left[e^{\ell(x)-s_\theta(x)}\right].$$

This motivates true evaluations of posterior draws in each detected mode. The identity does not guarantee an accurate finite-sample estimate: concentrated correction weights or missed regions can defeat it. Density-filtered boost rows are selected points, not those unfiltered posterior draws; the unweighted formula must not be applied to them.

Read [boost derivation](_generated/library/docs/boost.md), [mixing-based grouping](_generated/library/docs/boost_modes_mixing.md), and [density-filter qualifications](_generated/library/docs/posterior_density_boost.md).

## Evidence and the integration measure

For a normalised physical prior $\pi(x)$, Bayesian evidence is $Z=\int L(x)\pi(x)\,dx$. A uniformly normalised training box, a transformed coordinate prior and a native backend posterior can have different measures. Additive objective constants matter for evidence even when they cancel from a relative profile. The [evidence guide](_generated/library/docs/evidence.md) specifies the conversion and recorded prior ranges.

## Batch error and the handover test

Let $\widehat\ell_k$ be the network used to choose an ordinary acquisition batch $B_k$, and let $\ell$ be the native backend log objective. On the $m$ valid pairs, define

$$
e_i=\widehat\ell_k(x_i)-\ell(x_i),\qquad
\bar e=\frac1m\sum_i e_i,\qquad
s_k=\sqrt{\frac1m\sum_i(e_i-\bar e)^2},\qquad
r_k=\sqrt{\frac1m\sum_i e_i^2}.
$$

The implementation uses population scatter (denominator $m$), and also records $\max_i|e_i|$. Nonfinite predictions/returns and an explicitly supplied refused-value sentinel are left out. The computation is in float64. These errors are measured before fitting on $B_k$ and before ordinary injection is appended. Their reuse adds no evaluations, but adaptive batch selection means they are not a representative independent posterior test.

For configured consecutive count $q$, temperature $T$ and threshold $a$, the verdict at network $N$ is

$$
V_N=\max_{k=N-q,\ldots,N-1}\frac{s_k}{T},\qquad
\text{pass only if every measurement exists and }V_N<a.
$$

The default for this opt-in metric is $q=2$, $a=1.2$. A constant objective offset contributes to RMS but not scatter, motivating scatter for the ordinary-loop decision. That does not make relative offsets between separated modes harmless: they affect mode weights and need mode-specific checks. The reported criterion is a handover heuristic supported by the [measured suite](_generated/library/docs/tempered_convergence.md), not a bound on posterior or evidence error.

For genuinely unfiltered draws from a normalised surrogate posterior on the same domain and with the same prior convention, a separate identity gives

$$
Z_{\rm true}/Z_{\rm network}
=\mathbb E_{p_{\rm network}}\!\left[\exp(\ell-\widehat\ell)\right].
$$

`hpc/analysis/evidence_summary.py` estimates this from saved posterior true checks. Finite samples, correlation, missed support and low weight ESS limit the estimate. Density-filtered boost rows have a different selection distribution and cannot be inserted into this expectation uncorrected.
