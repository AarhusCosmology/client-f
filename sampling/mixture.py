"""The T = 1 sample of a network whose region has several ellipsoids.

One ensemble started on all of them keeps, where it does not cross between them, the shares
of its start and not the posterior's. Here every cell of the region (the points nearer to one
ellipsoid than to any other) has an ensemble of its own, the evidence of the network in each
cell is estimated by bridge sampling, and the sample is drawn from the cells in proportion.
Used by the derived chain (`client.py derived --chain`) and by hpc/analysis/report.py and
true_check.py; the boosts have their own mode search (sampling/modes.py), which does the same
for the modes it finds.
"""
import warnings

import numpy as np

# Below this efficiency of the bridge's proposal the evidence of a cell is not to be trusted.
MIN_RELIABLE_ESS = 0.1


def sample_region_mixture(surrogate, bounds, sampling, n_steps, thin, seed=None, diagnostics=None):
    """Sample disjoint cells of the saved region and mix by their current evidences.

    Nearest-ellipsoid cells partition the entire box, including overlaps: no posterior
    mass is counted twice. They need not be separate physical modes. Every cell gets an
    ensemble and a bridge estimate; starting populations never set the output weights.
    The equal-weight output is resampled and must not be treated as independent MCMC.
    """
    from scipy.special import logsumexp
    from sampling.modes import _ensemble, as_numpy, bridge_log_evidence

    region = surrogate.region
    lower, upper = np.asarray(bounds, float).T
    rng = np.random.default_rng(seed)
    values = as_numpy(surrogate.logpost)
    sampler, beta, target = _ensemble(surrogate.logpost, lower, upper, sampling.n_walkers, region)
    cells, draws, probabilities = [], [], []
    try:
        for k in range(len(region)):
            target.assign(k)
            sampler.run(n_steps=n_steps, initial_positions=region.start(sampling.n_walkers, rng, component=k),
                        burn_in=sampling.burn_in, thin=thin, chunk_size=sampling.chunk_size,
                        progress=getattr(sampling, 'progress', True))
            chain = np.array(sampler.chain(), copy=True)
            lp = np.array(sampler.log_prob(), copy=True)
            x, f = chain.reshape(-1, chain.shape[-1]), lp.ravel()
            if len(x) < max(40, 4 * len(lower)) or not np.isfinite(f).all():
                raise ValueError('Not enough finite region-cell samples for bridge weights; increase the chain length')
            inside = lambda y, k=k: (np.all((y >= lower) & (y <= upper), axis=1) & (region.labels(y) == k))
            log_z, spread, efficiency = bridge_log_evidence(x, f, values, inside, lower, upper, 6, rng)
            if not np.isfinite([log_z, spread, efficiency]).all() or efficiency <= 0:
                raise ValueError('Region-cell bridge estimate failed; no posterior mixture was written')
            if efficiency < MIN_RELIABLE_ESS:
                warnings.warn(f'Region cell {k}: bridge efficiency {efficiency:.3g} is low; '
                              'posterior mixture weights may be unreliable', RuntimeWarning)
            draws.append(x)
            probabilities.append(f)
            cells.append(dict(cell=k, log_evidence=log_z, log_evidence_spread=spread, efficiency=efficiency,
                              samples=len(x), acceptance=float(np.mean(sampler.acceptance_fraction().numpy())),
                              max_tau=float(sampler.max_tau)))
    finally:
        sampler.reset()
    weights = np.exp(np.array([c['log_evidence'] for c in cells]) - logsumexp([c['log_evidence'] for c in cells]))
    count = len(draws[0])
    labels = rng.choice(len(cells), size=count, p=weights)
    output, log_prob = np.empty_like(draws[0]), np.empty_like(probabilities[0])
    for k, cell in enumerate(cells):
        selected = labels == k
        # Without replacement: a cell is asked for no more rows than it has (the output is as
        # long as one cell's chain), and drawn with replacement the output held 63 to 77%
        # distinct rows where 93% and more were to be had.
        rows = rng.choice(len(draws[k]), size=int(selected.sum()), replace=False)
        output[selected], log_prob[selected] = draws[k][rows], probabilities[k][rows]
        cell.update(weight=float(weights[k]), output_samples=int(selected.sum()))
    unreliable = [c['cell'] for c in cells if c['efficiency'] < MIN_RELIABLE_ESS]
    info = dict(method='region-cell bridge mixture', cells=cells, resampled=True, unreliable_cells=unreliable,
                note='Current surrogate weights, not true-likelihood corrections. Resampling does not create independent samples; '
                     'inspect cell acceptance, autocorrelation and bridge efficiency/spread. No mode-discovery certificate.')
    if diagnostics is not None:
        diagnostics.update(info)
    print('Region-cell posterior weights: ' + ', '.join(f'{w:.4f}' for w in weights)
          + (f'; NOT RELIABLE for cell(s) {unreliable}: the bridge\'s proposal fits them badly (efficiency below '
             f'{MIN_RELIABLE_ESS:g}). A longer chain helps where the chain was too short for the fit; a cell whose '
             'shape the Gaussian mixture does not follow stays marked at any length' if unreliable else ''))
    return output.reshape(chain.shape), log_prob.reshape(lp.shape)
