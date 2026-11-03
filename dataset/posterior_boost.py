"""Experimental posterior-focused boost selection using CLiENT's existing filter.

Opt-in through acquisition.boost.density_filter; disabled by default.
Input clouds must be posterior samples *including residence multiplicity*, not
unique jump-chain states. Each mode supplies an equal-size cloud: the target is
q_bal = sum(q_k)/K, independent of the surrogate's mode evidence. With `shares`
(acquisition.boost.balance_floor, dataset/boost.py mode_shares) mode k's cloud is
that size times its share instead: q = sum(s_k q_k)/sum(s_k). Selection can
accept unequal numbers per mode because existing coverage enters the deficit.
Do not pass selected points to boost_report's unweighted evidence correction.
"""
from types import SimpleNamespace
import numpy as np
from dataset.acquisition import select_points


def select_posterior_boost(dataset, mode_samples, n, *, seed=None, pool_factor=10, shares=None):
    """Return distinct new points and selection diagnostics; never mutate dataset.

    Uses the same whitening, k-NN estimator on both sides, and sequential deficit
    updates as normal acquisition. The target temperature is locally overridden
    to one; the run's hot acquisition setting is left unchanged. A finite pool
    may return fewer than n points; there is no unfiltered quota-filling fallback.
    """
    if n < 1 or pool_factor < 1 or not mode_samples:
        raise ValueError('positive budget/pool_factor and at least one mode required')
    clouds=[np.asarray(x,dtype=np.float32) for x in mode_samples]
    ndim=dataset.inputs.shape[1]
    if any(x.ndim!=2 or x.shape[1]!=ndim or not len(x) or not np.isfinite(x).all() for x in clouds):
        raise ValueError('mode samples must be nonempty finite arrays in parameter order')
    rng=np.random.default_rng(seed)
    if shares is None:
        size=min(len(x) for x in clouds)
        sizes=[size]*len(clouds)
    else:
        shares=np.asarray(shares,dtype=float)
        if shares.shape!=(len(clouds),) or not np.all(np.isfinite(shares)) or not np.all(shares>0) or shares.max()!=1:
            raise ValueError('shares: one positive number per mode, the largest 1')
        # The largest common size the clouds allow, each cut to its share of it, at least one state.
        size=min(int(len(x)/s) for x,s in zip(clouds,shares))
        sizes=[min(len(x),max(1,int(round(size*s)))) for x,s in zip(clouds,shares)]
    # Uniform subsampling of states preserves residence probabilities. Unlike
    # resampling, it does not create extra copies to force the mode quota.
    balanced=np.concatenate([x if len(x)==m else x[rng.choice(len(x),m,replace=False)] for x,m in zip(clouds,sizes)])
    # The ordinary selector collapses consecutive repeats. Group ALL identical
    # coordinates first, preserving their counts, including float32 rounding.
    order=np.lexsort(balanced.T[::-1]); balanced=balanced[order]
    # select_points draws a pool of pool_factor*n distinct states and leaves the rest for the
    # reference sample. One or two modes can hold fewer distinct states than that pool: a real
    # run's single mode had 198 885 for a pool of 200 000 (n 10 000, pool_factor 20; 25 October
    # 2026) and the selector's error stopped the run. Only where that error would come is the
    # pool made smaller, so that the reference keeps min(training points, half the states);
    # every other call is the one it was.
    distinct=len(np.unique(balanced,axis=0)); n_current=len(dataset.inputs)
    used=pool_factor
    if n_current>dataset.n_neighbors and distinct-min(pool_factor*n,distinct)<=dataset.n_neighbors:
        used=max(1,min(pool_factor,(distinct-min(n_current,distinct//2))//n))
    view=SimpleNamespace(inputs=dataset.inputs,knn_index=dataset.knn_index,
                         n_neighbors=dataset.n_neighbors,target_temperature=1.)
    points,info=select_points(view,balanced[:,None,:],np.zeros((len(balanced),1)),
                             n,1.,used,batch_size=1,rng=rng,exclude_existing=True)
    info.update(target='equal-mode T=1 posterior',modes=len(clouds),states_per_mode=size,
                requested=n,selected=len(points),evidence_correction_valid=False)
    if shares is not None:
        info.update(target='floored-share T=1 posterior',states_per_mode=sizes,shares=shares.tolist())
    if used!=pool_factor:
        info.update(pool_factor_requested=pool_factor,pool_factor_used=used,distinct_states=distinct)
    return points,info
