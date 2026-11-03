"""Profile-directed designs with near-ridge radii and explicit boundary handling.

The design is for training, not a posterior sample. Broad conditional-Gaussian draws
are mixed with fixed quadratic radii, ridge anchors and adaptive crossing nodes.
No proposal is clipped onto the prior box. A finite-sweep truncated-Gaussian Gibbs
fallback is recorded when rejection is inefficient; it is not claimed to be iid.
"""
import numpy as np
from scipy.stats import truncnorm

from dataset.peak_samples import _fit_quadratic
from profiles.metric import crossings


def select_nodes(profiles, names, bounds, broad_nodes=7):
    """Retain broad coverage, then weight the peak and 1/2-sigma crossings more."""
    bounds = np.asarray(bounds, float)
    nodes = []
    for name, rec in profiles['profiles'].items():
        i = names.index(name)
        lo, hi = bounds[i]
        grid, logl = np.asarray(rec['grid']), np.asarray(rec['logl'])
        entries = []
        def add(value, weight, role):
            value = float(value)
            for e in entries:
                if abs(e['value'] - value) <= 1e-10*(hi-lo):
                    e['weight'] = max(e['weight'], weight)
                    e['roles'].append(role)
                    return
            entries.append(dict(parameter=name, index=i, value=value, weight=weight, roles=[role]))
        for value in np.linspace(lo, hi, broad_nodes):
            add(value, 1., 'coverage')
        add(profiles['best_x'][i], 3., 'peak')
        for level in (1., 4.):
            for value in crossings(grid, 2*(profiles['best_logl']-logl), level):
                add(value, 4., f'crossing_{level:g}')
        for e in entries:
            j = int(np.argmin(np.abs(grid - e['value'])))
            e['start'] = np.asarray(rec['xopt'][j], float)
        nodes.extend(sorted(entries, key=lambda e:e['value']))
    return nodes


def allocate_counts(nodes, budget, rng):
    if not nodes or budget < 0:
        raise ValueError("nonnegative budget and at least one node required")
    if budget < len(nodes):
        weights=np.array([n['weight'] for n in nodes],float);counts=np.zeros(len(nodes),int)
        if budget:counts[rng.choice(len(nodes),size=budget,replace=False,p=weights/weights.sum())]=1
        return counts
    minimum = min(3, budget//len(nodes))
    weights = np.array([n['weight'] for n in nodes], float)
    ideal = (budget-minimum*len(nodes))*weights/weights.sum()
    counts = minimum+np.floor(ideal).astype(int)
    remainder = budget-int(counts.sum())
    # Randomize ties, retain largest-remainder allocation otherwise.
    order = np.lexsort((rng.random(len(nodes)), -(ideal-np.floor(ideal))))
    counts[order[:remainder]] += 1
    return counts


def _truncated_gibbs(P, low, high, temperature, rng, sweeps=60):
    delta = np.zeros(len(low))
    for _ in range(sweeps):
        for j in rng.permutation(len(low)):
            mean = -(P[j] @ delta-P[j,j]*delta[j])/P[j,j]
            sd = np.sqrt(temperature/P[j,j])
            delta[j] = truncnorm.rvs((low[j]-mean)/sd, (high[j]-mean)/sd,
                                     loc=mean, scale=sd, random_state=rng)
    return delta


def make_design(inputs, targets, profiles, names, bounds, n_points, optimise, *,
                seed=42, ridge_eig=1e-3, max_rejections=128):
    """Return exactly n_points distinct proposals plus full node/draw provenance.

    ``optimise(i, value, starts)`` returns (logL, full point, diagnostics). The caller
    evaluates true likelihoods; this function never uses analytic ground truth.
    """
    x, y, bounds = np.asarray(inputs,float), np.asarray(targets,float), np.asarray(bounds,float)
    if (x.ndim != 2 or x.shape[1] != len(names) or len(names) < 2
            or y.shape != (len(x),) or bounds.shape != (len(names),2)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.isfinite(bounds).all() or np.any(bounds[:,1] <= bounds[:,0])
            or max_rejections < 1):
        raise ValueError("invalid training data, bounds or rejection budget")
    fit = _fit_quadratic(x, y, ridge_eig=ridge_eig)
    precision = -(fit['V'] * fit['w_neg']) @ fit['V'].T
    rng = np.random.default_rng(seed)
    nodes = select_nodes(profiles, names, bounds)
    counts = allocate_counts(nodes, n_points, rng)
    lo, hi = bounds.T
    draws, records, node_records = [], [], []
    used = {row.tobytes() for row in x}
    for node_id, (node, count) in enumerate(zip(nodes, counts)):
        i, value = node['index'], node['value']
        peak, anchor, diagnostic = optimise(i, value, [node['start'], profiles['best_x']])
        anchor = np.asarray(anchor, float)
        if (anchor.shape != (len(names),) or not np.isfinite(anchor).all()
                or np.any(anchor < lo) or np.any(anchor > hi) or anchor[i] != value):
            raise ValueError("profile optimizer returned an infeasible anchor")
        mask = np.arange(len(names)) != i
        P = precision[np.ix_(mask,mask)]
        covariance = np.linalg.solve(P, np.eye(len(P)))
        L = np.linalg.cholesky((covariance+covariance.T)/2)
        std = fit['std'][mask]
        low, high = (lo[mask]-anchor[mask])/std, (hi[mask]-anchor[mask])/std
        for k in range(int(count)):
            role = 'anchor' if k == 0 else ('near_ridge' if k % 2 else 'broad')
            deficit = [0.05, 0.2, 0.5, 1.0][(k//2) % 4] if role == 'near_ridge' else None
            temperature = [0.25, 0.5, 1.0][(k//2) % 3] if role == 'broad' else None
            attempts, fallback, point = 0, False, anchor.copy()
            if role == 'anchor' and point.tobytes() in used:
                role, deficit = 'near_ridge', 0.05
            if role != 'anchor':
                while True:
                    attempts += 1
                    if attempts <= max_rejections:
                        u = rng.normal(size=len(P))
                        if role == 'near_ridge':
                            u *= np.sqrt(2*deficit)/np.linalg.norm(u)
                        else:
                            u *= np.sqrt(temperature)
                        delta = L @ u
                    else:
                        fallback = True
                        t = temperature if role == 'broad' else 2*deficit/len(P)
                        delta = _truncated_gibbs(P, low, high, t, rng)
                    point = anchor.copy()
                    point[mask] += std*delta
                    if (np.all(point >= lo) and np.all(point <= hi)
                            and point.tobytes() not in used):
                        break
                    if attempts > max_rejections+10:
                        raise RuntimeError("unable to generate a distinct feasible conditional draw")
            used.add(point.tobytes())
            displacement = (point[mask]-anchor[mask])/std
            realised = .5*displacement@P@displacement
            draws.append(point)
            records.append(dict(node=node_id, mode=role, requested_deficit=deficit,
                                temperature=temperature, proposal_attempts=attempts,
                                gibbs_fallback=fallback, realised_quadratic_deficit=float(realised),
                                nuisance_boundary=bool(np.any((point[mask]==lo[mask]) | (point[mask]==hi[mask])))))
        node_records.append({**{k:v for k,v in node.items() if k!='start'},
                             'count':int(count), 'anchor':anchor.tolist(),
                             'surrogate_logl':float(peak), 'optimization':diagnostic})
    result = np.asarray(draws)
    assert result.shape == (n_points,len(names))
    manifest = dict(method='adaptive_profile_radius_mixture_v1',seed=seed,n_points=n_points,
                    ridge_eig=ridge_eig,quadratic_rms=fit['resid'],
                    positive_eigenvalues=int((fit['w']>0).sum()),
                    floored_eigenvalues=int((np.abs(fit['w']) < ridge_eig*np.max(abs(fit['w']))).sum()),
                    nodes=node_records,draws=records,
                    near_ridge_radii_deficits=[0.05,0.2,0.5,1.0],broad_temperatures=[.25,.5,1.],
                    clipped_points=0,gibbs_fallbacks=sum(r['gibbs_fallback'] for r in records),
                    nonstationary_nodes=sum(not n['optimization']['stationary'] for n in node_records),
                    max_rejections=max_rejections,gibbs_sweeps=60)
    return result, manifest
