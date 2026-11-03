"""The training of a profile study with `retrain: true`: the 75/25 preset of added rows, the
broad design, the independent validation set and the frozen-scaler fit (profiles/workflow.py).

Functions consume the same objective convention as the baseline checkpoint.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import numpy as np
import tensorflow as tf
from dataset.profile_augmentation import _truncated_gibbs
from training.losses import solve_ess_floor
from profiles.metric import as_dchi2


def save(path, value):
    def convert(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,default=convert,allow_nan=False))
    temp.replace(path)


@dataclass(frozen=True)
class ProfileTrainingPreset:
    added_points: int = 2000
    broad_fraction: float = .75
    first_epochs: int = 400
    polish_epochs: int = 1600
    initial_lr: float = 1e-5
    transition_lr: float = 1e-6
    final_lr: float = 1e-7
    batch_size: int = 128
    seed: int = 42

    def __post_init__(self):
        if not isinstance(self.added_points,int) or self.added_points<0: raise ValueError('augmentation count must be nonnegative')
        if not 0 <= self.broad_fraction <= 1: raise ValueError('broad_fraction must be in [0,1]')
        if min(self.first_epochs,self.polish_epochs)<2: raise ValueError('each cosine phase needs >=2 epochs')
        if not 0 < self.final_lr <= self.transition_lr <= self.initial_lr: raise ValueError('invalid LR schedule')

    def learning_rate(self, epoch):
        if epoch < self.first_epochs:
            return self.transition_lr+.5*(self.initial_lr-self.transition_lr)*(1+np.cos(np.pi*epoch/(self.first_epochs-1)))
        k=min(epoch-self.first_epochs,self.polish_epochs-1)
        return self.final_lr+.5*(self.transition_lr-self.final_lr)*(1+np.cos(np.pi*k/(self.polish_epochs-1)))



def broad_design(original, profile, metadata, bounds, fit, optimizer, seed, n_points):
    """Paper recipe: 13 full-box nodes, conditional Gaussian T=.25/.5/1, clipping.

    This reconstructs the published recipe with updated shared baseline anchors;
    it does not claim to reproduce the unavailable original generator or RNG.
    """
    rng = np.random.default_rng(seed)
    precision = -(fit['V'] * fit['w_neg']) @ fit['V'].T
    nodes = [(name, v) for name in profile['names'] for v in np.linspace(*bounds[metadata.param_names.index(name)], 13)]
    points, records = [], []
    used = set(map(tuple, original[:, :-1]))
    for identifier, (name, value) in enumerate(nodes):
        index = metadata.param_names.index(name)
        rec = profile['profiles'][name]
        start = rec['xopt'][int(np.argmin(abs(np.array(rec['grid'])-value)))]
        _, anchor, quality = optimizer(index, value, [start, profile['best_x']])
        mask = np.arange(len(bounds)) != index
        P = precision[np.ix_(mask, mask)]
        covariance = np.linalg.solve(P, np.eye(len(P)))
        L = np.linalg.cholesky((covariance+covariance.T)/2)
        count = n_points//len(nodes) + int(identifier < n_points % len(nodes))
        for k in range(count):
            temperature = [.25, .5, 1.][k % 3]
            for attempt in range(100):
                proposed = anchor.copy()
                proposed[mask] += fit['std'][mask] * (L @ rng.normal(size=len(P))) * np.sqrt(temperature)
                point = np.clip(proposed, bounds[:, 0], bounds[:, 1])
                point[index] = value
                if tuple(point) not in used: break
            else: raise RuntimeError('Repeated broad-design points')
            used.add(tuple(point)); points.append(point)
            records.append(dict(node=identifier, parameter=name, value=value, temperature=temperature,
                                clipped=bool(np.any(point != proposed)), stationary_anchor=quality['stationary']))
    return np.array(points), records

def without_refused(global_data, ridge_data, profile_pairs, nuisance_pairs):
    """Held-out sets without the rows the likelihood refused. The pairs are of ridge rows: those
    touching a refused row go and the others are renumbered. With no pair left of a kind, or no
    row of a set, the score that selects the checkpoint cannot be computed, and the study stops:
    a parameter's profile pairs share one row, its central node."""
    keep_global, keep_ridge = np.isfinite(global_data[:, -1]), np.isfinite(ridge_data[:, -1])
    number = np.cumsum(keep_ridge) - 1

    def surviving(pairs):
        pairs = np.asarray(pairs, int).reshape(-1, 2)
        return number[pairs[keep_ridge[pairs].all(axis=1)]]

    found = dict(global_rows=keep_global.sum(), ridge_rows=keep_ridge.sum(),
                 profile_pairs=len(surviving(profile_pairs)), nuisance_pairs=len(surviving(nuisance_pairs)))
    empty = [name for name, count in found.items() if not count]
    if empty:
        raise RuntimeError(f"the likelihood refused validation points, and none of {', '.join(empty)} is left: the score "
                           "that selects the checkpoint cannot be computed; refused_points.json lists them")
    return global_data[keep_global], ridge_data[keep_ridge], surviving(profile_pairs), surviving(nuisance_pairs)


def validation_set(original, profile, metadata, bounds, fit, optimizer, truth, seed=20260925, *, global_count=512, nodes_per_parameter=9, canonicalize=None):
    rng = np.random.default_rng(seed)
    global_points = []
    original_rows = set(map(tuple, original[:, :-1]))
    attempts=0
    while len(global_points) < global_count:
        attempts+=1
        if attempts>10000*global_count:raise ValueError('could not draw distinct in-domain global validation points')
        p = original[rng.integers(len(original)), :-1] + .02*fit['std']*rng.normal(size=len(bounds))
        if np.all(p >= bounds[:, 0]) and np.all(p <= bounds[:, 1]) and tuple(p) not in original_rows:
            global_points.append(p)
    precision = -(fit['V'] * fit['w_neg']) @ fit['V'].T
    ridge, centres, profile_pairs, nuisance_pairs, provenance = [], [], [], [], []
    selected = profile['names']
    for name in selected:
        index = metadata.param_names.index(name)
        rec = profile['profiles'][name]
        grid = np.array(rec['grid'])
        if profile.get('grid_fractions',{}).get(name, .5) == 1:
            low, high = bounds[index]
        else:
            relevant = grid[as_dchi2(profile, name) <= 9]
            low, high = relevant[0], relevant[-1]
        parameter_centres = []
        mask = np.arange(len(bounds)) != index
        P = precision[np.ix_(mask, mask)]
        covariance = np.linalg.solve(P, np.eye(len(P)))
        L = np.linalg.cholesky((covariance+covariance.T)/2)
        for node in range(nodes_per_parameter):
            value = low + (high-low)*(node + rng.uniform(.2, .8))/nodes_per_parameter
            start = rec['xopt'][int(np.argmin(abs(grid-value)))]
            _, anchor, quality = optimizer(index, value, [start, profile['best_x']])
            centre = len(ridge)
            ridge.append(anchor); centres.append(centre); parameter_centres.append(centre)
            for deficit in [.2, 1.]:
                fallback = False
                for attempt in range(128):
                    u = rng.normal(size=len(P)); u *= np.sqrt(2*deficit)/np.linalg.norm(u)
                    delta = L @ u
                    point = anchor.copy(); point[mask] += fit['std'][mask]*delta
                    if np.all(point >= bounds[:, 0]) and np.all(point <= bounds[:, 1]): break
                else:
                    fallback = True
                    delta = _truncated_gibbs(P, (bounds[mask, 0]-anchor[mask])/fit['std'][mask],
                                              (bounds[mask, 1]-anchor[mask])/fit['std'][mask],
                                              2*deficit/len(P), rng)
                    point = anchor.copy(); point[mask] += fit['std'][mask]*delta
                assert np.isfinite(point).all()
                tolerance = 1e-12 * (bounds[:, 1]-bounds[:, 0])
                assert np.all(point >= bounds[:, 0]-tolerance) and np.all(point <= bounds[:, 1]+tolerance)
                point = np.clip(point, bounds[:, 0], bounds[:, 1])  # roundoff only
                nuisance_pairs.append([centre, len(ridge)])
                ridge.append(point)
                provenance.append(dict(parameter=name, node=node, deficit=deficit, gibbs_fallback=fallback,
                                       stationary_anchor=quality['stationary']))
        profile_pairs.extend([[parameter_centres[nodes_per_parameter//2], j] for j in parameter_centres if j != parameter_centres[nodes_per_parameter//2]])
    global_points, ridge = np.array(global_points), np.array(ridge)
    if canonicalize is not None:
        all_points=canonicalize(np.vstack([global_points,ridge]))
        global_points,ridge=all_points[:len(global_points)],all_points[len(global_points):]
    evaluate=getattr(truth,'evaluate_many',lambda xs:np.array([truth.loglkl(x) for x in xs]))
    global_data = np.column_stack([global_points, evaluate(global_points)])
    ridge_data = np.column_stack([ridge, evaluate(ridge)])
    return (*without_refused(global_data, ridge_data, profile_pairs, nuisance_pairs), provenance)

class SelectCheckpoints(tf.keras.callbacks.Callback):
    def __init__(self, validation, peak, c, out):
        super().__init__()
        self.validation, self.peak, self.c, self.out = validation, peak, c, out
        self.records, self.denominators = [], None
        self.best = {name:dict(score=float('inf'), epoch=None, weights=None) for name in ['pointwise','contrast']}

    def measure(self):
        residuals, components = {}, {}
        for name, key in [('global','global_data'), ('ridge','ridge_data')]:
            data = self.validation[key]
            pred = np.concatenate([np.asarray(self.model(data[j:j+512, :-1], training=False)).reshape(-1)
                                   for j in range(0,len(data),512)])
            residuals[name] = pred-data[:, -1]
            components[name] = float(np.mean((residuals[name]/(self.peak-data[:, -1]+self.c))**2))
        for name in ['profile', 'nuisance']:
            pairs = self.validation[f'{name}_pairs']
            components[name] = float(np.mean((residuals['ridge'][pairs[:, 0]]-residuals['ridge'][pairs[:, 1]])**2))
        if self.denominators is None:
            self.denominators = {k:max(v,1e-12) for k,v in components.items()}
        scores = dict(pointwise=.5*components['global']+.5*components['ridge'],
                      contrast=.5*components['global']/self.denominators['global']
                         +.25*components['profile']/self.denominators['profile']
                         +.25*components['nuisance']/self.denominators['nuisance'])
        assert np.isfinite(list(scores.values())).all()
        return scores, components

    def record(self, epoch, logs=None):
        scores, components = self.measure()
        self.records.append(dict(epoch=epoch, scores=scores, components=components))
        for rule, score in scores.items():
            if score < self.best[rule]['score']:
                self.best[rule] = dict(score=score, epoch=epoch, weights=self.model.get_weights(), optimizer=[v.numpy().copy() for v in self.model.optimizer.variables])
        if logs is not None: logs.update({f'val_{k}':v for k,v in scores.items()})
        save(self.out/'validation_history.json', self.records)

    def on_train_begin(self, logs=None): self.record(0)
    def on_epoch_end(self, epoch, logs=None): self.record(epoch+1, logs)


def fit_profile_model(model, data, validation, output, *, preset=None, original_rows=None, counts=None):
    """Frozen-scaler training and paired selectors, independent of acquisition/backend."""
    preset=preset or ProfileTrainingPreset()
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    original_rows=len(data) if original_rows is None else original_rows
    dim=data.shape[1]-1; peak=float(data[:,-1].max())
    c0=.5*dim*(1-2/(9*dim)+3*np.sqrt(2/(9*dim)))**3
    c=solve_ess_floor(data[:,-1],peak,c0,.1)
    def msre(y,pred):
        y=tf.reshape(y,tf.shape(pred))
        return tf.reduce_mean(tf.square((pred-y)/(peak-y+c)))
    tf.keras.utils.set_random_seed(preset.seed)
    fixed=[v.numpy().copy() for v in model.non_trainable_weights]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=preset.initial_lr),loss=msre,jit_compile=True)
    model.optimizer.build(model.trainable_variables)
    monitor=SelectCheckpoints(validation,peak,c,out)
    order=np.random.default_rng(preset.seed).permutation(len(data))
    history=model.fit(data[order,:-1],data[order,-1],epochs=preset.first_epochs+preset.polish_epochs,
        batch_size=preset.batch_size,verbose=2,callbacks=[
            tf.keras.callbacks.LearningRateScheduler(lambda e,lr:preset.learning_rate(e)),monitor])
    for a,b in zip(fixed,model.non_trainable_weights): np.testing.assert_array_equal(a,b.numpy())
    model.save(out/'model_final.keras')
    selected={}
    for rule,best in monitor.best.items():
        model.set_weights(best['weights'])
        for var,value in zip(model.optimizer.variables,best['optimizer']): var.assign(value)
        scores,_=monitor.measure()
        np.testing.assert_allclose(scores[rule],best['score'],rtol=1e-7,atol=1e-12)
        model.save(out/f'model_{rule}.keras')
        selected[rule]=dict(epoch=best['epoch'],score=best['score'])
    save(out/'history.json',history.history)
    save(out/'training.json',dict(preset=asdict(preset),selected=selected,primary_selector='contrast',
        loss_peak=peak,loss_c=c,normalizers=monitor.denominators,fixed_scalers_verified=True,
        counts=counts,original_rows=original_rows,added_rows=len(data)-original_rows,
        caveat='Direct generation and uninterrupted RNG are not bitwise reproduction of the earlier matrix.'))
    return selected
