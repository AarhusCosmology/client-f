"""Augment an existing checkpoint in nested rounds and compare its cosmological profiles.

Example:
  python benchmarking/augment_profiles.py RUN -it 5 --reference EXACT.json -o OUTPUT

The +2000 dataset includes the first +1000 and adapts its next 1000 to that model.
Both models are independently warm-started from the original checkpoint, using the
same LR schedule and fixed original validation rows. Augmented node groups, not
individual neighboring draws, are assigned to validation. No MCMC is required.
"""
import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarking.profile_likelihood import finite_or_none, grid_for
from profiles.surrogate import _bounded_optimizer as surrogate_optimizer, scan_profiles
from dataset.profile_augmentation import make_design

COSMO = ['omega_b','omega_cdm','theta_s_100','ln10_10_A_s','n_s','tau_reio']


def save_json(path, value):
    Path(path).write_text(json.dumps(finite_or_none(value),indent=2,allow_nan=False))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile_model(model, data, md, params, true_lk, *, seed=42, n_grid=61, grid_frac=.5):
    bounds=np.array(list(md.bounds.values()),float)
    indices=[md.param_names.index(n) for n in params]
    grids=[grid_for(*bounds[i],n_grid,grid_frac) for i in indices]
    optimiser=surrogate_optimizer(model,bounds,seed=seed,precision='float64',
                                  gtol=1e-4,ftol=1e-15,retry_starts=2)
    rng=np.random.default_rng(seed)
    starts=[*data[np.argsort(data[:,-1])[-20:],:-1],bounds.mean(1),
            *rng.uniform(bounds[:,0],bounds[:,1],(10,len(bounds)))]
    start=time.monotonic()
    peak,anchor,records,quality=scan_profiles(optimiser,starts,grids,indices,
                                            max_passes=15,value_tol=1e-6)
    profiles={}
    for name,rec in zip(params,records):
        truth=np.array([true_lk.loglkl(x) for x in rec['xopt']])
        profiles[name]=dict(grid=rec['grid'].tolist(),logl=rec['logl'].tolist(),
                            xopt=rec['xopt'].tolist(),optimization=rec['diagnostics'],
                            true_logl=truth.tolist(),raw=(2*(truth-rec['logl'])).tolist())
    payload=dict(schema_version=2,names=params,full_parameter_names=md.param_names,
                 n_grid=n_grid,grid_frac=grid_frac,best_logl=float(peak),best_x=anchor.tolist(),
                 true_at_best=float(true_lk.loglkl(anchor)),profiles=profiles,
                 optimization_quality=quality,true_evaluation_kind='at_surrogate_optima',
                 optimizer_settings=dict(optimiser.options,precision='float64',retry_starts=2,seed=seed),
                 seconds=time.monotonic()-start)
    return payload,optimiser


def group_validation(manifest, rng, fraction=.1):
    count=len(manifest['nodes'])
    held=set(rng.choice(count,max(1,int(round(fraction*count))),replace=False).tolist())
    return np.array([r['node'] in held for r in manifest['draws']],bool),sorted(held)


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('run_dir',type=Path)
    p.add_argument('-it','--iteration',type=int,required=True)
    p.add_argument('-o','--output',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--params',default=','.join(COSMO))
    p.add_argument('--budgets',default='1000,2000')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--learning-rate',type=float,default=3e-5)
    p.add_argument('--n-grid',type=int,default=61)
    p.add_argument('--grid-frac',type=float,default=.5)
    p.add_argument('--max-epochs',type=int,default=None,
                   help='override saved epoch ceiling (primarily for smoke checks)')
    a=p.parse_args()
    budgets=[int(v) for v in a.budgets.split(',')]
    if budgets != sorted(set(budgets)) or not budgets or budgets[0]<=0:
        p.error('budgets must be positive and strictly increasing')
    if (a.n_grid<2 or not 0<a.grid_frac<=1 or not np.isfinite(a.learning_rate)
            or a.learning_rate<=0 or (a.max_epochs is not None and a.max_epochs<1)):
        p.error('invalid grid, learning rate or epoch ceiling')
    if a.output.exists() and any(a.output.iterdir()):
        p.error('output directory must be empty; existing experiments are never overwritten')
    a.output.mkdir(parents=True,exist_ok=True)
    import tensorflow as tf
    from config.config import Config
    from run.run import settings_file
    from likelihood.base import build_likelihood
    from likelihood.surrogate import SurrogateMetadata
    from model.network import build_model,load_model,model_background,transfer_warm_start
    from training.training import train_model,save_history
    from training.losses import build_loss

    run=a.run_dir.resolve()
    cfg=Config.from_yaml(settings_file(run))
    md=SurrogateMetadata.load(run/'metadata.json')
    params=a.params.split(',')
    if len(set(params)) != len(params) or any(n not in md.param_names for n in params):
        p.error('profile parameters must be unique metadata names')
    source_model=run/f'trained_models/model_it_{a.iteration}.keras'
    source_data=run/f'training_data/data_it_{a.iteration}.csv'
    original=np.loadtxt(source_data,delimiter=',',skiprows=1)
    true_lk=build_likelihood(cfg.likelihood.wrapper,
                            str(run/'likelihood_input'/Path(cfg.likelihood.input).name))
    if true_lk.param_names != md.param_names:
        raise ValueError('likelihood and saved model parameter order differs')
    metadata=dict(source_run=str(run),iteration=a.iteration,source_model=str(source_model),
                  source_data=str(source_data),source_model_sha256=sha(source_model),
                  source_data_sha256=sha(source_data),reference=str(a.reference.resolve()),
                  reference_sha256=sha(a.reference),original_points=len(original),
                  budgets=budgets,seed=a.seed,params=params,learning_rate=a.learning_rate,
                  training_warm_start='function_preserving_from_original_for_each_budget',
                  design='nested: each additional round follows the previous augmented model',
                  profile_precision='float64',n_grid=a.n_grid,grid_frac=a.grid_frac,
                  training_config=asdict(cfg.training),model_config=asdict(cfg.model),
                  epoch_ceiling=a.max_epochs or cfg.training.n_epochs,
                  tensorflow_version=tf.__version__,numpy_version=np.__version__)
    save_json(a.output/'experiment.json',metadata)
    md.save(a.output/'metadata.json')
    (a.output/'reference.json').write_bytes(a.reference.read_bytes())
    state=a.output/'status.txt'
    state.write_text('profiling original checkpoint\n')
    baseline=load_model(source_model)
    profiles,optimiser=profile_model(baseline,original,md,params,true_lk,seed=a.seed,
                                   n_grid=a.n_grid,grid_frac=a.grid_frac)
    save_json(a.output/'profiles_0.json',profiles)
    data=original.copy()
    validation=np.zeros(len(data),bool)
    validation[np.random.default_rng(a.seed).choice(len(data),int(.1*len(data)),replace=False)]=True
    previous=0
    for round_index,budget in enumerate(budgets):
        tf.keras.utils.set_random_seed(a.seed)
        state.write_text(f'designing {budget-previous} additional points for +{budget}\n')
        new,manifest=make_design(data[:,:-1],data[:,-1],profiles,md.param_names,
                                 np.array(list(md.bounds.values())),budget-previous,optimiser,
                                 seed=a.seed+1000*round_index)
        values=np.array([true_lk.loglkl(x) for x in new])
        if not np.isfinite(values).all():
            raise RuntimeError('true likelihood returned non-finite augmented targets')
        val_new,held=group_validation(manifest,np.random.default_rng(a.seed+round_index))
        manifest.update(round=round_index,total_budget=budget,held_out_nodes=held,
                        source_model_sha256=sha(source_model if previous==0 else a.output/f'model_plus{previous}.keras'),
                        true_likelihood_calls=len(new))
        save_json(a.output/f'design_round{round_index+1}.json',manifest)
        data=np.vstack([data,np.column_stack([new,values])])
        validation=np.r_[validation,val_new]
        np.savetxt(a.output/f'data_plus{budget}.csv',data,delimiter=',',
                   header=','.join(md.param_names+['loglkl']),comments='')
        np.save(a.output/f'validation_mask_plus{budget}.npy',validation)
        state.write_text(f'training +{budget} on {len(data)} total points\n')
        training=data[~validation]
        training=training[np.random.default_rng(a.seed).permutation(len(training))]
        # A boost network's background stays what it was, as in the loop (model_background).
        model=build_model(training[:,:-1],training[:,-1],cfg.model.n_layers,
                          cfg.model.n_neurons,cfg.model.activation,background=model_background(baseline))
        transfer_warm_start(baseline,model)
        loss=build_loss(cfg.training.loss,cfg.training.sigma_level,len(md.parameters),
                        float(training[:,-1].max()),targets=training[:,-1],
                        ess_floor=cfg.training.msre_ess_floor)
        start=time.monotonic()
        history,metrics=train_model(model,training[:,:-1],training[:,-1],loss,
                    a.learning_rate,a.max_epochs or cfg.training.n_epochs,cfg.training.batch_size,
                    cfg.training.validation_split,cfg.training.patience,
                    lr_schedule='plateau',lr_factor=cfg.training.lr_factor,
                    lr_patience=cfg.training.lr_patience,lr_min=cfg.training.lr_min,
                    lr_grace=cfg.training.lr_grace,
                    validation_data=(data[validation,:-1],data[validation,-1]))
        metrics.update(seconds=time.monotonic()-start,epochs_run=len(history.history['loss']),
                       points=len(data),train_points=len(training),validation_points=int(validation.sum()))
        model.save(a.output/f'model_plus{budget}.keras')
        save_history(history.history,a.output/f'history_plus{budget}.csv')
        save_json(a.output/f'training_plus{budget}.json',metrics)
        state.write_text(f'profiling +{budget}\n')
        profiles,optimiser=profile_model(model,data,md,params,true_lk,seed=a.seed,
                                        n_grid=a.n_grid,grid_frac=a.grid_frac)
        save_json(a.output/f'profiles_{budget}.json',profiles)
        previous=budget
    getattr(true_lk,'close',lambda:None)()
    state.write_text('training and grid profiles complete; ready for comparison\n')


if __name__=='__main__':
    main()
