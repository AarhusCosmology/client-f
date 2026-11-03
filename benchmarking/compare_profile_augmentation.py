"""Compare original/+1000/+2000 profiles, refining crossings beyond the plotting grid."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarking.profile_likelihood import finite_or_none
from profiles.metric import dpm,as_dchi2
from profiles.surrogate import _bounded_optimizer as surrogate_optimizer


def refined_metrics(profile, reference, optimiser, names):
    output={}
    for name in profile['names']:
        i=names.index(name)
        g=np.asarray(profile['profiles'][name]['grid'])
        xopt=np.asarray(profile['profiles'][name]['xopt'])
        d=as_dchi2(profile,name)
        rg=np.asarray(reference['profiles'][name]['grid'])
        rd=as_dchi2(reference,name)
        centre,scale=(rg[0]+rg[-1])/2,(rg[-1]-rg[0])/2
        u=(rg-centre)/scale
        a,b,c=np.polyfit(u,rd,2)
        quadratic=bool(a>0 and np.max(abs(np.polyval([a,b,c],u)-rd)) < 1e-7*max(1,np.max(abs(rd)))
                       and abs(c-b*b/(4*a))<1e-6)
        output[name]={}
        for label,level in [('1 sigma',1.),('2 sigma',4.)]:
            evaluations=[]
            def f(value):
                j=int(np.argmin(abs(g-value)))
                v,x,diag=optimiser(i,value,[xopt[j],profile['best_x']])
                evaluations.append(dict(value=float(value),logl=float(v),
                                        stationary=diag['stationary'],
                                        projected_gradient_inf=diag['projected_gradient_inf']))
                return 2*(profile['best_logl']-v)-level
            roots=[]
            for j in range(len(g)-1):
                left,right=d[j]-level,d[j+1]-level
                if left*right<0:
                    # Re-evaluate brackets with the same nuisance optimizer; if an
                    # updated optimum invalidates the bracket, flag it rather than
                    # silently reporting an interpolated root as refined.
                    fl,fr=f(g[j]),f(g[j+1])
                    if fl*fr>0:
                        continue
                    roots.append(float(brentq(f,g[j],g[j+1],xtol=1e-8*(g[-1]-g[0]),rtol=1e-12)))
                elif left==0:
                    if abs(f(g[j]))<1e-6:
                        roots.append(float(g[j]))
            if d[-1]==level and abs(f(g[-1]))<1e-6:
                roots.append(float(g[-1]))
            roots=sorted(set(roots))
            truth=[]
            if quadratic:
                truth=(centre+scale*np.sort(np.roots([a,b,c-level]))).tolist()
            valid=bool(quadratic and len(roots)==2 and len(truth)==2
                       and all(g[0]<=v<=g[-1] for v in truth)
                       and all(e['logl']<=profile['best_logl']+1e-6 for e in evaluations))
            score=float(np.sum(abs(np.array(roots)-truth))/(truth[1]-truth[0])) if valid else None
            output[name][label]=dict(score=score,emulator_roots=roots,reference_roots=truth,
                                     valid=valid,reference_quadratic=quadratic,
                                     all_evaluations_stationary=all(e['stationary'] for e in evaluations),
                                     evaluations=evaluations)
    return output


def peak_bias(profile):
    """Twice the true value at the surrogate's best point minus the surrogate's; None when the
    likelihood refused that point (the null of finite_or_none)."""
    true=profile.get('true_at_best')
    return None if true is None else float(2*(true-profile['best_logl']))


def check_grids(profile, reference, label):
    """The comparison subtracts node by node and plots both on the reference's grid: another
    --n-grid fails only after the refined crossings, and another --grid-frac does not fail at all."""
    for name in profile['names']:
        own,theirs=np.asarray(profile['profiles'][name]['grid'],float),np.asarray(reference['profiles'][name]['grid'],float)
        if own.shape!=theirs.shape or not np.allclose(own,theirs,rtol=1e-9,atol=0):
            raise ValueError(f"{label}: the grid of {name} is not the reference's "
                             f"({len(own)} nodes on [{own[0]:g}, {own[-1]:g}] and {len(theirs)} on [{theirs[0]:g}, {theirs[-1]:g}])")


def compare(directory):
    from model.network import load_model
    from likelihood.surrogate import SurrogateMetadata
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory=Path(directory)
    settings=json.loads((directory/'experiment.json').read_text())
    reference=json.loads((directory/'reference.json').read_text())
    md=SurrogateMetadata.load(directory/'metadata.json')
    budgets=[0,*settings['budgets']]
    profiles={b:json.loads((directory/f'profiles_{b}.json').read_text()) for b in budgets}
    for budget,profile in profiles.items():
        check_grids(profile,reference,f'profiles_{budget}.json')
    scores={}
    csv_rows=[]
    for budget,profile in profiles.items():
        modelpath=settings['source_model'] if budget==0 else directory/f'model_plus{budget}.keras'
        optimiser=surrogate_optimizer(load_model(modelpath),np.array(list(md.bounds.values())),
                                      seed=settings['seed'],precision='float64',gtol=1e-4,
                                      ftol=1e-15,retry_starts=2)
        grid_scores=dpm(profile,reference)
        refined=refined_metrics(profile,reference,optimiser,md.param_names)
        shape=np.concatenate([as_dchi2(profile,n)-as_dchi2(reference,n) for n in profile['names']])
        raw=np.concatenate([np.asarray(profile['profiles'][n]['raw'],float) for n in profile['names']])
        summary={}
        for label in ['1 sigma','2 sigma']:
            vals=[refined[n][label]['score'] for n in profile['names'] if refined[n][label]['score'] is not None]
            summary[label]=dict(median=float(np.median(vals)) if vals else None,
                                maximum=max(vals) if vals else None,n=len(vals),
                                grid_median=float(np.nanmedian([grid_scores[label][n][0] for n in profile['names']])))
        for n in profile['names']:
            csv_rows.append([budget,n,grid_scores['1 sigma'][n][0],grid_scores['2 sigma'][n][0],
                             refined[n]['1 sigma']['score'],refined[n]['2 sigma']['score']])
        scores[budget]=dict(summary=summary,grid_scores=grid_scores,refined=refined,
                           shape_rms=float(np.sqrt(np.mean(shape**2))),
                           shape_max_abs=float(np.max(abs(shape))),raw_rms=float(np.sqrt(np.nanmean(raw**2))),
                           peak_bias=peak_bias(profile),
                           optimization_quality=profile['optimization_quality'],
                           nonstationary_grid_nodes=sum(not d['stationary'] for n in profile['names']
                                                       for d in profile['profiles'][n]['optimization']))
        print(f"+{budget}: refined metrics {summary}; shape RMS {scores[budget]['shape_rms']:.6g}",flush=True)
    (directory/'comparison.json').write_text(json.dumps(finite_or_none(scores),indent=2,allow_nan=False))
    with open(directory/'profile_metrics.csv','w') as f:
        w=csv.writer(f);w.writerow(['added_points','parameter','grid_dPM_1sigma','grid_dPM_2sigma',
                                    'refined_dPM_1sigma','refined_dPM_2sigma']);w.writerows(csv_rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','text.usetex':False,'font.size':11})
    colors=['#69727e','#2176b5','#d65335']
    for residual in [False,True]:
        fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        for ax,name in zip(axes.flat,settings['params']):
            g=np.asarray(reference['profiles'][name]['grid'])
            truth=as_dchi2(reference,name)
            if not residual:ax.plot(g,truth,color='black',lw=1.8,label='Exact profile')
            for color,budget in zip(colors,budgets):
                y=as_dchi2(profiles[budget],name)
                ax.plot(g,y-truth if residual else y,color=color,lw=1.5,
                        label='Original' if budget==0 else f'+{budget:,} points',
                        ls='--' if budget==0 else '-')
            if residual:
                ax.axhline(0,color='black',lw=.7)
                ax.set_ylabel(r'$\Delta\chi^2_{\rm emulator}-\Delta\chi^2_{\rm exact}$')
            else:
                ax.axhline(1,color='#aaa',ls=':',lw=.8)
                ax.axhline(4,color='#aaa',ls=':',lw=.8)
                visible=g[truth<=9]
                if len(visible)>1:ax.set_xlim(visible[0],visible[-1])
                ax.set_ylim(-.15,9)
                ax.set_ylabel(r'$\Delta\chi^2$')
            label=md.parameters[md.param_names.index(name)].label
            ax.set_xlabel('$'+label.strip('$')+'$')
            ax.grid(alpha=.18)
            ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,4),useOffset=False)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside upper center',ncol=len(labels),frameon=False)
        target=f"{len(md.parameters)}D · original iteration {settings['iteration']}"
        fig.suptitle(target+(' — profile errors' if residual else ' — cosmological profiles'),y=1.07)
        stem='profile_residuals' if residual else 'profile_comparison'
        fig.savefig(directory/f'{stem}.png',dpi=180,bbox_inches='tight')
        fig.savefig(directory/f'{stem}.pdf',bbox_inches='tight')
        plt.close(fig)
    lines=[f"# {len(md.parameters)}D profile augmentation comparison",'',
           'Smaller metrics are better. Refined scores solve surrogate crossings and compare to the quadratic exact reference.',
           '', '| Added points | Median dPM 1σ | Max dPM 1σ | Median dPM 2σ | Profile-shape RMS | Raw residual RMS |',
           '|---:|---:|---:|---:|---:|---:|']
    for budget,s in scores.items():
        one,two=s['summary']['1 sigma'],s['summary']['2 sigma']
        fmt=lambda x:'undefined' if x is None else f'{x:.5g}'
        lines.append(f"| {budget} | {fmt(one['median'])} | {fmt(one['maximum'])} | {fmt(two['median'])} | {s['shape_rms']:.5g} | {s['raw_rms']:.5g} |")
    lines.extend(['','The 2,000-point set contains the 1,000-point set; its second round follows the +1,000 model.',
                  'Each augmented model starts from the original checkpoint, with function-preserving scaler compensation and LR 3e-5 with plateau annealing.',
                  'The comparison tests this combined method; it does not isolate the effects of acquisition, learning rate and warm-start changes.',
                  'One seed per target. Numerical diagnostics are in comparison.json; local stationarity does not certify global optima.',
                  'Grid points were not used as explicit validation targets; whole augmentation node groups were held out for model selection.',
                  'No claim is made about unprofiled parameters or posterior accuracy from these profile-only tests.'])
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (directory/'status.txt').write_text('complete\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('directory',type=Path)
    compare(parser.parse_args().directory)
