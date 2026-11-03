"""Check generated schema coverage/defaults and local guide links without heavy imports."""
import ast
from dataclasses import fields
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'_ext'))
from prepare import configuration, LINK, prepare


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main():
    configs=load('docs_run_config',ROOT/'config/config.py')
    profiles=load('docs_profile_config',ROOT/'profiles/config.py')
    schema=configuration(ROOT)
    checked=0
    for entry in schema:
        module=profiles if entry['cls']=='ProfileConfig' else configs
        cls=getattr(module,entry['cls'])
        assert {f.name for f in fields(cls)}=={f['name'] for f in entry['fields']},entry['cls']
        if cls.__name__=='Config': continue
        values={}
        for f in entry['fields']:
            if f['default']=='required' or f['default'].endswith('otherwise required)'):
                values[f['name']]= {'int':4,'float':1.,'str':'example'}.get(f['type'])
        for key, value in {'wrapper': 'cobaya', 'sampler': 'aies', 'metric': 'gaussian_posterior_drift', 'validation_split': 0.2,
                           'sampling_strategy': 'lhs'}.items():
            if key in values: values[key] = value
        obj=cls.from_dict(values) if hasattr(cls,'from_dict') else cls()
        for f in entry['fields']:
            if f['default']=='required' or f['default'].endswith(('(section absent)', 'otherwise required)')): continue
            expected=ast.literal_eval(f['default'])
            assert getattr(obj,f['name'])==expected,(entry['section'],f['name'],getattr(obj,f['name']),expected)
            checked+=1
    # Explicit user-facing claims and the absent/null/empty distinctions.
    raw=__import__('yaml').safe_load((HERE/'examples/gaussian_client.yaml').read_text())
    cfg=configs.Config.from_dict(raw)
    assert cfg.acquisition.audit is None and cfg.acquisition.boost is None
    assert cfg.training.warm_start
    assert cfg.training.lr_schedule=='none' and cfg.training.msre_ess_floor==.1
    assert cfg.acquisition.n_inject==cfg.prior.n_inject==0
    assert profiles.ProfileConfig().retrain is False and profiles.ProfileConfig().added_points==0
    assert profiles.ProfileConfig.load(parameters=['x'],added_points=10).retrain
    assert profiles.ProfileConfig().chain_length==2
    assert configs.AcquisitionConfig.from_dict(dict(raw['acquisition'],audit=None)).audit is not None
    try: configs.AcquisitionConfig.from_dict(dict(raw['acquisition'],boost={}))
    except ValueError: pass
    else: raise AssertionError('An empty boost must not be documented as valid')
    conditional=next(f for e in schema if e['cls']=='ConvergenceConfig' for f in e['fields'] if f['name']=='threshold')
    assert conditional['default'].startswith(str(configs.ConvergenceConfig.TEMPERED_THRESHOLD))
    c=configs.ConvergenceConfig.from_dict({'metric':'tempered_batch_scatter','max_iterations':30})
    assert c.threshold==1.2 and c.consecutive==2
    try: configs.ConvergenceConfig.from_dict({'metric':'gaussian_posterior_drift','max_iterations':30})
    except KeyError: pass
    else: raise AssertionError('Drift threshold must remain required')
    boost=configs.BoostConfig.from_dict({'n':1000,'start_iteration':15})
    assert boost.boosts is None and boost.density_filter is False
    assert all(getattr(boost,k) is None for k in ('background','region','mixing','region_quantile','train_domain','mode_depth','balance_floor'))
    for name in ('recommended_near_gaussian','recommended_non_gaussian'):
        recommended=configs.Config.from_dict(__import__('yaml').safe_load((ROOT/'input'/(name+'.yaml')).read_text()))
        assert recommended.acquisition.boost.density_filter and recommended.training.lr_schedule=='plateau',name
    assert profiles.ProfileConfig().region=='run' and profiles.ProfileConfig().surrogate_optimizer=='hybrid'
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'profile.yaml'
        path.write_text('parameters: [x]\nregion: null\n')
        assert profiles.ProfileConfig.load(path).region is None
    assert profiles.ProfileConfig.load(parameters=['x'],region=0.8).region==0.8
    assert 'tensorflow' not in sys.modules and 'cobaya' not in sys.modules
    prepare(ROOT,HERE)
    provenance=json.loads((HERE/'_generated/provenance.json').read_text())
    broken=[]
    for rel in provenance['guide_sha256']:
        p=ROOT/rel
        for match in LINK.finditer(p.read_text()):
            url=match[2].split('#')[0]
            if not url or ':' in url or url.startswith('//'): continue
            dest=(p.parent/url).resolve()
            if dest.is_relative_to(ROOT) and not dest.exists(): broken.append({'page':rel,'target':url})
    assert not broken,'Missing repository-relative links: '+json.dumps(broken)
    print(json.dumps({'schema_fields':sum(len(e['fields']) for e in schema),
                      'defaults_checked_against_parser':checked,'guides':len(provenance['guide_sha256']),
                      'status':'passed'}))

if __name__=='__main__': main()
