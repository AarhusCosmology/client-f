"""Small real CLI runs: continued unchanged; identity drift refused, then accepted; a run
stopped inside its audit or inside an acquisition batch continued; a boost run with a region
continued twice without changing a committed file; a boost run that the drift criterion
does not stop; a run started again at an earlier network, refused and then rewound."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import yaml
ROOT=Path(__file__).resolve().parents[1]

def invoke(args, expected=0, wrapper=None, timeout=120):
    command=[sys.executable,str(ROOT/'client.py'),*map(str,args)] if wrapper is None else [sys.executable,str(wrapper),*map(str,args)]
    result=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
    if (result.returncode==0)!=(expected==0):
        raise AssertionError(result.stdout)
    return result.stdout

with tempfile.TemporaryDirectory() as temporary:
    d=Path(temporary);config=yaml.safe_load((ROOT/'docs/site/examples/gaussian_client.yaml').read_text())
    config['likelihood']['input']=str(ROOT/'docs/site/examples/gaussian_likelihood.yaml')
    config['training']['n_epochs']=2
    config['acquisition'].update(n_append=10,pool_factor=2)
    config['acquisition']['audit']=dict(rounds=1,starts=2,steps=2,epochs=1,tau=.001,fraction=.2,k=2)
    cfg=d/'config.yaml';cfg.write_text(yaml.safe_dump(config))
    # Manufacture an optimistic trained model so the audit must spend true calls.
    optimistic=d/'optimistic.py'
    optimistic_code="import sys\nsys.path.insert(0,"+repr(str(ROOT))+")\nimport client\noriginal=client.save_model\nseen=set()\ndef save(model,path):\n    if id(model) not in seen:\n        model.get_layer('loglkl_denormalization').target_mean.assign_add(100.)\n        seen.add(id(model))\n    original(model,path)\nclient.save_model=save\n"
    optimistic.write_text(optimistic_code+'client.main()\n')
    invoke([cfg,'-i',1,'-o',d/'ok'],wrapper=optimistic)
    run=next((d/'ok').iterdir())
    identity=json.loads((run/'objective_identity.json').read_text())
    invoke([run,'-i',1],wrapper=optimistic)
    assert identity==json.loads((run/'objective_identity.json').read_text())
    summary=json.loads((run/'objective_calls/summary.json').read_text())
    assert summary['stages']['initial/design']['attempts']==128
    assert sum(v.get('attempts',0) for k,v in summary['stages'].items() if k.startswith('audit/'))>0
    input_file=next((run/'likelihood_input').iterdir());input_file.write_text(input_file.read_text()+'\n# changed target input\n')
    refused=invoke([run,'-i',0],expected=1)
    assert 'identity changed' in refused and 'changed: input' in refused and '--accept-identity-change' in refused
    # A reviewed change is accepted once; the old record is kept and the next start needs no flag.
    assert 'Objective identity change accepted' in invoke([run,'-i',0,'--accept-identity-change'])
    assert len(list((run/'objective_identity_history').glob('*.json')))==1
    assert json.loads((run/'objective_identity.json').read_text())['accepted_changes'][0]['differences']==['changed: input']
    invoke([run,'-i',0])
    # Interrupt between the audited network write and the audit report/data commit.
    wrapper=d/'interrupt.py'
    wrapper.write_text("import sys\nsys.path.insert(0,"+repr(str(ROOT))+")\nimport training.audit\ndef stop(*a,**k):\n    raise RuntimeError('injected audit commit failure')\ntraining.audit.save_audit=stop\nimport client\nclient.main()\n")
    text=invoke([cfg,'-i',1,'-o',d/'broken'],expected=1,wrapper=wrapper)
    assert 'injected audit commit failure' in text
    broken=next((d/'broken').iterdir())
    assert (broken/'trained_models/model_it_0.keras').exists()
    # The network was saved, its audit was not committed: it is not taken as audited. The
    # continuation sets the phase's files aside and trains and audits network 0 again.
    text=invoke([broken,'-i',0])
    assert 'Phase training_it_0 did not finish' in text and 'Training model on' in text
    aside=next((broken/'interrupted').iterdir())
    assert aside.name.startswith('training_it_0-') and (aside/'trained_models/model_it_0.keras').exists()
    assert (broken/'trained_models/model_it_0.keras').exists() and (broken/'audit/audit_it_0.json').exists()
    assert json.loads((broken/'phases/training_it_0.json').read_text())['status']=='complete'
    invoke([broken,'-i',0])
    # A job that ends inside a batch of true evaluations (a time limit): the next one does the
    # acquisition again with the saved network.
    killer=d/'killed.py'
    killer.write_text("import sys, os\nsys.path.insert(0,"+repr(str(ROOT))+")\nimport client\noriginal=client.evaluate_true\ncalls=[]\n"
                      "def evaluate(likelihood, samples):\n    calls.append(1)\n    if len(calls)==2:\n        os._exit(137)\n"
                      "    return original(likelihood, samples)\nclient.evaluate_true=evaluate\nclient.main()\n")
    plain=yaml.safe_load(cfg.read_text());del plain['acquisition']['audit']
    plain_cfg=d/'plain.yaml';plain_cfg.write_text(yaml.safe_dump(plain))
    invoke([plain_cfg,'-i',2,'-o',d/'killed'],expected=1,wrapper=killer)
    killed=next((d/'killed').iterdir())
    assert json.loads((killed/'phases/acquisition_it_0.json').read_text())['status']=='incomplete'
    assert not (killed/'training_data/data_it_1.csv').exists()
    text=invoke([killed,'-i',1])
    assert 'Phase acquisition_it_0 did not finish' in text and 'Loading existing model' in text
    assert (killed/'training_data/data_it_1.csv').exists() and (killed/'trained_models/model_it_1.keras').exists()
    # Boosts with a region and the audit: a continued run reads the region its saved network
    # was audited in. Built again from a training set that holds that audit's points it could
    # differ, and the file is a committed one: the start after that would be refused.
    import hashlib
    boosted=yaml.safe_load((ROOT/'docs/site/examples/gaussian_client.yaml').read_text())
    boosted['likelihood']['input']=str(ROOT/'docs/site/examples/gaussian_likelihood.yaml')
    boosted['training']['n_epochs']=150
    boosted['acquisition']['boost']=dict(n=40,start_iteration=1,region=2.0,walkers=32,cool_steps=50,steps=200,components=1)
    boosted['acquisition']['audit']=dict(tau=.001,rounds=2,starts=16,steps=60,epochs=2,k=2)
    boosted_cfg=d/'boosted.yaml';boosted_cfg.write_text(yaml.safe_dump(boosted))
    invoke([boosted_cfg,'-i',4,'-o',d/'boosted'],timeout=600)
    boost_run=next((d/'boosted').iterdir())
    region=boost_run/'boost/region_it_3.npz'
    assert region.exists()
    before=hashlib.sha256(region.read_bytes()).hexdigest()
    for _ in range(2):
        text=invoke([boost_run,'-i',1],timeout=600)
        assert 'Region of the saved network' in text
        assert hashlib.sha256(region.read_bytes()).hexdigest()==before
    # The drift criterion belongs to the ordinary loop: met at a boost iteration (here at once,
    # with a threshold nothing misses) it used to end the run at its first boost.
    drifting=yaml.safe_load(boosted_cfg.read_text())
    del drifting['acquisition']['audit']
    drifting['acquisition']['boost'].pop('region')
    drifting['training']['n_epochs']=20
    drifting['convergence'].update(threshold=1e9,max_iterations=3)
    drifting_cfg=d/'drifting.yaml';drifting_cfg.write_text(yaml.safe_dump(drifting))
    text=invoke([drifting_cfg,'-o',d/'drifting'],timeout=600)
    assert 'Convergence criterion met' not in text and 'the boosts have begun: not a criterion' in text
    assert (next((d/'drifting').iterdir())/'trained_models/model_it_2.keras').exists()
    # -s N in a run that has gone further did iteration N again in place: every phase stayed
    # complete and the next start without -s went on from the earlier line's latest network,
    # without the points of the iteration done again. Refused now; --rewind sets the later
    # iterations aside first.
    invoke([plain_cfg,'-i',3,'-o',d/'rewound'])
    rewound=next((d/'rewound').iterdir())
    data=lambda k: (rewound/f'training_data/data_it_{k}.csv').read_bytes()
    assert (rewound/'trained_models/model_it_2.keras').exists()
    earlier={k:data(k) for k in (1,2)}
    refused=invoke([rewound,'-s',1,'-i',1],expected=1)
    assert 'has gone on to iteration 2' in refused and '--rewind' in refused, refused[-2000:]
    assert data(2)==earlier[2] and not (rewound/'interrupted').exists()
    text=invoke([rewound,'-s',1,'-i',1,'--rewind'])
    assert '--rewind:' in text and 'are set aside in interrupted/rewind_to_it_1-' in text, text[-3000:]
    (aside,)=(rewound/'interrupted').iterdir()
    assert (aside/'training_data/data_it_2.csv').read_bytes()==earlier[2] and (aside/'trained_models/model_it_2.keras').exists()
    assert (aside/'phases/acquisition_it_1.json').exists() and (aside/'REWIND.json').exists()
    assert data(1)==earlier[1] and data(2)!=earlier[2]                 # iteration 1 was done again, from the same network
    assert all(json.loads(f.read_text())['status']=='complete' for f in (rewound/'phases').glob('*.json'))
    text=invoke([rewound,'-i',1])                                      # and the next start goes on from the new line
    assert 'data_it_2.csv' in text and 'did not finish' not in text, text[-3000:]
    assert data(3).startswith(data(2).rstrip(b'\n'))
    # A job killed while rewinding has already moved some phase records, while later
    # networks/data may still exist. A plain continuation finishes that recorded operation
    # before choosing its starting network, instead of silently resuming the old line.
    before_rewind = data(3)
    killer.write_text("import sys, os\nsys.path.insert(0,"+repr(str(ROOT))+")\nimport run.storage\n"
                      "original=run.storage.shutil.move\ndef move(source,target):\n"
                      "    if '/trained_models/' in source: os._exit(137)\n"
                      "    return original(source,target)\nrun.storage.shutil.move=move\nimport client\nclient.main()\n")
    invoke([rewound,'-s',1,'--rewind','-i',1],expected=1,wrapper=killer)
    (pending,)=[p for p in rewound.glob('interrupted/rewind_to_it_*/REWIND.json')
                if json.loads(p.read_text()).get('status')=='pending']
    text=invoke([rewound,'-i',1])
    assert 'Finished interrupted rewind to iteration 1' in text and 'data_it_1.csv' in text, text[-3000:]
    assert not (rewound/'trained_models/model_it_3.keras').exists()
    assert (rewound/'trained_models/model_it_2.keras').exists()
    assert (pending.parent/'training_data/data_it_3.csv').read_bytes()==before_rewind
    assert json.loads(pending.read_text())['status']=='complete'
print('Pipeline recovery smoke passed',flush=True)
