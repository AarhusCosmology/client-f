"""Stage the guides and generate configuration tables without scientific imports."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

API_FILES = ['config/config.py', 'profiles/config.py', 'likelihood/base.py',
             'likelihood/surrogate.py', 'run/run.py', 'dataset/boost.py',
             'dataset/posterior_boost.py', 'dataset/augment.py', 'sampling/modes.py',
             'sampling/region.py', 'training/audit.py', 'profiles/workflow.py',
             'profiles/residual_newton.py', 'profiles/hybrid_surface.py',
             'profiles/region.py', 'utils/held_out.py', 'convergence/convergence.py']
LINK = re.compile(r'(!?\[[^\]\n]*\]\()([^\s)]+)(\))')


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def configuration(root):
    """AST-derived field/type/default inventory; parser defaults override constructors."""
    sections = {'LikelihoodConfig':'likelihood', 'PriorConfig':'prior',
                'AcquisitionConfig':'acquisition', 'AuditConfig':'acquisition.audit',
                'BoostConfig':'acquisition.boost', 'ModelConfig':'model',
                'TrainingConfig':'training', 'SamplingConfig':'sampling',
                'DerivedConfig':'derived', 'ConvergenceConfig':'convergence',
                'Config':'top level', 'ProfileConfig':'profile YAML'}
    result = []
    for rel in ['config/config.py', 'profiles/config.py']:
        source = (root / rel).read_text()
        lines = source.splitlines()
        for cls in ast.parse(source).body:
            if not isinstance(cls, ast.ClassDef) or cls.name not in sections:
                continue
            method = next((m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name=='from_dict'), None)
            gets, required = {}, set()
            if method:
                for n in ast.walk(method):
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id=='d' and n.func.attr=='get' and n.args and isinstance(n.args[0],ast.Constant):
                        gets[n.args[0].value] = ast.unparse(n.args[1]) if len(n.args)>1 else 'None'
                    if isinstance(n, ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='d' and isinstance(n.slice,ast.Constant):
                        required.add(n.slice.value)
            fields=[]
            constants={t.id: ast.unparse(n.value) for n in cls.body if isinstance(n, ast.Assign)
                       for t in n.targets if isinstance(t, ast.Name)}
            for n in cls.body:
                if not isinstance(n,ast.AnnAssign) or not isinstance(n.target,ast.Name): continue
                key=n.target.id
                default=gets.get(key, 'required' if n.value is None else ast.unparse(n.value))
                # Keys nested through conditional d[...] calls are optional.
                if key in ('audit','boost','derived') and cls.name in ('Config','AcquisitionConfig'): default='None (section absent)'
                if cls.name == 'ConvergenceConfig' and key == 'threshold':
                    default=constants['TEMPERED_THRESHOLD'] + ' (tempered_batch_scatter only; otherwise required)'
                if default.startswith('field('): default='[]' if 'list' in default else '{}'
                description=lines[n.lineno-1].split('#',1)[1].strip() if '#' in lines[n.lineno-1] else ''
                fields.append(dict(name=key, type=ast.unparse(n.annotation), default=default, note=description, line=n.lineno))
            result.append(dict(section=sections[cls.name], cls=cls.name, source=rel, fields=fields))
    return result


def prepare(root, site):
    try:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        revision=os.environ.get('CLIENT_DOCS_REVISION','main')
    repo='https://github.com/AarhusCosmology/client/blob/'+revision+'/'
    generated=site/'_generated'
    if generated.exists(): shutil.rmtree(generated)
    selected={p.resolve() for p in (root/'docs').glob('*.md')}
    selected.update((root/n).resolve() for n in ['README.md','UPGRADING.md','CHANGES.md','PERFORMANCE.md','AGENTS.md','hpc/README.md'])
    # Bring linked Markdown reports into the same searchable manual.
    pending=list(selected)
    while pending:
        p=pending.pop()
        for m in LINK.finditer(p.read_text()):
            target=m[2].split('#')[0]
            if ':' in target or not target: continue
            q=(p.parent/target).resolve()
            if q.is_relative_to(root) and q.suffix=='.md' and q.is_file() and q not in selected and not q.is_relative_to(site):
                selected.add(q); pending.append(q)
    mapping={p:generated/'library'/p.relative_to(root) for p in selected}
    hashes={}
    for p,dest in sorted(mapping.items()):
        raw=p.read_text(); hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
        def replace(m):
            target=m[2]; name,sep,anchor=target.partition('#')
            if ':' in name or not name or name.startswith('//'): return m[0]
            q=(p.parent/name).resolve()
            if not q.is_relative_to(root): return m[0]
            if q in mapping:
                uri=os.path.relpath(mapping[q],dest.parent).replace(os.sep,'/')
            elif q == site/'README.md':
                uri=os.path.relpath(site/'maintenance.md',dest.parent).replace(os.sep,'/')
            elif q.is_file() and q.suffix.lower() in ('.png','.jpg','.jpeg','.svg','.pdf'):
                asset=generated/'assets'/q.relative_to(root)
                asset.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(q,asset)
                uri=os.path.relpath(asset,dest.parent).replace(os.sep,'/')
            else:
                uri=repo+str(q.relative_to(root))
            return m[1]+uri+(sep+anchor if sep else '')+m[3]
        text=LINK.sub(replace,raw)
        # Preserve reports verbatim below a provenance note. No empirical claim is updated.
        note='```{note}\nMaintained source: ['+str(p.relative_to(root))+']('+repo+str(p.relative_to(root))+'). Measurements keep their original settings; consult the [current defaults]('+os.path.relpath(site/'configuration.md',dest.parent)+') before starting a new run.\n```\n\n'
        first_heading = re.search(r'^#+ ', text, re.M)
        title = ('# ' + {'CHANGES.md':'What this version adds', 'UPGRADING.md':'Migration notes'}.get(p.name,p.stem) + '\n\n') if first_heading and first_heading.group().startswith('##') else ''
        write(dest,title+note+text)
    library=['# Guides and reports','', 'These pages are built from the repository originals. Reports describe their measured configurations; they are not automatically new defaults.','', '```{toctree}', ':maxdepth: 1', '']
    for p in sorted(selected): library.append(str(mapping[p].relative_to(site)).removesuffix('.md'))
    library+=['```',''];write(site/'library.md','\n'.join(library))
    schema=configuration(root)
    config=['# Configuration fields','', 'Generated from the source at build time. `required` means the parser requires a value. `None` corresponds to YAML `null`; absence of an optional section disables it. Values below are defaults when the field is omitted, not recommended budgets for every problem. See [configuration conventions](../../configuration.md) for activation and validation rules.','']
    for entry in schema:
        config += ['## '+entry['section'],'', '[Source: '+entry['cls']+']('+repo+entry['source']+')','', '| Setting | Type | Omitted-field default | Inline source note |','|---|---|---|---|']
        for f in entry['fields']:
            values=[f['name'],f['type'],f['default']]
            values=[s.replace('|','\\|') for s in values]
            config.append('| '+' | '.join('`'+s+'`' for s in values)+' | '+f['note'].replace('|','\\|')+' |')
        config.append('')
    write(generated/'reference/configuration.md','\n'.join(config))
    write(generated/'reference/configuration.json',json.dumps(schema,indent=2)+'\n')
    for rel in API_FILES:
        p=generated/'api_source'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/rel,p)
        init=p.parent/'__init__.py'
        if not init.exists(): write(init, '"""Selected interfaces in ' + p.parent.name + '."""\n')
    write(generated/'provenance.json',json.dumps({'revision':revision,'guide_sha256':hashes,'api_sources':API_FILES},indent=2)+'\n')
    return revision
