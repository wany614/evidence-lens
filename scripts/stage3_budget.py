"""Measure the registered real pilot repeat and freeze a single source/step budget before production."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from evidence_lens._bundle import encoded,load,sha
from evidence_lens._verify import require
from make_handoff import publish
from stage3_reference import located,perform


def compare(a,b):
    require(a.dtype==b.dtype and a.shape==b.shape,'Repeat output grid differs')
    finite=np.isfinite(a)&np.isfinite(b)
    diff=np.abs(a[finite].astype(np.float64)-b[finite].astype(np.float64))
    return {'total':int(a.size),'paired_finite':int(finite.sum()),'validity_disagreements':int((np.isfinite(a)!=np.isfinite(b)).sum()),
            'bitwise_equal':a.tobytes()==b.tobytes(),'max_abs':float(diff.max()) if diff.size else None,'mean_abs':float(diff.mean()) if diff.size else None}


def analyse(spec,roots):
    samples=[]
    init=set();checkpoint=set();steps=set();pixels=set()
    for i,t in enumerate(spec['trials']):
        gen,pred=roots[i]
        r=load(gen/'reconstruction.json')
        m=load(pred/'prediction.json')
        require(r['run_id']==t['reconstruction']['run_id'] and m['run_id']==t['render']['run_id'] and m['reconstruction_run_id']==r['run_id']
            and m['depth_verified'] and sha(gen/'scene.npz')==r['checkpoint_sha256']==m['checkpoint_sha256'] and sha(gen/'initial.npz')==r['initial_checkpoint_sha256']
            and r['final_gaussian_count']==r['initial_gaussian_count']==4096 and len(m['views'])==len(r['generator_source_ids'])+6,'Pilot samples or common reconstruction differ')
        require(r['optimizer_sha256']==spec['optimizer_sha256'] and tuple(r['generator_source_ids'])==({'M2':('s00','s03'),'M4':('s00','s01','s02','s03')}[t['condition']])
            and all(s['samples']*len(r['generator_source_ids'])==r['gaussian_optimization_steps'] and
                    all(v>0 for v in s['gradient_nonzero_steps'].values()) for s in r['source_sampling'].values()),'Source contribution differs')
        init.add(r['initial_parameter_sha256']);checkpoint.add(r['initial_checkpoint_sha256']);steps.add(r['gaussian_optimization_steps']);pixels.add(r['sampled_source_pixels'])
        samples.append({**t,'camera_sha256':r['source_camera_sha256'],'source_rgb_sha256':r['source_rgb_sha256'],'steps':r['gaussian_optimization_steps'],
             'sampled_source_pixels':r['sampled_source_pixels'],'source_sampling':r['source_sampling'],'initial_checkpoint_sha256':r['initial_checkpoint_sha256'],
             'final_parameter_sha256':r['final_parameter_sha256'],'final_gaussian_count':r['final_gaussian_count'],'delta_l2':r['parameter_delta_l2'],
             'loss_initial':r['loss_initial_by_source'],'loss_final':r['loss_final_by_source'],'optimization_seconds':r['optimization_seconds'],'render_seconds':m['stage_seconds'],
             'optimization_torch_gpu_allocated_bytes':r['runtime']['torch_cuda_allocated_peak_bytes'],'optimization_torch_gpu_reserved_bytes':r['runtime']['torch_cuda_reserved_peak_bytes'],
             'render_torch_gpu_allocated_bytes':m['runtime']['torch_cuda_allocated_peak_bytes'],'device':r['runtime']['device']})
    require(len(init)==len(checkpoint)==len(steps)==len(pixels)==1 and steps=={64} and pixels=={64*256**2},'M2/M4 or repeat used different initialization or total budget')
    for fam in {s['family'] for s in samples}:
        filtered=[s for s in samples if s['family']==fam and s['phase']=='engineering-pilot']
        require(len(filtered)==2 and {s['condition'] for s in filtered}=={'M2','M4'} and
                all(filtered[0]['source_rgb_sha256'][s]==filtered[1]['source_rgb_sha256'][s] for s in ('s00','s03')),
                'Paired source endpoints differ')
    first=next(i for i,s in enumerate(samples) if s['condition']=='M2' and s['phase']=='engineering-pilot')
    repeat=next(i for i,s in enumerate(samples) if s['phase']=='engineering-repeat')
    require(samples[first]['family']==samples[repeat]['family'],'Repeat family differs')
    ga,ra=roots[first];gb,rb=roots[repeat]
    with np.load(ga/'scene.npz',allow_pickle=False) as p,np.load(gb/'scene.npz',allow_pickle=False) as q:
        delta={k:compare(p[k],q[k]) for k in p.files}
    views={}
    for v in load(ra/'prediction.json')['views']:
        with np.load(ra/(v+'.npz'),allow_pickle=False) as p,np.load(rb/(v+'.npz'),allow_pickle=False) as q:
            views[v]={k:compare(p[k],q[k]) for k in ('rgb','z','alpha')}
    return {'samples':samples,'common_initial_parameter_sha256':next(iter(init)),'common_initial_checkpoint_sha256':next(iter(checkpoint)),
            'engineering_repeat':{'original':samples[first]['reconstruction'],'repeat':samples[repeat]['reconstruction'],'parameter_comparison':delta,
              'views':views,'interpretation':'One independent same-input training-family repeat measures numerical variability under this CUDA software/hardware and schedule.'}}


def work(a):
    spec=load(a.request)
    roots=list(zip(a.generation,a.prediction,strict=True))
    evidence=analyse(spec,roots)
    b=load(a.optimizer)
    require(sha(a.optimizer)==spec['optimizer_sha256'] and spec['steps'] in b['optimization']['production_candidate_updates'] and
            len(evidence['samples'])==5,'Final production budget not registered')
    proof=load(a.canary)
    require(proof['passed'] and proof['batch_sha256']==spec['batch_sha256'],'Missing actual input-boundary canary')
    out={'schema':'el-stage3-production-budget-1','run_id':os.environ['EL_RUN_ID'],'catalog_sha256':spec['catalog_sha256'],
         'batch_sha256':spec['batch_sha256'],'worker_sha256':spec['worker_sha256'],'optimizer_sha256':sha(a.optimizer),
         'optimization_steps':spec['steps'],'pixels_per_condition':spec['steps']*256**2,'gaussian_count':4096,'condition_settings_identical':True,
         'init_sha256':evidence['common_initial_parameter_sha256'],'pilot_evidence':evidence,'scientific_production_restarts_from_initial':True,
         'pilot_samples_in_scientific_dataset':0,'decision':spec['reason'],'input_boundary_run':spec['canary'],'reference_registration':spec['registration']}
    (a.out/'production-budget.json').write_bytes(encoded(out))


def entry(a):
    store,project,assets=a.store.resolve(),a.project.resolve(),a.assets.resolve()
    packet=load(a.batch);base=store/'stage3'/packet['catalog_sha256'];dst=base/'production-budget.json'
    if dst.exists() and a.action=='record':
        old=load(dst);require(load(located(store,old['budget'])/'payload/production-budget.json')['optimization_steps']==a.updates,'Immutable production budget differs')
        return old
    boundary=load(a.batch.parent/'source-boundary-audit.json')
    canary=boundary['run'];registration=load(base/'registration.json')['registration']
    budget=located(store,registration)/'payload/optimizer.json'
    roles={'optimizer':(budget,'registration'),'canary':(located(store,canary)/'payload/canary.json','canary')}
    parents={'registration':registration,'canary':canary}
    req={'catalog_sha256':packet['catalog_sha256'],'batch_sha256':sha(a.batch),'worker_sha256':packet['worker_sha256'],'steps':a.updates,
         'reason':a.reason,'registration':registration,'canary':canary,'optimizer_sha256':sha(budget),'trials':[]}
    args=['{python}','{source}/scripts/stage3_budget.py','work','--request','{input:request}','--optimizer','{input:optimizer}','--canary','{input:canary}','--out','{out}']
    roots=[]
    for i,job in enumerate(packet['jobs']):
        p=base/'engineering'/job['trial_id']/'selection.json'
        chosen=load(p);r=chosen['selected']
        require(chosen['selection_status']=='selected' and chosen['job_sha256']==job['job_sha256'] and chosen['phase']==job['phase'],'Wrong pilot selection')
        req['trials'].append({**{k:job[k] for k in ('family','trial_id','phase','condition')},'selection_sha256':sha(p),'reconstruction':r['reconstruction'],'render':r['render']})
        paths=[]
        for role,key in [('generation','reconstruction'),('prediction','render')]:
            tag=role+str(i)
            parents[tag]=r[key]
            payload=located(store,r[key])/'payload';roles[tag]=(payload,tag); paths.append(payload)
            args+=['--'+role,'{input:'+tag+'}']
        roots.append(tuple(paths))
    if a.action=='inspect':return analyse(req,roots)
    with tempfile.TemporaryDirectory(dir=store/'work',prefix='stage3-pilot-budget-') as tmp:
        src=Path(tmp)/'engine';(src/'scripts').mkdir(parents=True)
        for root,n in [(assets,'stage3_budget.py'),(assets,'stage3_reference.py'),(project,'make_handoff.py'),(project,'formal_lineage.py')]:
            shutil.copyfile(root/'scripts'/n,src/'scripts'/n)
        record=Path(tmp)/'request.json';record.write_bytes(encoded(req));roles['request']=(record,None)
        cfg={'experiment_id':'stage3-production-budget','trial_id':'joint-production-budget-v1','dataset_hash':req['catalog_sha256'],'split':'train',
             'production_steps':a.updates,'recorded_input_sha256':sha(record),'recording':{'expected_output_bytes':190000}}
        pin=perform(store,project,src,cfg,'report',roles,parents,args)
        proof=located(store,pin)/'payload/production-budget.json'
        pointer={'schema':'el-stage3-production-budget-selection-1','catalog_sha256':req['catalog_sha256'],'worker_sha256':req['worker_sha256'],
                 'budget':pin,'member':'payload/production-budget.json','budget_sha256':sha(proof),'optimization_steps':a.updates}
        publish(dst,encoded(pointer))
        return pointer


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('inspect','record','work'))
    for n in ('store','project','assets','batch','request','optimizer','canary','out'):p.add_argument('--'+n,type=Path)
    for n in ('generation','prediction'):p.add_argument('--'+n,type=Path,action='append',default=[])
    p.add_argument('--updates',type=int);p.add_argument('--reason')
    a=p.parse_args()
    if a.action=='record':require(a.updates in (64,128,256) and a.reason,'A recorded choice and reason are required')
    if a.action=='work':work(a)
    else:print(json.dumps(entry(a),sort_keys=True))
