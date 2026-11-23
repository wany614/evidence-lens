"""Execute one SHA-pinned original multiview batch on the registered Windows CUDA worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

sys.dont_write_bytecode=True
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'src'))
from evidence_lens._bundle import encoded, files, load, local_path, sha
from evidence_lens._verify import require, verify_bundle
from worker_jobs import dispatch, frozen_json


SETS={'M2':['s00','s03'],'M4':['s00','s01','s02','s03']}


def exact(path,meta):
    require(path.is_file() and path.stat().st_size==meta['bytes'] and sha(path)==meta['sha256'],'Worker member hash differs: '+path.name)


def job_load(z, dest, entry):
    exact(z,entry)
    with zipfile.ZipFile(z) as archive:
        names=archive.namelist()
        require(len(names)==len(set(names)) and 'job.json' in names,'Repeated or missing job member')
        jobbytes=archive.read('job.json')
        require(hashlib.sha256(jobbytes).hexdigest()==entry['job_sha256'],'Job registration changed')
        job=json.loads(jobbytes)
        expected={'job.json','optimization.json','source_cameras.json','render_cameras.json',*('sources/'+s+'.png' for s in SETS[job['condition']])}
        require(job['schema']=='el-stage3-worker-job-1' and job['trial_id']==entry['trial_id'] and job['generator_source_ids']==SETS[job['condition']]
                and set(names)==expected and set(job['files'])==expected-{'job.json'},'Undeclared source image, target truth, or job member')
        for name in sorted(names):
            data=archive.read(name)
            if name!='job.json':
                x=job['files'][name]
                require(len(data)==x['bytes'] and hashlib.sha256(data).hexdigest()==x['sha256'],'Input manifest mismatch')
            target=dest/local_path(name)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
    return job


def launch(store,returns,job,stage,folder,job_sha,producer,retry=None,resume=None):
    ref=job['reference'];reg=job['registration'];rb=job['readback']
    parents=[ref,rb,reg]
    if job.get('production_budget'):parents.append(job['production_budget'])
    if job.get('opening_method'):parents.append(job['opening_method'])
    kind='generation' if stage=='optimize' else 'render'
    root=store/'stage3-control'/job['identity']['dataset_hash']/job['trial_id']
    root.mkdir(parents=True,exist_ok=True)
    config={**job['identity'],'experiment_id':job['experiment_id'],'trial_id':job['trial_id'],'phase':job['phase'],
            'formal_handoff_sha256':job_sha,'registration':reg,'upstream_reference':ref,'independent_readback':rb,
            'input_worker_sha256':job['worker_sha256'],'generator_source_ids':job['generator_source_ids'],
            'source_rgb_sha256':{s:job['files']['sources/'+s+'.png']['sha256'] for s in job['generator_source_ids']},
            'source_camera_sha256':job['files']['source_cameras.json']['sha256'],'optimizer':load(folder/'optimization.json'),
            'optimizer_sha256':job['files']['optimization.json']['sha256'],'optimization_steps':job['optimization_steps'],
            'production_budget':job.get('production_budget'),'opening_method':job.get('opening_method'),'recording':{'expected_output_bytes':19000000}}
    roles={'optimizer':(folder/'optimization.json',reg)}
    cmd=['{python}','{source}/src/evidence_lens/stage3_gaussians.py',stage,'--config','{config}','--out','{out}','--cameras','{input:cameras}']
    if stage=='optimize':
        roles['cameras']=(folder/'source_cameras.json',ref)
        for s in job['generator_source_ids']:
            roles[s]=(folder/'sources'/(s+'.png'),ref)
            cmd+=['--source',s,'{input:'+s+'}']
    else:
        require(producer and producer['status']=='succeeded','Render needs a successful joint reconstruction')
        pp={'run_id':producer['run'],'manifest_sha256':producer['manifest_sha256']}
        parents.append(pp)
        payload=store/'runs'/pp['run_id']/'payload'
        meta=load(payload/'reconstruction.json')
        require(meta['run_id']==pp['run_id'] and meta['source_rgb_sha256']==config['source_rgb_sha256'] and
                meta['checkpoint_sha256']==sha(payload/'scene.npz'),'Rendered scene does not belong to this condition')
        config.update(reconstruction_run_id=pp['run_id'],reconstruction_manifest_sha256=pp['manifest_sha256'],render_camera_sha256=job['files']['render_cameras.json']['sha256'])
        roles.update(cameras=(folder/'render_cameras.json',ref),checkpoint=(payload/'scene.npz',pp),reconstruction=(payload/'reconstruction.json',pp))
        cmd+=['--checkpoint','{input:checkpoint}','--reconstruction','{input:reconstruction}']
    cfg=root/(kind+'-config.json')
    frozen_json(cfg,config)
    def execute():
        args=[sys.executable,'-m','evidence_lens.recorder','--root',str(store),'--source-root',str(CODE),'--storage-policy',str(CODE/'configs/storage.json'),'--kind',kind,'--config',str(cfg)]
        for p in parents:
            args+=['--parent',p['run_id']+'='+p['manifest_sha256']]
        for role,(path,owner) in roles.items():
            args+=['--input',role+'='+str(path),'--input-parent',role+'='+owner['run_id']]
        if retry: args+=['--retry-of',retry]
        proc=subprocess.run([*args,'--',*cmd],capture_output=True,text=True,env={**os.environ,'PYTHONPATH':str(CODE/'src'),'PYTHONDONTWRITEBYTECODE':'1'})
        if proc.stderr: print(proc.stderr,file=sys.stderr,flush=True)
        require(proc.stdout.strip(),'Recorder did not issue a run receipt')
        return json.loads(proc.stdout.splitlines()[-1])
    # Reuse the original reserve/recover/pack recorder controller. Failed runs are returned intact.
    return dispatch(store,returns,job,job_sha,kind,producer['run'] if producer else None,retry,resume,execute)


def run(a):
    require(os.name=='nt','Scientific execution requires Windows')
    base,store,returns=CODE.parent,a.store.resolve(),a.returns.resolve()
    require((store.drive and store.drive.casefold()==Path(sys.executable).drive.casefold()) and base.drive.casefold()==store.drive.casefold(),
            'Stage extraction and Python/worker store must be on the registered research volume')
    req=base/'batch.json'
    require(sha(req)==a.batch_sha256,'Do not execute an unregistered batch')
    batch=load(req)
    actual=files(CODE)
    require(set(actual)==set(batch['worker_files']),'Unexpected worker member')
    for k,r in batch['worker_files'].items():exact(actual[k],r)
    store.mkdir(parents=True,exist_ok=True)
    returns.mkdir(parents=True,exist_ok=True)
    created=store/'stage3-job-cache';created.mkdir(exist_ok=True)
    # Keep the verified, previously used CUDA extension cache available via the caller's environment.
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE/'src'))
    summary=[]
    for entry in batch['jobs']:
        if a.trial and a.trial!=entry['trial_id']:continue
        with tempfile.TemporaryDirectory(prefix='stage3-',dir=created) as tmp:
            folder=Path(tmp)
            j=job_load(base/entry['path'],folder,entry)
            require(j['worker_sha256']==batch['worker_sha256'] and j['identity']['dataset_hash']==batch['catalog_sha256'],'Different protocol/worker')
            produced=None
            records=[]
            for step in ('optimize','render'):
                retry=a.retry if a.trial and step==a.stage else None
                resume=a.resume_preflight if a.trial and step==a.stage else None
                returned=launch(store,returns,j,step,folder,entry['job_sha256'],produced,retry,resume)
                records.append(returned)
                print(json.dumps({'trial':j['trial_id'],'stage':step,'return':returned},sort_keys=True),flush=True)
                if returned['status']!='succeeded':break
                produced=returned
            summary.append({'trial_id':j['trial_id'],'job_sha256':entry['job_sha256'],'runs':records})
    require(summary,'No registered job matched')
    primary=returns/(batch['batch']+'-worker-summary.json')
    receipt={'schema':'el-stage3-worker-completion-1','batch':batch['batch'],'batch_sha256':a.batch_sha256,'jobs':summary}
    if not a.trial and not primary.exists():frozen_json(primary,receipt)
    else:frozen_json(returns/(batch['batch']+'-'+hashlib.sha256(encoded(receipt)).hexdigest()[:16]+'.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('store','returns'):p.add_argument('--'+n,required=True,type=Path)
    p.add_argument('--batch-sha256',required=True);p.add_argument('--trial');p.add_argument('--stage',choices=('optimize','render'))
    p.add_argument('--retry');p.add_argument('--resume-preflight')
    a=p.parse_args()
    if (a.retry or a.resume_preflight) and not(a.trial and a.stage):p.error('Retry requires a single exact trial and stage')
    run(a)
