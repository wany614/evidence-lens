"""Record a fresh Mac numerical readback of a named C acquisition batch and its pinned native parents."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from evidence_lens._bundle import encoded,load,sha
from evidence_lens._verify import require,verify_bundle
from make_handoff import publish,sha_bytes
from stage3_intake import SETS,authenticate_reconstruction,authenticate_render,lineage
from stage3_postreview import audit
from stage3_reference import family_info,located,perform


def gate(store,catalog,family,split):
    if split=='test':
        from evidence_lens.stage3_gate import require_opening
        require_opening(store,catalog,family=family,shelf=store.parent/'cold-shelf')


def work(a):
    spec=load(a.inputs/'request.json'); packet=load(a.batch)
    require(spec['batch_sha256']==sha(a.batch) and spec['catalog_sha256']==packet['catalog_sha256'],'Batch digest changed')
    mod=Path(__file__).resolve().parents[1]/'src/evidence_lens/stage3_gaussians.py'
    require(sha(mod)==packet['worker_files']['src/evidence_lens/stage3_gaussians.py']['sha256'],'Projection source changed')
    loader=importlib.util.spec_from_file_location('actual_joint_model',mod)
    gauss=importlib.util.module_from_spec(loader);loader.loader.exec_module(gauss)
    results={}
    for row in packet['jobs']:
        tid=row['trial_id']
        sel=spec['trials'][tid]
        gate(a.store,packet['catalog_sha256'],row['family'],sel['split'])
        folder=a.inputs/tid
        sp=folder/'selection.json';ap=folder/'postreview-input-audit.json';jp=folder/'job.json'
        require(sha(sp)==sel['selection_sha256'] and sha(ap)==sel['audit_sha256'] and sha(jp)==row['job_sha256'],
                'A selected job, numerical readback or explicit parent differs')
        q=load(sp); j=load(jp); old=load(ap); pins=q['selected']
        require(q['batch_sha256']==spec['batch_sha256'] and q['trial_id']==j['trial_id']==tid and
                q['phase']==row['phase'] and q['condition']==row['condition'] and q['family']==row['family'] and
                pins['reference']==j['reference'] and pins['readback']==j['readback'] and pins['registration']==j['registration'] and
                q['production_budget']==j.get('production_budget'),'Original scientific selection differs from its source job')
        for x in q['attempts']:
            ref={'run_id':x['run_id'],'manifest_sha256':x['manifest_sha256']}
            returned={'run':x['run_id'],'manifest_sha256':x['manifest_sha256'],'status':x['status']}
            native=located(a.store,ref)
            lineage(native,j,row['job_sha256'],returned,SETS[q['condition']],pins.get('reconstruction') if x['stage']=='render' else None)
            if x['status']!='succeeded':
                for name in ('manifest.json','run.json','completion.json','stderr.log'):
                    require(sha(native/name)==sha(folder/'attempts'/x['run_id']/name),'Recorded failed-attempt evidence changed')
        if 'reconstruction' in pins:
            rec=located(a.store,pins['reconstruction'])
            gen=authenticate_reconstruction(rec,j,gauss)
            require(gen==q['numerical_readback']['reconstruction'],'Archived gradients/step budget/parameters changed')
        if 'render' in pins:
            render=authenticate_render(located(a.store,pins['render']),j,pins['reconstruction'],rec,gen)
            require(render==q['numerical_readback']['render'],'Returned depth/opacity/RGB arrays differ from original intake')
        fresh=audit(a.store,q,j,sha(sp),gauss)
        require(encoded(fresh)==encoded(old),'Independent numerical readback differs from condition proof')
        results[tid]={'selection_sha256':sha(sp),'postreview_sha256':sha(ap),'status':q['selection_status'],'phase':q['phase'],
                       'source_ids':list(SETS[q['condition']]),'parents':pins,'failure':q.get('failure'),'measurements':fresh['measurements']}
    require(len(results)==len(spec['trials'])==len(packet['jobs']),'Incomplete batch numerical readback')
    payload={'schema':'el-stage3-batch-postreview-1','run_id':os.environ['EL_RUN_ID'],'passed':True,'catalog_sha256':spec['catalog_sha256'],
             'batch_sha256':sha(a.batch),'batch':packet['batch'],'audit_code_sha256':sha(Path(__file__).with_name('stage3_postreview.py')),
             'evidence':'Selected original source PNGs, calibrated poses, initial and optimized tensors, gradients, checkpoint and all numerical renders.',
             'results':results}
    (a.out/'postreview.json').write_bytes(encoded(payload))


def record(a):
    store,assets,project=a.store.resolve(),a.assets.resolve(),a.project.resolve()
    require(sha(a.batch)==a.batch_sha256,'Wrong delivery batch')
    packet=load(a.batch);cat=packet['catalog_sha256'];batch=packet['batch'];base=store/'stage3'/cat
    marker=base/'batches'/batch/'postreview-archive.json'
    with marker.with_suffix('.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if marker.exists():
            saved=load(marker);sealed=located(store,saved['run'])
            require(saved['batch_sha256']==a.batch_sha256 and sha(sealed/'payload/postreview.json')==saved['postreview_sha256'] and
                    verify_bundle(sealed)['run']['execution_status']=='succeeded','Batch proof differs')
            return saved
        with tempfile.TemporaryDirectory(dir=store/'work',prefix='stage3-intake-proof-') as tmp:
            temp=Path(tmp);src=temp/'engine';inputs=temp/'selection-inputs'
            (src/'scripts').mkdir(parents=True);inputs.mkdir()
            names=('stage3_intake_audit.py','stage3_intake.py','stage3_postreview.py','stage3_reference.py')
            for name in names:shutil.copyfile(assets/'scripts'/name,src/'scripts'/name)
            for name in ('make_handoff.py','formal_lineage.py'):
                shutil.copyfile(project/'scripts'/name,src/'scripts'/name)
            worker='src/evidence_lens/stage3_gaussians.py'
            (src/worker).parent.mkdir(parents=True);shutil.copyfile(assets/worker,src/worker)
            req={'batch_sha256':a.batch_sha256,'catalog_sha256':cat,'trials':{}}
            parents={}
            for row in packet['jobs']:
                _,rowcat,info=family_info(store,assets,row['family'])
                require(rowcat==cat,'Source catalog changed')
                t=row['trial_id'];dest=inputs/t;dest.mkdir()
                original=base/('trials' if row['phase']=='production' else 'engineering')/t
                for name in ('selection.json','postreview-input-audit.json'):shutil.copyfile(original/name,dest/name)
                jobfile=a.batch.parent/row['path'];require(sha(jobfile)==row['sha256'],'Job ZIP changed')
                with zipfile.ZipFile(jobfile) as z:
                    jobdata=z.read('job.json');require(sha_bytes(jobdata)==row['job_sha256'],'Job SHA changed')
                    (dest/'job.json').write_bytes(jobdata)
                j=json.loads(jobdata);q=load(dest/'selection.json')
                pins=list(q['selected'].values())
                for attempt in q['attempts']:
                    pin={k:attempt[k] for k in ('run_id','manifest_sha256')}
                    if attempt['status']=='succeeded':
                        pins.append(pin)
                        continue
                    # The recorder only accepts successful batches as scientific parents. Preserve the failed attempt's actual
                    # readback evidence as inputs and recheck every declared native file against its original signed manifest.
                    failed=located(store,pin);saved_failed=dest/'attempts'/attempt['run_id'];saved_failed.mkdir(parents=True)
                    for name in ('manifest.json','run.json','completion.json','stderr.log'):
                        shutil.copyfile(failed/name,saved_failed/name)
                pins.extend(j[k] for k in ('production_budget','opening_method') if j.get(k))
                for pin in pins:
                    rid=pin['run_id']
                    require(rid not in parents or parents[rid]==pin,'Two manifests claimed for one science run')
                    parents[rid]=pin
                req['trials'][t]={'selection_sha256':sha(dest/'selection.json'),'audit_sha256':sha(dest/'postreview-input-audit.json'),'split':info['split']}
            (inputs/'request.json').write_bytes(encoded(req))
            cfg={'experiment_id':'stage3-postreview-input-readback','trial_id':batch+'-postreview','dataset_hash':cat,
                 'split':next(iter(req['trials'].values()))['split'],'batch_sha256':a.batch_sha256,'input_inventory_sha256':sha(inputs/'request.json'),
                 'recording':{'expected_output_bytes':240000}}
            roles={'batch':(a.batch,None),'inputs':(inputs,None)}
            command=['{python}','{source}/scripts/stage3_intake_audit.py','work','--batch','{input:batch}','--inputs','{input:inputs}',
                     '--out','{out}','--store',str(store)]
            found=perform(store,project,src,cfg,'archive_verify',roles,parents,command)
        proof=located(store,found)/'payload/postreview.json'
        saved={'schema':'el-stage3-postreview-archive-selection-1','batch':batch,'batch_sha256':a.batch_sha256,'catalog_sha256':cat,'run':found,
               'postreview_sha256':sha(proof),'trials':len(req['trials'])}
        publish(marker,encoded(saved))
        return saved


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('work','record'))
    for n in ('store','assets','project','batch','inputs','out'):p.add_argument('--'+n,type=Path)
    p.add_argument('--batch-sha256');args=p.parse_args()
    if args.action=='work':work(args)
    else:print(json.dumps(record(args),sort_keys=True))
