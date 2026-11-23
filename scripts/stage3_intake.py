"""Verify actual C Windows returns and materialize signed selections without trusting an arrival order."""
import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from evidence_lens._bundle import encoded,load,sha
from evidence_lens._verify import require,verify_bundle
from evidence_lens.archive import receive
from make_handoff import publish
from stage3_postreview import audit as postreview_audit
from stage3_reference import family_info,located


SETS={'M2':('s00','s03'),'M4':('s00','s01','s02','s03')}
TARGETS=tuple(f'v{i:02d}' for i in range(1,7))
PARAMS=('means','log_scales','quats','opacity_logits','color_logits')


def file_ref(owner,root,name):
    file=root/name
    return {**owner,'path':name,'sha256':sha(file),'bytes':file.stat().st_size}


def check_native_selection(assets,store,job):
    fam=job['identity']['scene_family_id']; c=job['condition']
    catalog,cat,row=family_info(store,assets,fam)
    base=store/'stage3'/cat
    official=load(base/'references'/(fam+'.json'))
    require(all(job[k]==official[k] for k in ('readback','registration','reference')) and cat==job['identity']['dataset_hash']
            and row['recipe_sha256']==job['identity']['scene_hash'],'Windows job belongs to another family')
    root=located(store,official['reference']);rb=located(store,official['readback'])/'payload/readback.json'
    require(load(rb)['passed'] and sha(rb)==official['readback_sha256'],'Independent native readback changed')
    ref=job['reference']
    sources={s:file_ref(ref,root,'payload/sources/'+s+'.png') for s in SETS[c]}
    sc=file_ref(ref,root,'payload/condition/'+c+'/source_cameras.json')
    rc=file_ref(ref,root,'payload/condition/'+c+'/render_cameras.json')
    full=file_ref(ref,root,'payload/cameras.json')
    require(all(x['sha256']==job['files']['sources/'+s+'.png']['sha256'] for s,x in sources.items()) and
            sc['sha256']==job['files']['source_cameras.json']['sha256'] and rc['sha256']==job['files']['render_cameras.json']['sha256'] and
            full['sha256']==official['camera_sha256'],'Job consumed different sources or calibration')
    return base,{'family':fam,'scene_family_id':fam,'scene_id':row['scene_id'],'split':row['split'],'condition':c,'condition_id':c,'trial_id':job['trial_id'],
        'phase':job['phase'],'catalog_sha256':cat,'generator_source_ids':list(SETS[c]),'target_ids':list(TARGETS),
        'source_images':sources,'cameras':full,'source_cameras':sc,'render_cameras':rc,
        'selected':{k:official[k] for k in ('registration','reference','readback')}}


def expected_parents(job):
    out=[job[x] for x in ('reference','readback','registration')]
    return out+[job[x] for x in ('production_budget','opening_method') if job.get(x)]


def lineage(root,job,jsha,returned,ids,producer=None):
    checked=verify_bundle(root)
    m=checked['run']; kind='generation' if producer is None else 'render'
    require(m['run_id']==returned['run'] and checked['manifest_sha256']==returned['manifest_sha256'] and
        m['execution_status']==returned['status'] and m['trial_id']==job['trial_id'] and m['run_kind']==kind and
        m['experiment_id']==job['experiment_id'] and all(m[k]==job['identity'][k] for k in ('scene_id','scene_family_id','scene_hash','camera_set_hash','dataset_hash','condition_id','split')),
        'Returned seal identity differs from original job')
    cfg=load(root/'config.resolved.yaml')
    parents=expected_parents(job)+([producer] if producer else [])
    require(checked['inputs']['parents']==parents and cfg['formal_handoff_sha256']==jsha and cfg['input_worker_sha256']==job['worker_sha256'] and
            cfg['optimization_steps']==job['optimization_steps'] and cfg['source_rgb_sha256']=={s:job['files']['sources/'+s+'.png']['sha256'] for s in ids},
            'Wrong Windows scientific inputs or parent chain')
    if m['execution_status']!='succeeded':
        return m
    roles={v['role']:v for v in checked['inputs']['inputs']}
    require(set(roles)==({'optimizer','cameras','checkpoint','reconstruction'} if producer else {'optimizer','cameras',*ids}),'Information boundary differs in sealed recorder inputs')
    def match(role,owner,digest):
        v=roles[role]
        require(v['parent_run_id']==owner['run_id'] and len(v['files'])==1 and v['files'][0]['sha256']==digest,'Changed input '+role)
    match('optimizer',job['registration'],job['files']['optimization.json']['sha256'])
    match('cameras',job['reference'],job['files']['render_cameras.json' if producer else 'source_cameras.json']['sha256'])
    if not producer:
        for s in ids:match(s,job['reference'],job['files']['sources/'+s+'.png']['sha256'])
    else:
        p=load(root/'payload/prediction.json')
        match('checkpoint',producer,p['checkpoint_sha256']);match('reconstruction',producer,p['reconstruction_sha256'])
    return m


def authenticate_reconstruction(root,job,gauss):
    payload=root/'payload'
    m=load(payload/'reconstruction.json')
    ids=SETS[job['condition']]
    budget=load(root/'config.resolved.yaml')['optimizer']
    require(m['schema']=='el-stage3-reconstruction-1' and m['run_id']==root.name and m['backend_version']=='1.5.3' and
            m['source_camera_sha256']==job['files']['source_cameras.json']['sha256'] and m['source_rgb_sha256']=={s:job['files']['sources/'+s+'.png']['sha256'] for s in ids} and
            m['checkpoint_sha256']==sha(payload/'scene.npz') and m['initial_checkpoint_sha256']==sha(payload/'initial.npz') and m['trace_sha256']==sha(payload/'optimizer.jsonl'),
            'Incorrect numerical reconstruction lineage')
    steps=m['gaussian_optimization_steps']
    n=budget['initialization']['count']
    require(n==m['initial_gaussian_count']==m['final_gaussian_count'] and steps==job['optimization_steps'] and m['topology_events']==[] and
            n<=50000 and steps<=5000 and tuple(m['generator_source_ids'])==ids,'Gaussian count, source set or step budget changed')
    init=gauss.initial_values(budget['initialization'])
    gaps={}
    with np.load(payload/'initial.npz',allow_pickle=False) as first,np.load(payload/'scene.npz',allow_pickle=False) as last:
        require(set(first.files)==set(last.files)==set(PARAMS) and gauss.digest_values(first)==m['initial_parameter_sha256'] and
                gauss.digest_values(last)==m['final_parameter_sha256'],'Parameter content SHA differs')
        for k in PARAMS:
            a,b=first[k],last[k]
            require(a.dtype==b.dtype==np.float32 and a.shape==b.shape==init[k].shape and np.isfinite(a).all() and np.isfinite(b).all(),'Invalid parameter tensor')
            gaps[k]=float(np.max(np.abs(a-init[k])))
            delta=float(np.linalg.norm((b-a).astype(np.float64)))
            require(delta>0 and abs(delta-m['parameter_delta_l2'][k])<=max(1e-7,delta*1e-8),'A changed geometry/appearance parameter was not demonstrated')
    log=[json.loads(v) for v in (payload/'optimizer.jsonl').read_text().splitlines()]
    require(len(log)==steps,'One real sampled source required per update')
    counts={s:Counter() for s in ids}; imgs=Counter()
    for i,v in enumerate(log):
        s=ids[i%len(ids)]
        require(v['update']==i+1 and v['source']==s and v['image_sha256']==m['source_rgb_sha256'][s] and v['pixels']==256**2
                and np.isfinite(v['loss_display_srgb_mae']) and set(v['gradient_l2'])==set(PARAMS),'Unexpected sample or source-gradient trace')
        imgs[s]+=1
        for p,g in v['gradient_l2'].items():
            require(np.isfinite(g) and g>=0,'Invalid gradient')
            counts[s][p]+=int(g>0)
    for s in ids:
        u=m['source_sampling'][s]
        require(imgs[s]==steps//len(ids)==u['samples'] and u['pixels']==imgs[s]*256**2 and
                dict(counts[s])==u['gradient_nonzero_steps'] and all(g>0 for g in counts[s].values()),'No measured input contribution')
    require(m['sampled_source_pixels']==steps*256**2,'Total pixel budget differs')
    return {'initial_parameter_sha256':m['initial_parameter_sha256'],'initial_checkpoint_sha256':m['initial_checkpoint_sha256'],
            'final_parameter_sha256':m['final_parameter_sha256'],'checkpoint_sha256':m['checkpoint_sha256'],'initial_mac_max_abs':gaps,
            'steps':steps,'sampled_pixels':m['sampled_source_pixels'],'source_sampling':m['source_sampling'],'geometry_and_appearance_l2_delta':m['parameter_delta_l2'],
            'loss_initial_by_source':m['loss_initial_by_source'],'loss_final_by_source':m['loss_final_by_source'],'gaussian_count':n,
            'optimization_seconds':m['optimization_seconds'],'runtime':m['runtime']}


def authenticate_render(root,job,pred_owner,rec,audit):
    p=root/'payload';m=load(p/'prediction.json'); views=SETS[job['condition']]+TARGETS
    require(m['schema']=='el-stage3-prediction-1' and m['run_id']==root.name and m['reconstruction_run_id']==pred_owner['run_id'] and
        m['reconstruction_sha256']==sha(rec/'payload/reconstruction.json') and m['checkpoint_sha256']==audit['checkpoint_sha256'] and
        m['source_rgb_sha256']=={s:job['files']['sources/'+s+'.png']['sha256'] for s in SETS[job['condition']]} and
        m['source_camera_sha256']==job['files']['source_cameras.json']['sha256'] and m['render_camera_sha256']==job['files']['render_cameras.json']['sha256'] and
        (m['rgb_encoding'],m['depth_semantics'],m['alpha_semantics'])==('display_srgb_0_1','expected_camera_z_m','accumulated_opacity') and
        tuple(m['generator_source_ids'])==SETS[job['condition']] and tuple(m['target_ids'])==TARGETS and m['depth_verified'] and
        set(m['views'])==set(views) and sha(p/'depth-probe.json')==m['depth_probe_sha256'],'Incompatible returned render/camera/ED')
    probe=load(p/'depth-probe.json')
    require(probe['passed'] and len(probe['samples'])==2 and all(abs(v['actual_z']-v['expected_z'])<=1e-4 and abs(v['alpha']-v['expected_alpha'])<=5e-4 and
        abs(v['actual_D']-v['expected_D'])<=1e-4 for v in probe['samples']) and abs(probe['translated_alpha']-.72)<=5e-4 and
        abs(probe['translated_z']-2.5)<=1e-4 and probe['empty_alpha']==0,'Off-axis opacity/ED readback failed')
    totals={}
    for vid in views:
        f=p/(vid+'.npz'); png=p/(vid+'.png'); v=m['views'][vid]
        require(sha(f)==v['sha256'] and sha(png)==v['png_sha256'] and f.stat().st_size==v['bytes'],'Changed prediction image')
        with np.load(f,allow_pickle=False) as a:
            require(set(a.files)=={'rgb','alpha','z'},'Prediction fields changed')
            rgb,alpha,z=a['rgb'],a['alpha'],a['z']
            require(rgb.shape==(256,256,3) and alpha.shape==z.shape==(256,256) and rgb.dtype==alpha.dtype==z.dtype==np.float32 and
                np.isfinite(rgb).all() and np.all((rgb>=0)&(rgb<=1)) and np.isfinite(alpha).all() and np.all((alpha>=0)&(alpha<=1)) and
                np.isnan(z[alpha==0]).all() and np.all(np.isnan(z)|((z>0)&np.isfinite(z))),'Invalid numeric output domain')
            with Image.open(png) as im:require(np.array_equal(np.asarray(im),np.floor(rgb*255+.5).astype(np.uint8)),'Prediction display bytes differ')
            totals[vid]={'rgb_valid':int(np.all(np.isfinite(rgb),-1).sum()),'positive_z':int(np.isfinite(z).sum()),'alpha_zero':int((alpha==0).sum())}
    return {'views':totals,'depth_probe_sha256':m['depth_probe_sha256'],'stage_seconds':m['stage_seconds'],'runtime':m['runtime']}


def published_attempts(selection,received,job_sha):
    previous={v['run_id']:v for v in selection['attempts']}
    for kind in ('generation','render'):
        for receipt in received.get((job_sha,kind),()):
            attempt=previous.get(receipt['run'])
            require(attempt is not None,'Additional return after selection requires explicit review; original remains immutable')
            require((receipt['manifest_sha256'],receipt['status'],kind,receipt['sha256'])==
                    (attempt['manifest_sha256'],attempt['status'],attempt['stage'],attempt['return_sha256']),
                    'Received attempt changed after its immutable selection')


def publish_postreview(store,selection,chosen,job,gauss):
    digest=hashlib.sha256(encoded(chosen)).hexdigest()
    if selection.exists():
        require(sha(selection)==digest,'Noncanonical selection bytes')
    audit=postreview_audit(store,chosen,job,digest,gauss)
    publish(selection.parent/'postreview-input-audit.json',encoded(audit))


def retry_predecessor_pending(archive,store):
    with zipfile.ZipFile(archive) as z:
        previous=json.loads(z.read('run.json')).get('retry_of_run_id')
    if not previous:
        return False
    from evidence_lens._bundle import RUN_ID
    require(isinstance(previous,str) and RUN_ID.fullmatch(previous),'Invalid original ID of returned retry')
    return not any((s/'runs'/previous/'manifest.json').is_file() for s in (store,store.parent/'cold-shelf'))


def intake(a):
    store,assets,project,returns=a.store.resolve(),a.assets.resolve(),a.project.resolve(),a.returns.resolve()
    require(sha(a.batch)==a.batch_sha256,'Batch selection digest differs')
    packet=load(a.batch)
    mod=assets/'src/evidence_lens/stage3_gaussians.py'
    require(sha(mod)==packet['worker_files']['src/evidence_lens/stage3_gaussians.py']['sha256'],'Validate with the worker actually dispatched')
    loader=importlib.util.spec_from_file_location('joint_parameters',mod)
    gauss=importlib.util.module_from_spec(loader);loader.loader.exec_module(gauss)
    received={}
    for f in sorted(returns.glob('*.receipt.json')):
        r=load(f)
        if r['handoff_sha256'] in {j['job_sha256'] for j in packet['jobs']}:
            received.setdefault((r['handoff_sha256'],r['run_kind']),[]).append(r)
    result={}
    for row in packet['jobs']:
        tid=row['trial_id'];jsha=row['job_sha256']
        if a.trial and a.trial!=tid:continue
        jobfile=a.batch.parent/row['path']
        require(sha(jobfile)==row['sha256'],'Mac job container differs')
        with zipfile.ZipFile(jobfile) as z:
            raw=z.read('job.json')
            require(__import__('hashlib').sha256(raw).hexdigest()==jsha,'Mac job receipt differs')
            j=json.loads(raw)
        base,chosen=check_native_selection(assets,store,j)
        mode='trials' if j['phase']=='production' else 'engineering'
        selection=base/mode/tid/'selection.json';selection.parent.mkdir(parents=True,exist_ok=True)
        with selection.with_suffix('.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if selection.exists():
                old=load(selection)
                require(old['job_sha256']==jsha,'Existing selection belongs to a different dispatch')
                published_attempts(old,received,jsha)
                publish_postreview(store,selection,old,j,gauss)
                result[tid]=old['selection_status'];continue
            selected={};attempts=[];reports={};pending=None;pre=None
            for kind in ('generation','render'):
                usable=[]
                for er in sorted(received.get((jsha,kind),()),key=lambda h:h['run']):
                    zipname=returns/er['package']
                    if not zipname.is_file() or zipname.stat().st_size!=er['bytes']:
                        continue
                    require(er['package']==er['run']+'.zip' and er['trial_id']==tid and sha(zipname)==er['sha256'],'Corrupt transport')
                    p={'run_id':er['run'],'manifest_sha256':er['manifest_sha256']}
                    evidence_dir=base/'receipts'/tid
                    marker=evidence_dir/(er['run']+'.json')
                    if marker.exists():
                        x=load(marker)
                        require(x['manifest_sha256']==p['manifest_sha256'] and x['container_sha256']==er['sha256'],'Prior transfer differs')
                    else:
                        if retry_predecessor_pending(zipname,store):
                            pending='waiting_for_retry_parent';continue
                        x=receive(zipname,store,er['sha256'],project/'configs/storage.json',shelf=store.parent/'cold-shelf')
                        if x['status']=='deferred_capacity':
                            pending='waiting_capacity';break
                        require(x['status'] in ('accepted','identical') and (x['run_id'],x['manifest_sha256'])==(p['run_id'],p['manifest_sha256']),'Native transfer declined')
                        publish(marker,encoded(x))
                    native=located(store,p); rec=lineage(native,j,jsha,er,SETS[j['condition']],pre)
                    attempts.append({**p,'stage':kind,'status':rec['execution_status'],'reason':rec['failure_reason'],
                        'retry_of':rec['retry_of_run_id'],'return_sha256':er['sha256'],'return_bytes':er['bytes'],'intake_receipt_sha256':sha(marker)})
                    if rec['execution_status']=='succeeded':usable.append(p)
                require(len(usable)<=1,'Multiple successful scientific attempts must be reconciled')
                if not usable:
                    pending=pending or ('retained_failure' if any(v['stage']==kind for v in attempts) else 'waiting_for_'+kind)
                    break
                p=usable[0]
                root=located(store,p)
                if kind=='generation':
                    pre=p;selected['reconstruction']=p
                    reports['reconstruction']=authenticate_reconstruction(root,j,gauss)
                else:
                    selected['render']=p
                    reports['render']=authenticate_render(root,j,pre,located(store,pre),reports['reconstruction'])
            if pending:
                if a.terminal is None or not attempts or a.terminal not in {x['run_id'] for x in attempts if x['status']!='succeeded'}:
                    result[tid]=pending;continue
                failure=next(x for x in attempts if x['run_id']==a.terminal)
                chosen.update(schema='el-stage3-failed-trial-1',selection_status='terminal_failure',failure=failure)
            else:
                chosen.update(schema='el-stage3-trial-selection-1',selection_status='selected')
            chosen['selected'].update(selected)
            chosen.update(job_sha256=jsha,batch_sha256=a.batch_sha256,attempts=attempts,numerical_readback=reports,
                          production_budget=j.get('production_budget'),optimizer_sha256=j['files']['optimization.json']['sha256'],
                          optimization_steps=j['optimization_steps'],worker_sha256=j['worker_sha256'])
            publish_postreview(store,selection,chosen,j,gauss)
            publish(selection,encoded(chosen))
            result[tid]=chosen['selection_status']
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('store','assets','project','returns','batch'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--batch-sha256',required=True);p.add_argument('--trial');p.add_argument('--terminal')
    print(json.dumps(intake(p.parse_args()),sort_keys=True))
