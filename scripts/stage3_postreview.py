"""Independently verify the C intake using actual source RGB, complete Gaussian arrays and calibrated poses."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evidence_lens._bundle import encoded,load,sha
from evidence_lens._verify import require,retry_matches,verify_bundle
from stage3_reference import located

TARGETS=tuple(f'v{i:02d}' for i in range(1,7))
FIELDS=('means','log_scales','quats','opacity_logits','color_logits')


def cameras_and_distances(ref,c,ids,optimizer):
    full=load(ref/'payload/cameras.json')['views']
    source=load(ref/'payload/condition'/c/'source_cameras.json')['views']
    render=load(ref/'payload/condition'/c/'render_cameras.json')['views']
    require(set(source)==set(ids) and set(render)==set(ids)|set(TARGETS),'Unexpected camera set')
    near,far=optimizer['render']['near_m'],optimizer['render']['far_m']
    for k,v in render.items():
        require(encoded(v)==encoded(full[k]) and (k not in source or encoded(v)==encoded(source[k])),'Source and render calibrations disagree')
        K=np.asarray(v['K'],dtype=np.float64)
        edge=np.array(K,copy=True);edge[:2,2]+=.5
        require(K.shape==(3,3) and np.array_equal(K[2],[0,0,1]) and K[0,1]==K[1,0]==0 and
            K[0,2]==K[1,2]==127.5 and K[0,0]==K[1,1] and K[0,0]>0 and v['width']==v['height']==256 and
            np.array_equal(v['K_edge'],edge) and np.array_equal(np.asarray(v['clip_m'],np.float32),np.asarray([near,far],np.float32)),
            'Skew, principal point, homogeneous row or rendered clip differs from the optimizer')
    loc={k:np.asarray(v['c2w'],dtype=np.float64)[:3,3] for k,v in render.items()}
    nearest={}
    for target in TARGETS:
        distance,source_id=min((float(np.linalg.norm(loc[target]-loc[s])),s) for s in ids)
        nearest[target]={'source_id':source_id,'distance_m':distance}
    return source,render,nearest


def audit(store,chosen,job,selection_sha,gauss):
    require(chosen['job_sha256'] and chosen['trial_id']==job['trial_id'] and chosen['selection_status'] in ('selected','terminal_failure'),
            'No authenticated selection')
    pins=chosen['selected']
    selected_succeeded=chosen['selection_status']=='selected'
    ids=tuple(job['generator_source_ids'])
    reg=located(store,pins['registration'])
    opt=reg/'payload/optimizer.json'
    optimizer=load(opt)
    osha=sha(opt)
    require(osha==job['files']['optimization.json']['sha256'],'Original optimizer hash differs')
    formal=job['phase']=='production'
    frozen=None
    if formal:
        bpin=job['production_budget'];frozen=load(located(store,bpin)/'payload/production-budget.json')
        require(frozen['optimizer_sha256']==osha and frozen['worker_sha256']==job['worker_sha256']
            and frozen['optimization_steps']==job['optimization_steps']==256 and frozen['pixels_per_condition']==256*256*256,
            'Production initial scene, registered source pixels or 256-step budget differ')
    rr=located(store,pins['reference'])
    source,render,nearest=cameras_and_distances(rr,job['condition'],ids,optimizer)
    actual_images={s:sha(rr/'payload/sources'/(s+'.png')) for s in ids}
    require(all(actual_images[s]==chosen['source_images'][s]['sha256']==job['files']['sources/'+s+'.png']['sha256'] for s in ids),'Source photographs differ')
    measures={'nearest_source_by_target':nearest,'nearest_source_tie_rule':'minimum Euclidean world-centre distance, then lexicographic source ID',
              'camera_clip_native_m':{k:v['clip_m'] for k,v in render.items()},'clip_matches_optimizer_float32':True,
              'source_camera_sha256':sha(rr/'payload/condition'/job['condition']/'source_cameras.json'),
              'render_camera_sha256':sha(rr/'payload/condition'/job['condition']/'render_cameras.json'),'source_images_sha256':actual_images}
    history={}
    for a in chosen['attempts']:
        pin={'run_id':a['run_id'],'manifest_sha256':a['manifest_sha256']}
        attempted=located(store,pin);meta=load(attempted/'run.json')
        require(meta['trial_id']==job['trial_id'] and meta['run_kind']==a['stage'] and meta['execution_status']==a['status'] and
                meta.get('retry_of_run_id')==a['retry_of'],'Retry position or archived scientific attempt differs')
        history[meta['run_id']]=(meta,attempted)
    for meta,attempted in history.values():
        old=meta['retry_of_run_id']
        if old:
            require(old in history,'Selected retry is missing its unsuccessful original')
            oldmeta,orig=history[old]
            require(oldmeta['execution_status']!='succeeded','Retry names a successful scientific ancestor')
            now=verify_bundle(attempted);then=verify_bundle(orig)
            retry_matches(then,now['run'],now['inputs'],now['command']['argv_template'])
    for tag,kind in [('reconstruction','generation'),('render','render')]:
        selected={r['run_id'] for r,_ in history.values() if r['run_kind']==kind and r['execution_status']=='succeeded'}
        require(selected==({pins[tag]['run_id']} if tag in pins else set()),'Successful attempt differs from selected state')
    if 'reconstruction' in pins:
        rec=located(store,pins['reconstruction'])/'payload';r=load(rec/'reconstruction.json')
        require(r['optimizer_sha256']==osha and r['source_rgb_sha256']==actual_images and r['source_camera_sha256']==measures['source_camera_sha256']
            and r['gaussian_optimization_steps']==job['optimization_steps'] and r['sampled_source_pixels']==job['optimization_steps']*256**2,
            'Returned run did not optimize the declared sources or budget')
        init=gauss.initial_values(optimizer['initialization'])
        with np.load(rec/'initial.npz',allow_pickle=False) as before,np.load(rec/'scene.npz',allow_pickle=False) as after:
            init_sha=gauss.digest_values(before)
            require(init_sha==r['initial_parameter_sha256']==gauss.digest_values(init) and
                    (not formal or init_sha==frozen['init_sha256'] and r['initial_checkpoint_sha256']==frozen['pilot_evidence']['common_initial_checkpoint_sha256']),
                    'Actual parameter initialization differs from Mac or the true frozen production budget')
            count=r['initial_gaussian_count']
            require(count==r['final_gaussian_count']==4096,'Unexpected Gaussian count')
            unchanged=np.ones(count,bool)
            per_field={};gap={}
            for k in FIELDS:
                x,y=before[k],after[k]
                require(x.dtype==y.dtype==np.float32 and x.shape==y.shape==init[k].shape and np.array_equal(x,init[k]),'Mac initial array differs')
                gap[k]=float(np.max(np.abs(x-init[k])))
                bit_equal=(np.ascontiguousarray(x).view(np.uint32).reshape(count,-1)==np.ascontiguousarray(y).view(np.uint32).reshape(count,-1)).all(1)
                per_field[k]=int(bit_equal.sum());unchanged &= bit_equal
            require(all(v==0 for v in gap.values()),'A nonzero initial Mac difference exceeds the fixed gate')
        measures.update(initial_parameter_sha256=init_sha,initial_mac_max_abs=gap,initial_checkpoint_sha256=r['initial_checkpoint_sha256'],
                        unchanged_gaussian_rows=int(unchanged.sum()),unchanged_rows_by_field=per_field,
                        sampled_source_pixels=r['sampled_source_pixels'],optimization_steps=r['gaussian_optimization_steps'])
    if 'render' in pins:
        dest=located(store,pins['render'])/'payload';pred=load(dest/'prediction.json')
        require(pred['source_camera_sha256']==measures['source_camera_sha256'] and pred['render_camera_sha256']==measures['render_camera_sha256'] and
            pred['reconstruction_run_id']==pins['reconstruction']['run_id'],'Source final-loss check requires the actual common checkpoint')
        rechecked={}
        for s in ids:
            with Image.open(rr/'payload/sources'/(s+'.png')) as im:
                source_rgb=np.asarray(im.convert('RGB')).astype(np.float64)/255
            f=dest/(s+'.npz')
            require(sha(f)==pred['views'][s]['sha256'],'Source prediction changed')
            with np.load(f,allow_pickle=False) as z:
                loss=float(np.mean(np.abs(z['rgb'].astype(np.float64)-source_rgb),dtype=np.float64))
            reported=r['loss_final_by_source'][s]
            gap=abs(loss-reported)
            require(np.isfinite(loss) and gap<=1e-5,'Recorded terminal training loss differs from source image versus final source render')
            rechecked[s]={'png_sha256':actual_images[s],'source_prediction_sha256':sha(f),'final_loss_recomputed':loss,'final_loss_optimizer':reported,'abs_difference':gap}
        measures['final_loss_by_source']=rechecked
        measures['final_loss_comparison_max_abs']=max(v['abs_difference'] for v in rechecked.values())
        measures['final_loss_abs_tolerance']=1e-5
    return {'schema':'el-stage3-postreview-input-audit-1','passed':True,'catalog_sha256':chosen['catalog_sha256'],'trial_id':job['trial_id'],
        'phase':job['phase'],'selection_status':chosen['selection_status'],'selection_sha256':selection_sha,'job_sha256':chosen['job_sha256'],
        'worker_sha256':job['worker_sha256'],'audit_code_sha256':sha(Path(__file__)),'optimizer_sha256':osha,'selected':pins,
        'production_budget':job.get('production_budget'),'failure_disposition':None if selected_succeeded else chosen['failure'],'measurements':measures}
