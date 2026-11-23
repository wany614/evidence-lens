"""Measure joint Gaussian acquisition costs from named native receipts, retaining unsuccessful attempts separately."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile

from evidence_lens._bundle import encoded,load,sha
from evidence_lens._verify import require,verify_bundle
from make_handoff import publish
from stage3_reference import located,perform


COLUMNS=('trial_id','family','split','condition','phase','selection_status','source_count','planned_views','complete_views',
         'actual_updates','source_pixels','initial_count','final_count','unchanged_full_rows','trained_sources','optimization_seconds',
         'generation_stage_seconds','render_stage_seconds','generation_wall_seconds','render_wall_seconds',
         'gen_torch_allocated_peak_bytes','gen_torch_reserved_peak_bytes','render_torch_allocated_peak_bytes','render_torch_reserved_peak_bytes',
         'gen_rss_peak_bytes','render_rss_peak_bytes','gpu_model','device_capacity_bytes','return_zip_bytes_all','failed_attempts','selection_sha256')
ATTEMPT_COLUMNS=('trial_id','family','split','condition','phase','stage','run_id','manifest_sha256','status','retry_of','wall_seconds','rss_peak_bytes',
                 'telemetry_samples','gpu_process_sample_count','gpu_process_sampled_peak_bytes','torch_allocated_peak_bytes','torch_reserved_peak_bytes',
                 'transfer_zip_bytes','transfer_zip_sha256','intake_receipt_sha256','failure_reason')


def write_csv(path,columns,rows):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,columns,extrasaction='ignore',lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def telemetry(p):
    with p.open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    samples=[int(r['gpu_process_memory_bytes']) for r in rows if r['gpu_process_memory_bytes']]
    missing=Counter(json.loads(r['missing_reason']).get('gpu_process_memory_bytes','not_reported') for r in rows if not r['gpu_process_memory_bytes'])
    rss=[int(r['tree_rss_bytes']) for r in rows if r['tree_rss_bytes']]
    return {'telemetry_samples':len(rows),'gpu_process_sample_count':len(samples),'gpu_process_sampled_peak_bytes':max(samples,default=None),
            'gpu_counter_missing_reasons':dict(missing),'sampled_rss_peak':max(rss,default=None)}


def measure_trial(store,chosen,companion,cell):
    tid=chosen['trial_id'];q=chosen['selected'];a=companion['measurements'];ids=chosen['generator_source_ids']
    require(companion['passed'] and companion['selected']==q and cell['selection_sha256']==companion['selection_sha256'] and
            cell['measurements']==a and cell['status']==chosen['selection_status'],'Fresh and prior intake disagree')
    attributes={k:chosen[k] for k in ('trial_id','family','condition','split','phase')}
    row={**attributes,'selection_sha256':cell['selection_sha256'],'selection_status':chosen['selection_status'],'source_count':len(ids),
         'planned_views':len(ids)+6,'complete_views':0,'trained_sources':0,'source_ids':ids,'selection':q,'failed_attempts':0,'return_zip_bytes_all':0,
         'recorded_failure':chosen.get('failure'),'nearest_source_by_target':a['nearest_source_by_target']}
    attempts=[]
    for x in chosen['attempts']:
        pin={k:x[k] for k in ('run_id','manifest_sha256')};sealed=located(store,pin);v=verify_bundle(sealed);meta=v['run']
        require(v['manifest_sha256']==pin['manifest_sha256'] and meta['trial_id']==tid and meta['run_kind']==x['stage'] and
                meta['execution_status']==x['status'] and meta['retry_of_run_id']==x['retry_of'],'Original attempt identity differs')
        tr=store/'stage3'/chosen['catalog_sha256']/'receipts'/tid/(pin['run_id']+'.json')
        transfer=load(tr)
        require(sha(tr)==x['intake_receipt_sha256'] and transfer['status'] in ('accepted','identical') and transfer['run_id']==pin['run_id'] and
                transfer['manifest_sha256']==pin['manifest_sha256'] and transfer['container_sha256']==x['return_sha256'] and
                transfer['bytes']==x['return_bytes'],'Transfer measurement differs from selected attempt')
        sample=telemetry(sealed/'telemetry.csv')
        require(sample.pop('sampled_rss_peak')==meta['sampled_peak_tree_rss_bytes'],'Original sampled process-tree RSS differs')
        z={**attributes,**pin,'stage':x['stage'],'status':meta['execution_status'],'retry_of':meta['retry_of_run_id'],
           'failure_reason':meta['failure_reason'],'wall_seconds':meta['elapsed_seconds_monotonic'],'rss_peak_bytes':meta['sampled_peak_tree_rss_bytes'],
           'transfer_zip_bytes':x['return_bytes'],'transfer_zip_sha256':x['return_sha256'],'intake_receipt_sha256':x['intake_receipt_sha256'],**sample}
        row['return_zip_bytes_all']+=x['return_bytes'];attempts.append(z)
        if x['status']!='succeeded':
            row['failed_attempts']+=1
            continue
        is_gen=x['stage']=='generation';stage='gen' if is_gen else 'render';method='reconstruction' if is_gen else 'render'
        require(q[method]==pin,'Unexpected additional successful native stage')
        original=load(sealed/'payload'/('reconstruction.json' if is_gen else 'prediction.json'))
        run=original['runtime'];native=chosen['numerical_readback'][method]
        require(run==native['runtime'] and run['gsplat']=='1.5.3','Original backend/allocator differs')
        for label in ('allocated','reserved'):
            key='torch_cuda_'+label+'_peak_bytes';z['torch_'+label+'_peak_bytes']=run[key]
            row[stage+'_torch_'+label+'_peak_bytes']=run[key]
        require(row.get('gpu_model',run['device'])==run['device'] and row.get('device_capacity_bytes',run['gpu_total_bytes'])==run['gpu_total_bytes'],
                'Generation and rendering hardware differ')
        row.update(gpu_model=run['device'],device_capacity_bytes=run['gpu_total_bytes']);row[stage+'_rss_peak_bytes']=z['rss_peak_bytes']
        if is_gen:
            require(a['optimization_steps']==original['gaussian_optimization_steps'] and a['sampled_source_pixels']==original['sampled_source_pixels'] and
                    native['source_sampling']==original['source_sampling'] and set(ids)==set(original['source_sampling']), 'Final source sampling differs')
            row.update(actual_updates=a['optimization_steps'],source_pixels=a['sampled_source_pixels'],initial_count=original['initial_gaussian_count'],
                       final_count=original['final_gaussian_count'],unchanged_full_rows=a['unchanged_gaussian_rows'],trained_sources=len(original['source_sampling']),
                       source_sampling=original['source_sampling'],source_images_sha256=original['source_rgb_sha256'],optimization_seconds=original['optimization_seconds'],
                       generation_stage_seconds=original['stage_seconds'],generation_wall_seconds=z['wall_seconds'],initial_parameter_sha256=a['initial_parameter_sha256'],
                       final_parameter_sha256=original['final_parameter_sha256'],unchanged_rows_by_field=a['unchanged_rows_by_field'],read_seconds=original['read_seconds'])
        else:
            require(original['run_id']==pin['run_id'] and original['reconstruction_run_id']==q['reconstruction']['run_id'] and len(original['views'])==len(ids)+6,
                    'Wrong reconstruction or render view count')
            row.update(complete_views=len(original['views']),render_stage_seconds=original['stage_seconds'],render_wall_seconds=z['wall_seconds'],
                       per_view_render_seconds={v:i['seconds'] for v,i in original['views'].items()})
    return row,attempts


def summaries(rows):
    out={}
    for scope in ('engineering','production'):
        selected=[r for r in rows if (r['phase']=='production')==(scope=='production')]
        if not selected:continue
        summary={'trials':len(selected),'selection_statuses':dict(Counter(r['selection_status'] for r in selected)),
                 'planned_views':sum(r['planned_views'] for r in selected),'complete_views':sum(r['complete_views'] for r in selected),
                 'source_pixels_successful_reconstructions':sum(r.get('source_pixels',0) for r in selected),
                 'all_returned_zip_bytes':sum(r['return_zip_bytes_all'] for r in selected),'failed_attempts':sum(r['failed_attempts'] for r in selected),'by_condition':{}}
        for c in ('M2','M4'):
            group=[r for r in selected if r['condition']==c];scope_results={}
            for key in ('optimization_seconds','generation_wall_seconds','render_wall_seconds','render_stage_seconds','gen_torch_allocated_peak_bytes',
                        'gen_torch_reserved_peak_bytes','render_torch_allocated_peak_bytes','render_torch_reserved_peak_bytes','gen_rss_peak_bytes','render_rss_peak_bytes'):
                values=[r[key] for r in group if r.get(key) is not None]
                scope_results[key]={'n':len(values),'min':min(values),'median':statistics.median(values),'max':max(values)} if values else {'n':0}
            summary['by_condition'][c]=scope_results
        out[scope]=summary
    return out


def work(a):
    request=load(a.inputs/'request.json');info=load(a.inputs/'catalog.json');cat=sha(a.inputs/'catalog.json')
    require(cat==request['catalog_sha256'],'Source catalog mismatch')
    if request['scope']=='full':
        from evidence_lens.stage3_gate import require_opening
        require_opening(a.store,cat,opening_sha256=request['opening_sha256'],shelf=a.store.parent/'cold-shelf')
    allowed={r['scene_family_id']:r for r in info['families']};rows=[];attempts=[];seen=set()
    for batch in request['batches']:
        ap=located(a.store,batch['audit'])/'payload/postreview.json';audit=load(ap)
        require(sha(ap)==batch['audit_sha256'] and audit['passed'] and audit['batch_sha256']==batch['batch_sha256'],'Missing original cohort readback')
        for tid,cell in audit['results'].items():
            d=a.inputs/'selections'/tid;sp=d/'selection.json';pp=d/'postreview-input-audit.json'
            require(sha(sp)==cell['selection_sha256'] and sha(pp)==cell['postreview_sha256'] and tid not in seen,'Duplicated or substituted trial')
            chosen=load(sp);f=allowed[chosen['family']];require(f['split']==chosen['split'] and (f['split']!='test' or request['scope']=='full'),'Unopened input')
            row,log=measure_trial(a.store,chosen,load(pp),cell);rows.append(row);attempts.extend(log);seen.add(tid)
    expected={tid for r in info['families'] if r['split']!='test' or request['scope']=='full' for tid in r['trials'].values()}
    formal=[x for x in rows if x['phase']=='production'];pilots=[x for x in rows if x['phase']!='production']
    require({x['trial_id'] for x in formal}==expected and len(pilots)==5 and sum(x['phase']=='engineering-repeat' for x in pilots)==1,'Cost matrix omits a registered trial or repeat')
    write_csv(a.out/'scientific-costs.csv',COLUMNS,formal);write_csv(a.out/'engineering-costs.csv',COLUMNS,pilots);write_csv(a.out/'stage-attempts.csv',ATTEMPT_COLUMNS,attempts)
    doc={'schema':'el-stage3-joint-acquisition-costs-1','run_id':os.environ['EL_RUN_ID'],'scope':request['scope'],'catalog_sha256':cat,
         'opening_sha256':request['opening_sha256'],'batches':request['batches'],'summary':summaries(rows),'trials':rows,'attempts':attempts,
         'source_notes':{'optimization_seconds':'Windows perf_counter around all Adam updates and the final CUDA synchronize. Includes host gradient reads and JSONL logging.',
                         'stage_seconds':'Process-local wallclock for the worker including initialization and its output handling to the point of measurement.',
                         'wall_seconds':'The separate immutable recorder elapsed_seconds_monotonic for each attempt, including worker launch.',
                         'torch_peaks':'The GPU allocator reports allocated and reserved bytes separately. Device capacity is the CUDA reported total.',
                         'gpu_process_sampled_peak_bytes':'Maximum of actual available driver PID-tree GPU samples. Null when absent. Missing reasons and sample counts are retained.',
                         'rss_peak_bytes':'Maximum sampled process-tree resident bytes observed by the recorder at its configured interval.',
                         'transfer_zip_bytes':'Original received ZIP sizes include source, inputs, payload and recorder evidence for every returned attempt.'},
         'tables':{p.name:sha(p) for p in a.out.glob('*.csv')}}
    (a.out/'costs.json').write_bytes(encoded(doc))


def record(a):
    project,assets,store=a.project.resolve(),a.assets.resolve(),a.store.resolve();cf=assets/'protocol/stage3-families-v1.json';cat=sha(cf);base=store/'stage3'/cat
    catalog=load(cf);marker=base/'costs'/(a.scope+'.json');parents={};combined=[]
    if a.scope=='full':
        from evidence_lens.stage3_gate import require_opening
        require(a.opening_sha256,'The real opening digest is required for the 32-condition ledger')
        opening=require_opening(store,cat,opening_sha256=a.opening_sha256,shelf=store.parent/'cold-shelf');parents['method']=opening['method']
    if marker.exists():
        saved=load(marker);r=located(store,saved['run']);require(sha(r/'payload/costs.json')==saved['costs_sha256'] and
                saved['batch_arguments']==a.batch and saved['opening_sha256']==a.opening_sha256,'Existing ledger uses different sources');return saved
    with tempfile.TemporaryDirectory(dir=store/'work',prefix='stage3-acquisition-cost-') as temporary:
        temp=Path(temporary);src=temp/'engine';(src/'scripts').mkdir(parents=True);inputs=temp/'inputs';inputs.mkdir();shutil.copyfile(cf,inputs/'catalog.json')
        for name in ('stage3_acquisition_costs.py','stage3_reference.py'):shutil.copyfile(assets/'scripts'/name,src/'scripts'/name)
        for name in ('make_handoff.py','formal_lineage.py'):shutil.copyfile(project/'scripts'/name,src/'scripts'/name)
        for i,arg in enumerate(a.batch):
            name,expected=arg.split('=');require(name.isascii() and name.replace('-','').isalnum(),'Invalid batch name')
            where=base/'batches'/name;packet=load(where/'batch.json');require(sha(where/'batch.json')==expected,'Unpinned acquisition batch')
            family_split={r['scene_family_id']:r['split'] for r in catalog['families']}
            require(all(family_split[t['family']]!='test' or a.scope=='full' for t in packet['jobs']),'Engineering/train/dev ledger cannot access test selections')
            ar=load(where/'postreview-archive.json');require(ar['batch_sha256']==expected and ar['catalog_sha256']==cat,'Wrong audit parent')
            parents['audit'+str(i)]=ar['run'];combined.append({'batch':name,'batch_sha256':expected,'audit':ar['run'],'audit_sha256':ar['postreview_sha256']})
            for row in packet['jobs']:
                tid=row['trial_id'];dest=inputs/'selections'/tid;dest.mkdir(parents=True,exist_ok=False)
                root=base/('trials' if row['phase']=='production' else 'engineering')/tid
                for n in ('selection.json','postreview-input-audit.json'):shutil.copyfile(root/n,dest/n)
        req={'catalog_sha256':cat,'scope':a.scope,'opening_sha256':a.opening_sha256,'batches':combined};(inputs/'request.json').write_bytes(encoded(req))
        cfg={'experiment_id':'stage3-acquisition-resources','trial_id':'stage3-acquisition-'+a.scope,'dataset_hash':cat,'split':'mixed',
             'inputs_sha256':sha(inputs/'request.json'),'recording':{'expected_output_bytes':1200000}}
        argv=['{python}','{source}/scripts/stage3_acquisition_costs.py','work','--inputs','{input:inputs}','--store',str(store),'--out','{out}']
        pin=perform(store,project,src,cfg,'report',{'inputs':(inputs,None)},parents,argv)
    r=located(store,pin);proof=r/'payload/costs.json';q=load(proof);result={'schema':'el-stage3-acquisition-costs-selection-1','run':pin,
         'costs_sha256':sha(proof),'opening_sha256':a.opening_sha256,'batch_arguments':a.batch,'tables':q['tables'],'scope':a.scope}
    publish(marker,encoded(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('record','work'));p.add_argument('--scope',choices=('development','full'))
    for n in ('project','assets','store','inputs','out'):p.add_argument('--'+n,type=Path)
    p.add_argument('--batch',action='append',default=[]);p.add_argument('--opening-sha256');args=p.parse_args()
    if args.action=='work':work(args)
    else:print(json.dumps(record(args),sort_keys=True))
