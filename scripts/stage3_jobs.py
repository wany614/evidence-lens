"""Create exact-source C jobs, verify truth removal leaves both real pilot inputs invariant, and pin the Windows runner."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

from evidence_lens._bundle import encoded,load,sha
from evidence_lens._verify import require
from make_handoff import publish,sha_bytes,zip_bytes
from stage3_reference import family_info,located

CORE=('__init__','_bundle','_capacity','_dedup','_index','_metrics','_snapshot','_telemetry','_verify','archive','recorder')
NEW=('scripts/stage3_windows.py','src/evidence_lens/stage3_gaussians.py')
SETS={'M2':['s00','s03'],'M4':['s00','s01','s02','s03']}


def signature(data):
    return {'sha256':sha_bytes(data),'bytes':len(data)}


def code_files(project,assets):
    original=['scripts/worker_jobs.py','configs/storage.json',*('src/evidence_lens/'+n+'.py' for n in CORE)]
    return {**{x:(project/x).read_bytes() for x in original},**{x:(assets/x).read_bytes() for x in NEW}}


def capture(payload,cond):
    parts={'source_cameras.json':(payload/'condition'/cond/'source_cameras.json').read_bytes(),
           'render_cameras.json':(payload/'condition'/cond/'render_cameras.json').read_bytes()}
    for view in SETS[cond]:
        parts['sources/'+view+'.png']=(payload/'sources'/(view+'.png')).read_bytes()
    require(set(json.loads(parts['source_cameras.json'])['views'])==set(SETS[cond]) and
            set(json.loads(parts['render_cameras.json'])['views'])==set(SETS[cond])|{f'v{i:02d}' for i in range(1,7)},'Camera subset violation')
    return parts


def build_job(packets,settings,job):
    data={**packets,'optimization.json':settings}
    entry={**job,'files':{k:signature(v) for k,v in sorted(data.items())}}
    raw=encoded(entry)
    return zip_bytes({**data,'job.json':raw}),sha_bytes(raw)


def canary(payload,cells):
    """Rebuild identical jobs from (a) permitted files only and (b) those plus wrong truth placeholders."""
    results={}
    with tempfile.TemporaryDirectory() as work:
        stripped=Path(work)/'inputs'
        for i in ('s00','s01','s02','s03'):
            f=stripped/'sources'/(i+'.png'); f.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(payload/'sources'/(i+'.png'),f)
        for c in ('M2','M4'):
            sub=stripped/'condition'/c;sub.mkdir(parents=True)
            for n in ('source_cameras','render_cameras'):shutil.copyfile(payload/'condition'/c/(n+'.json'),sub/(n+'.json'))
        for state in ('removed','substituted'):
            if state=='substituted':
                for path in ('scene/world.blend','reference/v01.npz','reference/v02.png','ground-truth/labels.npz','reference.json'):
                    f=stripped/path;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b'changed, deliberately invalid private oracle material')
            for c,(settings,job,expected) in cells.items():
                new,_=build_job(capture(stripped,c),settings,job)
                require(sha_bytes(new)==expected,'Prohibited truth altered the optimizer interface')
                results.setdefault(c,{})[state]=sha_bytes(new)
        a,b=capture(stripped,'M2'),capture(stripped,'M4')
        endpoints={}
        for s in SETS['M2']:
            ac,bc=(json.loads(x['source_cameras.json'])['views'][s] for x in (a,b))
            require(sha_bytes(a['sources/'+s+'.png'])==sha_bytes(b['sources/'+s+'.png']) and encoded(ac)==encoded(bc),'Endpoint RGB/calibration changed')
            endpoints[s]={'rgb_sha256':sha_bytes(a['sources/'+s+'.png']),'canonical_camera_record_sha256':sha_bytes(encoded(ac))}
        return {'identical_job_when_external_truth':results,'shared_endpoint_checks':endpoints}


def batch(a):
    project,assets,store=a.project.resolve(),a.assets.resolve(),a.store.resolve()
    require(a.batch.isascii() and a.batch.replace('-','').isalnum() and len(a.families)==len(set(a.families)),'Bad batch name or duplicated family')
    worker=code_files(project,assets)
    wsha=sha_bytes(zip_bytes(worker))
    rows=[];tests={};jobs={};header=None
    for name in a.families:
        catalog,cat,row=family_info(store,assets,name)
        base=store/'stage3'/cat
        refsel=load(base/'references'/(name+'.json'))
        reg=refsel['registration']
        registered=located(store,reg)/'payload'
        rp,rb=refsel['reference'],refsel['readback']
        payload=located(store,rp)/'payload'
        reference=load(payload/'reference.json')
        require(reference['run_id']==rp['run_id'] and reference['camera_sha256']==catalog['camera_sha256'] and
                sha(payload/'reference.json')==refsel['reference_sha256'] and
                sha(located(store,rb)/'payload/readback.json')==refsel['readback_sha256']
                and load(located(store,rb)/'payload/readback.json')['passed'],'Reference readback differs')
        budget=(registered/'optimizer.json').read_bytes(); h=sha_bytes(budget)
        require(h==load(registered/'registration.json')['pilot_optimizer_sha256'],'Optimizer differs from original registration')
        proto=json.loads(budget)
        if a.phase=='engineering-pilot':
            require(row['pilot'] and row['split']=='train','Pilot may only use the two registered training families')
            steps=proto['optimization']['engineering_pilot_updates']; production=None
        else:
            freeze=load(base/'production-budget.json')
            require(freeze['catalog_sha256']==cat and freeze['worker_sha256']==wsha,'Worker or roster differs from production freeze')
            production=freeze['budget']; doc=load(located(store,production)/'payload/production-budget.json')
            require(doc['catalog_sha256']==cat and doc['worker_sha256']==wsha and doc['optimizer_sha256']==h and
                    doc['optimization_steps'] in proto['optimization']['production_candidate_updates'],'Invalid sealed production budget')
            steps=doc['optimization_steps']
        header=(base,cat)
        wanted=[('M2',False),('M4',False)]
        if a.repeat and row['pilot'] and name==catalog['pilot_families'][0] and a.phase=='engineering-pilot':
            wanted.append(('M2',True))
        cells={}
        for c,repeat in wanted:
            phase='engineering-repeat' if repeat else a.phase
            tid=row['trials'][c]+({'production':'','engineering-pilot':'-pilot','engineering-repeat':'-pilot-repeat01'}[phase])
            data=capture(payload,c)
            for s in SETS[c]:
                require(signature(data['sources/'+s+'.png'])==reference['source_images'][s]['source'],'Original source bytes differ')
            for k in ('source_cameras','render_cameras'):
                require(signature(data[k+'.json'])==reference['condition_cameras'][c][k],'Original camera bytes differ')
            identity={'scene_family_id':name,'scene_id':row['scene_id'],'scene_version':row['scene_version'],'scene_hash':row['recipe_sha256'],
                      'dataset_id':catalog['dataset_id'],'dataset_hash':cat,'split':row['split'],'condition_id':c,
                      'camera_set_id':'stage3-256-'+c,'camera_set_hash':sha_bytes(data['render_cameras.json']),'trial_id':tid}
            job={'schema':'el-stage3-worker-job-1','phase':phase,'experiment_id':'stage3-joint-'+phase,'trial_id':tid,'identity':identity,
                 'condition':c,'generator_source_ids':SETS[c],'target_ids':catalog['target_ids'],'registration':reg,'reference':rp,'readback':rb,
                 'worker_sha256':wsha,'production_budget':production,'optimization_steps':steps,'optimization_source_pixels':steps*256*256}
            if row['split']=='test':
                from evidence_lens.stage3_gate import require_opening
                job['opening_method']=require_opening(store,cat,family=name,shelf=store.parent/'cold-shelf')['method']
            package,jsha=build_job(data,budget,job)
            path='jobs/'+tid+'.zip'
            jobs[path]=package
            rows.append({'path':path,'trial_id':tid,'condition':c,'phase':phase,'family':name,'job_sha256':jsha,**signature(package)})
            if not repeat: cells[c]=(budget,job,sha_bytes(package))
        if row['pilot']:
            tests[name]=canary(payload,cells)
    base,cat=header
    manifest={'schema':'el-stage3-windows-batch-1','batch':a.batch,'catalog_sha256':cat,'worker_sha256':wsha,'worker_files':{k:signature(v) for k,v in worker.items()},
              'jobs':rows,'source_only_canary':tests,'contains_target_truth':False,'job_protocol':'Archive contains each job separately. Optimizer recorder only stages own source PNGs and source cameras. Renderer stages its allowed cameras plus its own selected checkpoint.'}
    batchbytes=encoded(manifest)
    out=base/'batches'/a.batch
    publish(out/'batch.json',batchbytes)
    for n,data in jobs.items():publish(out/n,data)
    package=zip_bytes({'batch.json':batchbytes,**{'worker/'+k:v for k,v in worker.items()},**jobs})
    pack=out/(a.batch+'.zip');publish(pack,package)
    results={'schema':'el-stage3-batch-delivery-1','batch':a.batch,'catalog_sha256':cat,'batch_sha256':sha_bytes(batchbytes),'worker_sha256':wsha,
             'container_sha256':sha(pack),'container_bytes':pack.stat().st_size,'container_name':pack.name,'trials':[v['trial_id'] for v in rows]}
    publish(out/'delivery.json',encoded(results))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('project','assets','store'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--batch',required=True);p.add_argument('--families',nargs='+',required=True)
    p.add_argument('--phase',choices=('engineering-pilot','production'),required=True);p.add_argument('--repeat',action='store_true')
    print(json.dumps(batch(p.parse_args()),sort_keys=True))
