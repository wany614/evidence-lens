"""Run the real two-family source-access boundary canary under the immutable recorder."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

from evidence_lens._bundle import encoded,load,sha
from evidence_lens._verify import require
from make_handoff import publish,sha_bytes
from stage3_jobs import build_job,canary,capture
from stage3_reference import located,perform


def audit(a):
    packet=load(a.batch)
    rows=[r for r in packet['jobs'] if r['phase']=='engineering-pilot']
    homes=sorted({r['family'] for r in rows})
    require(len(homes)==2 and len(rows)==4 and len(a.reference)==2,'Only registered pilot canary pairs')
    result={}
    for f,original in zip(homes,a.reference,strict=True):
        cells={}
        for row in (r for r in rows if r['family']==f):
            source=a.jobs/Path(row['path']).name
            require(sha(source)==row['sha256'],'Changed delivery ZIP')
            with zipfile.ZipFile(source) as z:
                obj=json.loads(z.read('job.json'));meta=obj.pop('files')
                budget=z.read('optimization.json')
            require(meta['optimization.json']['sha256']==sha_bytes(budget) and obj['identity']['scene_family_id']==f,'Input-only job identity differs')
            package,jobsha=build_job(capture(original,obj['condition']),budget,obj)
            require(sha_bytes(package)==row['sha256'] and jobsha==row['job_sha256'],'Primary original job cannot be reproduced')
            cells[obj['condition']]=(budget,obj,row['sha256'])
        require(set(cells)=={'M2','M4'},'Pilot conditions missing')
        actual=canary(original,cells)
        require(actual==packet['source_only_canary'][f],'Removed/replaced-truth readback disagrees with delivery')
        result[f]=actual
    return {'schema':'el-stage3-source-only-canary-1','run_id':os.environ['EL_RUN_ID'],'catalog_sha256':packet['catalog_sha256'],
            'batch_sha256':sha(a.batch),'scope':'two real train pilot source packs; original, all truth absent, and external substituted wrong truth',
            'research_generation_or_model_updates':0,'checks':result,'passed':True}


def record(a):
    store,assets,project=a.store.resolve(),a.assets.resolve(),a.project.resolve()
    packet=load(a.batch)
    base=store/'stage3'/packet['catalog_sha256']
    marker=base/'batches'/packet['batch']/'source-boundary-audit.json'
    if marker.exists():return load(marker)
    families=sorted(packet['source_only_canary'])
    require(len(families)==2 and len([j for j in packet['jobs'] if j['phase']=='engineering-pilot'])==4,'Unexpected engineering pilot')
    with tempfile.TemporaryDirectory(dir=store/'work',prefix='stage3-input-audit-') as tmp:
        src=Path(tmp)/'engine';(src/'scripts').mkdir(parents=True)
        for n in ('stage3_canary','stage3_jobs','stage3_reference'):
            shutil.copyfile(assets/('scripts/'+n+'.py'),src/('scripts/'+n+'.py'))
        for name in ('make_handoff.py','formal_lineage.py'):
            shutil.copyfile(project/'scripts'/name,src/'scripts'/name)
        args=['{python}','{source}/scripts/stage3_canary.py','work','--batch','{input:batch}','--jobs','{input:jobs}','--out','{out}']
        roles={'batch':(a.batch,None),'jobs':(a.batch.parent/'jobs',None)}
        parents={}
        for i,f in enumerate(families):
            receipt=load(base/'references'/(f+'.json'))
            key='ref'+str(i)
            parents[key]=receipt['reference']
            roles[key]=(located(store,parents[key])/'payload',key)
            args+=['--reference','{input:'+key+'}']
            parents['readback'+str(i)]=receipt['readback']
        cfg={'experiment_id':'stage3-source-only-boundary-v2','trial_id':packet['batch'],'dataset_hash':packet['catalog_sha256'],'split':'train',
             'boundary_batch_sha256':sha(a.batch),'recording':{'expected_output_bytes':100000}}
        if a.supersedes:
            rid,h=a.supersedes.split('=')
            old=located(store,{'run_id':rid,'manifest_sha256':h})
            meta=load(old/'run.json')
            require(meta['experiment_id']=='stage3-source-only-boundary' and meta['trial_id']==packet['batch'] and meta['execution_status']=='failed',
                    'Prior operational attempt must be this same failed canary')
            cfg['supersedes_operational_attempt']={'run_id':rid,'manifest_sha256':h,'stderr_sha256':sha(old/'stderr.log'),
                'cause':'The first Mac canary snapshot omitted formal_lineage, a transitive import of its archive helper. No geometry, image, or model was produced.'}
        pin=perform(store,project,src,cfg,'archive_verify',roles,parents,args)
        proof=located(store,pin)/'payload/canary.json'
        result={'schema':'el-stage3-boundary-audit-selection-1','run':pin,'proof_sha256':sha(proof),'batch_sha256':sha(a.batch),
                'supersedes_operational_attempt':cfg.get('supersedes_operational_attempt')}
        publish(marker,encoded(result))
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('work','record'));p.add_argument('--batch',type=Path,required=True)
    for n in ('store','project','assets','jobs','out'):p.add_argument('--'+n,type=Path)
    p.add_argument('--reference',type=Path,action='append',default=[]);p.add_argument('--supersedes')
    a=p.parse_args()
    if a.action=='work':
        result=audit(a);(a.out/'canary.json').write_bytes(encoded(result))
    else:print(json.dumps(record(a)))
