"""Acquire one registered half of the four-family C confirmation cohort after its real opening is supplied."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import zipfile

from evidence_lens._bundle import encoded,load,sha
from evidence_lens._verify import require,verify_bundle
from evidence_lens.stage3_gate import require_opening
from make_handoff import publish,sha_bytes
from stage3_jobs import batch as build_batch
from stage3_reference import CAT,located,main as native_reference


def utc_value(timestamp):
    require(isinstance(timestamp,str) and timestamp.endswith('Z'),'Missing recorded UTC acquisition time')
    return datetime.fromisoformat(timestamp[:-1]+'+00:00')


def acquire(a):
    store,assets,project=a.store.resolve(),a.assets.resolve(),a.project.resolve();catalog=load(assets/CAT);cat=sha(assets/CAT)
    require(a.part in (1,2) and a.opening_sha256,'Supply the exact opening SHA and half 1 or 2')
    # This must precede every read of a local study reference, source image, reconstruction or confirmatory selection.
    marker=require_opening(store,cat,opening_sha256=a.opening_sha256,shelf=store.parent/'cold-shelf')
    planned=[r['scene_family_id'] for r in catalog['families'] if r['split']=='test']
    require(planned==catalog['test_families']==marker['test_families'] and len(planned)==4 and marker['conditions']==['M2','M4'],
            'The confirmed cohort must be the original registered four families and two source conditions')
    families=planned[(a.part-1)*2:a.part*2];references={};split_name=f'v11-c-joint-test{a.part:02d}'
    for f in families:
        ref=native_reference(argparse.Namespace(action='reference',store=store,assets=assets,project=project,family=f,blender_python=a.blender_python))
        require(ref['family']==f and ref['split']=='test','Wrong native confirmation set')
        for key in ('reference','readback'):
            actual=verify_bundle(located(store,ref[key]));meta=actual['run']
            require(actual['manifest_sha256']==ref[key]['manifest_sha256'] and meta['execution_status']=='succeeded' and
                    meta['scene_family_id']==f and meta['split']=='test' and utc_value(meta['started_at_utc'])>=utc_value(marker['opened_at_utc']),
                    'A confirmation image or readback precedes its real method opening')
        references[f]={k:ref[k] for k in ('registration','reference','readback','reference_sha256','readback_sha256','camera_sha256')}
    delivery=build_batch(argparse.Namespace(project=project,assets=assets,store=store,batch=split_name,phase='production',families=families,repeat=False))
    path=store/'stage3'/cat/'batches'/split_name;spec=load(path/'batch.json')
    required={(f,c) for f in families for c in ('M2','M4')}
    require({(r['family'],r['condition']) for r in spec['jobs']}==required and len(spec['jobs'])==4,'Incomplete predetermined confirmation batch')
    for row in spec['jobs']:
        original=path/row['path'];require(sha(original)==row['sha256'],'Source-only archive changed');accepted=references[row['family']]
        with zipfile.ZipFile(original) as z:
            raw=z.read('job.json');receipt=json.loads(raw)
            require(sha_bytes(raw)==row['job_sha256'] and receipt['opening_method']==marker['method'] and
                    receipt['reference']==accepted['reference'] and receipt['readback']==accepted['readback'] and receipt['optimization_steps']==256 and
                    receipt['optimization_source_pixels']==16777216 and receipt['identity']['split']=='test' and
                    set(z.namelist())=={'job.json','optimization.json','source_cameras.json','render_cameras.json',*('sources/'+s+'.png' for s in receipt['generator_source_ids'])},
                    'Confirmatory Windows job and its actual method/inputs disagree')
    result={'schema':'el-stage3-postopening-reference-batch-1','catalog_sha256':cat,'opening_sha256':a.opening_sha256,'method':marker['method'],
            'method_sha256':marker['method_sha256'],'opened_at_utc':marker['opened_at_utc'],'families':families,'references':references,'delivery':delivery}
    publish(path/'opening-origin.json',encoded(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('project','assets','store','blender-python'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--opening-sha256',required=True);p.add_argument('--part',type=int,choices=(1,2),required=True)
    print(json.dumps(acquire(p.parse_args()),sort_keys=True))
