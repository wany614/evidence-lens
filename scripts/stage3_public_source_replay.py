"""Regenerate the fixed development example solely from its public authored recipe, software and calibration."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from evidence_lens._bundle import encoded,load,sha
from evidence_lens._verify import require
from make_handoff import publish
from stage3_reference import located,perform


def work(a):
    world,cams=a.world,a.cameras
    cfg=load(world/'config.json');require(cfg['scene_family_id']=='scene_057_offset_bell_cage' and cfg['split']=='development','The registered public example is required')
    with tempfile.TemporaryDirectory() as raw:
        tmp=Path(raw);fresh=tmp/'produced';fresh.mkdir()
        entry=world/'code/src/evidence_lens/stage3_native.py'
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':''}
        subprocess.run([str(a.blender_python),'-B',str(entry),'produce','--out',str(fresh),'--config',str(world/'config.json'),'--settings',str(world/'settings.json'),
            '--recipe',str(world/'recipe.json'),'--cameras',str(cams)],check=True,env=env,cwd=tmp)
        regenerated=load(fresh/'reference.json');require(sha(fresh/'cameras.json')==sha(cams)==regenerated['camera_sha256'],'Original public camera calibration differs')
        data={}
        for s in regenerated['source_ids']:
            original=a.sources/(s+'.png');new=fresh/'sources'/(s+'.png')
            require(new.read_bytes()==original.read_bytes(),'Published original source is not regenerated exactly')
            data[s]={'source_png_sha256':sha(original),'regenerated_png_sha256':sha(new),'bytes':new.stat().st_size,'png_bytes_equal':True}
        for v in regenerated['target_ids']:
            w=a.truth/(v+'.npz');new=fresh/'reference'/(v+'.npz')
            checks={}
            with np.load(w,allow_pickle=False) as one,np.load(new,allow_pickle=False) as two:
                require(set(one.files)==set(two.files),'Target array inventory differs')
                for k in one.files:
                    x,y=one[k],two[k];checks[k]=x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
                require(all(checks.values()),'A source render or true RGB, Position, ID, ED support, per-source visibility or reason changed: '+v)
            o=a.truth/(v+'.png');n=fresh/'reference'/(v+'.png');require(o.read_bytes()==n.read_bytes(),'Published target PNG bytes differ')
            data[v]={'public_target_npz_sha256':sha(w),'regenerated_target_npz_sha256':sha(new),'all_numerical_fields_bitwise_equal':checks,
                     'public_target_png_sha256':sha(o),'regenerated_target_png_sha256':sha(n),'png_bytes_equal':True}
        require(len(data)==10,'All four source photographs and six targets are required')
        result={'schema':'el-stage3-standalone-public-world-replay-1','run_id':os.environ['EL_RUN_ID'],'subject':'scene_057_offset_bell_cage','passed':True,
                'public_inputs':{'config':sha(world/'config.json'),'recipe':sha(world/'recipe.json'),'settings':sha(world/'settings.json'),'cameras':sha(cams),
                                 'native_entry':sha(entry)},'sources_in_produce_subprocess':['authored config','recipe','settings','cameras','public Python code'],
                'result':'All four source PNGs and six target PNGs were reproduced byte-for-byte. All public target numerical arrays were reproduced bit-for-bit.',
                'comparisons':data}
    (a.out/'public-world-replay.json').write_bytes(encoded(result))


def record(a):
    store,project,assets=a.store.resolve(),a.project.resolve(),a.assets.resolve();cat=sha(assets/'protocol/stage3-families-v1.json');area=store/'stage3'/cat/'public-example'
    q=load(area/'selection.json');pin=q['run'];case=located(store,pin)/'payload';marker=area/'full-world-replay.json'
    if marker.exists():
        receipt=load(marker);proof=located(store,receipt['run'])/'payload/public-world-replay.json';require(receipt['public_case']==pin and sha(proof)==receipt['replay_sha256'],'Independent replay pointer mismatches');return receipt
    roles={'world':(case/'REFERENCE/world','public'),'cameras':(case/'REFERENCE/cameras.json','public'),'sources':(case/'CASE/sources','public'),
           'truth':(case/'REFERENCE/reference','public')}
    with tempfile.TemporaryDirectory(dir=store/'work',prefix='stage3-public-standalone-') as raw:
        s=Path(raw)/'engine';(s/'scripts').mkdir(parents=True)
        for n in ('stage3_public_source_replay.py','stage3_reference.py'):shutil.copyfile(assets/'scripts'/n,s/'scripts'/n)
        for n in ('make_handoff.py','formal_lineage.py'):shutil.copyfile(project/'scripts'/n,s/'scripts'/n)
        cmd=['{python}','{source}/scripts/stage3_public_source_replay.py','work','--out','{out}','--blender-python',str(a.blender_python.absolute())]
        for n in roles:cmd+=['--'+n,'{input:'+n+'}']
        cfg={'experiment_id':'stage3-public-world-replay','trial_id':'scene_057_offset_bell_cage-standalone','scene_family_id':'scene_057_offset_bell_cage','split':'development',
             'dataset_hash':cat,'public_case':pin,'recording':{'expected_output_bytes':65000000}}
        run=perform(store,project,s,cfg,'archive_verify',roles,{'public':pin},cmd)
    proof=located(store,run)/'payload/public-world-replay.json';require(load(proof)['passed'],'Public native replay failed')
    receipt={'schema':'el-stage3-public-world-replay-selection-1','public_case':pin,'run':run,'replay_sha256':sha(proof)};publish(marker,encoded(receipt));return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('record','work'))
    for name in ('project','store','assets','world','cameras','sources','truth','out','blender-python'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.action=='work':work(a)
    else:print(json.dumps(record(a),sort_keys=True))
