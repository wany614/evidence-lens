"""Record the C preregistration, a native family and a separate reopened-Blender readback."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from evidence_lens._bundle import encoded,load,sha
from evidence_lens._verify import require,verify_bundle
from make_handoff import publish

CAT='protocol/stage3-families-v1.json'
OPT='protocol/stage3-optimizer-v1.json'
ENGINE=('scene.py','passes.py','validation.py')


def family_info(store,assets,family):
    doc=load(assets/CAT)
    key=sha(assets/CAT)
    row=next((v for v in doc['families'] if v['scene_family_id']==family),None)
    require(row is not None,'Unknown family')
    if row['split']=='test':
        from evidence_lens.stage3_gate import require_opening
        require_opening(store,key,family=family,shelf=store.parent/'cold-shelf')
    return doc,key,row


def engine_copy(project,assets,folder,registration):
    names=[(assets,'scripts/stage3_registration_worker.py' if registration else 'src/evidence_lens/stage3_native.py'),
           (assets,'scripts/stage3_scene_designs.py'),(assets,'scripts/stage3_register_reference.py'),(assets,'scripts/stage3_reference.py'),
           (project,'src/evidence_lens/blender_reference/formal_designs.py'),(project,'src/evidence_lens/blender_reference/formal_catalog.py'),
           (project,'scripts/stage2_reference_catalog.py')]
    if registration:
        names += [(assets,'src/evidence_lens/stage3_native.py')]
    names += [(project,'src/evidence_lens/blender_reference/'+x) for x in ENGINE]
    for root,n in names:
        dst=folder/n
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/n,dst)


def perform(store,project,src,cfg,kind,inputs,parents,worker):
    cfgpath=src.parent/'resolved.json'
    cfgpath.write_bytes(encoded(cfg))
    cmd=[sys.executable,'-m','evidence_lens.recorder','--root',str(store),'--source-root',str(src),'--storage-policy',str(project/'configs/storage.json'),
         '--kind',kind,'--config',str(cfgpath)]
    for key,pin in parents.items():
        cmd+=['--parent',pin['run_id']+'='+pin['manifest_sha256']]
    for role,(path,parent) in inputs.items():
        cmd+=['--input',role+'='+str(path)]
        if parent:
            cmd+=['--input-parent',role+'='+parents[parent]['run_id']]
    proc=subprocess.run([*cmd,'--',*worker],env={**os.environ,'PYTHONPATH':str(project/'src')},capture_output=True,text=True)
    if proc.stderr:
        print(proc.stderr,file=sys.stderr,flush=True)
    if proc.returncode:
        print(proc.stdout,file=sys.stderr,flush=True)
        raise RuntimeError('Recorder exited '+str(proc.returncode))
    result=json.loads(proc.stdout.splitlines()[-1])
    checked=verify_bundle(Path(result['path']))
    require(checked['run']['execution_status']=='succeeded','Reference stage failed')
    return {'run_id':result['run_id'],'manifest_sha256':checked['manifest_sha256']}


def located(store,pin):
    candidates=[p/'runs'/pin['run_id'] for p in (store,store.parent/'cold-shelf') if (p/'runs'/pin['run_id']).is_dir()]
    require(len(candidates)==1 and sha(candidates[0]/'manifest.json')==pin['manifest_sha256'],'Original source receipt differs')
    return candidates[0]


def main(a):
    store,project,assets=a.store.resolve(),a.project.resolve(),a.assets.resolve()
    if a.action=='register':
        cat=load(assets/CAT)
        key=sha(assets/CAT)
        row=None
    else:
        cat,key,row=family_info(store,assets,a.family)
    base=store/'stage3'/key
    sel=base/'registration.json' if row is None else base/'references'/(a.family+'.json')
    sel.parent.mkdir(parents=True,exist_ok=True)
    with sel.with_suffix('.lock').open('a+b') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        if sel.exists():
            old=load(sel)
            require(old['catalog_sha256']==key,'Existing registration catalog differs')
            pins=[old['registration']] if row is None else [old['reference'],old['readback']]
            for pin in pins: located(store,pin)
            return old
        cfg={'dataset_id':cat['dataset_id'],'dataset_hash':key,'camera_set_id':'stage3-four-source-six-target-256-v1','camera_set_hash':cat['camera_sha256'],
             'experiment_id':'stage3-reference-registration' if row is None else 'stage3-native-reference',
             'trial_id':'stage3-reference-registration-v1' if row is None else a.family+'-reference','recording':{'expected_output_bytes':200000 if row is None else 70000000}}
        with tempfile.TemporaryDirectory(dir=store/'work',prefix='stage3-native-') as tmp:
            src=Path(tmp)/'engine'
            engine_copy(project,assets,src,row is None)
            if row is None:
                paths={'catalog':assets/CAT,'study':assets/cat['preregistration_path'],'settings':assets/cat['reference_settings_path'],
                       'cameras':assets/cat['camera_path'],'recipes':assets/'configs/c-reference-v1/recipes','budget':assets/OPT}
                worker=['{python}','{source}/scripts/stage3_registration_worker.py','--out','{out}']
                for n in paths: worker+=['--'+n,'{input:'+n+'}']
                rp=perform(store,project,src,cfg,'archive_verify',{k:(v,None) for k,v in paths.items()},{},worker)
                final={'schema':'el-stage3-registration-selection-1','catalog_sha256':key,'registration':rp,
                       'registration_sha256':sha(located(store,rp)/'payload/registration.json')}
            else:
                rp=load(base/'registration.json')['registration']
                registered=located(store,rp)/'payload'
                require(sha(registered/'catalog.json')==key and sha(registered/'recipes'/(a.family+'.json'))==row['recipe_sha256'],'Selected registration differs')
                cfg.update(scene_family_id=a.family,scene_version='v001',scene_id=row['scene_id'],split=row['split'],scene_hash=row['recipe_sha256'],
                           condition_id='shared-reference',registered_reference=rp)
                python=a.blender_python.absolute()
                entry='{source}/src/evidence_lens/stage3_native.py'
                files={'settings':(registered/'settings.json','registration'),'cameras':(registered/'cameras.json','registration'),
                       'recipe':(registered/'recipes'/(a.family+'.json'),'registration'),'preregistration':(registered/'registration.json','registration')}
                worker=[str(python),entry,'produce','--out','{out}','--config','{config}']
                for n in ('settings','cameras','recipe'): worker+=['--'+n,'{input:'+n+'}']
                stage=base/'references'/('partial-'+a.family+'.json')
                if stage.exists():
                    partial=load(stage)
                    require(partial['registration']==rp and partial['catalog_sha256']==key,'Different prior reference stage')
                    ref=partial['reference']
                else:
                    ref=perform(store,project,src,cfg,'scene_render',files,{'registration':rp},worker)
                    publish(stage,encoded({'registration':rp,'reference':ref,'catalog_sha256':key}))
                payload=located(store,ref)/'payload'
                rb=perform(store,project,src,cfg,'archive_verify',{'reference':(payload,'reference')},{'reference':ref},
                           [str(python),entry,'readback','--reference','{input:reference}','--out','{out}'])
                proof=load(located(store,rb)/'payload/readback.json')
                require(proof['passed'] and proof['reference_sha256']==sha(payload/'reference.json'),'Verification does not bind original bytes')
                final={'schema':'el-stage3-reference-selection-1','catalog_sha256':key,'family':a.family,'split':row['split'],
                       'registration':rp,'reference':ref,'readback':rb,'reference_sha256':sha(payload/'reference.json'),
                       'readback_sha256':sha(located(store,rb)/'payload/readback.json'),'camera_sha256':sha(payload/'cameras.json')}
        publish(sel,encoded(final))
        return final


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('register','reference'))
    for n in ('project','store','assets'): p.add_argument('--'+n,required=True,type=Path)
    p.add_argument('--family')
    p.add_argument('--blender-python',type=Path,default=Path('.venv-reference/bin/python'))
    print(json.dumps(main(p.parse_args()),sort_keys=True))
