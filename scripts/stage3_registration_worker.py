"""Verify and preserve the C roster, authored recipes, source calibration, and fixed pilot budget."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

def hashfile(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def execute(a):
    out=a.out
    roster=json.loads(a.catalog.read_text())
    rows=roster['families']
    if [r['split'] for r in rows]!=['train']*8+['development']*4+['test']*4:
        raise ValueError('Unregistered split sequence')
    if len({r['pair_structure_sha256'] for r in rows})!=len({r['geometry_fingerprint'] for r in rows}) or len({r['pair_structure_sha256'] for r in rows})!=16:
        raise ValueError('Duplicate world fingerprints')
    for tag,pin,name in [('study','preregistration_sha256','reference-study.json'),('cameras','camera_sha256','cameras.json'),('settings','reference_settings_sha256','settings.json')]:
        src=getattr(a,tag)
        if hashfile(src)!=roster[pin]:
            raise ValueError('Registration checksum differs '+tag)
        shutil.copyfile(src,out/name)
    (out/'recipes').mkdir()
    for r in rows:
        f=a.recipes/Path(r['recipe_path']).name
        if hashfile(f)!=r['recipe_sha256']:
            raise ValueError('Changed scene recipe')
        shutil.copyfile(f,out/'recipes'/f.name)
    shutil.copyfile(a.catalog,out/'catalog.json')
    shutil.copyfile(a.budget,out/'optimizer.json')
    record={'schema':'el-stage3-reference-registration-1','run_id':os.environ['EL_RUN_ID'],'catalog_sha256':hashfile(a.catalog),
            'camera_sha256':hashfile(a.cameras),'pilot_optimizer_sha256':hashfile(a.budget),'recipe_count':len(rows),
            'pilot_families':roster['pilot_families'],'test_families':roster['test_families'],'research_images_generated':0,
            'members':{p.relative_to(out).as_posix():hashfile(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    (out/'registration.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('out','catalog','study','cameras','recipes','settings','budget'):p.add_argument('--'+n,required=True,type=Path)
    execute(p.parse_args())
