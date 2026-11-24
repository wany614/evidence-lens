"""Seal narrow T0 transport files and their independently verifiable historical provenance."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import zipfile


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def spec(p):
    return {'bytes': p.stat().st_size, 'sha256': digest(p)}


def load(p):
    return json.loads(p.read_bytes())


def encoded(d):
    return (json.dumps(d, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


def unzip_worker(path, intended):
    check(digest(path) == intended == 'aa65bc5a412cdafdac2e9ffa5f64f675ba8253904100dafbc086284b5a9632c1', 'Changed original Windows worker')
    with zipfile.ZipFile(path) as z:
        doc = json.loads(z.read('SOURCE-MANIFEST.json'))
        check(doc['tree_sha256'] == '02fe9a25281a72b43a4eb038b37d0a2c5b6e291175f0ac0b1c8c0b5761247e3c' and
              sorted(z.namelist()) == sorted([*doc['files'], 'SOURCE-MANIFEST.json']), 'Original worker roster')
        for name, row in doc['files'].items():
            data = z.read(name)
            check(row['bytes'] == len(data) and hashlib.sha256(data).hexdigest() == row['sha256'], 'Original worker file ' + name)
    return doc['tree_sha256']


def bridge(cfg, inputs, out):
    origins = inputs['origins']
    for role, pin in cfg['origins'].items():
        root = origins / 'proofs' / pin['run_id']
        manifest = load(root / 'manifest.json')
        rows = {r['path']: r for r in manifest['files']}
        run = load(root / 'run.json')
        check(digest(root / 'manifest.json') == load(root / 'completion.json')['manifest_sha256'] == pin['manifest_sha256']
              and run['run_id'] == pin['run_id'] and run['execution_status'] == 'succeeded'
              and digest(root / 'run.json') == rows['run.json']['sha256'], 'Historical seal differs')
        original = origins / 'members' / role
        check(spec(original) == {k: rows[pin['member']][k] for k in ('bytes','sha256')} and digest(original) == pin['sha256'], 'Historical member differs: ' + role)
        dest = out / role
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, dest)
    joined = load(out / 'joined.json')
    evidence = []
    for row in joined['cells']:
        gp = row['original_runs']['generation']
        role = 'historical/' + gp['run_id'] + '.json'
        parent = cfg['origins'][role]
        g = load(out / role)
        check(gp['run_id'] == g['run_id'] == parent['run_id'] and gp['manifest_sha256'] == parent['manifest_sha256'], 'Wrong historic optimizer')
        evidence.append({'run_id': g['run_id'], 'manifest_sha256': parent['manifest_sha256'], 'object_id': row['object_id'],
                         'condition':row['condition'], 'recipe_sha256':g['identity']['recipe_sha256'], 'actual_method':g['actual_method'],
                         'initial_points':g['checkpoints']['0']['gaussians'], 'completed_steps':g['last_completed_step']})
    check(len(evidence) == 8 and len({r['run_id'] for r in evidence}) == 8 and
          all(r['completed_steps'] == 5000 for r in evidence), 'Incomplete R2 history')
    freeze = load(out / 'old-freeze.json')
    check(freeze['recipe_sha256'] == digest(out / 'recipe.json') and freeze['stop_step'] == freeze['schedule_steps'] == 5000 and
          load(out / 'recipe.json')['topology']['cap'] == 20000, 'Incorrect frozen recipe')
    tree = unzip_worker(out / 'worker.zip', freeze['worker_source']['sha256'])
    check(load(out / 'depth-probe.json')['passed'] and load(out / 'resource-probe.json')['passed'], 'Original ED evidence failed')
    doc = {'schema':'el-t0-recovery-bridge-1','run_id':os.environ['EL_RUN_ID'],'origins':cfg['origins'],
           'prior_fits':evidence,'completed_final_recipe_fallback':sum(r['actual_method']=='endpoint_stratified_inverse_depth' and
           r['initial_points']==2048 and r['recipe_sha256']==freeze['recipe_sha256'] for r in evidence),
           'worker_tree_sha256':tree,'recipe_sha256':freeze['recipe_sha256'],'original_target_photos_read':0}
    (out/'bridge.json').write_bytes(encoded(doc))


def write_zip(path, files):
    with zipfile.ZipFile(path,'x') as archive:
        for name, data in sorted(files.items()):
            info=zipfile.ZipInfo(name, (1980,1,1,0,0,0));info.create_system=3;info.external_attr=0o100644<<16
            archive.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=5)


def dispatch(cfg, inputs, out):
    ids=cfg['generator_source_ids']; obj=cfg['scene_family_id']; cond=cfg['condition_id']; trial=cfg['scientific_trial_id']
    init=load(inputs['initmeta']); rc=load(inputs['rendercameras']); sc=load(inputs['cameras']); recipe=load(inputs['recipe'])
    check(ids==({'M2':['s00','s03'],'M4':['s00','s01','s02','s03']}[cond]) and set(sc['views'])==set(ids)
          and set(rc['views'])==set(ids)|{f'v{i:02d}' for i in range(1,7)} and all(sc['views'][s]==rc['views'][s] for s in ids),'Incorrect condition cameras')
    check(init['run_id']==cfg['initialization']['run_id'] and init['object_id']==obj and init['condition']==cond and
          init['source_camera_sha256']==digest(inputs['cameras']) and init['init_npz_sha256']==digest(inputs['init']) and
          init['source_rgb_sha256']=={s:digest(inputs[s]) for s in ids},'Initialization origin differs')
    forced=cfg['forced_fallback_fixture']; eligible=init['eligibility']['qualified']
    check((not forced or (obj=='set1' and cond=='M2' and init.get('forced_fallback_fixture') is True)) and
          (init['actual_method']=='known_pose_sift_triangulation' if eligible and not forced else
           init['actual_method']=='endpoint_stratified_inverse_depth' and init['N']==2048 and init['fallback']['total']==2048),'Initializer decision differs')
    check(digest(inputs['recipe'])=='57730c706aaaa2f88c7f39ff179ba04ff971f31e599bf3409a3d545c61d72c98' and
          recipe['stop_step']==5000 and recipe['topology']['cap']==20000 and cfg['requested_render_steps']==[5000], 'Wrong T0 schedule')
    tree=unzip_worker(inputs['worker'],cfg['worker_sha256']); parent_of=cfg['input_origin_ids']
    members={'init.npz':'init','init.json':'initmeta','source_cameras.json':'cameras','render_cameras.json':'rendercameras',**{'sources/'+s+'.png':s for s in ids}}
    files={n:spec(inputs[k]) for n,k in members.items()}
    job={'schema':'el-recovery-cell-1','purpose':'engineering' if forced else 'formal','identity':{'scene_family_id':obj,'scene_id':obj,'condition_id':cond,'split':'engineering' if forced else 'test'},
         'object_id':obj,'condition':cond,'trial_id':trial,'experiment_id':cfg['windows_experiment'],'stop_step':5000,
         'recipe_sha256':digest(inputs['recipe']),'parents':cfg['windows_parents'],'input_parents':{**{k:parent_of[v] for k,v in members.items()},'recipe':parent_of['recipe']},'files':files}
    jobbytes=encoded(job); (out/'job.json').write_bytes(jobbytes)
    packed={'cell/'+n:inputs[k].read_bytes() for n,k in members.items()}
    packed.update({'cell/job.json':jobbytes,'recipe.json':inputs['recipe'].read_bytes(),'worker.zip':inputs['worker'].read_bytes()})
    ready={'schema':'el-t0-worker-ticket-1','run_id':os.environ['EL_RUN_ID'],'trial_id':trial,'object_id':obj,'condition':cond,
           'job_sha256':hashlib.sha256(jobbytes).hexdigest(),'worker_tree_sha256':tree,'forced_fallback_fixture':forced,
           'initialization':cfg['initialization'],'source':cfg['source'],'freeze':cfg.get('freeze'),'D_qualification':cfg.get('D_qualification'),
           'actual_method':init['actual_method'],'initial_points':init['N'],'source_eligibility':init['eligibility'],
           'render_steps':[5000], 'target_reference_members':0,'members':{n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in packed.items()}}
    packed['READY.json']=encoded(ready);(out/'READY.json').write_bytes(packed['READY.json'])
    write_zip(out/'dispatch.zip',packed)
    shutil.copyfile(Path(__file__).with_name('t0_worker_run.py'),out/'t0_worker_run.py')
    (out/'ticket.json').write_bytes(encoded({'schema':'el-t0-transport-ticket-1','run_id':os.environ['EL_RUN_ID'],
        'trial_id':trial,'package':spec(out/'dispatch.zip'),'launcher':spec(out/'t0_worker_run.py'),'job_sha256':ready['job_sha256'],'worker_sha256':cfg['worker_sha256']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=('bridge','dispatch'))
    for k in ('config','out'):parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--input',action='append',nargs=2,required=True)
    a=parser.parse_args(); (bridge if a.operation=='bridge' else dispatch)(load(a.config),{k:Path(p) for k,p in a.input},a.out)
