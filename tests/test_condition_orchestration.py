"""Isolated recorder-backed six-pixel transport, stress, independent check and selection."""

import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from evidence_lens import archive, recorder
from evidence_lens._bundle import BundleError, encoded, load, save, sha

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import condition_jobs as jobs
import condition_mac as mac
import condition_stage as worker
import make_handoff
from formal_lineage import VerifiedRuns
from test_stress_conditions import small_ply
from test_trial_selection import build_frozen_fixture


def test_q1_mock_transport_real_derivation_cpu_science_and_timing_isolation(tmp_path, monkeypatch, capsys):
    scientific_ply = tmp_path / 'scientific-fixture.ply'
    small_ply(scientific_ply, 0.1234567)
    store, b0_path = build_frozen_fixture(tmp_path, monkeypatch, capsys, scientific_ply)
    b0 = load(b0_path)
    directory = store / 'formal' / ('5'*64)
    directory.mkdir(parents=True)
    baseline_worker = make_handoff.zip_bytes(make_handoff.source_files(formal=True))
    (directory / 'worker-source.zip').write_bytes(baseline_worker)
    save(directory / 'worker-source.json', {'sha256':sha(directory / 'worker-source.zip')})
    ids = {k:v['run_id'] for k,v in b0['selected'].items()}
    family = {'scene_family_id':b0['scene_family_id'],'scene_id':b0['scene_id'],'scene_version':'v001','trial_id':b0['trial_id'],
              'split':'test','dataset_id':'fixture','dataset_hash':'5'*64}
    method = tmp_path / 'method.json'; save(method, {'synthetic':'method release fixture'})
    ctx = {'store':store,'family':family,'condition':'Q1','quality_trial':'fixture-Q1','baseline':b0,'b0_ids':ids,
           'b0_selection_sha256':sha(b0_path),'verified':VerifiedRuns(store),'method_file':method,
           'release':{'method_freeze_sha256':sha(method)},'first_test':family['scene_family_id']}
    container = tmp_path / 'condition.zip'
    export = jobs.make_job(ctx, None, None, container)
    handoff = tmp_path / 'worker'
    with zipfile.ZipFile(container) as z:
        assert {n for n in z.namelist() if not n.startswith('code/')} == {'handoff.json','input.png','cameras.json'}
        assert 'code/scripts/condition_stage.py' in z.namelist() and not any('blender_reference' in n for n in z.namelist())
        z.extractall(handoff)
    job, digest = worker.read_job(handoff)
    subprocess.run([sys.executable, '-B', str(handoff/'code/scripts/condition_stage.py'), '--help'],
                   cwd=handoff, check=True, capture_output=True)
    assert export['primary'] == 'derive' and job['baseline_generation']['run_id'] == ids['generation']
    with pytest.raises(FileNotFoundError):
        jobs.make_job(ctx, None, 1, tmp_path/'early-timing.zip')
    cfg = tmp_path / 'worker-config.json'
    settings = {**load(ROOT/'configs/windows-b0.json'), **job['identity'], 'formal_handoff_sha256':digest,
                'worker_source_sha256':job['worker_source_sha256'],'timing_repeat':None,
                'method_freeze_sha256':sha(method),'b0_selection_sha256':sha(b0_path)}
    save(cfg, settings)
    def seal(kind, parents, inputs, command, trial=None, experiment=jobs.QUALITY):
        args = ['--root',str(store),'--source-root',str(ROOT),'--config',str(cfg),'--kind',kind,'--trial',trial or ctx['quality_trial'],'--experiment',experiment]
        for rid in parents:
            args += ['--parent',rid]
        for name,(rid, filename) in inputs.items():
            args += ['--input',f'{name}={store/"runs"/rid/"payload"/filename}']
        result = recorder.main([*args, '--', '{python}', *command]); captured = capsys.readouterr()
        assert result == 0, (captured.out,captured.err)
        rid = json.loads(captured.out.strip().splitlines()[-1])['run_id']
        archive.receive(store/'runs'/rid, store)
        return rid
    bgen, ref = ids['generation'], ids['reference']
    derived = seal('derivation',[bgen],{'ply':(bgen,'scene.ply'),'generation':(bgen,'generation.json')},
                   ['{source}/scripts/stress_condition.py','Q1','--ply','{input:ply}','--generation','{input:generation}','--out','{out}'])
    real_derived = load(store/'runs'/derived/'payload/generation.json')
    assert real_derived['forward_calls_in_this_stage'] == 0
    # This simulation intentionally uses a copied tiny unit render. A real Windows
    # adapter is required for research data; here only the transport/lineage is tested.
    code = ('import sys,shutil,os,json,pathlib;out=pathlib.Path(sys.argv[1]);'
            'shutil.copytree(sys.argv[2],out,dirs_exist_ok=True,copy_function=shutil.copyfile);m=json.loads((out/"prediction.json").read_text());'
            'm.update(producer_run_id=os.environ["EL_RUN_ID"],generation_run_id=sys.argv[3],ply_sha256=sys.argv[4],phases={});'
            '(out/"prediction.json").write_text(json.dumps(m))')
    render = seal('render',[ref,derived],{'rgb':(ref,'input.png'),'cameras':(ref,'cameras.json'),'ply':(derived,'scene.ply'),'generation':(derived,'generation.json')},
                  ['-c',code,'{out}',str(store/'runs'/ids['render']/'payload'),derived,real_derived['ply_sha256']])
    args = SimpleNamespace(stage='score',producer_run=derived,render_run=render,preparation_run=None,score_run=None,evaluation_run=None,recompute_run=None,retry_of=None,cpu_python=None)
    args.score_run = mac.run_stage(ctx, args)['run_id']
    args.stage='evaluate'; args.evaluation_run = mac.run_stage(ctx, args)['run_id']
    args.stage='verify'; args.recompute_run = mac.run_stage(ctx, args)['run_id']
    args.stage='select'; selection = mac.run_stage(ctx,args)
    assert mac.run_stage(ctx,args)['status'] == 'identical'
    record = load(Path(selection['selection']))
    assert record['selected']['preparation']['run_id'] == derived and record['selected']['generation']['run_id'] == bgen
    assert record['sources']['input_png']['run_id'] == ref and record['evidence']['q1_field_check']['vertices'] == 3
    assert record['evidence']['q1_field_check']['struct_binary16_values_checked'] == 9
    assert record['evidence']['target_denominators']['v01']['rgb_evaluable'] == 0
    timing = jobs.make_job(ctx,None,1,tmp_path/'timed.zip')
    assert timing['trial_id'] == 'fixture-Q1-timing-01' and timing['primary'] == 'derive'
    with zipfile.ZipFile(tmp_path/'timed.zip') as z:
        timed_job = json.loads(z.read('handoff.json'))
    timed_id = timing['trial_id']
    save(cfg, dict(settings, **{'trial_id':timed_id,'timing_repeat':1,'warmup':timed_job['warmup'],
                               'formal_handoff_sha256':timing['handoff_sha256']}))
    tp = seal('derivation',[bgen],{'ply':(bgen,'scene.ply'),'generation':(bgen,'generation.json')},
              ['{source}/scripts/stress_condition.py','Q1','--ply','{input:ply}','--generation','{input:generation}','--out','{out}'], timed_id,jobs.TIMING)
    tr = seal('render',[ref,tp],{'rgb':(ref,'input.png'),'cameras':(ref,'cameras.json'),'ply':(tp,'scene.ply'),'generation':(tp,'generation.json')},
              ['-c',code,'{out}',str(store/'runs'/ids['render']/'payload'),tp,real_derived['ply_sha256']],timed_id,jobs.TIMING)
    from condition_timing import explicit_timing
    timed = explicit_timing(ctx,1,None,tp,tr)
    assert load(Path(timed['timing_receipt']))['included_in_quality_or_training'] is False
    assert not (store/'trials'/timed_id/'trial_selection.json').exists()
    with pytest.raises(BundleError, match='different condition dispatch'):
        mac.run_stage(ctx, SimpleNamespace(**dict(vars(args), producer_run=tp, render_run=tr)))
    with pytest.raises(BundleError, match='first test'):
        jobs.make_job(dict(ctx, first_test='different'),None,2,tmp_path/'ineligible.zip')


def test_r1_recorder_derivation_and_worker_contains_only_exact_reduced_source(tmp_path, capsys):
    import numpy as np
    from PIL import Image
    from test_formal_lineage import seal
    from stress_condition import half_source

    store, project = tmp_path / 'store', tmp_path / 'project'
    (project/'scripts').mkdir(parents=True)
    (project/'scripts/stub.py').write_text('# isolated synthetic parent\n')
    family = {'scene_id':'resolution/v001','scene_family_id':'resolution','scene_version':'v001','trial_id':'resolution-B0',
              'condition_id':'B0','split':'test','dataset_id':'fixture','dataset_hash':'b'*64}
    cfg = tmp_path/'config.json'; save(cfg, family)
    original = tmp_path/'original'; original.mkdir()
    yy, xx = np.indices((512,512)); pixels = np.stack([xx,yy,xx+yy],axis=-1).astype('uint8')
    Image.fromarray(pixels).save(original/'input.png')
    view = {'width':512,'height':512,'K':[[400,0,255.5],[0,400,255.5],[0,0,1]],
            'K_edge':[[400,0,256],[0,400,256],[0,0,1]],'c2w':np.eye(4).tolist()}
    cameras = {'schema':'el-camera-1','pixel_center':'integer','views':dict.fromkeys(jobs.VIEWS,view)}
    save(original/'cameras.json', cameras)
    (original/'private.blend').write_bytes(b'unit world')
    save(original/'reference.json', {'schema':'el-reference-1','validation_passed':True,'scene_path':'private.blend',
         'input_sha256':sha(original/'input.png'),'camera_sha256':sha(original/'cameras.json'),'blend_sha256':sha(original/'private.blend'),
         'views':dict.fromkeys(jobs.VIEWS,{})})
    ref = seal(store, project, cfg, 'scene_render', original, capsys)['run_id']
    proof = tmp_path/'readback'; proof.mkdir(); save(proof/'reference_readback.json',{'fixture':True})
    rb = seal(store,project,cfg,'archive_verify',proof,capsys,[ref])['run_id']
    stage = tmp_path/'generation';stage.mkdir();(stage/'scene.ply').write_bytes(b'baseline PLY fixture')
    save(stage/'generation.json', {'source_size_wh':[512,512],'network_input_wh':[1536,1536],'focal_length_px':400.})
    gen = seal(store,project,cfg,'generation',stage,capsys,[ref])['run_id']
    source_dir = store/'formal'/family['dataset_hash'];source_dir.mkdir(parents=True)
    (source_dir/'worker-source.zip').write_bytes(make_handoff.zip_bytes(make_handoff.source_files(formal=True)))
    save(source_dir/'worker-source.json',{'sha256':sha(source_dir/'worker-source.zip')})
    method = tmp_path/'method.json';save(method,{'fixture':True})
    ctx = {'family':family,'store':store,'condition':'R1','quality_trial':'resolution-R1','b0_ids':{'reference':ref,'readback':rb,'generation':gen},
           'b0_selection_sha256':'a'*64,'verified':VerifiedRuns(store),'release':{'method_freeze_sha256':sha(method)},
           'baseline':{'evidence':{'scene_sha256':sha(original/'private.blend')}},'method_file':method,'first_test':family['scene_family_id']}
    prep = mac.prepare(ctx)
    assert mac.prepare(ctx) == prep
    package = tmp_path/'r1.zip'; jobs.make_job(ctx,prep['run_id'],None,package)
    extracted = tmp_path/'r1';
    with zipfile.ZipFile(package) as z:
        z.extractall(extracted)
    job, _ = worker.read_job(extracted)
    assert job['source'] == prep and {n for n in job['files'] if not n.startswith('code/')} == {'input.png','cameras.json'}
    expected, camera = half_source(pixels, cameras)
    with Image.open(extracted/'input.png') as actual:
        assert actual.size == (256,256) and np.array_equal(np.asarray(actual),expected)
    assert json.loads((extracted/'cameras.json').read_text()) == camera
    assert all(camera['views'][v] == cameras['views'][v] for v in jobs.VIEWS if v != 'source')
    from condition_selection import r1_focal_scaling
    save(stage/'generation.json', {'source_size_wh':[256,256],'network_input_wh':[1536,1536],'focal_length_px':200.})
    half_gen = seal(store,project,cfg,'generation',stage,capsys,[prep['run_id']])['run_id']
    geometric = r1_focal_scaling(ctx,{'reference':ref,'preparation':prep['run_id'],'producer':half_gen})
    assert geometric['B0']['effective_internal_focal_px'] == geometric['R1']['effective_internal_focal_px'] == 1200.
    save(stage/'generation.json', {'source_size_wh':[256,256],'network_input_wh':[1536,1536],'focal_length_px':201.})
    wrong = seal(store,project,cfg,'generation',stage,capsys,[prep['run_id']])['run_id']
    with pytest.raises(BundleError, match='focal telemetry'):
        r1_focal_scaling(ctx,{'reference':ref,'preparation':prep['run_id'],'producer':wrong})
    with pytest.raises(BundleError, match='derivation'):
        jobs.make_job(ctx,gen,None,tmp_path/'wrong.zip')
