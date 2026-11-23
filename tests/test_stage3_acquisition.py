"""Checks for the C source boundary and numeric interfaces, without instantiating any registered test world."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import zipfile

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from stage3_reference import family_info
from stage3_jobs import build_job,canary,capture,signature
from make_handoff import encoded,zip_bytes
from stage3_windows import job_load
from stage3_budget import compare


def local_module():
    path=ROOT/'src/evidence_lens/stage3_gaussians.py'
    loader=importlib.util.spec_from_file_location('el_test_stage3_joint',path)
    module=importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def input_fixture(tmp_path):
    full=json.loads((ROOT/'configs/stage3-cameras-v1.json').read_text())
    rgb=tmp_path/'sources';rgb.mkdir()
    for i in range(4):(rgb/f's{i:02d}.png').write_bytes(b'allowed source byte '+bytes([i]))
    for c,ss in [('M2',['s00','s03']),('M4',['s00','s01','s02','s03'])]:
        base=tmp_path/'condition'/c;base.mkdir(parents=True)
        for role,ids in [('source_cameras',ss),('render_cameras',ss+[f'v{i:02d}' for i in range(1,7)])]:
            (base/(role+'.json')).write_bytes(encoded({**full,'views':{s:full['views'][s] for s in ids}}))
    return tmp_path


def test_roster_and_opening_guard_precede_any_test_read(tmp_path,monkeypatch):
    roster=json.loads((ROOT/'protocol/stage3-families-v1.json').read_text())
    assert [r['split'] for r in roster['families']]==['train']*8+['development']*4+['test']*4
    def reject(*args,**kwargs):raise PermissionError('test has not been opened')
    monkeypatch.setitem(sys.modules,'evidence_lens.stage3_gate',types.SimpleNamespace(require_opening=reject))
    with pytest.raises(PermissionError,match='not been opened'):
        family_info(tmp_path,ROOT,roster['test_families'][0])
    assert family_info(tmp_path,ROOT,roster['pilot_families'][0])[2]['pilot']


def test_fixed_source_and_target_cameras():
    g=local_module()
    camera=g.read(ROOT/'configs/stage3-cameras-v1.json')
    groups=list(g.SETS['M4'])+list(g.TARGETS)
    views=g.valid_cameras(camera,groups)
    centers={s:np.array(views[s]['c2w'])[:3,3] for s in groups}
    flex=centers['s03']-centers['s00']
    assert np.linalg.norm(flex)==pytest.approx(.72,abs=1e-6)
    for i,s in enumerate(g.SETS['M4']):np.testing.assert_allclose(centers[s],centers['s00']+i*flex/3,atol=1e-6)
    assert len({tuple(x) for x in centers.values()})==len(groups)
    assert all(np.array(c['K'])[0,2]==127.5 and c['K_edge'][0][2]==128 for c in views.values())
    with pytest.raises(ValueError):g.valid_cameras(camera,g.SETS['M2'])


def test_identical_multiview_initialization_and_budget(tmp_path):
    g=local_module()
    b=g.read(ROOT/'protocol/stage3-optimizer-v1.json')
    init=g.initial_values(b['initialization'])
    assert list(init)==list(g.PARAMS) and init['means'].shape==(4096,3)
    for name in ('M2','M4'):
        np.savez(tmp_path/(name+'.npz'),**g.initial_values(b['initialization']))
        for k,v in g.initial_values(b['initialization']).items():assert np.array_equal(v,init[k])
    assert (tmp_path/'M2.npz').read_bytes()==(tmp_path/'M4.npz').read_bytes()
    assert g.digest_values(init)==g.digest_values(g.initial_values(b['initialization']))
    for steps in (64,128,256):
        assert sum(steps//2*256**2 for _ in g.SETS['M2'])==sum(steps//4*256**2 for _ in g.SETS['M4'])


def test_truth_removal_and_unexpected_source_rejection(tmp_path):
    folder=tmp_path/'capture'
    folder.mkdir()
    pay=input_fixture(folder)
    cells={}
    for c,names in [('M2',['s00','s03']),('M4',['s00','s01','s02','s03'])]:
        job={'schema':'el-stage3-worker-job-1','condition':c,'generator_source_ids':names,'trial_id':'fixture-'+c}
        b,jsha=build_job(capture(pay,c),b'{}',job)
        path=tmp_path/(c+'.zip');path.write_bytes(b)
        case={**signature(b),'job_sha256':jsha,'trial_id':'fixture-'+c}
        assert job_load(path,tmp_path/('extract-'+c),case)['generator_source_ids']==names
        cells[c]=(b'{}',job,signature(b)['sha256'])
        if c=='M2':
            with zipfile.ZipFile(path) as z: contents={n:z.read(n) for n in z.namelist()}
            contents['sources/s01.png']=b'other condition'
            bad=zip_bytes(contents);wrong=tmp_path/'wrong.zip';wrong.write_bytes(bad)
            with pytest.raises(Exception,match='Undeclared source'):
                job_load(wrong,tmp_path/'bad',{**case,**signature(bad)})
    proof=canary(pay,cells)
    assert set(proof['shared_endpoint_checks'])=={'s00','s03'}
    assert all(x['removed']==x['substituted'] for x in proof['identical_job_when_external_truth'].values())


def test_repeat_diagnostic_keeps_missing_and_finite_support():
    a=np.array([1,np.nan,3],np.float32);b=np.array([1,2,3.25],np.float32)
    q=compare(a,b)
    assert q['validity_disagreements']==1 and q['paired_finite']==2 and q['max_abs']==.25 and not q['bitwise_equal']
