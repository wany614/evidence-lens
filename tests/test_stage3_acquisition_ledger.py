"""Small fabricated views exercise paired reconstruction denominators and missing hardware counters."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from evidence_lens._bundle import encoded,sha
from stage3_acquisition_costs import telemetry
import stage3_reconstruction_ledger as ledger


def test_missing_driver_samples_remain_unavailable(tmp_path):
    counter=tmp_path/'telemetry.csv'
    with counter.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['gpu_process_memory_bytes','missing_reason','tree_rss_bytes']);w.writerow(['',json.dumps({'gpu_process_memory_bytes':'exit_255'}),123])
    fields=telemetry(counter)
    assert fields['gpu_process_sample_count']==0 and fields['gpu_process_sampled_peak_bytes'] is None
    assert fields['sampled_rss_peak']==123 and fields['gpu_counter_missing_reasons']=={'exit_255':1}


def test_true_common_rgb_and_depth_domains_and_missing_prediction(tmp_path,monkeypatch):
    monkeypatch.setattr(ledger,'located',lambda store,pin:store/pin['run_id'])
    ref=tmp_path/'reference/payload';(ref/'reference').mkdir(parents=True);meta={'views':{}}
    size=(1,3);eye=np.ones(size,np.float32);rgb=np.zeros((*size,3),np.float32)
    truth={'rgb':rgb,'z':eye,'position':np.ones((*size,3),np.float32),'edge':np.zeros(size,bool),**{'visibility_'+s:np.ones(size,np.uint8) for s in ledger.SOURCES['M4']}}
    records={};pins={'run_id':'reference','manifest_sha256':'a'*64}
    for cond in ledger.SOURCES:
        cell={'selection':{'reference':pins,'render':{'run_id':cond},'reconstruction':{'run_id':cond+'-scene'}}};records[cond]=cell
        p=tmp_path/cond/'payload';p.mkdir(parents=True);m={'views':{},'run_id':cond,'reconstruction_run_id':cond+'-scene','depth_verified':True}
        for v in ledger.TARGETS:
            original=ref/'reference'/(v+'.npz');np.savez(original,**truth);meta['views'][v]={'npz':{'sha256':sha(original)}}
            predict=rgb.copy();predict[...,0,:]=np.float32(.05);predict[...,1,:]=.2;predict[...,2,:]=.4
            alpha=eye.copy();dp=eye.copy();dp+=.1
            if cond=='M2':predict[...,2,:]=np.nan;alpha[...,0]=.3
            else:predict[...,0,:]=np.nan;alpha[...,1]=.4
            f=p/(v+'.npz');np.savez(f,rgb=predict,alpha=alpha,z=dp);m['views'][v]={'sha256':sha(f)}
        (p/'prediction.json').write_bytes(encoded(m))
    (ref/'reference.json').write_bytes(encoded(meta))
    result=ledger.family_quality(tmp_path,records)
    assert result['own']['M2']['E']==12 and result['own']['M2']['Y']==6
    assert result['paired']['M2']['E']==result['paired']['M4']['E']==6
    assert result['paired']['M2']['Z']==result['paired']['M4']['Z']==6
    assert result['M4_minus_M2']['rgb_mae']==0
    del records['M4']['selection']['render'];no=ledger.family_quality(tmp_path,records)
    assert no['own']['M4']['V']==18 and no['own']['M4']['E']==0
    assert no['paired']['M2']['E']==0 and no['M4_minus_M2']['depth_mae_m'] is None


def test_confirmation_launcher_stops_before_any_real_science(tmp_path,monkeypatch):
    import argparse
    import pytest
    import stage3_open_references as launcher
    def deny(*args,**kwargs):raise ValueError('fixture has no sealed model')
    def unexpected(*args,**kwargs):pytest.fail('touched reference or image before actual opening')
    monkeypatch.setattr(launcher,'require_opening',deny);monkeypatch.setattr(launcher,'native_reference',unexpected);monkeypatch.setattr(launcher,'build_batch',unexpected)
    with pytest.raises(ValueError,match='fixture has no sealed model'):
        launcher.acquire(argparse.Namespace(project=ROOT,assets=ROOT,store=tmp_path,part=1,opening_sha256='1'*64,blender_python=Path('unused')))
    assert list(tmp_path.iterdir())==[]
