"""Engineering checks for the separately published, independently evaluated public control."""
import json
import os
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

STAGING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGING/'scripts'))
import vision_control_attachment as control


def named(p):
    return p['run_id']+'='+p['manifest_sha256']


def test_only_explicit_measured_control_records_and_members(tmp_path):
    with pytest.raises(ValueError,match='original public study records'):
        control.material_sources(SimpleNamespace(case_export=named(control.RAW_PIN),reference_export=named(control.REFERENCE_PIN),
                                                evaluation=named(control.RAW_PIN)),tmp_path)
    data=tmp_path/'excess.zip'
    with zipfile.ZipFile(data,'w') as z:z.writestr('research/unselected-view.png',b'other research')
    with pytest.raises(ValueError,match='fixed positive allowlist'):
        control.read_public(data,'a'*64,'b'*40,'c'*64,{})


def test_full_grid_control_replay_includes_unknown_state_and_exact_cutoff(tmp_path):
    a,b=tmp_path/'reference-scores',tmp_path/'clean-scores';a.mkdir();b.mkdir()
    x=np.zeros((512,512),np.float32);x[0]=np.nan;x[1,2]=.5
    ok=np.isfinite(x);state=np.where(ok,x>=.5,255).astype(np.uint8)
    data={'l0_reason':np.zeros(x.shape,np.uint8)}
    for key in control.METHODS.values():data.update({key:x,key+'_valid':ok,key+'_state':state})
    for v in control.VIEWS:
        np.savez_compressed(a/(v+'.npz'),**data);os.link(a/(v+'.npz'),b/(v+'.npz'))
    rows,passed=control.risk_readback(a,b)
    assert passed and sum(r['pixels'] for v in rows.values() for r in v.values())==11010048
    (b/'v02.npz').unlink();key=control.METHODS['L2-vision-ft'];data[key]=x.copy();data[key][1,2]=np.nextafter(np.float32(.5),np.float32(0))
    data[key+'_state']=state.copy();data[key+'_state'][1,2]=0;np.savez_compressed(b/'v02.npz',**data)
    rows,passed=control.risk_readback(a,b)
    assert not passed and rows['v02']['L2-vision-ft']['threshold_state_difference']==1 and rows['v02']['L2-vision-ft']['maximum_absolute_difference']<.0002


def test_authorized_actual_control_replays_all_original_arrays_and_fixed_figures(tmp_path):
    store=os.getenv('EL_PUBLIC_CONTROL_STORE')
    if not store:pytest.skip('set EL_PUBLIC_CONTROL_STORE to the owner store containing the registered public control')
    args=SimpleNamespace(store=Path(store).resolve(),shelf=None,case_export=named(control.CASE_PIN),reference_export=named(control.REFERENCE_PIN),
                         evaluation=named(control.RAW_PIN),output=tmp_path/'validated')
    result=control.materials(args)
    assert result['case_sha256']==control.CASE_INDEX and result['reference_sha256']==control.REFERENCE_INDEX and result['replayed_six_native_views']
    assert result['materials_sha256']=='fd9dc5c3b1e4100e2add0605fb0a04d580f1ee177509c5542302a8494dcc9ab0'
    inventory=json.loads((args.output/'materials.json').read_text());assert set(inventory['files'])==control.SEED_NAMES and len(inventory['staged_scoring_inputs'])==10
    assert all(n.startswith('REFERENCE/') for n in control.REF_NAMES) and control.CASE_NAMES.isdisjoint(control.REF_NAMES)
