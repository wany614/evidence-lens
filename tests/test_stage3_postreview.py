"""Readback regression fixtures use synthetic calibration, queue records and a minimal Blender DNA block."""
import json
from pathlib import Path
import struct
import sys
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from evidence_lens._bundle import encoded
from stage3_intake import published_attempts,retry_predecessor_pending
from stage3_postreview import cameras_and_distances
from stage3_public_blend import decode,portable_copy


def camera_fixture(folder,change):
    rig=json.loads((ROOT/'configs/stage3-cameras-v1.json').read_text())
    c='M2';ids=['s00','s03'];targets=[f'v{i:02d}' for i in range(1,7)]
    if change in ('principal','skew','clip','row'):
        for v in rig['views'].values():
            if change=='principal':v['K'][0][2]=128
            if change=='skew':v['K'][0][1]=.125
            if change=='clip':v['clip_m'][0]=.051
            if change=='row':v['K'][2][0]=.125
    out=folder/'payload';p=out/'condition'/c;p.mkdir(parents=True)
    (out/'cameras.json').write_bytes(encoded(rig))
    for key,names in [('source',ids),('render',ids+targets)]:
        subset={**rig,'views':{s:rig['views'][s] for s in names}}
        if change=='source' and key=='source':subset['views']['s03']['c2w'][0][3]+=.01
        (p/(key+'_cameras.json')).write_bytes(encoded(subset))
    return c,ids,json.loads((ROOT/'protocol/stage3-optimizer-v1.json').read_text())


@pytest.mark.parametrize('change',('principal','skew','clip','row','source'))
def test_bad_source_camera_rejected(tmp_path,change):
    with pytest.raises(ValueError):
        cameras_and_distances(tmp_path,*camera_fixture(tmp_path,change))


def test_measured_nearest_source(tmp_path):
    _,_,measured=cameras_and_distances(tmp_path,*camera_fixture(tmp_path,None))
    assert set(measured)=={f'v{i:02d}' for i in range(1,7)}
    assert measured['v05']['distance_m']==pytest.approx(.34713111,abs=1e-7)


def test_durable_dispatch_catches_late_results_and_waits_for_retry(tmp_path):
    doc={'attempts':[{'run_id':'rid','manifest_sha256':'m','status':'succeeded','stage':'render','return_sha256':'z'}]}
    er={'run':'rid','manifest_sha256':'m','status':'succeeded','sha256':'z'}
    published_attempts(doc,{},'job')
    published_attempts(doc,{('job','render'):[er]},'job')
    with pytest.raises(ValueError,match='after selection'):
        published_attempts(doc,{('job','render'):[er,{**er,'run':'other'}]},'job')
    with pytest.raises(ValueError,match='changed'):
        published_attempts(doc,{('job','render'):[{**er,'sha256':'different'}]},'job')
    rid='20260101T010101010101Z-0000000000000000'
    retry=tmp_path/'retry.zip'
    with zipfile.ZipFile(retry,'w') as z:z.writestr('run.json',json.dumps({'retry_of_run_id':rid}))
    assert retry_predecessor_pending(retry,tmp_path/'store')
    prior=tmp_path/'cold-shelf/runs'/rid;prior.mkdir(parents=True)
    (prior/'manifest.json').write_text('{}')
    assert not retry_predecessor_pending(retry,tmp_path/'store')


def native_fixture():
    def fields(tag,names):
        b=tag+struct.pack('<I',len(names))+b''.join(n+b'\0' for n in names)
        return b+b'\0'*((-len(b))%4)
    dna=(b'SDNA'+fields(b'NAME',[b'title[96]',b'dir[1090]'])+fields(b'TYPE',[b'char',b'FileSelectParams'])+
         b'TLEN'+struct.pack('<HH',1,2088)+b'STRC'+struct.pack('<I6H',1,1,2,0,0,0,1))
    def block(kind,data):return struct.pack('<4siQii',kind,len(data),0,0,1)+data
    contents=bytearray(2088)
    contents[:9]=b'file-menu'
    contents[96:109]=b'/home/fixture'
    contents[130]=135
    contents[1800]=170
    return b'BLENDER-v405'+block(b'DATA',contents)+block(b'DNA1',dna)+block(b'ENDB',b'')


def test_blender_ui_directory_cleared_and_other_bytes_preserved(tmp_path):
    orig=tmp_path/'local.blend';public=tmp_path/'public.blend'
    original=native_fixture()
    orig.write_bytes(original)
    proof=portable_copy(orig,public)
    assert proof['cleared_fields']==1 and proof['changed_byte_count']==14
    after=decode(public);start=12+24+96
    assert after[:start]==original[:start] and after[start+1090:]==original[start+1090:]
    assert after[start:start+1090]==bytes(1090)
