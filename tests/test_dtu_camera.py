"""Pure geometry, crop support, exact return inputs and holdout roles."""
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

from evidence_lens.dtu_camera import (SOURCES,TARGETS,camera_outputs,convert_rgb,crop_geometry,decompose,digest,encoded)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from dtu_convert import opening
from dtu_handoff_diagnostic import MAIN_RUNNER,RUNNER,source_variant
from dtu_reconstruct_windows import OPT_SHA,unpack


def test_full_integer_center_crop_and_source_color():
    c=crop_geometry(1600,1200);a=256/1200
    assert (c["left"],c["top"],c["side"])==(200,0,1200)
    principal=np.array([823.2,619.07,1])
    output=np.asarray(c["affine"])@principal
    assert np.allclose(output[:2],[(823.2-200+.5)*a-.5,(619.07+.5)*a-.5])
    assert np.max(abs(output[:2]-127.5))>4
    rgb=np.zeros((1200,1600,3),dtype=np.uint8);rgb[:,:200,0]=240;rgb[:,1400:,1]=220
    rgb[:,200:1400]=[31,57,210]
    b=io.BytesIO();Image.fromarray(rgb).save(b,format="PNG")
    result=convert_rgb(b.getvalue(),c)
    assert np.all(np.asarray(Image.open(io.BytesIO(result)))==[31,57,210])
    assert convert_rgb(result,crop_geometry(256,256))==result
    with pytest.raises(ValueError,match="native three-channel"):
        convert_rgb(result,c)


def test_negative_projective_scale_rotation_units_origin():
    a=.28
    r=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    k=np.array([[1450.,0,817.],[0,1490.,591.],[0,0,1]])
    centre=np.array([17,-193,37.5])
    p=-.013*k@np.column_stack([r,-r@centre])
    inferred,pose=decompose(p,.001,1)
    assert np.allclose(pose[:3,3],centre*.001)
    assert np.allclose(pose[:3,:3],r.T)
    assert np.allclose(inferred[:2,2],[816,590])
    with pytest.raises(ValueError):decompose(p,0,1)
    with pytest.raises(ValueError):decompose(p,.001,2)


def test_known_geometry_no_old_camera_substitution():
    old=json.loads((ROOT/"configs/stage3-cameras-v1.json").read_bytes())
    # Repeated global physical scale and rigid pose changes preserve the source/target layout.
    raw={}; distance_ratio=.55
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    intrinsic=np.array([[2600.,0,825.],[0,2900.,595.],[0,0,1]])
    for i,(_,cam) in enumerate(old["views"].items(),1):
        c=np.array(cam["c2w"])
        centre=(rotation@c[:3,3]*distance_ratio+[1.2,-.7,.1])*1000
        u,_,vt=np.linalg.svd(c[:3,:3])
        view=(rotation@u@vt).T
        raw[i]=intrinsic@np.column_stack([view,-view@centre])
    q=camera_outputs(raw,1600,1200,.001,0,old)
    assert len(set(q["ids"].values()))==10
    assert q["frame"]["endpoint_ids"]==[q["ids"]["s00"],q["ids"]["s03"]]
    assert q["frame"]["additional_scale"]==1
    ratio=q["fit"]["dtu"]["endpoint_distance_m"]/q["fit"]["stage3"]["endpoint_distance_m"]
    assert abs(ratio-distance_ratio)<1e-9
    for n in SOURCES+TARGETS:
        k=np.asarray(q["cameras"]["views"][n]["K"])
        assert np.isclose(k[0,0],2600*256/1200) and not np.isclose(k[0,2],127.5)
        assert q["checks"][n]["max_abs_reprojection_px"]<1e-8


def test_reference_quarantine_roles_and_sha(tmp_path):
    source={"schema":"el-dtu-opening-1","purpose":"source-only reconstruction and frozen score","freeze_sha256":"a"*64,"object_ids":["set6"]}
    target={**source,"schema":"el-dtu-reference-gate-1","purpose":"one independent evaluation of the complete registered matrix"}
    p=tmp_path/"marker"
    with pytest.raises(ValueError):opening("set6",None,None)
    for doc,role,blocked in [(source,"sources","references"),(target,"references","sources")]:
        p.write_bytes(encoded(doc));h=digest(p.read_bytes())
        assert opening("set6",p,h,role)==doc
        with pytest.raises(ValueError):opening("set6",p,"0"*64,role)
        with pytest.raises(ValueError):opening("set6",p,h,blocked)


def test_set1_scoped_diagnostic_does_not_mutate_original_source():
    original=(ROOT/RUNNER).read_bytes()
    assert digest(original)==MAIN_RUNNER
    variant=source_variant(original)
    assert b'j["object_id"]=="set1"' in variant and b'j["optimization_steps"]==1024' in variant
    assert b'"optimization_steps":1024' in variant and b'd1-dtu-debug-only-1024' in variant
    assert (ROOT/RUNNER).read_bytes()==original
    with pytest.raises(Exception):source_variant(original+b"\n")


def test_nested_worker_job_rejects_unauthorized_photo_even_with_matching_outer_sha(tmp_path):
    import hashlib
    import zipfile
    from evidence_lens._bundle import encoded
    from evidence_lens._bundle import sha
    good_names={"source_cameras.json","render_cameras.json","optimization.json","sources/s00.png","sources/s03.png"}
    job={"schema":"el-dtu-job-1","condition":"M2","trial_id":"d1-dtu-set1-M2-256","experiment_id":"d1-dtu-source-only-256",
         "generator_source_ids":["s00","s03"],"optimization_steps":256,"target_ids":list(TARGETS),"object_id":"set1",
         "files":{n:{"bytes":1,"sha256":OPT_SHA} for n in good_names}}
    raw=encoded(job)
    for extra in ("targets/v01.png","sources/s01.png"):
        z=tmp_path/extra.replace("/","_")
        with zipfile.ZipFile(z,"w") as archive:
            archive.writestr("job.json",raw)
            for n in good_names|{extra}:archive.writestr(n,b"G")
        entry={"bytes":z.stat().st_size,"sha256":sha(z),"job_sha256":hashlib.sha256(raw).hexdigest(),"trial_id":job["trial_id"]}
        with pytest.raises(Exception,match="Target image, unauthorized source"):
            unpack(z,tmp_path/"staging",entry)
        assert not (tmp_path/"staging").exists()
