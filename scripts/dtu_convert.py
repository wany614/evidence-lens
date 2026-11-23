"""Convert one explicit source or independent evaluation roster from a sealed DTU download."""
import argparse
import json
import re
import sys
import zlib
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from evidence_lens.dtu_camera import SOURCES,TARGETS,convert_rgb,digest,encoded,record,require,write

PURPOSE="source-only reconstruction and frozen score"


def opening(object_id, path, expected, role="sources"):
    require(object_id in ("set1","set6"),"Unknown registered sample")
    if object_id=="set1":
        require(path is None and expected is None,"Development cannot borrow a test receipt")
        return None
    require(path is not None and re.fullmatch("[0-9a-f]{64}",expected or ""),"Set6 requires the specific immutable method opening")
    raw=path.read_bytes()
    require(digest(raw)==expected,"Opening identity changed")
    obj=json.loads(raw)
    kind,purpose=("el-dtu-opening-1",PURPOSE) if role=="sources" else ("el-dtu-reference-gate-1","one independent evaluation of the complete registered matrix")
    require(role in ("sources","references") and obj["schema"]==kind and obj["object_ids"]==["set6"] and obj["purpose"]==purpose and
            re.fullmatch("[0-9a-f]{64}",obj.get("freeze_sha256","")),"Wrong real-object opening")
    return obj


def main(a):
    opening(a.object_id,a.opening,a.opening_sha256,a.role)
    native_roles=SOURCES if a.role=="sources" else TARGETS
    image_paths=dict(a.image)
    require(len(a.image)==len(image_paths) and set(image_paths)==set(native_roles),"Wrong sealed photographic roster")
    planraw=a.plan.read_bytes(); plan=json.loads(planraw)
    cropraw=a.crop.read_bytes(); crop=json.loads(cropraw)
    require(plan["schema"]=="el-dtu-camera-plan-1" and plan["converter"]==crop["id"]=="dtu-crop256-v1" and
            digest(cropraw)==plan["crop_sha256"] and plan["lighting_id"]==3,"Crop/camera identity mismatch")
    metadata=json.loads(a.members.read_bytes())
    srcids,tarids=plan["source_view_ids"],plan["target_view_ids"]
    original={k:"SampleSet/MVS Data/Rectified/scan"+a.object_id[3:]+"/rect_%03d_3_r5000.png"%({**srcids,**tarids}[k]) for k in native_roles}
    files={x["member"]:x for x in metadata["files"]}
    require(len(metadata["files"])==len(files) and set(files)==set(original.values()),"Expected independently retrieved source-only or target-only members")
    index={}
    for k in native_roles:
        name=original[k]; receipt=files[name]; data=Path(image_paths[k]).read_bytes()
        require(record(data)=={t:receipt[t] for t in ("bytes","sha256")} and "%08x"%(zlib.crc32(data)&0xffffffff)==receipt["crc32"],
                "Wrong DTU original member, CRC or SHA for "+k)
        png=convert_rgb(data,crop)
        path=("sources/" if a.role=="sources" else "targets/")+k+".png"
        write(a.out/path,png)
        row={"official_view_id":{**srcids,**tarids}[k],"original_member":name,"original_sha256":receipt["sha256"],
             "original_bytes":receipt["bytes"],"original_crc32":receipt["crc32"],"rgb":{"path":path,**record(png)}}
        if a.role=="references":
            row.update(valid_mask={"kind":"all_true","width":256,"height":256},
                       object_mask=None,object_mask_reason="The official ObsMask is a 3D evaluation volume. No verified image-space object segmentation is supplied for these rectified views.")
        index[k]=row
    common={"object_id":a.object_id,"camera_plan_sha256":digest(planraw),"camera_sha256":plan["camera_sha256"],
            "view_set_sha256":plan["view_set_sha256"],"crop_sha256":plan["crop_sha256"],"lighting_id":3,
            "source_view_ids":srcids,"target_view_ids":tarids,"opening_sha256":a.opening_sha256,
            "download_members_sha256":digest(a.members.read_bytes()),"archive":metadata["archive"],"views":index}
    if a.role=="sources":
        write(a.out/"source-images.json",encoded({"schema":"el-dtu-source-images-1",**common}))
    else:
        write(a.out/"references.json",encoded({"schema":"el-dtu-references-1",**common}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role",choices=("sources","references"),required=True);p.add_argument("--object-id",required=True)
    for n in ("plan","crop","members","out"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--image",nargs=2,action="append",required=True,metavar=("LOGICAL_ID","FILENAME"))
    p.add_argument("--opening",type=Path);p.add_argument("--opening-sha256")
    main(p.parse_args())
