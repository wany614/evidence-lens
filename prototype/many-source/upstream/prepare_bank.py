"""Build a many-source audit bank from the supported native gsplat export and its registered source archive."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
from PIL import Image

from evidence_lens_bundle.common import encoded, json_bytes, parts, read_file, require, sha
from evidence_lens_many.bank import Bank, build_manifest


def file_spec(root, relative):
    path=root
    for part in parts(relative):
        path=path/part
        require(not path.is_symlink(),"symlink in external upstream asset")
    with path.open("rb") as f:
        h=hashlib.file_digest(f,"sha256").hexdigest()
    return {"path":relative,"bytes":path.stat().st_size,"sha256":h}


def frames(raw):
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        require(len(archive.files)==3 and set(archive.files)=={"rgb","z","alpha"},"incomplete native frame: expected RGB, alpha and optical-axis Z")
        rgb,z,alpha=(archive[k] for k in ("rgb","z","alpha"))
    require(rgb.dtype==z.dtype==alpha.dtype==np.float32 and rgb.shape==(252,378,3) and z.shape==alpha.shape==(252,378),"incompatible native frame shape/dtype")
    require(np.isfinite(rgb).all() and np.isfinite(alpha).all() and (rgb>=0).all() and (rgb<=1).all()
            and (alpha>=0).all() and (alpha<=1).all() and np.all(np.isfinite(z[alpha>1e-6])) and np.all(z[alpha>1e-6]>0)
            and np.all(np.isnan(z[alpha<=1e-6])),"native display, opacity or expected-depth support differs")


def checked(data, spec):
    require(len(data)==spec["bytes"] and sha(data)==spec["sha256"],"native source or prediction hash differs")
    return data


def classify(path):
    path=Path(path)
    if path.is_file() and path.suffix.lower()==".ply":
        with path.open("rb") as f:
            header=f.read(32768).split(b"end_header",1)[0]
        return {"status":"incomplete","asset":"gaussian_ply" if b"scale_0" in header and b"opacity" in header else "point_cloud_ply",
                "missing":["source access manifest","camera/frame calibration","source photographs","renderer photometry","expected camera-Z and opacity","source self-render evidence"],
                "reconstruction_provenance":"unverified"}
    if path.is_file() and path.suffix.lower()==".npz":
        try:
            frames(path.read_bytes())
        except (ValueError, KeyError) as exc:
            return {"status":"incomplete","asset":"native_frame","missing":[str(exc)],"reconstruction_provenance":"unverified"}
        return {"status":"incomplete","asset":"standalone_native_frame","missing":["native checkpoint and camera/source provenance"],
                "reconstruction_provenance":"unverified"}
    if path.is_dir() and (path/"asset.json").is_file():
        manifest=json_bytes((path/"asset.json").read_bytes())
        require(manifest.get("schema")=="el-many-external-asset-1","unsupported reconstruction export")
        return {"status":"native_manifest_present","scene_id":manifest.get("scene_id"),"registered_frames":len(manifest.get("frames",{}))}
    return {"status":"incomplete","missing":["supported native asset manifest","source-only calibrated observations"],"reconstruction_provenance":"unverified"}


def materialize(data_zip,asset,training,parity,probe,read_pred,out,identities=None):
    with Path(data_zip).open("rb") as source_stream:
        require(hashlib.file_digest(source_stream,"sha256").hexdigest()==asset["source_package"]["sha256"],"changed original source package")
    require(asset["schema"]=="el-many-external-asset-1" and training["schema"]=="el-many-official-fit-1"
            and parity["schema"]=="el-many-native-checkpoint-parity-1" and probe["passed"] is True,"incomplete native upstream evidence")
    require(asset["checkpoint"]==training["native_checkpoint"] and asset["run_id"]==training["run_id"]
            and asset["gaussians"]==training["gaussians"] and 0<asset["gaussians"]<=60000,"native checkpoint identity or cardinality differs")
    R,A,H=asset["R"],asset["A"],asset["H"]
    require(R==A==training["R"] and len(set(R))==len(R) and set(R).isdisjoint(H) and set(training["source_visits"])==set(R)
            and min(training["source_visits"].values())>0 and sum(training["source_visits"].values())==training["training_steps"]==6000
            and training["target_photo_opens"]==training["internal_holdout"]==0,"native training provenance missing or incomplete")
    native={"means","scales","quats","opacities","sh0","shN"}
    require(set(training["native_parameters"])==native and all(row["equal_to_loaded_native"] for row in training["native_parameters"].values()),
            "native tensor parity missing")
    require(set(asset["frames"])==set(parity["frames"])==set(R+H),"missing registered native prediction")
    with zipfile.ZipFile(data_zip) as z:
        scene=json_bytes(z.read("scene.json"))
        require(scene["schema"]=="el-many-source-data-1" and scene["R"]==R and scene["A"]==A and scene["H"]==H
                and scene["unit"]==asset["unit"] and scene["frame_receipt"]==asset["frame_receipt"] and scene["scene_id"]==asset["scene_id"]
                and scene["configuration"]==asset["configuration"] and set(scene["source_rgb"])==set(R),"changed bank roles or source frame")
        cameras={k:{c:scene["views"][k][c] for c in ("K","c2w","width","height")} for k in R+H}
        require(asset["views"]==cameras,"native and source cameras differ")
        sources,targets={},{}
        (out/"source").mkdir();(out/"renders").mkdir()
        for i,key in enumerate(R+H):
            native_file=asset["frames"][key];data=checked(read_pred(key,native_file),native_file)
            frames(data)
            name=f"renders/{i:04d}.npz";(out/name).write_bytes(data)
            prediction={"path":name,"sha256":native_file["sha256"],"bytes":len(data)}
            if key in scene["source_rgb"]:
                photo=scene["source_rgb"][key];content=checked(z.read(photo["path"]),photo)
                with Image.open(io.BytesIO(content)) as im:
                    require(im.mode=="RGB" and im.size==(378,252) and im.format=="PNG","incorrect analyzed R observation")
                name=f"source/{i:04d}.png";(out/name).write_bytes(content)
                sources[key]={"original_sha256":scene["original_rgb"][key]["sha256"],"photo":{"path":name,"sha256":photo["sha256"],"bytes":len(content)},"render":prediction}
            else:targets[key]={"render":prediction}
    frame={**scene["frame_receipt"],"matrix":scene["frame_receipt"]["world_to_deployment"]}
    permission={**scene["permission"],"source_ids":scene["permission"]["A"],"terms":scene["permission"]["license"]}
    require(permission["source_ids"]==A and permission["H"]==H and permission["R"]==R,"source acquisition grant disagrees with native visitation")
    bank=build_manifest(scene_id=scene["scene_id"],checkpoint_sha256=asset["checkpoint"]["sha256"],renderer_id="nerfstudio-project/gsplat 1.5.3 simple_trainer/"+asset["official_commit"],
        reconstruction_provenance="verified" if identities else "unverified",reconstruction_ids=R,accessible_ids=A,target_ids=H,metric_anchor_ids=frame.get("metric_anchor_ids",[]),
        frame_receipt=frame,cameras=cameras,grids=scene["grids"],sources=sources,targets=targets,depth_verified=True,
        depth_probe_sha256=asset["depth_probe"]["sha256"],permission=permission,
        provenance={"configuration":scene["configuration"],"unit":scene["unit"],"native_run_id":asset["run_id"],
                    "source_acquisition":scene["run_id"],"native_evidence":identities or {},
                    "calibration":frame,"rgb_encoding":asset["rgb_encoding"],"z_semantics":asset["z_semantics"]})
    (out/"bank.json").write_bytes(encoded(bank))
    verified=Bank(out,sha(encoded(bank))).inspect()
    require(not verified["unavailable_source_ids"] and not verified["missing_target_prediction_ids"],"incomplete scoring bank")
    return verified


def from_directory(source, native, out, trusted_asset=None, trusted_source=None, trusted_lineage=None):
    state=classify(native)
    require(state["status"]=="native_manifest_present","incomplete external input: "+", ".join(state.get("missing",[])))
    raw=read_file(native,"asset.json");meta=json_bytes(raw)
    ids=None
    if trusted_asset is not None or trusted_source is not None:
        require(trusted_asset==sha(raw) and trusted_source==meta["source_package"]["sha256"],"external asset or source provenance pin differs")
        ids={"externally_supplied_native_member_sha256":trusted_asset,"externally_supplied_source_archive_sha256":trusted_source}
    if trusted_lineage is not None:
        witness=read_file(native,"lineage.json")
        require(ids is not None and sha(witness)==trusted_lineage,"incorrect additional lineage witness")
        ids=json_bytes(witness)
        require(ids["asset"]["sha256"]==trusted_asset and ids["data"]["sha256"]==trusted_source
                and ids["checkpoint"]["sha256"]==meta["checkpoint"]["sha256"],"signed original lineage differs from supplied native/source")
    def doc(name):
        spec=meta[name]
        return json_bytes(checked(read_file(native,spec["path"]),spec))
    checkpoint=meta["checkpoint"]
    require(file_spec(native,checkpoint["path"])==checkpoint,"unverified native checkpoint")
    require(not out.exists(),"use a fresh output directory")
    out.mkdir(parents=True)
    try:
        return materialize(source,meta,doc("training"),doc("native_parity"),doc("depth_probe"),
                           lambda key,spec:read_file(native,spec["path"]),out,ids)
    except BaseException:
        shutil.rmtree(out)
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("inspect","prepare"))
    p.add_argument("--asset",type=Path,required=True);p.add_argument("--sources",type=Path);p.add_argument("--out",type=Path)
    p.add_argument("--trusted-asset-sha256");p.add_argument("--trusted-source-sha256");p.add_argument("--trusted-lineage-sha256")
    a=p.parse_args()
    res=classify(a.asset) if a.action=="inspect" else from_directory(a.sources,a.asset,a.out,a.trusted_asset_sha256,a.trusted_source_sha256,a.trusted_lineage_sha256)
    print(json.dumps(res,indent=2,sort_keys=True))
