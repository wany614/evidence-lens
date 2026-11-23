"""Record one source-only R2 initialization and its separate permitted Windows input package."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from evidence_lens._bundle import encoded,load,sha
from evidence_lens._verify import require,verify_bundle
def located(store,pin):
    from stage3_reference import located as original
    return original(store,pin)



INIT="024a4a08552e748afafc3ead27ba745d6f96e84757572e5fbf9aec387653d086"
STUDY="82619d87327135f0f999fab67b468de121b2fb95c77fab3af55b90285d0b2c8d"
REG={"run_id":"20261123T211839939983Z-b5f7663e8f694647","manifest_sha256":"38b1bfb26f985603055ef46c15602a70b399b3db128c5772e79a21bac1030327"}
SETS={"M2":("s00","s03"),"M4":("s00","s01","s02","s03")}
TARGETS=tuple(f"v{i:02d}" for i in range(1,7))
OLD={
    ("control057","M4"):(("20261123T161606028919Z-a8035c52a31d4515","30a6a2eea5de838e439a2bbb5b36de652b462603f75e029d5a318e78798793fa"),
                         ("20261123T161614103426Z-1a9611a4aeb54ddd","873cf27ee8662cf4daa15083ed74783a1c951a874848e5707ef4cab22027e24e")),
    ("set1","M2"):(("20261123T194437433991Z-8fac72aeaecb4df2","e5b4587a712f0ff8b614e27c9cf26a7d1e59f5549215bdacbe246cdfb9ca1ae6"),
                   ("20261123T194443666569Z-9d97763ba58549af","3b55d63322d5e86e842a952fcec1124fd69352cb0e9854cd35a2e4ea68a3ffb9")),
    ("set1","M4"):(("20261123T194448137507Z-f65aac2e9de54f81","81dcd2e836aad143c9711ead1e7ba8890a73453560498813a7446e06c0d44832"),
                   ("20261123T194454282387Z-32b4acc030cd4fae","c729e0cb797ec08ea4801f5f0d773ba7ef43917be3ab87ad491fd477e1610743")),
    ("set6","M2"):(("20261123T195811818489Z-81eef22303cf4b1f","08baee5b404b66940f837383dc6f754bd82ebc2327993ca10996bca10872878d"),
                   ("20261123T195818268548Z-6f1414bb1f0e42e6","9abcba59ab4badb4869628a5a992ab4d718c2de21f76654d97d5b734a963e860")),
    ("set6","M4"):(("20261123T195822973108Z-f2bcfc40c74147f4","9b54f0d1bb371dd6dc662ea7e77c5cc481d8305440dd9e7dfae811503c628f6d"),
                   ("20261123T195829364853Z-5b86125c06f74c42","224abd8d2077f785e8c845f152560197b103e268de9e3141963817a0f7cb87c1")),
}


def signature(path):
    return {"bytes":path.stat().st_size,"sha256":sha(path)}


def member(store,pin,name):
    root=located(store,pin)
    manifest=load(root/"manifest.json")
    listed={x["path"]:x for x in manifest["files"]}
    require(name in listed and sha(root/"run.json")==listed["run.json"]["sha256"] and load(root/"run.json")["execution_status"]=="succeeded","Unsuccessful input parent")
    path=root/name
    require(path.is_file() and not path.is_symlink() and signature(path)=={k:listed[name][k] for k in ("bytes","sha256")},"A pinned input member changed: "+name)
    return path


def source_role(store,pin,role,name):
    listing=load(member(store,pin,"inputs.json"))
    item=next((x for x in listing["inputs"] if x["role"]==role),None)
    require(item and item["kind"]=="file" and item["path"]=="inputs/"+role+"/"+name and len(item["files"])==1,"Wrong historic source role")
    return member(store,pin,item["path"])


def copy_script(project,src,name):
    dest=src/"scripts"/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(project/"scripts"/name,dest)


def package(a):
    cfg=load(a.config);ids=SETS[cfg["condition_id"]]
    meta=load(a.init_meta)
    require(meta["run_id"]==cfg["init_pin"]["run_id"] and meta["object_id"]==cfg["scene_family_id"] and
            meta["condition"]==cfg["condition_id"] and meta["initialization_protocol_sha256"]==INIT and
            meta["generator_source_ids"]==list(ids) and meta["init_npz_sha256"]==sha(a.init) and
            meta["source_camera_sha256"]==sha(a.cameras),"Wrong source-only initialization")
    src=load(a.cameras);ren=load(a.render_cameras)
    require(set(src["views"])==set(ids) and set(ren["views"])==set(ids+TARGETS) and all(src["views"][s]==ren["views"][s] for s in ids),"Incorrect renderer camera subset")
    images={s:Path(f) for s,f in a.source}
    require(set(images)==set(ids) and len(images)==len(a.source) and {s:sha(f) for s,f in images.items()}==meta["source_rgb_sha256"],"Wrong authorized source photos")
    data={"init.npz":a.init,"init.json":a.init_meta,"source_cameras.json":a.cameras,"render_cameras.json":a.render_cameras,
          **{"sources/"+s+".png":f for s,f in images.items()}}
    read={k:f.read_bytes() for k,f in data.items()}
    for k,b in read.items():
        dest=a.out/"handoff"/k;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(b)
    # This descriptive file goes next to the assets; the optimizer must stage only its authorized inputs.
    info={"schema":"el-recovery-input-handoff-1","run_id":os.environ["EL_RUN_ID"],"object_id":cfg["scene_family_id"],"condition":cfg["condition_id"],
          "init_pin":cfg["init_pin"],"registration":cfg["registration"],"source_parent":cfg["source_parent"],"render_camera_parent":cfg["render_camera_parent"],
          "regression_recipe_freeze":cfg.get("regression_recipe_freeze"),"registered_study_sha256":STUDY,"initialization_protocol_sha256":INIT,
          "generator_source_ids":list(ids),"target_ids":list(TARGETS),"actual_method":meta["actual_method"],"N":meta["N"],"scene_scale_m":meta["scene_scale_m"],
          "parameter_sha256":meta["parameter_sha256"],"source_rgb_sha256":meta["source_rgb_sha256"],"source_camera_sha256":sha(a.cameras),"render_camera_sha256":sha(a.render_cameras),
          "files":{k:signature(f) for k,f in data.items()}}
    (a.out/"handoff.json").write_bytes(encoded(info))
    # File order, permissions and timestamps fixed for re-delivery.
    from zipfile import ZIP_DEFLATED,ZipFile,ZipInfo
    with ZipFile(a.out/"input.zip","x",compression=ZIP_DEFLATED) as z:
        for n,b in sorted(read.items()):
            q=ZipInfo(n,date_time=(1980,1,1,0,0,0));q.create_system=3;q.compress_type=ZIP_DEFLATED;q.external_attr=0o100444<<16
            z.writestr(q,b)


def validate_regression_freeze(store,received):
    require(received is not None,"Set6 recovery initialization requires the separately sealed final recipe and common stop")
    p={k:received[k] for k in ("run_id","manifest_sha256")};chk=member(store,p,received["member"])
    require(received["member"].startswith("payload/") and sha(chk)==received["sha256"],"Changed regression freeze")
    d=load(chk)
    require(d["schema"]=="el-recovery-recipe-freeze-1" and d.get("registration",{}).get("run_id")==REG["run_id"] and
            d.get("initialization_protocol_sha256")==INIT and d.get("recipe_sha256") and d.get("common_stop_step") in (500,1500,3000,5000),
            "External regression requires the real sealed complete recipe")
    return p


def record(a):
    from make_handoff import publish
    from stage3_reference import perform
    project,store=a.project.resolve(),a.store.resolve()
    if a.object_id=="057":a.object_id="control057"
    require((a.object_id,a.condition) in OLD,"Not a registered geometric input cell")
    freeze=load(a.recipe_freeze) if a.recipe_freeze else None
    fpin=validate_regression_freeze(store,freeze) if a.object_id=="set6" else None
    reg=member(store,REG,"payload/registration.json");registration=load(reg)
    require(registration["registered_files"]["protocol/recovery-initialization-v1.json"]==INIT and registration["protocol_sha256"]==STUDY,"Different research registration")
    protocol=project/"protocol/recovery-initialization-v1.json";require(sha(protocol)==INIT,"Changed source geometry recipe")
    make=lambda pair: {"run_id":pair[0],"manifest_sha256":pair[1]}
    original_gen,original_render=map(make,OLD[a.object_id,a.condition]);ids=SETS[a.condition]
    paths={s:source_role(store,original_gen,s,s+".png") for s in ids}
    camera=source_role(store,original_gen,"cameras","source_cameras.json")
    cfg={"dataset_id":"reconstruction-recovery-v1","dataset_hash":STUDY,"scene_family_id":a.object_id,"scene_id":a.object_id,
         "condition_id":a.condition,"split":"regression" if a.object_id=="set6" else "development","trial_id":"r2-source-"+a.object_id+"-"+a.condition,
         "experiment_id":"recovery-source-init-v1","generator_source_ids":list(ids),"source_parent":original_gen,"registration":REG,
         "source_rgb_sha256":{s:sha(f) for s,f in paths.items()},"source_camera_sha256":sha(camera),
         "initialization_protocol_sha256":INIT,"recipe_sha256":INIT,"regression_recipe_freeze":freeze,"recording":{"expected_output_bytes":2000000}}
    scratch=store.parent/"recovery-private-scratch";scratch.mkdir(exist_ok=True)
    base=store/"recovery"/"inputs"/INIT/a.object_id/a.condition;base.mkdir(parents=True,exist_ok=True)
    with (base/"lock").open("a+b") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        complete=base/"selection.json"
        if complete.exists():
            s=load(complete)
            require(s["object_id"]==a.object_id and s["condition"]==a.condition and s["regression_recipe_freeze"]==freeze,"Prior selection differs")
            member(store,s["handoff_pin"],"payload/handoff.json");member(store,s["init_pin"],"payload/init.json")
            return s
        parents={"registered":REG,"original":original_gen}
        if fpin:parents["freeze"]=fpin
        partial=base/"initialization.json"
        if partial.exists():
            prior=load(partial);require(prior["regression_recipe_freeze"]==freeze,"Prior source preparation freeze differs");ip=prior["pin"]
        else:
            inputs={"cameras":(camera,"original"),"registration":(reg,"registered"),"rules":(protocol,None),**{s:(f,"original") for s,f in paths.items()}}
            if fpin:inputs["freeze"]=(member(store,fpin,freeze["member"]),"freeze")
            with tempfile.TemporaryDirectory(dir=scratch) as tmp:
                engine=Path(tmp)/"engine";copy_script(project,engine,"recovery_triangulate.py")
                worker=[str(a.geometry_python.absolute()),"{source}/scripts/recovery_triangulate.py","--cameras","{input:cameras}","--protocol","{input:rules}","--config","{config}","--out","{out}"]
                for s in ids:worker+=["--source",s,"{input:"+s+"}"]
                ip=perform(store,project,engine,cfg,"derivation",inputs,parents,worker)
            publish(partial,encoded({"pin":ip,"regression_recipe_freeze":freeze}))
        render_camera=source_role(store,original_render,"cameras","render_cameras.json")
        inputs={"cameras":(camera,"original"),"render_cameras":(render_camera,"calibration"),"init_meta":(member(store,ip,"payload/init.json"),"init"),
                "init":(member(store,ip,"payload/init.npz"),"init"),**{s:(f,"original") for s,f in paths.items()}}
        cfg={**cfg,"experiment_id":"recovery-input-handoff-v1","trial_id":cfg["trial_id"]+"-package","init_pin":ip,"render_camera_parent":original_render}
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            engine=Path(tmp)/"engine";copy_script(project,engine,"recovery_inputs.py")
            worker=["{python}","{source}/scripts/recovery_inputs.py","package","--config","{config}","--out","{out}"]
            for k in ("cameras","render_cameras","init_meta","init"):worker+=["--"+k.replace("_","-"),"{input:"+k+"}"]
            for s in ids:worker+=["--source",s,"{input:"+s+"}"]
            hp=perform(store,project,engine,cfg,"derivation",inputs,{**parents,"calibration":original_render,"init":ip},worker)
        meta=load(member(store,ip,"payload/init.json"))
        selected={"object_id":a.object_id,"condition":a.condition,"trial_id":cfg["trial_id"],"regression_recipe_freeze":freeze,"init_pin":ip,"handoff_pin":hp,
                  "init_meta_sha256":sha(member(store,ip,"payload/init.json")),"handoff_sha256":sha(member(store,hp,"payload/handoff.json")),
                  "init_npz_sha256":meta["init_npz_sha256"],"parameter_sha256":meta["parameter_sha256"],"actual_method":meta["actual_method"],
                  "N":meta["N"],"scene_scale_m":meta["scene_scale_m"],"triangulated_points":meta["eligibility"]["independent_tracks"],
                  "input_zip":signature(member(store,hp,"payload/input.zip"))}
        publish(complete,encoded(selected))
        return selected


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest="action",required=True)
    q=subs.add_parser("record")
    q.add_argument("--store",type=Path,default=Path(".research"));q.add_argument("--project",type=Path,default=Path("."))
    q.add_argument("--object-id",choices=("057","control057","set1","set6"),required=True);q.add_argument("--condition",choices=("M2","M4"),required=True)
    q.add_argument("--geometry-python",type=Path,required=True);q.add_argument("--recipe-freeze",type=Path)
    q=subs.add_parser("package");q.add_argument("--source",nargs=2,action="append",required=True)
    for n in ("config","out","cameras","render-cameras","init-meta","init"):q.add_argument("--"+n,type=Path,required=True)
    args=p.parse_args()
    if args.action=="record":print(json.dumps(record(args),sort_keys=True))
    else:package(args)
