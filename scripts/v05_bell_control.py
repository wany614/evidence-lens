"""Record the authored bell source, Gaussian render and independent reference roles separately."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

from dual_records import selected, stage
from evidence_lens._bundle import encoded, load, save, sha
from evidence_lens._verify import require
from v05_bell_common import CHECKPOINT, HISTORICAL_MANIFEST, HISTORICAL_RUN, OLD, SIZES, SOURCES, TARGETS, projections

ROOT = Path(__file__).resolve().parents[1]
OLD_FILES = {"cameras": "CASE/cameras.json", "model": "CASE/M4/scene.npz", "reconstruction": "CASE/M4/reconstruction.json",
             "settings": "REFERENCE/world/settings.json", "world": "REFERENCE/world/offset-bell-cage.blend",
             "budget": "CASE/M4/render-config.json", "core": "CASE/code/stage3_gaussians.py"}


def historical(store, role):
    manifest = store / "history/runs" / HISTORICAL_RUN / "manifest.json"
    require(sha(manifest) == HISTORICAL_MANIFEST, "changed original public bell manifest")
    member = "payload/" + OLD_FILES[role]
    row, = [r for r in load(manifest)["files"] if r["path"] == member]
    return {"run_id": HISTORICAL_RUN, "manifest_sha256": HISTORICAL_MANIFEST, "member":member,"sha256":row["sha256"]}


def work(action, cfg, args, out):
    if action == "register":
        save(out/"registration.json", {"schema":"el-v05-bell-processing-1","run_id":os.environ["EL_RUN_ID"],
             "parent":cfg["sources"]["eligibility"], "superseded_registration":cfg["sources"]["first_registration"],
             "superseded_unissued_ticket":cfg["sources"]["first_packet"], "objects":["scene_057_offset_bell_cage"],"source_order":SOURCES,
             "target_order":TARGETS,"analysis_wh":{name:list(wh) for name,wh in SIZES.items()},
             "authorized_originals":cfg["historical_materials"],"observation_scale":2,
             "new_source_acquisition":{"schema":"el-upstream-layout-authority-1","fixed_layouts":{"M2":[SOURCES[0],SOURCES[-1]],"M4":SOURCES}, "optional_pool":[]},
             "measurement":"original 256-trained M4 checkpoint unchanged; original Blender horizontal lens and poses; native source and target camera images at twice the analyzed width/height",
             "native":"native Gaussian display RGB, opacity and expected optical-axis Z with calibrated per-grid K, no depth interpolation",
             "old":"same 896x504 landscape observations resized to 256x144 and centered with top and bottom 56; native Gaussian render at 256x144 in the same field; invalid padding (RGB=0, alpha=0, Z=NaN)",
             "scope":"all 4 original development source poses and all 6 registered target poses; native square and four rectangles plus a separate legacy landscape letterbox; no new fit",
             "target_right":"independent reference renderer and evaluator only after scoring is sealed; only six landscape target images required",
             "license":{"original-scene-and-derived-photography":"CC-BY-4.0","model-and-render-code":"Apache-2.0"}})
        return
    data = load(args["registration"])
    require(data["schema"] == "el-v05-bell-processing-1" and data["authorized_originals"]["model"]["sha256"] == CHECKPOINT,
            "incorrect original bell registration")
    cameras = load(args["cameras"])["views"]
    p = projections(cameras)
    raw = {"projections.json":encoded(p),"run_gpu.py":(ROOT/"scripts/v05_bell_gpu.py").read_bytes()}
    names = {"model":"scene.npz","reconstruction":"reconstruction.json","budget":"render-config.json","core":"runtime/stage3_gaussians.py"}
    for k, name in names.items():
        require(sha(args[k]) == data["authorized_originals"][k]["sha256"], "registered file changed: "+k)
        raw[name] = args[k].read_bytes()
    ticket = {"schema":"el-v05-bell-gpu-ticket-1","run_id":os.environ["EL_RUN_ID"],"checkpoint_sha256":CHECKPOINT,
              "files":{n:hashlib.sha256(v).hexdigest() for n,v in raw.items()},"allowed_photographic_inputs":[],
              "source_order":SOURCES,"target_order":TARGETS,"originals":{k:data["authorized_originals"][k] for k in names},
              "registration":cfg["sources"]["registration"],"analysis_groups":[*SIZES,OLD],"expected_native_frames":(len(SIZES)+1)*(len(SOURCES)+len(TARGETS))}
    raw["ticket.json"] = encoded(ticket)
    archive = out/"bell-gpu.zip"
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED) as z:
        for k, v in sorted(raw.items()):
            z.writestr(k,v)
    save(out/"dispatch.json",{"schema":"el-v05-bell-gpu-dispatch-1","run_id":os.environ["EL_RUN_ID"],"archive":archive.name,
                             "archive_sha256":sha(archive),"archive_bytes":archive.stat().st_size,"checkpoint_sha256":CHECKPOINT,
                             "registration":cfg["sources"]["registration"],"source_photo_reads":0,"target_reference_reads":0,
                             "windows_command":"python -I -B run_gpu.py --out EMPTY_RECORDER_PAYLOAD"})


def control(store, action, opening=None, retry=None):
    if action == "register":
        pins={"eligibility":selected(store,"a-grid-registration"),"first_registration":selected(store,"a-bell-register"),
              "first_packet":selected(store,"a-bell-packet")}
        details={"historical_materials":{k:historical(store,k) for k in OLD_FILES}}
        kind,member,estimate="archive_verify","registration.json",15000
    else:
        pins={"registration":selected(store,"a-bell-register-six"),"cameras":historical(store,"cameras")}
        details={}
        if action == "packet":
            pins.update({k:historical(store,k) for k in ("model","reconstruction","budget","core")})
            kind,member,estimate="dataset_build","dispatch.json",6_000_000
        else:
            require(action in ("sources","references"),"unknown bell operation")
            pins.update({k:historical(store,k) for k in ("world","settings")})
            if action == "references":
                require(opening == "a-bell-reference-opening", "sealed landscape score opening required")
                pins["opening"]=selected(store,opening)
            exe=os.environ.get("EVIDENCE_LENS_BLENDER_PYTHON",str(ROOT/".venv-reference/bin/python"))
            command=[exe,"-B","{source}/scripts/v05_bell_photos.py",action,"--out","{out}","--config","{config}"]
            for k in pins:command += ["--input",k,"{input:"+k+"}"]
            name = "a-bell-source-observations" if action=="sources" else "a-bell-references"
            if action=="sources":
                fixes=[]
                for attempt in sorted((store/"work/controller/a-bell-sources").glob("attempt-*.jsonl")):
                    final=json.loads(attempt.read_text().splitlines()[-1])
                    fixes.append({k:final[k] for k in ("run_id","manifest_sha256","execution_status","failure_reason")})
                details["prior_control_attempts"]=fixes
            return stage(store,name,"render",pins,details,command,"payload/photos.json",200_000_000,retry=retry)
    command=["{python}","-B","{source}/scripts/v05_bell_control.py",action+"-worker","--out","{out}","--config","{config}"]
    for k in pins:command += ["--input",k,"{input:"+k+"}"]
    return stage(store,"a-bell-"+action+"-six",kind,pins,details,command,"payload/"+member,estimate)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("register","packet","sources","references","register-worker","packet-worker"))
    for key in ("store","config","out"):p.add_argument("--"+key,type=Path)
    p.add_argument("--input",action="append",nargs=2,default=[])
    p.add_argument("--opening")
    p.add_argument("--retry",help="Original failed recorder attempt for a diagnosed infrastructure recovery")
    a=p.parse_args()
    if a.action.endswith("-worker"):
        work(a.action[:-7],load(a.config),{k:Path(v) for k,v in a.input},a.out)
    else:print(json.dumps(control(a.store.resolve(),a.action,a.opening,a.retry)))
