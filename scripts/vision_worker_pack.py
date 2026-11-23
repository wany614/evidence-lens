"""Make a deterministic Windows training source handoff from named project files and a Git-pinned CPU core."""
import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

VISION=("__init__","base","batch","configure","contract","deploy","model","native","selection","train")
CORE=("__init__","_bundle","comparison_board","evaluation","features","formal_inputs","formal_metrics","l1","l1_data","l1_freeze",
      "labels","risk_model","scoring_io")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--core-repo",type=Path,required=True)
    p.add_argument("--core-commit",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    require_commit=re.fullmatch("[a-f0-9]{40}",a.core_commit)
    if not require_commit:
        p.error("use a full public CPU core commit")
    data={}
    for name in ("scripts/vision_worker.py","scripts/vision_worker_pack.py","configs/vision-base-v1.json",
                 *("src/evidence_lens_vision/"+v+".py" for v in VISION)):
        f=a.source_root/name
        if not f.is_file() or f.is_symlink():
            raise ValueError("missing regular worker source "+name)
        data[name]=f.read_bytes()
    git=["git","-C",str(a.core_repo),"show"]
    for name in ("LICENSE","NOTICE",*("src/evidence_lens/"+v+".py" for v in CORE)):
        data[name]=subprocess.check_output([*git,a.core_commit+":"+name])
    entries={n:{"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()} for n,raw in sorted(data.items())}
    identity={"schema":"el-vision-worker-source-1","public_cpu_core_commit":a.core_commit,"files":entries,
              "capabilities":["pin-base","verify-base","configure","pilot","train","select","infer","infer-native","infer-dataset-development"],
              "test_opening":"The Mac study test entry additionally imports the registered independent evaluation authority."}
    data["worker-source.json"]=(json.dumps(identity,sort_keys=True,separators=(",",":"))+"\n").encode()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as out:
        for n,raw in sorted(data.items()):
            info=zipfile.ZipInfo(n,(1980,1,1,0,0,0))
            info.external_attr=0o100644<<16
            info.create_system=3
            out.writestr(info,raw,compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
    with zipfile.ZipFile(a.output) as z:
        if z.namelist()!=sorted(data) or any(z.read(n)!=v for n,v in data.items()):
            raise ValueError("source ZIP readback differs")
    print(json.dumps({"bytes":a.output.stat().st_size,"files":len(data),"sha256":hashlib.sha256(a.output.read_bytes()).hexdigest()}))


if __name__=="__main__":
    main()
