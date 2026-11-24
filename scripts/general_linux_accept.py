"""Run the actual release acceptance in a fresh network-off Linux container on the recorded image."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require


def replay(a):
    spec=load(a.manifest)
    require(sha(a.manifest)==a.manifest_sha256,"different release inventory")
    image=json.loads(subprocess.check_output(["docker","image","inspect",a.image]))[0]
    require(image["Id"]==a.image and image["Os"]=="linux","use an exact installed Linux image digest")
    tool=Path(__file__).with_name("general_clean_accept.py")
    command=["docker","run","--rm","--network","none","--read-only","--tmpfs","/tmp:rw,exec,size=1g",
             "-v",str(a.out.resolve())+":/result","-v",str(tool.resolve())+":/app/accept.py:ro",
             "-v",str(a.wheelhouse.resolve())+":/wheels:ro",
             "-v",str(a.manifest.resolve())+":/release/release-manifest.json:ro"]
    for name,r in spec["assets"].items():
        if r["role"] in ("wheel","original_replays"):
            require(Path(name).name==name and sha(a.downloads/name)==r["sha256"],"different downloaded bytes")
            command.extend(["-v",str((a.downloads/name).resolve())+":/release/"+name+":ro"])
    command.extend([a.image,"python","-I","-B","/app/accept.py","--manifest-sha256",a.manifest_sha256,
                    "--downloads","/release","--default-only","--offline","--wheelhouse","/wheels",
                    "--out","/result/replay"])
    ran=subprocess.run(command,capture_output=True,text=True)
    if ran.returncode:
        raise ValueError("Linux public replay failed: "+ran.stderr[-2200:])
    p=a.out/"replay/acceptance.json";result=load(p)
    require(result["platform"]["os"]=="Linux" and image["Architecture"]=="arm64" and
            result["platform"]["arch"] in ("aarch64","arm64") and list(result["groups"])==["default"]
            and result["code_commit"]==spec["code_commit"],"Linux acceptance failed")
    save(a.out/"linux.json",{"schema":"el-general-linux-default-1","run_id":os.environ["EL_RUN_ID"],
         "image":a.image,"container_architecture":image["Architecture"],"network":"none","read_only_root":True,
         "clean_acceptance":result,"acceptance_sha256":sha(p),
         "wheelhouse":{q.name:sha(q) for q in sorted(a.wheelhouse.glob("*.whl"))},
         "distribution_boundary":"The same published bytes may be transported into this network-off system."})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("out","downloads","wheelhouse","manifest"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--manifest-sha256",required=True);p.add_argument("--image",required=True)
    replay(p.parse_args())
