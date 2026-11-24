"""Download and verify the immutable public general release without an account or token."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

BASE="https://github.com/wany614/evidence-lens/releases/download/"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def fetch(url, target, expected_sha, size=None):
    if target.exists() or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("a new download destination and a complete SHA-256 are required")
    t=time.monotonic()
    limit=size if size is not None else 1024**2
    tmp=target.with_suffix(target.suffix+".partial")
    try:
        h=hashlib.sha256()
        with tmp.open("xb") as dest:
            q=urllib.request.Request(url,headers={"User-Agent":"Evidence-Lens-public-replay","Accept":"application/octet-stream"})
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(q,timeout=120) as reply:
                if reply.status != 200:
                    raise ValueError("anonymous download returned HTTP " + str(reply.status))
                n=0
                while block:=reply.read(1024**2):
                    n+=len(block)
                    if n>limit:
                        raise ValueError("download exceeds the declared size")
                    h.update(block);dest.write(block)
        if h.hexdigest()!=expected_sha or (size is not None and n!=size) or digest(tmp)!=expected_sha:
            raise ValueError("public asset readback differs: "+target.name)
        tmp.rename(target)
    finally:
        tmp.unlink(missing_ok=True)
    return {"url":url,"name":target.name,"bytes":n,"sha256":expected_sha,"anonymous":True,"environment_proxy":False,"http_status":200,
            "seconds_including_verification":time.monotonic()-t}


def download(out, tag, commit, manifest_sha, everything=False):
    if not re.fullmatch(r"v[0-9][0-9A-Za-z.-]*",tag) or not re.fullmatch("[0-9a-f]{40}",commit):
        raise ValueError("supply the published tag and full commit")
    out.mkdir(parents=True,exist_ok=True)
    base=BASE+tag+"/"
    manifest="release-manifest.json"
    receipts=[fetch(base+manifest,out/manifest,manifest_sha)]
    spec=json.loads((out/manifest).read_bytes())
    if spec["schema"]!="el-general-release-1" or spec["tag"]!=tag or spec["code_commit"]!=commit:
        raise ValueError("public release identity differs")
    selected={v["role"]:k for k,v in spec["assets"].items()}
    if len(selected)!=len(spec["assets"]) or set(selected)!={"wheel","sources","original_replays","static_trajectories","measurements"}:
        raise ValueError("unexpected general release roles")
    sums={**{name:a["sha256"] for name,a in spec["assets"].items()},manifest:manifest_sha}
    expected_sums="".join(sums[name]+"  "+name+"\n" for name in sorted(sums)).encode("ascii")
    receipts.append(fetch(base+"SHA256SUMS",out/"SHA256SUMS",hashlib.sha256(expected_sums).hexdigest(),len(expected_sums)))
    for name,entry in spec["assets"].items():
        if not re.fullmatch("[A-Za-z0-9_.-]+",name) or name in (".","..") or type(entry["bytes"]) is not int or entry["bytes"]<=0:
            raise ValueError("invalid asset name or declared size")
        if everything or entry["role"] in ("wheel","original_replays"):
            receipts.append(fetch(base+name,out/name,entry["sha256"],entry["bytes"]))
    result={"schema":"el-general-public-download-1","code_commit":commit,"tag":tag,"anonymous_requests":receipts}
    (out/"public-readback.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path)
    for name in ("tag","commit","manifest-sha256"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--all",action="store_true")
    a=p.parse_args()
    print(json.dumps(download(a.out,a.tag,a.commit,a.manifest_sha256,a.all),sort_keys=True))
