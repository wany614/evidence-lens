"""Verify an independently pinned B prerelease from its anonymous GitHub distribution and run fresh CPU inference."""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import venv
import zipfile
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener

HERE=Path(__file__).resolve()
sys.path.insert(0,str(HERE.parent))
import vision_release as rel
from vision_public_smoke import CONTROL_INPUTS, unpack_archive

REPO="wany614/evidence-lens"
CPU="evidence-lens-"+rel.TAG+"-cpu.zip"
NUMERIC="evidence-lens-"+rel.TAG+"-numerical.zip"
WEIGHT="dinov2_vits14_pretrain.pth"
SOURCES=("verify_vision_public.py","vision_release.py","vision_public_smoke.py","vision_release_native.py")


def opener():
    return build_opener(ProxyHandler({}),HTTPSHandler())


def stream(op,url,sink,limit,expected=None):
    request=Request(url,headers={"User-Agent":"Evidence-Lens-anonymous-vision-verifier"})
    count=0;h=hashlib.sha256()
    with op.open(request,timeout=120) as response:
        host=(urlsplit(response.geturl()).hostname or "").lower()
        allowed=(host in {"github.com","api.github.com","dl.fbaipublicfiles.com"} or host.endswith((".githubusercontent.com",".fbcdn.net")))
        rel.require(urlsplit(response.geturl()).scheme=="https" and allowed,"unexpected public download endpoint")
        while True:
            b=response.read(1024**2)
            if not b:break
            count+=len(b);rel.require(count<=limit,"public download exceeded its declared size")
            h.update(b);sink.write(b)
    val={"bytes":count,"sha256":h.hexdigest()}
    if expected:rel.require(val==expected,"anonymous bytes differ from the independent file pin")
    return val


def fetch(op,url,limit):
    import io
    b=io.BytesIO();stream(op,url,b,limit);return b.getvalue()


def public_metadata(op,repo,tag,commit):
    rel.require(repo==REPO and re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+-rc\.[0-9]+",tag)
                and re.fullmatch(r"[a-f0-9]{40}",commit),"invalid expected GitHub identity")
    api="https://api.github.com/repos/"+repo
    def get(endpoint):return json.loads(fetch(op,api+endpoint,16*2**20))
    project=get("");release=get("/releases/tags/"+tag);ref=get("/commits/"+tag)
    rel.require(project["full_name"]==repo and project["private"] is False and ref["sha"]==commit and release["tag_name"]==tag
                and release["prerelease"] is True and release["draft"] is False and
                release["html_url"]=="https://github.com/"+repo+"/releases/tag/"+tag,"anonymous repository, tag or commit differs")
    names={r["name"]:r for r in release["assets"]}
    rel.require(len(names)==len(release["assets"]),"duplicate public release asset")
    prefix="https://github.com/"+repo+"/releases/download/"+tag+"/"
    for n,r in names.items():
        rel.require(rel.safe_name(n)==n and "/" not in n and r["browser_download_url"]==prefix+n
                    and type(r["size"]) is int and 0<r["size"]<2*1024**3 and r["state"]=="uploaded","unexpected GitHub attachment")
    return release,names,ref["commit"]["tree"]["sha"]


def git_source(op,tree_sha,base):
    api="https://api.github.com/repos/"+REPO+"/git/trees/"+tree_sha+"?recursive=1"
    tree=json.loads(fetch(op,api,16*2**20))
    rel.require(tree["sha"]==tree_sha and tree["truncated"] is False,"incomplete public source tree")
    entries={r["path"]:r for r in tree["tree"]}
    for n in SOURCES:
        e=entries["scripts/"+n];raw=(base/n).read_bytes()
        git_sha=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
        rel.require(e["type"]=="blob" and e["mode"] in ("100644","100755") and e["size"]==len(raw) and e["sha"]==git_sha,
                    "local verifier differs from public commit: "+n)
    return {n:rel.sig(base/n) for n in SOURCES}


def download(op,asset,expected,folder):
    n=asset["name"];final=folder/rel.safe_name(n)
    rel.require(expected["bytes"]==asset["size"] and not final.exists(),"asset size or destination differs: "+n)
    partial=final.with_name(n+".partial")
    with partial.open("xb") as f:received=stream(op,asset["browser_download_url"],f,expected["bytes"],expected)
    if asset.get("digest"):rel.require(asset["digest"]=="sha256:"+received["sha256"],"GitHub SHA-256 differs for "+n)
    partial.rename(final)
    return dict(received,source=asset["browser_download_url"],asset_id=asset["id"])


def same_candidate(c,tag,commit):
    rel.require(tag==rel.TAG and c["schema"]=="el-vision-public-candidate-1" and c["version"]==tag and c["code_commit"]==commit
                and set(c["assets"]) in ({CPU,NUMERIC},{CPU,NUMERIC,WEIGHT}) and set(c["supporting_assets"])=={"SHA256SUMS","RELEASE_NOTES.md"}
                and c["base"]["sha256"]==rel.BASE and c["base"]["bytes"]==88283115 and c["base"]["original_distribution"]==rel.BASE_URL
                and set(c["models"])=={"L1",*rel.BRANCHES} and set(c["weights"])=={"L1-full","L1-geometry",*rel.BRANCHES} and rel.HASH.fullmatch(c["release_sha256"]),
                "production B candidate, genuine three-model inventory and original base are required")
    for m in c["models"].values():
        rel.require(rel.RID.fullmatch(m["run_id"]) and rel.HASH.fullmatch(m["manifest_sha256"]) and rel.HASH.fullmatch(m["model_sha256"]),
                    "invalid selected producer or model pin")
    for spec in (*c["assets"].values(),*c["supporting_assets"].values(),*c["weights"].values()):
        rel.require(type(spec["bytes"]) is int and 0<spec["bytes"]<2*1024**3 and rel.HASH.fullmatch(spec["sha256"]),"invalid candidate file pin")


def install(temp,requirements,wheelhouse):
    env=temp/"clean-cpu";venv.EnvBuilder(with_pip=True).create(env)
    exe=env/("Scripts/python.exe" if os.name=="nt" else "bin/python")
    pip=[str(exe),"-I","-m","pip","--isolated","install","--disable-pip-version-check","--no-input"]
    config={k:v for k,v in os.environ.items() if not k.startswith(("PIP_","PYTHON","GH_","GITHUB_"))}
    config["PIP_CONFIG_FILE"]=os.devnull
    if platform.system() in ("Linux","Windows") and not wheelhouse:
        subprocess.run([*pip,"--index-url","https://download.pytorch.org/whl/cpu","torch==2.8.0"],check=True,env=config)
    locator=["--no-index","--find-links",str(wheelhouse.resolve())] if wheelhouse else ["--index-url","https://pypi.org/simple"]
    subprocess.run([*pip,*locator,"-r",str(requirements)],check=True,env=config)
    return exe


def extract_control(archive, case_sha, folder):
    """Copy the pinned case and ten scoring inputs. Evaluation assets stay in the attachment."""
    import shutil
    rel.require(rel.HASH.fullmatch(case_sha),"invalid external public control case pin")
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();indices=[n for n in names if n.split("/")[-1]=="case.json"]
        rel.require(len(names)==len(set(names)) and len(indices)==1,"expected one unambiguous public case.json in the attachment")
        index=rel.safe_name(indices[0]);prefix=index[:-len("case.json")]
        info=z.getinfo(index)
        rel.require(info.file_size<2**20,"oversized external case manifest")
        raw=z.read(info);rel.require(rel.fingerprint(raw)==case_sha,"public control manifest differs from its independent pin")
        doc=json.loads(raw)
        rel.require(doc["schema"]=="el-vision-public-adapter-case-1" and set(doc["files"])==CONTROL_INPUTS,
                    "external archive lacks the registered ten scoring inputs")
        rel.write(folder,"case.json",raw)
        for n in sorted(CONTROL_INPUTS):
            spec=doc["files"][n];info=z.getinfo(prefix+n)
            rel.require(info.file_size==spec["bytes"] and 0<info.file_size<=32*1024**2 and rel.HASH.fullmatch(spec["sha256"]),
                        "public control size or pin differs: "+n)
            p=folder/n;p.parent.mkdir(exist_ok=True,parents=True)
            with z.open(info) as src,p.open("xb") as dst:shutil.copyfileobj(src,dst,2**20)
            rel.require(rel.sig(p)==spec,"public control changed member: "+n)
    return doc


def verify(a):
    rel.require(a.tag==rel.TAG and rel.HASH.fullmatch(a.candidate_sha256),"use the exact B public tag and separately obtained candidate.json SHA-256")
    op=opener();out=a.output.resolve()
    rel.require(not a.native_case or not out.is_relative_to(a.native_case.resolve()),"grade output must remain outside the private source case")
    out.mkdir(exist_ok=True,parents=True)
    rel.require(not any(out.iterdir()),"choose an empty anonymous verification directory")
    public=out/"downloads";public.mkdir()
    start=time.monotonic();phase="public release metadata"
    try:
        release,assets,tree=public_metadata(op,REPO,a.tag,a.commit)
        local_sources=git_source(op,tree,HERE.parent)
        idx=assets["candidate.json"]
        rel.require(idx["size"]<2**20,"oversized release candidate index")
        proof={"candidate.json":download(op,idx,{"sha256":a.candidate_sha256,"bytes":idx["size"]},public)}
        c=rel.read(public/"candidate.json");same_candidate(c,a.tag,a.commit)
        explicit={}
        for item in a.attachment:
            name,sep,digest=item.partition("=")
            rel.require(sep and rel.HASH.fullmatch(digest) and name in assets and name not in explicit,"invalid extra independent attachment pin")
            explicit[name]={"sha256":digest,"bytes":assets[name]["size"]}
        planned={**c["assets"],**c["supporting_assets"]}
        rel.require(not (set(explicit)&(set(planned)|{"candidate.json"})) and set(assets)=={*planned,*explicit,"candidate.json"},
                    "public asset list and independent attachment inventory differ")
        rel.require(bool(a.public_case_asset)==bool(a.public_case_sha256) and (not a.public_case_asset or a.public_case_asset in explicit),
                    "the independently pinned external public-plane attachment and its case.json hash are required together")
        for n,s in {**planned,**explicit}.items():
            phase=n;proof[n]=download(op,assets[n],s,public)
        expected_sums="".join(s["sha256"]+"  "+n+"\n" for n,s in sorted(c["assets"].items()))
        rel.require((public/"SHA256SUMS").read_bytes()==expected_sums.encode(),"public checksum list and candidate differ")
        normalize=lambda s:s.replace("\r\n","\n").rstrip("\n")
        rel.require(normalize(release["body"])==normalize((public/"RELEASE_NOTES.md").read_text(encoding="utf-8")),
                    "public release body differs from the actual signed-source numerical summary")
        phase="original generic DINOv2 checkpoint"
        if WEIGHT in c["assets"]:
            base=proof[WEIGHT]
            rel.require(base["sha256"]==rel.BASE and base["bytes"]==88283115,"mirrored original base differs")
        else:
            p=public/(WEIGHT+".partial")
            with p.open("xb") as dest:
                base=stream(op,rel.BASE_URL,dest,88283115,{"bytes":88283115,"sha256":rel.BASE})
            p.rename(public/WEIGHT);base["source"]=rel.BASE_URL
        phase="inner archive and quantitative attachment"
        with tempfile.TemporaryDirectory(prefix="evidence-lens-anonymous-") as raw:
            temp=Path(raw);bundle=temp/"bundle"
            doc=unpack_archive(public/CPU,c["assets"][CPU]["sha256"],bundle,c["release_sha256"])
            rel.require(c["members"]==len(doc["files"])+2 and doc["code_commit"]==a.commit and doc["models"]==c["models"] and doc["method_freeze"]["run_id"]==c["method"]["run_id"]
                        and doc["method_freeze"]["manifest_sha256"]==c["method"]["manifest_sha256"] and doc["report"]==c["report"]
                        and doc["example"]["grade_run"]==c["example_grade"] and
                        all(doc["native_validation"][k]==c["native_grade"][k] for k in ("run_id","manifest_sha256"))
                        and rel.sha(bundle/"smoke.py")==local_sources["vision_public_smoke.py"]["sha256"],"downloaded study, grade, commit or models differ")
            measured={k:rel.sig(bundle/"models"/"L1"/(k+".json")) for k in ("L1-full","L1-geometry")}
            measured.update({k:rel.sig(bundle/"models"/k/"weights.safetensors") for k in rel.BRANCHES})
            rel.require(measured==doc["weights"]==c["weights"],"original trained coefficients or FP32 visual weights differ")
            with zipfile.ZipFile(public/NUMERIC) as numeric:
                expected=sorted(n.split("/",1)[1] for n in doc["files"] if n.startswith("study/"))
                rel.require(numeric.namelist()==expected,"standalone public numerical members differ")
                for n in expected:
                    with numeric.open(n) as f:
                        rel.require(hashlib.file_digest(f,"sha256").hexdigest()==doc["files"]["study/"+n]["sha256"],"independent numeric attachment differs: "+n)
            phase="clean CPU installation and authored example"
            exe=install(temp,bundle/"requirements.txt",a.wheelhouse)
            subprocess.run([str(exe),"-I","-B",str(bundle/"smoke.py"),"--bundle",str(bundle),"--output",str(out/"cpu"),"--use-installed",
                            "--archive",str(public/CPU),"--archive-sha256",c["assets"][CPU]["sha256"],"--release-sha256",c["release_sha256"]],check=True,cwd=temp)
            smoke=rel.read(out/"cpu/smoke.json")
            rel.require(smoke["passed"] and not smoke["failed"] and smoke["archive_sha256"]==c["assets"][CPU]["sha256"]
                        and smoke["release_sha256"]==c["release_sha256"] and set(smoke["views"])==set(rel.VIEWS)
                        and all(len(v)==7 and all(r["passed"] for r in v.values()) for v in smoke["views"].values()) and len(smoke["map_files"])==36,
                        "anonymous CPU grade did not verify all 42 distributions")
            public_control=None
            if a.public_case_asset:
                phase="separate 512 public-plane control"
                case=extract_control(public/a.public_case_asset,a.public_case_sha256,temp/"public-control-input")
                dest=out/"public-control"
                subprocess.run([str(exe),"-I","-B",str(bundle/"smoke.py"),"--bundle",str(bundle),"--use-installed",
                                "--release-sha256",c["release_sha256"],"--archive",str(public/CPU),"--archive-sha256",c["assets"][CPU]["sha256"],
                                "--case",str(temp/"public-control-input"),"--case-sha256",a.public_case_sha256,"--output",str(dest)],check=True,cwd=temp)
                checks=rel.read(dest/"control-scores.json")
                rel.require(checks["schema"]=="el-vision-public-control-scores-1" and checks["status"]=="scored" and
                            checks["case_sha256"]==a.public_case_sha256 and checks["inputs"]==case["files"] and checks["condition"]=="public-plane" and
                            checks["models"]=={k:v["model_sha256"] for k,v in c["models"].items()} and set(checks["views"])==set(rel.VIEWS) and
                            all(set(v)==set(doc["methods"]) and all(r["pixels"]==512**2 and r["finite"]==r["low"]+r["high"] and
                                r["finite"]+r["unknown"]==r["pixels"] for r in v.values()) for v in checks["views"].values()) and
                            len(checks["map_files"])==36 and len(checks["score_files"])==6,"incomplete separately scored native public control")
                public_control={"attachment":a.public_case_asset,"attachment_sha256":explicit[a.public_case_asset]["sha256"],"case_sha256":a.public_case_sha256,
                                "receipt_sha256":rel.sha(dest/"control-scores.json"),"method_views":42,"coordinates":42*512**2}
            private=None
            rel.require(bool(a.native_case)==bool(a.native_case_sha256),"supply both private case and independently pinned digest")
            if a.native_case:
                phase="optional private formal native equivalence"
                subprocess.run([str(exe),"-I","-B",str(HERE.parent/"vision_release_native.py"),"check","--case",str(a.native_case.resolve()),
                    "--case-sha256",a.native_case_sha256,"--archive",str(public/CPU),"--archive-sha256",c["assets"][CPU]["sha256"],
                    "--release-sha256",c["release_sha256"],"--output",str(out/"formal-native")],check=True,cwd=temp)
                receipt=out/"formal-native/native-validation.json";d=rel.read(receipt)
                rel.require(d["passed"] and d["family"]==doc["native_validation"]["family"] and d["models"]==doc["models"],"wrong formal native grade")
                private={"native_validation_sha256":rel.sha(receipt),"family":d["family"],"checked_coordinates":512*512*18}
        result={"schema":"el-vision-anonymous-verification-1","status":"verified","repository":REPO,"version":a.tag,"code_commit":a.commit,
                "release_url":release["html_url"],"release_id":release["id"],"github_published_at":release["published_at"],
                "candidate_sha256":a.candidate_sha256,"release_sha256":c["release_sha256"],"assets":proof,"base":base,
                "models":c["models"],"measured_weights":measured,"source_files":local_sources,"runtime":smoke["runtime"],"smoke_sha256":rel.sha(out/"cpu/smoke.json"),
                "plane_method_views":42,"plane_pngs":36,"separate_public_control":public_control,
                "formal_native_public_receipt":doc["native_validation"],"local_private_native":private,
                "access":"anonymous HTTPS, no credentials, cookies or environment proxy","verified_utc":datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds":time.monotonic()-start}
        rel.write(out,"public-verification.json",rel.entries(result))
        return {"status":"verified","version":a.tag,"verification_sha256":rel.sha(out/"public-verification.json")}
    except Exception as exc:
        rel.write(out,"failure.json",rel.entries({"schema":"el-vision-anonymous-verification-failure-1","phase":phase,"version":a.tag,
                  "candidate_sha256":a.candidate_sha256,"type":type(exc).__name__,"recheck":"Preserve downloads and local CPU or native receipts from this attempt."}))
        raise


if __name__=="__main__":
    if sys.version_info<(3,11):raise SystemExit("The anonymous B verifier requires Python 3.11 or newer.")
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag",required=True);p.add_argument("--commit",required=True);p.add_argument("--candidate-sha256",required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--wheelhouse",type=Path)
    p.add_argument("--attachment",action="append",default=[],metavar="NAME=SHA256")
    p.add_argument("--public-case-asset",help="independently pinned additional ZIP containing the native 512 public plane control")
    p.add_argument("--public-case-sha256",help="independent SHA-256 of case.json in the selected external control ZIP")
    p.add_argument("--native-case",type=Path);p.add_argument("--native-case-sha256")
    print(json.dumps(verify(p.parse_args()),sort_keys=True))
