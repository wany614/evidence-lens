"""Publish the verified experimental assets and confirm their bytes over anonymous HTTPS."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from dual_records import bind,resolve,selected,stage
from evidence_lens._bundle import load,save,sha
from evidence_lens._verify import require
from general_public_download import fetch
from many_seal import MANIFEST,TAG
from v05_publish import cmd,REMOTE,REPO


def advance(a):
    require(cmd("git","-C",a.repo,"remote","get-url","--push","origin")==REMOTE,"wrong remote")
    git=["git","-C",a.repo]
    def remote():return cmd(*git,"ls-remote","origin","refs/heads/main").split()[0]
    before=remote()
    require(all(re.fullmatch(r"[0-9a-f]{40}",c) for c in (a.commit,a.prior)) and before in (a.prior,a.commit)
            and cmd(*git,"rev-parse",a.commit+"^{commit}")==a.commit,"local or remote source changed")
    subprocess.run([*git,"merge-base","--is-ancestor",a.prior,a.commit],check=True)
    if before!=a.commit:
        cmd(*git,"-c","credential.helper=","-c","credential.helper=!"+shlex.quote(str(a.gh))+" auth git-credential",
            "push","origin",a.commit+":refs/heads/main",env={**os.environ,"GH_PROMPT_DISABLED":"1","GIT_TERMINAL_PROMPT":"0"})
    after=remote()
    require(after==a.commit,"remote update differs")
    save(a.out/"git.json",dict(schema="el-dual-track-public-source-1",run_id=os.environ["EL_RUN_ID"],
         repo=REPO,ref="refs/heads/main",before=before,after=after))


def release(a,spec):
    def api(path):return json.loads(cmd(a.gh,"api","repos/"+REPO+path))
    info=api("")
    require(info["full_name"]==REPO and not info["private"] and info["permissions"]["push"] and
            api("/commits/"+spec["code_commit"])["sha"]==spec["code_commit"],"wrong public source")
    def rows():return [r for r in api("/releases?per_page=100") if r["tag_name"]==TAG]
    results=rows()
    if a.operation=="draft" and not results:
        require(not cmd("git","-C",a.repo,"ls-remote","--tags","origin","refs/tags/"+TAG),"tag already exists")
        cmd(a.gh,"release","create",TAG,"--repo",REPO,"--draft","--prerelease","--target",spec["code_commit"],
            "--title","Multi-source CPU evidence audit 0.1.0.dev0","--notes-file",a.notes)
        results=rows()
    require(len(results)==1 and results[0]["target_commitish"]==spec["code_commit"] and results[0]["prerelease"],
            "unexpected release source or status")
    item=results[0]
    local={p.name:(p,p.stat().st_size,sha(p)) for p in a.asset}
    require(set(local)==set(spec["assets"])|{MANIFEST,"SHA256SUMS"} and
            all(local[n][1:]==(d["bytes"],d["sha256"]) for n,d in spec["assets"].items()),"release asset roster changed")
    def remote(row):
        require(all(f["state"]=="uploaded" for f in row["assets"]),"nonterminal GitHub asset")
        return {f["name"]:(f["size"],f.get("digest","").removeprefix("sha256:")) for f in row["assets"]}
    seen=remote(item)
    require(set(seen)<=set(local) and all(seen[k]==local[k][1:] for k in seen),"pre-existing release asset differs")
    if a.operation=="draft":
        require(item["draft"],"already published")
        missing=[v[0] for k,v in local.items() if k not in seen]
        if missing:cmd(a.gh,"release","upload",TAG,*missing,"--repo",REPO)
        item=rows()[0]
    else:
        require(load(a.draft)["id"]==item["id"] and seen=={k:v[1:] for k,v in local.items()},"public draft identity differs")
        if item["draft"]:cmd(a.gh,"release","edit",TAG,"--repo",REPO,"--draft=false","--prerelease","--latest=false")
        item=api("/releases/tags/"+TAG)
        ref=api("/git/ref/tags/"+TAG)["object"]
        if ref["type"]=="tag":ref=api("/git/tags/"+ref["sha"])["object"]
        require(not item["draft"] and ref["type"]=="commit" and ref["sha"]==spec["code_commit"],"different public tag")
    require(remote(item)=={k:v[1:] for k,v in local.items()},"GitHub readback ledger differs")
    save(a.out/"github.json",dict(schema="el-many-github-release-1",run_id=os.environ["EL_RUN_ID"],repo=REPO,tag=TAG,
        code_commit=spec["code_commit"],id=item["id"],url=item["html_url"],draft=item["draft"],prerelease=item["prerelease"],
        manifest_sha256=sha(a.manifest),assets={n:dict(bytes=v[1],sha256=v[2]) for n,v in local.items()}))


def anonymous(a,spec):
    official=load(a.published)
    require(official["tag"]==TAG and not official["draft"] and official["manifest_sha256"]==sha(a.manifest)
            and official["code_commit"]==spec["code_commit"],"wrong public release")
    base="https://github.com/"+REPO+"/releases/download/"+TAG+"/"
    downloads=[]
    for n,f in sorted(official["assets"].items()):
        require(re.fullmatch(r"[A-Za-z0-9_.-]+",n) and Path(n).name==n,"nonportable public filename")
        downloads.append(fetch(base+n,a.out/n,f["sha256"],f["bytes"]))
    require(load(a.out/MANIFEST)==spec,"different anonymous manifest")
    save(a.out/"readback.json",dict(schema="el-many-public-downloads-1",run_id=os.environ["EL_RUN_ID"],repo=REPO,
        tag=TAG,code_commit=spec["code_commit"],status="published_experimental",url=official["url"],
        manifest_sha256=sha(a.manifest),asset_count=len(downloads),bytes=sum(f["bytes"] for f in official["assets"].values()),
        downloads=downloads))


def control(a):
    s=a.store
    if a.operation=="push":
        require(a.label and a.label.startswith(("b-","c-")),"record a B or C publication stage")
        pins={"seal":selected(s,a.seal_label)}
        return stage(s,a.label,"archive_verify",pins,{"before":a.prior,"after":a.commit,"repo":REPO},
            ["{python}","scripts/many_distribute.py","push","--repo",str(Path.cwd()),"--gh",str(a.gh.resolve()),
             "--prior",a.prior,"--commit",a.commit,"--out","{out}"],"payload/git.json",2200)
    pin=selected(s,"b-research-release")
    pins={"manifest":pin};spec=load(resolve((s,s/"history"),pin))
    cmdline=["{python}","scripts/many_distribute.py",a.operation,"--manifest","{input:manifest}","--out","{out}"]
    if a.operation=="download":
        pins["published"]=selected(s,"b-github-publish")
        cmdline+=["--published","{input:published}"]
        return stage(s,"b-anonymous-readback","archive_verify",pins,{"mode":"anonymous HTTPS of every public asset"},
            cmdline,"payload/readback.json",sum(v["bytes"] for v in spec["assets"].values())+100000)
    pins["source_push"]=selected(s,"b-push-release-source")
    pins["sums"]=bind(s,pin["run_id"],"payload/SHA256SUMS");pins["notes"]=bind(s,pin["run_id"],"payload/release-notes.md")
    for name,v in spec["assets"].items():pins[v["role"]]=v["parent"]
    cmdline+=["--repo",str(Path.cwd()),"--gh",str(a.gh.resolve()),"--notes","{input:notes}","--asset","{input:manifest}","{input:sums}"]
    cmdline+=["{input:"+v["role"]+"}" for v in spec["assets"].values()]
    if a.operation=="publish":
        pins["draft"]=selected(s,"b-github-draft")
        cmdline+=["--draft","{input:draft}"]
    return stage(s,"b-github-"+a.operation,"archive_verify",pins,{"repo":REPO,"prerelease":True,"tag":TAG},
        cmdline,"payload/github.json",13000)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation",choices=("push","draft","publish","download"))
    for k in ("store","repo","out","gh","draft","published","manifest","notes"):p.add_argument("--"+k,type=Path)
    for k in ("commit","prior","label","seal-label"):p.add_argument("--"+k)
    p.add_argument("--asset",type=Path,nargs="+")
    a=p.parse_args()
    if a.store:print(control(a))
    elif a.operation=="push":advance(a)
    elif a.operation=="download":anonymous(a,load(a.manifest))
    else:release(a,load(a.manifest))
