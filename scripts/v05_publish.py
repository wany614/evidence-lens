"""Publish and independently retrieve the verified grid release using the registered repository."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess

from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from dual_records import bind, resolve, selected, stage
from general_public_download import fetch
from v05_baseline import MANIFEST, TAG


REPO = "wany614/evidence-lens"
REMOTE = "https://github.com/" + REPO + ".git"


def cmd(*args, env=None):
    return subprocess.check_output(list(map(str, args)), text=True, env=env).strip()


def push(a):
    argv = ["git", "-C", a.repo]
    require(cmd(*argv, "remote", "get-url", "--push", "origin") == REMOTE, "wrong push target")
    def remote_main():
        return cmd(*argv, "ls-remote", "origin", "refs/heads/main").split()[0]
    before = remote_main()
    require(before in (a.prior, a.commit) and re.fullmatch(r"[0-9a-f]{40}", a.commit)
            and cmd(*argv,"rev-parse",a.commit+"^{commit}") == a.commit, "remote or local commit changed")
    subprocess.run([*argv,"merge-base","--is-ancestor",a.prior,a.commit],check=True)
    if before != a.commit:
        cmd(*argv,"-c","credential.helper=","-c","credential.helper=!"+shlex.quote(str(a.gh))+" auth git-credential",
            "push","origin",a.commit+":refs/heads/main",
            env={**os.environ,"GIT_TERMINAL_PROMPT":"0","GH_PROMPT_DISABLED":"1"})
    after=remote_main()
    require(after==a.commit,"different remote commit after fast-forward")
    save(a.out/"git.json", dict(schema="el-v05-branch-publication-1",run_id=os.environ["EL_RUN_ID"],
         repo=REPO,before=before,after=after,ref="refs/heads/main"))


def get_release(a, spec, assets):
    import json

    def api(path):
        return json.loads(cmd(a.gh,"api","repos/"+REPO+path))
    repo=api("")
    require(repo["full_name"]==REPO and not repo["private"] and repo["permissions"]["push"],"unexpected repository")
    require(api("/commits/"+spec["code_commit"])["sha"]==spec["code_commit"],"release source missing")
    def releases():
        return [r for r in api("/releases?per_page=100") if r["tag_name"]==TAG]
    rows=releases()
    if a.operation=="draft" and not rows:
        require(not cmd("git","-C",a.repo,"ls-remote","--tags","origin","refs/tags/"+TAG),"tag already exists")
        cmd(a.gh,"release","create",TAG,"--repo",REPO,"--draft","--target",spec["code_commit"],
            "--title","Evidence Lens v0.5.0 · Calibrated CPU image grids","--notes-file",a.notes,*assets)
        rows=releases()
    require(len(rows)==1,"release draft missing or ambiguous")
    release=rows[0]
    require(release["target_commitish"]==spec["code_commit"],"unexpected release target")
    hashes={p.name:(p.stat().st_size,sha(p)) for p in assets}
    remote={f["name"]:(f["size"],f.get("digest","").removeprefix("sha256:")) for f in release["assets"]
            if f["state"]=="uploaded"}
    require(hashes==remote,"GitHub uploaded assets differ from the exact local files")
    if a.operation=="publish":
        require(release["id"]==load(a.draft)["release_id"],"unexpected release ID")
        if release["draft"]:
            cmd(a.gh,"release","edit",TAG,"--repo",REPO,"--draft=false")
        release=api("/releases/tags/"+TAG)
        ref=api("/git/ref/tags/"+TAG)["object"]
        if ref["type"]=="tag":
            ref=api("/git/tags/"+ref["sha"])["object"]
        require(not release["draft"] and ref["type"]=="commit" and ref["sha"]==spec["code_commit"],
                "public tag differs from tested source")
    else:
        require(release["draft"],"this version is already public")
    save(a.out/"github.json", dict(schema="el-v05-release-publication-1",run_id=os.environ["EL_RUN_ID"],
         repo=REPO,tag=TAG,code_commit=spec["code_commit"],release_id=release["id"],
         draft=release["draft"],published_at=release["published_at"],url=release["html_url"],
         manifest_sha256=sha(a.manifest),notes_sha256=sha(a.notes),
         assets={n:dict(bytes=p[0],sha256=p[1]) for n,p in hashes.items()}))


def readback(a, spec):
    public=load(a.public)
    require(public["tag"]==TAG and public["code_commit"]==spec["code_commit"] and not public["draft"]
            and public["manifest_sha256"]==sha(a.manifest),"unexpected published manifest")
    base="https://github.com/"+REPO+"/releases/download/"+TAG+"/"
    downloads=[]
    for name,record in sorted(public["assets"].items()):
        require(re.fullmatch(r"[A-Za-z0-9_.-]+",name) and name not in (".",".."),"invalid artifact name")
        downloads.append(fetch(base+name,a.out/name,record["sha256"],record["bytes"]))
    require(load(a.out/MANIFEST)==spec,"public discovery returned a different source identity")
    save(a.out/"readback.json",dict(schema="el-v05-anonymous-readback-1",run_id=os.environ["EL_RUN_ID"],
         tag=TAG,code_commit=spec["code_commit"],distribution_status="published",url=public["url"],
         manifest_sha256=sha(a.manifest),downloads=downloads))


def controller(a):
    s=a.store
    m=selected(s,"a-baseline")
    r=(s,s/"history")
    spec=load(resolve(r,m))
    gh=a.gh.resolve() if a.gh else None
    if a.operation=="push":
        require(bool(a.prior and a.commit and a.label and gh and gh.is_file()),
                "provide exact predecessor, proposed commit, GitHub CLI and stage")
        return stage(s,a.label,"archive_verify",{"baseline":m},
            {"expected_predecessor":a.prior,"commit":a.commit,"repo":REPO},
            ["{python}","scripts/v05_publish.py","push","--repo",str(Path.cwd()),"--gh",str(gh),
             "--prior",a.prior,"--commit",a.commit,"--out","{out}"],"payload/git.json",2000)
    pins={"manifest":m}
    op=a.operation
    command=["{python}","scripts/v05_publish.py",op,"--manifest","{input:manifest}","--out","{out}"]
    if op=="download":
        pins["public"]=selected(s,"a-github-publish")
        return stage(s,"a-anonymous-readback","archive_verify",pins,{"method":"anonymous HTTPS, verify all assets"},
            command+["--public","{input:public}"],"payload/readback.json",
            sum(v["bytes"] for v in spec["assets"].values())+100000)
    require(op in {"draft","publish"} and a.notes.is_file() and gh and gh.is_file(),
            "release action, GitHub CLI or notes missing")
    pins.update({v["role"]:v["parent"] for v in spec["assets"].values()})
    pins["checksums"]=bind(s,m["run_id"],"payload/SHA256SUMS")
    pins["push"]=selected(s,"a-push-release-commit")
    command+=["--gh",str(gh),"--repo",str(Path.cwd()),"--notes","{input:notes}"]
    if op=="publish":
        pins["draft"]=selected(s,"a-github-draft")
        command+=["--draft","{input:draft}"]
    command+=["--assets",*["{input:"+v["role"]+"}" for v in spec["assets"].values()],
              "{input:manifest}","{input:checksums}"]
    return stage(s,"a-github-"+op,"archive_verify",pins,{"repo":REPO,"exact_tag":TAG,"operation":op},
                 command,"payload/github.json",15000,{"notes":a.notes.resolve()})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation",choices=["push","draft","publish","download"])
    for n in ("store","repo","gh","out","notes","manifest","draft","public"):
        p.add_argument("--"+n,type=Path)
    for n in ("prior","commit","label"):
        p.add_argument("--"+n)
    p.add_argument("--assets",type=Path,nargs="+")
    a=p.parse_args()
    if a.store: print(controller(a))
    elif a.operation=="push": push(a)
    elif a.operation=="download": readback(a,load(a.manifest))
    else: get_release(a,load(a.manifest),a.assets)
