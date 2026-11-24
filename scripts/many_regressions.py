"""Run the multi-source interface and scientific boundary regressions against a named source commit."""

import argparse
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile

from dual_records import selected,stage
from evidence_lens._bundle import save,sha
from evidence_lens._verify import require,verify_bundle


def run(a):
    repo=Path.cwd().resolve()
    commit=subprocess.check_output(["git","rev-parse","HEAD^{commit}"],text=True).strip()
    pins={k:selected(a.store,label) for k,label in
        (("study","b-study-registration"),("cpu","b-score-core-build"),("interface","b-development-science-interface"))}
    cmd=["{python}","scripts/many_regressions.py","--repo",str(repo),"--commit",commit,"--out","{out}"]
    previous={}
    for rid in a.previous:
        folder=next(d for d in (a.store/"runs"/rid,a.store/"quarantine"/rid) if d.is_dir())
        result=verify_bundle(folder)
        require(result["run"]["trial_id"]=="b-study-regressions" and result["run"]["execution_status"]=="failed",
                "different prior failure")
        previous[rid]=result["manifest_sha256"]
    return stage(a.store,"b-frozen-science-regressions","archive_verify",pins,{"science_code_commit":commit,
                 "tests":"boundary and CPU API regressions with declared protocol inputs","prior_regression_runs":previous,
                 "repair":"Include the tracked protocol directory in the temporary checkout. The first audit omitted it and the strict retry identity rejected changed code."},
                 cmd,"payload/checks.json",5000)


def check(a):
    require(re.fullmatch(r"[0-9a-f]{40}",a.commit),"exact source commit required")
    cases=["tests/test_many_science_boundaries.py","prototype/many-source/tests/test_bank.py"]
    raw=subprocess.check_output(["git","-C",a.repo,"archive","--format=tar",a.commit,"scripts","src","prototype",
                                 "protocol",cases[0],"pyproject.toml"])
    with tempfile.TemporaryDirectory(dir=a.out.parent) as temp:
        root=Path(temp)
        with tarfile.open(fileobj=io.BytesIO(raw)) as files:files.extractall(root,filter="data")
        env={**os.environ,"PYTHONPATH":os.pathsep.join(str(root/p) for p in ("src","scripts","prototype/many-source/src")),
             "PYTHONDONTWRITEBYTECODE":"1","PYTEST_DISABLE_PLUGIN_AUTOLOAD":"1"}
        log=subprocess.run([sys.executable,"-B","-m","pytest","-q","-p","no:cacheprovider",*cases],
                           cwd=root,env=env,text=True,capture_output=True)
        (a.out/"pytest.txt").write_text(log.stdout+log.stderr,encoding="utf-8")
        require(log.returncode==0,"targeted regression failed; see captured pytest output")
        match=re.search(r"(\d+) passed",log.stdout)
        require(match and int(match[1])>=12,"incomplete boundary or CPU regression suite")
        save(a.out/"checks.json",dict(schema="el-many-interface-tests-1",run_id=os.environ["EL_RUN_ID"],
            code_commit=a.commit,passed=int(match[1]),exit_code=log.returncode,log_sha256=sha(a.out/"pytest.txt"),
            files={name:sha(root/name) for name in cases},scientific_fit_count=0))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("store","repo","out"):p.add_argument("--"+n,type=Path)
    p.add_argument("--commit")
    p.add_argument("--previous",action="append",default=[])
    a=p.parse_args()
    if a.store:print(run(a))
    else:check(a)
