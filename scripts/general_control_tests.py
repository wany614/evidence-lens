"""Run targeted deployment regressions from the verified public source ZIP."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require

SUITES=("test_b2_asset.py","test_bundle.py","test_replay.py","test_resident.py","test_t1_confirmation_controls.py")


def run(a):
    build=load(a.build)
    zipname,= [n for n in build["assets"] if n.endswith(".zip")]
    archive=a.build.parent/zipname
    require(sha(archive)==build["assets"][zipname]["sha256"],"different public source archive")
    with tempfile.TemporaryDirectory(dir=a.out.parent) as td:
        checkout=Path(td)/"project";checkout.mkdir()
        with ZipFile(archive) as z:
            z.extractall(checkout)
        env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1",
             "PYTHONPATH":os.pathsep.join(str(checkout/n) for n in ("src","scripts"))}
        result=subprocess.run([sys.executable,"-B","-m","pytest","-q","-p","no:cacheprovider",
                               "--junitxml",str(a.out/"results.xml"),*["tests/"+n for n in SUITES]],
                              cwd=checkout,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        print(result.stdout,flush=True)
    require(result.returncode==0,"a pinned targeted regression failed")
    cases=list(ET.parse(a.out/"results.xml").getroot().iter("testcase"))
    require(cases and all(c.find("failure") is None and c.find("error") is None and c.find("skipped") is None
                          for c in cases),"missing or failed regression")
    save(a.out/"tests.json",{"schema":"el-general-release-tests-1","run_id":os.environ["EL_RUN_ID"],
                             "source_commit":build["code_commit"],"source_zip_sha256":sha(archive),
                             "suites":SUITES,"tests_passed":len(cases),"skipped":0,
                             "xml_sha256":sha(a.out/"results.xml")})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--build",type=Path,required=True)
    run(p.parse_args())
