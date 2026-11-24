"""Register the new study and bind every scientific stage to its own immutable inputs."""

import argparse
import os
from pathlib import Path

from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from dual_records import selected, stage


def register(s):
    pins = {k:selected(s, label) for k,label in
            (("baseline","a-baseline"),("published","a-github-publish"),("readback","a-anonymous-readback"))}
    return stage(s, "b-study-registration", "archive_verify", pins,
        {"scope":"before new heads, before new observations or independent confirmation evaluation"},
        ["{python}","scripts/many_study.py","--protocol","{input:protocol}","--baseline","{input:baseline}",
         "--readback","{input:readback}","--out","{out}"],"payload/study.json",14000,
        {"protocol":Path("protocol/many-source-study-v1.json").resolve()})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("store","protocol","baseline","readback","out"):
        p.add_argument("--"+n,type=Path)
    a=p.parse_args()
    if a.store: print(register(a.store))
    else:
        spec=load(a.protocol)
        require(spec["baseline_id"]=="sha256:"+sha(a.baseline) and
                load(a.readback)["distribution_status"]=="published" and
                load(a.readback)["manifest_sha256"]==sha(a.baseline),"independent baseline missing")
        save(a.out/"study.json",dict(schema="el-many-source-registration-1",run_id=os.environ["EL_RUN_ID"],
             protocol_sha256=sha(a.protocol),baseline=spec["baseline_id"],study=spec))
