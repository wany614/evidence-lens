"""Project a positive list of public study measurements and write the reader's installation entry."""

import argparse
import os
from pathlib import Path
import re

from dual_records import bind,selected,stage
from evidence_lens._bundle import load,save,sha
from evidence_lens._verify import require
from many_runtime import BASE_WHEEL,WHEEL
from many_seal import TAG


FILES={
    "status.json":("b-scientific-closeout","status.json"),
    "runs.csv":("b-scientific-closeout","runs.csv"),
    "resources-and-failures.md":("b-scientific-closeout","resources-and-failures.md"),
    "model-card.md":("b-model-package","model-card.md"),
    "feature-schema.json":("b-model-package","feature-schema.json"),
    "operating-point.json":("b-model-package","operating-point.json"),
    "model-package.json":("b-model-package","model-release.json"),
    "freeze-manifest.json":("b-model-freeze","freeze.json"),
    "data-roles.json":("b-data-registration","registration.json"),
    "upstream-recipe.json":("b-upstream-recipe","recipe.json"),
    "opening.json":("b-confirmation-h-opening","opening.json"),
    "independent-arithmetic.json":("b-independent-arithmetic-check","arithmetic.json"),
    "access-order.json":("b-data-final-access","access.json"),
    "input-alignment.json":("b-fit-input-alignment","alignment.json"),
    "checkpoint-updates.json":("b-checkpoint-update-attribution","updates.json"),
    "upstream.json":("b-upstream-closeout","upstream.json"),
    "results.md":("b-scientific-results","results.md")}
for name in ("scenes.csv","views.csv","comparability.csv","view-priority.csv","geometric-visibility.csv",
             "nearest-depth-approximation.csv","scene-domain-averages.csv","endpoints.json"):
    FILES[name]=("b-scientific-results",name)


def control(s):
    pins={}; mapping={}
    for i,(dest,(label,member)) in enumerate(FILES.items()):
        key=f"file{i}"
        pins[key]=bind(s,selected(s,label)["run_id"],"payload/"+member)
        mapping[key]=dest
    cmd=["{python}","scripts/many_docs.py","--config","{config}","--out","{out}"]
    for key in pins:cmd+=["--input",key,"{input:"+key+"}"]
    return stage(s,"b-public-documentation","report",pins,{"public_files":mapping,"phase":"B","tag":TAG},
                 cmd,"payload/public-docs.json",4500000)


def project(a):
    cfg=load(a.config);docs=a.out/"docs";docs.mkdir()
    base="https://github.com/wany614/evidence-lens/releases/download/"
    for k,path in a.input:
        src=Path(path);name=cfg["public_files"][k]
        require(name in FILES and sha(src)==cfg["sources"][k]["sha256"],"different study document")
        raw=src.read_text(encoding="utf-8")
        require(not re.search(r"/(?:Users|Volumes)/[\w.-]+|LAPTOP-[A-Z0-9]{6,}|ghp_[a-zA-Z0-9]{25,}|BEGIN [A-Z ]*PRIVATE KEY",raw),
                "private host or unintended material in public projection")
        if name=="results.md":
            raw=raw.replace("[Each method and domain](domain-ranks.csv)",
                "[Full tables including each method and domain]("+base+TAG+"/many-source-results.zip)")
        (docs/name).write_text(raw,encoding="utf-8")
    status=load(docs/"status.json");results=load(docs/"endpoints.json");model=load(docs/"model-package.json")
    verify=load(docs/"access-order.json")
    require(verify["confirmation_H_first_decodes_before_grant"]==0,"unexpected exposure in access report")
    download=base+TAG+"/"
    lines=["# Multi-source evidence audit","","The tool checks a fixed target render against a stated selection of source photographs. "
        "A limited query produces its risk ranking and identifies which sources are comparable. An explicit full audit can find a source that the limited query missed. "
        "Independent RGB measurements supply the error labels in this study.","",
        f"**Study result: {status['status']}**. The registered K16 comparison tested a fitted NumPy head against the `{status['strong_rule']}` rule. "
        "Each of the two external confirmation scenes contributed ten targets. Four independent target sets were opened together after all six confirmation score or failure receipts existed.","",
        "| ETH3D confirmation scene | Labelled errors / pixels | New errors / accepted | Rule errors / accepted | Mean target-level difference |",
        "|---|---:|---:|---:|---:|"]
    for r in status["external"]:
        delta="N/A" if r["new_minus_rule"] is None else f"{100*r['new_minus_rule']:+.3f} percentage points"
        lines.append(f"| {r['scene']} | {r['error_pixels']:,} / {r['labelled_pixels']:,} | "
            f"{r['new_errors']:,} / {r['quota']:,} | {r['rule_errors']:,} / {r['quota']:,} | {delta} |")
    lines+=["","[Open the numerical report](../../../examples/many-source-study/report.html) · [Measured results and P1 gap counts](results.md) · "
        "[Every numerical table, including risk–coverage curves]("+download+"many-source-results.zip)","","## Run the authored example on CPU","",
        "Download the original input ZIP into an empty working directory. Create a Python 3.12 virtual environment with `python -m venv .venv`. "
        "On macOS or Linux activate it with `source .venv/bin/activate`. In Windows PowerShell use `.\\.venv\\Scripts\\Activate.ps1`. "
        "[Authored input ZIP]("+download+"evidence-lens-b04-r16-example.zip) · [Two prepared reports with JSON and arrays]("+download+"evidence-lens-b04-r16-reports.zip)","",
        "```sh",
        "python -m pip install "+base+"v0.5.0/"+BASE_WHEEL+" "+download+WHEEL,
        "python -m zipfile -e evidence-lens-b04-r16-example.zip case","cd case",
        "evidence-lens-bank inspect --bank bank",
        "evidence-lens-bank score --bank bank --target h00 --budget 4 --strategy coverage --model model.json --model-sha256 "+model["weights_sha256"]+" --out finite",
        "evidence-lens-bank audit-bank --bank bank --target h00 --budget 4 --strategy coverage --model model.json --out audited",
        "```","","Open finite/report.html and audited/report.html. Each report carries the ordered query IDs, witnesses, scope states and the frozen rank. "
        "The original b04 target uses 16 source photographs. Its independent static-scene bank and optional CUDA upstream example have separate entry points. "
        "[Four-source panel](../../../examples/many-source/b04-layered-louvers/score/report.html) · [Full audit](../../../examples/many-source/b04-layered-louvers/audit-bank/report.html) · "
        "[Upstream adapter](../../../prototype/many-source/upstream/README.md)","","## Model and review receipts","",
        f"The trained parameter file is {model['model_bytes']:,} bytes. Its SHA-256 is `{model['weights_sha256']}`. "
        "The operating point is unset. Mac arm64 and an offline Linux arm64 container ran the original deployment example. "
        "Native Windows x86-64 supplied the source-only Gaussian fits. "
        "[Model ZIP]("+download+model["archive"]["name"]+") · [Model card](model-card.md) · [CPU platform results](deployment/README.md)","",
        "[Study protocol](protocol.md) · [Data roles](data-roles.json) · [Source and calibration](data/README.md) · [Frozen model and rule](freeze-manifest.json) · "
        "[Unified opening](opening.json) · [Actual reference access](access-order.json) · [Independent arithmetic](independent-arithmetic.json) · "
        "[Upstream recipe](upstream-recipe.json) · [Upstream fits](upstream.json) · [All source image and camera-grid checks](input-alignment.json) · [Checkpoint update counts](checkpoint-updates.json) · "
        "[Resource and failure ledger](resources-and-failures.md) · [Scope](limitations.md) · [Rights](rights.md)"]
    (docs/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    save(a.out/"public-docs.json",dict(schema="el-many-public-docs-1",run_id=os.environ["EL_RUN_ID"],
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(docs.iterdir())},
        source_receipts=cfg["sources"],phase="B",status=status["status"]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("store","out","config"):p.add_argument("--"+k,type=Path)
    p.add_argument("--input",nargs=2,action="append")
    a=p.parse_args()
    if a.store:print(control(a.store))
    else:project(a)
