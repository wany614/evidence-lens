"""Re-run a sealed evaluation with its recovered inputs and source bytes, then compare every output file."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from evidence_lens._bundle import RUN_ID, files, load, save, sha
from evidence_lens._verify import require, verify_bundle, verify_dependencies

ROOT = Path(__file__).resolve().parents[1]


def replay(source, inputs, spec, output):
    data = load(spec)
    original = data["evaluation"]
    require(RUN_ID.fullmatch(original), "invalid original evaluation ID")
    for member, digest in data["source_files"].items():
        require(sha(source / member) == digest, "recovered source differs from recorded source")
    replacements = {"{python}": sys.executable, "{out}": str(output / "recomputed"), "{run_id}": original}
    for row in data["inputs"]:
        require(row["path"].startswith("inputs/"), "invalid recovered input binding")
        replacements["{input:"+row["role"]+"}"] = str(inputs / row["path"][7:])
    template = data["command"]
    require(template[:3] == ["{python}","-m","evidence_lens.evaluation"], "only replay the original CPU evaluator")
    argv = [replacements.get(x,x) for x in template]
    require(all("{" not in arg for arg in argv), "unresolved replay binding")
    env = dict(os.environ, PYTHONPATH=str(source/"src"), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([argv[0], "-B", *argv[1:]], cwd=source, env=env, check=True)
    obtained = {name:sha(path) for name,path in files(output/"recomputed").items()}
    require(obtained == data["output_hashes"], f"recomputed output differs: {sorted(n for n in obtained.keys()|data['output_hashes'].keys() if obtained.get(n)!=data['output_hashes'].get(n))}")
    save(output/"replay-verification.json", {"schema":"el-restoration-comparison-1","original_run_id":original,
         "replay_run_id":os.environ["EL_RUN_ID"],"original_manifest_sha256":data["manifest_sha256"],
         "outcome":"all_payload_files_byte_identical","compared_files":len(obtained),"hashes":obtained,
         "tolerance":"Exact byte equality, including JSON, CSV, Parquet, six sets of labels, threshold curves, and the annotated PNG board.",
         "evaluation_id_policy":"The replay uses the original evaluation ID in its outputs to reproduce the board annotation and table metadata. The enclosing forensic run has its own new ID."})


def prepare(store, rid):
    parent = store/"runs"/rid
    record = verify_bundle(parent)
    require(record["run"]["run_kind"]=="evaluation" and record["run"]["execution_status"]=="succeeded","original evaluation missing")
    verify_dependencies(record,store)
    code = load(parent/"code_snapshot.json")["files"]
    with tempfile.TemporaryDirectory(dir=store) as tmp:
        tmp=Path(tmp)
        # Source and command records have their own run-manifest pins. The recorder's normal
        # scientific parent map covers input/output artifacts, so transport source separately.
        src=tmp/"recovered-source"
        shutil.copytree(parent/"source",src,copy_function=os.link)
        spec=tmp/"replay.json"
        save(spec,{"evaluation":rid,"manifest_sha256":record["manifest_sha256"],"source_files":{r["path"]:r["sha256"] for r in code},
                   "inputs":record["inputs"]["inputs"],"command":load(parent/"command.json")["argv_template"],
                   "output_hashes":{n[8:]:r["sha256"] for n,r in record["artifacts"].items() if n.startswith("payload/")}})
        conf=tmp/"config.json"
        save(conf,{"purpose":"Independent recovered evaluation comparison","condition_id":"B0","original_evaluation":rid,
                   "original_manifest":record["manifest_sha256"],"replay_spec_sha256":sha(spec)})
        cmd=[sys.executable,"-m","evidence_lens.recorder","--root",str(store),"--source-root",str(ROOT),"--config",str(conf),
             "--experiment","restored-evaluation","--trial","replay-"+rid,"--kind","archive_verify","--parent",rid,
             "--input","old-inputs="+str(parent/"inputs"),"--input","archive-code="+str(src),"--input","spec="+str(spec),"--",
             "{python}","{source}/scripts/restore_replay.py","worker","--source","{input:archive-code}","--inputs","{input:old-inputs}",
             "--spec","{input:spec}","--output","{out}"]
        subprocess.run(cmd,check=True,cwd=ROOT)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest="mode",required=True)
    a=commands.add_parser("record")
    a.add_argument("--restored-store",type=Path,required=True);a.add_argument("--evaluation",required=True)
    b=commands.add_parser("worker")
    for k in ("source","inputs","spec","output"):
        b.add_argument("--"+k,type=Path,required=True)
    args=p.parse_args()
    if args.mode=="record":prepare(args.restored_store.resolve(),args.evaluation)
    else:replay(args.source,args.inputs,args.spec,args.output)
