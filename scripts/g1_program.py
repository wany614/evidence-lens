"""Record the complete numerical implementations used by the independent T1 readers before opening."""

import argparse
import json
import os
from pathlib import Path

from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from g1_fixed_score import selected
from general_records import resolve, stage


PROGRAMS=("scripts/g1_fixed_score.py","scripts/g1_joint.py","scripts/g1_evaluate.py","scripts/g1_recount.py",
          "scripts/g1_program.py","scripts/g1_metric_control.py","scripts/t1_nbv_evaluate.py","scripts/t1_nbv_independent.py",
          "scripts/t1_terminals.py","scripts/t1_access.py","scripts/t1_probe_terminal.py","scripts/t1_receive.py","scripts/dtu_eval_board.py",
          "scripts/dtu_eval_metrics.py","scripts/vision_evaluate.py","scripts/recovery_eval_risk.py")


def main(a):
    store=a.store; roots=(store,store/"history")
    project=Path(__file__).resolve().parents[1]
    if a.action=="worker":
        config=load(a.config)
        entry=load(resolve(roots,config["sources"]["entry"]))
        f=load(resolve(roots,entry["freeze"]))
        hashes=config["scientific_evaluation_files"]
        require(all(sha(project/n)==v for n,v in hashes.items()) and
                all(hashes[n]==v for n,v in {**f["scientific_code_sha256"],**f["selection_code"]}.items()),
                "the registered readers differ from their immutable sources")
        save(a.out/"software.json",{"schema":"el-t1-evaluation-program-1","run_id":os.environ["EL_RUN_ID"],
             "entry":config["sources"]["entry"],"scientific_evaluation_files":hashes,
             "reference_files_read":0,"reference_gate":"closed","risk_training":0,"threshold_refits":0})
        return
    ep=selected(store,"g0-entry")
    g=load(resolve(roots,ep));f=load(resolve(roots,g["freeze"]))
    require(not (store/"work/controller/t1-opening/selected.json").exists(),"reference authority already issued")
    names=sorted(set(PROGRAMS)|set(f["scientific_code_sha256"])|set(f["selection_code"]))
    hashes={n:sha(project/n) for n in names}
    return stage(*roots,"t1-evaluation-program","archive_verify",{"entry":ep,"freeze":g["freeze"],"metric_control":selected(store,"t1-metric-control-final")},
                 {"scientific_evaluation_files":hashes},["{python}","{source}/scripts/g1_program.py","worker",
                  "--store",str(store),"--config","{config}","--out","{out}"],"payload/software.json",9000)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=["seal","worker"])
    for k in ("store","config","out"):p.add_argument("--"+k,type=Path)
    print(json.dumps(main(p.parse_args()),sort_keys=True))
