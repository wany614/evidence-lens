"""Select bounded paired training candidates by development evidence from their recorder seals."""
from pathlib import Path

import numpy as np

from .contract import BRANCHES, HASH, RUN, SEEDS, digest, output_dir, put, read, require


def choose(rows,stage):
    required=4 if stage=="learning-rate" else 8
    require(len(rows)==required and len({r["run_id"] for r in rows})==len(rows),"incomplete or duplicate candidate roster")
    same=("training_dataset_sha256","development_dataset_sha256","preprocessing_sha256","base_pin_sha256","source_sha256")
    for r in rows:
        require(r["schema"]=="el-vision-training-1" and r["phase"]=="train" and r["status"]=="completed"
                and all(r[k]==rows[0][k] for k in same) and r["branch"] in BRANCHES and r["seed"] in SEEDS
                and np.isfinite(r["best_development_family_equal_ap"]) and r["frozen_identical"] is True
                and r["optimizer_steps"]>0 and r["task_gradient_minibatches"]>=3,"incompatible or unproven formal training run")
    for seed in sorted({r["seed"] for r in rows}):
        paired={r["initial_head_sha256"] for r in rows if r["seed"]==seed}
        design={r["paired_design_sha256"] for r in rows if r["seed"]==seed}
        require(len(paired)==len(design)==1, "paired methods used different head initialization")
    selected={}
    registered={"L2-frozen":{(0.,1e-4),(0.,3e-4)},"L2-vision-ft":{(1e-5,3e-4),(3e-5,3e-4)}}
    def summarize(r):
        return {k:r[k] for k in ("run_id","manifest_sha256","model_sha256","seed","head_lr","backbone_lr",
                                  "best_epoch","best_development_family_equal_ap","threshold")}
    for branch in BRANCHES:
        entries=[r for r in rows if r["branch"]==branch]
        seed_a=[r for r in entries if r["seed"]==SEEDS[0]]
        require(len(seed_a)==2 and {(r["backbone_lr"],r["head_lr"]) for r in seed_a}==registered[branch],
                "missing registered seed-A LR candidate")
        winner=min(seed_a,key=lambda r:(-r["best_development_family_equal_ap"],r["backbone_lr"],r["head_lr"]))
        selected[branch]=summarize(winner)
        if stage=="primary":
            repeats=[r for r in entries if r["seed"]!=SEEDS[0]]
            require(len(repeats)==2 and {r["seed"] for r in repeats}==set(SEEDS[1:]) and
                    all((r["backbone_lr"],r["head_lr"])==(winner["backbone_lr"],winner["head_lr"]) for r in repeats),
                    "paired seeds B/C must retain selected learning rate")
    return {"schema":"el-vision-lr-selection-1" if stage=="learning-rate" else "el-vision-primary-selection-1",
            "seed":SEEDS[0],"chosen":selected,"all_candidates":[dict(summarize(r),branch=r["branch"]) for r in rows],
            "rule":"highest development family-equal AP, tie smaller learning rate; seed A remains primary",
            **{k:rows[0][k] for k in same}}


def selection(args):
    from evidence_lens.formal_inputs import PinnedRun
    out=output_dir(args.output)
    roster=read(args.roster)
    require(roster["schema"]=="el-vision-candidate-roster-1" and roster["stage"]==args.stage,"unexpected candidate roster stage")
    rows=[]
    for pin in roster["runs"]:
        require(RUN.fullmatch(pin["run_id"]) and HASH.fullmatch(pin["manifest_sha256"]),"invalid candidate seal")
        original=PinnedRun(args.store,pin)
        require(original.run["run_kind"]=="model_fit" and original.run["execution_status"]=="succeeded",
                "selection requires successful original model fits")
        record=read(original.open("payload/training.json"))
        require(record["run_id"]==pin["run_id"] and record["package"]["path"]=="package/model.json",
                "candidate run or package member differs")
        package=original.open("payload/package/model.json")
        require(digest(package)==record["package"]["sha256"] and read(package)["model_id"]==pin["run_id"],
                "candidate scientific model index differs")
        rows.append(dict(record,manifest_sha256=pin["manifest_sha256"],model_sha256=digest(package)))
    doc=choose(rows,args.stage)
    doc["roster_sha256"]=digest(args.roster)
    put(out/"selection.json",doc)
    return doc
