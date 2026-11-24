"""Score only the two T1 fixed layouts using the original, separately loaded risk release."""

import argparse
import json
from pathlib import Path

from dtu_eval import score_map
from dtu_score import METHODS, SCOPES, VIEWS, RELEASE_SHA
from evidence_lens._bundle import load, sha
from evidence_lens._verify import require
from evidence_lens.multisource_io import Case
from general_records import bind, resolve, stage
from t1_access import FIXED, OBJECTS, S0, SLOTS


def selected(store, name):
    return load(store / "work/controller" / name / "selected.json")


def fixed_case(roots, obj, cond, readback, entry, freeze):
    require(obj in OBJECTS and cond in ("M2", "M4"), "only a registered fixed layout may enter formal risk scoring")
    result = load(resolve(roots, readback))
    require(result["object_id"] == obj and result["condition"] == cond and result["scientific_sample"]
            and not result["forced_fallback_fixture"] and result["target_reference_files_opened"] == 0
            and result["recipe_sha256"] == entry["recipe_sha256"]
            and result["freeze"] == freeze["sources"]["reconstruction"],
            "readback belongs to another reconstruction or study")
    source = load(resolve(roots,result["source"]))
    desired = S0 if cond == "M2" else FIXED
    require(source["object_id"] == obj and source["logical_condition"] == cond+"-fixed"
            and source["entry"]["sha256"] == sha(resolve(roots,source["entry"]))
            and source["entry"]["run_id"] == entry["run_id"]
            and [source["sources"][slot]["official_view_id"] for slot in SLOTS[cond]] == desired
            and source["reference_photo_reads"] == 0,"wrong fixed source order or exposure")
    if result["status"] != "succeeded":
        require(result["status"] == "failed" and bool(result.get("failure_reasons") or result.get("reason"))
                and not result.get("verified"), "missing scientific failure disposition")
        return result, None, None
    require(result["verified"] and result["completed_steps"] == 5000 and result["cap"] == 20000,
            "incomplete frozen fit or verification")
    cp = bind(roots[0], readback["run_id"], "payload/case/case.json")
    case = Case(resolve(roots,cp).parent,cp["sha256"])
    require(case.condition == cond and case.sha256 == result["case_sha256"] and
            list(case.generator) == SLOTS[cond] and list(case.meta["target_ids"]) == list(VIEWS)
            and case.meta["source_cameras"]["sha256"] == source["source_camera_sha256"] and
            case.meta["render_cameras"]["sha256"] == source["render_camera_sha256"], "fixed native case identity differs")
    return result, cp, case


def verify_scores(roots, score_pin, case, models):
    path = resolve(roots,score_pin)
    sc = load(path)
    require(sc["schema"] == "el-dtu-frozen-scores-1" and sc["release_sha256"] == RELEASE_SHA
            and sc["case_sha256"] == case.sha256 and sc["condition"] == case.condition
            and sc["method_scopes"] == SCOPES and sc["models"] == models and set(sc["views"]) == set(VIEWS)
            and sc["checkpoint_sha256"] == case.meta["checkpoint_sha256"]
            and sc["reconstruction_run_id"] == case.meta["reconstruction_run_id"], "unbound frozen scores")
    states = []
    for view in VIEWS:
        row = sc["views"][view]
        if view in case.failed:
            require(row["status"] == "render_failed" and row["reason"] == case.failed[view]
                    and row["methods"] == {}, "unverified absent fixed prediction")
        else:
            require(row["status"] == "ready" and set(row["methods"]) == set(METHODS), "incomplete fixed score row")
        for method in METHODS:
            if view not in case.failed:
                score_map(path.parent,sc,method,view,case,models[method]["threshold"])
            states.append({"view":view,"method":method,"status":row["status"],"scope":SCOPES[method],
                           "score_file":row["methods"].get(method,{}).get("file"),"reason":row.get("reason")})
    return states


def score_available(store, python):
    roots=(store,store/"history")
    entry_pin=selected(store,"g0-entry")
    entry=load(resolve(roots,entry_pin))
    f=load(resolve(roots,entry["freeze"]))
    require(sha(Path(__file__).with_name("dtu_score.py")) == f["scientific_code_sha256"]["scripts/dtu_score.py"],
            "wrong frozen risk executable")
    weight_pin=selected(store,"g0-original-public-models")
    weights=resolve(roots,weight_pin)
    folder=weights.parent/"cpu"
    require(sha(folder/"release.json") == RELEASE_SHA and load(weights)["models"] == f["released_models"],
            "incorrect public weight archive")
    for obj in OBJECTS:
        for cond in ("M2","M4"):
            key=obj+"-"+cond+"-fixed"
            name="t1-score-"+key
            rbpath=store/"work/controller"/("t1-readback-"+key)/"selected.json"
            decpath=store/"work/controller"/("t1-decisions-"+obj)/"selected.json"
            if not rbpath.exists() or not decpath.exists():
                continue
            rb=load(rbpath)
            rr,cp,case=fixed_case(roots,obj,cond,rb,entry,f)
            if cp is None:
                print(key,rr["status"],flush=True)
                continue
            pin=stage(*roots,name,"model_score",{"entry":entry_pin,"case":cp,"readback":rb,
                      "release":f["sources"]["risk_release"],"public_weights":weight_pin,"decisions":load(decpath)},
                      {"scope":"fixed four-method formal matrix; no independent target references",
                       "method_scopes":SCOPES,"object_id":obj,"condition":cond},
                      [str(python.absolute()),"-I","-B","{source}/scripts/dtu_score.py",
                       "--case",str(resolve(roots,cp).parent),"--case-sha256",cp["sha256"],
                       "--cpu-dir",str(folder),"--release-sha256",RELEASE_SHA,"--threads","4","--output","{out}"],
                      "payload/scores.json",20000000)
            print(key,pin["run_id"],flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store",type=Path,required=True)
    p.add_argument("--cpu-python",type=Path,required=True)
    a=p.parse_args()
    score_available(a.store.resolve(),a.cpu_python)
