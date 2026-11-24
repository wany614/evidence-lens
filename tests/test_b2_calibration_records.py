"""Synthetic archive gates and a complete evaluation run with six artificial object identities."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import b2_calibration_production as prod
from b2_intake_photos import verify_commitment, valid_opening
from b2_calibration import ObjectData


def strong(i):
    return {"run_id":f"20260101T000000000000Z-{i:016x}", "manifest_sha256":f"{i:064x}",
            "member":"payload/fixture.json", "sha256":f"{i+1:064x}"}


def fixtures():
    ids=[f"fixture{i}" for i in range(6)]
    co={"splits":{"C6":ids},"camera":{"conditions":{c:{"source_cameras":{"sha256":"1"*64},
        "render_cameras":{"sha256":"2"*64}} for c in ("M2","M4")}}}
    pins={"cohort":strong(1),"method":strong(2),"candidates":strong(3),"opening":strong(4),
          **{"ref"+o:strong(i+20) for i,o in enumerate(ids)}}
    cells=[{"object_id":obj,"condition":c,"status":"failed","reason":"fixture failure","disposition":strong(32)}
           for obj in ids for c in ("M2","M4")]
    gate={"schema":"el-b2-c6-opening-1","object_ids":ids,"conditions":["M2","M4"],
          "target_ids":list(prod.VIEWS),"cohort":pins["cohort"],"calibration_method":pins["method"],
          "score_candidates":pins["candidates"],"predictions_and_risks_sealed":True,"scheduled_entries":288,
          "scored_entries":0,"terminal_cells":cells}
    groups={}
    for c in ("M2","M4"):
        groups[c]=[]
        for i,oid in enumerate(ids):
            risk=np.array([.1,.2,.3,.4],np.float32)
            valid=np.ones(4,bool)
            lab=np.array([0,0,0,0] if c=="M2" else [1,1,1,1],np.uint8)
            groups[c].append(ObjectData.from_views(oid,[(risk,valid,valid,lab)]))
    score={"schema":"el-b2-score-candidates-1","run_id":pins["candidates"]["run_id"],
           "cohort":pins["cohort"],"calibration_method":pins["method"],"candidate_thresholds":prod.candidate_map(groups),
           "C6_reference_reads":0,"T1_photo_reads":0,"views":[{"fixture":True}]}
    return co,pins,gate,groups,score


def test_collective_opening_rejects_incomplete_fold_or_wrong_authority():
    co,p,gate,_,score=fixtures()
    valid_opening(co,p["cohort"],p["method"],gate)
    verify_commitment(co,p["cohort"],p["method"],gate,score)
    with pytest.raises(ValueError):
        verify_commitment(co,p["cohort"],p["method"],gate,{**score,"T1_photo_reads":1})
    score["candidate_thresholds"]["M4"]["full"].pop()
    with pytest.raises(ValueError):
        verify_commitment(co,p["cohort"],p["method"],gate,score)
    gate["terminal_cells"].pop()
    with pytest.raises(ValueError):
        valid_opening(co,p["cohort"],p["method"],gate)


def test_complete_numeric_report_with_two_distinct_fit_outcomes(tmp_path,monkeypatch):
    co,p,gate,groups,score=fixtures()
    sources={p["cohort"]["run_id"]:co,p["opening"]["run_id"]:gate,p["candidates"]["run_id"]:score}
    monkeypatch.setattr(prod,"checked",lambda tables,pin:sources[pin["run_id"]])
    monkeypatch.setattr(prod,"inspect_method",lambda *a:{"models":{"MS-L1":{"threshold":.25}},"scope":{},
                                                       "target":{},"definitions":{"label":"synthetic threshold"}})
    monkeypatch.setattr(prod,"fetch_evidence",lambda *a:(groups,score["views"],[{"fixture":True}],[{"fixture":True}]))
    monkeypatch.setenv("EL_RUN_ID",strong(50)["run_id"])
    prod.perform(SimpleNamespace(action="fit",out=tmp_path),{"roots":[],"pinned":p})
    get=lambda n:json.loads((tmp_path/n).read_bytes())
    r=get("calibration-results.json")
    assert r["candidate_rows"]==2814 and len((tmp_path/"leave-one-out.csv").read_text().splitlines())==13
    assert r["per_object_candidate_rows"]==14472
    assert get("profiles/M2.json")["status"]=="empirical_calibration_only"
    assert get("profiles/M4.json")["status"]=="no_feasible_operating_point" and get("profiles/M4.json")["tau"] is None
    assert r["outcomes"]["M2"]["loo_feasible_folds"]==6 and r["outcomes"]["M4"]["loo_feasible_folds"]==0


def test_native_reader_keeps_failed_views_and_original_four_method_counts(tmp_path,monkeypatch):
    co,p,gate,_,_=fixtures()
    targets={v:i for i,v in enumerate(prod.VIEWS,1)}
    co["target_view_ids"]=targets
    co["objects"]=[{"object_id":obj,"target_members":{v:{"name":obj+v} for v in targets}} for obj in co["splits"]["C6"]]
    pins={obj:p["ref"+obj] for obj in co["splits"]["C6"]}
    for i,row in enumerate(gate["terminal_cells"]):
        row["disposition"]=strong(100+i)
    originals={row["disposition"]["run_id"]:{"object_id":row["object_id"],"condition":row["condition"],
               "status":"failed"} for row in gate["terminal_cells"]}
    for obj,ref in pins.items():
        originals[ref["run_id"]]={"schema":"el-b2-dtu-references-1","object_id":obj,"opening":p["opening"],
           "cohort":p["cohort"],"calibration_method":p["method"],"views":{v:{"official_view_id":i,
           "archive_member":obj+v,"valid_mask":{"kind":"all_true","width":256,"height":256},"rgb":{}} for v,i in targets.items()}}
    monkeypatch.setattr(prod,"checked",lambda _,pin:originals[pin["run_id"]])
    monkeypatch.setattr(prod,"locate",lambda *_:tmp_path/"unused")
    monkeypatch.setattr(prod,"photo",lambda *_:np.zeros((256,256,3),np.float32))
    method={"models":{m:{"threshold":0.1} for m in prod.METHODS}}
    groups,views,orig,objects=prod.fetch_evidence([],method,co,p,gate["terminal_cells"],pins)
    assert len(views)==72 and len(orig)==288 and len(objects)==48 and len(groups["M4"])==6
    assert all(row["V"]==65536 and row["E"]==0 and row["own_AP"] is None and row["common_prevalence"] is None for row in orig)
    assert all(row["coverage"]==0 and row["low_error"] is None for row in objects)
