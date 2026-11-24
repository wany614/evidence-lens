"""Run a recorded source acquisition or a collectively opened C6 evaluation acquisition."""
import argparse
import json
from pathlib import Path
from b2_intake_record import record_step,locate,sealed_member
from b2_intake_photos import SOURCES,valid_opening,verify_commitment,REFERENCE_CODE
from t0_dtu_archive import check

def acquire(roots,cohort_pin,obj,condition,opening=None,endpoints=None):
    cohort=json.loads(locate(roots,cohort_pin).read_bytes())
    check(obj in cohort["splits"]["C6"],"this batch only reads C6")
    prior=cohort["upstream_roster"]["run_id"]
    pins={"cohort":cohort_pin,"crop":sealed_member(roots,prior,"payload/crop.json")}
    opts=["--cohort","{input:cohort}","--crop","{input:crop}"]
    if condition:
        check(condition in SOURCES and opening is None and (condition=="M4")==bool(endpoints),"invalid fixed source authority")
        for typ in ("source","render"):
            k=typ+"Cam";pins[k]=sealed_member(roots,prior,"payload/condition/"+condition+"/"+typ+"_cameras.json")
            opts+=["--"+typ+"-camera","{input:"+k+"}"]
        if endpoints:
            pins["endpointProvenance"]=endpoints;opts+=["--endpoint-provenance","{input:endpointProvenance}"]
            for n in (0,3):
                k="endpoint"+str(n);pins[k]=sealed_member(roots,endpoints["run_id"],"payload/original/s0"+str(n)+".png")
                opts+=["--"+k,"{input:"+k+"}"]
        name="b2-C6-source-"+obj+"-"+condition;target="source-provenance.json"
    else:
        check(opening and not endpoints,"target acquisition needs independent opening")
        op=json.loads(locate(roots,opening).read_bytes());method=op["calibration_method"];locate(roots,method)
        valid_opening(cohort,cohort_pin,method,op)
        candidates=op["score_candidates"]
        verify_commitment(cohort,cohort_pin,method,op,json.loads(locate(roots,candidates).read_bytes()))
        pins.update(opening=opening,method=method,candidates=candidates);opts+=["--candidates","{input:candidates}","--opening","{input:opening}","--method","{input:method}"]
        name="b2-C6-reference-"+obj;target="references.json"
    cfg={"role":"sources" if condition else "references","object_id":obj,"condition":condition}
    return record_step(roots,name,"b2_intake_photos.py","payload/"+target,cfg,pins,opts,
                       extra_code=list(REFERENCE_CODE) if not condition else ["scripts/t0_dtu_convert.py","src/evidence_lens/dtu_camera.py"],estimate=32*1024**2,kind="dataset_build")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store",type=Path,required=True);p.add_argument("--batch-one",type=Path,required=True);p.add_argument("--history",type=Path,default=Path(".research"))
    for k in ("cohort-pin","endpoints-pin","opening-pin"):p.add_argument("--"+k,type=Path,required=k=="cohort-pin")
    p.add_argument("--object-id",required=True);p.add_argument("--condition",choices=tuple(SOURCES));a=p.parse_args()
    read=lambda f:json.loads(f.read_bytes()) if f else None
    print(json.dumps(acquire((a.store.resolve(),a.batch_one.resolve(),a.history.resolve()),read(a.cohort_pin),a.object_id,a.condition,read(a.opening_pin),read(a.endpoints_pin))))
