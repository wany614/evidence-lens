"""Record a registered DTU acquisition from a pinned freeze or one collective reference opening."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from evidence_lens._bundle import load, sha
from t0_records import bind, record, resolve
from t0_dtu_convert import authorise, pin, verify_reference_software


def acquire(store, history, freeze_pin, obj, condition, opening=None, endpoints=None, retry=None):
    roots = (store, history)
    freeze_pin = pin(freeze_pin)
    freeze = load(resolve(roots,freeze_pin))
    roster_pin = pin(freeze["sources"]["roster"])
    roster = load(resolve(roots,roster_pin))
    cfg = {"role":"sources" if condition else "references","condition":condition,"object_id":obj,
           "roster":roster_pin,"freeze":freeze_pin,"opening":pin(opening) if opening else None,
           "endpoints":pin(endpoints) if endpoints else None}
    authority = load(resolve(roots,opening)) if opening else None
    if condition is None and authority is not None:
        lock_pin = pin(authority["software_lock"])
        lock = load(resolve(roots,lock_pin))
        verify_reference_software(lock,freeze_pin,Path(__file__).resolve().parents[1])
        cfg["software_lock"] = lock_pin
    authorise(roster,cfg,freeze,authority)
    pins = {"roster":roster_pin,"freeze":freeze_pin,
            "crop":bind(store,roster_pin["run_id"],"payload/crop.json")}
    argv = ["{python}","{source}/scripts/t0_dtu_convert.py","--config","{config}","--out","{out}"]
    pins_to_args = {k:k for k in pins}
    if condition:
        for role, key in (("sourceCamera","source-cameras"),("renderCamera","render-cameras")):
            filename = key.replace("-","_") + ".json"
            pins[role] = bind(store,roster_pin["run_id"],f"payload/condition/{condition}/{filename}")
            pins_to_args[role] = key
        if (condition == "M4") != bool(endpoints):
            raise ValueError("M4 requires its exact previously received M2 endpoints")
        if endpoints:
            pins["endpointProvenance"] = pin(endpoints)
            pins_to_args["endpointProvenance"] = "endpoint-provenance"
            for n in (0,3):
                key = f"endpoint{n}"
                pins[key] = bind(store,endpoints["run_id"],f"payload/original/s0{n}.png")
                pins_to_args[key] = key
        receipt = "source-provenance.json"
    else:
        pins["softwareLock"] = cfg["software_lock"]
        pins_to_args["softwareLock"] = "software-lock"
        pins["opening"] = pin(opening)
        pins_to_args["opening"] = "opening"
        receipt = "references.json"
    for role, key in pins_to_args.items():
        argv += ["--"+key,"{input:"+role+"}"]
    suffix = condition or "evaluation-reference"
    return record(store,history,"dtu-acquire-"+obj+"-"+suffix,"dataset_build",pins,cfg,argv,
                  "payload/"+receipt,32*1024**2,retry=retry)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store",type=Path,required=True)
    p.add_argument("--history",type=Path,default=Path(".research"))
    p.add_argument("--freeze-pin",type=Path,required=True)
    p.add_argument("--object-id",required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--condition",choices=["M2","M4"])
    mode.add_argument("--reference-opening-pin",type=Path)
    p.add_argument("--endpoints-pin",type=Path)
    p.add_argument("--retry-of")
    a=p.parse_args()
    result=acquire(a.store.resolve(),a.history.resolve(),load(a.freeze_pin),a.object_id,a.condition,
                   load(a.reference_opening_pin) if a.reference_opening_pin else None,
                   load(a.endpoints_pin) if a.endpoints_pin else None,a.retry_of)
    print(json.dumps(result,sort_keys=True))
