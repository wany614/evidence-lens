"""Check the two released Q4 heads and the independent geometry interface on one development view."""

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from dual_records import bind,selected,stage
from evidence_lens._bundle import load,save,sha
from evidence_lens._verify import require
from many_science_control import cpu_command,runtime,source_path,suite


def control(s):
    pins=suite(s)
    for role,stage_name in (("bank","b-bank-b04-layered-louvers-r16"),("truth","b-truth-b04-layered-louvers"),
                            ("visibility","b-exact-visibility-b04-layered-louvers")):
        pins[role]=selected(s,stage_name)
    pins["old"]=bind(s/"history","20261124T041052425868Z-d308c134e69d4d4b","payload/cpu/release.json")
    cmd=cpu_command(runtime(s,pins),"many_scientific_probe.py")+["--out","{out}","--config","{config}"]
    for role in ("bank","truth","visibility","old"):cmd+=["--"+role,str(source_path(s,pins[role]))]
    return stage(s,"b-development-science-interface","archive_verify",pins,
        {"role":"development","target":"h00","condition":"R16","label_comparisons":0},cmd,"payload/verified.json",14000)


def verify(a):
    import numpy as np
    from evidence_lens_many import Bank,legacy_q4,select_sources,stream_features
    from many_data_access import EvaluatorTruth
    from many_data_visibility import VisibilityTruth
    from many_evaluate import exact_visibility
    p=load(a.config)["sources"]
    require(all(sha(getattr(a,k))==p[k]["sha256"] for k in ("old","bank","truth","visibility")),"different source pin")
    b=Bank(a.bank.parent,p["bank"]["sha256"])
    require(b.meta["scene_id"]=="b04-layered-louvers" and len(b.ids)==16,"development pilot required")
    reference=EvaluatorTruth(a.truth,p["truth"]["sha256"],bank=b)
    probe=VisibilityTruth(a.visibility,p["visibility"]["sha256"])
    probe.verify(reference)
    methods={}; cells={}; geom={}
    plan=select_sources(b,"h00",strategy="coverage")
    for step in stream_features(b,"h00",plan,budgets=(4,8,16,32,"all")):
        name=str(step.actual_k)
        cells[name]={"queries":list(step.queried_source_ids)}
        geom[name]={"comparable":step.comparable_count.copy()}
        if step.actual_k==4:
            for method in ("MS-L1","MS-ft"):
                result=legacy_q4(b,"h00",step.queried_source_ids,method,a.old.parent,threads=4)
                require(result["profile"]["queried_source_ids"]==cells[name]["queries"] and
                        np.isfinite(result["rank"]).any() and
                        np.array_equal(np.isfinite(result["rank"]),result["finite"]),"released rank domain mismatch")
                methods[method]={"profile":result["profile"],"finite":int(result["finite"].sum()),
                                 "pixels":result["finite"].size}
    require(set(geom)=={"4","8","16"},"aliased library sizes repeated or omitted")
    counts=exact_visibility(probe,"h00",cells,geom)
    require(all(v["probe_pixels"]==1944 and v["independent_target_surface"]>0 for v in counts.values()),
            "unusable physical-visibility probe")
    save(a.out/"verified.json",dict(schema="el-many-dev-science-interface-1",run_id=os.environ["EL_RUN_ID"],
         study="development only",bank=b.sha256,queries=cells,old_heads=methods,visibility=counts,
         reference_rgb_decodes=0,new_head_fits=0))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("store","config","bank","truth","visibility","old","out"):p.add_argument("--"+n,type=Path)
    a=p.parse_args()
    if a.store:print(control(a.store))
    else:verify(a)
