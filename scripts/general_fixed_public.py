"""Export only numerical fixed-layout T1 evidence and immutable protocol receipts to the public tree."""
import argparse
import csv
import json
from pathlib import Path
import shutil

from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from general_records import bind, resolve

TABLES=("method-views.csv","method-conditions.csv","matched-views.csv","matched-conditions.csv",
        "object-equal.csv","unknown-reasons.csv","source-fit.csv")
METHODS=("B-L1-transfer","MS-L1","MS-frozen","MS-ft")


def rows(path):
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value, precision=4):
    return format(float(value),f".{precision}f") if value!="" and value is not None else "N/A"


def publish(store,out):
    roots=(store,store/"history")
    def select(name):
        return load(store/"work/controller"/name/"selected.json")
    opened=select("t1-opening"); op=load(resolve(roots,opened));mp=op["matrix"]
    matr=load(resolve(roots,mp))
    evaluation=select("t1-fixed-evaluation"); evpath=resolve(roots,evaluation);ev=load(evpath)
    recount=select("t1-fixed-recount"); audited=load(resolve(roots,recount))
    require(ev["opening"]==opened and ev["matrix"]==mp and audited["primary"]==evaluation and ev["scheduled"]==288,
            "independent confirmation has no complete audit")
    require(set(ev["numeric_outputs"])==set(TABLES), "unexpected file in the public metric allowlist")
    out.mkdir(parents=True,exist_ok=True);fixed=out/"fixed";fixed.mkdir(exist_ok=True)
    pins={}
    def copy(pin,name):
        src=resolve(roots,pin);d=fixed/name
        if d.exists():
            require(sha(d)==pin["sha256"],"public evidence changed: "+name)
        else:
            shutil.copyfile(src,d)
        pins[name]=pin
    for n in TABLES:
        p=bind(store,evaluation["run_id"],"payload/public/"+n)
        require(p["sha256"]==ev["numeric_outputs"][n],"different numerical result")
        copy(p,n)
    for p,n in [(opened,"opening.json"),(op["evaluation_program"],"evaluation-program.json"),(evaluation,"evaluation.json"),
                (recount,"native-recount.json"),(mp,"sealed-matrix.json")]:
        copy(p,n)
    save(fixed/"origins.json",{"schema":"el-t1-public-numerical-evidence-1","files":pins,
                               "original_images_and_boards":"owner's private research store"})
    objects=rows(fixed/"method-conditions.csv"); quotas=rows(fixed/"matched-conditions.csv"); avg=rows(fixed/"object-equal.csv")
    lookup={(r["object_id"],r["condition"],r["method"]):r for r in objects}
    equal={(r["condition"],r["method"],r["metric"]):r for r in avg}
    lines = ["# Prospective T1 on six DTU objects","",
             f'The jointly sealed cohort supplied {ev["scored"]} scored slots among 288 scheduled. The source evidence, immutable fits and terminal failures had all been signed when the independent reader opened the 36 registered target photographs. The 42 logical reconstruction conditions have a separate observation and computation ledger. [Pre-opening matrix](fixed/sealed-matrix.json) · [Collective opening](fixed/opening.json) · [Native independent recount](fixed/native-recount.json)',
             "","## Per-object fixed-layout reconstruction and released MS-L1 threshold","",
             "| Object | Layout | Views scored / planned | RGB errors Y / E | Low L / V | Errors LY / L | Unknown U / E | Target MAE |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for obj in op["object_ids"]:
        for cond in ("M2","M4"):
            r=lookup[obj,cond,"MS-L1"]
            lines.append(f'| {obj} | {cond} | {r["views_scored"]}/{r["views_scheduled"]} | {r["Y"]}/{r["E"]} | {r["L"]}/{r["V"]} | {r["LY"]}/{r["L"]} | {r["U"]}/{r["E"]} | {fmt(r["MAE"])} |')
    lines+=["","A zero denominator makes the corresponding rate undefined. Missing M4 predictions retain every reference pixel in V. Their L/V equals zero and their error prevalence and LY/L are undefined. Finite support S, common support C, error counts and the absolute MAE sum accompany every method and view. [All 288 rows](fixed/method-views.csv) · [All 48 object–layout–method totals](fixed/method-conditions.csv) · [Unknown causes](fixed/unknown-reasons.csv)",
        "","## Four methods on identical current candidates","","Each physical object contributes equally to a defined mean. Common AP and prevalence use the same defined views. The 20% column uses the registered exact per-view quota from the full crop, restricted to common score support. Coverage at the original thresholds uses each method's native application domain.",
        "","| Layout | Method | Objects with defined AP / planned | Common AP | Prevalence | Errors at 20% | Original L/V | Original LY/L |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for cond in ("M2","M4"):
        for m in METHODS:
            def value(k):
                return fmt(equal[cond,m,k]["object_equal_mean"])
            n=equal[cond,m,"common_AP"]
            lines.append(f'| {cond} | {m} | {n["objects_defined"]}/{n["objects_scheduled"]} | {value("common_AP")} | {value("common_prevalence")} | {value("within_new_risk_20%_V")} | {value("coverage")} | {value("low_error")} |')
    lines+=["","## Errors among the accepted pixels at 20%","","| Object | Layout | "+" | ".join(METHODS)+" |","|---|---|---:|---:|---:|---:|"]
    matched={(r["object_id"],r["condition"],r["method"]):r for r in quotas if r["anchor"]=="20%_V"}
    for obj in op["object_ids"]:
        for cond in ("M2","M4"):
            counts=[]
            for m in METHODS:
                r=matched[obj,cond,m]
                counts.append((r["accepted_positive"]+"/"+r["accepted"]) if int(r["risk_defined_views"]) else "N/A")
            lines.append("| "+obj+" | "+cond+" | "+" | ".join(counts)+" |")
    lines+=["","Scores tied at a cutoff use the original label-independent pixel order. Every attainable and unattainable quota, cutoff group and optimistic/pessimistic tie count is retained. The two C6 profiles have no feasible operating point. [All matched quotas](fixed/matched-views.csv) · [Object-equal denominators](fixed/object-equal.csv) · [Source self-render errors](fixed/source-fit.csv) · [Prediction and reference hashes](fixed/evaluation.json)",
            "","[Camera selection, failures and paired independent target results](nbv.md) · [Reproduction and licensed access](reproduction.md) · [Study synthesis and T0 comparison](../../general-closeout/README.md)"]
    (out/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return pins


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();print(json.dumps(publish(a.store,a.out),sort_keys=True))
