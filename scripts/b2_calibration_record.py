"""Record the C6 algorithm, incremental risk fingerprints, collective score candidates and opened calibration."""
import argparse
import json
from pathlib import Path
import sys

import b2_calibration_production as production
import b2_intake
from b2_intake_record import ROOT, locate, record_step, sealed_member
from t0_dtu_archive import check, digest


def read(path):
    return json.loads(path.read_bytes())


def scientific_files():
    files = {
        "scripts/b2_calibration_record.py", "scripts/b2_calibration_fixture.py",
        "scripts/b2_intake_cohort.py", "tests/test_b2_calibration.py", "tests/test_b2_calibration_records.py",
    }
    for mod in tuple(sys.modules.values()):
        origin = getattr(mod, "__file__", None)
        if not origin:
            continue
        path = Path(origin).resolve()
        if any(path.is_relative_to(ROOT / part) for part in ("scripts", "src")):
            files.add(path.relative_to(ROOT).as_posix())
    return {name: digest((ROOT / name).read_bytes()) for name in sorted(files)}


def selected(store, stage):
    return read(store / "work/controller" / stage / "selected.json")


def original_cell(roots, obj, cond):
    tag = obj + "-" + cond
    base = roots[0] / "work/controller"
    score = base / ("c6-score-" + tag) / "selected.json"
    back = base / ("b2-c6-readback-" + tag) / "selected.json"
    failed = base / ("b2-c6-failure-" + tag) / "selected.json"
    if score.is_file() and back.is_file():
        rb = read(back)
        return {"status":"scored", "case":sealed_member(roots, rb["run_id"], "payload/case/case.json"),
                "readback":rb, "scores":read(score)}
    if failed.is_file():
        pf=read(failed)
        disposition=read(locate(roots,pf))
        return {"status":disposition["status"], "reason":disposition["reason"], "disposition":pf}
    return None


def cell_parents(cells):
    return {role+tag:cell[role] for tag,cell in cells.items()
            for role in ("scores","case","readback","disposition") if role in cell}


def action(args):
    roots=(args.store.resolve(), args.batch_one.resolve(), args.history.resolve())
    stage=roots[0]
    cohort=selected(stage,"b2-dtu-cohort")
    roster=read(locate(roots,cohort))
    base={"cohort":cohort}
    names=scientific_files()
    cfg={"read_only_roots":[str(r) for r in roots]}
    def run(tag,job,member,config,pins,estimate=8*1024**2):
        return record_step(roots,tag,"b2_calibration_production.py","payload/"+member,
                           {**cfg,**config},pins,worker_args=(job,),extra_code=tuple(names),estimate=estimate)
    if args.action=="lock":
        prior=selected(roots[1],"freeze")
        release=read(locate(roots,prior))["sources"]["release"]
        pins={**base,"prior_freeze":prior,"release":release,"validation":selected(stage,"b2-calibration-validation"),
              "reconstruction":selected(stage,"b2-reconstruction-seed-43171")}
        return run("b2-calibration-method","lock","calibration-method.json",{"scientific_files":names},pins)
    base["method"]=selected(stage,"b2-calibration-method")
    spec=read(locate(roots,base["method"]))
    for rel,h in spec["scientific_files"].items():
        check(digest((ROOT/rel).read_bytes())==h, "changed calibrated implementation: "+rel)
    if args.action in ("fingerprints","candidates"):
        terminal={}
        for obj in roster["splits"]["C6"]:
            for cond in ("M2","M4"):
                tag=obj+"-"+cond
                cell=original_cell(roots,obj,cond)
                if cell is not None:
                    pins={**base,**cell_parents({tag:cell})}
                    terminal[tag]=run("b2-risk-fingerprint-"+tag,"fingerprint","fingerprint.json",
                                      {"object_id":obj,"condition":cond,"cell":cell},pins)
        if args.action=="fingerprints":
            return {"fingerprints":terminal,"planned":12}
        matrix_path=args.matrix or stage/"work/controller/c6-closed-matrix.json"
        raw=matrix_path.read_bytes()
        matrix=json.loads(raw)
        check(matrix["schema"]=="el-b2-c6-closed-matrix-1" and matrix["cohort"]==cohort and
              len(terminal)==12 and set(matrix["cells"])==set(terminal),"incomplete C6 closure")
        for tag,fp in terminal.items():
            check(read(locate(roots,fp))["cell"]==matrix["cells"][tag],"score-only matrix differs from per-cell record")
        pins={**base,**cell_parents(matrix["cells"])}
        sealed=run("b2-C6-score-only-matrix","matrix","closed-matrix.json",{"matrix_snapshot":matrix,
                   "matrix_bytes_hex":raw.hex(),"matrix_sha256":digest(raw)},pins)
        check(sealed["sha256"]==digest(raw),"closed matrix byte identity")
        return run("b2-C6-score-candidates","candidates","score-candidates.json",{},
                   {**pins,"matrix":sealed,**{"fp"+tag:fp for tag,fp in terminal.items()}})
    check(args.opening and args.reference_index,"explicit collective opening and C6 references required")
    opening=read(args.opening);refs=read(args.reference_index)
    check(refs["opening"]==opening and refs["cohort"]==cohort and set(refs["references"])==set(roster["splits"]["C6"]),"wrong actual C6 reference list")
    return run("b2-C6-calibration","fit","calibration-results.json",{},
               {**base,"opening":opening,"candidates":selected(stage,"b2-C6-score-candidates"),
                **{"ref"+obj:fp for obj,fp in refs["references"].items()}},32*1024**2)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("lock","fingerprints","candidates","fit"))
    for key in ("store","batch-one","history","matrix","opening","reference-index"):
        parser.add_argument("--"+key,type=Path,required=key in ("store","batch-one"),
                            default=ROOT/".research" if key=="history" else None)
    print(json.dumps(action(parser.parse_args()),sort_keys=True))
