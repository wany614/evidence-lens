"""Recalculate storage and timing from all five registered pilot families and their actual archive containers."""

import argparse
import json
import shutil
from pathlib import Path

from evidence_lens._bundle import files, load, sha
from evidence_lens._capacity import bytes_on_tree
from evidence_lens._verify import require
from evidence_lens.scoring_io import new_output, write_json
from formal_lineage import family_record

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {"reference": 24, "readback": 24, "generation": 28, "render": 32, "score": 32,
          "evaluation": 32, "recompute": 32, "extract": 20}


def calculate(store, containers):
    catalog = ROOT / "protocol/families-v1.json"
    config = load(ROOT / "protocol/study-v1.json")
    families = config["storage"]["representative_families"]
    measured = {}
    for family in families:
        row = family_record(catalog, ROOT / "configs/formal-freeze-v1.json", family)
        selpath = store / "trials" / row["trial_id"] / "trial_selection.json"
        selected = load(selpath)
        ids = {role:record["run_id"] for role,record in selected["selected"].items()}
        ids["extract"] = load(store / "formal" / sha(catalog) / "extracts" / (family+".json"))["run_id"]
        data = {}
        for role, rid in ids.items():
            tree = store / "runs" / rid
            record = load(tree / "run.json")
            require(record["execution_status"] == "succeeded" and record["scene_family_id"] == family, "wrong pilot identity")
            data[role] = {"run_id": rid, "manifest_sha256":sha(tree/"manifest.json"),
                          "logical_bytes": sum(p.stat().st_size for p in files(tree).values()),
                          "elapsed_s": record["elapsed_seconds_monotonic"], "sampled_peak_rss_bytes":record["sampled_peak_tree_rss_bytes"]}
        packed = load(containers / (family+"-pack.json"))
        require(packed["family"] == family and all(packed["run_manifests"].get(r["run_id"]) == r["manifest_sha256"]
                for r in data.values()), "family archive does not contain the selected stages")
        measured[family] = {"selection_sha256": sha(selpath), "runs": data, "container": {k: packed[k] for k in ("name","bytes","sha256")},
                            "ply_bytes":(store/"runs"/ids["generation"]/"payload/scene.ply").stat().st_size}
    maxima = {k:max(r["runs"][k]["logical_bytes"] for r in measured.values()) for k in COUNTS}
    q1 = maxima["generation"] + max(r["ply_bytes"] for r in measured.values())
    phases = {k:{"planned":n, "unit_logical_bytes":maxima[k], "total":n*maxima[k]} for k,n in COUNTS.items()}
    phases["Q1-derivation"]={"planned":4, "unit_logical_bytes":q1, "total":4*q1}
    shared=load(ROOT/"configs/windows-b0.json")["generator"]["checkpoint_bytes"]
    science=sum(v["total"] for v in phases.values())+shared
    repeats=6*maxima["generation"]+3*q1+9*maxima["render"]
    reserve=config["storage"]["working_margin_fraction"]
    return {"schema":"el-five-family-budget-1","pilot_families":measured,"max_stage_projection":phases,
            "study_logical_with_one_shared_checkpoint_bytes":science, "timing_repeats_additional_logical_estimate":repeats,
            "projection_including_repeats_with_margin":int((science+repeats)*(1+reserve)),
            "shared_checkpoint_bytes":shared,"working_margin_fraction":reserve,
            "physical_family_container_range_bytes":[min(r["container"]["bytes"] for r in measured.values()),max(r["container"]["bytes"] for r in measured.values())],
            "distinct_project_store_file_bytes":bytes_on_tree(store),"volume_free_bytes":shutil.disk_usage(store).free,
            "caveats":["The logical projection counts repeated embedded inputs; the family ZIP stores exact duplicate bytes once within the same family.",
                       "R1 conservatively uses B0 size. Q1 adds one embedded PLY. The upstream checkpoint stays external to private cloud containers.",
                       "P4 fit, comparison outputs, temporal overlap, provider-managed cache and future failures need separate measurements."]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("store","containers","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    report=calculate(a.store.resolve(),a.containers)
    with new_output(a.output) as out:
        write_json(out/"budget.json",report)
        lines=["# Five-family resource measurements","", "| Family | Sealed family ZIP, MiB | Generation, s | Render, s | Generation RSS, GiB |",
               "| --- | ---: | ---: | ---: | ---: |"]
        for k,v in report["pilot_families"].items():
            g,r=v["runs"]["generation"],v["runs"]["render"]
            lines.append(f"| {k} | {v['container']['bytes']/2**20:.2f} | {g['elapsed_s']:.2f} | {r['elapsed_s']:.2f} | {g['sampled_peak_rss_bytes']/2**30:.2f} |")
        lines+=["",f"The 32-condition logical estimate is {report['study_logical_with_one_shared_checkpoint_bytes']/2**30:.2f} GiB. "
                f"Including nine timing repeats and a 50% margin gives {report['projection_including_repeats_with_margin']/2**30:.2f} GiB.","",*report["caveats"]]
        (out/"budget.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("pilot_families","max_stage_projection")}))
if __name__=="__main__":
    main()
