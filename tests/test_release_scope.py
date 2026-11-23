"""The prerelease tag, scope and scientific count disclosure come from the selected records."""
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from release_scope import check_scope, graded_views, python_version, summary_bytes
from release_smoke import METHODS, VIEWS, distributions, preserve_partial, verify_files


def test_tag_and_scope_refuse_mixed_scientific_sources():
    tag = "v0.1.1-rc.1"
    assert python_version(tag) == "0.1.1rc1" and python_version("v0.1.0") == "0.1.0"
    assert len(graded_views(tag)) == 6 and graded_views("v0.1.0") == ("v01",)
    for wrong in ("0.1.1rc1", "v0.1.1rc1", "v0.1.1-rc.0", "v0.1.1-rc.01", "v0.01.1"):
        with pytest.raises(ValueError, match="positive candidate"):
            python_version(wrong)
    pin = {"run_id": "20260101T000000000000Z-0123456789abcdef", "manifest_sha256": "d" * 64}
    scope = {"schema":"el-prerelease-scope-1","version":tag,"publication_status":"private_candidate",
             "preview_class":"technical_preview","formal_evidence_status":"original_study_under_review",
             "model_id":pin["run_id"],"model_index_sha256":"a"*64,"method_freeze_sha256":"b"*64,
             "report":pin,"example_scores":pin,"diagnostics":[dict(pin,path="payload/measurement.json",sha256="c"*64)],
             "finding":"Test measurement.","limitations":["One fixture."]}
    args = (tag,"b"*64,pin["run_id"],"a"*64,{"report":pin,"example_scores":pin})
    check_scope(scope,*args)
    for changed in (dict(scope,publication_status="published"),dict(scope,preview_class="experimental_research_preview"),
                    dict(scope,model_id="different"),dict(scope,diagnostics=[]),
                    dict(scope,diagnostics=[dict(pin,path="inputs/private.json",sha256="c"*64)])):
        with pytest.raises(ValueError):
            check_scope(changed,*args)


def test_summary_separates_low_unknown_and_total_outside_high(tmp_path):
    p = tmp_path / "counts.csv"
    def write(limits):
        with p.open("w",encoding="utf-8",newline="") as f:
            w=csv.writer(f);w.writerow(("condition","family","method","region","V","E","H","Y","HY","LY","UY"))
            for c in ("B0","R1","Q1"):
                for k in range(4):
                    w.writerow((c,str(k),"L1-full","all",*limits))
    scope={"preview_class":"technical_preview","version":"v0.1.1-rc.1","formal_evidence_status":"original_study_under_review",
           "finding":"Recorded fixture.","limitations":["Four related scenes."],"model_id":"test","model_index_sha256":"a"*64,
           "report":{"run_id":"report"},"example_scores":{"run_id":"demo"},"diagnostics":[]}
    write((20,15,6,10,5,3,2))
    rendered=summary_bytes(scope,p).decode()
    assert "| B0 | 80 | 60 | 24 | 40 | 20 | 12 | 8 | 20 |" in rendered
    assert "| B0 | 30.00% | 40.00% | 83.33% | 50.00% |" in rendered
    write((20,15,6,10,5,3,""))
    assert "| R1 | unavailable |" in summary_bytes(scope,p).decode()
    write((20,15,6,10,5,3,1))
    with pytest.raises(ValueError,match="partition"):
        summary_bytes(scope,p)


def test_smoke_names_changed_files_and_retains_failed_distribution(tmp_path):
    extracted=tmp_path/"bundle";extracted.mkdir()
    (extracted/"release.json").write_text(json.dumps({"schema":"el-public-model-release-1","version":"v0.1.1-rc.2","files":{"lost.bin":{}}}))
    (extracted/"extra.bin").write_bytes(b"x")
    with pytest.raises(ValueError,match="missing=.*lost.bin.*extra=.*extra.bin"):
        verify_files(extracted,None)
    values={"finite_expected":4,"finite_actual":4,"pixels":6,"max_absolute_score_difference":0.,
            "validity_equal":True,"states_equal":True,"passed":True}
    grade={"views":{v:{"methods":{m:dict(values) for m in METHODS},"l0_reason_codes_equal":True} for v in VIEWS}}
    assert sum(map(len,distributions(grade).values()))==30
    failing=grade["views"]["v04"]["methods"]["L1-geometry"]
    failing.update(finite_expected=0,finite_actual=0,max_absolute_score_difference=None)
    output=tmp_path/"checked"
    with pytest.raises(ValueError,match="v04/L1-geometry.*partial smoke evidence saved") as caught:
        with preserve_partial(tmp_path/"partial",output,"c"*64) as stage:
            (stage/"grade.json").write_text(json.dumps(grade))
            distributions(grade)
    failed,=tmp_path.glob("checked-failed-*")
    assert not output.exists() and str(failed) in str(caught.value)
    assert (failed/"results/grade.json").is_file() and json.loads((failed/"failure.json").read_text())["passed"] is False
    failing.update(values,max_absolute_score_difference=float("nan"))
    with pytest.raises(ValueError,match="v04/L1-geometry.*maximum"):
        distributions(grade)
    failing.update(values,states_equal=False)
    with pytest.raises(ValueError,match="v04/L1-geometry: states_equal"):
        distributions(grade)
