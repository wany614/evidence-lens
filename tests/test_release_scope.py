"""The prerelease tag, scope and scientific count disclosure come from the selected records."""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from release_scope import check_scope, graded_views, python_version, summary_bytes


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
            w=csv.writer(f);w.writerow(("condition","family","method","region","Y","HY","LY","UY"))
            for c in ("B0","R1","Q1"):
                for k in range(4):
                    w.writerow((c,str(k),"L1-full","all",*limits))
    scope={"preview_class":"technical_preview","version":"v0.1.1-rc.1","formal_evidence_status":"original_study_under_review",
           "finding":"Recorded fixture.","limitations":["Four related scenes."],"model_id":"test","model_index_sha256":"a"*64,
           "report":{"run_id":"report"},"example_scores":{"run_id":"demo"},"diagnostics":[]}
    write((10,5,3,2))
    assert "| B0 | 40 | 20 | 12 | 8 | 20 |" in summary_bytes(scope,p).decode()
    write((10,5,3,""))
    assert "| R1 | unavailable |" in summary_bytes(scope,p).decode()
    write((10,5,3,1))
    with pytest.raises(ValueError,match="partition"):
        summary_bytes(scope,p)
