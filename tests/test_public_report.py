"""Generate a public report from the sealed, explicitly synthetic end-to-end study fixture."""

import csv
import re
import runpy
from pathlib import Path

import pytest
from PIL import Image

from evidence_lens import formal_evaluation, formal_report
from evidence_lens._bundle import load, sha


def test_visibility_percentages_use_their_own_denominators():
    root = Path(__file__).resolve().parents[1]
    functions = runpy.run_path(str(root / "scripts/export_public_report.py"))
    families = ("a", "b", "c", "d")
    regional = {(f, k): (0, 0) for f in families for k in (*functions["VISIBILITY"], "all")}
    regional.update(
        {
            ("a", "all"): (2, 4),
            ("a", "observed"): (1, 3),
            ("a", "disoccluded"): (1, 1),
            ("b", "all"): (0, 1),
            ("b", "outside_input_fov"): (0, 1),
        }
    )
    report = functions["visibility_text"](families, regional)
    assert "| All evaluable pixels | 2 | 5 | 100.00 | 25.00 | 2/4 |" in report
    assert "| Visible in the source view | 1 | 3 | 37.50 | 33.33 | 1/4 |" in report
    assert "| Background or invalid reference geometry | 0 | 0 | 0.00 | — | 0/4 |" in report
    regional["a", "all"] = (2, 5)
    with pytest.raises(ValueError, match="partition"):
        functions["visibility_text"](families, regional)


def test_aggregate_report_plot_check_and_privacy(tmp_path, capsys, monkeypatch):
    pytest.importorskip("sklearn", minversion="1.9.1")
    pytest.importorskip("pyarrow")
    pytest.importorskip("matplotlib")
    import test_formal_evaluation
    from test_recorded_l1 import build_recorded_learning_fixture
    from test_release_model import scientific_report

    root = Path(__file__).resolve().parents[1]
    export = runpy.run_path(str(root / "scripts/export_public_report.py"))["export"]
    f = build_recorded_learning_fixture(tmp_path, capsys, monkeypatch, full_protocol=True)
    repo = f["repo"]
    monkeypatch.setattr(test_formal_evaluation, "ROOT", repo)
    monkeypatch.setattr(formal_evaluation, "__file__", str(repo / "src/evidence_lens/formal_evaluation.py"))
    monkeypatch.setattr(formal_report, "__file__", str(repo / "src/evidence_lens/formal_report.py"))
    pin = scientific_report(tmp_path, f, capsys)
    out = tmp_path / "public"
    with pytest.raises(ValueError, match="pinned successful study report"):
        export(f["store"], pin, out)
    built = export(f["store"], pin, out, fixture=True)
    report = out / "docs/results/README.md"
    public = load(out / "docs/results/provenance.json")
    assert (
        public["report_run"] == pin
        and public["training_run"] == f["fit"]
        and public["kind"] == "synthetic_fixture"
    )
    assert public["document_files"]["README.md"]["sha256"] == sha(report)
    assert set(built["binary_examples"]) == set(public["binary_examples"])
    assert len(built["binary_examples"]) == 4
    for rel, spec in built["binary_examples"].items():
        path = out / rel
        assert spec == {"format": "rgb8-png", "bytes": path.stat().st_size, "sha256": sha(path)}
        with Image.open(path) as png:
            assert png.mode == "RGB" and png.format == "PNG" and not png.info
    with (report.parent / "family-methods.csv").open() as stream:
        family = list(csv.DictReader(stream))
    assert len(family) == 60
    with (report.parent / "paired-changes.csv").open() as stream:
        paired = list(csv.DictReader(stream))
    assert len(paired) == 8 and any(int(row["matched_AP_views"]) < 6 for row in paired)
    source = report.read_text()
    assert "Synthetic integration fixture" in source and "Confirmed errors / reference" in source
    dev_member = "numeric/evidence/" + f["threshold"]["run_id"] + "/development.json"
    dev_path = f["store"] / "runs" / f["threshold"]["run_id"] / "payload/development.json"
    assert public["sources"][dev_member]["sha256"] == sha(dev_path)
    op = load(dev_path)["methods"]["L1-full"]
    assert f"**{100 * op['coverage']:.2f}%**" in source and "on the four development families" in source
    assert "common-mask AP" in source and "unadjusted for this shift in prevalence" in source
    assert "compared files" in source and "confidential-machine-fixture" not in source
    assert (
        str(tmp_path) not in source and str(tmp_path) not in (report.parent / "provenance.json").read_text()
    )
    for target in re.findall(r"\]\(([^)]+)\)", source):
        assert (report.parent / target).is_file() or (root / "docs/results" / target).resolve().is_file(), (
            target
        )
