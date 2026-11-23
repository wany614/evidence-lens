"""Publication projections on format-only fixtures, with no experiment files or held-out scenes."""

from copy import deepcopy
import json

import pytest

from evidence_lens.formal_inputs import signature, verified_file
from evidence_lens.multisource import CONDITIONS, METHODS, SCOPES, TARGETS
from evidence_lens.scoring_io import write_json
from stage3_risk_joint import ORIGINAL, audit_cells, copy_original_tables, native_rows
from stage3_risk_sources import CATALOG, asset


def test_original_table_set_and_bytes(tmp_path):
    pub = tmp_path / "payload/public-numerical"
    pub.mkdir(parents=True)
    p = {"run_id": "20200101T000000000000Z-1234567890abcdef"}
    header = {
        "models": {},
        "families": [],
        "method_sha256": "0" * 64,
        "opening_sha256": "1" * 64,
        "dataset_sha256": "2" * 64,
    }
    for n in ORIGINAL:
        path = pub / n
        if n == "study-summary.json":
            write_json(path, {**header, "scientific_run_id": p["run_id"], "split": "test"})
        else:
            path.write_bytes(b"fixture,empty\r\n")
    write_json(
        pub / "manifest.json",
        {
            "schema": "el-stage3-public-numerical-manifest-1",
            "source_run_id": p["run_id"],
            "files": [asset(pub, pub / n) for n in ORIGINAL],
        },
    )

    class Fixture:
        pin = p
        files = {x.relative_to(tmp_path).as_posix(): signature(x) for x in pub.iterdir()}

        def open(self, n):
            return verified_file(tmp_path, n, self.files[n])

    ev = {**header, "public_numerical": asset(tmp_path / "payload", pub / "manifest.json")}
    dst = tmp_path / "publish"
    dst.mkdir()
    copy_original_tables(Fixture(), ev, dst)
    assert (dst / "confirmation-manifest.json").read_bytes() == (pub / "manifest.json").read_bytes()
    assert all((dst / n).read_bytes() == (pub / n).read_bytes() for n in ORIGINAL)
    (pub / "public-extra.csv").write_text("outside scope")
    j = json.loads((pub / "manifest.json").read_bytes())
    j["files"].append(asset(pub, pub / "public-extra.csv"))
    write_json(pub / "manifest.json", j)
    Fixture.files["payload/public-numerical/manifest.json"] = signature(pub / "manifest.json")
    ev["public_numerical"] = asset(tmp_path / "payload", pub / "manifest.json")
    with pytest.raises(ValueError, match="unregistered confirmation"):
        copy_original_tables(Fixture(), ev, dst)


def test_acquisition_pairs_unique_and_failed_support_retained():
    d = {s: {"cells": []} for s in ("train", "development", "test")}
    trials = []
    for split, n in zip(d, (8, 4, 4), strict=True):
        for i in range(n):
            for cond in CONDITIONS:
                f = split + "-fixture-" + str(i)
                x = {
                    "family": f,
                    "condition": cond,
                    "selected": {},
                    "selection_sha256": "a" * 64,
                    "failure": {"stage": "terminal engineering fixture"},
                }
                d[split]["cells"].append(x)
                trials.append(
                    {
                        "family": f,
                        "condition": cond,
                        "split": split,
                        "phase": "production",
                        "selection_status": "terminal_failure",
                        "selection_sha256": "a" * 64,
                        "selection": {},
                    }
                )
    trials += [{"phase": "engineering-pilot"} for _ in range(4)] + [{"phase": "engineering-repeat"}]
    doc = {
        "schema": "el-stage3-joint-acquisition-costs-1",
        "catalog_sha256": CATALOG,
        "opening_sha256": "b" * 64,
        "trials": trials,
    }
    audit_cells(d, doc, "b" * 64)
    wrong = deepcopy(doc)
    wrong["trials"][1] = wrong["trials"][0]
    with pytest.raises(ValueError, match="repeated"):
        audit_cells(d, wrong, "b" * 64)
    wrong = deepcopy(doc)
    wrong["trials"][2]["selection_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="disagree"):
        audit_cells(d, wrong, "b" * 64)


def test_native_alias_does_not_get_a_second_timing():
    p = {"results": []}
    sources = {}
    for m in METHODS:
        views = []
        for c in CONDITIONS:
            for v in TARGETS:
                target = {"condition": c, "view": v, "modes": {}}
                for scope in SCOPES:
                    alias = "two" if (c, scope) == ("M2", "all") else None
                    entry = {"file": None, "alias_of": alias}
                    target["modes"][scope] = entry
                    p["results"].append(
                        {
                            "method": m,
                            "condition": c,
                            "view": v,
                            "scope": scope,
                            "alias_of": alias,
                            "status": "render_failed",
                            "threshold": 0.5,
                            "cpu": None,
                            "comparison": None,
                        }
                    )
                views.append(target)
        sources[m] = {"views": views}
    rows = native_rows(p, sources)
    assert len(rows) == 144 and sum(r["alias_of"] is None for r in rows) == 120
    assert all(r["launched_process_wall_seconds"] is None and r["max_abs"] is None for r in rows)
