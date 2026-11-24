"""Exercise source access, immutable queue receipts and a real multi-store recorder DAG."""

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evidence_lens._bundle import encoded, load, sha
from b2_worker_queue import completed, spec


def test_outside_calibration_is_rejected_before_source(tmp_path, monkeypatch):
    import b2_reconstruct as mod

    authority = tmp_path / "authority.json"
    cohort = tmp_path / "cohort.json"
    authority.write_bytes(
        encoded(
            {
                "optimizer_seed": 43171,
                "schema": "el-b2-reconstruction-origin-1",
                "study_sha256": sha(mod.STUDY),
            }
        )
    )
    cohort.write_bytes(encoded({"splits": {"C6": ["set48"], "T1": ["set100"]}}))
    opened = []

    def read(roots, pin):
        opened.append(pin)
        return {"authority": authority, "cohort": cohort}[pin]

    monkeypatch.setattr(mod, "resolve", read)
    with pytest.raises(ValueError, match="authorized fixed C6"):
        mod.prepare(tmp_path, tmp_path, "authority", "source", "set100", "M2", "M2-fixed", cohort="cohort")
    assert opened == ["authority", "cohort"]


def test_wrong_seed_rejected_before_exposed_data(tmp_path):
    import b2_reconstruct as mod

    with pytest.raises(ValueError, match="Unregistered seed"):
        mod.register(tmp_path, tmp_path, 137101)
    with pytest.raises(ValueError, match="exposed subjects"):
        mod.development_source(tmp_path, tmp_path, {}, "set100", {})


@pytest.mark.parametrize("chosen", [[7, 7], [24, 31], [7, 8], [7]])
def test_bad_pair_cannot_open_photos(tmp_path, monkeypatch, chosen):
    from b2_worker_development import supply
    import dtu_sample

    def fail(*args, **kwargs):
        pytest.fail("network used before selection check")

    monkeypatch.setattr(dtu_sample, "Network", fail)
    study = load(ROOT / "protocol/b2-reconstruction-v1.json")
    roles = {}
    for name, doc in {
        "study": study,
        "authority": {"base_recipe_sha256": study["base_recipe_sha256"]},
        "start": {"object_id": "set1", "checkpoint_sha256": "a" * 64},
        "decision": {
            "schema": "el-b2-nbv-selection-1",
            "object_id": "set1",
            "strategy_id": "risk",
            "pool": study["development_selection"]["candidate_ids"],
            "selected_official_ids": chosen,
            "m2_checkpoint_sha256": "a" * 64,
        },
    }.items():
        roles[name] = tmp_path / (name + ".json")
        roles[name].write_bytes(encoded(doc))
    cfg = {
        "object_id": "set1",
        "strategy_id": "risk",
        "layout": "M4-selected",
        "sources": {"decision": {"sha256": sha(roles["decision"])}},
    }
    with pytest.raises(ValueError, match="independently sealed"):
        supply(cfg, roles, tmp_path / "new")


def test_queue_reads_original_bytes_and_rejects_changed_files(tmp_path):
    job = {"trial_id": "b2-fixture", "job_sha256": "f" * 64, "package": {"sha256": "c" * 64}}
    f = tmp_path / "archive.zip"
    f.write_bytes(b"original")
    record = {
        "schema": "el-t0-windows-return-1",
        "trial_id": job["trial_id"],
        "job_sha256": job["job_sha256"],
        "package_sha256": job["package"]["sha256"],
        "runs": [],
        "status": "failed",
        "files": {f.name: spec(f)},
    }
    m = tmp_path / "completed.json"
    m.write_bytes(encoded(record))
    row = completed(job, tmp_path)
    assert row["files"] == 1 and row["return_bytes"] == len(b"original") + m.stat().st_size
    extra = tmp_path / "other.zip"
    extra.write_text("extra")
    with pytest.raises(ValueError, match="Unexpected"):
        completed(job, tmp_path)
    extra.unlink()
    f.write_text("changed")
    with pytest.raises(ValueError, match="Changed"):
        completed(job, tmp_path)


def test_two_readonly_store_roots_have_real_verified_ancestry(tmp_path, capsys):
    from evidence_lens import recorder
    from b2_reconstruct_history import register
    from b2_reconstruct_receive import historical_roots
    from t0_records import bind

    src = tmp_path / "repo/scripts"
    src.mkdir(parents=True)
    (src / "fixture.py").write_text(
        "from pathlib import Path\nimport sys\nPath(sys.argv[1],'x').write_text('value')\n"
    )
    policy = tmp_path / "config.json"
    policy.write_text("{}")

    def put(store, trial, parent=None):
        args = [
            "--root",
            str(store),
            "--source-root",
            str(src.parent),
            "--kind",
            "archive_verify",
            "--experiment",
            "b2-fixture",
            "--config",
            str(policy),
            "--trial",
            trial,
        ]
        if parent:
            args += ["--parent", parent["run_id"] + "=" + parent["manifest_sha256"]]
        assert recorder.main(args + ["--", "{python}", "{source}/scripts/fixture.py", "{out}"]) == 0
        rid = json.loads(capsys.readouterr().out)["run_id"]
        return bind(store, rid, "payload/x")

    old, first, new = [tmp_path / x for x in ("old", "first", "new")]
    a = put(old, "old")
    b = put(first, "first", a)
    c = put(new, "new", b)
    registry = tmp_path / "fixture-registry.json"
    registry.write_bytes(encoded({"schema": "el-b2-windows-batch-1", "jobs": [{"ticket": c}]}))
    pr = register(new, first, old, "synthetic", [registry])
    docs, anchors = historical_roots(new, first, old, pr["run_id"], pr["manifest_sha256"], pr["sha256"])
    assert docs == [sha(registry)] and set(anchors) == {b["run_id"]}
    proof = load(new / "runs" / pr["run_id"] / pr["member"])
    assert proof["verified_runs"] == 2 and proof["checked_manifests"][a["run_id"]] == a["manifest_sha256"]


def test_aliased_fit_is_charged_once_and_two_photo_accesses_are_counted():
    from b2_reconstruct_cost import unique_cost

    def metrics(rid, seconds):
        return {"run_id": rid, "recorder_elapsed_s": seconds, "sampled_tree_rss_peak_bytes": 18}

    common = metrics("same-fit", 32.5)
    rows = [
        {
            "cost": {
                "physical_trial_id": "common-trial",
                "windows_return_object_bytes_including_summary": 160,
                "source_candidate_http_body_bytes": 180,
                "stages": {"source": metrics(rid, 0.5), "generation": common},
            }
        }
        for rid in ("uniform-source", "risk-source")
    ]
    costs = unique_cost(rows)
    assert (
        costs["physical_reconstructions"] == 1
        and costs["windows_return_object_bytes_including_summary"] == 160
    )
    assert costs["source_candidate_http_body_bytes"] == 360 and costs["stages"]["source"]["unique_runs"] == 2
    assert costs["stages"]["generation"]["unique_runs"] == 1
    assert costs["stages"]["generation"]["sum_recorder_elapsed_s"] == 32.5
