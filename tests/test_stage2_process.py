"""Stage B's new dispatch edges, closed-cohort protection and terminal selection contract."""

import copy
import importlib
from pathlib import Path

import pytest

from evidence_lens._bundle import BundleError, encoded, load, new_id, sha


@pytest.fixture
def processor(monkeypatch):
    project = Path.cwd()
    stage = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project / "scripts"))
    monkeypatch.syspath_prepend(str(stage / "scripts"))
    return importlib.import_module("stage2_process")


def test_unknown_and_closed_rosters_never_read_data(processor, tmp_path, monkeypatch):
    project = Path.cwd()

    def unexpected(*args):
        pytest.fail("A scientific input was touched before enrollment or opening")

    monkeypatch.setattr(processor, "handoff", unexpected)
    monkeypatch.setattr(processor, "collect_returns", unexpected)
    catalog = Path(__file__).resolve().parents[1] / "protocol/stage2-families-v1.json"
    args = [
        "--project",
        str(project),
        "--store",
        str(tmp_path),
        "--catalog",
        str(catalog),
        "--catalog-sha256",
        sha(catalog),
        "--returns-dir",
        str(tmp_path),
    ]
    rows = load(catalog)["families"]
    for row in rows:
        if row["split"] == "test":
            with pytest.raises(BundleError, match="still sealed"):
                processor.main([*args, "--family", row["scene_family_id"]])
    with pytest.raises(BundleError, match="Unknown or duplicate family"):
        processor.main([*args, "--family", "scene_025_wrong"])
    assert not list(tmp_path.iterdir())


def synthetic_worker(processor, root, status="succeeded"):
    rid, gen, ref, check = [new_id() for _ in range(4)]

    def pin(name, digest):
        return {"run_id": name, "manifest_sha256": digest * 64}

    refp, genp, checkp = pin(ref, "a"), pin(gen, "b"), pin(check, "c")
    identity = {
        "trial_id": "fixture-B0",
        "scene_family_id": "fixture",
        "scene_id": "fixture/v001",
        "condition_id": "B0",
        "split": "train",
        "scene_version": "v001",
        "dataset_id": "fixture",
        "dataset_hash": "7" * 64,
        "scene_hash": "8" * 64,
        "camera_set_hash": "4" * 64,
    }
    dispatch = {
        "identity": identity,
        "reference": refp,
        "readback": checkp,
        "worker_source_sha256": "9" * 64,
        "handoff_sha256": "2" * 64,
    }
    originals = {"input_sha256": "3" * 64, "camera_sha256": "4" * 64}
    gen_meta = {
        "artifacts": {
            "payload/scene.ply": {"sha256": "5" * 64},
            "payload/generation.json": {"sha256": "6" * 64},
        }
    }

    class Verified:
        def get(self, run_id, kind=None):
            assert run_id == gen and kind == "generation"
            return gen_meta

    ctx = {
        "store": root,
        "project": root,
        "row": identity,
        "refmeta": originals,
        "dispatch": dispatch,
        "verified": Verified(),
    }
    directory = root / "runs" / rid
    directory.mkdir(parents=True)
    settings = {
        **identity,
        "formal_handoff_sha256": dispatch["handoff_sha256"],
        "worker_source_sha256": dispatch["worker_source_sha256"],
        "upstream_reference": refp,
        "independent_readback": checkp,
    }
    (directory / "config.resolved.yaml").write_bytes(encoded(settings))
    inputs = [
        {
            "parent_run_id": ref,
            "files": [{"sha256": originals["camera_sha256"]}, {"sha256": originals["input_sha256"]}],
        },
        {"parent_run_id": gen, "files": [{"sha256": "5" * 64}, {"sha256": "6" * 64}]},
    ]
    run = {
        "run": {
            **identity,
            "run_id": rid,
            "run_kind": "render",
            "experiment_id": processor.EXP,
            "execution_status": status,
        },
        "inputs": {"parents": [refp, genp], "inputs": inputs},
    }
    return ctx, run, refp, genp


def test_cross_family_producer_bytes_and_readback_refused(processor, tmp_path):
    ctx, rec, ref, gen = synthetic_worker(processor, tmp_path)
    processor.validate_stage(ctx, rec, "render", ref, gen)
    for change, error in (
        (lambda r: r["run"].update(scene_family_id="a different family"), "another family"),
        (lambda r: r["inputs"]["parents"][1].update(manifest_sha256="0" * 64), "exactly the selected"),
        (lambda r: r["inputs"]["inputs"][1]["files"].pop(), "exact worker input"),
        (lambda r: r["inputs"]["inputs"][0]["files"].pop(), "exact worker input"),
    ):
        bad = copy.deepcopy(rec)
        change(bad)
        with pytest.raises(BundleError, match=error):
            processor.validate_stage(ctx, bad, "render", ref, gen)
    settings = tmp_path / "runs" / rec["run"]["run_id"] / "config.resolved.yaml"
    metadata = load(settings)
    metadata["independent_readback"]["manifest_sha256"] = "0" * 64
    settings.write_bytes(encoded(metadata))
    with pytest.raises(BundleError, match="another handoff"):
        processor.validate_stage(ctx, rec, "render", ref, gen)


def test_corrupt_return_never_reaches_store(processor, tmp_path, monkeypatch):
    ctx, rec, ref, gen = synthetic_worker(processor, tmp_path / "store")
    rid = rec["run"]["run_id"]
    folder = tmp_path / "return"
    folder.mkdir()
    (folder / (rid + ".zip")).write_bytes(b"test")
    (folder / (rid + ".receipt.json")).write_bytes(
        encoded(
            {
                "trial_id": ctx["row"]["trial_id"],
                "run": rid,
                "schema": "el-worker-return-1",
                "run_kind": "render",
                "handoff_sha256": ctx["dispatch"]["handoff_sha256"],
                "package": rid + ".zip",
                "sha256": "8" * 64,
                "manifest_sha256": "1" * 64,
                "bytes": 4,
            }
        )
    )

    def unexpected(*args):
        pytest.fail("Corrupt transport reached the receiver")

    monkeypatch.setattr(processor, "receive", unexpected)
    with pytest.raises(BundleError, match="wrong length or digest"):
        processor.collect_returns(ctx, folder)


def test_terminal_failure_keeps_all_views_and_refuses_to_hide_success(processor, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from select_trial import transfer_evidence

    ctx, record, reference, _ = synthetic_worker(processor, tmp_path / "store", status="failed")
    rid = record["run"]["run_id"]
    ctx["opening_path"] = None
    ctx["catalog_hash"] = "7" * 64
    record["manifest_sha256"] = "d" * 64
    record["run"].update(
        run_kind="generation",
        failure_reason="synthetic fixture exhaustion",
        worker_exit_code=1,
        parent_run_ids=[reference["run_id"]],
    )
    record["inputs"]["parents"] = [reference]
    ctx["verified"] = SimpleNamespace(get=lambda _rid, kind=None: record)

    def accept(store, records):
        assert rid in records
        return {rid: [{"status": "accepted"}]}

    monkeypatch.setitem(transfer_evidence.__globals__, "transfer_evidence", accept)
    other = new_id()
    competitor = ctx["store"] / "runs" / other / "run.json"
    competitor.parent.mkdir()
    competitor.write_bytes(encoded({**record["run"], "run_id": other, "execution_status": "succeeded"}))
    pin = {"run_id": rid, "manifest_sha256": record["manifest_sha256"]}
    with pytest.raises(ValueError, match="would hide"):
        processor.terminal(ctx, pin)
    assert not (ctx["store"] / "trials").exists()
    competitor.unlink()
    path = processor.terminal(ctx, pin)
    result = load(path)
    assert result["schema"] == "el-vision-failed-trial-1" and result["selection_status"] == "failed"
    assert result["planned_targets"] == list(processor.TARGETS) and len(result["planned_targets"]) == 6
    assert result["reference"] == reference and result["failure"] == pin
    first = sha(path)
    assert processor.terminal(ctx, pin) == path and sha(path) == first
