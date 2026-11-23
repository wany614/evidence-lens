"""The queue rehearses transport, binding and restore without visiting any confirmation payload."""

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from evidence_lens._bundle import BundleError, encoded, load, sha
from evidence_lens._verify import verify_bundle

# Keep these tests runnable in either the stage-two overlay or its installed checkout.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from make_handoff import publish
import run_stage2_confirmation as win
import stage2_confirmation as mac
import stage2_confirmation_archive as archive
from stage2_confirmation_rehearsal import exercise

ROOT = Path.cwd().resolve()


def test_closed_gate_and_repeatable_input_free_plan(tmp_path, monkeypatch, capsys):
    catalog = ROOT / "protocol/stage2-families-v1.json"
    output = tmp_path / "kit"
    result = mac.build_kit(ROOT, catalog, mac.CATALOG, output)
    assert mac.build_kit(ROOT, catalog, mac.CATALOG, output) == result
    win.main(["--plan", str(output / "plan.json"), "--plan-sha256", result["plan_sha256"], "--dry-run"])
    dry = json.loads(capsys.readouterr().out)
    assert dry["trial_count"] == 24 and len(dry["batches"]) == 16
    base = tmp_path / "unopened"
    marker = base / "protocol/stage2" / mac.CATALOG / "test-opening.json"
    a = SimpleNamespace(
        catalog=catalog,
        catalog_sha256=mac.CATALOG,
        project=ROOT,
        store=base,
        method_sha256="2" * 64,
        freeze_run="20261123T000000000000Z-" + "a" * 16,
        freeze_manifest="3" * 64,
        opening_sha256="4" * 64,
        plan_sha256=result["plan_sha256"],
        plan=output / "plan.json",
    )
    with pytest.raises(BundleError, match="Opening absent"):
        mac.authority(a)
    assert not base.exists()
    publish(marker, encoded({"schema": "el-stage2-test-opening-1"}))
    with pytest.raises(BundleError, match="Opening absent"):
        mac.authority(a)
    assert not (base / "runs").exists() and not (base / "trials").exists()
    with zipfile.ZipFile(output / "windows-plan.zip") as z:
        assert set(z.namelist()) == {"plan.json", *mac.WINS}


def test_packing_deferral_recovers_and_independent_family_restoration(tmp_path):
    root = tmp_path / "fixture"
    proof = exercise(root)
    assert proof["receipt_before_zip"] == "in_transit" and proof["windows_stage_launches"] == 1
    assert proof["receiver_dispositions"] == ["accepted", "identical"]
    store = root / "receiver"
    rid = proof["producer_run_id"]
    pin = {rid: proof["manifest_sha256"]}
    extra = root / "extra.json"
    publish(extra, encoded({"test": "synthetic transport only"}))
    doc, origins = archive.file_map(
        store, root / "shelf", pin, {"protocol/example.json": extra}, None, {"synthetic": True}
    )
    path = root / "private.zip"
    packed = archive.write_archive(doc, origins, path, 0)
    assert archive.write_archive(doc, origins, path, 0) == packed
    independent = root / "remote-return.zip"
    shutil.copyfile(path, independent)
    restored = archive.verify_archive(independent, packed["container_sha256"], root / "blank-store")
    assert restored["restored"] and restored["logical_files"] > 8 and restored["run_manifests"] == pin
    assert verify_bundle(root / "blank-store/runs" / rid)["manifest_sha256"] == proof["manifest_sha256"]
    assert (root / "blank-store/protocol/example.json").read_bytes() == extra.read_bytes()
    with pytest.raises(BundleError, match="exclusive"):
        archive.verify_archive(independent, packed["container_sha256"], root / "blank-store")
    bad = root / "changed.zip"
    shutil.copyfile(path, bad)
    with bad.open("ab") as dst:
        dst.write(b"changed")
    with pytest.raises(BundleError, match="Wrong private"):
        archive.verify_archive(bad, packed["container_sha256"])


def test_mixed_failed_and_delayed_originals_keep_all_attempts(tmp_path):
    tid = "synthetic-B0"
    hh = "a" * 64
    job = {"trial_id": tid, "handoff_sha256": hh}
    jobs = [job]

    def add(ix, kind, status, blob):
        rid = f"20261123T00000000000{ix}Z-" + str(ix) * 16
        r = {
            "schema": "el-worker-return-1",
            "trial_id": tid,
            "handoff_sha256": hh,
            "run": rid,
            "manifest_sha256": str(ix) * 64,
            "package": rid + ".zip",
            "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "run_kind": kind,
            "status": status,
            "exit_code": 0 if status == "succeeded" else 1,
        }
        publish(tmp_path / (rid + ".receipt.json"), encoded(r))
        return rid, r

    older, _ = add(1, "generation", "failed", b"original")
    publish(tmp_path / (older + ".zip"), b"original")
    assert win.return_state(tmp_path, jobs)[tid]["status"] == "sealed_failure_review"
    new, _ = add(2, "generation", "succeeded", b"retry")
    publish(tmp_path / (new + ".zip"), b"retry")
    final, _ = add(3, "render", "succeeded", b"rendered")
    (tmp_path / (final + ".zip")).write_bytes(b"ren")
    assert win.return_state(tmp_path, jobs)[tid]["status"] == "in_transit"
    (tmp_path / (final + ".zip")).write_bytes(b"rendered")
    done = win.return_state(tmp_path, jobs)[tid]
    assert done["status"] == "returned" and [x["run_id"] for x in done["attempts"]] == [older, new, final]
    (tmp_path / (final + ".zip")).write_bytes(b"changed!")
    with pytest.raises(ValueError, match="integrity"):
        win.return_state(tmp_path, jobs)


def test_wrong_tickets_and_mismatched_stress_opening_rejected(tmp_path):
    kit = tmp_path / "kit"
    meta = mac.build_kit(ROOT, ROOT / "protocol/stage2-families-v1.json", mac.CATALOG, kit)
    plan = load(kit / "plan.json")
    r = plan["families"][0]
    b = r["batches"][1]
    auth = {
        "opening_sha256": "3" * 64,
        "method_sha256": "4" * 64,
        "freeze_manifest_sha256": "5" * 64,
        "freeze_run_id": "20261123T000000000000Z-" + "f" * 16,
    }
    jobs = [
        {"trial_id": t, "name": t + ".zip", "bytes": 1, "sha256": "a" * 64, "handoff_sha256": "b" * 64}
        for t in b["trials"]
    ]
    batch = {
        "schema": "el-windows-condition-batch-1",
        "batch_id": b["batch"],
        "runner_sha256": plan["windows_sources"][b["runner"]],
        "transport_sha256": plan["windows_sources"][win.RUNNERS["B0"]],
        "catalog_sha256": mac.CATALOG,
        "jobs": jobs,
        "opening": auth,
    }
    dest = tmp_path / "batch.json"
    dest.write_bytes(encoded(batch))
    assert win.check_batch(dest, sha(dest), plan, b, auth) == batch
    batch["opening"] = dict(auth, freeze_manifest_sha256="6" * 64)
    dest.write_bytes(encoded(batch))
    with pytest.raises(ValueError, match="Stress opening"):
        win.check_batch(dest, sha(dest), plan, b, auth)
    batch["jobs"] = list(reversed(jobs))
    dest.write_bytes(encoded(batch))
    with pytest.raises(ValueError, match="native batch"):
        win.check_batch(dest, sha(dest), plan, b, auth)
    with pytest.raises(ValueError, match="Wrong pinned"):
        win.check_plan(kit / "plan.json", "0" * 64, mac.windows_source(ROOT)[mac.WINS[0]])


def test_mac_queue_resumes_same_synthetic_trials_and_closes_all_conditions(tmp_path, monkeypatch):
    # All target families here are knobs in a scheduler fixture, with no scene or rendered data.
    store = tmp_path / "store"
    exchange = tmp_path / "exchange"
    store.mkdir()
    exchange.mkdir()
    plan_path = tmp_path / "metadata.json"
    publish(plan_path, encoded({"purpose": "synthetic_scheduler"}))
    ph = sha(plan_path)
    auth = {
        "schema": "fixture",
        "method_sha256": "1" * 64,
        "opening_sha256": "2" * 64,
        "freeze_manifest_sha256": "4" * 64,
        "freeze_run_id": "20261123T000000000000Z-" + "d" * 16,
        "plan_sha256": ph,
    }
    a = SimpleNamespace(
        project=ROOT,
        store=store,
        shelf=tmp_path / "cold",
        catalog=tmp_path / "none",
        catalog_sha256="7" * 64,
        exchange=exchange,
        python=Path(sys.executable),
        blender_python=Path(sys.executable),
        plan=plan_path,
        plan_sha256=ph,
    )
    sources = {n: sha(f) for n, f in mac.windows_source(ROOT).items()}
    plan = {"catalog_sha256": a.catalog_sha256, "windows_sources": sources, "families": []}
    for i in range(8):
        family = f"synthetic_queue_{i}"
        plan["families"].append(
            {
                "family": family,
                "batches": [
                    {
                        "kind": kind,
                        "runner": win.RUNNERS[kind],
                        "batch": f"synthetic-{i}-{kind}",
                        "trials": [family + "-" + c for c in cs],
                    }
                    for kind, cs in [("B0", ["B0"]), ("stress", ["R1", "Q1"])]
                ],
            }
        )
    by_id = {b["batch"]: b for row in plan["families"] for b in row["batches"]}
    calls = []

    def fake_invoke(_, script, flags):
        flags = list(map(str, flags))
        calls.append((script, tuple(flags)))
        if script == "stage2_reference.py":
            return
        if "--batch" in flags:
            label = flags[flags.index("--batch") + 1]
            block = by_id[label]
            folder = exchange / "Jobs" / label
            batch = {
                "schema": "el-windows-b0-batch-1"
                if block["kind"] == "B0"
                else "el-windows-condition-batch-1",
                "batch_id": label,
                "runner_sha256": sources[block["runner"]],
                "jobs": [],
            }
            if block["kind"] == "B0":
                batch["family_catalog_sha256"] = a.catalog_sha256
            else:
                batch.update(
                    catalog_sha256=a.catalog_sha256,
                    transport_sha256=sources[win.RUNNERS["B0"]],
                    opening=auth,
                )
            for t in block["trials"]:
                key = hashlib.sha256(t.encode()).hexdigest()
                batch["jobs"].append(
                    {"trial_id": t, "name": t + ".zip", "bytes": 1, "sha256": key, "handoff_sha256": key}
                )
            publish(folder / "batch.json", encoded(batch))
            return
        family = flags[flags.index("--family") + 1]
        c = flags[flags.index("--condition") + 1] if "--condition" in flags else "B0"
        t = family + "-" + c
        publish(
            store / "trials" / t / "trial_selection.json",
            encoded({"trial_id": t, "selection_status": "selected"}),
        )

    monkeypatch.setattr(mac, "invoke", fake_invoke)

    def accepted(_, reg, row, block, t):
        return {
            "trial_id": t,
            "condition_id": t.rsplit("-", 1)[1],
            "batch": block["batch"],
            "selection_sha256": sha(store / "trials" / t / "trial_selection.json"),
            "selection_status": "selected",
            "run_pins": {},
        }

    monkeypatch.setattr(mac, "accepted", accepted)

    def arrivals(b):
        folder = exchange / "Returns" / b["batch"]
        batch = load(exchange / "Jobs" / b["batch"] / "batch.json")
        for ix, j in enumerate(batch["jobs"]):
            t = j["trial_id"]
            c = t.rsplit("-", 1)[1]
            for n, kind in enumerate((win.ROLES[c], "render")):
                tok = hashlib.sha256((t + kind).encode()).hexdigest()
                rid = "20261123T000000000000Z-" + tok[:16]
                blob = tok.encode()
                record = {
                    "schema": "el-worker-return-1",
                    "trial_id": t,
                    "handoff_sha256": j["handoff_sha256"],
                    "run": rid,
                    "manifest_sha256": tok,
                    "package": rid + ".zip",
                    "bytes": len(blob),
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "run_kind": kind,
                    "status": "succeeded",
                    "exit_code": 0,
                }
                publish(folder / (rid + ".zip"), blob)
                publish(folder / (rid + ".receipt.json"), encoded(record))

    for row in plan["families"]:
        for block in row["batches"]:
            waiting = mac.step(a, plan, {}, auth)
            assert waiting["batch"] == block["batch"] and waiting["status"] == "waiting_for_return"
            assert mac.step(a, plan, {}, auth) == waiting
            arrivals(block)
        family = mac.step(a, plan, {}, auth)
        assert family["status"] == "family_complete" and family["family"] == row["family"]
    final = mac.step(a, plan, {}, auth)
    assert final == mac.step(a, plan, {}, auth) and final["conditions"] == 24
    assert len(load(Path(final["selection_registry"]))["trials"]) == 24
    assert len(calls) == 8 * (1 + 2 + 3)


def test_windows_queued_quota_waits_for_explicit_resume_and_ignores_duplicate_launch(tmp_path, monkeypatch):
    project = ROOT
    kit = tmp_path / "kit"
    meta = mac.build_kit(ROOT, ROOT / "protocol/stage2-families-v1.json", mac.CATALOG, kit)
    plan = load(kit / "plan.json")
    row = plan["families"][0]
    b = row["batches"][0]
    trial = b["trials"][0]
    # Native queue input contains only arbitrary synthetic transport bytes, never a held-out image, array or job handoff.
    exchange = tmp_path / "exchange"
    state = tmp_path / "state"
    state.mkdir()
    store = tmp_path / "store"
    store.mkdir()
    folder = exchange / "Jobs" / b["batch"]
    ret = exchange / "Returns" / b["batch"]
    blob = b"synthetic body, not a model input"
    key = hashlib.sha256(blob).hexdigest()
    path = folder / (trial + ".zip")
    publish(path, blob)
    publish(folder / b["runner"], (project / "scripts" / b["runner"]).read_bytes())
    job = {"trial_id": trial, "name": path.name, "bytes": len(blob), "sha256": key, "handoff_sha256": key}
    batchfile = folder / "batch.json"
    publish(
        batchfile,
        encoded(
            {
                "schema": "el-windows-b0-batch-1",
                "batch_id": b["batch"],
                "runner_sha256": plan["windows_sources"][b["runner"]],
                "family_catalog_sha256": mac.CATALOG,
                "jobs": [job],
            }
        ),
    )
    aa = "3" * 64
    ticket = exchange / "Jobs" / win.QUEUE / "ready" / (b["batch"] + ".json")
    publish(
        ticket,
        encoded(
            {
                "schema": "el-stage2-confirmation-batch-ticket-1",
                "family": row["family"],
                "batch": b["batch"],
                "trials": [trial],
                "authorization_sha256": aa,
                "plan_sha256": meta["plan_sha256"],
                "batch_sha256": sha(batchfile),
            }
        ),
    )
    a = SimpleNamespace(
        exchange=exchange,
        state=state,
        store=store,
        staged=tmp_path / "unread-staged",
        outgoing=tmp_path / "outgoing",
        weights=tmp_path / "unread-weights",
        plan_sha256=meta["plan_sha256"],
        authorization_sha256=aa,
        resume_batch=None,
    )
    calls = []

    def invoke(command, check):
        calls.append(command)
        if len(calls) == 1:
            publish(
                ret / "controller-000.json",
                encoded({"trial_id": trial, "handoff_sha256": key, "stderr_tail": win.QUOTA}),
            )
            return SimpleNamespace(returncode=1)
        for i, kind in enumerate(("generation", "render"), 1):
            rid = "20261123T000000000000Z-" + str(i) * 16
            data = b"synthetic record " + kind.encode()
            h = hashlib.sha256(data).hexdigest()
            publish(ret / (rid + ".zip"), data)
            publish(
                ret / (rid + ".receipt.json"),
                encoded(
                    {
                        "schema": "el-worker-return-1",
                        "trial_id": trial,
                        "handoff_sha256": key,
                        "run": rid,
                        "manifest_sha256": h,
                        "package": rid + ".zip",
                        "bytes": len(data),
                        "sha256": h,
                        "run_kind": kind,
                        "status": "succeeded",
                        "exit_code": 0,
                    }
                ),
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(win.subprocess, "run", invoke)
    assert win.once(a, plan, {})["status"] == "packaging_quota"
    assert win.once(a, plan, {})["status"] == "packaging_quota"
    assert len(calls) == 1
    a.resume_batch = b["batch"]
    assert win.once(a, plan, {})["status"] == "published"
    assert win.once(a, plan, {})["status"] == "awaiting_mac_ticket" and len(calls) == 2
    a.authorization_sha256 = "2" * 64
    with pytest.raises(ValueError, match="Unbound queue"):
        win.once(a, plan, {})
