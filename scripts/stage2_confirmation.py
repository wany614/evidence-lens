"""Plan and, only under explicit verified opening pins, operate the complete Mac B confirmation cohort."""

import argparse
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace
from pathlib import Path

from evidence_lens import _capacity
from evidence_lens._bundle import HASH, RUN_ID, encoded, load, sha
from evidence_lens._verify import require
from make_handoff import publish, sha_bytes, zip_bytes
import run_stage2_confirmation as win

CATALOG = "543adef7b3b3f0dbc11434738f5a7a6d4c6178622b63f452a5133eed6446dc79"
WINS = ("run_stage2_confirmation.py", "run_windows_batch.py", "run_conditions_batch.py")


def windows_source(project):
    return {n: (Path(__file__).with_name(n) if n == WINS[0] else project / "scripts" / n) for n in WINS}


def build_kit(project, catalog_path, digest, output):
    reg = win.signed(catalog_path, digest)
    require(
        digest == CATALOG and reg["schema"] == "el-vision-family-roster-1" and len(reg["families"]) == 24,
        "Wrong registered cohort",
    )
    rows = sorted((r for r in reg["families"] if r["split"] == "test"), key=lambda r: r["scene_family_id"])
    files = {n: p.read_bytes() for n, p in windows_source(project).items()}
    plan = {
        "schema": "el-stage2-confirmation-plan-1",
        "queue": win.QUEUE,
        "catalog_sha256": digest,
        "display_rule": "v01",
        "windows_sources": {n: sha_bytes(v) for n, v in files.items()},
        "families": [],
    }
    for i, r in enumerate(rows, 41):
        f = r["scene_family_id"]
        blocks = []
        for kind, conds in [("B0", ["B0"]), ("stress", ["R1", "Q1"])]:
            blocks.append(
                {
                    "kind": kind,
                    "batch": f"v11-b-confirm-{i:03}-{kind.lower()}",
                    "runner": win.RUNNERS[kind],
                    "trials": [r["trials"][c] for c in conds],
                }
            )
        plan["families"].append({"family": f, "test_group": r["test_group"], "batches": blocks})
    files["plan.json"] = encoded(plan)
    for n, v in files.items():
        publish(output / n, v)
    pd = sha_bytes(files["plan.json"])
    win.check_plan(output / "plan.json", pd, output / WINS[0])
    bundle = zip_bytes(files)
    publish(output / "windows-plan.zip", bundle)
    result = {
        "schema": "el-stage2-confirmation-kit-1",
        "purpose": "metadata_and_source_only",
        "families": 8,
        "conditions": 24,
        "batches": 16,
        "catalog_sha256": digest,
        "plan_sha256": pd,
        "zip_sha256": sha_bytes(bundle),
        "zip_bytes": len(bundle),
        "windows_sources": plan["windows_sources"],
    }
    publish(output / "kit.json", encoded(result))
    return result


def authority(a):
    from stage2_opening import verify_opening

    reg = win.signed(a.catalog, a.catalog_sha256)
    require(
        a.catalog_sha256 == CATALOG
        and all(
            isinstance(v, str) and HASH.fullmatch(v)
            for v in (a.method_sha256, a.opening_sha256, a.freeze_manifest, a.plan_sha256)
        )
        and RUN_ID.fullmatch(a.freeze_run),
        "Exact owner opening, method, recorder and plan pins are required",
    )
    plan = win.check_plan(a.plan, a.plan_sha256, windows_source(a.project)[WINS[0]])
    require(
        plan["catalog_sha256"] == a.catalog_sha256
        and all(sha(p) == plan["windows_sources"][n] for n, p in windows_source(a.project).items()),
        "Queue differs from native Windows source or catalog",
    )
    path = a.store / "protocol/stage2" / a.catalog_sha256 / "test-opening.json"
    require(path.is_file() and sha(path) == a.opening_sha256, "Opening absent or differs from owner pin")
    opening = verify_opening(a.store, reg, a.catalog_sha256)
    require(
        (opening["freeze_run_id"], opening["freeze_manifest_sha256"], opening["method_sha256"])
        == (a.freeze_run, a.freeze_manifest, a.method_sha256)
        and opening["test_family_ids"] == [r["family"] for r in plan["families"]],
        "Owner pins differ from the recorder opening",
    )
    record = {
        "schema": "el-stage2-confirmation-authorization-1",
        "plan_sha256": a.plan_sha256,
        "queue": win.QUEUE,
        "opening_sha256": a.opening_sha256,
        **{
            k: opening[k]
            for k in (
                "catalog_sha256",
                "test_family_ids",
                "freeze_run_id",
                "freeze_manifest_sha256",
                "method_sha256",
            )
        },
    }
    return plan, reg, record


def invoke(a, script, flags):
    p = a.project / "scripts" / script
    env = dict(os.environ, PYTHONPATH=str(a.project / "src"), PYTHONUTF8="1")
    cmd = [str(a.python), str(p), *map(str, flags)]
    result = subprocess.run(
        cmd,
        env=env,
        cwd=a.project,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    if result.returncode:
        event = {
            "schema": "el-stage2-confirmation-controller-event-1",
            "script": script,
            "exit_code": result.returncode,
            "arguments": list(map(str, flags)),
            "output": result.stdout,
        }
        data = encoded(event)
        h = sha_bytes(data)
        publish(a.store / "stage2" / a.catalog_sha256 / "confirmation" / "events" / (h + ".json"), data)
        known = (
            "projected free disk below configured floor",
            "projected active batch exceeds configured quota",
            "deferred by configured disk limits",
            "would exceed disk limits",
            "would breach the free disk floor",
        )
        if any(m in result.stdout for m in known):
            raise CapacityHold(h, script)
        raise RuntimeError(
            script
            + " exited "
            + str(result.returncode)
            + "; diagnostic "
            + h
            + "\n"
            + result.stdout[-7000:]
        )


class CapacityHold(RuntimeError):
    def __init__(self, digest, script):
        super().__init__(script + " held by storage limits")
        self.digest = digest
        self.script = script


def optional_archive(a, row, dest, authorization_digest):
    from stage2_confirmation_archive import snapshot

    folder = getattr(a, "private_archive", None)
    if folder is None:
        return None
    f = row["family"]
    marker = dest / "archives" / (f + ".local.json")
    family_hash = sha(dest / "families" / (f + ".json"))
    if marker.is_file():
        record = load(marker)
        require(
            record["family_sha256"] == family_hash
            and record["authorization_sha256"] == authorization_digest
            and (folder / (f + ".zip")).is_file(),
            "Existing family archive or authority differs",
        )
        return record
    args = SimpleNamespace(
        project=a.project,
        catalog=a.catalog,
        store=a.store,
        shelf=a.shelf,
        catalog_sha256=a.catalog_sha256,
        family=f,
        family_sha256=family_hash,
        authorization_sha256=authorization_digest,
        output=folder / (f + ".zip"),
    )
    proof = snapshot(args)
    record = {
        "schema": "el-stage2-family-local-archive-1",
        "family_sha256": family_hash,
        "authorization_sha256": authorization_digest,
        "provider_status": "awaiting_independent_download",
        **{"proof": proof},
    }
    publish(marker, encoded(record))
    return record


def save_notes(folder, jobs, dest):
    rows = win.notes(folder, jobs)
    for r in rows:
        publish(dest / "controllers" / r["file"], (folder / r["file"]).read_bytes())
    return rows


def accepted(a, reg, r, b, trial, expected_sha=None):
    from vision_sources import pinned_cell
    from evidence_lens.formal_inputs import compact_pin

    c = trial.rsplit("-", 1)[1]
    path = a.store / "trials" / trial / "trial_selection.json"
    h = sha(path)
    require(expected_sha is None or h == expected_sha, "Previously accepted selection changed")
    row = next(x for x in reg["families"] if x["scene_family_id"] == r["family"])
    cell = pinned_cell([a.store, a.shelf], path, h, row, c)
    q = load(path)
    pins = q.get("selected") or {k: q[k] for k in ("reference", "failure")}
    if c in ("R1", "Q1") and q["selection_status"] == "selected":
        base = a.store / "trials" / row["trials"]["B0"] / "trial_selection.json"
        bo = load(base)
        require(
            q["b0_selection_sha256"] == sha(base)
            and compact_pin(q["selected"]["reference"]) == compact_pin(bo["selected"]["reference"]),
            "Broken paired B0 selection",
        )
        if c == "Q1":
            require(
                compact_pin(q["selected"]["generation"]) == compact_pin(bo["selected"]["generation"])
                and compact_pin(q["selected"]["producer"]) == compact_pin(q["selected"]["preparation"]),
                "Wrong Q1 generation",
            )
    return {
        "trial_id": trial,
        "condition_id": c,
        "selection_sha256": h,
        "selection_status": q["selection_status"],
        "run_pins": {k: {n: v[n] for n in ("run_id", "manifest_sha256")} for k, v in pins.items()},
        "batch": b["batch"],
    }


def step(a, plan, reg, auth):
    dest = a.store / "stage2" / a.catalog_sha256 / "confirmation"
    queue = a.exchange / "Jobs" / win.QUEUE
    auth_b = encoded(auth)
    ad = sha_bytes(auth_b)
    for path in (dest / "authorization.json", queue / "authorization.json"):
        publish(path, auth_b)
    publish(queue / "plan.json", a.plan.read_bytes())
    for n, p in windows_source(a.project).items():
        publish(queue / n, p.read_bytes())
    common = ["--store", a.store, "--project", a.project]
    scientific = common + ["--catalog", a.catalog, "--catalog-sha256", a.catalog_sha256]

    def waiting(kind, **rest):
        return dict(status=kind, authorization_sha256=ad, **rest)

    def reservation(bytes_needed):
        return _capacity.check(a.store, bytes_needed, load(a.project / "configs/storage.json"))

    families = {}
    for row in plan["families"]:
        f = row["family"]
        selections = {}
        completed = dest / "families" / (f + ".json")
        if completed.is_file():
            prior = load(completed)
            require(
                prior["authorization_sha256"] == ad and prior["family"] == f,
                "Family marker belongs to other opening",
            )
            for t, entry in prior["trials"].items():
                require(
                    sha(a.store / "trials" / t / "trial_selection.json") == entry["selection_sha256"],
                    "Selection changed after family closure",
                )
            require(
                set(prior["trials"]) == {t for b in row["batches"] for t in b["trials"]},
                "Incomplete family closure",
            )
            optional_archive(a, row, dest, ad)
            families[f] = prior
            continue
        for b in row["batches"]:
            label = b["batch"]
            ticket_path = queue / "ready" / (label + ".json")
            local = dest / "tickets" / (label + ".json")
            jobdir = a.exchange / "Jobs" / label
            if not local.is_file():
                space = reservation(150_000_000 if b["kind"] == "B0" else 50_000_000)
                if space["reason"]:
                    return waiting("mac_capacity_hold", family=f, batch=label, capacity=space)
                if b["kind"] == "B0":
                    invoke(
                        a,
                        "stage2_reference.py",
                        common + ["--family", f, "--blender-python", a.blender_python],
                    )
                    invoke(
                        a,
                        "stage2_jobs.py",
                        scientific + ["--families", f, "--batch", label, "--exchange", a.exchange],
                    )
                else:
                    if selections[row["batches"][0]["trials"][0]]["selection_status"] != "selected":
                        return waiting(
                            "paired_conditions_require_baseline_disposition", family=f, batch=label
                        )
                    invoke(
                        a,
                        "stage2_conditions.py",
                        [
                            "dispatch",
                            *scientific,
                            "--families",
                            f,
                            "--conditions",
                            "R1",
                            "Q1",
                            "--batch",
                            label,
                            "--exchange",
                            a.exchange,
                        ],
                    )
                bp = jobdir / "batch.json"
                bd = sha(bp)
                win.check_batch(bp, bd, plan, b, auth)
                pub = {
                    "schema": "el-stage2-confirmation-batch-ticket-1",
                    "plan_sha256": a.plan_sha256,
                    "authorization_sha256": ad,
                    "batch": label,
                    "batch_sha256": bd,
                    "family": f,
                    "trials": b["trials"],
                }
                publish(local, encoded(pub))
            ticket = load(local)
            require(ticket["authorization_sha256"] == ad and ticket["batch"] == label, "Wrong local ticket")
            publish(ticket_path, local.read_bytes())
            batch = win.check_batch(jobdir / "batch.json", ticket["batch_sha256"], plan, b, auth)
            folder = a.exchange / "Returns" / label
            log = dest / "returns" / label
            diag = save_notes(folder, batch["jobs"], log)
            for t in b["trials"]:
                marker = dest / "accepted" / (t + ".json")
                selected = a.store / "trials" / t / "trial_selection.json"
                if marker.is_file():
                    entry = load(marker)
                    require(
                        entry["authorization_sha256"] == ad
                        and entry["trial_id"] == t
                        and sha(selected) == entry["selection_sha256"],
                        "Accepted trial was replaced",
                    )
                    selections[t] = entry
                    continue
                if not selected.is_file():
                    states = win.return_state(folder, batch["jobs"])
                    state = states[t]
                    if state["status"] != "returned":
                        return waiting(
                            state["status"],
                            family=f,
                            trial=t,
                            batch=label,
                            controller_notes=diag,
                            attempts=state["attempts"],
                        )
                    space = reservation(180_000_000)
                    if space["reason"]:
                        return waiting("mac_capacity_hold", family=f, batch=label, trial=t, capacity=space)
                    if b["kind"] == "B0":
                        invoke(
                            a, "stage2_process.py", scientific + ["--family", f, "--returns-dir", folder]
                        )
                    else:
                        invoke(
                            a,
                            "stage2_conditions.py",
                            [
                                "process",
                                *scientific,
                                "--family",
                                f,
                                "--condition",
                                t.rsplit("-", 1)[1],
                                "--returns-dir",
                                folder,
                            ],
                        )
                entry = accepted(a, reg, row, b, t)
                entries = win.return_state(folder, batch["jobs"])[t]
                require(
                    entries["status"] in ("returned", "sealed_failure_review"),
                    "Canonical selection has no complete original Windows return",
                )
                for original in entries["attempts"]:
                    name = original["run_id"] + ".receipt.json"
                    raw = (folder / name).read_bytes()
                    require(
                        sha_bytes(raw) == original["receipt_sha256"],
                        "Return envelope changed after validation",
                    )
                    publish(log / "envelopes" / name, raw)
                entry.update(
                    authorization_sha256=ad,
                    returns=entries["attempts"],
                    ticket_sha256=sha(local),
                    controller_notes=diag,
                )
                publish(marker, encoded(entry))
                selections[t] = entry
            publish(
                queue / "acknowledged" / (label + ".json"),
                encoded(
                    {
                        "schema": "el-stage2-confirmation-mac-ack-1",
                        "authorization_sha256": ad,
                        "batch": label,
                        "trials": {t: selections[t] for t in b["trials"]},
                    }
                ),
            )
        group = {
            "schema": "el-stage2-confirmation-family-1",
            "family": f,
            "authorization_sha256": ad,
            "trials": selections,
        }
        publish(completed, encoded(group))
        families[f] = group
        optional_archive(a, row, dest, ad)
        return waiting("family_complete", family=f, completed=len(families), family_sha256=sha(completed))
    trials = {
        t: {"selection_sha256": v["selection_sha256"]}
        for item in families.values()
        for t, v in item["trials"].items()
    }
    require(len(families) == 8 and len(trials) == 24, "Incomplete final cohort")
    selection = {
        "schema": "el-vision-selected-trials-1",
        "catalog_sha256": a.catalog_sha256,
        "split": "test",
        "trials": trials,
    }
    target = dest / "selection-registry.json"
    publish(target, encoded(selection))
    return waiting(
        "complete",
        families=8,
        conditions=24,
        selection_registry=str(target),
        selection_registry_sha256=sha(target),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_subparsers(dest="action", required=True)

    def catalog_flags(cmd):
        for n in ("project", "catalog"):
            cmd.add_argument("--" + n, required=True, type=Path)
        cmd.add_argument("--catalog-sha256", required=True)

    kit = actions.add_parser("kit")
    catalog_flags(kit)
    kit.add_argument("--output", required=True, type=Path)
    live = actions.add_parser("run")
    catalog_flags(live)
    for k in ("store", "shelf", "exchange", "plan", "python", "blender-python"):
        live.add_argument("--" + k, required=True, type=Path)
    for k in ("plan-sha256", "opening-sha256", "method-sha256", "freeze-run", "freeze-manifest"):
        live.add_argument("--" + k, required=True)
    live.add_argument(
        "--private-archive",
        type=Path,
        help="Optional original family archive directory outside the active store",
    )
    live.add_argument("--once", action="store_true")
    live.add_argument("--poll", type=int, default=15)
    a = p.parse_args(argv)
    a.project = a.project.resolve()
    a.catalog = a.catalog.resolve()
    if a.action == "kit":
        print(json.dumps(build_kit(a.project, a.catalog, a.catalog_sha256, a.output)))
        return
    for k in ("store", "shelf", "exchange", "python", "blender_python", "plan"):
        setattr(
            a, k, getattr(a, k).absolute() if k in ("python", "blender_python") else getattr(a, k).resolve()
        )
    require(1 <= a.poll <= 60, "Polling interval outside 1..60 seconds")
    plan, reg, op = authority(a)
    if a.private_archive:
        a.private_archive = a.private_archive.resolve()
        require(
            not a.private_archive.is_relative_to(a.store), "Place private archives outside the active store"
        )
    base = a.store / "stage2" / a.catalog_sha256 / "confirmation"
    with win.lock(base / "operator.sqlite3"):
        previous = None
        while True:
            try:
                result = step(a, plan, reg, op)
            except CapacityHold as ex:
                result = {"status": "mac_capacity_hold", "script": ex.script, "event_sha256": ex.digest}
            if result != previous:
                print(json.dumps(result), flush=True)
                previous = result
            if a.once or result["status"] in (
                "complete",
                "mac_capacity_hold",
                "sealed_failure_review",
                "paired_conditions_require_baseline_disposition",
            ):
                return
            if result["status"] != "family_complete":
                time.sleep(a.poll)


if __name__ == "__main__":
    main()
