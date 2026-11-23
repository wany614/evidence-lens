"""Reconstruct the original Windows worker or register a separately named research replication."""

import argparse
import io
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

from evidence_lens._bundle import encoded, load, local_path, sha, utc
from evidence_lens._index import writer_gate
from evidence_lens._verify import require
from evidence_lens.blender_reference.formal_catalog import CATALOG, LOCK, STUDY, validate, value_hash
from make_handoff import publish, sha_bytes, zip_bytes
from select_trial import KINDS, freeze
from source_candidate import audit, git

ROOT = Path(__file__).resolve().parents[1]
WORKER_COMMIT = "ab18e7d4ae627c550f11eb08095739830055ea91"
WORKER_SHA = "3ef8f32ee5deb62065a56e9e64221a84c2e2e8fd907037f1724f857125967143"
ORIGINAL_CATALOG = "fb7fabef06d006c992bc38cb03c283a41d90238e0b174664a1357f274bb36baa"
ADDENDUM = "protocol/analysis-addendum-v1.json"
POINTS = "protocol/operating-points-v1.json"
WORKER_DESCRIPTOR = "protocol/windows-worker-source-v1.json"


def worker_from_source(project):
    descriptor = load(ROOT / WORKER_DESCRIPTOR)
    require(descriptor["schema"] == "el-historical-worker-source-1" and descriptor["source_commit"] == WORKER_COMMIT
            and descriptor["zip_sha256"] == WORKER_SHA and len(descriptor["files"]) == 19, "invalid historical source descriptor")
    members = {}
    for name, entry in descriptor["files"].items():
        path = project / local_path(name)
        require(path.is_file() and not path.is_symlink(), f"missing worker source {name}; provide the historical Git checkout or --worker-zip")
        data = entry["historical_text"].encode("utf-8") if "historical_text" in entry else path.read_bytes()
        if sha_bytes(data) != entry["sha256"]:
            checkout_sha = sha_bytes(data)
            edits = (entry.get("edits") if checkout_sha == entry.get("checkout_sha256") else
                     entry.get("checkout_variants", {}).get(checkout_sha))
            require(edits is not None, f"{name} differs from both recorded versions; supply the pinned checkout or --worker-zip")
            lines = data.decode("utf-8").splitlines(keepends=True)
            for edit in reversed(edits):
                start = edit["line"] - 1
                end = start + len(edit["remove"])
                require(lines[start:end] == edit["remove"], f"worker reconstruction line mismatch in {name}")
                lines[start:end] = edit["insert"]
            data = "".join(lines).encode("utf-8")
        require(sha_bytes(data) == entry["sha256"], f"historical blob mismatch for {name}")
        members[name] = data
    return zip_bytes(members)


def worker(store, catalog, git_root, supplied_zip=None):
    """Install only a byte-identical original package before the first formal dispatch."""
    if supplied_zip is None:
        try:
            found = subprocess.run(["git", "-C", str(git_root), "cat-file", "-e", WORKER_COMMIT + "^{commit}"], capture_output=True).returncode == 0
        except FileNotFoundError:
            found = False
        if found:
            def blob(name):
                return git(git_root, "show", WORKER_COMMIT + ":" + local_path(name))
            names = json.loads(blob("configs/windows-worker-files.json"))["files"]
            data = zip_bytes({n: blob(n) for n in names if n != "README.md"})
        else:
            data = worker_from_source(git_root)
    else:
        data = supplied_zip.read_bytes()
    require(sha_bytes(data) == WORKER_SHA, "worker bytes differ from the original formal B0 package")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        members = {n: sha_bytes(z.read(local_path(n))) for n in z.namelist()}
    require(len(members) == 19 and "README.md" not in members, "unexpected historical worker inventory")
    require(load(catalog)["schema"] == "el-formal-families-1", "invalid family catalog")
    directory = store / "formal" / sha(catalog)
    receipt = {"schema": "el-formal-worker-source-1", "sha256": WORKER_SHA, "files": members}
    with writer_gate(store):
        # Check both destinations before publishing either member of the pair.
        for name, content in (("worker-source.zip", data), ("worker-source.json", encoded(receipt))):
            path = directory / name
            if path.exists() or path.is_symlink():
                require(not path.is_symlink() and path.is_file() and sha(path) == sha_bytes(content),
                        "a different worker is already frozen for this catalog")
        publish(directory / "worker-source.zip", data)
        publish(directory / "worker-source.json", encoded(receipt))
    return {"worker_source_sha256": WORKER_SHA, "members": len(members), "path": str(directory / "worker-source.zip")}


def pilot_evidence(store, project, row, lock):
    trial = row["trial_id"]
    selected_file = store / "trials" / trial / "trial_selection.json"
    selected = load(selected_file)
    require(selected["scene_id"] == row["scene_id"] and selected["scene_family_id"] == row["scene_family_id"]
            and selected["trial_id"] == trial and selected["experiment_id"] == "p2-single-image"
            and selected["condition_id"] == "B0", "replication requires its own selected tabletop B0 pilot")
    ids = {role: selected["selected"][role]["run_id"] for role in KINDS}
    old = row["reuse"]
    require(sha(selected_file) != old["selection_sha256"] and all(ids[r] != old[r + "_run_id"] for r in ("reference", "readback")),
            "the historical private tabletop cannot stand in for a new replication")
    freeze(store, trial, ids)  # Reopen the complete sealed chain and scientific gates.
    windows, cpu = (load(project / name) for name in ("configs/windows-b0.json", "configs/p2-cpu.json"))
    for role in ("generation", "render", "score", "evaluation", "recompute"):
        cfg = load(store / "runs" / ids[role] / "config.resolved.yaml")
        template = windows if role in ("generation", "render") else cpu
        require(all(key in cfg and cfg[key] == value for key, value in template.items()),
                f"pilot {role} settings differ from the frozen B0 protocol")
    generated = load(store / "runs" / ids["generation"] / "payload/generation.json")
    require(generated["checkpoint_sha256"] == windows["generator"]["checkpoint_sha256"], "pilot used a different checkpoint")
    reuse = {"selection_sha256": sha(selected_file)}
    recipe = load(project / "configs/reference.json")["reference"]
    for role in ("reference", "readback"):
        bundle = store / "runs" / ids[role]
        cfg = load(bundle / "config.resolved.yaml")
        require(value_hash(cfg["reference"]) == row["recipe_sha256"] and cfg["reference"] == recipe,
                "pilot used a different authored tabletop recipe")
        for rel, expected in lock["frozen_files"].items():
            if rel.startswith("src/evidence_lens/blender_reference/") and rel.endswith(("/produce.py", "/scene.py", "/passes.py", "/validation.py", "/verify_reference.py")):
                require(sha(bundle / "source" / rel) == expected, "pilot reference implementation differs from the registered source")
        reuse[role + "_run_id"] = ids[role]
        reuse[role + "_manifest_sha256"] = selected["selected"][role]["manifest_sha256"]
        reuse[role + "_config_sha256"] = sha(bundle / "config.resolved.yaml")
    ref = store / "runs" / ids["reference"] / "payload"
    meta = load(ref / "reference.json")
    require(meta["camera_sha256"] == row["camera_sha256"] == sha(ref / "cameras.json"), "pilot camera differs from authored tabletop")
    scene_file = ref / "scene" / (row["scene_family_id"] + "_v001.blend")
    require(meta["input_sha256"] == sha(ref / "input.png") and meta["blend_sha256"] == sha(scene_file),
            "pilot scene or source bytes differ from the sealed manifest")
    reuse.update(input_sha256=meta["input_sha256"], blend_sha256=meta["blend_sha256"])
    return reuse


def registration(blobs, name, reuse, stamp, commit):
    """Rebind lineage and dates. The scientific recipe and original validation code are copied verbatim."""
    origin = {path: sha_bytes(blobs[path]) for path in (STUDY, CATALOG, LOCK, ADDENDUM, POINTS)}
    require(origin[CATALOG] == ORIGINAL_CATALOG, "register from the original source checkout")
    identity = {"name": name, "source_commit": commit, "original_files_sha256": origin}
    study, catalog, lock, addendum, points = (json.loads(blobs[path]) for path in (STUDY, CATALOG, LOCK, ADDENDUM, POINTS))
    study.update(registered_at_utc=stamp, replication=identity)
    study["known_at_registration"] = {"families_seen": ["scene_001_tabletop"], "engineering_selection_sha256": reuse["selection_sha256"]}
    study_sha = sha_bytes(encoded(study))
    catalog.update(registered_at_utc=stamp, replication=identity, preregistration_sha256=study_sha)
    catalog["families"][0]["reuse"] = reuse
    catalog_sha = sha_bytes(encoded(catalog))
    lock.update(registered_at_utc=stamp, replication=identity, catalog_sha256=catalog_sha, preregistration_sha256=study_sha)
    lock["frozen_files"].update({STUDY: study_sha, CATALOG: catalog_sha})
    addendum.update(time_utc=stamp, study_sha256=study_sha, known=["Only the separately recorded tabletop pilot in this replication store."],
                    replication=identity)
    points.update(parent_study_sha256=study_sha, parent_addendum_sha256=sha_bytes(encoded(addendum)), replication=identity)
    return {p: encoded(v) for p, v in ((STUDY, study), (CATALOG, catalog), (LOCK, lock), (ADDENDUM, addendum), (POINTS, points))}


def committed_source(project):
    commit = git(project, "rev-parse", "--verify", "HEAD").decode().strip()
    entries = {}
    for item in git(project, "ls-tree", "-rz", commit).split(b"\0"):
        if not item:
            continue
        header, name = item.split(b"\t", 1)
        mode, kind, oid = header.split()
        require(kind == b"blob" and mode in (b"100644", b"100755"), "source registration requires regular committed files")
        entries[local_path(name.decode())] = (int(mode, 8), oid.decode())
    _, blobs = audit(project, entries)  # Only registered source, never untracked research data, local settings or Git history.
    return commit, blobs, entries


def register(store, project, out, name):
    require(not out.is_symlink(), "replication output cannot be a symbolic link")
    project, store, out = project.resolve(), store.resolve(), out.resolve()
    require(re.fullmatch(r"[a-z][a-z0-9-]{1,62}", name) and name != "evidence-lens-families-v1",
            "give the independent replication a distinct lowercase name")
    require(not out.exists() and not any(out.is_relative_to(p) or p.is_relative_to(out) for p in (project, store)),
            "choose a new project directory outside the source checkout and research store")
    catalog, _, _, lock = validate(project)
    require(lock["catalog_sha256"] == ORIGINAL_CATALOG, "historical source template required")
    for path in (store / "runs").glob("*/run.json"):
        recorded = load(path)
        require(recorded.get("scene_family_id") in (None, "scene_001_tabletop") and recorded.get("dataset_hash") is None,
                "register the replication in a pilot-only store before recording other families")
    pilot = pilot_evidence(store, project, catalog["families"][0], lock)
    commit, blobs, entries = committed_source(project)
    require(all(sha_bytes(blobs[p]) == expected for p, expected in lock["frozen_files"].items()) and sha_bytes(blobs[LOCK]) == sha(project / LOCK),
            "working registration and committed source differ")
    out.parent.mkdir(parents=True, exist_ok=True)
    with writer_gate(store / "replication-registrations"):
        intent = store / "protocol/replication-registration.json"
        identity = {"schema": "el-replication-registration-1", "name": name, "source_commit": commit,
                    "pilot_selection_sha256": pilot["selection_sha256"]}
        previous = load(intent) if intent.is_file() else {}
        stamp = previous.get("registered_at_utc", utc())
        replacements = registration(blobs, name, pilot, stamp, commit)
        record = dict(identity, registered_at_utc=stamp, catalog_sha256=sha_bytes(replacements[CATALOG]))
        require(not previous or previous == record, "this research store already belongs to a different registration")
        with tempfile.TemporaryDirectory(dir=out.parent, prefix=".replication-") as temp:
            stage = Path(temp) / name
            stage.mkdir()
            for rel, blob in blobs.items():
                path = stage / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(replacements.get(rel, blob))
                path.chmod(entries[rel][0] & 0o777)
            validate(stage)
            require(not out.exists() and not out.is_symlink(), "replication directory already exists")
            publish(intent, encoded(record))  # An interrupted registration can recreate its exact catalog in a new path.
            stage.rename(out)
    return {"replication": name, "project": str(out), "catalog_sha256": sha(out / CATALOG), "tabletop_selection_sha256": pilot["selection_sha256"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="action", required=True)
    w = subs.add_parser("worker", help="seed the historical 19-file worker before the first formal handoff")
    w.add_argument("--store", required=True, type=Path)
    w.add_argument("--families", type=Path, default=ROOT / CATALOG)
    w.add_argument("--git-root", type=Path, default=ROOT, help="full Git checkout or extracted published source tree")
    w.add_argument("--worker-zip", type=Path, help="alternative exact original ZIP for a source-only checkout")
    r = subs.add_parser("register", help="build a new source project with a separately named 24-family registration")
    r.add_argument("--store", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--name", required=True)
    args = p.parse_args()
    try:
        result = worker(args.store.resolve(), args.families, args.git_root, args.worker_zip) if args.action == "worker" else register(args.store, ROOT, args.out, args.name)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        p.exit(1, f"{error}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
