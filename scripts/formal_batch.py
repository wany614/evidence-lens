"""Prepare compact, pinned B0 learning extracts and compose one complete split dataset."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from evidence_lens._bundle import RUN_ID, encoded, files, load, save, sha
from evidence_lens._capacity import check as space_check, policy
from evidence_lens._index import writer_gate
from evidence_lens._verify import require, verify_bundle
from formal_lineage import EXPERIMENT, family_record
from mac_trial import select_files
from make_handoff import publish
from select_trial import KINDS, freeze as verify_selection

ROOT = Path(__file__).resolve().parents[1]
TARGETS = tuple(f"v{i:02}" for i in range(1, 7))


def family_pin(store, catalog, freeze_file, family_id):
    family = family_record(catalog, freeze_file, family_id)
    require(family["split"] in {"train", "development"}, "test families cannot enter the learning extract")
    selection = store / "trials" / family["trial_id"] / "trial_selection.json"
    selected = load(selection)
    require(selected.get("condition_id") == "B0" and selected.get("selection_status") == "selected" and
            all(selected[k] == family[k] for k in ("scene_id", "scene_family_id", "trial_id")), "selected B0 family differs from its catalog")
    row = next(r for r in load(catalog)["families"] if r["scene_family_id"] == family_id)
    if "reuse" in row:
        require(family["split"] == "train" and sha(selection) == row["reuse"]["selection_sha256"],
                "engineering training selection must match its pre-registered byte hash")
    else:
        require(selected["experiment_id"] == EXPERIMENT, "formal learning needs a formally selected B0 chain")
    ids = {role: selected["selected"][role]["run_id"] for role in KINDS}
    # Replays the sealed chain, parent inventory, transfer receipts, seven cameras,
    # numerical gates and independent readback once, including the shared checkpoint.
    verify_selection(store, family["trial_id"], ids)
    if "reuse" not in row:
        for role in ("reference", "readback", "generation", "render", "score", "evaluation", "recompute"):
            run = load(store / "runs" / ids[role] / "run.json")
            require(all(run.get(k) == family[k] for k in ("dataset_id", "dataset_hash", "split", "scene_id", "scene_family_id", "condition_id")),
                    "selected run is outside its recorded formal family/split")
    pin = {"schema": "el-learning-family-1", **{k: family[k] for k in ("scene_family_id", "scene_id", "trial_id", "condition_id", "cv_fold")},
           "split": "dev" if family["split"] == "development" else "train", "split_manifest_sha256": family["dataset_hash"],
           "selection_sha256": sha(selection)}
    for role in ("score", "evaluation"):
        record = selected["selected"][role]
        pin[role + "_run_id"] = ids[role]
        pin[role + "_manifest_sha256"] = record["manifest_sha256"]
        pin[role + "_receipt_sha256"] = selected["evidence"][role + "_metadata_sha256"]
    return family, selected, pin


def check_extract(payload, family, pin):
    data = load(payload / "family.json")
    require(data["schema"] == "el-learning-extract-1" and data["pin"] == pin and
            set(data["views"]) == set(TARGETS) and pin["scene_family_id"] == family["scene_family_id"], "extract pin differs")
    require(set(files(payload)) == {"family.json", *(v + ".npz" for v in TARGETS)}, "unexpected or missing compact extract files")
    for view, info in data["views"].items():
        path = payload / (view + ".npz")
        require(path.stat().st_size == info["bytes"] and sha(path) == info["sha256"], f"changed compact {view} file")
    return sha(payload / "family.json")


def prepare(store, catalog, freeze_file, family_id, retry=None):
    family, selected, pin = family_pin(store, catalog, freeze_file, family_id)
    directory = store / "formal" / family["dataset_hash"] / "extracts"
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / (family_id + ".json")
    with writer_gate(directory / ".locks" / family_id):
        if receipt.is_file():
            existing = load(receipt)
            require(existing["family_pin"] == pin, "stored extract used different source evidence")
            path = store / "runs" / existing["run_id"]
            require(sha(path / "manifest.json") == existing["manifest_sha256"] and
                    check_extract(path / "payload", family, pin) == existing["extract_manifest_sha256"], "stored compact extract differs")
            return existing
        # Recover a successful extract if the controller exited before publishing its pointer.
        attempts = []
        for path in sorted((store / "runs").glob("*/run.json")):
            run = load(path)
            if run["run_kind"] == "dataset_build" and run["trial_id"] == family["trial_id"] and run["experiment_id"] == "p4-learning-extract":
                require(load(path.parent / "config.resolved.yaml")["family_pin"] == pin, "conflicting existing extract preparation")
                attempts.append(run)
        successful = [r for r in attempts if r["execution_status"] == "succeeded"]
        require(len(successful) <= 1, "ambiguous successful extract preparations")
        if successful:
            rid = successful[0]["run_id"]
        else:
            if attempts and attempts[-1].get("worker_exit_code") is not None:
                require(retry == attempts[-1]["run_id"], "failed scientific preparation requires explicit --retry-of")
            with tempfile.TemporaryDirectory(dir=store, prefix="learning-family-") as tmp:
                temp = Path(tmp)
                family_file, config = temp / "family-pin.json", temp / "prepare.json"
                save(family_file, pin)
                save(config, {**family, "family_pin": pin, "config_version": "formal-learning-extract-1"})
                score_id, eval_id = pin["score_run_id"], pin["evaluation_run_id"]
                scores = select_files(store / "runs" / score_id / "payload", temp / "scores",
                                      ["score.json", "feature_schema.json", *(v + ".npz" for v in TARGETS)])
                labels = select_files(store / "runs" / eval_id / "payload", temp / "evaluation",
                                      ["evaluation.json", *("labels/" + v + ".npz" for v in TARGETS)])
                cmd = [sys.executable, "-m", "evidence_lens.recorder", "--root", str(store), "--source-root", str(ROOT),
                       "--config", str(config), "--kind", "dataset_build", "--trial", family["trial_id"], "--experiment", "p4-learning-extract",
                       "--input", f"family={family_file}", "--input", f"catalog={catalog}", "--input", f"scores={scores}",
                       "--input-parent", "scores=" + score_id, "--input", f"evaluation={labels}", "--input-parent", "evaluation=" + eval_id]
                for role in ("score", "evaluation", "recompute"):
                    rec = selected["selected"][role]
                    cmd.extend(["--parent", rec["run_id"] + "=" + rec["manifest_sha256"]])
                if retry:
                    cmd.extend(["--retry-of", retry])
                worker = ["{python}", "{source}/scripts/train_l1.py", "prepare", "--family", "{input:family}", "--catalog", "{input:catalog}",
                          "--scores", "{input:scores}", "--evaluation", "{input:evaluation}", "--output", "{out}"]
                done = subprocess.run([*cmd, "--", *worker], cwd=ROOT, text=True, stdout=subprocess.PIPE)
                require(done.returncode == 0, f"recorded learning extract failed: {done.stdout.strip()}")
                rid = json.loads(done.stdout)["run_id"]
        result = verify_bundle(store / "runs" / rid)
        require(result["run"]["execution_status"] == "succeeded" and result["run"]["run_kind"] == "dataset_build", "extract failed")
        extract_hash = check_extract(store / "runs" / rid / "payload", family, pin)
        evidence = {"schema": "el-formal-learning-receipt-1", "run_id": rid, "manifest_sha256": result["manifest_sha256"], "family_pin": pin,
                    "extract_manifest_sha256": extract_hash, "selected": {key: {k: rec[k] for k in ("run_id", "manifest_sha256")}
                                                                          for key, rec in selected["selected"].items()}}
        publish(receipt, encoded(evidence))
        return evidence


def dataset(store, catalog, freeze_file, split, out):
    catalog_hash = sha(catalog)
    require(load(freeze_file)["catalog_sha256"] == catalog_hash, "dataset catalog differs from the freeze")
    roster = [row for row in load(catalog)["families"] if row["split"] == {"train":"train", "dev":"development"}[split]]
    require(len(roster) == {"train":16, "dev":4}[split] and len({r["scene_family_id"] for r in roster}) == len(roster),
            "learning dataset needs its complete preassigned roster")
    index = {"schema":"el-learning-index-1", "split":split, "condition_id":"B0", "split_manifest_sha256":catalog_hash, "families":[]}
    content = {"catalog.json": (catalog, catalog_hash)}
    receipts = store / "formal" / catalog_hash / "extracts"
    for row in roster:
        family_id = row["scene_family_id"]
        family = family_record(catalog, freeze_file, family_id)
        rec = load(receipts / (family_id + ".json"))
        rid, pin = rec["run_id"], rec["family_pin"]
        require(RUN_ID.fullmatch(rid) and pin["split"] == split and pin["split_manifest_sha256"] == catalog_hash
                and all(pin[k] == family[k] for k in ("scene_family_id", "scene_id", "trial_id", "cv_fold")), "extract is not a member of this split")
        sealed = store / "runs" / rid
        require(sha(sealed / "manifest.json") == rec["manifest_sha256"] and load(sealed / "completion.json")["manifest_sha256"] == rec["manifest_sha256"],
                "extract producer no longer matches its pinned manifest")
        source = sealed / "payload"
        require(check_extract(source, family, pin) == rec["extract_manifest_sha256"], "frozen extract changed")
        data = load(source / "family.json")
        index["families"].append({"scene_family_id":family_id, "extract_manifest_sha256":rec["extract_manifest_sha256"]})
        content[family_id + "/family.json"] = (source / "family.json", rec["extract_manifest_sha256"])
        for v in TARGETS:
            content[family_id + "/" + v + ".npz"] = (source / (v + ".npz"), data["views"][v]["sha256"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with writer_gate(receipts / ".locks" / ("dataset-" + split)):
        if out.exists():
            require(not out.is_symlink() and set(files(out)) == {"dataset.json", *content} and (out / "dataset.json").read_bytes() == encoded(index)
                    and all(sha(out / name) == ref[1] for name, ref in content.items()), "conflicting immutable split dataset")
        else:
            # The bounded learning files are immutable and reused by same-volume hard link.
            # A portable copy fallback still needs to fit the existing storage policy.
            estimate = sum(p.stat().st_size for p, _ in content.values()) + len(encoded(index))
            space = space_check(store, estimate, policy(None, ROOT))
            require(not space["reason"], space["reason"] or "")
            with tempfile.TemporaryDirectory(dir=out.parent, prefix=".dataset-") as tmp:
                stage = Path(tmp) / "dataset"; stage.mkdir()
                for name, (src, expected) in content.items():
                    dst = stage / name; dst.parent.mkdir(parents=True, exist_ok=True)
                    if name == "catalog.json":
                        shutil.copyfile(src, dst)
                    else:
                        try:
                            os.link(src, dst)
                        except OSError:
                            shutil.copyfile(src, dst)
                    require(sha(dst) == expected, "staged learning dataset changed")
                    dst.chmod(dst.stat().st_mode & ~0o222)
                save(stage / "dataset.json", index)
                (stage / "dataset.json").chmod(0o444)
                stage.rename(out)
    return {"dataset":str(out), "dataset_sha256":sha(out / "dataset.json"), "split":split, "families":len(roster)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("prepare", "dataset"):
        cmd = sub.add_parser(stage)
        cmd.add_argument("--store", required=True, type=Path)
        cmd.add_argument("--families", type=Path, default=ROOT / "protocol/families-v1.json")
        cmd.add_argument("--freeze", type=Path, default=ROOT / "configs/formal-freeze-v1.json")
        if stage == "prepare":
            cmd.add_argument("--family", required=True)
            cmd.add_argument("--retry-of")
        else:
            cmd.add_argument("--split", required=True, choices=("train", "dev"))
            cmd.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    store, catalog, lock = args.store.resolve(), args.families.resolve(), args.freeze.resolve()
    result = (prepare(store, catalog, lock, args.family, args.retry_of) if args.stage == "prepare" else
              dataset(store, catalog, lock, args.split, args.out.absolute()))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
