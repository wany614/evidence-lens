"""Export the numerical source diagnostics and registered decisions for a completed recovery study."""

import argparse
from pathlib import Path

from evidence_lens._bundle import encoded, load, sha
from evidence_lens._verify import require
from recovery.contract import field_names
from dtu_record_gate import verified_member
from make_handoff import publish


DECISIONS = {"registration": "el-recovery-registration-1", "r1-decision": "el-recovery-r1-decision-1",
             "freeze": "el-recovery-final-freeze-1", "admission": "el-recovery-recipe-admission-1",
             "control-review": "el-recovery-control-review-1", "selection": "el-recovery-quality-selection-1",
             "resource-probe": "el-recovery-resource-probe-1"}
INIT_FIELDS = ("object_id", "condition", "actual_method", "N", "generator_source_ids", "frame", "scene_scale_m",
               "shared_endpoints", "matching", "eligibility", "fallback", "parameters", "versions",
               "initialization_protocol_sha256", "parameter_sha256", "init_npz_sha256", "source_tracks_sha256",
               "source_camera_sha256", "source_rgb_sha256")


def export(store, spec, out):
    records = []
    names = set()

    def add(name, pin, raw):
        require(Path(name).name == name and name.endswith(".json"), "nonportable public destination")
        require(name not in names, "duplicate public table")
        names.add(name)
        dest = out / name
        publish(dest, raw)
        records.append({"path": name, "source": pin, "bytes": dest.stat().st_size, "sha256": sha(dest)})

    for d in spec["decisions"]:
        role, pin = d["role"], d["source"]
        require(role in DECISIONS, "unknown decision role")
        p = verified_member(store, pin)
        v = load(p)
        require(v["schema"] == DECISIONS[role] and v["run_id"] == pin["run_id"], "unrelated decision record")
        suffix = {"admission": v.get("recipe_id"), "control-review": v.get("recipe_sha256", "")[:16],
                  "resource-probe": v.get("candidate_recipe_sha256", "")[:16]}.get(role)
        suffix = "-" + suffix if suffix else ""
        add(role + suffix + ".json", pin, p.read_bytes())
        if role == "admission" and "resource_basis" in v:
            proof = v["resource_basis"]["fresh_capacity"]
            raw = verified_member(store, proof)
            measured = load(raw)
            require(proof["member"] == "payload/forced-topology.json"
                    and measured["fixture"] == "synthetic-20k-controlled-branches"
                    and measured["passed"] is True and measured["nominal_resolution"] == [256, 256]
                    and len(measured["steps"]) == 5, "unknown recipe capacity measurement")
            original = store / "runs" / proof["run_id"]
            inputs = load(original / "inputs.json")["inputs"]
            applied = [r["role"] for r in inputs if r["role"] in {"recipe", "candidate"} and
                       len(r["files"]) == 1 and r["files"][0]["sha256"] == v["recipe_sha256"]]
            require(len(applied) == 1, "original capacity record lacks the admitted recipe")
            snapshot = load(original / "code_snapshot.json")["snapshot_sha256"]
            primary = load(verified_member(store, v["resource_basis"]["unchanged_renderer_and_resume"]))
            require(snapshot == primary["actual_worker_snapshots"]["topology"], "capacity worker differs")
            add("capacity-" + v["recipe_id"] + ".json", proof, raw.read_bytes())
            add("capacity-binding-" + v["recipe_id"] + ".json", pin, encoded({
                "schema": "el-recovery-public-capacity-binding-1", "native_observation": proof,
                "source_snapshot_sha256": snapshot, "recipe_input_role": applied[0],
                "original_inputs_manifest_sha256": sha(original / "inputs.json"),
                "recipe_sha256": v["recipe_sha256"], "admission": pin,
            }))
    for pin in spec.get("jobs", []):
        p = verified_member(store, pin)
        job = load(p)
        require(job["schema"] == "el-recovery-cell-1" and job["purpose"] == "formal"
                and job["experiment_id"] == "reconstruction-recovery-v1"
                and set(job["files"]) == field_names(job["condition"]), "unrelated reconstruction job")
        add("job-" + job["trial_id"] + ".json", pin, p.read_bytes())
    for pin in spec["initializers"]:
        v = load(verified_member(store, pin))
        require(v["schema"] == "el-recovery-init-1" and v["run_id"] == pin["run_id"], "unrelated initializer")
        obj, cond = v["object_id"], v["condition"]
        require(obj in {"set1", "set6", "scene_057_offset_bell_cage"} and cond in {"M2", "M4"},
                "not a registered recovery condition")
        table = {"schema": "el-recovery-public-initializer-1", "source": pin, **{k: v[k] for k in INIT_FIELDS}}
        add("initialization-" + obj + "-" + cond + ".json", pin, encoded(table))
    publish(out / "members.json", encoded({"schema": "el-recovery-public-authority-1", "files": records}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "pins", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    export(a.store.resolve(), load(a.pins), a.out)
