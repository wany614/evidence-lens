"""Verify disclosed recorder and local engineering incidents separately from scientific family outcomes."""

from pathlib import Path

from evidence_lens._bundle import load
from evidence_lens._verify import verify_bundle
from evidence_lens.formal_inputs import compact_pin, regular, require
from evidence_lens.scoring_io import file_hash
from vision_sources import find_run, member, signed

SCOPES = {
    "terminal_readback_setup": (
        "stage3-terminal-optimization-readback",
        "The original terminal-checkpoint readback named an undeclared input parent. The corrected archive verification read the completed Windows optimizer products.",
    ),
    "tensor_audit_alias": (
        "stage3-neural-tensor-comparison",
        "The independent validator expected a particular JSON key order for the equivalent M2 score-two and score-all entries. The corrected validator checked the same original tensors and score arrays.",
    ),
    "public_case_export": (
        "stage3-public-case",
        "The first public case export named a reference control member outside the parent's payload inventory. The corrected export bound the original control-file hashes to the full signed manifest.",
    ),
}


def verified_failure(roots, pin):
    for base in roots:
        for name in ("runs", "quarantine"):
            root = base / name / pin["run_id"]
            if not root.is_dir():
                continue
            result = verify_bundle(root)
            require(
                result["manifest_sha256"] == pin["manifest_sha256"]
                and result["run"]["run_id"] == pin["run_id"]
                and result["run"]["execution_status"] == "failed",
                "engineering failure seal differs",
            )
            return root, result["run"]
    raise ValueError("missing explicitly recorded engineering incident " + pin["run_id"])


def observations(roots, request):
    rows = []
    require(set(request["recorded"]) == set(SCOPES), "incomplete independent engineering incident roster")
    for key, entry in request["recorded"].items():
        experiment, note = SCOPES[key]
        orig = compact_pin(entry["failure"])
        good = compact_pin(entry["recovery"])
        f, failed = verified_failure(roots, orig)
        success = find_run(roots, good)
        rec = entry["recovery"]
        p, spec = member(success, rec["member"], rec["sha256"])
        require(
            failed["experiment_id"] == experiment
            and success.run["execution_status"] == "succeeded"
            and success.run["experiment_id"] == experiment,
            "engineering recovery attaches to a different task",
        )
        rows.append(
            {
                "scope": key,
                "status": "failed",
                "source": orig,
                "original_record_sha256": file_hash(regular(f, "run.json")),
                "original_worker_exit_code": failed["worker_exit_code"],
                "original_recorder_exit_code": failed["exit_code"],
                "original_wall_seconds": failed["elapsed_seconds_monotonic"],
                "original_peak_tree_rss_bytes": failed["sampled_peak_tree_rss_bytes"],
                "recovered_by": {
                    **good,
                    "member": p.relative_to(success.root).as_posix(),
                    "sha256": spec["sha256"],
                },
                "recovery_wall_seconds": success.run["elapsed_seconds_monotonic"],
                "reason": note,
            }
        )
    native = request["local_cache_case"]
    require(
        native["member"] == "work/v11-c-src/public-replay-import-cache-incident/receipt.json",
        "unexpected local engineering observation",
    )
    event = signed(roots[0] / native["member"], native["sha256"])
    ref = compact_pin(event["reference"])
    run = find_run(roots, ref)
    require(
        verify_bundle(run.root)["manifest_sha256"] == ref["manifest_sha256"],
        "reference original failed integrity readback",
    )
    require(
        set(event["generated_cache_digests"])
        == {"passes.cpython-311.pyc", "scene.cpython-311.pyc", "validation.cpython-311.pyc"}
        and len(event["scope"]) > 0,
        "wrong recorded cache incident",
    )
    rows.append(
        {
            "scope": "public_example_rehearsal_import",
            "status": "local_engineering_incident",
            "recorder_run_id": None,
            "observed_utc": event["actual_time_utc"],
            "observation_sha256": native["sha256"],
            "affected_original": ref,
            "original_bundle_verification_passed": True,
            "extra_cache_files": len(event["generated_cache_digests"]),
            "reason": "An engineering rehearsal imported three sealed Blender helper modules and created three unlisted Python bytecode caches. The caches were relocated intact. The original reference bundle passed verification. Subsequent replay disabled bytecode writes.",
        }
    )
    return rows
