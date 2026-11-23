import pytest

from evidence_lens.formal_inputs import PinnedRun
from test_vision_data import seal_fixture
from vision_public_study import engineering_readback


def test_engineering_observations_keep_names_and_paths_private(tmp_path):
    h = "a" * 64
    rid = "20000101T000000000000Z-0000000000000099"
    events = [
        {
            "event": "windows_transfer_deferred",
            "scope": "acquisition_transport",
            "outcome": "resumed_same_identity",
            "secondary_source_sha256": dict.fromkeys(
                ("control", "observation", "resolution", "selection"), h
            ),
            "elapsed_seconds": None,
            "private_path": "/Users/example/private",
        },
        {
            "event": "isolated_mac_bytecode_cache",
            "scope": "engineering_pilot",
            "outcome": "invalid_source_snapshot",
            "secondary_source_sha256": dict.fromkeys(
                ("orphan_manifest", "orphan_events", "orphan_snapshot"), h
            ),
            "elapsed_seconds": 1.25,
            "run_id": rid,
            "unplanned_bytecode_files": 10,
            "child_exit_code": 0,
        },
        {
            "event": "release_example_input_lineage",
            "scope": "release_example",
            "outcome": "rejected_then_repaired",
            "secondary_source_sha256": dict.fromkeys(("reject_manifest", "reject_events"), h),
            "elapsed_seconds": 0.2,
            "run_id": rid,
            "child_started": False,
        },
    ]
    for i in range(2):
        unique = rid[:-1] + str(i)
        if i:
            events[0]["secondary_source_sha256"]["resolution"] = "/Users/example/private"
        pin = seal_fixture(
            tmp_path,
            unique,
            "archive_verify",
            "succeeded",
            {"experiment_id": "stage2-engineering-incident-readback"},
            {
                "payload/engineering-events.json": {
                    "schema": "el-stage2-engineering-events-1",
                    "run_id": unique,
                    "events": events,
                }
            },
        )
        field = PinnedRun(tmp_path, pin)
        if i:
            with pytest.raises(ValueError, match="observation identity"):
                engineering_readback(field)
        else:
            parsed = engineering_readback(field)
            assert "/Users/" not in str(parsed) and parsed["events"][0]["elapsed_seconds"] is None
            assert (
                parsed["events"][-1]["scope"] == "release_example"
                and not parsed["events"][-1]["child_started"]
            )
