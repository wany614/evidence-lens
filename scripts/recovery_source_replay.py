"""Reproduce the registered source-only initialization and Windows ZIP from immutable inputs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from evidence_lens._bundle import encoded, load, sha
from evidence_lens._verify import require

SCRIPTS = ("recovery_source_replay.py", "recovery_inputs.py", "recovery_triangulate.py")


def run_worker(args):
    """Run SIFT in its own process. Expose the old output to the comparator only after it exits."""
    from argparse import Namespace
    from recovery_inputs import SETS, package

    cfg = load(args.config)
    sources = [(source, Path(path)) for source, path in args.source]
    require([s for s, _ in sources] == list(SETS[cfg["condition_id"]]), "Unexpected replay source roles")
    with tempfile.TemporaryDirectory(prefix="geometry-replay-") as temp:
        scratch = Path(temp)
        original_config = scratch / "original-config.json"
        package_config = scratch / "original-handoff-config.json"
        original_config.write_bytes(encoded(cfg["original_config"]))
        package_config.write_bytes(encoded(cfg["original_handoff_config"]))
        require(
            sha(original_config) == cfg["original_config_sha256"]
            and sha(package_config) == cfg["original_handoff_config_sha256"],
            "Different sealed run configurations",
        )
        computed = scratch / "computed"
        computed.mkdir()
        command = [
            args.geometry_python,
            str(Path(__file__).with_name("recovery_triangulate.py")),
            "--protocol",
            args.protocol,
            "--cameras",
            args.cameras,
            "--config",
            original_config,
            "--out",
            computed,
        ]
        for source, path in sources:
            command.extend(["--source", source, path])
        subprocess.run(command, check=True)
        current = load(computed / "init.json")
        original = load(args.original_meta)
        for name in ("init.npz", "source-tracks.npz"):
            require(sha(computed / name) == cfg["original_file_sha256"][name], "Replay differs: " + name)
        volatile = {"elapsed_seconds", "run_id"}
        numeric = {key: value for key, value in current.items() if key not in volatile}
        require(
            numeric == {key: value for key, value in original.items() if key not in volatile},
            "Different matches, eligible observations, fallback, camera, source hashes or parameters",
        )
        dispatch = scratch / "dispatch"
        dispatch.mkdir()
        package(
            Namespace(
                config=package_config,
                init_meta=args.original_meta,
                init=computed / "init.npz",
                cameras=args.cameras,
                render_cameras=args.render_cameras,
                source=sources,
                out=dispatch,
            )
        )
        require(
            sha(dispatch / "input.zip") == cfg["original_file_sha256"]["input.zip"], "Transport ZIP differs"
        )
        (args.out / "replayed-init.json").write_bytes((computed / "init.json").read_bytes())
        import hashlib

        result = {
            "schema": "el-recovery-source-replay-1",
            "run_id": os.environ["EL_RUN_ID"],
            "passed": True,
            "object_id": cfg["scene_family_id"],
            "condition": cfg["condition_id"],
            "parents": cfg["original_pins"],
            "programs": cfg["program_sha256"],
            "compared_sha256": cfg["original_file_sha256"],
            "scientific_json_sha256": hashlib.sha256(encoded(numeric)).hexdigest(),
            "metadata_differences_allowed": sorted(volatile),
            "verified_points": current["N"],
            "parameter_sha256": current["parameter_sha256"],
            "source_rgb_sha256": current["source_rgb_sha256"],
            "source_camera_sha256": current["source_camera_sha256"],
            "render_camera_sha256_for_zip": sha(args.render_cameras),
            "initialization_inputs": [
                "source_cameras.json",
                *[s + ".png" for s, _ in sources],
                "registered rules and identity",
            ],
            "zip_metadata": "Original run metadata and timestamps are preserved in init.json for byte-identical redelivery.",
        }
        (args.out / "source-replay.json").write_bytes(encoded(result))


def record(args):
    import hashlib
    from make_handoff import publish
    from recovery_inputs import INIT, SETS, copy_script, member, source_role
    from stage3_reference import perform

    project, store = args.project.resolve(), args.store.resolve()
    base = store / "recovery/inputs" / INIT / args.object_id / args.condition
    selected = load(base / "selection.json")
    require(
        selected["object_id"] == args.object_id and selected["condition"] == args.condition,
        "Unexpected original selection",
    )
    pins = {"initialization": selected["init_pin"], "handoff": selected["handoff_pin"]}
    init, handoff = pins["initialization"], pins["handoff"]
    programs = {name: sha(project / "scripts" / name) for name in SCRIPTS}
    key = hashlib.sha256(encoded(programs)).hexdigest()
    dest = base / ("source-replay-" + key + ".json")
    if dest.is_file():
        existing = load(dest)
        require(
            sha(member(store, existing, existing["member"])) == existing["sha256"], "Replaced replay record"
        )
        return existing
    original_cfg = member(store, init, "config.resolved.yaml")
    old = load(original_cfg)
    ids = SETS[args.condition]
    require(
        old["scene_family_id"] == args.object_id and old["generator_source_ids"] == list(ids),
        "Wrong source registration",
    )
    signatures = {
        name: sha(member(store, init, "payload/" + name)) for name in ("init.npz", "source-tracks.npz")
    }
    signatures["input.zip"] = sha(member(store, handoff, "payload/input.zip"))
    handoff_cfg = member(store, handoff, "config.resolved.yaml")
    cfg = {
        **old,
        "experiment_id": "recovery-source-byte-replay-v1",
        "trial_id": "source-replay-" + old["trial_id"] + "-" + key[:12],
        "original_pins": pins,
        "program_sha256": programs,
        "original_file_sha256": signatures,
        "original_config": old,
        "original_handoff_config": load(handoff_cfg),
        "original_config_sha256": sha(original_cfg),
        "original_handoff_config_sha256": sha(handoff_cfg),
        "recording": {"expected_output_bytes": 65536},
    }
    inputs = {
        "original_meta": (member(store, init, "payload/init.json"), "initialization"),
        "protocol": (
            source_role(store, init, "rules", "recovery-initialization-v1.json"),
            "initialization",
        ),
        "cameras": (source_role(store, init, "cameras", "source_cameras.json"), "initialization"),
        "render_cameras": (source_role(store, handoff, "render_cameras", "render_cameras.json"), "handoff"),
        **{s: (source_role(store, init, s, s + ".png"), "initialization") for s in ids},
    }
    scratch = store.parent / "recovery-private-scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        engine = Path(tmp) / "engine"
        for name in SCRIPTS:
            copy_script(project, engine, name)
        command = [
            "{python}",
            "{source}/scripts/recovery_source_replay.py",
            "worker",
            "--config",
            "{config}",
            "--out",
            "{out}",
            "--geometry-python",
            str(args.geometry_python.absolute()),
        ]
        for role in (
            "original_meta",
            "protocol",
            "cameras",
            "render_cameras",
        ):
            command.extend(["--" + role.replace("_", "-"), "{input:" + role + "}"])
        for s in ids:
            command.extend(["--source", s, "{input:" + s + "}"])
        pin = perform(store, project, engine, cfg, "archive_verify", inputs, pins, command)
    proof = "payload/source-replay.json"
    pin.update(member=proof, sha256=sha(member(store, pin, proof)))
    publish(dest, encoded(pin))
    return pin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    entry = commands.add_parser("record")
    entry.add_argument("--store", type=Path, default=Path(".research"))
    entry.add_argument("--project", type=Path, default=Path("."))
    entry.add_argument("--condition", choices=("M2", "M4"), required=True)
    entry.add_argument("--object-id", choices=("scene_057_offset_bell_cage", "set1"), required=True)
    entry.add_argument("--geometry-python", type=Path, required=True)
    entry = commands.add_parser("worker")
    for field in (
        "config",
        "original-meta",
        "protocol",
        "cameras",
        "render-cameras",
        "out",
        "geometry-python",
    ):
        entry.add_argument("--" + field, type=Path, required=True)
    entry.add_argument("--source", nargs=2, action="append", required=True)
    args = parser.parse_args()
    if args.action == "worker":
        run_worker(args)
    else:
        print(json.dumps(record(args)))
