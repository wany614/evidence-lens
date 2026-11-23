"""Build or unpack the finite, truth-free stage C neural scientific source bundle."""

import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

B_COMMIT = "fb2fbdc232cf2be706ca88e4ad9ddfe884013cc3"
BASE_FILES = (
    "src/evidence_lens/__init__.py",
    "configs/storage.json",
    "scripts/worker_jobs.py",
    "scripts/vision_gpu_job.py",
    *(
        "src/evidence_lens/" + x + ".py"
        for x in (
            "_bundle",
            "_capacity",
            "_dedup",
            "_index",
            "_metrics",
            "_snapshot",
            "_telemetry",
            "_verify",
            "_worker",
            "recorder",
            "archive",
            "formal_metrics",
            "evaluation",
            "risk_model",
            "features",
            "scoring_io",
            "labels",
            "formal_inputs",
            "comparison_board",
            "l1",
            "l1_data",
            "l1_freeze",
        )
    ),
    *(
        "src/evidence_lens_vision/" + x + ".py"
        for x in ("__init__", "base", "configure", "contract", "deploy", "model", "native", "train")
    ),
)
NEURAL_FILES = (
    *(
        "src/evidence_lens_multirisk/" + x + ".py"
        for x in ("__init__", "data", "model", "train", "deploy", "selection", "inference", "proof")
    ),
    "scripts/multi_risk_pack.py",
    "scripts/multi_risk_worker.py",
    "scripts/multi_risk_job.py",
)
SCORING_FILES = (
    "protocol/multisource-v1.json",
    "protocol/stage3-learning-v1.json",
    *("src/evidence_lens/" + x + ".py" for x in ("multisource", "multisource_io", "multisource_model")),
)
CONTROL_FILES = (
    "protocol/stage3-study-v1.json",
    "src/evidence_lens/stage3_gate.py",
    "src/evidence_lens/stage3_decision.py",
)
REFERENCE_FILES = ("protocol/stage3-families-v1.json",)


def require(ok, why):
    if not ok:
        raise ValueError(why)


def hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def entries(value):
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def all_names():
    return set(BASE_FILES + NEURAL_FILES + SCORING_FILES + CONTROL_FILES + REFERENCE_FILES)


def pack(a):
    data = {
        n: subprocess.check_output(["git", "-C", str(a.repository), "show", B_COMMIT + ":" + n])
        for n in BASE_FILES
    }
    for folder, names in (
        (a.neural, NEURAL_FILES),
        (a.scoring, SCORING_FILES),
        (a.control, CONTROL_FILES),
        (a.reference, REFERENCE_FILES),
    ):
        root = (folder or a.repository).resolve()
        for n in names:
            p = root / n
            require(
                p.is_file() and p.resolve() == p and n not in data,
                "unavailable or duplicate declared source: " + n,
            )
            data[n] = p.read_bytes()
    index = {
        "schema": "el-stage3-neural-scientific-source-1",
        "B_source_commit": B_COMMIT,
        "files": {n: {"bytes": len(raw), "sha256": hash_bytes(raw)} for n, raw in sorted(data.items())},
    }
    data["worker-source.json"] = entries(index)
    a.output.parent.mkdir(exist_ok=True, parents=True)
    with zipfile.ZipFile(a.output, "x", compression=zipfile.ZIP_DEFLATED) as dst:
        for n, raw in sorted(data.items()):
            i = zipfile.ZipInfo(n, (1980, 1, 1, 0, 0, 0))
            i.compress_type = zipfile.ZIP_DEFLATED
            i.create_system = 3
            i.external_attr = 0o100644 << 16
            dst.writestr(i, raw)
    return {
        "sha256": hash_bytes(a.output.read_bytes()),
        "index_sha256": hash_bytes(data["worker-source.json"]),
        "files": len(data),
    }


def extracted(pack, expected, dest):
    data = Path(pack).read_bytes()
    require(hash_bytes(data) == expected, "worker archive differs from the explicit SHA")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        expected_names = all_names() | {"worker-source.json"}
        require(
            names == sorted(expected_names) and all(i.external_attr >> 16 == 0o100644 for i in z.infolist()),
            "worker source member inventory differs",
        )
        index = json.loads(z.read("worker-source.json"))
        require(
            index["schema"] == "el-stage3-neural-scientific-source-1"
            and index["B_source_commit"] == B_COMMIT
            and set(index["files"]) == all_names(),
            "worker source provenance differs",
        )
        dest.mkdir(exist_ok=True, parents=True)
        require(not any(dest.iterdir()), "use an empty worker root")
        for n in names:
            raw = z.read(n)
            if n != "worker-source.json":
                require(
                    index["files"][n] == {"bytes": len(raw), "sha256": hash_bytes(raw)},
                    "invalid source member " + n,
                )
            p = dest / n
            p.parent.mkdir(exist_ok=True, parents=True)
            p.write_bytes(raw)
    return index


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("pack")
    s.add_argument("--repository", type=Path, default=Path("."))
    for n in ("neural", "scoring", "control", "reference"):
        s.add_argument("--" + n, type=Path)
    s.add_argument("--output", type=Path, required=True)
    e = sub.add_parser("extract")
    e.add_argument("--archive", type=Path, required=True)
    e.add_argument("--sha256", required=True)
    e.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(
        json.dumps(
            pack(a) if a.action == "pack" else extracted(a.archive, a.sha256, a.output), sort_keys=True
        )
    )
