"""Read sealed, deployment-only frozen cases and score maps for the numerical bridge."""

import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from evidence_lens.multisource_io import Case
from .bundle import Bundle
from .common import (
    RELEASE_INDEX_SHA256,
    digest,
    exact_keys,
    json_bytes,
    member,
    profile,
    read_file,
    require,
    sha,
)
from .legacy import adapt_authenticated_case
from .numerics import METHODS, checked_arrays


def pinned(pin, filename):
    exact_keys(pin, ("directory", "sha256"), "deployment artifact pin")
    digest(pin["sha256"], filename)
    directory = Path(pin["directory"])
    raw = read_file(directory, filename)
    require(sha(raw) == pin["sha256"], "frozen deployment artifact differs: " + filename)
    return directory, json_bytes(raw)


def check_joint_closure(closure, cells):
    """Read only the recorder seal, its files index and the closed label-blind matrix."""
    exact_keys(closure, ("directory", "run_id", "manifest_sha256", "matrix_sha256"), "joint closure")
    folder = Path(closure["directory"])
    digests = {k: digest(closure[k], k) for k in ("manifest_sha256", "matrix_sha256")}
    run = closure["run_id"]
    raw = read_file(folder.parent, "manifest.json")
    require(sha(raw) == digests["manifest_sha256"], "joint closure run manifest differs")
    manifest = json_bytes(raw)
    completion = json_bytes(read_file(folder.parent, "completion.json"))
    require(
        manifest["run_id"] == completion["run_id"] == run
        and completion["manifest_sha256"] == digests["manifest_sha256"]
        and completion["execution_status"] == "succeeded",
        "joint matrix lacks a successful recorder seal",
    )
    source = read_file(folder, "sealed-matrix.json")
    matches = [f for f in manifest["files"] if f["path"] == "payload/sealed-matrix.json"]
    require(
        len(matches) == 1
        and matches[0]["sha256"] == sha(source) == digests["matrix_sha256"]
        and len(source) == matches[0]["bytes"],
        "sealed matrix is not a member of its recorder run",
    )
    matrix = json_bytes(source)
    require(
        matrix["schema"] == "el-t1-closed-matrix-1"
        and matrix["run_id"] == run
        and len(matrix["cells"]) == 42
        and len(matrix["fixed_score_terminals"]) == 288
        and len(matrix["fixed_cells"]) == 12
        and len(set(matrix["objects"])) == 6,
        "full frozen T1 closure is required before engineering comparison",
    )
    fixed = {(c["object_id"], c["condition"] + "-fixed"): c for c in matrix["fixed_cells"]}
    require(
        set(fixed)
        == {(o, layout) for o in matrix["objects"] for layout in ("M2-fixed", "M4-fixed")}
        == {(c["object_id"], c["layout"]) for c in cells}
        and len(cells) == 12,
        "T1 roster differs from the closed fixed matrix",
    )
    for cell in cells:
        original = fixed[cell["object_id"], cell["layout"]]
        if original["status"] == "failed":
            failure = cell.get("terminal_failure")
            require(
                isinstance(failure, dict)
                and failure.get("reason") == original["reason"]
                and failure.get("receipt_sha256") == digests["matrix_sha256"],
                "failure must cite the exact terminal disposition in the closed matrix",
            )
        else:
            require(
                original["status"] == "succeeded"
                and "terminal_failure" not in cell
                and all(
                    cell[k]["sha256"] == original[old]["sha256"]
                    for k, old in (("case", "case"), ("readback", "readback"), ("historical", "scores"))
                ),
                "frozen case, readback or scores do not belong to the joint matrix",
            )
    return {"run_id": run, **digests}


def historical_case(cell, temporary, models_dir, *, t1):
    case_dir, _ = pinned(cell["case"], "case.json")
    case = Case(case_dir, cell["case"]["sha256"])
    _, readback = pinned(cell["readback"], "reconstruction-readback.json")
    schema = readback["schema"]
    require(
        schema == "el-t1-reconstruction-readback-1"
        if t1
        else schema in ("el-t0-reconstruction-readback-1", "el-b2-reconstruction-readback-1"),
        "wrong readback generation",
    )
    require(
        readback["verified"] is True
        and readback["status"] == "succeeded"
        and readback["scientific_sample"] is True
        and readback["forced_fallback_fixture"] is False
        and readback["target_reference_files_opened"] == 0
        and readback["case_sha256"] == case.sha256
        and readback["object_id"] == cell["object_id"]
        and readback["condition"] == case.condition
        and cell["layout"] == case.condition + "-fixed"
        and readback.get("logical_condition", readback.get("layout", cell["layout"])) == cell["layout"],
        "readback does not authenticate this fixed deployment cell",
    )
    require(
        case.pred["depth_verified"]
        and readback["depth_probe"]["sha256"] == case.pred["depth_probe_sha256"]
        and readback["render_camera_sha256"] == case.meta["render_cameras"]["sha256"]
        and readback["source_camera_sha256"] == case.meta["source_cameras"]["sha256"],
        "readback depth/camera mismatch",
    )
    gen, ren = (readback["origins"][k] for k in ("generation", "render"))
    require(
        gen["run_id"] == case.pred["reconstruction_run_id"]
        and ren["run_id"] == readback["source_renderer_id"],
        "wrong generator or renderer",
    )
    # Only the authenticated JSON geometry/cameras are opened from the fixed reference layout.
    g = Bundle(cell["geometry_bundle"]["directory"], cell["geometry_bundle"]["sha256"])
    require(
        g.order == tuple(case.generator)
        and g.targets == tuple(case.meta["target_ids"])
        and g.condition == case.condition
        and g.primary == "s00"
        and g.endpoints == ("s00", "s03"),
        "registered layout differs from the prepared geometry exemplar",
    )
    for name, c in case.cameras.items():
        reference = g.cameras[name]
        require(
            c.shape == reference.shape
            and np.array_equal(c.K, reference.K)
            and np.array_equal(c.c2w, reference.c2w),
            "metric layout camera differs from authenticated geometry exemplar",
        )
    dest = temporary / "prepared"
    adapt_authenticated_case(
        case,
        dest,
        input_domain="Registered fixed DTU deployment grid",
        generator={"run_id": gen["run_id"], "receipt_sha256": gen["manifest_sha256"]},
        renderer={"run_id": ren["run_id"], "receipt_sha256": ren["manifest_sha256"]},
        geometry=g.meta["geometry"],
        proof_sha256=cell["readback"]["sha256"],
    )
    root, old = pinned(cell["historical"], "scores.json")
    require(
        old["schema"] == "el-dtu-frozen-scores-1"
        and old["release_sha256"] == RELEASE_INDEX_SHA256
        and old["case_sha256"] == case.sha256
        and old["condition"] == case.condition
        and old["checkpoint_sha256"] == case.pred["checkpoint_sha256"]
        and old["generator_source_ids"] == list(case.generator)
        and old["source_camera_sha256"] == case.meta["source_cameras"]["sha256"]
        and old["render_camera_sha256"] == case.meta["render_cameras"]["sha256"]
        and set(old["views"]) == set(case.meta["target_ids"]),
        "wrong prior CPU score matrix",
    )
    for method, spec in profile()["models"].items():
        require(
            all(
                old["models"][method][k] == spec[k]
                for k in ("model_id", "model_index_sha256", "weights_sha256", "threshold")
            ),
            "archived model differs: " + method,
        )
    reasons = temporary / "frozen-reasons"
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(Path(__file__).with_name("frozen_reasons.py")),
            "--models",
            str(models_dir),
            "--case",
            str(case_dir),
            "--case-sha256",
            case.sha256,
            "--out",
            str(reasons),
        ],
        check=True,
    )
    provenance = {
        "scores_sha256": cell["historical"]["sha256"],
        "score_run_id": old["run_id"],
        "readback_sha256": cell["readback"]["sha256"],
        "case_sha256": case.sha256,
        "geometry_template_sha256": g.sha256,
        "release_sha256": old["release_sha256"],
        "archived_threads": old["threads"],
        "archived_environment": old.get("environment"),
        "reason_provenance": "Not stored in historical 3-array NPZ; reconstructed from original CPU feature runtime, frozen B validity predicates and the historical finite mask.",
        "reason_runtime": json.loads((reasons / "origin.json").read_bytes()),
    }

    def read_view(view):
        rows = {}
        position = old["views"][view]
        if position["status"] == "render_failed":
            require(view in case.failed and position["methods"] == {}, "unmatched archived render failure")
            return None
        require(
            position["status"] == "ready"
            and set(position["methods"]) == set(METHODS)
            and view not in case.failed,
            "missing or unexpected archived methods",
        )
        with np.load(reasons / (view + ".npz"), allow_pickle=False) as support:
            for method in METHODS:
                entry = position["methods"][method]
                require(
                    entry["target_prediction_sha256"] == case.meta["renders"][view]["sha256"]
                    and entry["source_ids"]
                    == (["s00"] if method == "B-L1-transfer" else list(case.generator))
                    and entry["scope"] == ("one" if method == "B-L1-transfer" else "all"),
                    "historical score consumes different evidence",
                )
                raw = member(root, entry["file"], expected=f"{view}/{method}/scores.npz")
                with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                    arrays = dict(archive)
                why = support["primary" if method == "B-L1-transfer" else "multisource"].copy()
                if method in ("MS-ft", "MS-frozen"):
                    why[(why == 0) & ~arrays["valid"]] = 8
                rows[method] = checked_arrays(arrays, old["models"][method]["threshold"], legacy_reason=why)
        return rows

    return Bundle(dest), read_view, provenance
