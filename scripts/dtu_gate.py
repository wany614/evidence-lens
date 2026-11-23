"""Hash-bind the object roster, reference quarantine and single D1 opening."""
import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

SOURCES = ("s00", "s01", "s02", "s03")
TARGETS = tuple(f"v{i:02d}" for i in range(1, 7))
METHODS = {"B-L1-transfer": "one", "MS-L1": "all", "MS-frozen": "all", "MS-ft": "all"}


def file_hash(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def pinned(path, expected):
    if not path or not expected or file_hash(path) != expected:
        raise ValueError("missing or changed pinned evidence")
    return json.loads(Path(path).read_bytes())


def verify_members(spec, views_path, marker_path, freeze_path):
    """Check roles before any photo bytes are requested."""
    names = spec.get("members", [])
    if not names:
        return
    photos = [n for n in names if "/Rectified/" in n]
    if not photos:
        metadata = {
            "SampleSet/ReadMe.txt", "SampleSet/CVPR2014bib.txt", "SampleSet/Datasets included.pdf",
            *("SampleSet/Matlab evaluation code/" + f for f in
              ("BaseEvalMain_web.m", "BaseEval2Obj_web.m", "GetUsedSets.m", "PointCompareMain.m")),
            *("SampleSet/MVS Data/Calibration/cal18/" + f for f in
              ("calib_data.mat", "Calib_Results_left.m", "Calib_Results_left.mat",
               "Calib_Results_right.m", "Calib_Results_right.mat"))
        }
        camera = re.compile(r"SampleSet/MVS Data/Calibration/cal18/pos_(?:00[1-9]|0[1-5][0-9]|06[0-4])\.txt")
        if any(name not in metadata and not camera.fullmatch(name) for name in names):
            raise ValueError("unregistered documentation or calibration member")
        return
    view_set = pinned(views_path, spec["view_set_sha256"])
    obj, role = spec["object_id"], spec["access_role"]
    if obj not in ("set1", "set6") or role not in ("source", "truth") or view_set["lighting_id"] != 3:
        raise ValueError("unknown D1 image role")
    ids = view_set["source_view_ids" if role == "source" else "target_view_ids"]
    if set(ids) != set(SOURCES if role == "source" else TARGETS):
        raise ValueError("unregistered image roles")
    expected = {f"SampleSet/MVS Data/Rectified/scan{obj[3:]}/rect_{int(i):03d}_3_r5000.png" for i in ids.values()}
    if set(names) != expected or len(names) != len(expected):
        raise ValueError("only registered members of a single image role may be fetched")
    if obj == "set6":
        marker = pinned(marker_path, spec["authority_sha256"])
        freeze = pinned(freeze_path, marker["freeze_sha256"])
        kind = "el-dtu-opening-1" if role == "source" else "el-dtu-reference-gate-1"
        if (marker["schema"] != kind or marker["object_ids"] != [obj] or
                freeze["protocol"]["external_objects"] != [obj] or
                freeze["view_set_sha256"] != spec["view_set_sha256"]):
            raise ValueError("external photograph requested outside frozen matrix")


def emit(folder, name, data, registry=None):
    folder.mkdir(parents=True, exist_ok=True)
    data["utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    data["run_id"] = os.environ["EL_RUN_ID"]
    raw = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with (folder / name).open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    if registry is not None:
        # Atomic hard link makes the first decision immutable under concurrent callers.
        registry.mkdir(parents=True, exist_ok=True)
        os.link(folder / name, registry / name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["freeze", "opening", "reference"])
    for name in ("request", "out", "source"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("protocol", "view-set", "camera-plan", "release", "freeze", "opening"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--evidence", action="append", type=Path, default=[])
    p.add_argument("--provenance", action="append", type=Path, default=[])
    a = p.parse_args()
    spec = json.loads(a.request.read_text())
    if a.action == "freeze":
        protocol = pinned(a.protocol, spec["protocol_sha256"])
        views = pinned(a.view_set, spec["view_set_sha256"])
        plan = pinned(a.camera_plan, spec["camera_plan_sha256"])
        release = pinned(a.release, spec["candidate_sha256"])
        if (protocol["external_objects"] != ["set6"] or protocol["development_objects"] != ["set1"] or
                protocol["scopes"] != METHODS or set(views["source_view_ids"]) != set(SOURCES) or
                set(views["target_view_ids"]) != set(TARGETS) or plan["view_set_sha256"] != spec["view_set_sha256"] or
                protocol["models"] != release["models"] or release["default_method"] != "MS-L1"):
            raise ValueError("freeze differs from registered study, camera plan or published model")
        # The recorder snapshot, including each numerical script and the exact optimizer, is immutable.
        pins = spec["code_sha256"]
        if not pins or any(file_hash(a.source / n) != h for n, h in pins.items()):
            raise ValueError("scientific code differs from registered bytes")
        if not spec.get("required_receipts"):
            raise ValueError("freeze requires sealed validation and development lineage")
        snapshot = json.loads((a.source.parent / "code_snapshot.json").read_bytes())
        emit(a.out, "freeze.json", {"schema":"el-dtu-freeze-1", "protocol":protocol, "view_set":views,
             "source_commit":snapshot["git_commit"], "source_snapshot_sha256":snapshot["snapshot_sha256"],
             "camera_plan":plan, **{k:spec[k] for k in ("candidate_sha256","view_set_sha256","camera_plan_sha256",
             "protocol_sha256","code_sha256","required_receipts","diagnostic_decision")}}, a.registry)
        return
    freeze = pinned(a.freeze, spec["freeze_sha256"])
    if freeze["schema"] != "el-dtu-freeze-1" or freeze["protocol"]["external_objects"] != ["set6"]:
        raise ValueError("unrecognized frozen cohort")
    if file_hash(a.registry / "freeze.json") != spec["freeze_sha256"]:
        raise ValueError("authority was frozen under a different registry")
    common = {"freeze_sha256":spec["freeze_sha256"], "object_ids":["set6"]}
    if a.action == "opening":
        emit(a.out, "opening.json", {**common, "schema":"el-dtu-opening-1",
             "purpose":"source-only reconstruction and frozen score", "exposed":"registered external source images and calibration only"}, a.registry)
    else:
        opening = pinned(a.opening, spec["opening_sha256"])
        cells = spec["terminal_cells"]
        if (opening["schema"] != "el-dtu-opening-1" or opening["freeze_sha256"] != spec["freeze_sha256"] or
                set(cells) != {"M2","M4"}):
            raise ValueError("reference release requires terminal outcomes from both registered conditions")
        if file_hash(a.registry / "opening.json") != spec["opening_sha256"]:
            raise ValueError("unregistered source opening")
        pins = {file_hash(path): path for path in a.evidence}
        provenance = {file_hash(path): path for path in a.provenance}
        for condition, cell in cells.items():
            if not re.fullmatch(r"[0-9a-f]{64}", cell.get("manifest_sha256", "")):
                raise ValueError("missing original terminal manifest hash")
            if cell["status"] not in {"ready","render_failed","score_failed"} or set(cell["methods"]) != set(METHODS):
                raise ValueError("incomplete method or failure roster")
            document = pinned(pins.get(cell["sha256"]), cell["sha256"])
            if cell["status"] == "ready" and (document.get("schema") != "el-dtu-frozen-scores-1" or
                    document.get("condition") != condition or document.get("method_scopes") != METHODS or
                    document.get("run_id") != cell["run_id"]):
                raise ValueError("terminal score receipt differs from declared condition or methods")
            if cell["status"] == "ready":
                case = pinned(provenance.get(cell["case_provenance"]["sha256"]), cell["case_provenance"]["sha256"])
                if (case["opening_sha256"] != spec["opening_sha256"] or case["object_id"] != "set6" or
                        case["condition"] != condition or case["case_sha256"] != document["case_sha256"] or
                        case["generation"]["run_id"] != document["reconstruction_run_id"] or
                        case["camera_plan_sha256"] != freeze["camera_plan_sha256"] or
                        document["models"] != freeze["protocol"]["models"] or set(document["views"]) != set(TARGETS) or
                        any(view.get("status") == "ready" and set(view["methods"]) != set(METHODS)
                            for view in document["views"].values())):
                    raise ValueError("terminal score receipt is not from the opened external case")
            if cell["status"] != "ready" and (document.get("execution_status") not in {"failed", "interrupted", "cancelled"} or
                    document.get("run_id") != cell["run_id"]):
                raise ValueError("missing independent terminal failure receipt")
        emit(a.out, "reference-gate.json", {**common, "schema":"el-dtu-reference-gate-1",
             "opening_sha256":spec["opening_sha256"], "terminal_cells":cells,
             "purpose":"one independent evaluation of the complete registered matrix"}, a.registry)


if __name__ == "__main__":
    main()
