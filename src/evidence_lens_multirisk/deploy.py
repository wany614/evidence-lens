"""Load and score a complete trained multi-source CPU package without ground truth."""

import importlib.util
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from evidence_lens_vision.contract import bound, digest, file_record, output_dir, put, read, require
from evidence_lens_vision.model import RiskModel
from .data import source_ids, validated_preprocessing
from .model import B_INDEX, B_WEIGHTS, METHODS, architecture, native_score

RUNTIME = {
    "evidence_lens": (
        "__init__",
        "features",
        "scoring_io",
        "l1",
        "risk_model",
        "multisource",
        "multisource_io",
        "multisource_model",
    ),
    "evidence_lens_vision": ("__init__", "model", "contract"),
    "evidence_lens_multirisk": ("__init__", "model", "data", "deploy"),
}
RUNTIME_MEMBERS = {f"runtime/{pkg}/{n}.py" for pkg, names in RUNTIME.items() for n in names}
PACKAGE_MEMBERS = {
    "weights.safetensors",
    "preprocessing.json",
    "feature-schema.json",
    "architecture.json",
    "B-model.json",
    "DINOv2-LICENSE",
    "LICENSE",
    "NOTICE",
    "score-native.py",
    "requirements.txt",
} | RUNTIME_MEMBERS


def export(model, directory, prep_path, schema_path, upstream_index, identity, threshold):
    dest = output_dir(directory)
    state = {n: t.detach().cpu().contiguous() for n, t in model.state_dict().items()}
    save_file(state, str(dest / "weights.safetensors"), metadata={"format": "pt"})
    pre = read(prep_path)
    put(dest / "architecture.json", architecture(pre["columns"]))
    for p, name in (
        (prep_path, "preprocessing.json"),
        (schema_path, "feature-schema.json"),
        (upstream_index, "B-model.json"),
    ):
        shutil.copyfile(p, dest / name)
    # These bytes accompany the registered, upstream-trained public model and will also travel with its C descendants.
    base = Path(upstream_index)
    b = read(base)
    for name in ("DINOv2-LICENSE", "LICENSE", "NOTICE"):
        orig = bound(base.parent, b["files"][name])
        shutil.copyfile(orig, dest / name)
    for pkg, names in RUNTIME.items():
        target = dest / "runtime" / pkg
        target.mkdir(parents=True)
        for n in names:
            spec = importlib.util.find_spec(pkg if n == "__init__" else pkg + "." + n)
            require(
                spec is not None and spec.origin and Path(spec.origin).is_file(),
                "missing deployed source: " + pkg + "." + n,
            )
            shutil.copyfile(spec.origin, target / (n + ".py"))
    launcher = (
        "from pathlib import Path\nimport sys\n"
        "sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))\n"
        "from evidence_lens_multirisk.deploy import standalone\nstandalone(Path(__file__).resolve().parent)\n"
    )
    (dest / "score-native.py").write_bytes(launcher.encode())
    (dest / "requirements.txt").write_bytes(
        b"numpy>=1.26,<3\ntorch>=2.8,<3\nsafetensors>=0.5,<1\nPillow>=11,<13\n"
    )
    files = {
        p.relative_to(dest).as_posix(): file_record(p, dest) for p in sorted(dest.rglob("*")) if p.is_file()
    }
    require(set(files) == PACKAGE_MEMBERS, "unexpected release file in trained package")
    idx = {
        **identity,
        "schema": "el-multisource-risk-model-1",
        "threshold": float(threshold),
        "files": files,
        "precision": "float32",
        "complete_backbone_included": True,
        "encoder": "dinov2_vits14",
        "source_B_index_sha256": B_INDEX,
        "source_B_weights_sha256": B_WEIGHTS,
        "licensed_base": "Apache-2.0",
        "total_parameters": sum(t.numel() for t in model.parameters()),
        "tensor_count": len(state),
    }
    put(dest / "model.json", idx)
    return idx


def load_model(folder, expected, device="cpu"):
    folder = Path(folder)
    require(digest(folder / "model.json") == expected, "deployed model index SHA differs")
    idx = read(folder / "model.json")
    require(
        idx["schema"] == "el-multisource-risk-model-1"
        and idx["method"] in METHODS
        and idx["source_B_index_sha256"] == B_INDEX
        and idx["source_B_weights_sha256"] == B_WEIGHTS
        and idx["precision"] == "float32"
        and idx["complete_backbone_included"] is True
        and set(idx["files"]) == PACKAGE_MEMBERS,
        "incomplete multi-source model index",
    )
    checked = {n: bound(folder, s) for n, s in idx["files"].items()}
    require(
        all(p.relative_to(folder).as_posix() == n for n, p in checked.items())
        and digest(checked["B-model.json"]) == B_INDEX
        and digest(checked["feature-schema.json"]) == idx["feature_schema_sha256"],
        "wrong original or feature schema",
    )
    pre = validated_preprocessing(
        checked["preprocessing.json"],
        idx["files"]["preprocessing.json"]["sha256"],
        idx["training_dataset_sha256"],
        idx["feature_schema_sha256"],
    )
    require(
        read(checked["architecture.json"]) == architecture(pre["columns"]) and np.isfinite(idx["threshold"]),
        "architecture/operating point differs",
    )
    model = RiskModel(features=len(pre["columns"]))
    tensors = load_file(str(checked["weights.safetensors"]), device="cpu")
    require(
        set(tensors) == set(model.state_dict())
        and len(tensors) == idx["tensor_count"] == 181
        and all(t.dtype == torch.float32 and torch.isfinite(t).all().item() for t in tensors.values())
        and sum(t.numel() for t in tensors.values()) == idx["total_parameters"],
        "invalid complete FP32 checkpoint",
    )
    model.load_state_dict(tensors, strict=True)
    return model.to(device).eval(), pre, idx


def score_grid(model_info, rgb, x, ok, device, *, rectangular=False):
    model, pre, idx = model_info
    return native_score(model, rgb, x, ok, pre, device, idx["threshold"], rectangular=rectangular)


def score_native(package, index_sha, case, case_sha, view, scope, output, threads, model_info=None):
    from evidence_lens.multisource_io import load_native

    torch.set_num_threads(threads)
    out = output_dir(output)
    start = time.perf_counter()
    m = model_info if model_info is not None else load_model(package, index_sha)
    loaded = time.perf_counter()
    rgb, features = load_native(Path(case), view, scope, expected_case_sha256=case_sha)
    require(rgb.shape == (256, 256, 3), "the registered C deployment grid is 256 by 256")
    case_doc = read(Path(case) / "case.json")
    require(digest(Path(case) / "case.json") == case_sha, "native case identity changed during scoring")
    prepared = time.perf_counter()
    arrays = score_grid(m, rgb, features["X"], features["scoreable"], "cpu")
    scored = time.perf_counter()
    np.savez_compressed(out / "scores.npz", **arrays)
    result = {
        "schema": "el-multisource-native-score-1",
        "model_sha256": index_sha,
        "method": m[2]["method"],
        "case_sha256": case_sha,
        "view": view,
        "scope": scope,
        "condition": case_doc["condition"],
        "generator_source_ids": case_doc["generator_source_ids"],
        "scorer_source_ids": source_ids(case_doc["condition"], scope),
        "shape": list(rgb.shape[:2]),
        "supported": int(arrays["valid"].sum()),
        "native_unknown": int((~arrays["valid"]).sum()),
        "threshold": m[2]["threshold"],
        "threads": threads,
        "timings": {
            "model_load_seconds": loaded - start,
            "source_loading_and_multiview_projection_seconds": prepared - loaded,
            "image_preprocessing_encoder_head_seconds": scored - prepared,
            "score_write_seconds": time.perf_counter() - scored,
            "complete_seconds": time.perf_counter() - start,
            "model_preloaded": model_info is not None,
        },
        "score": file_record(out / "scores.npz", out),
    }
    put(out / "native-score.json", result)
    return result


def standalone(package):
    import argparse
    import json

    p = argparse.ArgumentParser(
        description="Score one original-grid predicted view from an explicitly allowed source subset."
    )
    for n in ("model-sha256", "case-sha256", "view", "scope"):
        p.add_argument("--" + n, required=True)
    for n in ("case", "output"):
        p.add_argument("--" + n, required=True, type=Path)
    p.add_argument("--threads", default=4, type=int)
    a = p.parse_args()
    print(
        json.dumps(
            score_native(
                package, a.model_sha256, a.case, a.case_sha256, a.view, a.scope, a.output, a.threads
            ),
            sort_keys=True,
        )
    )
