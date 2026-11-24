"""Lightweight CPU deployment for the frozen single-source transfer and the fitted multi-source LR."""

from pathlib import Path
import json

import numpy as np

from .features import extract_features
from .l1 import feature_rows, load_package, predict_rows, risk_state
from .multisource import COLUMNS, NUMERIC, OPTIONAL, schema_sha256
from .multisource_io import Case, read_member, require
from .scoring_io import file_hash


def validate_rows(x):
    require(
        x.dtype == np.float32 and x.ndim >= 2 and x.shape[-1] == len(COLUMNS),
        "raw features must be native float32 in multisource-v1 column order",
    )
    require(not np.isinf(x).any(), "infinite feature value")
    flags = x[..., len(NUMERIC) :]
    require(np.all((flags == 0) | (flags == 1)), "non-binary missing or empty indicator")
    for j, k in enumerate(OPTIONAL):
        require(
            np.array_equal(flags[..., j].astype(bool), ~np.isfinite(x[..., NUMERIC.index(k)])),
            "missing flag differs from numeric column: " + k,
        )


def checked_preprocessing(p):
    require(
        p.get("schema") == "el-stage3-risk-preprocessing-1"
        and p.get("feature_schema_sha256") == schema_sha256()
        and p.get("columns") == list(COLUMNS)
        and p.get("numeric_count") == len(NUMERIC),
        "preprocessing schema or feature order mismatch",
    )
    for k in ("medians", "means", "scales"):
        a = np.asarray(p[k], np.float64)
        require(
            a.shape == (len(NUMERIC),) and np.isfinite(a).all() and (k != "scales" or np.all(a > 0)),
            "invalid preprocessing " + k,
        )
    return p


def transform(raw, preprocessing, dtype=np.float32):
    """Impute and standardize in float64; binary flags keep their raw values."""
    checked_preprocessing(preprocessing)
    validate_rows(raw)
    x = raw.astype(np.float64)
    q = x[..., : len(NUMERIC)]
    q[:] = (np.where(np.isfinite(q), q, preprocessing["medians"]) - preprocessing["means"]) / preprocessing[
        "scales"
    ]
    return x.astype(dtype, copy=False)


def light_risk(deploy, coefficients, preprocessing):
    raw, ok = deploy["X"], deploy["scoreable"]
    require(raw.shape[:-1] == ok.shape and ok.dtype == bool, "feature mask shape mismatch")
    require(
        coefficients["schema"] == "el-stage3-lr-coefficients-1"
        and coefficients["method"] == "MS-L1"
        and coefficients["feature_schema_sha256"] == schema_sha256()
        and coefficients["columns"] == list(COLUMNS)
        and coefficients["classes"] == [0, 1],
        "incompatible MS-L1 model",
    )
    w = np.asarray(coefficients["coefficients"], np.float64)
    b = coefficients["intercept"]
    require(
        w.shape == (len(COLUMNS),) and np.isfinite(w).all() and np.isscalar(b) and np.isfinite(b),
        "invalid MS-L1 coefficients",
    )
    score = np.full(ok.shape, np.nan, np.float32)
    ids = np.flatnonzero(ok)
    for start in range(0, len(ids), 65536):
        selected = ids[start : start + 65536]
        x = transform(raw.reshape(-1, len(COLUMNS))[selected], preprocessing, np.float64)
        logits = x @ w + b
        score.ravel()[selected] = np.exp(-np.logaddexp(0.0, -logits))
    return score


def load_light(package, model_sha256, *, require_frozen=True):
    p = Path(package)
    require(file_hash(p / "model.json") == model_sha256, "MS-L1 model index digest differs")
    m = json.loads((p / "model.json").read_bytes())
    require(
        m.get("schema") == "el-stage3-light-package-1"
        and m.get("method") == "MS-L1"
        and m.get("feature_schema_sha256") == schema_sha256()
        and m.get("status") in ("fit", "frozen")
        and (not require_frozen or m["status"] == "frozen"),
        "incompatible MS-L1 package or status",
    )
    for n in ("coefficients.json", "preprocessing.json"):
        require(file_hash(p / n) == m["files"][n]["sha256"], "MS-L1 package member differs: " + n)
    c = json.loads(read_member(p, m["files"]["coefficients.json"], "coefficients.json"))
    prep = checked_preprocessing(
        json.loads(read_member(p, m["files"]["preprocessing.json"], "preprocessing.json"))
    )
    require(
        m["weights_sha256"] == m["files"]["coefficients.json"]["sha256"]
        and c["preprocessing_sha256"] == m["files"]["preprocessing.json"]["sha256"],
        "MS-L1 training identity mismatch",
    )
    if m["status"] == "frozen":
        tau = m.get("threshold")
        require(
            isinstance(tau, (float, int)) and np.isfinite(tau) and 0 <= tau <= 1.000001,
            "invalid frozen MS-L1 cutoff",
        )
    return m, c, prep


def load_transfer(package, index_hash, weights_hash, tau):
    model = load_package(package, expected_index_sha256=index_hash)
    require(
        model["file_hashes"]["L1-full.json"]["sha256"] == weights_hash
        and model["thresholds"]["L1-full"] == tau,
        "single-source transfer differs from the frozen parent",
    )
    return model


def transfer_risk(target, target_camera, source, model, depth_verified):
    image, frame, camera = source
    f = extract_features(image, frame, target, camera, target_camera, depth_verified)
    return transfer_feature_risk(f, model)


def transfer_feature_risk(f, model):
    out = np.full(f["scoreable"].shape, np.nan, np.float32)
    ids = np.flatnonzero(f["scoreable"])
    for start in range(0, len(ids), 65536):
        selected = ids[start : start + 65536]
        out.ravel()[selected] = predict_rows(feature_rows(f, selected), model["models"]["L1-full"])
    return out


def native_score(
    case_dir, view, scope, method, package, index_sha256, *, weights_sha256=None, tau=None, case_sha256=None
):
    case = Case(case_dir, case_sha256)
    if method == "B-L1-transfer":
        model = load_transfer(package, index_sha256, weights_sha256, tau)
        frame = case.frame(view)
        require(frame is not None, "failed target")
        risk = transfer_risk(
            frame, case.cameras[view], case.sources("one")["s00"], model, case.pred["depth_verified"]
        )
    elif method == "MS-L1":
        model, coef, prep = load_light(package, index_sha256)
        if tau is not None:
            require(model["threshold"] == tau, "registered model threshold differs")
        tau = model["threshold"]
        _, deploy = case.features(view, scope)
        risk = light_risk(deploy, coef, prep)
    else:
        raise ValueError("method must be B-L1-transfer or MS-L1")
    return {"risk": risk, "valid": np.isfinite(risk), "state": risk_state(risk, tau)}


def main(argv=None):
    import argparse
    from .multisource import SCOPES, TARGETS
    from .scoring_io import new_output, write_json

    p = argparse.ArgumentParser(description=__doc__)
    for key in ("case", "package", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--method", choices=("B-L1-transfer", "MS-L1"), required=True)
    p.add_argument("--model-sha256", required=True)
    p.add_argument("--case-sha256", required=True)
    p.add_argument("--weights-sha256")
    p.add_argument("--tau", type=float)
    p.add_argument("--view", choices=TARGETS, required=True)
    p.add_argument("--scope", choices=SCOPES, required=True)
    a = p.parse_args(argv)
    with new_output(a.output) as out:
        result = native_score(
            a.case,
            a.view,
            a.scope,
            a.method,
            a.package,
            a.model_sha256,
            weights_sha256=a.weights_sha256,
            tau=a.tau,
            case_sha256=a.case_sha256,
        )
        np.savez_compressed(out / "scores.npz", **result)
        write_json(
            out / "score.json",
            {
                "schema": "el-stage3-native-score-1",
                "method": a.method,
                "case_sha256": a.case_sha256,
                "model_sha256": a.model_sha256,
                "view": a.view,
                "scope": a.scope,
                "file": {
                    "path": "scores.npz",
                    "sha256": file_hash(out / "scores.npz"),
                    "bytes": (out / "scores.npz").stat().st_size,
                },
            },
        )


if __name__ == "__main__":
    main()
