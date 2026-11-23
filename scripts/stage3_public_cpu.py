"""Run four public native risk models or independently replay their registered development example."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from time import perf_counter

METHODS = ("B-L1-transfer", "MS-L1", "MS-frozen", "MS-ft")
SCOPES = ("one", "two", "all")
VIEWS = tuple(f"v{i:02d}" for i in range(1, 7))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def checked_json(path, digest_value):
    info = Path(path).read_bytes()
    require(
        isinstance(digest_value, str) and hashlib.sha256(info).hexdigest() == digest_value,
        "incorrect published identity for " + Path(path).name,
    )
    return json.loads(info)


def member(root, spec):
    name = spec["path"]
    parts = name.split("/")
    require(all(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", p) for p in parts), "nonportable member name")
    p = Path(root) / name
    require(
        p.is_file()
        and p.resolve() == p
        and p.stat().st_size == spec["bytes"]
        and digest(p) == spec["sha256"],
        "declared file missing or altered: " + name,
    )
    return p


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def source_scope(c, s):
    require(c in ("M2", "M4") and s in SCOPES, "invalid reconstruction or scoring set")
    return (
        ["s00"] if s == "one" else ["s00", "s03"] if s == "two" or c == "M2" else ["s00", "s01", "s02", "s03"]
    )


class Runtime:
    def __init__(self, root, expected, threads):
        self.root = Path(root).resolve()
        self.meta = checked_json(self.root / "release.json", expected)
        require(
            self.meta["schema"] == "el-stage3-cpu-release-1"
            and self.meta["version"] == "v0.3.0-rc.1"
            and set(self.meta["models"]) == set(METHODS)
            and 1 <= threads <= 32,
            "unexpected Stage C release or thread count",
        )
        self.specs = self.meta["models"]
        self.models = {}
        # The published index pins these exact import bytes from a trained package before executable imports.
        first = self.root / "models/MS-frozen"
        index = checked_json(first / "model.json", self.specs["MS-frozen"]["model_index_sha256"])
        for n, s in index["files"].items():
            if n.startswith("runtime/"):
                require(member(first, s) == first / n, "runtime path differs from index")
        sys.path.insert(0, str(first / "runtime"))
        self.threads = threads

    def load(self, name):
        from evidence_lens.multisource_model import load_light, load_transfer

        require(name in self.specs, "unavailable risk model")
        if name not in self.models:
            spec = self.specs[name]
            folder = self.root / "models" / name
            target = spec["model_index_sha256"]
            if name == "B-L1-transfer":
                self.models[name] = load_transfer(folder, target, spec["weights_sha256"], spec["threshold"])
            elif name == "MS-L1":
                model, coef, pre = load_light(folder, target)
                require(
                    model["model_id"] == spec["model_id"]
                    and model["threshold"] == spec["threshold"]
                    and model["weights_sha256"] == spec["weights_sha256"],
                    "light model differs",
                )
                self.models[name] = (coef, pre)
            else:
                import torch
                from evidence_lens_multirisk.deploy import load_model

                torch.set_num_threads(self.threads)
                obj = load_model(folder, target)
                require(
                    obj[2]["phase"] == "formal"
                    and obj[2]["method"] == name
                    and obj[2]["seed"] == 1729
                    and obj[2]["model_id"] == spec["model_id"]
                    and obj[2]["threshold"] == spec["threshold"]
                    and obj[2]["files"]["weights.safetensors"]["sha256"] == spec["weights_sha256"],
                    "wrong primary neural package",
                )
                self.models[name] = obj
        return self.models[name]

    def score(self, name, case, view, scope):
        import numpy as np
        from evidence_lens.l1 import risk_state
        from evidence_lens.multisource_model import light_risk, transfer_risk

        require(view in VIEWS and scope in SCOPES, "undeclared target or source scope")
        params = self.load(name)
        spec = self.specs[name]
        if name == "B-L1-transfer":
            target = case.frame(view)
            require(target is not None, "original prediction failed")
            risk = transfer_risk(
                target, case.cameras[view], case.sources("one")["s00"], params, case.pred["depth_verified"]
            )
        else:
            rgb, feat = case.features(view, scope)
            require(rgb.shape == (256, 256, 3), "original 256-pixel prediction required")
            if name == "MS-L1":
                risk = light_risk(feat, *params)
            else:
                from evidence_lens_multirisk.deploy import score_grid

                return score_grid(params, rgb, feat["X"], feat["scoreable"], "cpu")
        return {"risk": risk, "valid": np.isfinite(risk), "state": risk_state(risk, spec["threshold"])}


def arrays(source, tau):
    import numpy as np

    with np.load(source, allow_pickle=False) as z:
        require(set(z.files) == {"risk", "valid", "state"}, "invalid native score fields")
        row = {k: z[k] for k in z.files}
    r, v, s = (row[k] for k in ("risk", "valid", "state"))
    require(
        r.dtype == np.float32
        and v.dtype == bool
        and s.dtype == np.uint8
        and r.shape == v.shape == s.shape == (256, 256)
        and np.array_equal(v, np.isfinite(r))
        and np.all(np.isnan(r[~v]))
        and np.array_equal(s[v], r[v].astype(np.float64) >= tau)
        and np.all(s[~v] == 255),
        "invalid native numerical values, support or frozen decision",
    )
    return row


def comparison(row, gold, scope, limit, strict_state):
    import numpy as np

    require(np.array_equal(row["valid"], gold["valid"]), "native support differs for " + scope)
    mask = row["valid"]
    d = np.abs(row["risk"][mask].astype(np.float64) - gold["risk"][mask])
    crossings = int(np.count_nonzero(row["state"] != gold["state"]))
    if strict_state:
        require(crossings == 0, "native threshold decisions differ for " + scope)
    maximum = float(d.max(initial=0.0))
    require(maximum <= limit, "native score differs for " + scope + ": " + str(maximum))
    return {
        "valid": int(mask.sum()),
        "unknown": int((~mask).sum()),
        "max_abs_error": maximum,
        "mean_abs_error": float(d.mean()) if len(d) else None,
        "threshold_crossings": crossings,
        "equal_values": int(np.count_nonzero(row["risk"][mask] == gold["risk"][mask])),
    }


def risk_output(runtime, case, method, view, scope, out):
    import numpy as np
    from PIL import Image

    start = perf_counter()
    result = runtime.score(method, case, view, scope)
    p = out / "scores.npz"
    np.savez_compressed(p, **result)
    risk, valid, state = (result[k] for k in ("risk", "valid", "state"))
    require(
        risk.shape == valid.shape == state.shape == (256, 256)
        and np.all((risk[valid] >= 0) & (risk[valid] <= 1)),
        "cannot render a non-native or out-of-range risk grid",
    )
    # Same scalar ramp and three deployable state colours as the independent scientific boards.
    ramp = np.array(
        [[23, 71, 96], [37, 138, 151], [137, 200, 162], [245, 205, 123], [207, 63, 68]], np.float64
    )
    points = np.linspace(0, 1, len(ramp))
    unknown = (113, 118, 127)
    low = (75, 168, 184)
    high = (230, 99, 67)
    colors = np.full((*risk.shape, 3), unknown, np.uint8)
    decisions = colors.copy()
    colors[valid] = np.rint(
        np.stack([np.interp(risk[valid], points, ramp[:, ch]) for ch in range(3)], axis=-1)
    ).astype(np.uint8)
    decisions[state == 0] = low
    decisions[state == 1] = high
    images = {}
    for label, data in (("risk", colors), ("state", decisions)):
        path = out / (label + ".png")
        Image.fromarray(data).save(path)
        images[label] = {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
    r = {
        "schema": "el-stage3-public-native-score-1",
        "version": runtime.meta["version"],
        "method": method,
        **runtime.specs[method],
        "case_sha256": case.sha256,
        "condition": case.condition,
        "generator_source_ids": list(case.generator),
        "scorer_source_ids": source_scope(case.condition, "one" if method == "B-L1-transfer" else scope),
        "requested_scope": scope,
        "view": view,
        "valid": int(result["valid"].sum()),
        "shape": list(result["risk"].shape),
        "seconds": perf_counter() - start,
        "score": {"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)},
        "images": images,
        "display": {
            "grid": [256, 256],
            "risk_range": [0, 1],
            "risk_ramp_srgb8": ramp.astype(int).tolist(),
            "low_srgb8": list(low),
            "high_srgb8": list(high),
            "unknown_srgb8": list(unknown),
            "low_rule": "finite risk strictly below the frozen threshold",
            "spatial_resampling": False,
        },
    }
    write(out / "score.json", r)
    return r


def check_error(reference, prediction, expected, row):
    import numpy as np

    with np.load(reference, allow_pickle=False) as z:
        ref = z["rgb"]
    with np.load(prediction, allow_pickle=False) as z:
        pred = z["rgb"]
    require(
        ref.shape == pred.shape == (256, 256, 3) and ref.dtype == pred.dtype == np.float32,
        "incompatible original reference or prediction",
    )
    good = lambda a: np.isfinite(a).all(2) & (a >= 0).all(2) & (a <= 1).all(2)
    v = good(ref)
    e = v & good(pred)
    mae = np.full(v.shape, np.nan, np.float32)
    mae[e] = np.mean(np.abs(pred[e] - ref[e]), axis=1, dtype=np.float32)
    y = np.full(v.shape, 255, np.uint8)
    y[e] = (mae[e] > np.float32(0.05)).astype(np.uint8)
    with np.load(expected, allow_pickle=False) as d:
        require(
            set(("V", "E", "Y", "rgb_mae")) <= set(d.files)
            and np.array_equal(v, d["V"])
            and np.array_equal(e, d["E"])
            and np.array_equal(y, d["Y"])
            and np.array_equal(mae, d["rgb_mae"], equal_nan=True),
            "independent original RGB error/label pixels differ",
        )
    counts = {"V": int(v.sum()), "E": int(e.sum()), "Y": int((y == 1).sum())}
    require(all(counts[k] == row[k] for k in counts), "independent native RGB denominators differ")
    return counts


def replay(runtime, example, supplied_sha, out):
    import numpy as np
    from evidence_lens.multisource_io import Case

    example = Path(example).resolve()
    files = checked_json(example / "example.json", supplied_sha)
    require(
        files["schema"] == "el-stage3-public-example-1"
        and files["family"].startswith("scene_057_")
        and files["method_sha256"] == runtime.meta["method_sha256"],
        "fixed development example or method does not match",
    )
    expected = example / "EXPECTED"
    example_index = checked_json(
        expected / "expected-example.json", files["files"]["EXPECTED/expected-example.json"]["sha256"]
    )
    require(
        example_index["schema"] == "el-stage3-public-expected-example-1"
        and example_index["family"] == files["family"]
        and example_index["catalog_sha256"] == runtime.meta["catalog_sha256"]
        and example_index["study_sha256"] == runtime.meta["study_sha256"],
        "wrong independently measured example",
    )
    rows = example_index["cases"]
    bykey = {(r["condition"], r["view"]): r for r in rows}
    require(
        len(bykey) == len(rows) == 12 and set(bykey) == {(c, v) for c in ("M2", "M4") for v in VIEWS},
        "missing original representative target",
    )
    # The CPU method receives only its deployment CASE directory. Reference and expectations are
    # independently opened here after the measured score for each native grid has been computed.
    results = []
    errors = []
    indices = {}
    scientific_indices = {}
    for m in METHODS:
        n = f"EXPECTED/{m}/native-scores.json"
        ix = checked_json(example / n, files["files"][n]["sha256"])
        require(
            ix["schema"] == "el-stage3-native-example-scores-1"
            and ix["method"] == m
            and ix["family"] == files["family"]
            and all(
                ix[k] == runtime.specs[m][k]
                for k in ("model_id", "model_index_sha256", "weights_sha256", "threshold")
            )
            and ix["catalog_sha256"] == runtime.meta["catalog_sha256"],
            "original native comparison identity differs for " + m,
        )
        indexed = {(r["condition"], r["view"]): r for r in ix["views"]}
        require(
            len(indexed) == len(ix["views"]) == 12 and set(indexed) == set(bykey),
            "missing original native comparison target",
        )
        indices[m] = indexed
        scientific = checked_json(expected / (m + "-scores.json"), example_index["scores"][m]["sha256"])
        require(
            scientific["schema"] == "el-stage3-public-expected-scores-1"
            and all(
                scientific[k] == ix[k]
                for k in (
                    "method",
                    "family",
                    "catalog_sha256",
                    "threshold",
                    "model_index_sha256",
                    "model_id",
                    "weights_sha256",
                )
            ),
            "scientific expected maps refer to another released model",
        )
        scientific_indices[m] = {(r["condition"], r["view"]): r for r in scientific["views"]}
        require(
            len(scientific["views"]) == len(scientific_indices[m]) == 12
            and set(scientific_indices[m]) == set(bykey),
            "incomplete scientific expected score index",
        )
    for c in ("M2", "M4"):
        case = Case(example / "CASE" / c, files["files"][f"CASE/{c}/case.json"]["sha256"])
        for v in VIEWS:
            for name in METHODS:
                indexed = indices[name][c, v]
                require(
                    indexed["case_sha256"] == case.sha256 and indexed["status"] == "ready",
                    "incorrect native comparison reference",
                )
                actual = {}
                for s in SCOPES:
                    ex = indexed["modes"][s]
                    alias = "two" if (c, s) == ("M2", "all") else None
                    require(
                        ex["alias_of"] == alias and ex["source_ids"] == source_scope(c, s),
                        "unexpected native comparison input scope",
                    )
                    if alias:
                        require(
                            ex["file"] == indexed["modes"][alias]["file"],
                            "M2 all and two must identify the exact same gold",
                        )
                        measured = actual[alias]
                    else:
                        measured = runtime.score(name, case, v, s)
                    actual[s] = measured
                    canonical = member(expected / name, ex["file"])
                    gold = arrays(canonical, runtime.specs[name]["threshold"])
                    value = comparison(measured, gold, f"{name}/{c}/{v}/{s}", 0.0001, False)
                    sci = scientific_indices[name][c, v]
                    require(
                        sci["case_sha256"] == case.sha256
                        and sci["status"] == "ready"
                        and sci["modes"][s]["alias_of"] == alias
                        and sci["modes"][s]["source_ids"] == source_scope(c, s),
                        "scientific expected map uses another condition",
                    )
                    prior = sci["modes"][s]["file"]
                    official = arrays(member(expected, prior), runtime.specs[name]["threshold"])
                    scientific_value = comparison(
                        measured, official, f"scientific {name}/{c}/{v}/{s}", 0.0001, False
                    )
                    p = out / "scores" / name / c / s / (v + ".npz")
                    p.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(p, **measured)
                    results.append(
                        {
                            "method": name,
                            "condition": c,
                            "view": v,
                            "scope": s,
                            "alias_of": alias,
                            "prediction_case_sha256": case.sha256,
                            "expected_sha256": ex["file"]["sha256"],
                            "measured_sha256": digest(p),
                            "effective_scorer_source_ids": source_scope(
                                c, "one" if name == "B-L1-transfer" else s
                            ),
                            "scientific_score_sha256": prior["sha256"],
                            "scientific_comparison": scientific_value,
                            **value,
                        }
                    )
            row = bykey[c, v]
            refnames = [
                n
                for n, entry in files["files"].items()
                if n.startswith("REFERENCE/")
                and n.endswith("/" + v + ".npz")
                and entry["sha256"] == row["reference_sha256"]
            ]
            require(len(refnames) == 1, "missing or ambiguous ground-truth source for " + v)
            ref = member(example, {"path": refnames[0], **files["files"][refnames[0]]})
            pred = member(case.root, case.meta["renders"][v])
            require(
                digest(pred) == row["prediction_sha256"], "independent reference mistake for " + c + "/" + v
            )
            errors.append(
                {"condition": c, "view": v, **check_error(ref, pred, member(expected, row["error"]), row)}
            )
    doc = {
        "schema": "el-stage3-public-cpu-replay-1",
        "status": "passed",
        "version": runtime.meta["version"],
        "catalog_sha256": runtime.meta["catalog_sha256"],
        "method_sha256": runtime.meta["method_sha256"],
        "example_sha256": supplied_sha,
        "models": runtime.specs,
        "family": files["family"],
        "scheduled_comparisons": len(results),
        "distinct_method_scope_views": sum(r["alias_of"] is None for r in results),
        "score_comparisons": results,
        "reference_evaluations": errors,
        "max_abs_error": max(r["max_abs_error"] for r in results),
        "threshold_crossings": sum(r["threshold_crossings"] for r in results if not r["alias_of"]),
        "comparison_rule": "Equal support, each state derived by the frozen strict-low threshold, pointwise maximum absolute FP32 risk difference <= 1e-4; every crossing is reported.",
    }
    require(
        doc["scheduled_comparisons"] == 144 and doc["distinct_method_scope_views"] == 120,
        "incomplete conditional/source matrix",
    )
    write(out / "grade.json", doc)
    return doc


def main():
    if sys.version_info < (3, 11):
        raise SystemExit("Stage C requires Python 3.11 or newer.")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("run", "replay"):
        a = sub.add_parser(name)
        a.add_argument("--release-sha256", required=True)
        a.add_argument("--output", type=Path, required=True)
        a.add_argument("--threads", type=int, default=4)
        if name == "run":
            a.add_argument("--case", type=Path, required=True)
            a.add_argument("--case-sha256", required=True)
            a.add_argument("--view", choices=VIEWS, required=True)
            a.add_argument("--scope", choices=SCOPES, required=True)
            a.add_argument("--method", choices=METHODS)
        else:
            a.add_argument("--example", type=Path, required=True)
            a.add_argument("--example-sha256", required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(not any(out.iterdir()), "use a fresh output folder")
    runtime = Runtime(Path(__file__).resolve().parent, a.release_sha256, a.threads)
    if a.action == "run":
        from evidence_lens.multisource_io import Case

        proof = risk_output(
            runtime,
            Case(a.case.resolve(), a.case_sha256),
            a.method or runtime.meta["default_method"],
            a.view,
            a.scope,
            out,
        )
    else:
        proof = replay(runtime, a.example, a.example_sha256, out)
    print(
        json.dumps(
            {
                k: v
                for k, v in proof.items()
                if k not in ("score_comparisons", "reference_evaluations", "models")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
