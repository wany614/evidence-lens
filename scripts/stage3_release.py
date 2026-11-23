"""Build the Stage C public CPU, authored example and numerical archives from original research seals."""

import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve()
TAG = "v0.3.0-rc.1"
METHODS = ("B-L1-transfer", "MS-L1", "MS-frozen", "MS-ft")
CONS = ("M2", "M4")
VIEWS = tuple(f"v{i:02d}" for i in range(1, 7))
CAT = "f14ca80777cf72e6f99075fb860e24cd38a6b4d9db342011173e9b216d3c66f5"
STUDY = "dd227c25648af4ab5e4e42d3fdd07702a8e46b3def36810ac9d65c55b07a7b23"
FEATURES = "1f444f0dc9b7fce5c998bb677da20b2c18c25312d7c918e9bf17c6b290457e63"
FORMAL_WORKER = "3cba08e523112084ad352dadcb104d3bbd9a97ec503966dc1efb8eec9cbed045"
FORMAL_WORKER_INDEX = "eef4b94b574af15ff03091bab9c9b3bf5f8ab82c15878b48051f1668553bf8d2"
FAMILY = "scene_057_offset_bell_cage"
LICENSES = {"Apache-2.0", "CC-BY-4.0"}
PERMISSIONS = (
    "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "licenses/CC-BY-4.0.txt",
    "licenses/ASSET-SCOPE.json", "licenses/multisource-assets.json",
)
DOCS = ("README.md", "MODEL_CARD.md", "REBUILD.md")
PROOFS = (
    "training.json",
    "setup.json",
    "initialization.json",
    "initial-head.safetensors",
    "curves.json",
    "gradient-witnesses.json",
    "parameter-changes.json",
    "independent-proof/verified-updates.json",
)
PUBLIC_TABLES = {
    "study-summary.json",
    "family-metrics.csv",
    "region-metrics.csv",
    "risk-coverage.csv",
    "paired-reconstruction.csv",
    "source-ablation.csv",
    "matched-all-scope-ap.csv",
    "resources.csv",
}
LIGHT = {
    "MS-L1": {"model.json", "coefficients.json", "preprocessing.json"},
    "B-L1-transfer": {
        "model.json",
        "feature_schema.json",
        "L1-full.json",
        "L1-geometry.json",
        "thresholds.json",
    },
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def add_notices(repository, commit, directories):
    from vision_release import git_blob, sig, write

    files = {}
    for name in PERMISSIONS:
        data = git_blob(repository, commit, name)
        for directory in directories:
            write(directory, name, data)
        files[name] = {**sig(directories[0] / name), "role": "license_and_attribution",
                       "source": {"commit": commit, "path": name}}
    return files


def public_maps(root, owner, release_type, original):
    from release_model import safe_text
    from vision_release import copy, safe_name, sha, sig

    allow = owner.json("payload/public-members.json")
    require(
        allow["schema"] == "el-stage3-public-members-1"
        and isinstance(allow["files"], list)
        and allow["files"],
        "expected a sealed publication allowlist",
    )
    rows = {}
    sources = {}
    for row in allow["files"]:
        name = safe_name(row["path"])
        lic = row["license"]
        src = row["source"]
        require(
            name not in rows
            and lic in LICENSES
            and all(k in src for k in ("run_id", "manifest_sha256", "member", "sha256")),
            "duplicate path or missing original publication authority: " + name,
        )
        ext = Path(name).suffix.lower()
        if release_type == "example":
            require(
                name.split("/")[0] in ("CASE", "REFERENCE", "EXPECTED")
                and ext
                in (
                    ".npz",
                    ".npy",
                    ".png",
                    ".json",
                    ".jsonl",
                    ".blend",
                    ".py",
                    ".md",
                    ".txt",
                    ".toml",
                    ".csv",
                ),
                "outside the authored public case, ground truth or expected results: " + name,
            )
        else:
            require(
                ext in (".json", ".csv", ".md", ".png", ".txt", ".jsonl")
                and not name.startswith(("CASE/", "REFERENCE/", "EXPECTED/", "models/")),
                "unexpected scientific report member: " + name,
            )
        orig = original(src)
        p = orig.file(src["member"])
        require(
            sha(p) == src["sha256"] and ("bytes" not in row or row["bytes"] == p.stat().st_size),
            "source bytes differ: " + name,
        )
        if ext in (".json", ".jsonl", ".csv", ".md", ".txt", ".toml"):
            safe_text(name, p.read_bytes())
        copy(root, name, p)
        rows[name] = {**sig(p), "license": lic}
        sources[name] = {k: src[k] for k in ("run_id", "manifest_sha256", "member", "sha256")}
    return rows, sources


def check_example(root, table, freeze, method):
    from evidence_lens.multisource_io import Case
    from evidence_lens_vision.contract import bound, read
    from vision_release import sha

    require(
        {
            "CASE/case-index.json",
            "CASE/common-initial.npz",
            "REFERENCE/reference.json",
            "EXPECTED/expected-example.json",
        }
        <= set(table),
        "original registered example, common Gaussian initialization or independent labels are missing",
    )
    public = read(root / "CASE/case-index.json")
    ref = read(root / "REFERENCE/reference.json")
    expected = read(root / "EXPECTED/expected-example.json")
    require(
        public["family"] == ref["family"] == expected["family"] == FAMILY
        and expected["schema"] == "el-stage3-public-expected-example-1"
        and expected["catalog_sha256"] == CAT
        and expected["study_sha256"] == STUDY
        and expected["conditions"] == list(CONS)
        and expected["targets"] == list(VIEWS)
        and expected["primary_view"] == "v01",
        "incorrect authored family or scientific role",
    )
    cases = {}
    endpoints = {}
    for c in CONS:
        base = root / "CASE" / c
        case = Case(base, table[f"CASE/{c}/case.json"]["sha256"])
        p = case.pred
        rec = read(base / "reconstruction.json")
        cases[c] = case.sha256
        require(
            case.condition == c
            and p["depth_verified"] is True
            and set(p["views"]) == set(case.generator) | set(VIEWS)
            and sha(base / "scene.npz") == case.meta["checkpoint_sha256"] == rec["checkpoint_sha256"]
            and sha(root / "CASE/common-initial.npz") == rec["initial_checkpoint_sha256"]
            and rec["run_id"] == case.meta["reconstruction_run_id"]
            and rec["gaussian_optimization_steps"] == 256
            and rec["final_gaussian_count"] == 4096,
            "incomplete selected public joint reconstruction " + c,
        )
        for k in ("source_cameras", "render_cameras", "prediction"):
            bound(base, case.meta[k])
        for n, s in case.meta["source_images"].items():
            inp = bound(base, s)
            require(
                sha(inp) == sha(root / "CASE/sources" / (n + ".png")), "shared original source RGB differs"
            )
            if n in ("s00", "s03"):
                require(endpoints.setdefault(n, s["sha256"]) == s["sha256"], "changed condition endpoint")
        for v, s in case.meta["renders"].items():
            bound(base, s)
            require((base / "predictions" / (v + ".png")).is_file(), "missing authorized native preview")
        for s in case.cameras.values():
            require(s.shape == (256, 256), "incorrect public native dimensions")
    require(
        len({r["case_sha256"] for r in expected["cases"]}) == 2
        and {(r["condition"], r["view"]) for r in expected["cases"]} == {(c, v) for c in CONS for v in VIEWS}
        and all(r["case_sha256"] == cases[r["condition"]] for r in expected["cases"]),
        "incomplete scored representative example",
    )
    for method_name in METHODS:
        ix = read(root / "EXPECTED" / method_name / "native-scores.json")
        original = read(bound(root / "EXPECTED", expected["scores"][method_name]))
        size = method["models"][method_name]
        for table_ix, root_dir, sch in (
            (ix, root / "EXPECTED" / method_name, "el-stage3-native-example-scores-1"),
            (original, root / "EXPECTED", "el-stage3-public-expected-scores-1"),
        ):
            require(
                table_ix["schema"] == sch
                and table_ix["method"] == method_name
                and table_ix["family"] == FAMILY
                and all(
                    table_ix[k] == size[k]
                    for k in ("model_id", "model_index_sha256", "weights_sha256", "threshold")
                ),
                "wrong gold model or calibrations for " + method_name,
            )
            views = table_ix["views"]
            require(
                len(views) == 12
                and {(r["condition"], r["view"]) for r in views} == {(c, v) for c in CONS for v in VIEWS},
                "incomplete expected native risks",
            )
            for r in views:
                require(
                    r["case_sha256"] == cases[r["condition"]]
                    and r["status"] == "ready"
                    and set(r["modes"]) == {"one", "two", "all"},
                    "incorrect gold source or status",
                )
                for scope, entry in r["modes"].items():
                    require(
                        entry["alias_of"] == ("two" if (r["condition"], scope) == ("M2", "all") else None),
                        "incorrect gold alias",
                    )
                    bound(root_dir, entry["file"])
                if r["condition"] == "M2":
                    require(
                        r["modes"]["two"]["file"] == r["modes"]["all"]["file"],
                        "M2 endpoint hypotheses differ",
                    )
    return cases


def scientific_report(allow, method, marker, find):
    """Verify the actual opened cohort and its sealed numerical members once."""
    from vision_release import sha

    source = allow.json("payload/public-members.json")
    candidates = {
        frozenset((("run_id", r["source"]["run_id"]), ("manifest_sha256", r["source"]["manifest_sha256"])))
        for r in source["files"]
    }
    for rid, h in allow.parents().items():
        candidates.add(frozenset((("run_id", rid), ("manifest_sha256", h))))
    candidates.add(frozenset(allow.pin.items()))
    matched = []
    public_hashes = {r["source"]["sha256"] for r in source["files"]}
    joints = []
    for pair in sorted(candidates, key=lambda kv: dict(kv)["run_id"]):
        run = find(dict(pair))
        if (run.run.run["run_kind"], run.run.run["experiment_id"]) != ("report", "stage3-joint-study-report"):
            continue
        j = run.json("payload/report.json")
        require(
            j["schema"] == "el-stage3-joint-study-report-1"
            and j["catalog_sha256"] == CAT
            and j["study_sha256"] == STUDY
            and j["method_sha256"] == marker["method_sha256"]
            and j["opening_sha256"] == marker["opening_sha256"],
            "joint study report is for a different registration or opening",
        )
        ep = j["public_numerical"]
        require(
            ep["path"] == "public-numerical/manifest.json"
            and sha(run.file("payload/" + ep["path"])) == ep["sha256"]
            and ep["sha256"] in public_hashes,
            "published combined numerical manifest differs",
        )
        jm = run.json("payload/" + ep["path"])
        require(
            jm["schema"] == "el-stage3-public-joint-manifest-1" and len(jm["files"]) > len(PUBLIC_TABLES),
            "missing combined development, test and engineering evidence",
        )
        jac = {row["path"]: row for row in jm["files"]}
        require(
            len(jac) == len(jm["files"]) and "confirmation-manifest.json" in jac,
            "missing original held-out manifest",
        )
        for n, s in jac.items():
            raw = run.file("payload/public-numerical/" + n)
            require(
                raw.stat().st_size == s["bytes"] and sha(raw) == s["sha256"] and s["sha256"] in public_hashes,
                "positive-listed joint report lacks a scientific member: " + n,
            )
        h = j["heldout"]
        require(
            h["member"] == "payload/evaluation.json"
            and run.parents().get(h["run_id"]) == h["manifest_sha256"]
            and sha(find(h).file(h["member"])) == h["sha256"],
            "held-out evaluation is not the report's original direct parent",
        )
        joints.append((run.pin, h, ep["sha256"], jac["confirmation-manifest.json"]["sha256"]))
    require(
        len(joints) == 1, "positive-listed publication must identify one original joint scientific report"
    )
    joint, heldout, joint_manifest, confirmation_manifest = joints[0]
    candidates = {frozenset((k, heldout[k]) for k in ("run_id", "manifest_sha256"))}
    for pair in sorted(candidates, key=lambda kv: dict(kv)["run_id"]):
        p = dict(pair)
        run = find(p)
        if "payload/evaluation.json" not in run.run.files:
            continue
        ev = run.json("payload/evaluation.json")
        if ev.get("schema") != "el-stage3-risk-evaluation-1" or ev.get("split") != "test":
            continue
        require(
            ev["method_sha256"] == marker["method_sha256"]
            and ev["opening_sha256"] == marker["opening_sha256"]
            and ev["catalog_sha256"] == CAT
            and ev["study_sha256"] == STUDY
            and ev["families"] == method["test_families"]
            and set(ev["methods"]) == set(METHODS)
            and ev["dispositions"]["planned_cells"] == 8
            and ev["dispositions"]["planned_target_views"] == 48
            and ev["dispositions"]["scheduled_scorer_modes"] == 144,
            "scientific report differs from complete signed held-out cohort",
        )
        for k in ("views", "families", "macros", "score-ablation", "reconstruction", "resources"):
            s = ev["files"][k]
            require(
                sha(run.file("payload/" + s["path"])) == s["sha256"],
                "incomplete original research table " + k,
            )
        ep = ev["public_numerical"]
        require(
            ep["path"] == "public-numerical/manifest.json"
            and sha(run.file("payload/" + ep["path"])) == ep["sha256"] == confirmation_manifest
            and ep["sha256"] in public_hashes,
            "sealed held-out numerical manifest missing in the publication list",
        )
        numeric = run.json("payload/" + ep["path"])
        require(
            numeric["schema"] == "el-stage3-public-numerical-manifest-1"
            and numeric["source_run_id"] == ev["run_id"]
            and {s["path"] for s in numeric["files"]} == PUBLIC_TABLES
            and len(numeric["files"]) == len(PUBLIC_TABLES),
            "public table set differs from its scientific registration",
        )
        for s in numeric["files"]:
            raw = run.file("payload/public-numerical/" + s["path"])
            require(
                sha(raw) == s["sha256"] and raw.stat().st_size == s["bytes"] and s["sha256"] in public_hashes,
                "positive-listed report lacks original held-out table " + s["path"],
            )
        summary = run.json("payload/public-numerical/study-summary.json")
        require(
            summary["schema"] == "el-stage3-public-numerical-study-1"
            and summary["split"] == "test"
            and summary["method_sha256"] == ev["method_sha256"]
            and summary["families"] == ev["families"]
            and summary["dispositions"] == ev["dispositions"]
            and summary["scientific_run_id"] == ev["run_id"],
            "scientific numerical summary no longer matches its source",
        )
        matched.append(
            {
                "run_id": p["run_id"],
                "manifest_sha256": p["manifest_sha256"],
                "evaluation_sha256": sha(run.file("payload/evaluation.json")),
                "dispositions": ev["dispositions"],
                "public_manifest_sha256": ep["sha256"],
                "joint_report": joint,
                "joint_public_manifest_sha256": joint_manifest,
                "summary_sha256": sha(run.file("payload/public-numerical/study-summary.json")),
                "table_sha256": {
                    k: ev["files"][k]["sha256"]
                    for k in ("views", "families", "macros", "score-ablation", "reconstruction", "resources")
                },
            }
        )
    require(
        len(matched) == 1,
        "public numerical allowlist must identify exactly one original blind held-out evaluation",
    )
    return matched[0]


def release_notes(summary, candidate):
    disposition = summary["dispositions"]

    def show(rec):
        return (
            "undefined" if rec["mean"] is None else format(rec["mean"], ".4f")
        ) + f" ({rec['defined_families']}/{rec['planned_families']})"

    axes = {(r["condition"], r["method"]): r for r in summary["family_equal_overall"] if r["scope"] == "all"}
    require(set(axes) == {(c, m) for c in CONS for m in METHODS}, "missing public four-method value")
    lines = [
        "# Evidence Lens " + TAG,
        "",
        f"The pre-test default is **{candidate['default_method']}**. The scientific confirmation returned {disposition['returned_target_views']} of {disposition['planned_target_views']} registered targets across four held-out families. Each target was rendered from separately fitted two- and four-image Gaussian reconstructions on its native 256-pixel grid.",
        "",
        "The table uses each reconstruction's full permitted source set. AP compares the four risk methods on the same finite, RGB-evaluable pixels. Low coverage counts RGB-evaluable pixels with a finite risk strictly below the frozen threshold, divided by reference-valid pixels. Low error is the fraction of low-risk pixels whose native three-channel mean absolute display-RGB error exceeds 0.05. Each entry gives the family-equal result and defined/scheduled families.",
        "",
        "| Reconstruction | Risk method | Four-method AP | Low coverage L/V | Low error LY/L |",
        "|---|---|---:|---:|---:|",
    ]
    for c in CONS:
        for m in METHODS:
            row = axes[c, m]
            lines.append(
                f"| {c} | {m} | {show(row['common_ap'])} | {show(row['low_coverage'])} | {show(row['low_error'])} |"
            )
    lines.extend(
        [
            "",
            "The numerical ZIP includes risk coverage, source-count interventions on fixed reconstructions, matched-pixel AP, failure dispositions and Gaussian optimization resources. The example ZIP contains the registered development-057 inputs, independent Blender reference and original expected maps. Each neural model includes all trained FP32 parameters.",
            "",
            "Check the three ZIPs against SHA256SUMS and extract the CPU and example archives into adjacent `cpu` and `example` directories. With Python 3.11 or 3.12, the default MS-L1 model needs only NumPy and Pillow. Install `requirements-light.txt` in a fresh environment to score a target. [Run the example and open its risk and state maps](https://github.com/wany614/evidence-lens/blob/v0.3.0-rc.1/docs/releases/v0.3.0-rc.1/README.md)",
            "",
            "For the 144-comparison replay on Linux or Windows, install `torch==2.8.0` from `https://download.pytorch.org/whl/cpu` first. On all three platforms, install `requirements.txt` and run the following from the CPU directory in that environment.",
            "",
            "```sh",
            f"python3 -I -B cpu.py replay --release-sha256 {candidate['assets']['cpu']['index_sha256']} --example ../example --example-sha256 {candidate['assets']['example']['index_sha256']} --output new-replay",
            "```",
            "",
            "On Windows use the active virtual environment's python.exe. Model computations receive the named predicted images, depth, alpha, source images and camera calibration. The verifier separately reads original RGB and reference labels.",
            "",
        ]
    )
    return "\n".join(lines).encode()


def models(out, method, original, repo, commit, store, shelf):
    from stage3_terminal_recovery import EXPERIMENT, fetch_failure, recovery_identity, source_equivalence
    from evidence_lens.stage3_decision import validate_fit_origin
    import hashlib
    from release_model import safe_text
    from multi_risk_pack import all_names
    from vision_release import copy, git_blob, sha
    from evidence_lens_multirisk.deploy import PACKAGE_MEMBERS
    from evidence_lens_multirisk.model import B_INDEX, B_WEIGHTS
    from evidence_lens_vision.contract import read

    models = {}
    provenance = {}
    for name in METHODS:
        r = method["models"][name]
        parent = original(r["index"])
        member_name = r["index"]["member"]
        p = parent.file(member_name)
        weight = parent.file(r["weights"]["member"])
        require(
            sha(p) == r["model_index_sha256"]
            and sha(weight) == r["weights_sha256"]
            and r["model_id"] == parent.pin["run_id"],
            "frozen original training identity differs",
        )
        d = read(p)
        names = LIGHT.get(name)
        prefix = member_name.rsplit("/", 1)[0] + "/"
        if name in ("MS-ft", "MS-frozen"):
            require(
                (parent.run.run["run_kind"], parent.run.run["experiment_id"])
                in (("model_fit", "stage3-risk-neural-fit"), ("derivation", EXPERIMENT))
                and d["schema"] == "el-multisource-risk-model-1"
                and d["method"] == name
                and d["phase"] == "formal"
                and d["seed"] == 1729
                and d["threshold"] == r["threshold"]
                and d["source_B_weights_sha256"] == B_WEIGHTS
                and d["source_B_index_sha256"] == B_INDEX
                and d["precision"] == "float32"
                and d["complete_backbone_included"] is True
                and d["tensor_count"] == 181
                and set(d["files"]) == PACKAGE_MEMBERS,
                "engineering or incomplete model in publication request",
            )
            origin = validate_fit_origin(parent.run, store, shelf)
            names = PACKAGE_MEMBERS | {"model.json"}
            fit = parent.json("payload/training.json")
            proof = parent.json("payload/independent-proof/verified-updates.json")
            require(
                proof["phase"] == fit["phase"] == "formal"
                and proof["run_id"] == d["model_id"] == r["model_id"]
                and proof["method"] == fit["method"] == name
                and proof["verified_tensors"] == 181
                and proof["checked_original_backbone"] == 175
                and proof["source_B_weights_sha256"] == B_WEIGHTS
                and proof["all_exported_tensor_hashes_match"] is True
                and min(proof["nonzero_task_gradient_minibatches"].values()) >= 3
                and proof["optimizer_steps"] == fit["optimizer_steps"] > 0
                and proof["model_index_sha256"] == fit["model_index_sha256"] == r["model_index_sha256"]
                and proof["weights_sha256"] == fit["weights_sha256"] == r["weights_sha256"]
                and fit["threshold"] == r["threshold"]
                and (
                    proof["backbone_changed_scalars"] > 0
                    if name == "MS-ft"
                    else proof["backbone_changed_scalars"] == 0
                ),
                "independent learning evidence differs from selected formal model",
            )
            recovered = parent.run.run["run_kind"] == "derivation"
            if recovered:
                rec = parent.json("payload/recovery.json")
                recovery_identity(parent.run.run, fit, d, rec)
                failed = fetch_failure(store, shelf)
                old = failed.open("inputs/worker/worker.zip")
                _, _, source_check = source_equivalence(old, parent.file("inputs/worker/worker.zip"))
                require(
                    source_check == rec["source_equivalence"]
                    and rec["terminal_readback"]["run_id"] in parent.parents()
                    and parent.parents()[rec["terminal_readback"]["run_id"]]
                    == rec["terminal_readback"]["manifest_sha256"]
                    and sha(failed.open("payload/fit/package/weights.safetensors")) == sha(weight)
                    and sha(failed.open("payload/fit/training-state.pt"))
                    == rec["terminal_checkpoint_sha256"],
                    "terminal optimizer, checkpoint and recovered weights differ",
                )
                certificate = rec["terminal_readback"]
                entry = original(certificate).file(certificate["member"])
                require(sha(entry) == certificate["sha256"], "invalid independent terminal readback")
                copy(out, "evidence/formal-worker-03.zip", old)
                copy(out, "evidence/" + name + "/terminal-optimization.json", entry)
                for suffix in (
                    "recovery.json",
                    "optimization-original/setup.json",
                    "optimization-original/broken-model.json",
                ):
                    path = parent.file("payload/" + suffix)
                    safe_text(suffix, path.read_bytes())
                    copy(out, "evidence/" + name + "/" + suffix, path)
            for suffix in PROOFS:
                proof_path = parent.file("payload/" + suffix)
                if suffix.endswith(".json"):
                    safe_text(suffix, proof_path.read_bytes())
                copy(out, "evidence/" + name + "/" + suffix, proof_path)
            frozen_worker = parent.file("inputs/worker/worker.zip")
            require(
                sha(frozen_worker) == d["worker_zip_sha256"] == FORMAL_WORKER,
                "stored training executable differs from the model index",
            )
            with zipfile.ZipFile(frozen_worker) as z:
                approved = all_names() | {"worker-source.json"}
                require(
                    z.namelist() == sorted(approved)
                    and all(i.external_attr >> 16 == 0o100644 for i in z.infolist()),
                    "formal worker has an unexpected member",
                )
                raw_index = z.read("worker-source.json")
                wi = json.loads(raw_index)
                require(
                    wi["schema"] == "el-stage3-neural-scientific-source-1"
                    and hashlib.sha256(raw_index).hexdigest() == FORMAL_WORKER_INDEX
                    and set(wi["files"]) == all_names()
                    and d["code_files"] == {k: s["sha256"] for k, s in wi["files"].items()},
                    "stored executable input and scientific source table differ",
                )
                for k, s in wi["files"].items():
                    data = z.read(k)
                    require(
                        s == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()},
                        "changed formal source " + k,
                    )
            published = out / "evidence" / "formal-worker.zip"
            if published.exists():
                require(sha(published) == d["worker_zip_sha256"], "selected models used different workers")
            else:
                copy(out, "evidence/formal-worker.zip", frozen_worker)
        elif name == "MS-L1":
            require(
                d["schema"] == "el-stage3-light-package-1"
                and d["status"] == "frozen"
                and d["model_id"] == r["model_id"]
                and d["threshold"] == r["threshold"],
                "unfrozen multi-source logistic regression",
            )
        else:
            require(
                d["schema"] == "el-l1-package-1"
                and d["status"] == "frozen"
                and d["model_id"] == r["model_id"],
                "wrong transferred B model",
            )
        commits = {}
        for n in sorted(names):
            f = parent.file(prefix + n)
            if n != "model.json":
                spec = d["files"][n]
                require(
                    sha(f) == (spec["sha256"] if isinstance(spec, dict) else spec),
                    "package member changed " + name + "/" + n,
                )
            if n.startswith("runtime/"):
                g = "src/" + n.removeprefix("runtime/")
                body = git_blob(repo, commit, g)
                require(
                    body == f.read_bytes()
                    and fit.get("export_source_sha256", fit["source_sha256"]).get(g) == sha(f),
                    "shipping runtime and training source differ: " + g,
                )
                commits[n] = {"source": g, "sha256": sha(f)}
            copy(out, "models/" + name + "/" + n, f)
        models[name] = {k: r[k] for k in ("model_id", "model_index_sha256", "weights_sha256", "threshold")}
        provenance[name] = {
            "source": parent.pin,
            "model_member": member_name,
            "weights_member": r["weights"]["member"],
            "shipping_commit": commit,
            "runtime": commits,
            **({"optimization": origin} if name in ("MS-ft", "MS-frozen") else {}),
        }
    require(len({r["weights_sha256"] for r in models.values()}) == 4, "duplicate model parameters")
    return models, provenance


def build(a):
    from vision_release import Original, archive_dir, copy, entries, git_blob, pin, sha, sig, write
    from evidence_lens.stage3_gate import require_opening, get_frozen_method
    from evidence_lens.scoring_io import new_output

    require(a.version == TAG and a.commit and a.repository.is_dir(), "unexpected release version")
    cached = {}

    def get(spec):
        key = spec["run_id"], spec["manifest_sha256"]
        if key not in cached:
            cached[key] = Original(a.store, a.shelf, spec)
        return cached[key]

    freeze, report, example = (get(pin(v)) for v in (a.method, a.report, a.example))
    marker = require_opening(a.store, CAT, shelf=a.shelf)
    method = get_frozen_method(a.store, marker, a.shelf)
    require(
        freeze.pin == marker["method"]
        and sha(freeze.file("payload/method-freeze.json")) == marker["method_sha256"]
        and method["run_id"] == freeze.pin["run_id"]
        and method["study_sha256"] == STUDY
        and method["feature_schema_sha256"] == FEATURES,
        "method does not identify the actual pretest opening",
    )
    opening = sha(a.store / "protocol/stage3" / CAT / "test-opening.json")
    report_receipt = scientific_report(report, method, {**marker, "opening_sha256": opening}, get)
    with new_output(a.output) as dest, tempfile.TemporaryDirectory(prefix="stage3-public-build-") as scratch:
        work = Path(scratch)
        cpu = work / "cpu"
        case = work / "example"
        numerical = work / "numerical"
        for location in (cpu, case, numerical):
            location.mkdir()
        report_table, report_sources = public_maps(numerical, report, "numerical", get)
        example_table, example_sources = public_maps(case, example, "example", get)
        enrolled, provenance = models(cpu, method, get, a.repository, a.commit, a.store, a.shelf)
        case_hash = check_example(case, example_table, freeze, method)
        prefix = "docs/releases/" + TAG + "/"
        for n in DOCS:
            write(cpu, n, git_blob(a.repository, a.commit, prefix + n))
        write(cpu, "cpu.py", git_blob(a.repository, a.commit, "scripts/stage3_public_cpu.py"))
        for name in ("requirements.txt", "requirements-light.txt"):
            write(cpu, name, git_blob(a.repository, a.commit, prefix + name))
        notices = add_notices(a.repository, a.commit, (cpu, case, numerical))
        require(not (notices.keys() & (example_table.keys() | report_table.keys())),
                "archived research member collides with a tagged license or notice")
        for k in (
            "catalog",
            "study",
            "feature_schema",
            "reconstruction_protocol",
            "production_budget",
            "learning_protocol",
            "neural_selection",
            "thresholds",
            "tensor_comparison",
            "cpu_validation",
        ):
            s = method["components"][k]
            p = get(s).file(s["member"])
            require(sha(p) == s["sha256"], "different recorded scientific component " + k)
            copy(cpu, "evidence/" + k + ".json", p)
            if k == "tensor_comparison":
                from evidence_lens_vision.contract import read

                proof = read(p)
                nested = [
                    proof["B_roster"],
                    *(r[x] for r in proof["fits"] for x in ("independent_readback", "scientific_readback")),
                ]
                require(
                    len(nested) == 9 and len({r["path"] for r in nested}) == 9,
                    "four independent scientific/tensor proof pairs required",
                )
                for r in nested:
                    data = get(s).file("payload/" + r["path"])
                    require(
                        sha(data) == r["sha256"] and data.stat().st_size == r["bytes"],
                        "incorrect scientific readback member",
                    )
                    copy(cpu, "evidence/tensor-comparison/" + r["path"], data)
        copy(cpu, "evidence/method-freeze.json", freeze.file("payload/method-freeze.json"))
        write(cpu, "evidence/model-sources.json", entries(provenance))
        shared = {
            "version": TAG,
            "repository": "https://github.com/wany614/evidence-lens",
            "commit": a.commit,
            "attribution": "wany614",
            "catalog_sha256": CAT,
            "study_sha256": STUDY,
            "feature_schema_sha256": FEATURES,
            "method_sha256": marker["method_sha256"],
            "method_source": freeze.pin,
            "opening_sha256": opening,
        }
        example_meta = {
            **shared,
            "schema": "el-stage3-public-example-1",
            "family": FAMILY,
            "case_sha256": case_hash,
            "source": example.pin,
            "allowlist_sha256": sha(example.file("payload/public-members.json")),
            "files": {**example_table, **notices},
        }
        write(case, "example.json", entries(example_meta))
        write(
            numerical,
            "numerical.json",
            entries(
                {
                    **shared,
                    "schema": "el-stage3-public-numerical-1",
                    "source": report.pin,
                    "heldout": report_receipt,
                    "files": {**report_table, **notices},
                    "allowlist_sha256": sha(report.file("payload/public-members.json")),
                }
            ),
        )
        files = {f.relative_to(cpu).as_posix(): sig(f) for f in sorted(cpu.rglob("*")) if f.is_file()}
        release = {
            **shared,
            "schema": "el-stage3-cpu-release-1",
            "default_method": method["default_method"],
            "models": enrolled,
            "sources": method["components"],
            "heldout": report_receipt,
            "expected_example_sha256": sha(case / "example.json"),
            "numerical_sha256": sha(numerical / "numerical.json"),
            "license": {
                "parameters_and_code": "Apache-2.0",
                "authored_media_and_records": "CC-BY-4.0",
                "upstream_weights": "DINOv2 Apache-2.0",
            },
            "files": files,
        }
        write(cpu, "release.json", entries(release))
        assets = {}
        for category, folder, index_name in (
            ("cpu", cpu, "release.json"),
            ("example", case, "example.json"),
            ("numerical", numerical, "numerical.json"),
        ):
            n = f"evidence-lens-{category}-{TAG}.zip"
            members = archive_dir(folder, dest / n)
            assets[category] = {
                "name": n,
                **sig(dest / n),
                "members": len(members),
                "index": index_name,
                "index_sha256": sha(folder / index_name),
            }
        summary = {
            **shared,
            "schema": "el-stage3-candidate-1",
            "default_method": method["default_method"],
            "assets": assets,
            "models": enrolled,
            "cells": report_receipt["dispositions"],
        }
        write(dest, "candidate.json", entries(summary))
        write(
            dest,
            "RELEASE_NOTES.md",
            release_notes(get(report_receipt).json("payload/public-numerical/study-summary.json"), summary),
        )
        write(dest, "SHA256SUMS", "".join(f"{v['sha256']}  {v['name']}\n" for v in assets.values()).encode())
        write(
            dest,
            "private-sources.json",
            entries(
                {
                    "candidate": summary,
                    "reports": report_sources,
                    "example": example_sources,
                    "models": provenance,
                    "originals": [{"pin": o.pin, "files": o.files} for _, o in sorted(cached.items())],
                }
            ),
        )
        if a.rebuild_of:
            for p in dest.iterdir():
                require(sig(p) == sig(a.rebuild_of / p.name), "independent rebuild differs: " + p.name)
    return summary


def isolated(a):
    import re

    require(
        re.fullmatch("[a-f0-9]{40}", a.commit)
        and subprocess.check_output(
            ["git", "-C", str(a.repository), "show", a.commit + ":scripts/stage3_release.py"]
        )
        == HERE.read_bytes(),
        "execute the builder pinned to the shipping source commit",
    )
    with tempfile.TemporaryDirectory(prefix="stage3-shipping-source-") as scratch:
        root = Path(scratch)
        archive = root / "source.zip"
        subprocess.run(
            [
                "git",
                "-C",
                str(a.repository),
                "archive",
                "--format=zip",
                "--output",
                str(archive),
                a.commit,
                "--",
                "scripts",
                "src",
            ],
            check=True,
        )
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                n = entry.filename
                if entry.is_dir() or not n.endswith(".py"):
                    continue
                require(
                    all(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", t) for t in n.split("/"))
                    and (entry.external_attr >> 16 & 0o170000) != 0o120000,
                    "unexpected scientific source",
                )
                dst = root / n
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(z.read(entry))
        boot = """import argparse,json,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path[:0]=[str(root/'scripts'),str(root/'src')]
import stage3_release as r
a=argparse.Namespace(**json.load(sys.stdin))
for k in ('store','shelf','repository','output','rebuild_of'):
    if getattr(a,k):setattr(a,k,Path(getattr(a,k)))
print(json.dumps(r.build(a),sort_keys=True))
"""
        info = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items()}
        subprocess.run(
            [sys.executable, "-I", "-B", "-c", boot, str(root)], input=json.dumps(info), text=True, check=True
        )


if __name__ == "__main__":
    if sys.version_info < (3, 11):
        raise SystemExit("Stage C releases require Python 3.11 or newer.")
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("repository", "store", "shelf", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("version", "commit", "method", "report", "example"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--rebuild-of", type=Path)
    isolated(p.parse_args())
