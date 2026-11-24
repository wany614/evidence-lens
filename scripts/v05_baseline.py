"""Assemble the standalone finite-grid baseline from sealed recorder parents."""

import argparse
import os
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from dual_records import bind, resolve, selected, stage


TAG = "v0.5.0"
WEIGHTS = "20261124T041052425868Z-d308c134e69d4d4b"
WHEEL = "evidence_lens-0.5.0-py3-none-any.whl"
SOURCE = "evidence-lens-source-v0.5.0.zip"
BUNDLES = "v05-bell-bundles.zip"
OBSERVATIONS = "v05-bell-source-observations.zip"
REFERENCES = "v05-bell-independent-references.zip"
PROOF = "v05-acceptance-and-examples.zip"
MANIFEST = "BASELINE_MANIFEST.json"
PARENTS = {
    "start": "a-start", "protocol": "a-grid-registration", "geometry": "a-analytic-geometry",
    "gaussians": "a-upstream-gaussian-contract", "upstream_render": "a-recovery-render-agreement",
    "neural_mac": "a-neural-dispatch-mac", "build": "a-release-build",
    "tests": "a-immutable-source-tests", "mac": "a-mac-install", "linux": "a-linux-installed-default",
    "windows": "a-windows-installed-intake", "windows_vs_mac": "a-installed-differences-windows",
    "linux_vs_mac": "a-installed-differences-linux", "platforms": "a-platform-public-tables",
    "bundle_index": "a-bell-bundle-publication", "raw_sources": "a-bell-source-delivery",
    "scores": "a-bell-scores-sealed", "opening": "a-bell-reference-opening",
    "references": "a-bell-references", "comparison_program": "a-bell-evaluation-program",
    "gpu_intake": "a-bell-gpu-readback", "reference_delivery": "a-bell-reference-delivery",
    "upstream_summary": "a-upstream-report", "upstream_negative_controls": "a-upstream-wheel-tests",
}


def run_ledger(store):
    entries = []
    folders = [f for bucket in ("runs", "quarantine") for f in (store / bucket).iterdir()]
    for folder in sorted(folders, key=lambda f: f.name):
        if not (folder / "completion.json").is_file():
            continue
        info = load(folder / "run.json")
        trial = info["trial_id"]
        if not (trial.startswith("a-") or info.get("experiment_id") == "dual-track-a"
                or "dual-track-v1-A" in trial):
            continue
        complete = load(folder / "completion.json")
        files = load(folder / "manifest.json")["files"]
        entries.append(dict(run_id=folder.name, trial=trial, kind=info["run_kind"], bucket=folder.parent.name,
            status=complete["execution_status"], exit_code=complete["exit_code"], parents=info["parent_run_ids"],
            manifest_sha256=complete["manifest_sha256"], start_utc=info["started_at_utc"],
            sealed_utc=complete["sealed_at_utc"], worker_wall_seconds=info["elapsed_seconds_monotonic"],
            sampled_tree_peak_rss_bytes=info["sampled_peak_tree_rss_bytes"],
            sealed_file_count=len(files), sealed_file_bytes=sum(x["bytes"] for x in files),
            payload_bytes=sum(x["bytes"] for x in files if x["path"].startswith("payload/"))))
    return entries


def evidence(a):
    s = a.store
    if (s/"work/controller/a-baseline-evidence/selected.json").is_file():
        pin = selected(s, "a-baseline-evidence")
        resolve((s, s/"history"), pin)
        return pin
    pins = {role: selected(s, label) for role, label in PARENTS.items()}
    pins["comparison"] = selected(s, a.comparison_label)
    pins["report_example"] = bind(s, pins["mac"]["run_id"], "payload/profiles-vision/landscape/MS-ft/report.html")
    paths = [Path("docs/releases/v0.5.0"), Path("docs/verification/v0.5.0")]
    files = [f for d in paths for f in sorted(d.rglob("*")) if f.is_file()
             and f.suffix in (".md", ".csv", ".json") and f.name != MANIFEST]
    board = Path("examples/offset-bell-cage/v05-landscape-comparison.png")
    files += [board, Path("protocol/v05-grid-acceptance.json"),
              Path("requirements/v05-default.txt"), Path("requirements/v05-vision.txt")]
    files.extend(Path("scripts") / ("v05_" + suffix + ".py") for suffix in (
        "bell_evaluation_runtime", "bell_reference_delivery", "local_release", "platform_compare",
        "platform_records", "platform_report", "recovery_control", "recovery_gpu", "source_tests",
        "upstream_summary", "upstream_tests"))
    files.extend(Path("tests") / name for name in ("test_v05_asset_contract.py", "test_v05_platform_compare.py"))
    require(all(f.is_file() for f in files) and len({p.as_posix() for p in files}) == len(files)
            and all(Path(n) in files for n in [
                "docs/releases/v0.5.0/ACCEPTANCE.md", "docs/verification/v0.5.0/platforms.json",
                "docs/releases/v0.5.0/bell-comparison/README.md"]), "missing acceptance materials")
    extras = {"file" + str(i): f.resolve() for i, f in enumerate(files)}
    names = {k: v.relative_to(Path.cwd()).as_posix() for k, v in extras.items()}
    return stage(s, "a-baseline-evidence", "archive_verify", pins,
        {"files": names, "rights": "original code Apache-2.0; authored research material CC BY 4.0"},
        ["{python}", "scripts/v05_baseline.py", "evidence-worker", "--store", str(s),
         "--config", "{config}", "--out", "{out}", "--example", "{input:report_example}",
         "--inputs", *["{input:" + k + "}" for k in extras]], "payload/evidence.json", 20_000_000, extras)


def evidence_worker(a):
    config = load(a.config)
    parents = config["sources"]
    def times(key):
        pin = parents[key]
        folder = a.store / "runs" / pin["run_id"]
        seal = load(folder / "completion.json")
        require(seal["manifest_sha256"] == pin["manifest_sha256"] and seal["execution_status"] == "succeeded",
                "different recorded terminal state " + key)
        return load(folder / "run.json")["started_at_utc"], seal["sealed_at_utc"]
    order = {key: times(key) for key in ("comparison_program", "scores", "opening", "references", "comparison")}
    require(max(order["comparison_program"][1], order["scores"][1]) < order["opening"][0]
            and order["opening"][1] < order["references"][0] and order["references"][1] < order["comparison"][0],
            "evaluation program, score closure, access and evaluation are out of order")
    files = {}
    for i, path in enumerate(a.inputs):
        key = "file" + str(i)
        name = config["files"][key]
        require(sha(path) == config["extra_sha256"][key], "different public artifact " + name)
        if path.suffix != ".png":
            raw = path.read_text(encoding="utf-8")
            require(not any(t in raw for t in ("/Users/", "/Volumes/", "D:\\Evidence", "Bearer ")),
                    "private path or token marker in " + name)
        files[name] = path
    require(sha(a.example) == parents["report_example"]["sha256"]
            and not any(t in a.example.read_text() for t in ("/Users/", "/Volumes/", "file:///")),
            "authored HTML differs or contains a private path")
    files["examples/offset-bell-cage/v05-landscape-ms-ft.html"] = a.example
    ledger = run_ledger(a.store)
    require(ledger and any(r["status"] == "failed" for r in ledger), "failure history missing")
    disk_kib = int(subprocess.check_output(["du", "-sk", str(a.store / "runs")], text=True).split()[0])
    save(a.out / "operations.json", dict(schema="el-v05-operation-ledger-1",
        scope="Native and received stage A runs sealed before the evidence assembly. Bindings and received copies may overlap.",
        resources="Per-run manifest lengths include snapshots. du counts allocated filesystem blocks under runs, with hard links counted once.",
        local_runs_allocated_bytes_at_snapshot=disk_kib*1024, runs=ledger))
    files["docs/releases/v0.5.0/operations.json"] = a.out / "operations.json"
    save(a.out / "access-order.json", dict(schema="el-v05-reference-access-order-1",
        stages={k:dict(pin=parents[k],started_utc=v[0],sealed_utc=v[1]) for k,v in order.items()}))
    files["docs/releases/v0.5.0/access-order.json"] = a.out / "access-order.json"
    archive = a.out / PROOF
    with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=6) as z:
        for name, path in sorted(files.items()):
            z.write(path, name)
    save(a.out / "evidence.json", dict(schema="el-v05-public-evidence-1", run_id=os.environ["EL_RUN_ID"],
        parents=config["sources"], archive=PROOF, bytes=archive.stat().st_size, sha256=sha(archive),
        members={n: {"sha256":sha(f), "bytes":f.stat().st_size} for n,f in sorted(files.items())}))


def inventory(s):
    build = selected(s, "a-release-build")
    proof = selected(s, "a-baseline-evidence")
    def asset(parent, name):
        return bind(s, parent["run_id"], "payload/" + name)
    pins = dict(build=build, evidence=proof, weights=bind(s/"history", WEIGHTS, "payload/cpu/release.json"),
        wheel=asset(build,WHEEL), sources=asset(build,SOURCE), proof=asset(proof,PROOF),
        bundles=asset(selected(s,"a-bell-bundle-publication"),BUNDLES),
        observations=asset(selected(s,"a-bell-source-delivery"),OBSERVATIONS),
        references=asset(selected(s,"a-bell-reference-delivery"),REFERENCES))
    public = ["wheel", "sources", "proof", "bundles", "observations", "references"]
    return stage(s, "a-baseline", "archive_verify", pins, {"tag":TAG, "asset_roles":public},
        ["{python}", "scripts/v05_baseline.py", "inventory-worker", "--config", "{config}",
         "--build", "{input:build}", "--weights", "{input:weights}", "--proof", "{input:evidence}",
         "--out", "{out}", "--inputs", *["{input:" + k + "}" for k in public]], "payload/" + MANIFEST, 30_000)


def inventory_worker(a):
    config = load(a.config)
    build, model, proof = (load(x) for x in (a.build, a.weights, a.proof))
    require(build["version"] == "0.5.0" and model["version"] == "v0.3.0-rc.1", "release version mismatch")
    names = [WHEEL, SOURCE, PROOF, BUNDLES, OBSERVATIONS, REFERENCES]
    rights = ["Apache-2.0 with stated model and dependency notices", "per-file public source inventory",
              "original documentation and example CC BY 4.0; code Apache-2.0",
              "authored data CC BY 4.0", "authored data CC BY 4.0", "authored evaluation data CC BY 4.0"]
    assets = {}
    for role, name, path, license_ in zip(config["asset_roles"], names, a.inputs, rights, strict=True):
        pin = config["sources"][role]
        require(path.name == name and sha(path) == pin["sha256"], "asset identity differs: " + name)
        assets[name] = dict(role=role, bytes=path.stat().st_size, sha256=pin["sha256"], rights=license_, parent=pin)
    require(assets[PROOF]["sha256"] == proof["sha256"] and
            all(assets[n]["sha256"] == v["sha256"] for n,v in build["assets"].items()), "build drift")
    result = dict(schema="el-v05-baseline-1", tag=TAG, code_commit=build["code_commit"],
        source_tree=build["source_tree"], source_build_run=build["run_id"], baseline_run=os.environ["EL_RUN_ID"],
        repository="wany614/evidence-lens", assets=assets, verification=proof["parents"],
        risk_release=model["version"], weights_commit=model["commit"], default_method="MS-L1",
        model_weights=model["models"], model_release_sha256=sha(a.weights),
        feature_schema_sha256=model["feature_schema_sha256"], registered_preprocessing=proof["parents"]["protocol"],
        preprocessing=proof["members"]["protocol/v05-grid-acceptance.json"],
        reference_scope="Previously exposed authored bell, six views, engineering description.",
        stage_a_risk_fits=0, stage_a_threshold_fits=0,
        historical={"T1_scored":240,"T1_registered":288,"T1_missing":48,"C6_M2_tau":None,"C6_M4_tau":None},
        publication="Read the independent GitHub release and anonymous download receipt for distribution status.")
    save(a.out / MANIFEST, result)
    sums={k:v["sha256"] for k,v in assets.items()}
    sums[MANIFEST]=sha(a.out / MANIFEST)
    (a.out/"SHA256SUMS").write_text("".join(sums[n] + "  " + n + "\n" for n in sorted(sums)), encoding="ascii")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["evidence","evidence-worker","inventory","inventory-worker"])
    for name in ("store","config","out","build","weights","proof","example"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--inputs",nargs="+",type=Path)
    p.add_argument("--comparison-label",default="a-bell-landscape-comparison-packaged")
    a=p.parse_args()
    if a.action=="evidence": print(evidence(a))
    elif a.action=="inventory": print(inventory(a.store))
    elif a.action=="evidence-worker": evidence_worker(a)
    else: inventory_worker(a)
