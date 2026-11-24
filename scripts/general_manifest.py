"""Create the release inventory from the sealed source build and redistributable example packages."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require


MEASUREMENT_TABLES = {
    "g3-configurations.csv", "g3-platform-backends.csv", "g3-cold-starts.csv",
    "g3-thread-candidates.csv", "g3-thread-candidate-frames.csv", "g3-frames.csv",
    "g4-offline-parity.csv", "g4-offline-frames.csv", "g4-loads.csv",
    "g4-version-events.csv", "g4-version-checks.csv", "g4-frames.csv",
    "process-resource-samples.csv",
}


def make(build_path, gold, trajectories, measurements, measurement_index, weights_receipt, out):
    build=load(build_path)
    assets={n:{**e,"role":"wheel" if n.endswith(".whl") else "sources"} for n,e in build["assets"].items()}
    immutable={"original_replays":(gold,"0ed367c24019004d5126c823d899db3941f0fe9dd24473394c3cd6b581c7cfa0"),
               "static_trajectories":(trajectories,"933d16cd27549618bc7adf43a1a86988daaf187fc06282a9d21775ac9a37b806")}
    digests={role:sha(path) for role,(path,_) in immutable.items()}
    require(all(digests[role]==expected for role,(_,expected) in immutable.items()),"wrong registered original asset")
    import zipfile
    with zipfile.ZipFile(gold) as z:
        expected=hashlib.sha256(z.read("expected.json")).hexdigest()
    require(expected=="7aed41037e7c4ec7f0a8583d33bdbe5291d6b02bf0ef8df325a13d337490f9cc","unexpected original replay oracle")
    for path,role in [(gold,"original_replays"),(trajectories,"static_trajectories")]:
        assets[path.name]={"role":role,"bytes":path.stat().st_size,"sha256":digests[role]}
    measured=load(measurement_index)
    require(measurements.name==measured["file"]=="evidence-lens-cpu-replay-measurements.zip"
            and sha(measurements)==measured["sha256"] and measurements.stat().st_size==measured["bytes"],
            "different measurement file or inventory")
    with zipfile.ZipFile(measurements) as z:
        tables=json.loads(z.read("tables.json"))
        expected_members=MEASUREMENT_TABLES | {"tables.json","replay-example.html","replay-example-arrays.npz"}
        require(tables["schema"]=="el-static-engineering-measurements-1"
                and set(tables["sources"])=={"mac","windows"}
                and all(set(row)=={"g3","g4"} for row in tables["sources"].values())
                and all(re.fullmatch("[0-9a-f]{64}", row.get("manifest_sha256") or "")
                        and re.fullmatch("[0-9a-f]{64}", row.get("receipt_sha256") or "")
                        for phases in tables["sources"].values() for row in phases.values())
                and set(tables["tables"])==MEASUREMENT_TABLES
                and len(z.namelist())==len(expected_members) and set(z.namelist())==expected_members,
                "measurement archive lacks a platform receipt or contains additional material")
        for name, item in tables["tables"].items():
            raw = z.read(name)
            require(item["bytes"] == len(raw) and item["sha256"] == hashlib.sha256(raw).hexdigest(),
                    "different public timing table " + name)
        for name, key in (("replay-example.html", "html_sha256"), ("replay-example-arrays.npz", "arrays_sha256")):
            require(hashlib.sha256(z.read(name)).hexdigest() == tables["html_example"][key],
                    "different original frame output " + name)
    assets[measurements.name]={"role":"measurements","bytes":measured["bytes"],"sha256":measured["sha256"]}
    require(len(assets)==5 and all(sha(build_path.parent/n)==q["sha256"] for n,q in build["assets"].items()),
            "source build output differs")
    weights=load(weights_receipt)
    require(weights["anonymous"] and weights["http_status"]==200 and weights["verified_members"]==116
            and weights["sha256"]=="f555956840870309217cc85fe9b339ef53762679bbd2138b7f149d8ea451a7b7"
            and weights["release_sha256"]=="150b04e702f5386b5808ac17545cd073e3d4d837ea789e6ae167168ac086084d",
            "original weights download or verified model index differs")
    manifest={"schema":"el-general-release-1","tag":"v"+build["version"],"code_commit":build["code_commit"],
              "source_build_run":build["run_id"],"inventory_run":os.environ["EL_RUN_ID"],
              "model_release":"v0.3.0-rc.1","risk_weights_retrained":False,"original_gold_sha256":expected,
              "cpu_release_index_sha256":weights["release_sha256"],"weights_download_run":weights["recorder_run"],
              "assets":assets}
    save(out/"release-manifest.json",manifest)
    sums={**{n:v["sha256"] for n,v in assets.items()},"release-manifest.json":sha(out/"release-manifest.json")}
    (out/"SHA256SUMS").write_text("".join(v+"  "+n+"\n" for n,v in sorted(sums.items())),encoding="ascii")
    return manifest


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("build","gold","trajectories","measurements","measurement-index","weights-receipt","out"):
        p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();print(json.dumps(make(a.build,a.gold,a.trajectories,a.measurements,a.measurement_index,a.weights_receipt,a.out),sort_keys=True))
