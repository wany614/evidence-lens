"""Create the release inventory from the sealed source build and redistributable example packages."""
import argparse
import json
import os
from pathlib import Path
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require


def make(build_path, gold, trajectories, out):
    build=load(build_path)
    assets={n:{**e,"role":"wheel" if n.endswith(".whl") else "sources"} for n,e in build["assets"].items()}
    import zipfile
    with zipfile.ZipFile(gold) as z:
        import hashlib
        expected=hashlib.sha256(z.read("expected.json")).hexdigest()
    require(expected=="7aed41037e7c4ec7f0a8583d33bdbe5291d6b02bf0ef8df325a13d337490f9cc","unexpected original replay oracle")
    for path,role in [(gold,"original_replays"),(trajectories,"static_trajectories")]:
        assets[path.name]={"role":role,"bytes":path.stat().st_size,"sha256":sha(path)}
    require(len(assets)==4 and all(sha(build_path.parent/n)==q["sha256"] for n,q in build["assets"].items()),
            "source build output differs")
    manifest={"schema":"el-general-release-1","tag":"v"+build["version"],"code_commit":build["code_commit"],
              "source_build_run":build["run_id"],"inventory_run":os.environ["EL_RUN_ID"],
              "model_release":"v0.3.0-rc.1","risk_weights_retrained":False,"original_gold_sha256":expected,
              "cpu_release_index_sha256":"150b04e702f5386b5808ac17545cd073e3d4d837ea789e6ae167168ac086084d",
              "assets":assets}
    save(out/"release-manifest.json",manifest)
    sums={**{n:v["sha256"] for n,v in assets.items()},"release-manifest.json":sha(out/"release-manifest.json")}
    (out/"SHA256SUMS").write_text("".join(v+"  "+n+"\n" for n,v in sorted(sums.items())),encoding="ascii")
    return manifest


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("build","gold","trajectories","out"):
        p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();print(json.dumps(make(a.build,a.gold,a.trajectories,a.out),sort_keys=True))
