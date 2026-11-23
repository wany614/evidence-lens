"""Export the allowed Windows worker source, one PNG and cameras from a sealed reference."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from evidence_lens._verify import verify_bundle


def build(bundle, out):
    root = Path(__file__).resolve().parents[1]
    checked = verify_bundle(bundle)
    run = checked["run"]
    if run["execution_status"] != "succeeded" or run["run_kind"] != "scene_render":
        raise ValueError("handoff needs an accepted reference run")
    ref = json.loads((bundle/"payload/reference.json").read_text())
    if ref["validation_passed"] is not True:
        raise ValueError("reference did not pass its numerical checks")
    files = {n: (bundle/"payload"/n).read_bytes() for n in ("input.png", "cameras.json")}
    if tuple(hashlib.sha256(files[n]).hexdigest() for n in ("input.png","cameras.json")) != (
            ref["input_sha256"],ref["camera_sha256"]):
        raise ValueError("input or camera pin differs from the accepted reference")
    sources = json.loads((root/"configs/windows-worker-files.json").read_text())["files"]
    for name in sources:
        path = root/name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"missing worker source: {name}")
        files["code/"+name] = path.read_bytes()
    job = {"schema":"el-worker-handoff-1", "experiment_id":"p2-single-image", "trial_id":run["trial_id"],
           "reference":{"run_id":run["run_id"],"manifest_sha256":checked["manifest_sha256"]},
           "protocol":"1.0", "condition":"B0", "files":{n:hashlib.sha256(d).hexdigest() for n,d in files.items()}}
    files["handoff.json"] = (json.dumps(job,indent=2,sort_keys=True)+"\n").encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out,"x",compression=zipfile.ZIP_DEFLATED) as archive:
        for n,d in sorted(files.items()):
            archive.writestr(n,d)
    print(json.dumps({"name":out.name,"bytes":out.stat().st_size,"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
                      "reference":job["reference"],"files":len(files)}))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    build(a.reference,a.out)
