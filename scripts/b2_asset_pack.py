"""Create an offline Windows CUDA ticket from original sealed fixture and pinned CPU package."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo
from evidence_lens_bundle.common import encoded, sha, write, require

LAUNCH = """import argparse, hashlib, json, os, sys
from pathlib import Path
base=Path(__file__).resolve().parent
m=json.loads((base/"ticket.json").read_bytes())
for name,record in m["files"].items():
    p=base/name
    if p.stat().st_size!=record["bytes"] or hashlib.sha256(p.read_bytes()).hexdigest()!=record["sha256"]:
        raise ValueError("worker ticket member differs: "+name)
sys.path.insert(0,str(base/"runtime"))
from b2_asset_windows import execute
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get("EL_RUN_ID"): raise ValueError("EL_RUN_ID must be set by the recorder")
    args.out.mkdir(parents=True,exist_ok=True)
    if any(args.out.iterdir()): raise ValueError("output must be an empty recorder payload")
    execute(base/"fixture",base/"protocol.json",args.out)
"""


def create(a):
    assets = {}
    fixed = ZipFile(a.wheel)
    for n in fixed.namelist():
        if n.startswith(("evidence_lens/", "evidence_lens_bundle/")) or n.endswith(
            ("licenses/LICENSE", "licenses/NOTICE", "CC-BY-4.0.txt")
        ):
            assets["runtime/" + n] = fixed.read(n)
    for f in (a.source / "src/evidence_lens_asset").glob("*.py"):
        assets["runtime/evidence_lens_asset/" + f.name] = f.read_bytes()
    assets["runtime/b2_asset_windows.py"] = (a.source / "scripts/b2_asset_windows.py").read_bytes()
    for f in a.fixture.rglob("*"):
        if f.is_file():
            assets["fixture/" + f.relative_to(a.fixture).as_posix()] = f.read_bytes()
    assets["protocol.json"] = a.protocol.read_bytes()
    assets["run_gpu.py"] = LAUNCH.encode()
    assets["README.txt"] = (
        b"Run in the registered Windows CUDA gsplat 1.5.3 environment, under the recorder. No Internet or installation. python -I -B run_gpu.py --out EMPTY_RECORDER_PAYLOAD. Return complete original output, manifest and cost. This authored fixture contains no DTU photos or research target data.\n"
    )
    ticket = {
        "schema": "el-b2-asset-worker-ticket-1",
        "operation": "original Gaussian interchange readback",
        "cuda": "gsplat 1.5.3",
        "cpu_wheel_sha256": sha(a.wheel.read_bytes()),
        "protocol_sha256": sha(assets["protocol.json"]),
        "fixture_sha256": sha(assets["fixture/fixture.json"]),
        "files": {n: {"sha256": sha(v), "bytes": len(v)} for n, v in sorted(assets.items())},
    }
    assets["ticket.json"] = encoded(ticket)
    out = a.out / "evidence-lens-b2-asset-cuda.zip"
    with ZipFile(out, "w", compression=ZIP_DEFLATED) as z:
        for n, v in sorted(assets.items()):
            i = ZipInfo(n, (2000, 1, 1, 0, 0, 0))
            i.compress_type = ZIP_DEFLATED
            z.writestr(i, v)
    with ZipFile(out) as z:
        require(z.testzip() is None, "worker transport CRC failed")
    write(
        a.out / "dispatch.json",
        {
            "file": out.name,
            "sha256": sha(out.read_bytes()),
            "bytes": out.stat().st_size,
            "ticket_sha256": sha(assets["ticket.json"]),
            "fixture_sha256": ticket["fixture_sha256"],
            "protocol_sha256": ticket["protocol_sha256"],
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("source", "fixture", "wheel", "protocol", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    create(p.parse_args())
