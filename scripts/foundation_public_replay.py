"""Run the existing offline browser acceptance on a fresh extraction of the anonymous exhibit ZIP."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
from zipfile import ZipFile

from dual_records import bind, selected, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from foundation_release import VIEWER


def verify(a):
    download = load(a.receipt)
    item = next(x for x in download["downloads"] if x["name"] == VIEWER)
    require(item["anonymous"] and sha(a.asset) == item["sha256"], "different public exhibit")
    with tempfile.TemporaryDirectory() as temp, ZipFile(a.asset) as archive:
        root = Path(temp)
        rows = [line.split("  ", 1) for line in archive.read("CHECKSUMS.txt").decode("ascii").splitlines()]
        names = set(archive.namelist())
        require(len(names) == len(archive.namelist()) == len(rows) + 1 and
                names == {name for _, name in rows} | {"CHECKSUMS.txt"}, "ZIP and checksum list differ")
        for expected, name in rows:
            path = PurePosixPath(name)
            require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == name,
                    "nonportable archive member")
            blob = archive.read(name)
            require(hashlib.sha256(blob).hexdigest() == expected, "changed archive member")
            target = root.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
        result = json.loads(subprocess.check_output(
            [a.node, a.driver, a.chrome, root / "index.html", root / "browser-output"], text=True, timeout=175))
    require(result["external_requests"] == 0 and not result["script_errors"] and
            len(result["verified_conditions"]) == 29 and len(result["independent_references_opened"]) == 7,
            "browser did not reach the complete public exhibit")
    save(a.out / "offline-replay.json", dict(schema="el-foundation-public-offline-replay-1",
         run_id=os.environ["EL_RUN_ID"], public_code_commit=download["code_commit"], archive=item,
         archive_files_checked=len(rows), repeated_conditions=29, evaluated_views=7,
         network_requests=result["external_requests"], script_errors=result["script_errors"],
         evaluated=result["independent_references_opened"], mobile_view=result["mobile"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("store", "node", "chrome", "out", "receipt", "asset", "driver"):
        p.add_argument("--" + key, type=Path)
    a = p.parse_args()
    if a.store:
        prior = selected(a.store, "c-anonymous-readback")
        print(stage(a.store, "c-public-offline-exhibit", "archive_verify", {"download":prior,
            "asset":bind(a.store, prior["run_id"], "payload/" + VIEWER)}, {"mode":"fresh offline extraction"},
            ["{python}", "scripts/foundation_public_replay.py", "--receipt", "{input:download}",
             "--asset", "{input:asset}", "--out", "{out}", "--driver", "{input:driver}",
             "--node", str(a.node), "--chrome", str(a.chrome)], "payload/offline-replay.json", 6000,
            extra_files={"driver": Path("scripts/foundation_viewer_smoke.mjs").resolve()}))
    else:
        verify(a)
