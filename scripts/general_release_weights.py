"""Download and authenticate the original four-model CPU distribution using a public release URL."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile


NAME = "evidence-lens-cpu-v0.3.0-rc.1.zip"
URL = "https://github.com/wany614/evidence-lens/releases/download/v0.3.0-rc.1/" + NAME
BYTES = 164854827
ZIP_SHA = "f555956840870309217cc85fe9b339ef53762679bbd2138b7f149d8ea451a7b7"
INDEX_SHA = "150b04e702f5386b5808ac17545cd073e3d4d837ea789e6ae167168ac086084d"


def check(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(out):
    out.mkdir(parents=True, exist_ok=True)
    check(not any(out.iterdir()), "choose an empty output directory")
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        scratch = Path(tmp)
        archive = scratch / NAME
        request = Request(URL, headers={"User-Agent":"Evidence-Lens-anonymous-model-replay"})
        with urlopen(request, timeout=100) as response, archive.open("xb") as f:
            status = response.status
            check(status == 200, "public download returned HTTP " + str(status))
            shutil.copyfileobj(response, f)
        check(archive.stat().st_size == BYTES and digest(archive) == ZIP_SHA, "public CPU archive differs")
        download_s = time.monotonic() - t0
        with ZipFile(archive) as package:
            names = package.namelist()
            index_name, = (n for n in names if n in ("release.json","cpu/release.json"))
            prefix = index_name.removesuffix("release.json")
            raw = package.read(index_name)
            check(hashlib.sha256(raw).hexdigest() == INDEX_SHA, "original release index differs")
            meta = json.loads(raw)
            check(set(meta["models"]) == {"MS-L1","MS-ft","MS-frozen","B-L1-transfer"}, "incomplete model set")
            permits = {**meta["files"], "release.json":{"bytes":len(raw),"sha256":INDEX_SHA}}
            check(len(names) == len(set(names)) and set(names) == {prefix+n for n in permits},
                  "unlisted or missing CPU archive member")
            target = scratch / "cpu"
            for name,spec in permits.items():
                parts = name.split("/")
                check(all(part not in ("",".","..") and "\\" not in part and ":" not in part for part in parts),
                      "nonportable member path")
                member = target.joinpath(*parts)
                member.parent.mkdir(parents=True, exist_ok=True)
                with package.open(prefix + name) as source, member.open("xb") as dest:
                    shutil.copyfileobj(source, dest)
                check(member.stat().st_size == spec["bytes"] and digest(member) == spec["sha256"],
                      "member integrity mismatch: " + name)
        target.rename(out / "cpu")
    receipt = {"schema":"el-general-public-weights-1","url":URL,"http_status":status,"anonymous":True,
               "sha256":ZIP_SHA,"bytes":BYTES,"release_sha256":INDEX_SHA,"verified_members":len(permits),
               "models":meta["models"],"download_and_archive_hash_seconds":download_s,
               "total_seconds":time.monotonic()-t0,"recorder_run":os.getenv("EL_RUN_ID")}
    (out / "download.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(fetch(parser.parse_args().out),sort_keys=True))
