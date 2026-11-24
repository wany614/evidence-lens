"""Build the public general wheel and source archive from one allowlisted, immutable Git tree."""
import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import tomllib
import zipfile

from bundle_build import run
from source_candidate import audit, git


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def inventory(repo, commit):
    check(bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "supply an exact Git commit")
    check(git(repo, "rev-parse", commit + "^{commit}").decode().strip() == commit, "commit unavailable")
    entries = {}
    for raw in git(repo, "ls-tree", "-rz", commit).split(b"\0"):
        if raw:
            head, name = raw.split(b"\t", 1)
            mode, kind, oid = head.split()
            check(mode in (b"100644", b"100755") and kind == b"blob", "unsupported source tree entry")
            entries[name.decode("utf-8")] = int(mode, 8), oid.decode("ascii")
    policy, blobs = audit(repo, entries)
    check(all(policy["license_status"][x]["public_export"] is True for x in set(policy["files"].values())),
          "public rights missing")
    return entries, blobs


def check_wheel(wheel, expected, meta, files):
    from email.parser import BytesParser
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        check(len({n.casefold() for n in names}) == len(names), "duplicate wheel member")
        info = "evidence_lens-" + meta["version"] + ".dist-info/"
        metadata = {"METADATA", "RECORD", "WHEEL", "top_level.txt", "entry_points.txt"}
        licenses = {"licenses/" + n for n in meta["license-files"]}
        check(set(names) == set(expected) | {info + n for n in metadata | licenses},
              "wheel differs from the explicit package and notices inventory")
        current = BytesParser().parsebytes(z.read(info+"METADATA"))
        check(current["Name"] == meta["name"] and current["Version"] == meta["version"] and
              current["License-Expression"] == meta["license"] and set(current.get_all("License-File")) == set(meta["license-files"]),
              "wheel project metadata differs from the pinned project")
        for n in meta["license-files"]:
            check(z.read(info+"licenses/"+n) == files[n], "different wheel notice " + n)
        check(z.read(info+"entry_points.txt").decode().strip().splitlines() ==
              ["[console_scripts]", "evidence-lens = evidence_lens_bundle.cli:main"], "different CLI")
        rows = list(csv.reader(io.StringIO(z.read(info + "RECORD").decode())))
        check(len(rows) == len(names) and {r[0] for r in rows} == set(names), "incomplete wheel RECORD")
        for name, code, size in rows:
            raw = z.read(name)
            if name == info+"RECORD":
                check(code == size == "", "self-referential wheel digest")
            else:
                hash64=base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
                check(code == "sha256="+hash64 and size == str(len(raw)), "wheel RECORD mismatch " + name)
            if name in expected:
                check(raw == expected[name], "packaged code or model differs from exact commit " + name)
    return len(names)


def build(repo, commit, out):
    entries, files = inventory(repo, commit)
    meta = tomllib.loads(files["pyproject.toml"].decode())["project"]
    version = meta["version"]
    source_zip = out / ("evidence-lens-source-v" + version + ".zip")
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip, "x") as z:
        z.comment = ("source commit " + commit).encode("ascii")
        for name, raw in sorted(files.items()):
            zi = zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0))
            zi.create_system = 3
            zi.external_attr = entries[name][0] << 16
            z.writestr(zi, raw, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    expected = {key[4:]: raw for key,raw in files.items() if key.startswith("src/")}
    check(expected and any(n.startswith("evidence_lens_bundle/assets/MS-L1/") for n in expected),
          "bundled linear model missing")
    with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
        project = Path(scratch) / "project"
        keys = ["pyproject.toml", "README.md", *meta["license-files"], *("src/"+s for s in expected)]
        for name in keys:
            dest = project.joinpath(*PurePosixPath(name).parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(files[name])
        run([shutil.which("uv"), "build", "--offline", "--wheel", "--out-dir", out, project])
    wheel, = out.glob("*.whl")
    check(wheel.name == "evidence_lens-"+version+"-py3-none-any.whl", "unexpected wheel tag")
    n = check_wheel(wheel, expected, meta, files)
    report = {"schema":"el-general-build-1", "run_id":os.getenv("EL_RUN_ID"), "code_commit":commit,
              "source_tree":git(repo,"rev-parse",commit+"^{tree}").decode().strip(), "version":version,
              "risk_weights_release":"v0.3.0-rc.1", "source_files":len(entries), "wheel_record_members":n,
              "default_dependencies":meta["dependencies"], "source_date_epoch":946684800,
              "assets":{p.name:{"bytes":p.stat().st_size,"sha256":digest(p.read_bytes())} for p in [source_zip,wheel]}}
    (out/"build.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n", encoding="utf-8")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--commit",required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(build(a.repo,a.commit,a.out),sort_keys=True))
