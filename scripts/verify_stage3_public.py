"""Fetch the public multi-source release or grade an explicit local candidate in a clean CPU environment."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zipfile

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from verify_vision_public import REPO, SOURCES, download, fetch, install, installer_environment, opener, public_metadata
from vision_release import HASH, entries, read, require, safe_name, sha, sig

TAG = "v0.3.0-rc.1"
INDEX = {"cpu": "release.json", "example": "example.json", "numerical": "numerical.json"}
SUPPORT = {"candidate.json", "RELEASE_NOTES.md", "SHA256SUMS"}


def extract_checked(path, dest, description):
    dest.mkdir()
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = {safe_name(i.filename) for i in infos}
        require(len(infos) == len(names) == description["members"] and sum(i.file_size for i in infos) < 3 * 2**30,
                "Unexpected ZIP length or duplicate members")
        for i in infos:
            require(not i.is_dir() and i.external_attr >> 28 in (0, 8), "Only regular public files are allowed")
            out = dest/i.filename
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(i) as src, out.open("xb") as sink:
                shutil.copyfileobj(src, sink)
    index = dest/description["index"]
    require(sha(index) == description["index_sha256"], "Public index digest differs")
    data = read(index)
    require(names == set(data["files"]) | {index.name}, "Public index does not enumerate the entire archive")
    for name, row in data["files"].items():
        require(sig(dest/name) == {k: row[k] for k in ("bytes", "sha256")}, "Public member differs: " + name)
    return data


def inspect_source(op, tree, public, records):
    api = "https://api.github.com/repos/" + REPO + "/git/trees/" + tree + "?recursive=1"
    doc = json.loads(fetch(op, api, 16*2**20))
    require(doc["sha"] == tree and doc["truncated"] is False, "Incomplete public Git tree")
    blobs = {x["path"]: x for x in doc["tree"] if x["type"] == "blob"}
    pairs = [(public/"cpu.py", "scripts/stage3_public_cpu.py")]
    verification_scripts = (HERE.name, *SOURCES)
    for name in verification_scripts:
        pairs.append((HERE.parent/name, "scripts/"+name))
    release_dir = "docs/releases/" + TAG + "/"
    for n in ("README.md", "REBUILD.md", "MODEL_CARD.md", "requirements.txt", "requirements-light.txt"):
        pairs.append((public/n, release_dir+n))
    sources = read(public/"evidence/model-sources.json")
    for method, row in sources.items():
        for rel, v in row["runtime"].items():
            p = public/"models"/method/rel
            require(safe_name(v["source"]).startswith("src/") and sha(p) == v["sha256"],
                    "Model runtime differs from its research source")
            pairs.append((p, v["source"]))
    source_paths = {p for p, _ in pairs}
    executable_files = {p for p in (public/"models").rglob("*.py") if "runtime" in p.relative_to(public).parts}
    require(executable_files <= source_paths, "A shipped runtime module has no matching source in the tagged commit")
    for path, name in pairs:
        b = path.read_bytes()
        require(blobs[name]["mode"] in ("100644", "100755") and len(b) == blobs[name]["size"]
                and hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest() == blobs[name]["sha"],
                "Shipping runtime differs from public commit: " + name)
    records["tagged_runtime_files"] = len(pairs)
    records["verification_sources"] = {name:sig(HERE.parent/name) for name in verification_scripts}


def same_native_scores(light, full):
    """Compare same-host arrays after decompression, independent of ZIP metadata."""
    names = {"risk.npy", "state.npy", "valid.npy"}
    with zipfile.ZipFile(light) as left, zipfile.ZipFile(full) as right:
        require(set(left.namelist()) == set(right.namelist()) == names
                and len(left.namelist()) == len(right.namelist()) == 3, "Unexpected standalone score members")
        hashes = {}
        for name in sorted(names):
            value = left.read(name)
            require(value == right.read(name), "Standalone and complete CPU results differ: " + name)
            hashes[name] = hashlib.sha256(value).hexdigest()
    return hashes


def grade(args):
    require(HASH.fullmatch(args.candidate_sha256) and re.fullmatch(r"[a-f0-9]{40}", args.commit),
            "Expected an independent candidate SHA-256 and Git commit")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(not any(out.iterdir()), "Use an empty receipt directory")
    t = time.monotonic()
    remote = args.local is None
    audience = "anonymous_github" if remote else "local_candidate"
    record = {"schema": "el-stage3-external-verification-1", "mode": audience, "version": TAG,
              "commit": args.commit, "platform": {"system": platform.system(), "machine": platform.machine(),
                                                   "version": platform.release(), "python": platform.python_version()}}
    try:
        with tempfile.TemporaryDirectory(prefix="el-native-replay-") as scratch:
            root = Path(scratch)
            op = opener()
            if remote:
                release, urls, tree = public_metadata(op, REPO, TAG, args.commit)
                record["release_url"] = release["html_url"]
                record["public_git_tree"] = tree
                downloads = root/"downloads"
                downloads.mkdir()
                for name, r in sorted(urls.items()):
                    digest = r["digest"].removeprefix("sha256:")
                    require(HASH.fullmatch(digest), "Missing GitHub asset SHA-256")
                    if name == "candidate.json":
                        require(digest == args.candidate_sha256, "Public candidate identity differs")
                    require(name in SUPPORT or name in {f"evidence-lens-{k}-{TAG}.zip" for k in INDEX},
                            "Unexpected public attachment")
                    download(op, r, {"bytes": r["size"], "sha256": digest}, downloads)
                require(set(urls) == SUPPORT | {f"evidence-lens-{k}-{TAG}.zip" for k in INDEX}, "Incomplete public attachments")
            else:
                downloads = args.local.resolve()
            candidate = downloads/"candidate.json"
            require(sha(candidate) == args.candidate_sha256, "Wrong candidate")
            c = read(candidate)
            require(c["schema"] == "el-stage3-candidate-1" and c["repository"] == "https://github.com/"+REPO
                    and c["version"] == TAG and c["commit"] == args.commit and set(c["assets"]) == set(INDEX),
                    "Unrecognized multi-source candidate")
            items = {}
            metadata = {}
            for key, index in INDEX.items():
                a = c["assets"][key]
                n = f"evidence-lens-{key}-{TAG}.zip"
                require(a["name"] == n and a["index"] == index and sig(downloads/n) == {k:a[k] for k in ("bytes", "sha256")},
                        "Candidate and attachment differ: " + n)
                items[n] = a["sha256"]
                if key == "numerical":
                    continue
                metadata[key] = extract_checked(downloads/n, root/key, a)
                require(all(metadata[key][k] == c[k] for k in ("commit","version","study_sha256","method_sha256","opening_sha256")),
                        "Package detached from the held-out study")
            lines = (downloads/"SHA256SUMS").read_text(encoding="utf8").splitlines()
            require(len(lines) == len(items) and set(lines) == {h+"  "+n for n,h in items.items()}, "Unexpected checksum inventory")
            record["assets"] = {n:sig(downloads/n) for n in sorted(SUPPORT | set(items))}
            cpu = root/"cpu"
            if remote:
                inspect_source(op, tree, cpu, record)
            profile = root/"scoring-process"
            profile.mkdir()
            env = installer_environment(profile)
            env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            release_sha = c["assets"]["cpu"]["index_sha256"]
            ex_sha = c["assets"]["example"]["index_sha256"]
            require(metadata["cpu"]["expected_example_sha256"] == ex_sha
                    and metadata["cpu"]["numerical_sha256"] == c["assets"]["numerical"]["index_sha256"]
                    and metadata["cpu"]["models"] == c["models"], "Example or model package differs")
            # Supply the first process only a deployable case. Restore evaluation data from the tested ZIP afterward.
            deploy = profile/"case"
            shutil.copytree(root/"example"/"CASE"/"M4", deploy)
            prefix = "CASE/M4/"
            declared = {n.removeprefix(prefix) for n in metadata["example"]["files"] if n.startswith(prefix)}
            require({p.relative_to(deploy).as_posix() for p in deploy.rglob("*") if p.is_file()} == declared,
                    "Extra or missing deployment case member")
            hidden = root/"example"
            shutil.rmtree(hidden)  # This verifier created the temporary extraction above.
            case_sha = sha(deploy/"case.json")
            require(case_sha == metadata["example"]["case_sha256"]["M4"], "Example points to a different four-source case")
            program = profile/"light-scorer"
            program.mkdir()
            for rel in ("cpu.py", "release.json", "models/MS-frozen/model.json"):
                dst = program/rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cpu/rel, dst)
            for rel in ("models/MS-L1", "models/MS-frozen/runtime"):
                shutil.copytree(cpu/rel, program/rel)
            with tempfile.TemporaryDirectory(prefix="light-only-", dir=root) as scratch:
                light_root = Path(scratch)
                light = install(light_root, cpu/"requirements-light.txt", args.wheelhouse, pytorch_cpu=False)
                receipt = read(light_root/"installation.json")
                names = {n.lower() for n in receipt["installed_distributions"]}
                require({"pillow","numpy"} <= names <= {"pillow","numpy","pip","setuptools","wheel"},
                        "The lightweight scorer installation includes unexpected packages")
                shutil.copyfile(light_root/"installation.json", out/"installation-light.json")
                subprocess.run([str(light),"-I","-B",str(program/"cpu.py"),"run","--release-sha256",release_sha,
                                "--case",str(deploy),"--case-sha256",case_sha,"--view","v01","--scope","all",
                                "--output",str(out/"example")], env=env, cwd=profile, check=True)
                require(read(out/"example"/"score.json")["method"] == c["default_method"] == "MS-L1",
                        "The lightweight process did not run the frozen default")
            for name in ("risk.png", "state.png"):
                with (out/"example"/name).open("rb") as f:
                    header = f.read(24)
                require(header[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II",header[16:24]) == (256,256),
                        "Expected native risk or state visualization")
            example_asset = c["assets"]["example"]
            extract_checked(downloads/example_asset["name"], hidden, example_asset)
            numerical = c["assets"]["numerical"]
            report = extract_checked(downloads/numerical["name"], root/"numerical", numerical)
            require(all(report[k] == c[k] for k in ("commit","version","study_sha256","method_sha256","opening_sha256")),
                    "Numerical package detached from the held-out study")
            exe = install(root, cpu/"requirements.txt", args.wheelhouse)
            shutil.copyfile(root/"installation.json", out/"installation.json")
            cmd = [str(exe), "-I", "-B", str(cpu/"cpu.py")]
            subprocess.run([*cmd,"replay","--release-sha256",release_sha,"--example",str(hidden),"--example-sha256",ex_sha,
                            "--output",str(root/"scores")], env=env, cwd=profile, check=True)
            src = root/"scores"/"grade.json"
            proof = read(src)
            require(proof["status"] == "passed" and proof["scheduled_comparisons"] == 144
                    and proof["distinct_method_scope_views"] == 120 and len(proof["reference_evaluations"]) == 12
                    and math.isfinite(proof["max_abs_error"]) and 0 <= proof["max_abs_error"] <= 0.0001
                    and type(proof["threshold_crossings"]) is int and proof["threshold_crossings"] >= 0,
                    "Public native replay missed a required method, scope, view or independent evaluation")
            same = same_native_scores(out/"example/scores.npz", root/"scores/scores/MS-L1/M4/all/v01.npz")
            shutil.copyfile(src,out/"grade.json")
            record.update(status="passed", candidate_sha256=args.candidate_sha256, grade_sha256=sha(src),
                          default_method=c["default_method"], scheduled_comparisons=144,
                          distinct_method_scope_views=120, threshold_crossings=proof["threshold_crossings"],
                          max_abs_risk_difference=proof["max_abs_error"], example_sha256=ex_sha,
                          standalone_case_sha256=case_sha, installation_sha256=sha(out/"installation.json"),
                          standalone_matches_graded_arrays=same,
                          lightweight_installation_sha256=sha(out/"installation-light.json"))
    except Exception as e:
        record.update(status="failed", failure_type=type(e).__name__, failure_stage=str(e)[:220])
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic()-t
        (out/"verification.json").write_bytes(entries(record))
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-sha256", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--local", type=Path)
    p.add_argument("--wheelhouse", type=Path)
    ns = p.parse_args()
    print(json.dumps(grade(ns), sort_keys=True))
