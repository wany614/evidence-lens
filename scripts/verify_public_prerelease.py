"""Download a tagged Evidence Lens L1 release anonymously and run the shipped CPU grader."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener


def require(ok, message):
    if not ok:
        raise ValueError(message)


def anonymous_get(opener, url):
    request = Request(url, headers={"User-Agent": "Evidence-Lens-public-verifier"})
    with opener.open(request, timeout=90) as response:
        return response.read()


def verify(args):
    require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9-]*/[a-zA-Z0-9_.-]+", args.repo), "Invalid repository")
    require(re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+-rc\.[0-9]+", args.tag), "Invalid prerelease tag")
    require(re.fullmatch(r"[a-f0-9]{40}", args.commit) and re.fullmatch(r"[a-f0-9]{64}", args.sha256),
            "Expected a full Git commit and an independent SHA-256")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(not any(out.iterdir()), "Choose an empty output directory")
    # urllib does not read a .netrc or cookie jar. Disable environment proxies and auth handlers.
    opener = build_opener(ProxyHandler({}), HTTPSHandler())
    api = "https://api.github.com/repos/" + args.repo
    started = time.monotonic()
    repo = json.loads(anonymous_get(opener, api))
    release = json.loads(anonymous_get(opener, api + "/releases/tags/" + args.tag))
    commit = json.loads(anonymous_get(opener, api + "/commits/" + args.tag))["sha"]
    require(repo["full_name"] == args.repo and repo["private"] is False, "Repository is not publicly readable")
    require(release["tag_name"] == args.tag and release["prerelease"] and not release["draft"]
            and commit == args.commit, "Public tag, commit or prerelease status differs")
    name = "evidence-lens-" + args.tag + "-cpu.zip"
    assets = {x["name"]: x for x in release["assets"]}
    asset = assets[name]
    base = "https://github.com/" + args.repo + "/releases/download/" + args.tag + "/"
    require(asset["browser_download_url"] == base + name and
            assets[name + ".sha256"]["browser_download_url"] == base + name + ".sha256",
            "Unexpected public asset URL")
    checksum = anonymous_get(opener, base + name + ".sha256").decode().split()
    require(checksum == [args.sha256, name], "Public sidecar differs from the independent expected digest")
    payload = anonymous_get(opener, base + name)
    digest = hashlib.sha256(payload).hexdigest()
    require(digest == args.sha256 and len(payload) == asset["size"], "Downloaded asset identity differs")
    if asset.get("digest"):
        require(asset["digest"] == "sha256:" + digest, "GitHub asset digest differs")
    package = out / name
    package.write_bytes(payload)
    seconds = time.monotonic() - started
    with tempfile.TemporaryDirectory() as temp:
        with zipfile.ZipFile(package) as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)), "Repeated ZIP member")
            for item in archive.infolist():
                parts = PurePosixPath(item.filename)
                require(item.external_attr >> 16 == 0o100644 and not parts.is_absolute() and
                        parts.as_posix() == item.filename and ".." not in parts.parts and
                        "\\" not in item.filename and ":" not in item.filename, "Unexpected ZIP member")
            release_bytes = archive.read("release.json")
            identity = json.loads(release_bytes)
            require(identity["code_commit"] == commit and identity["version"] == args.tag, "Internal release identity")
            archive.extractall(temp)
        command = [sys.executable, "-I", str(Path(temp) / "smoke.py"), "--bundle", temp, "--output", str(out / "cpu"),
                   "--archive", str(package), "--archive-sha256", digest,
                   "--release-sha256", hashlib.sha256(release_bytes).hexdigest()]
        if args.wheelhouse:
            command += ["--wheelhouse", str(args.wheelhouse.resolve())]
        subprocess.run(command, check=True, cwd=temp)
    smoke_bytes = (out / "cpu/smoke.json").read_bytes()
    smoke = json.loads(smoke_bytes)
    require(smoke["passed"] and smoke["verified_zip_sha256"] == digest and
            smoke["model_id"] == identity["model_id"], "CPU verification did not return the pinned model")
    receipt = dict(schema="el-public-download-verification-1", url=release["html_url"], release_id=release["id"],
                   asset_url=asset["browser_download_url"], asset_id=asset["id"], repository=args.repo, tag=args.tag,
                   commit=commit, sha256=digest, bytes=len(payload), members=len(names), model_id=identity["model_id"],
                   github_published_utc=release["published_at"], verified_utc=datetime.now(timezone.utc).isoformat(),
                   authentication="anonymous HTTPS without cookies, credentials or configured proxy",
                   download_and_metadata_seconds=seconds, smoke_sha256=hashlib.sha256(smoke_bytes).hexdigest(),
                   grade_seconds=smoke["grade_seconds"], install_seconds=smoke["install_seconds"],
                   max_score_difference=smoke["max_absolute_score_difference"], views=smoke["views"], methods=smoke["methods"],
                   finite_masks_equal=smoke["finite_masks_equal"], state_arrays_equal=smoke["state_arrays_equal"],
                   l0_reason_arrays_equal=smoke["l0_reason_arrays_equal"], map_count=len(smoke["map_pngs"]))
    (out / "public-verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="wany614/evidence-lens")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, help="Use compatible local dependency wheels during installation")
    verify(parser.parse_args())
