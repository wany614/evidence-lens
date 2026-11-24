"""Export a positive-listed offline exhibit and publish its separately versioned research materials."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from dual_records import bind, resolve, selected, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from many_distribute import anonymous, release


TAG = "humanities-foundation-v1"
MANIFEST = "C_MANIFEST.json"
VIEWER = "evidence-lens-foundation-viewer-v1.zip"
PHOTOS = "evidence-lens-foundation-source-photos-v1.zip"
TITLE = "Evidence Lens humanities foundation · edition 1"
EXAMPLE = "examples/evidence-foundation/"
DOCS = "docs/humanities-foundation/"
STAGES = {"registration": "c-foundation-registration", "editions": "c-primary-editions",
          "cases": "c-foundation-case-data", "viewer": "c-foundation-viewer",
          "case_readback": "c-foundation-case-readback",
          "baseline": "a-baseline", "research": "b-research-release", "public_B": "b-anonymous-readback"}
REQUIRED_DOCS = ("README.md", "MODEL_AND_DATA_BASIS.md", "CONCEPTS_AND_CLAIMS.md", "THEORY_SOURCES.md",
                 "OPEN_QUESTIONS.md", "CONTRIBUTIONS.md", "bibliography.bib", "primary-editions.json",
                 "registration.json", "case-data-check.json")


def git_blob(repo, commit, name):
    return subprocess.check_output(["git", "-C", str(repo), "cat-file", "blob", commit + ":" + name])


def write_member(archive, name, data):
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data, compress_type=ZIP_DEFLATED)


def package(a):
    cfg = load(a.config)
    commit = cfg["code_commit"]
    files = json.loads(git_blob(a.repo, commit, "configs/public-source.json"))["files"]
    def data(name):
        require(name in files, "file outside public source inventory: " + name)
        return git_blob(a.repo, commit, name)
    context = {k: hashlib.sha256(data(DOCS + k)).hexdigest() for k in REQUIRED_DOCS}
    members = {}
    subset = sorted(k for k in files if k.startswith(EXAMPLE))
    require({EXAMPLE + f for f in ("index.html", "case-index.json", "case-viewer.json")} <= set(subset),
            "exhibit or verified receipts missing")
    case_id = hashlib.sha256(data(EXAMPLE + "case-index.json")).hexdigest()
    require(case_id == cfg["sources"]["cases"]["sha256"], "case index differs from the recorded extraction")
    viewer_bytes = data(EXAMPLE + "case-viewer.json")
    require(hashlib.sha256(viewer_bytes).hexdigest() == cfg["sources"]["viewer"]["sha256"] and
            json.loads(viewer_bytes)["parents"]["case_data"] == cfg["sources"]["cases"],
            "viewer receipt differs from the recorded interface check")
    require(context["primary-editions.json"] == cfg["sources"]["editions"]["sha256"] and
            context["registration.json"] == cfg["sources"]["registration"]["sha256"] and
            context["case-data-check.json"] == cfg["sources"]["case_readback"]["sha256"],
            "reading, registration or case readback differs")
    cases = json.loads(data(EXAMPLE + "case-index.json"))
    require(cases["new_training_runs"] == cases["new_renders"] == 0 and
            [len(c["views"]) for c in cases["cases"]] == [6, 1] and
            [c["available_count"] for c in cases["cases"]] == [4, 128],
            "unexpected case roster or additional experiment")
    with ZipFile(a.out / VIEWER, "x") as archive:
        def insert(name, raw):
            require(name not in members and name != "CHECKSUMS.txt" and PurePosixPath(name).as_posix() == name and
                    not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts,
                    "invalid portable exhibit member")
            members[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            write_member(archive, name, raw)
        for name in subset:
            require(files[name] == "cc-original", "unverified exhibit rights: " + name)
            insert(name[len(EXAMPLE):], data(name))
        for notice in ("CC-BY-4.0.txt", "EXAMPLE-LICENSE.txt"):
            insert("licenses/" + notice, data("licenses/" + notice))
        insert("licenses/Apache-2.0.txt", data("LICENSE"))
        web = "https://github.com/wany614/evidence-lens/tree/" + commit + "/" + DOCS
        insert("OPEN.txt", ("Extract the entire directory and open index.html in a web browser. "
               "The case viewer, sources and evaluation displays run offline.\n"
               "Edition 1. Original exhibit, images and measurements by Evidence Lens / wany614. "
               "CC BY 4.0. The embedded program is Apache-2.0.\n"
               "Read the primary-source cards, model history and records of decisions at " + web + "\n").encode())
        checks = "".join(v["sha256"] + "  " + k + "\n" for k, v in sorted(members.items()))
        write_member(archive, "CHECKSUMS.txt", checks.encode())
    with ZipFile(a.out / VIEWER) as archived:
        require(archived.testzip() is None and len(archived.namelist()) == len(members) + 1 and
                set(archived.namelist()) == set(members) | {"CHECKSUMS.txt"},
                "portable exhibit package differs")
    save(a.out / "package.json", dict(schema="el-foundation-portable-exhibit-1",
         run_id=os.environ["EL_RUN_ID"], code_commit=commit, viewer_zip=VIEWER, members=members,
         sha256=sha(a.out / VIEWER), bytes=(a.out / VIEWER).stat().st_size, case_index_sha256=case_id,
         documents=context, source_archive=cases["source_photo_archive"], parents=cfg["sources"]))


def inventory(a):
    cfg = load(a.config)
    package_info = load(a.package)
    require(sha(a.package) == cfg["sources"]["package"]["sha256"] and
            all(package_info["parents"][role] == cfg["sources"][role] for role in STAGES) and
            package_info["case_index_sha256"] == cfg["sources"]["cases"]["sha256"],
            "package and release select different evidence")
    require(package_info["code_commit"] == cfg["code_commit"] and a.asset[0].name == VIEWER and
            sha(a.asset[0]) == package_info["sha256"] and a.asset[1].name == PHOTOS and
            sha(a.asset[1]) == package_info["source_archive"]["sha256"] and
            a.asset[1].stat().st_size == package_info["source_archive"]["bytes"], "wrong exhibit assets")
    assets = {}
    for p, role, rights in zip(a.asset, ("exhibit", "photos"),
        ("CC BY 4.0 original presentation, documentation and rendered examples. Embedded program Apache-2.0.",
         "CC BY 4.0 original synthetic camera observations, derived renders and grantable measurements. See the member inventory."),
        strict=True):
        pin = cfg["sources"][role]
        require(sha(p) == pin["sha256"], "asset differs from its selected receipt")
        assets[p.name] = dict(role=role, bytes=p.stat().st_size, sha256=pin["sha256"], parent=pin, license=rights)
    contents = dict(schema="el-humanities-foundation-release-1", version=1, tag=TAG,
        run_id=os.environ["EL_RUN_ID"], code_commit=cfg["code_commit"], status="completed",
        A_BASELINE_ID="sha256:" + cfg["sources"]["baseline"]["sha256"],
        B_MANIFEST_SHA256=cfg["sources"]["research"]["sha256"], case_index_sha256=package_info["case_index_sha256"],
        authored_target_views=7, cases=["offset-bell-cage / v01–v06", "b05-perforated-gallery / R128 / h00"],
        gallery_budgets=[4, 8, 16, 32, 128], training_and_calibration_in_C=0,
        documents=package_info["documents"], assets=assets, parents=cfg["sources"])
    save(a.out / MANIFEST, contents)
    sums = {name: v["sha256"] for name, v in assets.items()}
    sums[MANIFEST] = sha(a.out / MANIFEST)
    (a.out / "SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in sorted(sums.items())), encoding="utf-8")
    (a.out / "release-notes.md").write_text(
        "# Inspecting an image and its sources\n\n"
        "Extract the viewer ZIP and open index.html. The six original bell views have four available "
        "synthetic source images. The fixed gallery image lets you inspect queries of 4, 8, 16, 32 and 128 sources. "
        "Open evaluation separately to see the independent reference and measured RGB error. "
        "The larger source package supplies originals and numerical data for inspection.\n\n"
        f"[Read the model history, primary theory cards and contribution record](https://github.com/wany614/evidence-lens/tree/{cfg['code_commit']}/{DOCS})\n",
        encoding="utf-8")


def control(a):
    s = a.store
    program = ["{python}", "scripts/foundation_release.py", a.operation, "--out", "{out}"]
    if a.operation in ("package", "inventory"):
        require(a.commit and re.fullmatch("[0-9a-f]{40}", a.commit) and
                subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == a.commit,
                "the humanities release must name an exact source commit")
        subprocess.run(["git", "diff", "--quiet", a.commit, "--", "scripts/foundation_release.py",
                        "scripts/many_distribute.py", "configs/public-source.json"], check=True)
        pins = {role: selected(s, stage_name) for role, stage_name in STAGES.items()}
        details = {"code_commit": a.commit, "tag": TAG}
        program += ["--config", "{config}"]
        if a.operation == "package":
            return stage(s, "c-exhibit-package", "archive_verify", pins, details,
                         program + ["--repo", str(Path.cwd())], "payload/package.json", 35000000)
        pins["package"] = selected(s, "c-exhibit-package")
        pins["exhibit"] = bind(s, pins["package"]["run_id"], "payload/" + VIEWER)
        pins["photos"] = bind(s, pins["cases"]["run_id"], "payload/" + PHOTOS)
        return stage(s, "c-foundation-release", "archive_verify", pins, details,
                     program + ["--package", "{input:package}", "--asset", "{input:exhibit}", "{input:photos}"],
                     "payload/" + MANIFEST, 18000)
    pin = selected(s, "c-foundation-release")
    pins = {"manifest": pin}
    spec = load(resolve((s, s / "history"), pin))
    program += ["--manifest", "{input:manifest}"]
    details = {"tag": TAG, "code_commit": spec["code_commit"]}
    if a.operation == "download":
        pins["published"] = selected(s, "c-github-publish")
        return stage(s, "c-anonymous-readback", "archive_verify", pins, details,
                     program + ["--published", "{input:published}"], "payload/readback.json",
                     sum(item["bytes"] for item in spec["assets"].values()) + 30000)
    pins["source_push"] = selected(s, "c-push-release-source")
    pins["notes"] = bind(s, pin["run_id"], "payload/release-notes.md")
    pins["sums"] = bind(s, pin["run_id"], "payload/SHA256SUMS")
    program += ["--gh", str(a.gh.resolve()), "--repo", str(Path.cwd()), "--notes", "{input:notes}",
                "--asset", "{input:manifest}", "{input:sums}"]
    for entry in spec["assets"].values():
        role = entry["role"]
        pins[role] = entry["parent"]
        program.append("{input:" + role + "}")
    if a.operation == "publish":
        pins["draft"] = selected(s, "c-github-draft")
        program += ["--draft", "{input:draft}"]
    return stage(s, "c-github-" + a.operation, "archive_verify", pins, details, program, "payload/github.json", 6000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("package", "inventory", "draft", "publish", "download"))
    for key in ("store", "gh", "out", "config", "repo", "package", "manifest", "notes", "draft", "published"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--asset", nargs="+", type=Path)
    a = parser.parse_args()
    if a.store:
        print(control(a))
    elif a.operation == "package":
        package(a)
    elif a.operation == "inventory":
        inventory(a)
    elif a.operation == "download":
        anonymous(a, load(a.manifest), TAG, MANIFEST, "foundation", "published_foundation")
    else:
        release(a, load(a.manifest), TAG, MANIFEST, TITLE, "foundation")
