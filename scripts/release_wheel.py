"""Verify an installed-distribution wheel's bytes, RECORD and declared dependencies against Git."""

import base64
import csv
import hashlib
import io
import json
import re
import stat
import subprocess
import tomllib
import zipfile
from email.parser import BytesParser

from evidence_lens._bundle import local_path

PUBLIC_WHEEL_LICENSE = "Apache-2.0 AND CC-BY-4.0"
PUBLIC_WHEEL_NOTICES = ("LICENSE", "NOTICE", "licenses/CC-BY-4.0.txt", "THIRD_PARTY_NOTICES.md", "licenses/DEPENDENCY-SOURCES.json")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def aggregate(rows):
    return sha(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode())


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement(raw):
    # The project's supported specifiers are plain names, comma-separated bounds and optional extras.
    m = re.fullmatch(r"([\w.-]+)\s*([^;]*?)(?:\s*;\s*extra\s*==\s*[\"']([^\x22\x27]+)[\"'])?", raw.strip())
    require(m, f"unsupported wheel dependency declaration {raw!r}")
    name, bounds, extra = m.groups()
    bounds = bounds.strip().removeprefix("(").removesuffix(")")
    parts = [s.replace(" ", "") for s in bounds.split(",") if s.strip()]
    require(
        all(re.fullmatch(r"(?:===|==|>=|<=|!=|~=|>|<)[a-zA-Z0-9*.+!-]+", s) for s in parts),
        f"unsupported wheel dependency bounds for {name}",
    )
    return canonical(name), tuple(sorted(parts)), canonical(extra) if extra else ""


def read_wheel(path, project_toml, *, checkout=None, commit=None, text_check=None):
    py = tomllib.loads(project_toml.decode())["project"]
    dist, version = py["name"].replace("-", "_"), py["version"]
    public = isinstance(py.get("license"),str)
    notices = set(py.get("license-files", ()))
    if public:
        require(py["license"]==PUBLIC_WHEEL_LICENSE and notices==set(PUBLIC_WHEEL_NOTICES)
                and py.get("authors")==[{"name":"wany614"}] and py.get("urls",{}).get("Repository")=="https://github.com/wany614/evidence-lens",
                "public project license, attribution, notices or repository differs")
    else:
        require(not notices, "PEP 639 license files require an explicit expression")
    allowed = {"RECORD","WHEEL","METADATA","top_level.txt","licenses/LICENSE"}
    allowed.update("licenses/"+n for n in notices)
    if not public:
        allowed.add("LICENSE")
    expected_name = f"{dist}-{version}-py3-none-any.whl"
    require(
        path.name == expected_name and path.is_file() and not path.is_symlink(),
        "wheel identity, version or tag differs from the committed project",
    )
    info = f"{dist}-{version}.dist-info/"
    with zipfile.ZipFile(path) as z:
        items = z.infolist()
        names = [local_path(i.filename) for i in items]
        require(len({n.casefold() for n in names}) == len(names), "duplicate or case-colliding wheel member")
        require(
            all(not i.is_dir() and (i.external_attr >> 16 & 0o170000) in (0, stat.S_IFREG) for i in items),
            "wheel contains a directory link or special file",
        )
        records = list(csv.reader(io.StringIO(z.read(info + "RECORD").decode("utf-8"), newline="")))
        require(
            all(len(row) == 3 for row in records)
            and len(records) == len(names)
            and {r[0] for r in records} == set(names),
            "wheel RECORD inventory differs",
        )
        modules, metadata = {}, {}
        for name, check, size in records:
            data = z.read(name)
            if name == info + "RECORD":
                require(check == size == "", "wheel RECORD must have an empty self-hash")
            else:
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                require(
                    check == "sha256=" + digest and size == str(len(data)),
                    f"wheel RECORD mismatch for {name}",
                )
            if name.startswith("evidence_lens/") and name.endswith(".py"):
                modules[name] = sha(data)
                if text_check:
                    text_check(name, data)
            elif name.startswith(info) and name[len(info) :] in allowed:
                metadata[name[len(info) :]] = data
                if text_check:
                    text_check("LICENSE" if name.startswith(info+"licenses/") or name.endswith("/LICENSE") else name, data)
            else:
                raise ValueError(f"unexpected wheel member {name}")
        wheel, desc = (BytesParser().parsebytes(metadata[k]) for k in ("WHEEL", "METADATA"))
        require(
            wheel["Wheel-Version"] == "1.0"
            and wheel["Root-Is-Purelib"] == "true"
            and wheel.get_all("Tag") == ["py3-none-any"]
            and canonical(desc["Name"]) == canonical(py["name"])
            and desc["Version"] == version
            and desc["Summary"] == py["description"]
            and desc["Requires-Python"] == py["requires-python"],
            "wheel compatibility metadata differs from the project",
        )
        expected = {requirement(v) for v in py.get("dependencies", [])}
        extras = py.get("optional-dependencies", {})
        for extra, items in extras.items():
            expected.update((*requirement(v)[:2], canonical(extra)) for v in items)
        require(
            {requirement(v) for v in desc.get_all("Requires-Dist", [])} == expected
            and {canonical(x) for x in desc.get_all("Provides-Extra", [])} == {canonical(x) for x in extras},
            "wheel requirements differ from the committed pyproject",
        )
        require(
            {requirement(v)[0] for v in py["dependencies"]} == {"numpy", "pillow"},
            "public CPU inference requires only NumPy and Pillow",
        )
        licenses = [metadata[name] for name in ("licenses/LICENSE", "LICENSE") if name in metadata]
        require(len(licenses) == 1, "standalone CPU wheel needs its committed source license")
        if public:
            require(desc["License-Expression"]==PUBLIC_WHEEL_LICENSE and desc["Author"]=="wany614" and
                    "Repository, https://github.com/wany614/evidence-lens" in desc.get_all("Project-URL",[]) and
                    sorted(desc.get_all("License-File",[]))==sorted(notices) and
                    all("licenses/"+n in metadata for n in notices),"wheel public licensing and author metadata differ")
        if "top_level.txt" in metadata:
            require(
                metadata["top_level.txt"].splitlines() == [b"evidence_lens"],
                "wheel declares an unexpected top-level package",
            )
        readme = desc.get_payload(decode=True).replace(b"\r\n", b"\n").rstrip(b"\n")
    required = {
        f"evidence_lens/{n}.py"
        for n in ("__init__", "scoring_io", "features", "risk_model", "l1", "l1_demo", "score_maps")
    }
    require(required <= set(modules), "wheel omits public CPU inference or grading")
    if checkout is not None:
        raw = subprocess.check_output(
            ["git", "-C", str(checkout), "ls-tree", "-rz", commit, "--", "src/evidence_lens"]
        )
        entries = {}
        for item in raw.split(b"\x00"):
            if item:
                head, name = item.split(b"\t", 1)
                mode, kind, oid = head.split()
                name = name.decode("utf-8")
                if name.endswith(".py"):
                    require(
                        kind == b"blob" and mode in (b"100644", b"100755"),
                        "committed wheel code must be a regular file",
                    )
                    entries[name.removeprefix("src/")] = oid.decode()
        require(set(modules) == set(entries), "wheel module inventory differs from the release Git commit")
        for name, oid in entries.items():
            content = subprocess.check_output(["git", "-C", str(checkout), "cat-file", "blob", oid])
            require(sha(content) == modules[name], "wheel source differs from release commit: " + name)
        git = ["git", "-C", str(checkout), "show"]
        original = subprocess.check_output([*git, commit + ":LICENSE"])
        require(licenses[0] == original, "wheel license differs from the release commit")
        for name in notices:
            require(metadata["licenses/"+name] == subprocess.check_output([*git,commit+":"+name]),
                    "wheel notice differs from Git: "+name)
        path = local_path(py["readme"])
        committed_readme = (
            subprocess.check_output([*git, commit + ":" + path]).replace(b"\r\n", b"\n").rstrip(b"\n")
        )
        require(readme == committed_readme, "wheel description differs from the committed project README")
    result = {
        "name": expected_name,
        "distribution": py["name"],
        "version": version,
        "tag": "py3-none-any",
        "requires_python": py["requires-python"],
        "base_dependencies": py["dependencies"],
        "source_license_sha256": sha(licenses[0]),
        "readme_text_sha256": sha(readme),
        "module_count": len(modules),
        "modules_sha256": aggregate(modules),
        "modules": modules,
    }
    if public:
        result.update(license_expression=PUBLIC_WHEEL_LICENSE,attribution="wany614",repository="wany614/evidence-lens",
                      license_file_sha256={n:sha(metadata["licenses/"+n]) for n in sorted(notices)})
    return result
