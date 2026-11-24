"""JSON and file integrity for portable deployment evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath


DIGEST = re.compile(r"[a-f0-9]{64}")
PROFILE = "v0.3.0-rc.1/original-thresholds"
RELEASE_INDEX_SHA256 = "150b04e702f5386b5808ac17545cd073e3d4d837ea789e6ae167168ac086084d"
ASSETS = Path(__file__).with_name("assets")
SEMANTICS = {"rgb": "display_srgb_0_1", "z": "expected_camera_z_m", "alpha": "accumulated_opacity"}


class BundleError(ValueError):
    """An actionable input or integrity error."""


class Unsupported(BundleError):
    """Well-formed evidence with unsupported semantics or model domain."""


def require(ok, message, exception=BundleError):
    if not ok:
        raise exception(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(data):
    return (json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def write(path, obj):
    path.write_bytes(encoded(obj))


def json_bytes(data):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            require(key not in obj, "duplicate JSON key: " + key)
            obj[key] = value
        return obj

    def invalid(value):
        raise BundleError("nonfinite JSON literal: " + value)

    obj = json.loads(data, object_pairs_hook=unique, parse_constant=invalid)
    require(isinstance(obj, dict), "expected a JSON object")
    return obj


def digest(value, role):
    require(isinstance(value, str) and DIGEST.fullmatch(value), "invalid SHA-256 for " + role)
    return value


def text(value, role):
    require(
        isinstance(value, str) and bool(value.strip()) and value.isprintable(), "missing or invalid " + role
    )
    return value


def exact_keys(obj, names, role):
    require(isinstance(obj, dict) and set(obj) == set(names), "unexpected or missing " + role + " fields")


def view_objects(document, role):
    views = document.get("views")
    require(isinstance(views, dict), role + ".views must be an object keyed by view ID")
    for name, entry in views.items():
        require(isinstance(entry, dict), role + ".views[" + name + "] must be an object")


def row(path, data):
    return {"path": path, "bytes": len(data), "sha256": sha(data)}


def parts(path):
    require(isinstance(path, str) and path and "\\" not in path and ":" not in path, "invalid member path")
    ps = PurePosixPath(path)
    require(
        not ps.is_absolute()
        and ps.as_posix() == path
        and all(x not in (".", "..") and x.isprintable() for x in ps.parts),
        "unsafe member path",
    )
    return ps.parts


def read_file(root, path):
    result = Path(root)
    for piece in parts(path):
        result = result / piece
        require(not result.is_symlink(), "symlink in member: " + path)
    require(
        result.is_file() and result.resolve().is_relative_to(Path(root).resolve()),
        "missing or unsafe member: " + path,
    )
    return result.read_bytes()


def member(root, spec, *, prefix=None, expected=None):
    exact_keys(spec, ("path", "bytes", "sha256"), "artifact")
    path = spec["path"]
    names = parts(path)
    require(
        (prefix is None or (len(names) >= 2 and names[0] == prefix))
        and (expected is None or path == expected),
        "wrong artifact role: " + path,
    )
    digest(spec["sha256"], path)
    require(type(spec["bytes"]) is int and spec["bytes"] > 0, "invalid length: " + path)
    data = read_file(root, path)
    require(len(data) == spec["bytes"] and sha(data) == spec["sha256"], "artifact bytes differ: " + path)
    return data


def profile():
    raw = read_file(ASSETS, "profile.json")
    require(
        sha(raw) == "99587dd70b67a84debb81fa1dc84e93729c8980bc4531edecfb56112a15eb191",
        "frozen profile identity differs",
    )
    return json_bytes(raw)


def no_replace_file(path, data):
    """Publish complete file without replacing a concurrent writer's output."""
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temp = Path(f.name)
    try:
        temp.write_bytes(data)
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)
