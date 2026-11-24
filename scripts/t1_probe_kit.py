"""Issue the frozen probe executable and its permitted 14-camera geometry without evaluation poses."""

import argparse
import json
from pathlib import Path

from evidence_lens._bundle import encoded, load, sha
from evidence_lens._verify import require
from make_handoff import publish, zip_bytes
from t1_access import POOL
from t1_study import DEFAULT, PROJECT, authority, bind, locate, stage


def build(roots, exchange):
    entry, freeze, fr = authority(roots)
    refs = fr["sources"]
    geo = refs["query_cameras"]
    recipe = bind(roots, refs["reconstruction"], "payload/recipe.json")
    cmd = ["{python}", "-B", "{source}/scripts/t1_probe_kit.py", "worker", "--config", "{config}",
           "--cameras", "{input:cameras}", "--out", "{out}"]
    pin = stage(roots, "t1-probe-kit", "dataset_build", dict(entry=entry, freeze=freeze, selector=refs["selector"],
                cameras=geo, recipe=recipe), dict(scientific_files=fr["selection_code"]), cmd, "ticket.json", 6 * 1024**2)
    kit = bind(roots, pin, "payload/probe-kit.zip")
    for folder in (roots[0] / "transport/Handoffs/t1-v1", exchange):
        publish(folder / "probe-kit.zip", locate(roots, kit).read_bytes())
        publish(folder / "probe-kit-ticket.json", locate(roots, pin).read_bytes())
    return kit


def worker(cfg, geo, out):
    refs = cfg["sources"]
    require(sha(geo) == refs["cameras"]["sha256"] and set(load(geo)["views"]) ==
            {"s00", "s03", *[f"c{i:03}" for i in POOL], *[f"q{i:02}" for i in range(4)]},
            "Probe camera set contains an unregistered pose")
    files = {}
    for base in ("src/recovery", "src/evidence_lens"):
        for p in sorted((PROJECT / base).rglob("*.py")):
            files[p.relative_to(PROJECT).as_posix()] = p.read_bytes()
    for n in ("b2_probe_worker.py", "t1_probe_windows.py"):
        files["scripts/" + n] = (PROJECT / "scripts" / n).read_bytes()
    locked = cfg["scientific_files"]
    for n in ("scripts/b2_probe_worker.py", "src/recovery/scene.py", "src/recovery/engine.py"):
        require(sha(PROJECT / n) == locked[n], "Frozen scientific probe code changed")
    files["probe-authority.json"] = encoded(dict(entry=refs["entry"], freeze=refs["freeze"], selector=refs["selector"],
            cameras=refs["cameras"], pool=POOL, recipe_sha256=refs["recipe"]["sha256"],
            source_sha256={n: locked[n] for n in locked if n in files}))
    files["cameras.json"] = geo.read_bytes()
    p = out / "probe-kit.zip"
    p.write_bytes(zip_bytes(files))
    (out / "ticket.json").write_bytes(encoded(dict(schema="el-t1-probe-kit-1", sha256=sha(p), bytes=p.stat().st_size,
          permitted_views=sorted(load(geo)["views"]), camera_pin=refs["cameras"], entry=refs["entry"],
          candidate_photographs=0, evaluation_poses=0, evaluation_photographs=0)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["issue", "worker"])
    for n in ("config", "cameras", "out", "exchange"):
        p.add_argument("--" + n, type=Path)
    p.add_argument("--store", type=Path, default=DEFAULT)
    a = p.parse_args()
    if a.action == "worker":
        worker(load(a.config), a.cameras, a.out)
    else:
        print(json.dumps(build((a.store.resolve(), a.store.resolve() / "history"), a.exchange)))
