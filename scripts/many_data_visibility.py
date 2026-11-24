"""Evaluator-only exact subpixel source occlusion for the preregistered 7-pixel authored probe."""

import argparse
import ctypes
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np

from dual_records import bind, resolve, selected, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from many_data_prepare import confirmation_terminals


ROOT = Path(__file__).resolve().parents[1]
NEAR = 0.05
ENDPOINT = 1e-6


def dots(a, b):
    return np.einsum("ij,ij->i", a, b)


class Reference:
    """Exhaustive plane-intersection and Gram-matrix barycentrics, independent of the BVH and its triangle test."""

    def __init__(self, triangles):
        self.a = triangles[:, 0]
        self.e = triangles[:, 1] - self.a
        self.f = triangles[:, 2] - self.a
        self.normals = np.cross(self.e, self.f)
        self.area = np.linalg.norm(self.normals, axis=1)
        self.ee, self.ff, self.ef = dots(self.e, self.e), dots(self.f, self.f), dots(self.e, self.f)
        self.den = self.ee * self.ff - self.ef * self.ef

    def nearest(self, origin, direction, min_distance):
        projections = self.normals @ direction
        with np.errstate(invalid="ignore", divide="ignore"):
            t = dots(self.a - origin, self.normals) / projections
            offset = origin - self.a + t[:, None] * direction
            ep, fp = dots(offset, self.e), dots(offset, self.f)
            u = (self.ff * ep - self.ef * fp) / self.den
            v = (self.ee * fp - self.ef * ep) / self.den
        hit = (
            (np.abs(projections) > 1e-12 * self.area)
            & (t >= min_distance)
            & (u >= -1e-10)
            & (v >= -1e-10)
            & (u + v <= 1 + 1e-10)
        )
        return t[hit].min() if hit.any() else np.inf

    def check(self, target, sources, xy, world, z, inside, visible):
        k, ct = target
        # Two foreground rays closest to fixed indices and both grid endpoints; selection uses true geometry only.
        foreground = np.flatnonzero(np.isfinite(z))
        chosen = {0, len(z) - 1}
        if len(foreground):
            for p in (len(z) // 3, 2 * len(z) // 3):
                chosen.add(int(foreground[np.argmin(abs(foreground - p))]))
        source_checks = 0
        for j in sorted(chosen):
            ray = np.array([(xy[j, 0] - k[0, 2]) / k[0, 0], (xy[j, 1] - k[1, 2]) / k[1, 1], 1.0])
            norm = np.linalg.norm(ray)
            direction = ct[:3, :3] @ (ray / norm)
            distance = self.nearest(ct[:3, 3], direction, NEAR * norm)
            require(np.isfinite(distance) == np.isfinite(z[j]), "independent target hit/miss differs")
            if not np.isfinite(distance):
                require(
                    not inside[:, j].any() and not visible[:, j].any(),
                    "background has an alleged visible source",
                )
                continue
            point = ct[:3, 3] + distance * direction
            require(
                abs(z[j] - distance / norm) < 2e-8 and np.max(abs(world[j] - point)) < 2e-8,
                "wrong target surface point",
            )
            for i, (ks, cs) in enumerate(sources):
                d = point - cs[:3, 3]
                local = cs[:3, :3].T @ d
                inz = False
                observed = False
                if local[2] >= NEAR:
                    x, y = (ks @ local)[:2] / local[2]
                    inz = -0.5 <= x < 377.5 and -0.5 <= y < 251.5
                    if inz:
                        length = np.linalg.norm(d)
                        closest = self.nearest(cs[:3, 3], d / length, NEAR * length / local[2])
                        observed = closest > length - ENDPOINT
                require(
                    inz == inside[i, j] and observed == visible[i, j],
                    "direct exhaustive and BVH true visibility disagree",
                )
                source_checks += 1
        return {"target_rays": len(chosen), "source_checks": source_checks, "indices": sorted(chosen)}


class Intersector:
    def __init__(self, source, work):
        lib = work / ("occlusion.dylib" if sys.platform == "darwin" else "occlusion.so")
        subprocess.run(
            ["c++", "-std=c++17", "-O3", "-fPIC", "-shared", str(source), "-o", str(lib)], check=True
        )
        self.func = ctypes.CDLL(str(lib)).visible_samples
        ptr = ctypes.c_void_p
        self.func.argtypes = [
            ptr,
            ctypes.c_int,
            ptr,
            ptr,
            ptr,
            ptr,
            ctypes.c_int,
            ptr,
            ctypes.c_int,
            ptr,
            ptr,
            ptr,
            ptr,
        ]
        self.func.restype = ctypes.c_int

    def evaluate(self, triangles, target, sources, xy):
        def array(v, dtype=np.float64):
            return np.ascontiguousarray(v, dtype=dtype)

        def ptr(v):
            return v.ctypes.data

        vertices = array(triangles)
        kt, ct = (array(m) for m in target)
        ks = array([k for k, _ in sources])
        cs = array([c for _, c in sources])
        xy = array(xy, np.int32)
        n = len(xy)
        ns = len(sources)
        points = np.full((n, 3), np.nan)
        depth = np.full(n, np.nan)
        inside = np.zeros((ns, n), np.uint8)
        visible = np.zeros_like(inside)
        status = self.func(
            ptr(vertices),
            len(vertices),
            ptr(kt),
            ptr(ct),
            ptr(ks),
            ptr(cs),
            ns,
            ptr(xy),
            n,
            ptr(points),
            ptr(depth),
            ptr(inside),
            ptr(visible),
        )
        require(status == 0, "continuous segment implementation failed")
        return points, depth, inside.astype(bool), visible.astype(bool)


def controls(tracer):
    k = np.array([[100.0, 0.0, 189.0], [0.0, 100.0, 126.0], [0.0, 0.0, 1.0]])
    ct = np.eye(4)
    cs = ct.copy()
    cs[0, 3] = 1.0
    xy = [[189, 126]]
    far = [[-2.0, -2.0, 6.0], [2.0, -2.0, 6.0], [0.0, 2.0, 6.0]]

    def blocker(x, z, half):
        return [[x - half, -0.12, z], [x + half, -0.12, z], [x, 0.12, z]]

    cases = (
        ("no_blocker", [far], True),
        ("coincident_endpoint", [far, far], True),
        ("near_wall", [far, blocker(0.5, 3.0, 0.15)], False),
        ("parallel_surface_5cm_in_front", [far, blocker(0.05 / 6, 5.95, 0.003)], False),
        ("near_wall_outside_segment", [far, blocker(0.9, 3.0, 0.03)], True),
    )
    results = {}
    for name, scene, expected in cases:
        arr = np.asarray(scene)
        target = (k, ct)
        source = [(k, cs)]
        world, z, inside, visible = tracer.evaluate(arr, target, source, xy)
        require(
            abs(z[0] - 6) < 1e-12 and inside[0, 0] and visible[0, 0] == expected,
            "occlusion control failed: " + name,
        )
        Reference(arr).check(target, source, np.asarray(xy), world, z, inside, visible)
        results[name] = bool(visible[0, 0])
    return results


def mesh_stage(s):
    roles = selected(s, "b-data-registration")
    registered = load(resolve((s,), roles))
    pins = {"registration": roles}
    snapshot = {}
    for i, r in enumerate(registered["originals"]):
        source = selected(s, "b-source-" + r["id"])
        pins["data" + str(i)] = source
        snapshot[r["id"]] = bind(s, source["run_id"], "source/scripts/many_data_original.py")
    return stage(
        s,
        "b-visibility-source-geometry",
        "archive_verify",
        pins,
        {
            "purpose": "Promote each original acquisition's manifest-listed mesh program to an explicit evaluator asset",
            "snapshot_members": snapshot,
        },
        [
            "{python}",
            "-B",
            "{source}/scripts/many_data_visibility.py",
            "mesh-worker",
            "--store",
            str(s),
            "--config",
            "{config}",
            "--out",
            "{out}",
        ],
        "payload/source.json",
        90000,
    )


def mesh_worker(a):
    cfg = load(a.config)
    require(os.environ.get("EL_RUN_ID"), "use recorded source promotion")
    emitted = {}
    permitted = {p["run_id"] for key, p in cfg["sources"].items() if key.startswith("data")}
    for family, pin in cfg["snapshot_members"].items():
        require(
            pin["run_id"] in permitted and pin["member"] == "source/scripts/many_data_original.py",
            "source program outside its native acquisition",
        )
        src = resolve((a.store,), pin)
        name = "sourcegen-" + family + ".py"
        shutil.copyfile(src, a.out / name)
        require(sha(a.out / name) == pin["sha256"], "source promotion altered bytes")
        emitted[family] = {"path": name, "sha256": pin["sha256"], "source_snapshot": pin}
    save(
        a.out / "source.json",
        {"schema": "el-many-native-geometry-code-1", "run_id": os.environ["EL_RUN_ID"], "programs": emitted},
    )


def record(s, scene):
    s = s.resolve()
    pins = {
        "protocol": selected(s, "b-visibility-probe-registration"),
        "registration": selected(s, "b-data-registration"),
    }
    spec = load(resolve((s, s / "history"), pins["registration"]))
    row = next((r for r in spec["originals"] if r["id"] == scene), None)
    require(row is not None, "exact geometry is restricted to the six authored families")
    confirm = row["split"] == "confirmation"
    label = "b-confirmation-h-opening" if confirm else "b-train-dev-h-opening"
    pins["opening"] = selected(s, label)
    gatepath = resolve((s, s / "history"), pins["opening"])
    gate = load(gatepath)
    if confirm:
        all_parents = {
            (r["run_id"], r["manifest_sha256"])
            for r in load(gatepath.parents[1] / "manifest.json")["parents"]
        }
        for member in confirmation_terminals(gate, spec):
            require(
                (member["run_id"], member["manifest_sha256"]) in all_parents, "unlisted confirmation terminal"
            )
            resolve((s, s / "history"), member)
    else:
        require(
            gate["phase"] == "train-development"
            and gate["allowed_targets"].get(scene) == [f"h{i:02d}" for i in range(6)],
            "missing training/development permission",
        )
    pins["truth"] = selected(s, "b-truth-" + scene)
    pins["source"] = selected(s, "b-source-" + scene)
    owner = pins["source"]["run_id"]
    pins["source_package"] = bind(s, owner, "payload/R128.zip")
    mesh_pin = bind(s, owner, "source/scripts/many_data_original.py")
    promoted = mesh_stage(s)
    pins["geometry_receipt"] = promoted
    pins["original_mesh"] = bind(s, promoted["run_id"], "payload/sourcegen-" + scene + ".py")
    predecessor = []
    for prefix in ("b-visible-", "b-exact-visible-"):
        attempt = s / "work/controller" / (prefix + scene) / "attempt-01.jsonl"
        if attempt.is_file():
            failed = json.loads(attempt.read_text().splitlines()[-1])
            require(
                failed["execution_status"] == "failed" and "original_mesh" in failed["failure_reason"],
                "unexpected predecessor for the positive archived source-code input",
            )
            predecessor.append({k: failed[k] for k in ("run_id", "manifest_sha256", "failure_reason")})
    cmd = [
        "{python}",
        "-B",
        "{source}/scripts/many_data_visibility.py",
        "worker",
        "--scene",
        scene,
        "--config",
        "{config}",
        "--out",
        "{out}",
        "--solver",
        "{input:solver}",
    ]
    for k in pins:
        cmd += ["--input", k, "{input:" + k + "}"]
    return stage(
        s,
        "b-exact-visibility-" + scene,
        "evaluation",
        pins,
        {
            "purpose": "exact evaluator-only mesh occlusion, 1944 samples per view",
            "scene": scene,
            "split": row["split"],
            "gate": label,
            "original_mesh_source_member": mesh_pin,
            "technical_predecessor": predecessor,
        },
        cmd,
        "payload/visibility.json",
        3500000,
        {"solver": ROOT / "scripts/many_data_visibility.cpp"},
    )


def worker(a):
    require(os.environ.get("EL_RUN_ID"), "submit evaluator as a native recorded stage")
    clock, cpu = time.monotonic(), time.process_time()
    cfg = load(a.config)
    files = {k: Path(v) for k, v in a.input}
    require(
        cfg["original_mesh_source_member"]["run_id"] == cfg["sources"]["source"]["run_id"]
        and load(files["geometry_receipt"])["programs"][a.scene]["source_snapshot"]
        == cfg["original_mesh_source_member"]
        and cfg["original_mesh_source_member"]["sha256"] == sha(files["original_mesh"]),
        "different original geometry code from this family's R acquisition",
    )
    protocol = load(files["protocol"])
    require(
        protocol["analysis_size_wh"] == [378, 252]
        and protocol["sample_centres"]["grid_wh"] == [54, 36]
        and protocol["sample_centres"]["first_xy"] == [3, 3]
        and protocol["sample_centres"]["stride_xy"] == [7, 7],
        "different independent visibility registration",
    )
    truth = load(files["truth"])
    opening = load(files["opening"])
    require(
        truth["scene_id"] == a.scene
        and truth["opening_sha256"] == sha(files["opening"])
        and truth["inputs"]["source"] == cfg["sources"]["source"]
        and truth["split"] == cfg["split"]
        and opening["allowed_targets"][a.scene] == truth["H"],
        "geometry cannot be opened under this grant",
    )
    registration = load(files["registration"])
    if cfg["split"] == "confirmation":
        confirmation_terminals(opening, registration)
    else:
        require(opening["phase"] == "train-development", "unexpected training/development opening")
    src = load(files["source"])
    require(
        sha(files["source_package"]) == src["packages"]["R128"]["sha256"], "different source geometry cameras"
    )
    with zipfile.ZipFile(files["source_package"]) as z:
        scene = json.loads(z.read("scene.json"))
    require(
        scene["R"] == src["R"]
        and scene["H"] == src["H"] == truth["H"]
        and scene["unit"] == "m"
        and scene["run_id"] == src["run_id"]
        and scene["frame_receipt"]["world_to_deployment"] == np.eye(4).tolist(),
        "expected original full metric camera frame",
    )
    imported = importlib.util.spec_from_file_location("archived_original_mesh", files["original_mesh"])
    original = importlib.util.module_from_spec(imported)
    imported.loader.exec_module(original)
    faces, _ = original.geometry(a.scene)
    triangles = np.ascontiguousarray(faces[:, :, :3], np.float64)
    checker = Reference(triangles)
    coords = np.array([[(x, y) for x in range(3, 378, 7)] for y in range(3, 252, 7)], np.int32)
    xy = coords.reshape(-1, 2)

    def camera(view):
        c = scene["views"][view]
        return np.array(c["K"]), np.array(c["c2w"])

    sources = [camera(v) for v in scene["R"]]
    references = {}
    quality = {}
    archive = a.out / "visibility.zip"
    with tempfile.TemporaryDirectory(dir=a.out.parent) as folder:
        tracer = Intersector(a.solver, Path(folder))
        tests = controls(tracer)
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as pack:
            for v in scene["H"]:
                target = camera(v)
                world, z, inside, visible = tracer.evaluate(triangles, target, sources, xy)
                require(
                    not (visible & ~inside).any()
                    and np.isfinite(z).any()
                    and np.all(z[np.isfinite(z)] >= NEAR),
                    "invalid exact source or target support",
                )
                independent = checker.check(target, sources, xy, world, z, inside, visible)
                out = io.BytesIO()
                np.savez_compressed(
                    out,
                    xy=coords,
                    world_xyz=world.reshape(36, 54, 3),
                    target_optical_z=z.reshape(36, 54),
                    foreground=np.isfinite(z).reshape(36, 54),
                    in_frame=inside.reshape(len(sources), 36, 54),
                    visible=visible.reshape(len(sources), 36, 54),
                )
                content = out.getvalue()
                name = v + ".npz"
                pack.writestr(name, content)
                references[v] = {
                    "path": name,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "bytes": len(content),
                }
                quality[v] = {
                    "all_probe_centres": len(xy),
                    "target_foreground_probes": int(np.isfinite(z).sum()),
                    "source_in_frame_pairs": int(inside.sum()),
                    "true_visible_source_pairs": int(visible.sum()),
                    "independent_exhaustive_check": independent,
                }
    save(
        a.out / "visibility.json",
        {
            "schema": "el-many-exact-visibility-1",
            "run_id": os.environ["EL_RUN_ID"],
            "scene_id": a.scene,
            "split": cfg["split"],
            "unit": "m",
            "H": scene["H"],
            "source_ids": scene["R"],
            "probe_grid_wh": [54, 36],
            "inputs": cfg["sources"],
            "mesh_triangles": len(triangles),
            "mesh_world_vertices_sha256": hashlib.sha256(triangles.astype("<f8").tobytes()).hexdigest(),
            "solver_sha256": sha(a.solver),
            "method": "continuous segment/triangle intersection, positive source near plane, endpoint exclusion 1e-6 m",
            "synthetic_control_visibility": tests,
            "geometry": quality,
            "targets": references,
            "image_or_score_pixel_reads": 0,
            "archive": {"path": archive.name, "bytes": archive.stat().st_size, "sha256": sha(archive)},
            "resource": {
                "worker_wall_seconds": time.monotonic() - clock,
                "worker_cpu_seconds": time.process_time() - cpu,
                "child_cpu_seconds": sum(resource.getrusage(resource.RUSAGE_CHILDREN)[:2]),
                "peak_process_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
        },
    )


class VisibilityTruth:
    """Read one permitted geometry result, with Q subsets taken only after fixed exact source tests."""

    def __init__(self, index, expected_sha256):
        self.index = Path(index).resolve()
        require(sha(self.index) == expected_sha256, "different exact visibility receipt")
        self.meta = load(self.index)
        require(self.meta["schema"] == "el-many-exact-visibility-1", "unexpected evaluator geometry schema")
        spec = self.meta["archive"]
        self.archive = self.index.parent / spec["path"]
        require(
            self.archive.resolve().parent == self.index.parent
            and self.archive.stat().st_size == spec["bytes"]
            and sha(self.archive) == spec["sha256"],
            "different exact visibility arrays",
        )
        self._cached = None

    def verify(self, truth):
        require(
            self.meta["scene_id"] == truth.meta["scene_id"]
            and self.meta["split"] == truth.meta["split"]
            and self.meta["H"] == truth.meta["H"]
            and self.meta["inputs"]["truth"]["sha256"] == sha(truth.index)
            and self.meta["inputs"]["source"] == truth.meta["inputs"]["source"]
            and self.meta["inputs"]["opening"]["sha256"] == truth.meta["opening_sha256"],
            "wrong geometry, source or H grant",
        )

    def target(self, view, queries):
        queries = tuple(queries)
        ids = self.meta["source_ids"]
        require(
            view in self.meta["H"] and len(queries) == len(set(queries)) > 0 and set(queries) <= set(ids),
            "invalid target or source subset for exact geometry",
        )
        if self._cached is None or self._cached[0] != view:
            spec = self.meta["targets"][view]
            with zipfile.ZipFile(self.archive) as z:
                content = z.read(spec["path"])
            require(
                len(content) == spec["bytes"] and hashlib.sha256(content).hexdigest() == spec["sha256"],
                "different geometry member",
            )
            with np.load(io.BytesIO(content), allow_pickle=False) as arrays:
                a = {k: arrays[k] for k in arrays.files}
            require(
                a["xy"].shape == (36, 54, 2)
                and a["visible"].shape == a["in_frame"].shape == (len(ids), 36, 54)
                and a["foreground"].shape == a["target_optical_z"].shape == (36, 54)
                and a["world_xyz"].shape == (36, 54, 3)
                and a["visible"].dtype == a["foreground"].dtype == np.bool_,
                "incorrect exact visibility grid",
            )
            self._cached = (view, a)
        a = self._cached[1]
        indices = [ids.index(k) for k in queries]
        return {
            "xy": a["xy"],
            "foreground": a["foreground"],
            "target_optical_z": a["target_optical_z"],
            "world_xyz": a["world_xyz"],
            "visible": np.any(a["visible"][indices], axis=0),
            "in_frame": np.any(a["in_frame"][indices], axis=0),
            "per_source_visible": a["visible"][indices],
            "source_ids": list(queries),
        }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("record", "worker", "register", "mesh-worker"))
    for n in ("store", "out", "config", "solver"):
        p.add_argument("--" + n, type=Path)
    p.add_argument("--input", nargs=2, action="append")
    p.add_argument("--scene")
    a = p.parse_args()
    if a.action == "record":
        print(json.dumps(record(a.store, a.scene)), flush=True)
    elif a.action == "register":
        s = a.store.resolve()
        pin = stage(
            s,
            "b-visibility-probe-registration",
            "archive_verify",
            {"roles": selected(s, "b-data-registration"), "study": selected(s, "b-study-registration")},
            {
                "purpose": "exact authored surface occlusion, fixed 1944 centres per view, before confirmation access",
                "phase": "before triangle-visibility measurements",
            },
            [
                "{python}",
                "-I",
                "-c",
                "import shutil,sys; shutil.copyfile(sys.argv[1],sys.argv[2])",
                "{input:protocol}",
                "{out}/visibility-protocol.json",
            ],
            "payload/visibility-protocol.json",
            10000,
            {"protocol": ROOT / "protocol/many-source-visibility-v1.json"},
        )
        print(json.dumps(pin), flush=True)
    elif a.action == "mesh-worker":
        mesh_worker(a)
    else:
        worker(a)
