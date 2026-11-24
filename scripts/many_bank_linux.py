"""Offline arm64 Linux installation and independent comparison on the registered authored bank."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

ACTIONS = ("score", "audit-bank")
PROBE = {
    "scene": "b04-layered-louvers",
    "case": "R16/h00",
    "strategy": "coverage",
    "query_size": 4,
    "model": "actual b-head-final B-CPU-LR, tau=null",
    "cpu": "Linux arm64 base CPU environment, one OpenBLAS/OMP thread",
    "install": "A Linux locked image, clean new venv, exact A and B wheels plus the sealed NumPy and Pillow arm64 wheels, network off and root read-only",
    "deployment": "inspect, score, regenerate offline report, then run the explicit full audit and regenerate its report; model and prepared data from the authorized public example only",
    "comparison": "compare Mac and Linux descriptors, actual ordered Q, all numeric arrays, risk signed and absolute differences, 10%/20% lowest common finite pixels with row-major ties; report actual differences without a calibrated threshold or new training",
}


def need(value, msg):
    if not value:
        raise ValueError(msg)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, data):
    with Path(path).open("x", encoding="utf8") as file:
        json.dump(data, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


def command(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def unpack(archive, target):
    with ZipFile(archive) as z:
        names = z.namelist()
        checknames = set(names)
        need(
            len(names) == len(checknames) and sum(f.file_size for f in z.infolist()) <= 512 << 20,
            "invalid size or repeated package member",
        )
        bank = json.loads(z.read("bank/bank.json"))
        allowed = {"bank/bank.json", "model.json", "README.md", "index.json"}
        for key, data in bank["sources"].items():
            need(key in bank["accessible_ids"], "unexpected source")
            for role in ("photo", "render"):
                allowed.add("bank/" + data[role]["path"])
        allowed.update("bank/" + data["render"]["path"] for data in bank["targets"].values())
        need(allowed == checknames, "example contains members outside its deployment contract")
        index = json.loads(z.read("index.json"))
        lookup = {f["path"]: f for f in index["files"]}
        need(
            set(lookup) == checknames - {"index.json"} and len(lookup) == len(index["files"]),
            "incomplete archive member index",
        )
        for name in sorted(names):
            p = PurePosixPath(name)
            need(
                not p.is_absolute()
                and p.as_posix() == name
                and all(s not in ("", ".", "..") and "\\" not in s for s in p.parts),
                "unsafe deployment member",
            )
            info = z.getinfo(name)
            need((info.external_attr >> 16) & 0o170000 in (0, 0o100000), "expected regular archive member")
            data = z.read(info)
            if name in lookup:
                need(
                    len(data) == lookup[name]["bytes"]
                    and hashlib.sha256(data).hexdigest() == lookup[name]["sha256"],
                    "different archive member",
                )
            dst = target.joinpath(*p.parts)
            dst.parent.mkdir(exist_ok=True, parents=True)
            with dst.open("xb") as file:
                file.write(data)
    need(
        sha(target / "bank/bank.json") == index["bank_sha256"]
        and sha(target / "model.json") == index["model_sha256"],
        "wrong example bank or model",
    )
    return index


def installed(a):
    import importlib.metadata as md
    import importlib.util
    import sysconfig

    wheels = sorted(Path("/wheels").glob("*.whl"))
    packages = {}
    for dist in ("evidence-lens", "evidence-lens-many-source", "numpy", "pillow"):
        d = md.distribution(dist)
        need(
            Path(d.locate_file("")).resolve().is_relative_to(Path(sys.prefix).resolve()),
            "package outside fresh venv",
        )
        packages[dist] = d.version
    need(
        packages
        == {
            "evidence-lens": "0.5.0",
            "evidence-lens-many-source": "0.1.0.dev0",
            "numpy": "2.5.5",
            "pillow": "12.3.0",
        },
        "different dependency group",
    )
    need(
        importlib.util.find_spec("torch") is None
        and importlib.util.find_spec("evidence_lens_many.evaluation") is not None,
        "incorrect base installation",
    )
    checked = {}
    for wheel in (x for x in wheels if x.name.startswith("evidence_lens")):
        with ZipFile(wheel) as z:
            names = [k for k in z.namelist() if not ".dist-info/" in k and not k.endswith("/")]
            root = Path(sysconfig.get_path("purelib"))
            need(
                all((root / n).is_file() and (root / n).read_bytes() == z.read(n) for n in names),
                "installed product differs from wheel",
            )
            checked[wheel.name] = len(names)
    interfaces = sorted(p.name for p in Path("/sys/class/net").iterdir())
    need(
        interfaces == ["lo"]
        and platform.system() == "Linux"
        and platform.machine() in ("aarch64", "arm64"),
        "wrong OS or unexpected external interface",
    )
    return {
        "os": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "kernel": platform.release(),
        "interfaces": interfaces,
        "packages": packages,
        "verified_product_files": checked,
        "wheelhouse": {x.name: {"sha256": sha(x), "bytes": x.stat().st_size} for x in wheels},
        "base_cpu_torch_present": False,
    }


def native_worker(a):
    expected = read("/inputs/expected.json")
    need(sha("/inputs/example.zip") == expected["archive_sha256"], "different public example archive")
    need(
        {x.name: sha(x) for x in Path("/wheels").glob("*.whl")} == expected["wheels"],
        "wrong wheelhouse bytes",
    )
    case = Path("/tmp/case")
    index = unpack("/inputs/example.zip", case)
    need(
        index["bank_sha256"] == expected["bank_sha256"]
        and index["model_sha256"] == expected["model_sha256"],
        "portable example and final science weights differ",
    )
    venv = Path("/tmp/clean")
    subprocess.run([sys.executable, "-I", "-m", "venv", venv], check=True)
    py = venv / "bin/python"
    cli = venv / "bin/evidence-lens-bank"
    initial = json.loads(
        command(
            [
                py,
                "-I",
                "-c",
                'import json,importlib.metadata as d; print(json.dumps(sorted(x.metadata["Name"] for x in d.distributions())))',
            ]
        )
    )
    need(
        all(
            n.lower() not in ("evidence-lens", "evidence-lens-many-source", "numpy", "pillow")
            for n in initial
        ),
        "venv was not clean",
    )
    args = [
        py,
        "-I",
        "-m",
        "pip",
        "--disable-pip-version-check",
        "install",
        "--no-index",
        "--no-cache-dir",
        "--no-deps",
        *sorted(Path("/wheels").glob("*.whl")),
    ]
    command(args)
    checklog = command([py, "-I", "-m", "pip", "check"])
    install = json.loads(command([py, "-I", "-B", __file__, "check-installed"]))
    need(
        {n: r["sha256"] for n, r in install["wheelhouse"].items()} == expected["wheels"],
        "installed environment source changed",
    )
    install.update(
        initial_distributions=initial,
        pip_check=checklog,
        run_id=os.environ["EL_RUN_ID"],
        cpu_thread_environment={
            k: os.environ[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    )
    a.out.mkdir(exist_ok=True, parents=True)
    save(a.out / "installation.json", install)
    normal = [cli, "inspect", "--bank", case / "bank", "--bank-sha256", expected["bank_sha256"]]
    inspect = json.loads(command(normal))
    save(a.out / "inspect.json", inspect)
    need(
        inspect["scene_id"] == "b04-layered-louvers"
        and len(inspect["unique_accessible_ids"]) == 16
        and len(inspect["target_camera_ids"]) == 6
        and not inspect["unavailable_source_ids"]
        and not inspect["missing_target_prediction_ids"],
        "different authored development scene",
    )
    outputs = {}
    for act in ACTIONS:
        path = a.out / act
        command(
            [
                cli,
                act,
                "--bank",
                case / "bank",
                "--bank-sha256",
                expected["bank_sha256"],
                "--target",
                "h00",
                "--strategy",
                "coverage",
                "--budget",
                "4",
                "--model",
                case / "model.json",
                "--model-sha256",
                expected["model_sha256"],
                "--out",
                path,
            ]
        )
        replayed = path / "report-regenerated.html"
        report = json.loads(command([cli, "report", "--scores", path, "--out", replayed]))
        doc = read(path / "scores.json")
        need(
            sha(replayed) == report["report_sha256"] == sha(path / "report.html")
            and report["score_receipt_sha256"] == sha(path / "scores.json"),
            "different regenerated offline report",
        )
        need(
            doc["model"]["model_sha256"] == expected["model_sha256"]
            and doc["bank_sha256"] == expected["bank_sha256"]
            and doc["arrays"]["sha256"] == sha(path / "scores.npz")
            and doc["query_cost"]["source_photos_decoded"] == 4,
            "different model, bank or finite query",
        )
        need(doc["audit"]["requested"] == (act == "audit-bank"), "incorrect query scope")
        if act == "audit-bank":
            need(
                doc["audit"]["full_complete"]
                and doc["audit"]["full_scope_cost"]["source_photos_decoded"] == 16,
                "incomplete library audit",
            )
        outputs[act] = {
            "query": doc["queried_source_ids"],
            "scores_json_sha256": sha(path / "scores.json"),
            "scores_npz_sha256": sha(path / "scores.npz"),
            "original_and_regenerated_html_sha256": sha(replayed),
            "rank_eligible_pixels": doc["rank_eligible_pixels"],
            "canvas_pixels": doc["canvas_pixels"],
            "reason_counts": doc["reason_counts"],
            "scope_counts": doc["scope_counts"],
        }
    save(
        a.out / "linux.json",
        {
            "schema": "el-b-many-clean-linux-1",
            "run_id": os.environ["EL_RUN_ID"],
            "image_id": expected["image_id"],
            "archive_sha256": expected["archive_sha256"],
            "bank_sha256": expected["bank_sha256"],
            "model_sha256": expected["model_sha256"],
            "inputs_role": "portable deployment bank, final head, wheels and validator",
            "network": "none",
            "root_read_only": True,
            "installation_sha256": sha(a.out / "installation.json"),
            "inspection_sha256": sha(a.out / "inspect.json"),
            "native": install,
            "results": outputs,
        },
    )


def container(a):
    cfg = read(a.config)
    image = cfg["image_id"]
    spec = json.loads(command(["docker", "image", "inspect", image]))[0]
    need(
        spec["Id"] == image and spec["Architecture"] == "arm64" and spec["Os"] == "linux",
        "local Linux image differs",
    )
    roles = cfg["sources"]
    example = read(a.example)
    need(
        example["archive"]["sha256"] == roles["archive"]["sha256"]
        and example["head_sha256"] == roles["model"]["sha256"],
        "registered authored package or fitted head differs",
    )
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        wheels = t / "wheels"
        wheels.mkdir()
        debug = {}
        for k, p in a.wheels:
            p = Path(p)
            need(sha(p) == roles[k]["sha256"], "different wheel")
            shutil.copyfile(p, wheels / p.name)
            debug[p.name] = roles[k]["sha256"]
        expected = {
            "wheels": debug,
            "bank_sha256": example["bank_sha256"],
            "model_sha256": example["head_sha256"],
            "archive_sha256": example["archive"]["sha256"],
            "image_id": image,
        }
        save(t / "expected.json", expected)
        cmd = [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,exec,size=2g",
            "-e",
            "EL_RUN_ID=" + os.environ["EL_RUN_ID"],
            "-e",
            "PIP_NO_INDEX=1",
            "-e",
            "PIP_DISABLE_PIP_VERSION_CHECK=1",
            "-e",
            "OMP_NUM_THREADS=1",
            "-e",
            "OPENBLAS_NUM_THREADS=1",
            "-e",
            "MKL_NUM_THREADS=1",
        ]
        for src, dst, mode in (
            (a.out, "/out", "rw"),
            (Path(__file__), "/app/worker.py", "ro"),
            (a.archive, "/inputs/example.zip", "ro"),
            (t / "expected.json", "/inputs/expected.json", "ro"),
            (wheels, "/wheels", "ro"),
        ):
            cmd.extend(["-v", f"{src.resolve()}:{dst}:{mode}"])
        subprocess.run(
            [*cmd, image, "python", "-I", "-B", "/app/worker.py", "native", "--out", "/out"], check=True
        )
    result = read(a.out / "linux.json")
    need(
        sha(a.out / "installation.json") == result["installation_sha256"]
        and set(result["results"]) == set(ACTIONS),
        "incomplete native clean Linux replay",
    )


def comparison(a):
    import numpy as np

    pins = read(a.config)["sources"]
    linux = read(a.linux)
    mac = read(a.example)
    need(
        linux["archive_sha256"] == mac["archive"]["sha256"]
        and linux["bank_sha256"] == mac["bank_sha256"]
        and linux["model_sha256"] == mac["head_sha256"],
        "different deployment inputs",
    )
    reports = {}
    for i, act in enumerate(ACTIONS):
        macmeta, linmeta = read(a.mac_meta[i]), read(a.lin_meta[i])
        ma, la = a.mac_array[i], a.lin_array[i]
        identity = (
            "schema",
            "baseline_id",
            "bank_sha256",
            "scene_id",
            "checkpoint_sha256",
            "renderer_id",
            "analysis_profile",
            "analysis_wh",
            "target_id",
            "target_prediction",
            "target_camera_sha256",
            "frame_receipt",
            "metric_anchor_ids",
            "reconstruction_ids",
            "accessible_ids",
            "canonical_source_ids",
            "exact_photo_aliases",
            "holdout_camera_ids",
            "queried_source_ids",
            "source_selection",
            "missing_queried_material",
            "feature_schema_sha256",
            "model",
            "permission",
            "scope_names",
            "reason_names",
        )
        identity_checks = {key: macmeta[key] == linmeta[key] for key in identity}
        need(
            macmeta["arrays"]["sha256"] == sha(ma) == mac["reports"][act]["scores_npz_sha256"]
            and linmeta["arrays"]["sha256"] == sha(la) == linux["results"][act]["scores_npz_sha256"],
            "array archive hash mismatch",
        )
        need(
            all(identity_checks[k] for k in ("bank_sha256", "target_id", "model", "target_prediction")),
            "different inputs to platform comparison",
        )
        fields = {}
        with np.load(ma, allow_pickle=False) as m, np.load(la, allow_pickle=False) as l:
            need(
                set(m) == set(l)
                and {"finite", "rank", "reason", "first_witness", "gap_witness", "scope_state"} <= set(m),
                "different score members",
            )
            for key in sorted(m):
                old, new = m[key], l[key]
                need(
                    old.dtype == new.dtype and old.shape == new.shape == (252, 378),
                    "incompatible numeric members",
                )
                if new.dtype.kind == "f":
                    different = old.view(np.uint32) != new.view(np.uint32)
                else:
                    different = old != new
                fields[key] = {
                    "dtype": new.dtype.str,
                    "different_values_or_float_words": int(different.sum()),
                    "pixels": int(new.size),
                }
            mr, lr = m["rank"], l["rank"]
            vf = np.isfinite(mr) & np.isfinite(lr)
            need(
                np.array_equal(np.isfinite(mr), m["finite"])
                and np.array_equal(np.isfinite(lr), l["finite"]),
                "finite mask disagrees with rank",
            )
            indices = np.flatnonzero(vf)
            delta = lr[vf].astype(np.float64) - mr[vf].astype(np.float64)
            quotas = {}
            for q in (0.1, 0.2):
                n = int(len(indices) * q)
                old = indices[np.lexsort((indices, mr.ravel()[indices]))[:n]]
                new = indices[np.lexsort((indices, lr.ravel()[indices]))[:n]]
                left, right = np.setdiff1d(old, new), np.setdiff1d(new, old)
                quotas[str(q)] = {
                    "common_finite": len(indices),
                    "accepted_each": n,
                    "mac_only": len(left),
                    "linux_only": len(right),
                    "mac_only_indices_sha256": hashlib.sha256(left.astype("<u4").tobytes()).hexdigest(),
                    "linux_only_indices_sha256": hashlib.sha256(right.astype("<u4").tobytes()).hexdigest(),
                }
            summary = {
                "common_finite": int(vf.sum()),
                "finite_mac": int(m["finite"].sum()),
                "finite_linux": int(l["finite"].sum()),
                "signed_linux_minus_mac_min": float(delta.min()) if len(delta) else None,
                "signed_linux_minus_mac_max": float(delta.max()) if len(delta) else None,
                "max_abs": float(np.max(np.abs(delta))) if len(delta) else None,
                "mean_abs": float(np.mean(np.abs(delta))) if len(delta) else None,
                "mean_signed": float(np.mean(delta)) if len(delta) else None,
                "eligible_quotas": quotas,
                "operating_threshold": None,
            }
        reports[act] = {
            "query_ids_linux": linmeta["queried_source_ids"],
            "query_ids_mac": macmeta["queried_source_ids"],
            "identity_fields_equal": identity_checks,
            "all_identity_fields_equal": all(identity_checks.values()),
            "all_array_words_equal": all(
                v["different_values_or_float_words"] == 0 for v in fields.values()
            ),
            "fields": fields,
            "rank": summary,
            "linux_report_sha256": linux["results"][act]["original_and_regenerated_html_sha256"],
            "mac_report_sha256": mac["reports"][act]["html_sha256"],
            "identical_html_bytes": linux["results"][act]["original_and_regenerated_html_sha256"]
            == mac["reports"][act]["html_sha256"],
        }
    save(
        a.out / "comparison.json",
        {
            "schema": "el-b-many-linux-mac-comparison-1",
            "run_id": os.environ["EL_RUN_ID"],
            "sources": pins,
            "case": "b04-layered-louvers/R16/h00",
            "pixel_count": 378 * 252,
            "reference_access": "only native deployment scores and metadata",
            "platforms": ["Mac arm64", "Linux arm64 container"],
            "results": reports,
        },
    )


def controller(a):
    from dual_records import bind, resolve, selected, stage

    s = a.store.resolve()
    roots = (s, s / "history")
    a_l = selected(s, "a-linux-installed-default")
    record = read(resolve(roots, a_l))
    pins = {
        "a_linux": a_l,
        "runtime": selected(s, "b-score-runtime"),
        "engineering": selected(s, "b-bank-engineering-protocol"),
    }
    for kind in ("numpy", "pillow"):
        name = next(n for n in record["wheelhouse"] if n.startswith(kind + "-"))
        pins[kind] = bind(s, a_l["run_id"], f"inputs/{kind}/{name}")
        need(pins[kind]["sha256"] == record["wheelhouse"][name], "different A offline dependency")
    for role, name in (
        ("prototype", "evidence_lens_many_source-0.1.0.dev0-py3-none-any.whl"),
        ("basewheel", "evidence_lens-0.5.0-py3-none-any.whl"),
    ):
        pins[role] = bind(s, pins["runtime"]["run_id"], f"inputs/{role}/{name}")
    info = {"protocol": PROBE, "image_id": record["image_id"], "program_sha256": sha(Path(__file__))}
    label, action = "b-bank-linux-replay-protocol", "registration-worker"
    program = ["{python}", "-B", "{source}/scripts/many_bank_linux.py"]
    cmd = ["--config", "{config}", "--out", "{out}"]
    if a.action == "register":
        return stage(
            s,
            label,
            "archive_verify",
            pins,
            info,
            program + [action, *cmd],
            "payload/protocol.json",
            18000,
            retry=a.retry,
        )
    pins["protocol"] = selected(s, label)
    registered = read(resolve(roots, pins["protocol"]))
    need(
        registered["program_sha256"] == info["program_sha256"] and registered["protocol"] == PROBE,
        "different registered Linux procedure",
    )
    pins["example"] = selected(s, "b-bank-public-example")
    ex = pins["example"]["run_id"]
    if a.action == "linux":
        pins["archive"] = bind(s, ex, "payload/evidence-lens-b04-r16-example.zip")
        pins["model"] = bind(s, selected(s, "b-head-final")["run_id"], "payload/model.json")
        cmd += ["--example", "{input:example}", "--archive", "{input:archive}"]
        for role in ("numpy", "pillow", "prototype", "basewheel"):
            cmd += ["--wheels", role, "{input:" + role + "}"]
        return stage(
            s,
            "b-bank-linux-clean-replay",
            "transfer_verify",
            pins,
            info,
            program + ["container", *cmd],
            "payload/linux.json",
            19000000,
            retry=a.retry,
        )
    pins["linux"] = selected(s, "b-bank-linux-clean-replay")
    cmd += ["--linux", "{input:linux}", "--example", "{input:example}"]
    for i, action in enumerate(ACTIONS):
        for tag, original in (("lin", pins["linux"]["run_id"]), ("mac", ex)):
            for role, name in (("meta", "scores.json"), ("array", "scores.npz")):
                field = tag + "_" + role + str(i)
                pins[field] = bind(s, original, f"payload/{action}/{name}")
                cmd += ["--" + tag + "-" + role, "{input:" + field + "}"]
    return stage(
        s,
        "b-bank-linux-mac-comparison",
        "archive_verify",
        pins,
        info,
        program + ["compare-worker", *cmd],
        "payload/comparison.json",
        80000,
        retry=a.retry,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=(
            "register",
            "linux",
            "compare",
            "registration-worker",
            "container",
            "native",
            "check-installed",
            "compare-worker",
        ),
    )
    for n in ("store", "config", "out", "archive", "example", "linux"):
        p.add_argument("--" + n, type=Path)
    p.add_argument("--wheels", action="append", nargs=2)
    for n in ("lin-meta", "lin-array", "mac-meta", "mac-array"):
        p.add_argument("--" + n, action="append", type=Path)
    p.add_argument("--retry")
    a = p.parse_args()
    if a.action == "registration-worker":
        d = read(a.config)
        need(
            d["program_sha256"] == sha(Path(__file__)) and d["protocol"] == PROBE,
            "different registered Linux program",
        )
        save(
            a.out / "protocol.json",
            {
                "schema": "el-b-many-linux-registration-1",
                "run_id": os.environ["EL_RUN_ID"],
                "parents": d["sources"],
                "image_id": d["image_id"],
                "program_sha256": d["program_sha256"],
                "protocol": PROBE,
            },
        )
    elif a.action == "native":
        native_worker(a)
    elif a.action == "container":
        container(a)
    elif a.action == "check-installed":
        print(json.dumps(installed(a)))
    elif a.action == "compare-worker":
        comparison(a)
    else:
        print(json.dumps(controller(a)))
