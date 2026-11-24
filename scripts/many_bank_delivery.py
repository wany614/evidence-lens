"""Recorded deployment measurements, external-format checks and the releasable authored example."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile


FAMILY = "b04-layered-louvers"
VIEW = "h00"
PROTOCOL = {
    "schema": "el-b-bank-engineering-protocol-1",
    "representative": "first accepted b04-layered-louvers R16 or R128, ordered by upstream bank recorder completion UTC, h00, coverage policy, K16",
    "prefit": "one four-thread finite-query cold/prepared/hot run with explicit residual rule, no fitted head",
    "fitted": "one fresh verified CPU process for each of thread ceilings 1, 2 and 4; actual recomputed B-CPU-LR on the same admitted K16 and unchanged target",
    "hot": "seven serial repetitions after one separately measured source preparation; report every observation, linear p50 and p95, throughput and process CPU seconds",
    "memory": "128 MiB source LRU; native process maximum resident size, current RSS at named boundaries, source cache retained peak, byte reads and actual decoding",
    "scale": "one b04 R128 h00 cold scan covering unique K4, K16 and full K128; report rich state size, actual 128 distinct source IDs and native high water",
    "external": "only delivery_area/R34, source members and target predictions; verify/import complete bank, remove one granted self-render from a private temporary copy, separately withhold expected-depth verification, inspect bare PLY and reject scoring it",
    "example": "b04-layered-louvers/R16/h00; real final B-CPU-LR, coverage Q4 alone and with explicit full audit; include finite and audited results, independent offline reports, permitted deployment bank, fitted head, rights and commands",
    "scope": "offline engineering costs; no confirmation RGB, labels, simulated rates or scientific model selection",
}


def need(check, message):
    if not check:
        raise ValueError(message)


def hashfile(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def peak_rss():
    import psutil

    if sys.platform == "win32":
        return int(psutil.Process().memory_info().peak_wset)
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else 1024 * value)


def observe(function):
    import psutil

    proc = psutil.Process()
    before = {t.id: t.user_time + t.system_time for t in proc.threads()}
    t = time.perf_counter()
    cpu = time.process_time()
    value = function()
    wall = time.perf_counter() - t
    core = time.process_time() - cpu
    after = proc.threads()
    return value, {
        "wall_seconds": wall,
        "cpu_core_seconds": core,
        "current_rss_bytes": proc.memory_info().rss,
        "native_peak_rss_bytes": peak_rss(),
        "process_thread_count_after": len(after),
        "threads_with_observed_cpu_time_increment": sum(
            t.user_time + t.system_time - before.get(t.id, 0) > 1e-5 for t in after
        ),
    }


def cpu_variables(n):
    return {
        key: str(n)
        for key in (
            "VECLIB_MAXIMUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }


def installation(cfg, build, runtime):
    import evidence_lens_many as package
    from evidence_lens_many import FEATURE_SCHEMA_SHA256

    doc = load(runtime)
    expected = load(build)
    need(
        doc["prototype_wheel_sha256"] == expected["sha256"]
        and doc["feature_schema_sha256"] == FEATURE_SCHEMA_SHA256,
        "different verified deployment environment or feature schema",
    )
    dist = importlib.metadata.distribution("evidence-lens-many-source")
    need(
        Path(dist.locate_file("evidence_lens_many/__init__.py")).resolve()
        == Path(package.__file__).resolve(),
        "benchmark must use installed wheel",
    )
    for name, h in expected["source_members"].items():
        need(hashfile(dist.locate_file(name)) == h, "installed member changed: " + name)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_version": dist.version,
        "base_wheel_sha256": doc["base_wheel_sha256"],
        "prototype_wheel_sha256": expected["sha256"],
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "runtime": cfg["sources"]["runtime"],
    }


def performance_child(args):
    import numpy as np
    import psutil
    from evidence_lens_many import Bank, score_lr, select_sources, simple_ranks, stream_features
    from evidence_lens_bundle.bindings import array_sha

    cfg = load(args.config)
    pins = cfg["sources"]
    setup = installation(cfg, args.build, args.runtime)
    need(
        all(os.environ.get(k) == v for k, v in cpu_variables(args.threads).items()),
        "thread ceilings were not set before interpreter start",
    )
    n = cfg["source_count"]
    capacity = 128 << 20

    def new_bank():
        bank = Bank(args.bank, pins["bank"]["sha256"], cache_bytes=capacity)
        need(
            bank.meta["scene_id"] == FAMILY and len(bank.ids) == n and VIEW in bank.target_ids,
            "wrong registered workload",
        )
        return bank

    bank, admit = observe(new_bank)
    plan, selection = observe(lambda: select_sources(bank, VIEW))
    need(
        plan.cost["source_photos_decoded"] == plan.cost["source_renders_decoded"] == 0,
        "selector read source photos or depth",
    )
    model = None
    if args.model:
        need(hashfile(args.model) == pins["model"]["sha256"], "different final head")
        model = load(args.model)

    def process(step):
        return score_lr(step, model) if model is not None else simple_ranks(step)["residual"]

    def single(bank):
        step = next(stream_features(bank, VIEW, plan, budgets=(16,)))
        rank, cost = observe(lambda: process(step))
        return {
            "rank_sha256": array_sha(rank),
            "finite": int(np.isfinite(rank).sum()),
            "queried_source_ids": step.queried_source_ids,
            "scanning": step.cost,
            "head_or_rule": cost,
        }

    result = {
        "mode": args.mode,
        "requested_thread_ceiling": args.threads,
        "thread_environment": cpu_variables(args.threads),
        "numpy_blas": np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {}).get("name"),
        "registered_bank": pins["bank"],
        "source_count": n,
        "target": VIEW,
        "model": "B-CPU-LR" if model is not None else "rule_residual",
        "model_sha256": pins["model"]["sha256"] if model is not None else None,
        "source_LRU_bytes": capacity,
        "initial_bank_admission": admit,
        "geometry_selection": selection,
        "selection_details": plan.cost,
        "installed": setup,
    }
    if args.mode == "scale":
        history = []

        def scale():
            for step in stream_features(bank, VIEW, plan, budgets=(4, 16, "all")):
                rank, cost = observe(lambda: process(step))
                history.append(
                    {
                        "actual_k": step.actual_k,
                        "queries": step.queried_source_ids,
                        "nominal": step.nominal_budgets,
                        "rank_sha256": array_sha(rank),
                        "head_or_rule": cost,
                        "scan": step.cost,
                        "native_peak_rss_bytes": peak_rss(),
                    }
                )

        _, observed = observe(scale)
        need(
            [r["actual_k"] for r in history] == [4, 16, 128]
            and history[-1]["scan"]["source_photos_decoded"] == 128
            and history[-1]["scan"]["source_renders_decoded"] == 128
            and len(set(history[-1]["queries"])) == 128,
            "full scale test did not decode exactly 128 unique observations",
        )
        result.update(full_scan=history, scale_cost=observed)
    else:
        cold, cold_cost = observe(lambda: single(bank))
        result["spawn_to_cold_result_seconds"] = (time.monotonic_ns() - args.launch_ns) * 1e-9
        need(
            cold["scanning"]["source_renders_decoded"] == cold["scanning"]["source_photos_decoded"] == 16
            and len(set(cold["queried_source_ids"])) == 16,
            "cold finite query accessed different sources",
        )
        del bank
        gc.collect()
        bank, second_admission = observe(new_bank)

        def populate():
            for source in plan.ids[:16]:
                need(bank.source(source) is not None, "missing admitted source")

        _, prepare = observe(populate)
        need(bank.counters["source_photos_decoded"] == 16, "source preparation did not decode exactly Q16")
        preparation_counters = dict(bank.counters)
        observations = []
        for _ in range(7):
            sample, cost = observe(lambda: single(bank))
            need(
                sample["scanning"]["source_photos_decoded"]
                == sample["scanning"]["source_renders_decoded"]
                == 0
                and sample["scanning"]["cache_hits"] == 16
                and sample["rank_sha256"] == cold["rank_sha256"],
                "warm run re-read sources or changed its own cold ranking",
            )
            observations.append({**cost, **sample})
        times = [x["wall_seconds"] for x in observations]
        result.update(
            cold=cold,
            cold_cost=cold_cost,
            second_bank_admission=second_admission,
            first_source_preparation=prepare,
            source_preparation_counters=preparation_counters,
            hot=observations,
            hot_service_p50_s=float(np.percentile(times, 50, method="linear")),
            hot_service_p95_s=float(np.percentile(times, 95, method="linear")),
            hot_targets_per_second=len(times) / sum(times),
            hot_cpu_core_seconds=sum(x["cpu_core_seconds"] for x in observations),
        )
    result.update(
        cache=bank.counters,
        native_peak_rss_bytes=peak_rss(),
        current_rss_bytes=psutil.Process().memory_info().rss,
        process_cpu_seconds=time.process_time(),
        rss_budget_met=peak_rss() <= 2 << 30,
        lru_budget_met=bank.counters["cache_peak_bytes"] <= 512 << 20,
    )
    save(args.out, result)


def run_children(a, cfg, mode):
    threads = (1, 2, 4) if mode == "fitted" else (4,)
    runs = []
    for n in threads:
        dest = a.out / f"{n}-threads.json"
        argv = [
            sys.executable,
            "-I",
            "-B",
            str(Path(__file__).resolve()),
            "performance-child",
            "--mode",
            mode,
            "--threads",
            str(n),
            "--bank",
            str(a.bank),
            "--config",
            str(a.config),
            "--runtime",
            str(a.runtime),
            "--build",
            str(a.build),
            "--out",
            str(dest),
        ]
        if a.model:
            argv += ["--model", str(a.model)]
        argv += ["--launch-ns", str(time.monotonic_ns())]
        subprocess.run(argv, env={**os.environ, **cpu_variables(n)}, check=True)
        runs.append(load(dest))
    save(
        a.out / "performance.json",
        {
            "schema": "el-b-measured-bank-performance-1",
            "run_id": os.environ["EL_RUN_ID"],
            "protocol": cfg["sources"]["protocol"],
            "source_bank": cfg["sources"]["bank"],
            "mode": mode,
            "actual_case": FAMILY + f"/R{cfg['source_count']}/" + VIEW,
            "runs": runs,
            "all_rss_within_budget": all(r["rss_budget_met"] for r in runs),
            "all_source_caches_within_budget": all(r["lru_budget_met"] for r in runs),
        },
    )


def cli(*args, expected=0):
    command = [sys.executable, "-I", "-B", "-m", "evidence_lens_many.cli", *[str(a) for a in args]]
    t = time.perf_counter()
    p = subprocess.run(command, capture_output=True, text=True)
    need(
        p.returncode == expected, "unexpected deployed CLI exit " + str(p.returncode) + ": " + p.stderr[-450:]
    )
    return (json.loads(p.stdout) if expected == 0 else p.stderr.strip()), time.perf_counter() - t


def external(a, cfg):
    from evidence_lens_many import Bank

    pins = cfg["sources"]
    installation(cfg, a.build, a.runtime)
    bank = Bank(a.bank, pins["bank"]["sha256"])
    need(
        bank.meta["scene_id"] == "delivery_area" and len(bank.ids) == 34 and len(bank.target_ids) == 10,
        "registered external development bank required",
    )
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        copy = work / "imported"
        imported, tcopy = cli("import", "--bank", a.bank, "--bank-sha256", bank.sha256, "--out", copy)
        inspected, tall = cli("inspect", "--bank", copy, "--verify-members")
        need(
            imported["bank_sha256"] == inspected["bank_sha256"] == bank.sha256
            and not inspected["unavailable_source_ids"]
            and not inspected["missing_target_prediction_ids"]
            and inspected["verified_member_access"]["source_photos_decoded"] == 34
            and inspected["verified_member_access"]["source_renders_decoded"] == 34
            and inspected["verified_member_access"]["target_rgb_decoded"] == 10,
            "positive import did not verify every admitted member",
        )
        _, tscore = cli(
            "score",
            "--bank",
            copy,
            "--target",
            VIEW,
            "--budget",
            "4",
            "--rule",
            "residual",
            "--out",
            work / "positive",
        )
        result = load(work / "positive/scores.json")
        selected = result["queried_source_ids"]
        need(
            len(selected) == 4
            and result["query_cost"]["source_photos_decoded"] == 4
            and not result["audit"]["requested"],
            "positive finite scoring read outside Q4",
        )
        missing = selected[0]
        source = bank.meta["sources"][missing]["render"]
        (copy / source["path"]).unlink()
        partial, _ = cli("inspect", "--bank", copy)
        need(partial["unavailable_source_ids"] == [missing], "missing real source render was not discovered")
        _, taudit = cli(
            "audit-bank",
            "--bank",
            copy,
            "--target",
            VIEW,
            "--budget",
            "4",
            "--rule",
            "residual",
            "--out",
            work / "partial",
        )
        audit = load(work / "partial/scores.json")
        need(
            not audit["audit"]["full_complete"]
            and missing in audit["missing_queried_material"]
            and audit["scope_counts"]["audited_library_no_comparable_evidence"]
            == audit["scope_counts"]["subset_comparability_gap"]
            == 0,
            "incomplete library generated a complete-library conclusion",
        )
        no_depth = load(copy / "bank.json")
        no_depth["depth_verified"] = False
        no_depth["depth_probe_sha256"] = None
        (copy / "bank.json").unlink()
        save(copy / "bank.json", no_depth)
        rejection, _ = cli(
            "score",
            "--bank",
            copy,
            "--target",
            VIEW,
            "--budget",
            "4",
            "--rule",
            "coverage",
            "--out",
            work / "missing-depth",
            expected=2,
        )
        need(
            "verified expected camera-Z" in rejection and not (work / "missing-depth").exists(),
            "depth-incomplete external input produced full risk",
        )
        bare = work / "structure.ply"
        bare.write_text(
            "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n",
            encoding="ascii",
        )
        structure, _ = cli("inspect", "--bank", bare)
        ply_error, _ = cli(
            "score",
            "--bank",
            bare,
            "--target",
            VIEW,
            "--rule",
            "coverage",
            "--out",
            work / "bare-score",
            expected=2,
        )
        need(
            structure["status"] == "structural_inspection_only" and not (work / "bare-score").exists(),
            "bare PLY was admitted for scoring",
        )
        public = {
            "schema": "el-b-external-bank-acceptance-1",
            "run_id": os.environ["EL_RUN_ID"],
            "parents": pins,
            "external_scene": "delivery_area",
            "source_count": 34,
            "target_prediction_count": 10,
            "unit": bank.meta["frame_receipt"]["unit"],
            "verified_bank_sha256": bank.sha256,
            "checkpoint_sha256": bank.meta["checkpoint_sha256"],
            "positive": {
                "import_wall_s": tcopy,
                "verify_all_wall_s": tall,
                "verify_all_cost": inspected["verified_member_access"],
                "score_q4_wall_s": tscore,
                "actual_q": selected,
                "target": VIEW,
                "rule": "residual",
                "finite_pixels": result["rank_eligible_pixels"],
                "canvas_pixels": result["canvas_pixels"],
                "score_arrays_sha256": result["arrays"]["sha256"],
                "query_cost": result["query_cost"],
            },
            "incomplete": {
                "withheld_source": missing,
                "original_self_render_sha256": source["sha256"],
                "full_complete": audit["audit"]["full_complete"],
                "scope_counts": audit["scope_counts"],
                "audit_cost": audit["audit"]["full_scope_cost"],
                "audit_wall_s": taudit,
                "missing_depth_scoring_status": 2,
                "missing_depth_message": rejection,
            },
            "bare_ply": {
                "role": "generated one-vertex structure-only negative control",
                "fixture_sha256": hashfile(bare),
                "inspection": structure,
                "score_exit": 2,
                "score_error": ply_error,
            },
            "public_material": "counts, source IDs and hashes; the temporary external photos, predictions and graphical reports are private",
        }
        save(a.out / "external.json", public)


def add_zip(z, src, name):
    info = zipfile.ZipInfo(name, (2000, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    with Path(src).open("rb") as source, z.open(info, "w") as dst:
        shutil.copyfileobj(source, dst, 1 << 20)


def example(a, cfg):
    import numpy as np
    from evidence_lens_many import Bank

    setup = installation(cfg, a.build, a.runtime)
    pins = cfg["sources"]
    bank = Bank(a.bank, pins["bank"]["sha256"])
    need(
        bank.meta["scene_id"] == FAMILY and len(bank.ids) == 16 and len(bank.target_ids) == 6,
        "registered authored development example required",
    )
    need(
        hashfile(a.model) == pins["model"]["sha256"] and load(a.model)["schema"] == "el-b-cpu-lr-1",
        "missing final fitted head",
    )
    with tempfile.TemporaryDirectory() as temp:
        package = Path(temp) / "example"
        package.mkdir()
        cli("import", "--bank", a.bank, "--bank-sha256", bank.sha256, "--out", package / "bank")
        shutil.copyfile(a.model, package / "model.json")
        reports = {}
        for action in ("score", "audit-bank"):
            dest = a.out / action
            _, wall = cli(
                action,
                "--bank",
                package / "bank",
                "--bank-sha256",
                bank.sha256,
                "--target",
                VIEW,
                "--budget",
                "4",
                "--strategy",
                "coverage",
                "--model",
                package / "model.json",
                "--model-sha256",
                pins["model"]["sha256"],
                "--out",
                dest,
            )
            r = load(dest / "scores.json")
            need(
                r["model"]["name"] == "B-CPU-LR"
                and r["model"]["model_sha256"] == pins["model"]["sha256"]
                and r["canvas_pixels"] == 378 * 252
                and r["query_cost"]["source_photos_decoded"] == 4,
                "different actual learned model or source admission",
            )
            need(r["audit"]["requested"] == (action == "audit-bank"), "unexpected library audit scope")
            if action == "audit-bank":
                need(
                    r["audit"]["full_complete"]
                    and r["audit"]["full_scope_cost"]["source_photos_decoded"] == 16,
                    "public example must name the actual full-library cost",
                )
            with np.load(dest / "scores.npz", allow_pickle=False) as arr:
                gap = arr["gap_witness"]
                good = gap >= 0
                witnesses = {
                    bank.ids[int(k)]: int(v) for k, v in zip(*np.unique(gap[good], return_counts=True))
                }
            reports[action] = {
                "score_receipt_sha256": hashfile(dest / "scores.json"),
                "html_sha256": hashfile(dest / "report.html"),
                "scores_npz_sha256": r["arrays"]["sha256"],
                "actual_q": r["queried_source_ids"],
                "finite_pixels": r["rank_eligible_pixels"],
                "scope_counts": r["scope_counts"],
                "gap_witness_counts_by_source": witnesses,
                "wall_s_including_launch_and_html": wall,
            }
            data = (dest / "report.html").read_text()
            need(
                "data:image/png;base64," in data
                and "first comparable source" in data
                and "<script src=" not in data,
                "incomplete local report",
            )
        with np.load(a.out / "score/scores.npz") as s, np.load(a.out / "audit-bank/scores.npz") as u:
            need(
                np.array_equal(s["rank"].view(np.uint32), u["rank"].view(np.uint32))
                and np.array_equal(s["finite"], u["finite"]),
                "full audit changed finite-query ranking",
            )
        acquisition = bank.meta["permission"]
        text = (
            "# Authored many-source example\n\n"
            "This archive contains the registered b04-layered-louvers R16 development bank and trained B-CPU-LR weights. "
            "Scoring uses the original four selected source photographs and predictions. Full audit scans the sixteen authorized sources. "
            "Six held-out target camera IDs have deployment predictions. Independent target photographs are kept by the evaluator.\n\n"
            "Install Evidence Lens 0.5.0 and the accompanying evidence_lens_many_source-0.1.0.dev0 wheel. Run these commands from the extracted directory.\n\n"
            "```sh\nevidence-lens-bank inspect --bank bank\nevidence-lens-bank score --bank bank --target h00 --budget 4 --model model.json --out finite\n"
            "evidence-lens-bank audit-bank --bank bank --target h00 --budget 4 --model model.json --out audited\n```\n\n"
            "Open finite/report.html or audited/report.html. Higher model ranks indicate greater estimated RGB error. "
            "Click a target pixel to inspect its actual source witnesses.\n\n"
            "Original authored scene and derived material by wany614, Evidence Lens, under Creative Commons Attribution 4.0. "
            "https://creativecommons.org/licenses/by/4.0/\n"
        )
        (package / "README.md").write_text(text, encoding="utf-8")
        files = sorted(x for x in package.rglob("*") if x.is_file())
        index = [
            {"path": p.relative_to(package).as_posix(), "bytes": p.stat().st_size, "sha256": hashfile(p)}
            for p in files
        ]
        save(
            package / "index.json",
            {
                "schema": "el-b-public-bank-example-1",
                "bank_sha256": bank.sha256,
                "model_sha256": pins["model"]["sha256"],
                "license": acquisition,
                "files": index,
            },
        )
        zip_path = a.out / "evidence-lens-b04-r16-example.zip"
        with zipfile.ZipFile(zip_path, "x") as z:
            for p in [*files, package / "index.json"]:
                add_zip(z, p, p.relative_to(package).as_posix())
        with zipfile.ZipFile(zip_path) as z:
            need(
                len(z.namelist()) == len(index) + 1 and z.testzip() is None,
                "authored example archive failed reread",
            )
        save(
            a.out / "example.json",
            {
                "schema": "el-b-public-bank-example-receipt-1",
                "run_id": os.environ["EL_RUN_ID"],
                "parents": pins,
                "scene_id": FAMILY,
                "R": 16,
                "A": 16,
                "target": VIEW,
                "actual_q": reports["score"]["actual_q"],
                "bank_sha256": bank.sha256,
                "head_sha256": pins["model"]["sha256"],
                "installed": setup,
                "reports": reports,
                "archive": {
                    "path": zip_path.name,
                    "bytes": zip_path.stat().st_size,
                    "sha256": hashfile(zip_path),
                },
                "source_terms": acquisition,
                "separate_audit_same_rank_words": True,
            },
        )


def controller(args):
    from dual_records import bind, resolve, selected, stage

    store = args.store.resolve()
    roots = (store, store / "history")
    pins = {
        "study": selected(store, "b-study-registration"),
        "baseline": selected(store, "a-baseline"),
        "runtime": selected(store, "b-score-runtime"),
        "build": selected(store, "b-score-core-build"),
    }
    runtime = load(resolve(roots, pins["runtime"]))
    exe = store / runtime["relative_executable"]
    need(exe.is_file(), "verified installed runtime is missing")
    if args.action == "register":
        return stage(
            store,
            "b-bank-engineering-protocol",
            "archive_verify",
            pins,
            {"protocol": PROTOCOL, "script_sha256": hashfile(Path(__file__))},
            [
                "{python}",
                "-B",
                "{source}/scripts/many_bank_delivery.py",
                "registration-worker",
                "--config",
                "{config}",
                "--out",
                "{out}",
            ],
            "payload/protocol.json",
            13_000,
            retry=args.retry,
        )
    pins["protocol"] = selected(store, "b-bank-engineering-protocol")
    protocol = load(resolve(roots, pins["protocol"]))
    need(
        protocol["protocol"] == PROTOCOL and protocol["script_sha256"] == hashfile(Path(__file__)),
        "different engineering measurement program",
    )
    needs_model = args.action in ("fitted", "example")
    if needs_model:
        m = selected(store, "b-head-final")
        pins["model"] = bind(store, m["run_id"], "payload/model.json")
    if args.action == "prefit":
        found = []
        for n in (16, 128):
            name = f"b-bank-{FAMILY}-r{n}"
            if (store / "work/controller" / name / "selected.json").is_file():
                candidate = selected(store, name)
                found.append(
                    (load(store / "runs" / candidate["run_id"] / "run.json")["ended_at_utc"], n, candidate)
                )
        need(bool(found), "no authorized development bank has completed")
        _, count, pins["bank"] = min(found)
    elif args.action == "fitted":
        first = selected(store, "b-bank-performance-prefit")
        pins["prefit"] = first
        d = load(resolve(roots, first))
        count = d["runs"][0]["source_count"]
        pins["bank"] = d["source_bank"]
    else:
        count = {"scale": 128, "external": 34, "example": 16}[args.action]
        family = "delivery-area" if args.action == "external" else FAMILY
        pins["bank"] = selected(store, f"b-bank-{family}-r{count}")
    bank = resolve(roots, pins["bank"]).parent
    cmd = [
        str(exe),
        "-I",
        "-B",
        "{source}/scripts/many_bank_delivery.py",
        "worker",
        "--mode",
        args.action,
        "--bank",
        str(bank),
        "--config",
        "{config}",
        "--runtime",
        "{input:runtime}",
        "--build",
        "{input:build}",
        "--out",
        "{out}",
    ]
    if needs_model:
        cmd += ["--model", "{input:model}"]
    if args.action in ("fitted", "prefit", "scale"):
        label, member, kind, estimate = (
            "b-bank-performance-" + args.action,
            "performance.json",
            "model_score",
            280000,
        )
    elif args.action == "external":
        label, member, kind, estimate = "b-bank-external-acceptance", "external.json", "archive_verify", 30000
    else:
        label, member, kind, estimate = "b-bank-public-example", "example.json", "report", 150_000_000
    return stage(
        store,
        label,
        kind,
        pins,
        {
            "source_count": count,
            "mode": args.action,
            "registered_target": VIEW,
            "reference_access": "no evaluator inputs",
        },
        cmd,
        "payload/" + member,
        estimate,
        retry=args.retry,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=(
            "register",
            "prefit",
            "fitted",
            "scale",
            "external",
            "example",
            "registration-worker",
            "worker",
            "performance-child",
        ),
    )
    for key in ("store", "bank", "out", "config", "model", "build", "runtime"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--retry")
    p.add_argument("--mode")
    p.add_argument("--threads", type=int)
    p.add_argument("--launch-ns", type=int)
    a = p.parse_args()
    if a.action == "registration-worker":
        cfg = load(a.config)
        need(
            cfg["protocol"] == PROTOCOL and cfg["script_sha256"] == hashfile(Path(__file__)),
            "different registered measurement program",
        )
        save(
            a.out / "protocol.json",
            {
                "schema": "el-b-bank-engineering-registration-1",
                "run_id": os.environ["EL_RUN_ID"],
                "protocol": PROTOCOL,
                "parents": cfg["sources"],
                "script_sha256": hashfile(Path(__file__)),
            },
        )
    elif a.action == "performance-child":
        performance_child(a)
    elif a.action == "worker":
        cfg = load(a.config)
        if a.mode == "external":
            external(a, cfg)
        elif a.mode == "example":
            example(a, cfg)
        else:
            run_children(a, cfg, a.mode)
    else:
        print(json.dumps(controller(a)))
