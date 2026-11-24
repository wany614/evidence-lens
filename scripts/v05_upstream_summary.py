"""Seal the portable, numerical upstream evidence for A from verified recorder members."""

import argparse
import json
import os
from pathlib import Path

from dual_records import selected, stage
from evidence_lens._bundle import load, save
from evidence_lens_bundle.common import require
from v05_bell_common import CHECKPOINT, HISTORICAL_MANIFEST, HISTORICAL_RUN, OLD, SIZES, SOURCES, TARGETS


STAGES = {
    "engineering_registration": "a-grid-registration",
    "bell_registration": "a-bell-register-six",
    "bell_packet": "a-bell-packet-six",
    "bell_native_readback": "a-bell-gpu-readback",
    "bell_original_sources": "a-bell-source-observations",
    "bell_bundles": "a-bell-bundle-publication",
    "source_archive": "a-bell-source-delivery",
    "bell_opening": "a-bell-reference-opening",
    "independent_photography": "a-bell-references",
    "reference_archive": "a-bell-reference-delivery",
    "gaussians": "a-upstream-gaussian-contract",
    "installed_upstream_tests": "a-upstream-wheel-tests",
    "projection_ticket": "a-recovery-render-control-ticket",
    "projection_agreement": "a-recovery-render-agreement",
}


def control(s):
    pins = {key: selected(s, label) for key, label in STAGES.items()}
    cmd = ["{python}", "-B", "{source}/scripts/v05_upstream_summary.py", "worker", "--config", "{config}", "--out", "{out}"]
    for key in pins:
        cmd += ["--input", key, "{input:" + key + "}"]
    return stage(s, "a-upstream-report", "report", pins, {"role": "public numbers and provenance"}, cmd, "payload/evidence.json", 28000)


def worker(cfg, paths, out):
    rec = {key: load(path) for key, path in paths.items()}
    pins = cfg["sources"]
    bundles, originals, refs, gaussian = (rec[key] for key in ("bell_bundles", "source_archive", "reference_archive", "gaussians"))
    capture, gate, parity, native = (rec[key] for key in ("independent_photography", "bell_opening", "projection_agreement", "bell_native_readback"))
    require(bundles["checkpoint_sha256"] == CHECKPOINT and tuple(bundles["sources"]) == tuple(SOURCES)
            and bundles["targets"] == TARGETS and set(bundles["routes"]) == {*SIZES, OLD}, "changed bell arms")
    require(bundles["readback"] == pins["bell_native_readback"] and originals["bundles"] == pins["bell_bundles"]
            and refs["opening"] == pins["bell_opening"] and refs["references"] == pins["independent_photography"]
            and gate["registered_scores_sealed"] and gate["terminal_score_count"] == 48, "broken scoring and measurement lineage")
    require(parity["inputs"]["dispatch"] == pins["projection_ticket"] and gaussian["historical_input"] == rec["projection_ticket"]["checkpoint"],
            "checkpoint acceptance and render comparison used different native optimizer inputs")
    routes = {name: {k: row[k] for k in ("path", "sha256", "active_xywh", "analysis_size", "source_observation_sha256")}
              for name, row in bundles["routes"].items()}
    save(out / "evidence.json", {
        "schema": "el-v05-public-upstream-evidence-1", "run_id": os.environ["EL_RUN_ID"], "parents": pins,
        "bell": {"object_id": "scene_057_offset_bell_cage", "historical_run_id": HISTORICAL_RUN,
                 "historical_manifest_sha256": HISTORICAL_MANIFEST, "checkpoint_sha256": CHECKPOINT,
                 "gaussians": native["gaussians"], "native_renderer": native["native"], "native_frames": native["native_frames"],
                 "source_order": SOURCES, "target_order": TARGETS, "routes": routes,
                 "published_bundles": bundles["archive"], "original_source_count": originals["original_count"],
                 "rebuilt_analysis_sources": originals["reconstructed_analysis_pngs"], "source_mismatches": originals["mismatches"],
                 "source_archive": originals["archive"], "reference_images_in_prepared_bundles": bundles["reference_images_in_archive"],
                 "terminal_scores_before_reference_opening": gate["terminal_score_count"],
                 "reference_renderer": {"blender": capture["bpy"], "device": capture["cycles_device"],
                                        "samples_per_pixel": capture["cycles_samples"], "seed": capture["seed"]},
                 "independent_reference_count": refs["photographs"], "reference_size": refs["size"],
                 "reference_archive": refs["archive"]},
        "recovery_asset": {key: gaussian[key] for key in ("profile", "actual_gaussians", "registered_cap", "tensor_shapes",
                            "malformed_opacities_rows", "malformed_rejection", "optional_failure_fixed_M4",
                            "optional_failure_optional_branch", "successful_optional_branch", "historical_input")},
        "installed_upstream_tests": {key: rec["installed_upstream_tests"][key] for key in
                                    ("wheel_sha256", "wheel_code_commit", "pass", "failures", "skips", "test_source_sha256", "junit_sha256")},
        "same_checkpoint_render": {key: parity[key] for key in ("passed", "source_view", "original_predicted_frame",
                                   "raw_and_activated_parameters", "native_render", "clip_m", "recipe", "predictions",
                                   "photographs_opened", "old_experiment_reruns")},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("record", "worker"))
    for key in ("store", "config", "out"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--input", action="append", nargs=2, default=[])
    args = parser.parse_args()
    if args.action == "worker":
        worker(load(args.config), {k: Path(v) for k, v in args.input}, args.out)
    else:
        print(json.dumps(control(args.store.resolve())))
