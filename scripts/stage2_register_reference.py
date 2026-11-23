"""Materialize preregistered scene recipes without invoking Blender or opening scientific test data."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from b_scene_designs import all_designs
from stage2_reference_catalog import (
    BASE,
    CAMERA,
    CATALOG,
    ENGINE,
    LOCK,
    NOVEL,
    OLD_CATALOG,
    OLD_DESIGNS,
    STUDY,
    TARGETS,
    digest,
    geometry_fingerprint,
    pair_structure,
    read,
    require,
    topology,
    value_hash,
)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def register(project, assets):
    time = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    authored = all_designs()
    base = read(project / BASE)["reference"]
    old = read(project / OLD_DESIGNS)
    oldc = read(project / OLD_CATALOG)
    old_rows = oldc["families"]
    world = old["shared_world"]
    past_structure = {}
    frozen_materials = base["materials"]
    for r in old_rows:
        family = r["scene_family_id"]
        if r["ordinal"] == 1:
            fg = [o for o in base["objects"] if o["id"] >= 10]
        else:
            fg = old["families"][family]["objects"]
        past_structure[family] = pair_structure(fg)
        require(
            not topology(fg),
            "A reserved topology occurs in an earlier family: " + family,
        )
    guard = {
        "schema": "el-vision-reference-preregistration-1",
        "registered_at_utc": time,
        "historical_catalog_sha256": digest(project / OLD_CATALOG),
        "basis": "Second-study single-image RGB risk, original opaque static synthetic objects at nearby poses.",
        "source": "source",
        "targets": TARGETS,
        "primary_display": "v01 in every family and variant including failed or low-support cases",
        "matrix": {
            "B0": 24,
            "R1": 8,
            "Q1": 8,
            "new_learned_scene_calls": 32,
            "new_predicted_source_and_target_images": 280,
        },
        "reference_design": "The 24 B0 reference bundles share seven camera-equivalent views. R1 gets a half-resolution sole source and matching K. Its six native target references remain identical to B0. Q1 uses that family's B0 reference and camera set.",
        "shared_world": {
            "identity": "original studio floor and closed backdrop, internally lit opaque diffuse objects",
            "geometry_and_lighting": world,
            "materials": frozen_materials,
            "material_roughness": base["material_roughness"],
            "world_linear_rgb": base["world_color_linear"],
            "world_strength": base["world_strength"],
            "colour": base["color"],
        },
        "camera_sha256": digest(project / CAMERA),
        "camera_contract": "512 square source and six target views. OpenCV camera-to-world, X right, Y down, Z forward, metres, integer pixel centres.",
        "validation": {
            **{
                k: v
                for k, v in base["validation"].items()
                if k not in {"occluder_id", "revealed_object_id"}
            },
            "occlusion_profile": "opaque_mesh_pairs",
        },
        "render_settings": {
            k: base[k]
            for k in (
                "blender_version",
                "width",
                "height",
                "preview_width",
                "engine",
                "device",
                "samples",
                "denoiser",
                "seed",
                "repeat_seed",
                "filter_width",
                "pixel_filter",
            )
        },
        "scientific_label": "E is the valid predicted/reference RGB intersection. Y marks mean absolute error in three display-sRGB channels strictly >0.05. Real optical-axis Z, Position, object ID, visibility and edge are evaluation-only.",
        "family_blocking": "All camera, quantization and resize versions descend from one stated family and split. Initial four training families are part of the 12. Historical test families remain separate regression evidence.",
        "test_opening": "One recorder-sealed method freeze covers all eight unseen families, the four familiar-category layouts and four genuinely novel geometry templates. No target render, reference truth, selection result or sample is generated for these eight in this pipeline until the controller verifies the opening.",
        "hypotheses": [
            "Original geometric edge/support features and RGB consistency can rank excessive RGB error in previously unseen layouts, with family-level variation and preserved failures.",
            "A spatial visual descriptor updated by the RGB-error task can improve common-domain ranking beyond the frozen descriptor and original logistic baselines. A null or negative comparison is retained.",
            "The four novel topology families can yield different coverage and error from the four familiar-category layout families. Both four-family groups are descriptive with limited precision.",
            "R1 may increase false negatives or RGB error at fixed operating points. Q1 isolates colour rounding and can have a numerically small measured effect.",
        ],
        "research_scope": "Checkpoint-based historical generator uses its own upstream research terms. These Blender recipes, materials and reference images are original. Evidence Lens deployable models do not receive target truth or scene/split identity as features.",
        "novelty_audit": {
            "reserved_signatures": NOVEL,
            "method": "Geometry fingerprint excludes names/IDs/pigments. A separate canonical shape-and-normalized-pair-distance measure is invariant to rigid whole-layout pose and uniform scale. Geometric graph/face signatures are verified against all 24 historical families and 16 new train/development recipes.",
            "historical_foreground_pair_fingerprints": past_structure,
        },
    }
    write(assets / STUDY, guard)
    rows = []
    for i, c in enumerate(authored, 25):
        family = c.family
        require(family.startswith(f"scene_{i:03d}_"), "Wrong new ordinal")
        split = "train" if i <= 36 else "development" if i <= 40 else "test"
        mat_ids = sorted({o["material"] for o in c.objects})
        assert set(mat_ids) <= set(frozen_materials)
        ids = [o["id"] for o in c.objects]
        names = [o["name"] for o in c.objects]
        require(
            len(ids) == len(set(ids)) == len(set(names)) and min(ids) >= 10,
            "Duplicate mesh identity",
        )
        spec = {
            "objects": c.objects,
            "title": c.title,
            "occlusion_description": c.relation,
        }
        recipe_path = f"configs/b-reference-v1/recipes/{family}.json"
        write(assets / recipe_path, spec)
        flags = topology(c.objects)
        require(
            flags == ([NOVEL[i - 45]] if i >= 45 else []),
            "Unexpected topology classification for " + family + ": " + str(flags),
        )
        pshape = pair_structure(c.objects)
        require(pshape not in past_structure.values(), "Old normalized layout reused")
        conditions = ("B0", "R1", "Q1") if split == "test" else ("B0",)
        rows.append(
            {
                "ordinal": i,
                "scene_family_id": family,
                "scene_id": family + "/v001",
                "scene_version": "v001",
                "title": c.title,
                "split": split,
                "cv_fold": (i - 25) // 3 if split == "train" else None,
                "test_group": (
                    "familiar_layout" if i <= 44 else "novel_geometry_material"
                )
                if split == "test"
                else None,
                "pilot": i <= 28,
                "trials": {k: family + "-" + k for k in conditions},
                "trial_id": family + "-B0",
                "geometry_template": c.template,
                "asset_lineage": c.ancestry,
                "occlusion_description": c.relation,
                "object_count": len(c.objects),
                "reserved_geometry": flags,
                "geometry_fingerprint": geometry_fingerprint(
                    world["objects"] + c.objects
                ),
                "pair_structure_sha256": pshape,
                "material_ids": mat_ids,
                "material_sha256": value_hash(
                    {m: frozen_materials[m] for m in mat_ids}
                ),
                "recipe_path": recipe_path,
                "recipe_sha256": value_hash(spec),
                "recipe_file_sha256": digest(assets / recipe_path),
                "example_view": "v01",
                "view_ids": ["source", *TARGETS],
                "camera_sha256": digest(project / CAMERA),
                "camera_set_id": "formal-near-fixed-v1",
            }
        )
    catalog = {
        "schema": "el-vision-family-roster-1",
        "dataset_id": "evidence-lens-vision-families-v1",
        "experiment_id": "stage2-reference",
        "registered_at_utc": time,
        "preregistration_path": STUDY,
        "preregistration_sha256": digest(assets / STUDY),
        "pilot_families": [r["scene_family_id"] for r in rows if r["pilot"]],
        "historic_reuse": {
            "baseline_catalog_sha256": digest(project / OLD_CATALOG),
            "allowed_training_families": 16,
            "allowed_development_families": 4,
            "historical_regression_only": 4,
        },
        "condition_matrix": {"B0": 24, "R1": 8, "Q1": 8},
        "families": rows,
    }
    write(assets / CATALOG, catalog)
    files = {
        "scripts/" + x
        for x in (
            "b_scene_designs.py",
            "stage2_reference_catalog.py",
            "stage2_register_reference.py",
            "stage2_reference.py",
        )
    }
    files.add(STUDY)
    baseline = {
        *ENGINE,
        BASE,
        CAMERA,
        OLD_CATALOG,
        OLD_DESIGNS,
        "src/evidence_lens/blender_reference/formal_catalog.py",
        "src/evidence_lens/blender_reference/formal_designs.py",
        "scripts/formal_reference.py",
        "src/evidence_lens/labels.py",
    }
    lock = {
        "schema": "el-vision-reference-freeze-1",
        "catalog_path": CATALOG,
        "catalog_sha256": digest(assets / CATALOG),
        "registered_at_utc": time,
        "common_source_files": {s: digest(project / s) for s in sorted(baseline)},
        "stage_files": {s: digest(assets / s) for s in sorted(files)},
        "note": "Recipe file digests for all 24 are in the roster. Acquisition opens only the selected train/development recipe until the shared test-opening is authenticated.",
    }
    write(assets / LOCK, lock)
    return lock


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument(
        "--assets-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    a = p.parse_args()
    print(json.dumps(register(a.project, a.assets_root), sort_keys=True))
