"""Prepared-file and resident regressions on the finite ordered analysis grids."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from evidence_lens.scoring_io import Camera
from evidence_lens_bundle.bindings import binding_from_bundle, camera_identity, frame_meta_from_bundle
from evidence_lens_bundle.bundle import Bundle
from evidence_lens_bundle.calibration import select
from evidence_lens_bundle.common import encoded, row, sha
from evidence_lens_bundle.grids import PROFILE_SIZES, build_grids
from evidence_lens_bundle.numerics import checked_arrays, compare_arrays
from evidence_lens_bundle.resident import load_scorer
from evidence_lens_bundle.scoring import score
from test_bundle import S, T, fixture, replace


@pytest.mark.parametrize("profile", PROFILE_SIZES)
def test_file_and_resident_analysis_grid(tmp_path, profile):
    root = fixture(tmp_path / "bundle", analysis_profile=profile)
    bundle = Bundle(root)
    w, h = PROFILE_SIZES[profile]
    assert bundle.analysis_size == (w, h)
    receipt = score(root, tmp_path / "scores")
    assert receipt["schema"] == "el-bundle-scores-2"
    assert receipt["analysis_profile"] == profile and receipt["analysis_size"] == [w, h]
    assert receipt["grids"] == bundle.grids
    assert receipt["grids_sha256"] == bundle.validate()["grids_sha256"] == sha(encoded(bundle.grids))
    assert receipt["counts"]["canvas"] == len(T) * w * h

    sources = {s: bundle.source(s) for s in S}
    binding = binding_from_bundle(bundle)
    for cache_bytes in (0, 64 << 20):
        scorer = load_scorer(cache_bytes=cache_bytes)
        context = scorer.prepare_sources(sources, version_binding=binding)
        assert scorer.prepare_sources(sources, version_binding=binding).cache_hit == bool(cache_bytes)
        for view in receipt["views"]:
            name = view["id"]
            target = bundle.frame(name)
            meta = frame_meta_from_bundle(bundle, name, binding=binding)
            out = scorer.score_frame(context, target, bundle.cameras[name], frame_meta=meta)
            with np.load(tmp_path / "scores" / view["arrays"]["path"], allow_pickle=False) as raw:
                expected = checked_arrays(dict(raw), scorer.threshold, grid_shape=(h, w))
            diff = compare_arrays(expected, out, scorer.threshold, grid_shape=(h, w))
            assert diff["bit_exact"] and diff["threshold_crossings_common_finite"] == 0
            assert out["identity"]["output_arrays_sha256"] == diff["old_sha256"]
            assert out["identity"]["target_grid_sha256"] == sha(encoded(bundle.grids["views"][name]))
            assert out["profile"]["applicability"] == "transferred_original_threshold_uncalibrated"
            if target is None:
                assert np.all(out["unknown_reason"] == 7)
            else:
                assert out["unknown_reason"][0, 0] == 4 and out["valid"][h // 2, w // 2]
                assert out["valid"].sum() > w * h * 0.9
    if profile == "analysis-256x256-v1":
        path = Path(__file__).resolve().parents[1] / "docs/results/batch-two/calibration/profiles/M4.json"
        cal = select(bundle, "MS-L1", path, sha(path.read_bytes()), None, None)
        assert not cal["applied"] and cal["status"] == "outside_calibration_scope"
        assert any("original 256 x 256" in reason for reason in cal["mismatches"])


def test_cache_separates_grid_contracts_and_same_path_content(tmp_path):
    scorer = load_scorer()
    profiles = (None, "analysis-256x256-v1", "analysis-378x252-v1", "analysis-375x251-v1")
    keys = set()
    grids = []
    for i, profile in enumerate(profiles):
        root = fixture(tmp_path / str(i), analysis_profile=profile)
        bundle = Bundle(root)
        binding = binding_from_bundle(bundle)
        context = scorer.prepare_sources({s: bundle.source(s) for s in S}, version_binding=binding)
        assert context.key not in keys and not context.cache_hit
        keys.add(context.key)
        grids.append(bundle.grids)
        if i == 0:
            legacy = context.identity["source_arrays"]
        elif i == 1:
            assert context.identity["source_arrays"] == legacy
        assert scorer.cache_info()["peak_bytes"] <= scorer.cache_info()["capacity_bytes"]
    assert grids[2]["encoder"]["patch_grid"] == grids[3]["encoder"]["patch_grid"]
    assert grids[2]["encoder"]["pad_ltrb"] != grids[3]["encoder"]["pad_ltrb"]

    item = bundle.meta["sources"][S[0]]
    file = root / item["path"]
    with Image.open(file) as image:
        changed = image.copy()
    changed.putpixel((0, 0), (77, 30, 60))
    changed.save(file)
    ref = row(item["path"], file.read_bytes())
    replace(root, "bundle.json", ["sources", S[0]], ref)
    replace(root, "producer.json", ["source_rgb_sha256", S[0]], ref["sha256"])
    new = Bundle(root)
    change = binding_from_bundle(new)
    assert change["source_revision"] != binding["source_revision"]
    sources = {s: new.source(s) for s in S}
    assert scorer.prepare_sources(sources, version_binding=change).key != context.key
    with pytest.raises(ValueError, match="authenticated bundle"):
        scorer.prepare_sources(sources, version_binding=binding)
    assert (
        scorer.prepare_sources(sources, version_binding={**change, "renderer_revision": "v2"}).key not in keys
    )


def test_target_raw_transform_and_camera_bits_are_bound(tmp_path):
    profile = "analysis-256x256-v1"
    bundle = Bundle(fixture(tmp_path / "bundle", analysis_profile=profile))
    binding = binding_from_bundle(bundle)
    scorer = load_scorer()
    sources = {s: bundle.source(s) for s in S}
    context = scorer.prepare_sources(sources, version_binding=binding)
    meta = frame_meta_from_bundle(bundle, T[0], binding=binding)
    camera, frame = bundle.cameras[T[0]], bundle.frame(T[0])
    out = scorer.score_frame(context, frame, camera, frame_meta=meta)
    other = build_grids(
        profile,
        {
            "live": {
                "width": 512,
                "height": 512,
                "distortion": "none",
                "K": [[250.0, 0.0, 255.5], [0.0, 250.0, 255.5], [0.0, 0.0, 1.0]],
            }
        },
    )["views"]["live"]
    assert other["analysis_K"] == meta["target_grid"]["analysis_K"]
    with pytest.raises(ValueError, match="target grid differs"):
        scorer.score_frame(context, frame, camera, frame_meta={**meta, "target_grid": other})
    with pytest.raises(ValueError, match="registered camera grid"):
        scorer.score_frame(context, frame, camera, frame_meta={**meta, "target_view_id": "live"})
    for bad in ({"source_binding_sha256": None}, {"target_view_id": None}, {"target_grid": None}):
        with pytest.raises(ValueError):
            scorer.score_frame(context, frame, camera, frame_meta={**meta, **bad})
    live = {
        **meta,
        "target_view_id": "live",
        "target_grid": other,
        "bundle_sha256": None,
        "target_file_sha256": None,
        "target_array_sha256": None,
    }
    misses = scorer.cache_info()["misses"]
    variant = scorer.score_frame(context, frame, camera, frame_meta=live)
    assert variant["identity"]["target_grid_sha256"] != out["identity"]["target_grid_sha256"]
    assert scorer.cache_info()["misses"] == misses + 1
    k = camera.K.copy()
    k[0, 0] = np.nextafter(k[0, 0], np.inf)
    cam1 = Camera(camera.width, camera.height, k, camera.c2w)
    variant2 = scorer.score_frame(
        context,
        frame,
        cam1,
        frame_meta={**live, "target_camera": camera_identity(cam1, bundle.analysis_size)},
    )
    assert variant2["identity"]["target_camera"] != variant["identity"]["target_camera"]
    assert scorer.cache_info()["misses"] == misses + 2
    bad_binding = deepcopy(binding)
    bad_binding["grids"]["views"][T[0]]["crop_xywh"][0] += 1
    with pytest.raises(ValueError, match="grid digest"):
        scorer.prepare_sources(sources, version_binding=bad_binding)
    bad_binding["grids_sha256"] = sha(encoded(bad_binding["grids"]))
    with pytest.raises(ValueError, match="declared transformation"):
        scorer.prepare_sources(sources, version_binding=bad_binding)
    base = {**binding, "source_files": None}
    before = scorer.prepare_sources(sources, version_binding=base)
    im, rendered, source_cam = sources[S[0]]
    sources[S[0]] = im, rendered, Camera(source_cam.width, source_cam.height, k, source_cam.c2w)
    assert scorer.prepare_sources(sources, version_binding=base).key != before.key


def test_rectangular_requires_authenticated_producer_and_matching_analysis_camera(tmp_path):
    root = fixture(tmp_path / "bundle", analysis_profile="analysis-448x252-v1")
    replace(root, "producer.json", ["grids_sha256"], "0" * 64)
    with pytest.raises(ValueError, match="producer analysis grid provenance"):
        Bundle(root)
    replace(root, "cameras.json", ["views", T[0], "K", 0, 0], 62.0)
    with pytest.raises(ValueError, match="dimensions or K"):
        Bundle(root)
