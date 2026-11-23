# Stage C multiview acquisition

Sixteen original opaque Blender assemblies share one calibrated 256 × 256 studio capture. The eight training families include two engineering pilots. Four families are reserved for development and four for a single held-out opening. The roster, sixteen recipes, common lighting, materials, cameras and first pilot budget are preserved by a registration recorder run. The registered roster SHA-256 is `f14ca80777cf72e6f99075fb860e24cd38a6b4d9db342011173e9b216d3c66f5`. The compiled ten-camera JSON SHA-256 is `1b6cb179e40b14c5874e4bc89ff8eecef6aa3fa94836cbf37d2b25f9f637267e`.

M2 uses endpoint RGB inputs s00 and s03, separated by 0.72 m. M4 includes s01 and s02 at the intervening third points. The six target views v01 through v06 use separate positions. Camera matrices describe a metric world and an OpenCV camera with its positive axes pointing right, down and forward. The JSON K uses integer pixel centres. The renderer adds 0.5 to the two principal-point coordinates for gsplat. The scientific reference images use Blender Standard display-sRGB. Reference surface Z is the positive first-hit optical depth. Gaussian output Z is opacity-normalized expected optical depth.

## Recording references

Use Python with the project's recorder dependencies for the controller. The Blender subprocess uses the reference environment with bpy 4.5.16, OpenEXR, Pillow and NumPy. A new private store needs a writable `work/` directory. In the following examples, `$STUDY` names that store, `$PROJECT` names the source tree and `$BLENDER_PYTHON` names the reference Python.

```sh
mkdir -p "$STUDY/work"
PYTHONPATH=src:scripts python scripts/stage3_reference.py register --store "$STUDY" --project "$PROJECT" --assets "$PROJECT"
PYTHONPATH=src:scripts python scripts/stage3_reference.py reference --family scene_049_cable_stayed_gantry --store "$STUDY" --project "$PROJECT" --assets "$PROJECT" --blender-python "$BLENDER_PYTHON"
```

Each family produces a recorder `scene_render` and a separately sealed `archive_verify`. The reference bundle includes the original blend, settings, author recipe, native scripts, combined linear EXR, Blender Position, IDs, FP32 display RGB, camera-Z, and independent first-hit visibility for each source. The readback reopens the saved blend, checks each camera, reconstructs axial depth and integer-centre reprojection from the original channels, and casts a deterministic subset of the four-source rays again. The fixed v01 two-seed control measures stochastic variation with the same denoiser. Family reference selections live in `STORE/stage3/CATALOG_SHA/references/FAMILY.json`. The one-time C method gate controls all test references, jobs and selected-data access.

## Preparing and returning multiview jobs

`scripts/stage3_jobs.py` creates deterministic nested job ZIPs and an explicit worker allowlist. Each optimizer receives its permitted PNGs, its source-only camera file and the registered optimizer settings. A separate renderer consumes the selected common checkpoint, allowed source/target camera file and the optimizer's provenance. The renderer outputs one float32 `rgb,z,alpha` NPZ and one PNG for each of the eight M2 or ten M4 views. The checkpoint stores world-space Gaussian means, raw quaternions, log scales, opacity logits and diffuse linear-colour logits. Training uses Adam with one full source per step and computes the photometric loss in display sRGB. The parameter count stays fixed. The rasterizer is provided by the separately installed gsplat package. [Upstream gsplat 1.5.3 renderer](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/rendering.py)

```sh
PYTHONPATH=src:scripts python scripts/stage3_jobs.py --project "$PROJECT" --assets "$PROJECT" --store "$STUDY" --phase engineering-pilot --families scene_049_cable_stayed_gantry scene_050_nested_paper_theatre --batch v11-c-joint-pilot01 --repeat
```

Transfer the published ZIP by its delivery hash. Extract it on the Windows CUDA research volume. Run the shipped worker from the batch directory using the selected GPU Python and the exact enclosed batch JSON digest.

```text
python worker/scripts/stage3_windows.py --batch-sha256 BATCH_SHA256 --store WINDOWS_STUDY_STORE --returns WINDOWS_RETURN_DIRECTORY
```

The Mac receiver accepts the original ZIPs by their SHA-256 and their signed recorder lineage. It validates step traces, source hashes and gradients, before/after parameters, renderer camera-Z and alpha controls, array ranges and decoded images. Missing transfers retain a waiting status. Sealed failures remain recoverable by run identity. Formal selections are written under `STORE/stage3/CATALOG_SHA/trials/TRIAL/selection.json`. Engineering pilots and the single same-input repeat have distinct trial IDs under `engineering/`. Production optimizations restart from the common initialization after a recorded budget decision.

```sh
PYTHONPATH=src:scripts python scripts/stage3_intake.py --store "$STUDY" --project "$PROJECT" --assets "$PROJECT" --batch "$BATCH_JSON" --batch-sha256 "$BATCH_SHA256" --returns "$AUTHENTICATED_RETURN_DIR"
```

The project software and fixed configurations are original work under its Apache-2.0 terms. Original authored reference media and authorized research data use CC BY 4.0, with attribution to wany614. Full private scientific bundles contain all provenance. The explicitly released example and reference assets are enumerated independently for distribution. Blender and gsplat retain their respective upstream rights and are installed separately.
