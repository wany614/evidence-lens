# Windows research worker

Use a project-local virtual environment on the chosen research drive. Record Windows, free space, GPU, driver, Toolkit, MSVC, Ninja and package versions first. The recorder and producer can use the same Python environment on Windows. A full research run needs a CUDA-capable PyTorch 2.8.0, torchvision 0.23.0, gsplat 1.5.3 and an editable, clean SHARP checkout at the commit in the B0 configuration. Choose the PyTorch CUDA build to match the installed driver and the compiler toolchain for gsplat. Source, weights, package caches, temporary build products and runs belong on the research volume.

The tested worker used Windows 11 x64 build 26200, CPython 3.11.5, CUDA 12.8 compilation tools (nvcc 12.8.93), MSVC 19.44.35230 and Ninja 1.13.0. The NVIDIA driver was 610.47. The package pins are recorded in [Windows research constraints](../requirements/windows-worker-constraints.txt). Install the CUDA builds of PyTorch and torchvision from the official cu128 index. Install SHARP from the pinned source checkout. Use the constraints file with pip's `-c` option.

For the measured renderer the official gsplat 1.5.3 source distribution had SHA-256 `343f080c470e891ba3c697c03bdf0952cb982d1d535fdc24adb23c2cb4fba906`. With a working x64 MSVC and Windows SDK environment, set `TORCH_CUDA_ARCH_LIST=8.6` and `MAX_JOBS=4` for this GPU. The recorded build used `python -m pip wheel --no-deps --no-build-isolation --no-index --wheel-dir wheels path/to/gsplat-1.5.3`. Install the resulting wheel in the research environment. A small native CUDA probe caught an incomplete Visual C++ auxiliary build layout on the initial attempt. After repair, gsplat’s native extension loaded and completed the seven-view calibration and render. The constraints file and these tools are specific to this worker.

Download the official checkpoint from the URL in `configs/windows-b0.json`. Verify byte count and SHA-256 against this file. The worker also validates the checksum before the official predictor loads it. Code and weights have separate licenses. [SHARP installation and use](https://github.com/apple-aiml-research/ml-sharp/tree/aed6527499ef91cba3b54c18d49a870f25947190)

A handoff carries an exact copy of the source PNG, camera calibration, reference run ID/manifest hash and allowlisted worker source. The generating process reads the source image and known source focal length. The renderer reopens the PLY with the pinned SHARP reader, blends in linear RGB using gsplat, converts once to display sRGB and exports FP32 data, PNG previews and actual measured resource figures. The source self-render is a diagnostic and the other six views are targets.

The first generation used the binary little-endian SHARP layout. Each vertex stores 14 FP32 scalars covering three source-camera position coordinates in metres, three degree-zero sRGB SH coefficients, an opacity logit, three log scales and four quaternion components. The independent PLY readback found finite values in all 14 fields of all 1,179,648 vertices. Source-camera Z ranged from 0.7165 to 3.4290 m. The file stored an identity source extrinsic, sRGB tag 0 and image dimensions 512×512. Its boundary-origin intrinsic matrix agreed with the frozen camera to 0.000021 pixels. `scripts/inspect_sharp_ply.py` checks these metadata and records separate hashes for the vertex fields.

Transfer complete generation and render run containers with their SHA-256 values. The receiver resolves the reference run and any shared checkpoint by content hash before importing the generation run, then imports the rendering run. A transfer failure is recorded separately from computation. A known-depth calibration failure disables depth-based study. The source self-render's appearance is reported whatever the measured error.

## Execute the first handoff

On the Mac select an explicit successful reference run.

```sh
python scripts/make_handoff.py --reference "$STORE/runs/$REFERENCE_ID" --out "$OUT/windows-input.zip"
```

Transfer that ZIP independently of the reference archives. Match its reported digest before extraction to a new directory on Windows. Install the worker dependencies in the project environment. The source handoff runs directly from its `code` directory. Using PowerShell variables that point to the extracted handoff, private store, outbound ZIP directory, worker Python and verified checkpoint, run

```powershell
& $Python "$Handoff/code/scripts/windows_stage.py" generate --handoff $Handoff --store $Store --return-dir $Outgoing --weights $Weights
& $Python "$Handoff/code/scripts/windows_stage.py" render --handoff $Handoff --store $Store --return-dir $Outgoing --generation-run $GenerationId
```

Read the actual generation ID from the first invocation. The script packs failed and successful attempts separately. Keep original containers and their producer SHA-256 values. On the Mac import generation before render, using the independently acquired shared checkpoint and the already verified scene as dependencies.

```sh
python -m evidence_lens.archive add-asset "$WEIGHTS" --root "$STORE" --sha256 "$WEIGHTS_SHA256"
python -m evidence_lens.archive receive "$GEN_ZIP" --root "$STORE" --sha256 "$GEN_ZIP_SHA256"
python -m evidence_lens.archive receive "$RENDER_ZIP" --root "$STORE" --sha256 "$RENDER_ZIP_SHA256"
python scripts/mac_trial.py score --store "$STORE" --reference-run "$REFERENCE_ID" --readback-run "$READBACK_ID" --render-run "$RENDER_ID" --cpu-python "$CPU_PYTHON"
python scripts/mac_trial.py evaluate --store "$STORE" --reference-run "$REFERENCE_ID" --readback-run "$READBACK_ID" --render-run "$RENDER_ID" --score-run "$SCORE_ID"
python scripts/mac_trial.py verify --store "$STORE" --reference-run "$REFERENCE_ID" --readback-run "$READBACK_ID" --render-run "$RENDER_ID" --score-run "$SCORE_ID" --evaluation-run "$EVALUATION_ID"
```

For scoring, `CPU_PYTHON` may be the base-only virtual environment. Evaluation uses the research environment. Verification reads the fixed view's original RGB, camera-Z and alpha, reconstructs its labels and metric counts, then checks sampled L0 projections against the CPU output. The verification run records all source files and their parent IDs. Incomplete archives are quarantined with a separate receipt. Exactly matching repeated transfers preserve the first accepted result.

## Formal B0 acquisition

The private reference controller first seals a family's seven cameras and `reference_readback.json`. Supply both exact run directories and the frozen catalog. The first successful export freezes the restricted worker source archive for the catalog. Re-export uses those bytes. Each ZIP carries one PNG, its cameras, its own family identifiers, the two upstream manifest pins and the worker code. Its receipt reports the ZIP SHA-256 and `handoff_sha256`, which hashes `handoff.json`.

```sh
python scripts/make_handoff.py --reference "$STORE/runs/$REFERENCE_ID" --readback "$STORE/runs/$READBACK_ID" --families protocol/families-v1.json --out "$OUT/$FAMILY.zip"
```

Use the same two Windows invocations shown above. Run them from the extracted handoff's own `code/scripts/windows_stage.py`. A per-trial reservation precedes process launch. Repeating a stage recovers the already sealed result or reports the prior attempt. An active or previously claimed job with unknown disposition requires inspection of its local recorder state. Each outgoing `<run_id>.receipt.json` reports the exact run, manifest, package SHA-256, size, exit code, status and family trial ID. Use `--retry-of RUN_ID` for a failed scientific attempt and `--resume-preflight RUN_ID` for a sealed failure before worker launch. Outgoing packaging can resume independently after a disk-space stop.

Receive generation then render through `evidence_lens.archive receive` with the SHA from each producer receipt. For each Mac B0 stage add `--families protocol/families-v1.json --generation-run "$GENERATION_ID"` to the corresponding command above. The Mac checks the explicit generation, render, frozen export, family identity, accepted calibration and selected scene/readback. On held-out `evaluate` and `verify`, `--method-freeze FILE` defaults to `STORE/protocol/method-freeze-v1.json`. The recorded threshold selection and frozen model must agree with the catalog and study. Freeze the complete seven-role B0 selection with `scripts/select_trial.py` after the independent view check.

Training and development consume compact, immutable per-family extracts. The previously frozen tabletop chain is the preassigned first training family. Each prepare run records its six targets, source score and evaluation receipts, catalog and selected run manifests. The complete training or development directory contains only `dataset.json`, `catalog.json` and the compact per-family files. The worker checks the complete frozen roster before opening those extracts.

```sh
python scripts/formal_batch.py prepare --store "$STORE" --family "$FAMILY"
python scripts/formal_batch.py dataset --store "$STORE" --split train --out "$OUT/train"
python scripts/formal_batch.py dataset --store "$STORE" --split dev --out "$OUT/dev"
```

The project retains its 20 GiB active target and 30 GiB free-space floor. Staging uses same-volume hard links for read-only bounded input and learning files where supported. The content-addressed checkpoint can be verified in place. For a separate mounted archive, copy a completed trial under explicitly chosen storage and verify an independently retrieved copy against the original manifest SHA. The object store shares identical bytes across runs and families. Its manifest lists every original relative path, all frozen attempts, selection, available learning extract, shared weights and recorded receipts. The verification receipt reports the bytes actually reopened at the supplied location. These commands retain the active originals.

```sh
python scripts/formal_archive.py copy --store "$STORE" --family "$FAMILY" --vault "$ARCHIVE"
python scripts/formal_archive.py verify --vault "$RETRIEVED" --manifest "$RETRIEVED_MANIFEST" --sha256 "$MANIFEST_SHA" --receipt "$OUT/$FAMILY.readback.json"
```
