# Source photographs and evaluator geometry

Nine independently identified scenes produced 15 source libraries. The source files retain 957 original photographs and their SHA-256 identities. There are 1,053 analysis-image memberships because each controlled family uses its 16-source subset in a separate reconstruction. The complete source audit checked every archive member and repeated the native-to-analysis conversion for the first source photograph in each scene. All nine repeated conversions matched the stored PNG pixels exactly. [Source audit and native receipts](source-data-audit.json)

| Scene | Role | Source photographs per reconstruction | Registered H cameras | Origin |
|---|---|---:|---:|---|
| b01-staggered-arcade | Training | 16, 128 | 6 | Seed 41031, staggered arched bays |
| b02-terraced-court | Training | 16, 128 | 6 | Seed 41032, stepped court and retaining walls |
| b03-woven-truss | Training | 16, 128 | 6 | Seed 41033, intersecting truss planes |
| b04-layered-louvers | Development | 16, 128 | 6 | Seed 41034, tilted slats at multiple depths |
| b05-perforated-gallery | Confirmation | 16, 128 | 6 | Seed 41035, double gallery with portals and bridges |
| b06-corner-alcoves | Confirmation | 16, 128 | 6 | Seed 41036, recessed and side-facing alcoves |
| delivery_area | Development | 34 | 10 | ETH3D, 44 DSLR photographs |
| living_room | Confirmation | 55 | 10 | ETH3D, 65 DSLR photographs |
| bridge | Confirmation | 100 | 10 | ETH3D, 110 DSLR photographs |

## Source permissions and image coordinates

For each authored family, R128 contains r000 through r127. R16 takes every eighth entry starting at r000. Both configurations use h00 through h05 as their fixed targets. Scene geometry and original images are project-authored, credited to wany614 under CC BY 4.0. The frame uses metres. [Data registration](../../../../protocol/many-source-data-v1.json) · [Authored source generator](../../../../scripts/many_data_original.py) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

For each ETH3D scene, original camera-recorded file names are sorted by UTF-8 bytes. H contains the photograph at zero-based rank floor((2j+1)N/20) for each j from 0 through 9. R consists of all remaining names in the same order. The code extracts the official undistorted PINHOLE intrinsics and pose headers and discards the alternating observation line. Reconstruction uses only its permitted R photographs, while all R and H camera headers are available. The source package contains R analysis PNGs and their original identities. The original R JPEGs remain in a separate private archive. [Official scene catalog](https://www.eth3d.net/datasets) · [Camera conventions and file format](https://www.eth3d.net/documentation)

The original ETH3D acquisition may have used all calibrated images. The camera basis is OpenCV right, down and forward. For the ETH3D undistorted release, subtract 0.5 from each principal-point coordinate to convert corner-origin calibration to A's integer-centre convention. The centre crop and LANCZOS resize then produce a 378 × 252 display-sRGB analysis image. Every scene package stores the raw and adjusted K, the original image size, the crop transform and c2w. ETH3D uses its supplied coordinate frame with unit label `scene_units`. The selected photo/camera files provide no independent per-archive metric scale. [ETH3D camera format](https://www.eth3d.net/documentation) · [Official acquisition and registration pipeline](https://github.com/ETH3D/dataset-pipeline)

ETH3D credits T. Schöps, J. L. Schönberger, S. Galliani, T. Sattler, K. Schindler, M. Pollefeys and A. Geiger, *A Multi-View Stereo Benchmark with High-Resolution Images and Multi-Camera Videos*, CVPR 2017. The dataset uses CC BY-NC-SA 4.0. This repository supplies acquisition code, the official URLs, hashes and numerical measurements. [Dataset and licence](https://www.eth3d.net/) · [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Held-out access

The official downloadable archives store each whole acquisition as an opaque compressed file. The three HTTP response bodies totalled 2,419,531,415 bytes against the 5,000,000,000-byte archive download budget. Archive inventory and camera headers precede any selection of photo members. The deployment worker extracts individual R photo members under a positive allowlist. The separate evaluator opens registered H photos only after their split authorization. It logs decoder calls and the first H decode timestamp for each authorized view. Independent true authored camera-Z uses the same calibrated frame as the deployed predictions. [Opaque intake](../../../../scripts/many_data_download.py) · [Role-restricted ETH3D extraction](../../../../scripts/many_data_eth.py)

The pre-opening data audit measured 34 training/development H first decodes and 536 authored source or target optical-depth images. Seven independent triangle-ray checks per depth image gave 3,752 probes. Confirmation scenes had zero H photograph decodes at this checkpoint. For the 22 data-production stages, sealed payloads occupied 5,095,504,366 bytes. Sealed files plus the two seal files per run totalled 5,258,146,881 apparent bytes after deduplicating hardlinks. Worker CPU clocks cover 17 stages and total 152.12 s. The separate sampled process-tree measure covers intervals in 20 stages and totals 207.78 s. These figures cover the data role before the separate sparse occlusion study and exclude model training and scoring. [Recorded resource boundaries](source-data-audit.json)

The controlled visibility study fixes 1,944 analysis-pixel centres per target. An exact ray hits its first mesh triangle and each real source camera casts a continuous segment to that same world point. The source counts the point as visible if the projection falls inside its native pixel footprint and no intervening triangle occurs before the endpoint epsilon of one micrometre. Each Q uses just its listed source IDs. The separate dense audit compares the nearest sampled optical-Z values with tolerance. Source and target photographs and risk ranks do not determine the sparse probe locations. [Registered probe positions, tolerances and memberships](../../../../protocol/many-source-visibility-v1.json) · [Evaluator and independent exhaustive check](../../../../scripts/many_data_visibility.py)

## Reproduction

Run from the repository root with the research Python dependencies, a C++17 compiler and libarchive's `bsdtar`. Set `STORE` to the native research recorder root containing the verified A baseline and B study registration. The example below processes the external development scene. The other ETH3D IDs in the table use the same three commands. For an authored family, use its full scene ID with the `source` command. [Controller](../../../../scripts/many_data_control.py)

```sh
export PYTHONPATH=src:scripts
python scripts/many_data_control.py register --store "$STORE"
python scripts/many_data_control.py download --scene delivery_area --store "$STORE"
python scripts/many_data_control.py roster --scene delivery_area --store "$STORE"
python scripts/many_data_control.py source --scene delivery_area --store "$STORE"
python scripts/many_data_control.py source --scene b04-layered-louvers --store "$STORE"
```

The train/development permission is a separate native stage. After those source packages are ready, it enables their evaluator jobs. Sparse exact visibility takes its own hash-checked source mesh and observations from the family metadata. In a full clean replay its mesh-code provenance stage expects all six authored source receipts. The untouched confirmation path instead requires the common prediction-opening parent with every registered success or terminal failure. [Permission guard and verified H reader](../../../../scripts/many_data_access.py) · [Native source preparation](../../../../scripts/many_data_prepare.py)

```sh
python scripts/many_data_access.py record-development --store "$STORE"
python scripts/many_data_control.py truth --scene b04-layered-louvers --opening-label b-train-dev-h-opening --store "$STORE"
python scripts/many_data_visibility.py register --store "$STORE"
python scripts/many_data_visibility.py record --scene b04-layered-louvers --store "$STORE"
```

A source stage exports `R16.zip`, `R128.zip` or its actual external count, plus `R-originals.zip` and `source.json`. The source-only ZIP contains `scene.json` and its R analysis PNGs. The independent evaluator reads its own index and archive. The CPU deployment example uses b04-layered-louvers, R16, h00. The preregistered confirmation exhibit uses b05-perforated-gallery, R128, h00 at every registered budget. [File-level SHA-256 inventory](source-data-audit.json)

## Development cross-check

The first development R16 bank for b04-layered-louvers supplied h00 and the ordered Q4 r008, r120, r000, r080. The bank and the independently authorized evaluator matched on all 22 registered camera matrices, coordinate units, frame and original-source identities. Of the 1,944 fixed triangle-ray probes, 1,368 hit physical foreground. Within that foreground, 1,353 probes were both physically visible in Q4 and model-comparable, 7 were model-comparable only, and 8 were neither. The separate dense nearest-depth approximation had 66,732 foreground pixels. The native check decoded zero held-out RGB photographs. [Sealed numeric result](development-visibility-check.json)

## Real terminal failures

The failure evaluator consumes the actual unsuccessful native run and its terminal recorder receipt. A missing prediction leaves all valid reference pixels in the planned denominator and zero independently labelled positions. A failed ranking job can still have a usable rendered prediction. In that case the evaluator measures the surviving RGB errors, preserves the unavailable prediction count and marks P1, P2, risk ranks and quotas unavailable. Confirmation also requires the exact failed receipt in the unified H opening. [Failure and reference evaluator](../../../../scripts/many_data_terminal.py) · [Boundary tests](../../../../tests/test_many_science_boundaries.py)

The source-only fit receiver provides the terminal record for a genuinely final failed fit. For an irrecoverable score failure at a measured hard resource limit, the following commands record the stop and its native budget observation. The scoring run ID, resource and reason come from that actual incident. After the applicable H opening, the last command fills the independent reference denominators. Ordinary successful checkpoints use the standard score and evaluation stages.

```sh
python scripts/many_data_terminal.py record-cutoff --store "$STORE" --scene "$SCENE" --count "$R" --failed-run "$FAILED_NATIVE_RUN" --resource "$LIMIT" --final-reason "$OBSERVED_REASON"
python scripts/many_data_terminal.py score-terminal --store "$STORE" --scene "$SCENE" --count "$R" --failed-run "$FAILED_NATIVE_RUN" --cutoff-label "$SEALED_STOP_LABEL"
python scripts/many_data_terminal.py evaluate --store "$STORE" --scene "$SCENE" --count "$R"
```

The generated stop label is `b-score-stop-<scene-with-hyphens>-r<count>`. The hard-limit command accepts `cpu`, `gpu`, `archive` or `score_rss`. It verifies the failed run, its bank and frozen model, the resource amount, the registered limit and the evidence parent. The unit-test cases exercise counterfactual accounting only. Research failures require their actual native receipts.
