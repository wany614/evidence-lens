# Batch-two resource and platform receipts

The inventory closed at 02:49:16 UTC on 24 November 2026. It contains 243 sealed recorder originals, including 191 recorded on macOS and 52 on Windows. The Linux container replay was launched and measured by a Mac recorder. There were 242 successful executions and one failed engineering check. Every scientific reconstruction reached its frozen endpoint. [Per-run timings, statuses and measured bytes](runs.csv) · [Inventory scope and hashes](resources.json)

| Quantity measured | Batch-two observation |
|---|---:|
| C6 fixed / D selected and sensitivity, unique new reconstructions | 12 / 13 |
| Optimizer updates, total across unique reconstructions | 125,000 |
| Windows recorder elapsed, cumulative across 25 fits / 25 endpoint renders | 956.795 s / 135.656 s |
| Mac numerical readback, cumulative across 25 fitted results | 48.544 s |
| Reconstruction input packages / output packages and completion receipts | 12,287,766 B / 672,604,960 B |
| Official DTU HTTP requests / response bodies | 298 / 170,160,325 B |
| DTU HTTP bodies for C6 / development camera choices | 112,179,096 B / 57,981,229 B |
| Peak sampled process-tree RSS, Mac / Windows | 528,138,240 B / 1,571,262,464 B |
| Peak sampled GPU memory in use across the device | 978 MiB |
| Local batch-two store, measured file lengths with hard links deduplicated | 3,272,347,811 B |
| Free / total external volume bytes during measurement | 1,016,734,728,192 B / 1,023,999,787,008 B |

The cumulative fit and render times include their recorder process boundaries. The whole-device GPU counter can include other processes. Transfer measurements refer to different interfaces and measure retained or transmitted bytes at each boundary. The disk census includes a preserved quarantined CPU return. Its logical count also appears in the accepted transfer once its original lineage was verified. [Scientific stage measurements](../reconstruction/README.md) · [Selection and query timings](../active-views/probe-costs.csv)

## Replaying the exact CPU wheel

The wheel occupies 365,563 bytes and has SHA-256 `ca52e82c8e870560c05075dda4b5282311500580f9283fe234aa27643d6f8882`. The portable 50-member example occupies 2,092,578 bytes. It includes the original bead scene, COLMAP inputs, Gaussian representations, prepared bundle and expected numerical outputs. [Archive receipts and instructions](../../../tooling/assets/README.md)

| Runtime | Actual input and comparison | Result |
|---|---|---|
| macOS arm64, CPU | Clean wheel install and original three-target bead bundle | All 196,608 FP32 words, masks, states and unknown reasons matched. The three compressed arrays matched byte for byte. [Consumer replay](../../../tooling/assets/portable-readback.json) |
| Windows 11 AMD64, Python 3.11.5, NumPy 2.3.3, Pillow 11.3.0 | New offline venv on the D drive, same authored ZIP and original three-target bundle | 14.203 s of recorder time. All 196,608 FP32 words and all categorical outputs matched, including 33,366 finite scores. The HTML file and encoded images had different bytes. All 14 decoded RGB images matched in size and pixels. [Mac readback of the Windows original](../windows-cpu.json) |
| Linux aarch64, Python 3.12.16, NumPy 2.5.5, Pillow 12.3.0 | Network-disabled Docker, read-only mount and new tmpfs venv, original six-target M4 bell cage | 388,358 finite scores out of 393,216 pixels, maximum difference 0, identical masks and states. Two independent installs, including execution through HTML, took 4.267 and 3.644 s. [Platform and scope receipts](../linux-cpu.json) |
| Windows RTX 3060 Laptop, CUDA 12.8, gsplat 1.5.3 | Native and imported renders on the same device, two camera encodings, three Gaussian representations | 30/30 comparisons passed, including the depth support checks. [All channels and denominators](../../../tooling/assets/render-parity.csv) |

The Windows CPU example had 1,344 low, 32,022 high and 163,242 unknown pixels. These are deployment states. The replay used no target photographs.

## Failures and review

The first independent set48 scoring check (`20261124T020940844896Z-26af37772c174877`) read the historical method records at the wrong nesting level. The recorded replacement (`20261124T021007283105Z-08ae49e3901e4e91`) read the original schema and matched every FP32 word. The first authored COLMAP fixture also exposed the native reader's whitespace-delimited text filename convention. The revised original fixture passed native parsing before Windows render acceptance. The original fixtures and receipts remain archived. [Asset verification](../../../tooling/assets/verification.json)

One Windows CPU return was initially quarantined because the single-store receiver could not resolve a historical upstream. A separate audit (`20261124T024743873579Z-3cea63613ba64587`) verified the 829 historical run manifests, 2,993 dependency edges and 79 shared assets in their actual stores. The identical return ZIP was subsequently admitted. The Mac verifier (`20261124T024840489724Z-802b4dde406a4262`) compared the four arrays per target directly with the packaged originals. In the complete batch store, 52 transfers were accepted and the initial quarantine receipt was retained. [Verified Windows arrays](../windows-cpu.json)

After the first inventory, a second check also bound the Windows-measured ZIP digest, the wheel and prepared bundle inside it, the original HTML, and the native timing stages to the same returned evidence. One attempt read the ordered stages as a JSON object (`20261124T025251085374Z-0c2e5b1069f94a75`). A second stopped during recorder admission when it supplied the failed attempt as an input (`20261124T025728253416Z-c8991a35506c4e64`). The corrected check verified and cited both failed records in its configuration, read the native JSON array and passed on the same bytes (`20261124T025913936867Z-db39ad69bea64f5c`). It found zero word or state differences and identical dimensions and RGB pixels in all 14 embedded images. [Archive and native-stage verification](../windows-cpu-verification.json)

The cumulative inventory closed at 02:59:58 UTC with 247 sealed originals, 244 successful and three failed. It includes the first 243, the first census itself and the three verification attempts. The batch store then occupied 3,297,594,802 file bytes with hard links deduplicated. This count ends before the supplemental census process itself (`20261124T025958729214Z-9bf1bb2c875f4aee`). Both inventories record the same 298 DTU requests and 170,160,325 response-body bytes. [Cumulative inventory](after-close.json) · [All 247 rows](after-close-runs.csv)

Two successful records followed that cutoff. The census process took 1.677 s and produced 75,939 payload bytes. The final archive verifier (`20261124T030422988018Z-770cf828f0f945dd`) took 1.438 s and produced 5,094 payload bytes. It additionally required each of the nine native stages to appear in order, with exit code zero, from environment creation through the final comparison. Including these two records gives 246 successful and three failed originals. [Nine ordered stages and numerical results](../windows-cpu-verification.json)

All 84 focused bundle, DTU, reconstruction, calibration and selection tests passed across their respective dependency environments. The public-export tests passed separately. A wider historical regression attempt had 339 passes, 11 skips and two isolated child-process failures in an environment lacking PyTorch. The standalone rank/serialization case then passed under the PyTorch interpreter. The other regression performed four temporary optimizer updates on its synthetic fixture before that environment failure. Its full fit was left unrepeated. The fixture used no project training data and supplied no released weights.

Claude Opus independently reviewed the implemented selection, calibration and final freeze. The resulting checks explicitly bind the prospective inputs, verify equal per-view masks, compare original versus sensitivity worker/input identities and check query poses against the independent evaluation rig. The final access inventory lists sensor acquisition receipts with an HTTP-log presence check. It covers the batch store, including quarantine. [Final freeze and access inventory](../holdout/t1-freeze.json) · [Archive acquisition receipts](../holdout/acquisition-audit.json)
