# Import calibrated Gaussian assets

`evidence-lens inspect` reads the camera model, the authorized source photographs and a Gaussian asset on CPU. `render-asset` uses a CUDA device to turn those inputs into an Evidence Bundle. Once prepared, the bundle can be moved to a CPU-only machine. The default scoring and HTML commands need Python, NumPy and Pillow. [Prepared-bundle quick start](../README.md)

## Run the original bead study

From the source checkout, generate the original helical-bead scene in a fresh directory. It includes two source photographs from an analytic surface renderer, a native metric scene, three Gaussian representations and calibration in two COLMAP encodings. Target photographs are absent. The non-contiguous source IDs are `41` and `5`. The target IDs in order are `70`, `109` and `13`. [Original scene recipe](../../../scripts/b2_asset_fixture.py)

```sh
python -m pip install .
mkdir beads
python scripts/b2_asset_fixture.py --out beads
evidence-lens inspect --colmap beads/colmap/text --encoding text \
  --images beads/images --gaussians beads/graphdeco-sh3.ply \
  --gaussian-profile graphdeco-v1 --plan beads/plan.json --out inspection
```

The inspection writes the integer-centre metric cameras and an input receipt with the original hashes, source order, model size and rendering requirements. On a CUDA machine with gsplat 1.5.3 and its matching PyTorch extension installed, run the next command. The optional package extra is `renderer`. Verify the local CUDA, PyTorch and gsplat combination for that device. [gsplat installation instructions](https://docs.gsplat.studio/main/EXAMPLES_INSTALL.html)

```sh
evidence-lens render-asset --colmap beads/colmap/binary --encoding binary \
  --images beads/images --gaussians beads/graphdeco-sh3.ply \
  --gaussian-profile graphdeco-v1 --plan beads/plan.json --out prepared
evidence-lens validate --bundle prepared
evidence-lens score --bundle prepared --model MS-L1 --device cpu --out risk
evidence-lens report --scores risk --out report.html
```

Use a fresh output for each command. `inspect` and `render-asset` share the same view plan. The renderer runs an off-axis expected camera-Z probe and records the actual CUDA backend, native frame hashes, timing and allocator counters. Each source self-render and target render comes from the one supplied asset. The prepared bundle identifies the local asset-import event and its checkpoint digest. An external asset may lack an optimizer or training receipt.

## Supported input

| Artifact | Accepted representation | Interpretation |
|---|---|---|
| COLMAP calibration | Version 3.10 text or little-endian binary. SIMPLE_PINHOLE and PINHOLE. | Original camera and image IDs retain their meaning. Hamilton `qw qx qy qz` and translation represent world to camera. Camera centers and translations use the supplied metric scale. Subtract 0.5 from the COLMAP principal point for integer-centre Evidence Bundle coordinates. |
| Source images | Nested, relative PNG or JPEG paths. Three 8-bit channels, native 256 × 256, display sRGB. | Only source IDs authorize reading files. Reoriented EXIF, color profiles requiring conversion, missing photos and dimension disagreements require correction upstream. The COLMAP 3.10 text filename occupies one whitespace-delimited token. |
| Graphdeco-compatible PLY | `graphdeco-v1`, PLY 1.0 ASCII or little-endian binary, one vertex table, float32 attributes. | `x y z`, three `scale_i` log-axis scales, four `rot_i` wxyz quaternion components, logit `opacity`, `f_dc_0..2`. An optional whole `nx ny nz` triple is ignored. |
| Spherical harmonics | Complete SH0 or SH3. SH3 uses `f_rest_0..44` in channel-major order. | Real SH on camera-to-Gaussian directions. Basis index increases by degree, then order from −ℓ to +ℓ. Add 0.5 to the SH color, clamp the lower bound to zero, composite against black and clip display RGB to [0,1]. This profile treats the learned coefficients as display-domain values. |
| Project checkpoint | `evidence-lens-npz-v1`. Exact FP32 arrays `means`, `log_scales`, `quats`, `opacity_logits`, `color_logits`, shaped N×3, N×3, N×4, N and N×3. | Apply exponential scales, unit wxyz rotation and sigmoid opacity. Sigmoid color represents linear RGB. Apply the sRGB transfer **after** black-background compositing. |
| View plan | `el-asset-view-plan-1`. Explicit two/four sources, primary, two common endpoints, one or more targets, input-domain declaration and metric evidence. | For COLMAP views, `id` is the original decimal image ID as a string. A novel target can supply its own `camera_m_integer_center` with a distinct opaque ID. Targets require pose and intrinsics only. |

The plan requires a positive `meters_per_unit` and either a named explicit scale declaration or a surveyed baseline. The latter supplies two declared source IDs, their distance and measurement tolerance. Points, Gaussian radii and camera centers receive the same scale conversion. An externally declared scale remains an assumption of that asset. Distorted camera models, sparse XYZ/RGB point clouds, other SH degrees and incomplete parameter sets return a structured diagnostic. The public geometric import and PLY readers were written for this project. [COLMAP 3.10 format](https://github.com/colmap/colmap/blob/0bd66d901c7549051e21e8f648777f802eb20a73/doc/format.rst) · [COLMAP text and binary serialization](https://github.com/colmap/colmap/blob/0bd66d901c7549051e21e8f648777f802eb20a73/src/colmap/scene/reconstruction_io.cc) · [Graphdeco field serialization](https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/scene/gaussian_model.py) · [Graphdeco upstream license](https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/LICENSE.md) · [gsplat 1.5.3 rasterization](https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html)

## Apply a registered operating point

The original model cutoff and risk arrays remain available even when you supply an empirical profile. The caller obtains the profile SHA-256 and the exact deployment scope receipt from an independent trusted record. A fixed-layout DTU proof binds the original source and target camera tables, source PNGs, all render files, checkpoint, optimizer version and seed, renderer, metric rectification, resolution, input semantics and model identity. An external PLY or a selected-view reconstruction does not inherit that calibration. [Scope recorder](../../../scripts/b2_asset_scope.py)

```sh
evidence-lens score --bundle prepared --out profiled-risk \
  --calibration-profile profiles/M2.json --calibration-sha256 "$PROFILE_SHA256" \
  --scope-proof scope.json --scope-sha256 "$SCOPE_SHA256"
evidence-lens report --scores profiled-risk --out profiled-report.html
```

With matching evidence and a finite selected cutoff, each view gains `calibrated-state.npy`. Classification compares the FP32 score, promoted to float64, strictly below the selected threshold. Equality is high. Unknown pixels retain state 255. The report displays the extra state map, counts and calibration status next to the original. `no_feasible_operating_point` and `outside_calibration_scope` leave the original maps in effect. Missing scope evidence is reported explicitly. `evaluated_on_holdout` requires a declared holdout outcome separate from the observed calibration performance. A deployment report has access to predicted evidence only. Its coverage is the fraction of target pixels assigned a state.

The comparison protocol fixed all six encoding/asset combinations and all five cameras ahead of rendering. Read the admission criteria and the two separate CPU and CUDA workers to reproduce the checks. [Preregistered comparisons](acceptance-v1.json) · [CPU verifier](../../../scripts/b2_asset_cpu.py) · [CUDA native-versus-import verifier](../../../scripts/b2_asset_windows.py)

## Measured import fidelity

The original helical-bead scene exercised three Gaussian representations, two COLMAP encodings and five cameras. All 30 comparisons passed the limits fixed before rendering. A second process on Mac read the original FP32 files and compared independently constructed native and imported renders, channel by channel. [Open the original example](../../../examples/helical-beads/README.md) · [Per-view measurements](render-parity.csv) · [Recorded inputs and results](verification.json)

| Quantity | Values compared | Maximum absolute difference | Registered limit |
|---|---:|---:|---:|
| Display RGB | 5,898,240 channel values | 2.98 × 10⁻⁷ | 10⁻⁴ |
| Opacity | 1,966,080 pixels | 2.98 × 10⁻⁷ | 10⁻⁴ |
| Expected camera-Z | 337,362 positive-opacity pixels | 4.24 × 10⁻⁵ m | 10⁻⁴ m |
| Camera intrinsics and poses | Five cameras in each of six combinations | 1.11 × 10⁻¹⁶ | 10⁻⁸ |

Each pair agreed on exactly which pixels had finite depth. Windows used an RTX 3060 Laptop GPU, PyTorch 2.8.0+cu128 and gsplat 1.5.3. The Windows recorder measured 6.469 seconds for all native and imported renders, including the interpreter and worker. PyTorch reported 13,197,824 bytes of peak allocated memory and 27,262,976 bytes of peak reserved memory. Mac measured 0.131 seconds inside the scoring worker for the three prepared targets. These timings describe this 145-Gaussian example and their named stage boundaries. [Stage timings](measured-stage-times.csv)

In a clean CPU installation, six supported imports succeeded. The CLI returned `unsupported` for sparse points, missing opacity, incomplete harmonic coefficients, distorted cameras, missing or conflicting metric scale, a wrong source size and a request to render without CUDA. It returned `invalid` for a duplicate source ID and a missing source photo. Each diagnostic included the reason. [CLI verification and messages](verification.json)

## Calibration on the fixed layout

Both real calibration profiles reported `no_feasible_operating_point`. Across six calibration objects, the procedure found no cutoff meeting its joint low-risk error and coverage requirements. The two-source and four-source profiles both contain a null threshold. On the matched set48 deployment, the installed CPU tool retained the original MS-L1 threshold of 0.06428983435034752. All FP32 scores, validity flags and decisions matched the original release in all six targets per layout. [Scope and original score checks](verification.json)

| Matched deployment | Target pixels | Original low | Original high | Unknown |
|---|---:|---:|---:|---:|
| set48, two sources | 393,216 | 112,426 | 253,903 | 26,887 |
| set48, four sources | 393,216 | 166,075 | 151,345 | 75,796 |

The original bead scene gave `outside_calibration_scope` with the same two-source profile. An offline browser opened both authored reports, selected every target and verified the receipt download. It also opened the two private set48 reports. The public receipt contains numerical checks and page hashes. [Open the out-of-scope example](../../../examples/helical-beads/outside-scope.html) · [Offline browser readback](offline-readback.json)

## Replay the portable archive

`evidence-lens-assets-v0.4.0b1.zip` includes the universal Python wheel, the authored data and prepared bundle, the expected FP32 scores and the two numeric calibration profiles. The archive is 2,092,578 bytes with SHA-256 `e0a6365b2895fb85ab3565f0753c3036bfaf0abc04c7dda53a2fe5fa3aeabcc9`. Extract it into an empty directory. From that directory, install the wheel in a fresh Python 3.11+ environment and run the following commands. [Distribution inventory and clean-install result](portable.json)

```sh
python -m pip install ./evidence_lens-0.4.0b1-py3-none-any.whl
evidence-lens validate --bundle prepared
evidence-lens score --bundle prepared --out my-scores
evidence-lens report --scores my-scores --out my-report.html
```

An independent Mac process checked the digest and length of every indexed ZIP member, installed the enclosed wheel and compared its output with the separately archived original scores. All three numerical files matched byte for byte. Across 196,608 pixels it found zero differences in FP32 risk, validity, state or reason for unknown. [Independent archive replay](portable-readback.json) · [Replay verifier](../../../scripts/b2_asset_portable_verify.py)

The same wheel and bead archive ran in a clean offline Windows 11 CPU environment. A Mac readback checked every risk word, validity flag, state and unknown reason against the archive and all 14 decoded report images against the original. An offline aarch64 Linux container ran the released M4 bell-cage example in a fresh environment and reproduced all six target maps. [Windows archive and per-target checks](../../results/batch-two/windows-cpu-verification.json) · [Linux platform receipts](../../results/batch-two/linux-cpu.json)

The raw arrays stay in the archive. The Git example carries the two offline reports, the illustrated comparison and their original source hashes. Attribute the original geometry, source images, renders and reports to **wany614 / Evidence Lens** under CC BY 4.0. The scene generator and input adapters are Apache-2.0. The previously released weights and external software retain their own terms. [Example inventory](../../../examples/helical-beads/manifest.json) · [Scope of the original example license](ORIGINAL-EXAMPLE-NOTICE.txt)
