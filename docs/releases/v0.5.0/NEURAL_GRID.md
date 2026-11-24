# Rectangular inference with the released DINOv2 weights

MS-frozen and MS-ft accept native analysis grids in Evidence Bundle v2. Both use their original FP32 encoder, 32-channel projection, numeric preprocessing, risk head and cutoff. The original Evidence Bundle v1 follows its historical 256 × 256 analysis and 252 × 252 encoder path. An explicit v2 square profile places the same 256 × 256 analysis dimensions on a 266 × 266 encoder. [Original model card](../v0.3.0-rc.1/MODEL_CARD.md) · [Registered numerical tolerances](../../../protocol/v05-grid-acceptance.json)

| Route | Analysis W × H | Encoder W × H | Patch columns × rows | Right, bottom padding | Finite scored / analysis pixels in probe |
|---|---:|---:|---:|---:|---:|
| v1 legacy square | 256 × 256 | 252 × 252 | 18 × 18 | Historical resize | 62,662 / 65,536 |
| v2 square | 256 × 256 | 266 × 266 | 19 × 19 | 10, 10 | 62,662 / 65,536 |
| v2 landscape | 448 × 252 | 448 × 252 | 32 × 18 | 0, 0 | 108,494 / 112,896 |
| v2 portrait | 252 × 448 | 252 × 448 | 18 × 32 | 0, 0 | 108,493 / 112,896 |
| v2 3:2 | 378 × 252 | 378 × 252 | 27 × 18 | 0, 0 | 91,423 / 95,256 |
| v2 padded | 375 × 251 | 378 × 252 | 27 × 18 | 3, 1 | 90,328 / 94,125 |

## From analysis pixels to patch features

The v2 scorer normalizes the supplied display-sRGB pixels with the published channel means and standard deviations. It replaces an RGB triplet containing an unavailable channel with normalized zero. The smallest multiple of 14 on each axis supplies the encoder extent. Added rows and columns have normalized value zero and contribute zero observed area. Each patch carries the number of finite native RGB triplets divided by 196.

The ViT-S/14 checkpoint contains one CLS token, zero register tokens and a learned 37 × 37 positional lattice. Following the pinned upstream function, the implementation separates the CLS position and interpolates the patch table in FP32 with bicubic sampling, `align_corners=False`, and antialiasing disabled. The height and width scale factors are `(patch_rows + 0.1) / 37` and `(patch_columns + 0.1) / 37`. It restores the CLS token before all twelve transformer blocks and removes it before spatial projection. The row-major patch ID is `patch_y * patch_columns + patch_x`. The 375 × 251 and 378 × 252 profiles have byte-identical position tables for each model. [Official token preparation and 2-D positions](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/models/vision_transformer.py#L167-L218) · [Official ordinary ViT-S/14 model options](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/hub/backbones.py)

At integer native pixel `(x, y)`, the continuous patch coordinates are `((x + 0.5) / 14 - 0.5, (y + 0.5) / 14 - 0.5)`. Bilinear sampling uses border-clamped neighbours, multiplies each coefficient by the neighbour’s observed fraction, and divides their weighted feature sum by the combined fraction. The final head receives the sampled feature vector and the original 39 numerical columns. Outputs cover the native analysis extent. A pixel requires eligible geometric evidence, a finite RGB triplet and positive interpolated support.

## Independent Mac readback

The verifier imports 157 hash-checked source members from upstream commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`. It loads each model’s 175 backbone tensors into the official model and checks every tensor against the release. Its independent oracle constructs input pixels, executes the official token and backbone methods, applies the released projection and head, and samples projected features using double-precision PyTorch `grid_sample`. The asymmetric probe includes a missing 28 × 28 RGB block, a partial-channel edge pixel and separate geometric eligibility holes. Hand-counted northeast, next-row and southeast patch IDs are 31, 32 and 575 in landscape, 17, 18 and 575 in portrait, and 26, 27 and 485 on the 27-column grids. Deliberate transpose and half-pixel errors change the measured controls. [Executable verifier](../../../scripts/v05_verify_neural.py)

Mac arm64 used Python 3.12.16, NumPy 2.5.5 and PyTorch 2.8.0 with four CPU threads. For both models across all six routes, identical attention backends yielded bit-exact encoder input, positions, input tokens and normalized CLS/patch outputs. Against the independent sampling and head implementation, the largest final risk difference was `4.62e-7`. Both methods had zero finite-mask changes and zero threshold crossings. With the official default attention backend, all ten model/grid combinations on v2 passed every registered tolerance. Their largest final risk difference was `8.94e-7`. [Paired numerical receipt](neural-mac.json)

The initial run exposed one out-of-tolerance legacy patch scalar among 124,416 per model when PyTorch chose its default CPU attention kernel for the official reference. Maximum absolute differences were `2.56062e-4` for MS-frozen and `2.40803e-4` for MS-ft. The released implementation already selected PyTorch’s SDPA math backend. A separate paired run reproduced both outliers and the profiler identified the official default kernel as CPU flash attention. Matching the math backend made those intermediate tensors byte-identical. The original default-kernel comparison had at most `1.401e-6` final risk difference across the two legacy runs, zero finite-mask changes and zero threshold crossings. The paired check retained the registered tolerances and both dispatch results. [Original diagnostic](neural-mac-initial.json) · [Paired dispatch and profiler evidence](neural-mac.json)

| Recorded Mac evaluation | Run | Manifest SHA-256 | Numerical receipt SHA-256 |
|---|---|---|---|
| Initial, official CPU default | `20261124T103721991451Z-89cc35880a7a45c6` | `c7da2fc4a0bcb2dd5073c41d4a0f3f18f908a6a88a854f26082b5d3fa8f15bc6` | `d612f963f226de5ba64b11476b662ed361e4b993eb536f34c9b4830e18d58b3d` |
| Paired official default and released math backend | `20261124T104214538720Z-d2fdaadf100546dd` | `364ab8d092e6bd449ec30085d5677fcd9f76fbd40acae920cc28a28ea51bdbe8` | `bd57e8e2dd95b269b24a00fc8ce1f50b1a1a5f48ba12923db7a2f7a393bf49db` |

Two setup attempts terminated before model execution. The recorder rejected a changed-source retry (`20261124T104059465924Z-5975fc7625aa4881`) and a diagnostic input from an undeclared failed parent (`20261124T104152131150Z-1eeb3b9eaa4846be`). The paired evaluation records the initial diagnostic’s run ID, member and manifest hashes in its configuration. Its scientific parents remain the registered A protocol and released weights.

## Reproduce on a CPU

From the matching source checkout, install the optional vision and recorder dependencies and fetch the complete model release. The downloader checks the 164,854,827-byte archive against SHA-256 `f555956840870309217cc85fe9b339ef53762679bbd2138b7f149d8ea451a7b7`. The verifier checks the official source tree by its member hashes, so the upstream commit archive and an exact checkout have the same reference code. [Model distribution](../v0.4.0/MODEL_CARD.md) · [Recorder input and storage conventions](../../recording.md)

```sh
python -m pip install '.[vision,record]'
python scripts/general_release_weights.py --out weights
curl -fL -o dinov2.tar.gz https://codeload.github.com/facebookresearch/dinov2/tar.gz/7764ea0f912e53c92e82eb78a2a1631e92725fc8
python -m evidence_lens.recorder \
  --root .research --kind transfer_verify --config protocol/v05-grid-acceptance.json \
  --experiment local-rectangular-verification --trial official-paired \
  --input protocol=protocol/v05-grid-acceptance.json --input official=dinov2.tar.gz \
  --input models=weights/cpu -- \
  '{python}' '{source}/scripts/v05_verify_neural.py' --threads 4 \
  --protocol '{input:protocol}' --official-source '{input:official}' \
  --models '{input:models}' --out '{out}'
```

The command prints a run ID. Its full numerical receipt is `.research/runs/<run-id>/payload/neural.json`. Each shape reports eligible and unknown counts, input and token checks, per-stage errors, threshold crossings and original tensor hashes. The independent sampling calculations can differ in the last FP32 bits from the scorer’s four-neighbour implementation.
