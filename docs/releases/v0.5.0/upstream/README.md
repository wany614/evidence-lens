# Gaussian assets and the authored bell study

The release can inspect native recovery checkpoints on a CPU and render their Gaussian parameters on CUDA. The authored bell study uses a separate 4,096-Gaussian Stage C model. Both paths preserve the measured camera poses and use expected camera-Z in metres.

## Supply a calibrated reconstruction

Use a COLMAP 3.10 PINHOLE or SIMPLE_PINHOLE model, two or four ordered source RGB8 photographs, a Gaussian file and a view plan. Targets require calibrated camera entries. An `el-asset-view-plan-2` plan adds an `analysis_profile` such as `analysis-448x252-v1` to the original plan. The optional `display_size` is a width/height pair with the same aspect ratio as the analysis grid. When the plan supplies a `grids` record, the importer verifies it against the derived original camera and analysis intrinsics. [Example view plan and commands](../../../tooling/assets/README.md)

The inspection receipt binds the original photo bytes, normalized analysis PNGs, source order, original camera sizes, per-view crop and pixel-centre transform. The source filter samples the registered integer crop with Pillow LANCZOS in display sRGB. The renderer computes display RGB, accumulated opacity and metric expected camera-Z directly on each analysis grid. Zero opacity gives undefined camera-Z.

For an `el-recovery-checkpoint-1` optimizer file, select `--gaussian-profile evidence-lens-recovery-pt-v1` and supply `recovery_clip_m` as the original numeric near/far pair in the plan. COLMAP omits these planes. Use `metric.meters_per_unit = 1` for native recovery assets and supply cameras in the same metric world frame. Inspection of this profile needs a CPU-capable PyTorch installation with support for `torch.load(weights_only=True)`. Rendering calls gsplat 1.5.3 on CUDA with the original screen-space variance and black background. Prepared bundles can subsequently be scored on a CPU. [Decoder](../../../../src/evidence_lens_asset/gaussian.py) · [Renderer](../../../../src/evidence_lens_asset/render.py)

| Native optimizer field | Shape | Quantity supplied to gsplat |
| --- | --- | --- |
| `means` | N × 3 | World position in metres |
| `scales` | N × 3 | Exponentiated log standard deviations in metres |
| `quats` | N × 4 | Unit quaternion after normalization |
| `opacities` | N | Sigmoid of the opacity logit |
| `colors` | N × 3 | Sigmoid of degree-zero linear RGB logits |

The decoder reads the actual number of rows, checks finite FP32 tensors and requires every field to agree on N. It checks the original optimizer step, positive cap, screen-space variance and calibrated metric units. An engineering control used an exposed M2 endpoint checkpoint with 16,251 Gaussians at step 5,000 under a cap of 20,000. Removing one opacity row triggered rejection. Three additional tests passed against the released wheel in a fresh CPU environment, covering branch permissions, camera/photograph provenance and malformed native checkpoint fields. [Numerical evidence and original run/member digests](evidence.json)

For this checkpoint, the release code reproduced all five native optimizer tensors byte for byte, both before and after their CUDA activations. Its 256 × 256 rendering of source camera `s00` matched the original archived prediction. All 196,608 RGB values, 65,536 alpha values, 65,536 expected camera-Z values and their finite masks were identical. The camera used clipping planes of 0.05 and 30 metres and the fitted screen-space variance of 0.3. Windows rendered from the checkpoint and source camera only. Mac verified the returned frame against the separately held original prediction. This control covers one previously exposed source view. [Independent frame comparison](../../../../scripts/v05_recovery_control.py)

## Register source access

A v2 view plan can supply `source_authorization` with a branch, its registration and the registration SHA-256. A fixed M2 or M4 branch follows its preregistered ordered list. An optional branch requires the sealed selection or failure record and its SHA-256. A successful selection grants exactly the ordered pair between the registered endpoints. A failed selection grants its two endpoints. The independent fixed M4 grant remains available. The importer verifies the requested order before reading any source image. The checks are prospective and apply when a plan includes this authority. [Grant validator](../../../../src/evidence_lens_asset/permissions.py)

## Reproduce the bell comparison

The original authored Offset Bell Cage scene supplies four source poses (`s00`–`s03`) and six target poses (`v01`–`v06`). All six routes use the same Stage C M4 checkpoint, fitted previously to square source exposures. For this study Blender Cycles generates new photographs from the original physical scene, lens and poses. Each producer records the fitting-source digests and the scoring-source digests separately. The Gaussian renderer runs once per registered route/camera. [Native projection and active rectangles](../../../../scripts/v05_bell_common.py) · [Independent photography](../../../../scripts/v05_bell_photos.py)

| Prepared bundle | Original source grid | Analysis grid | Active image area |
| --- | --- | --- | --- |
| `legacy-letterbox` | 896 × 504 | 256 × 256 | 256 × 144, first row 56 |
| `landscape` | 896 × 504 | 448 × 252 | Full grid |
| `square` | 512 × 512 | 256 × 256 | Full grid |
| `portrait` | 504 × 896 | 252 × 448 | Full grid |
| `photo` | 756 × 504 | 378 × 252 | Full grid |
| `odd` | 750 × 502 | 375 × 251 | Full grid |

The legacy and native landscape arms observe the same 16:9 frustum. For the legacy input, LANCZOS resizes the landscape source to 256 × 144. The native Gaussian renderer produces its own 256 × 144 channels for that arm. Both are centred on a 256-square canvas. Its 56 rows above and below the image have zero RGB and opacity and undefined camera-Z. The actual target evaluation uses the common content area and accounts separately for missing support. The new `square` route uses a square frustum and a 266 × 266 patch-encoder input.

The `v05-bell-bundles.zip` download contains the six prepared scoring inputs, with four ordered source photographs and ten Gaussian renders per route. The companion `v05-bell-source-observations.zip` contains the 20 original source RGB8 PNGs. Its `index.json` links every one of the 24 derived source PNG digests to an original. Rebuilding them from the originals reproduced every byte. [Source reconstruction checks](../../../../scripts/v05_bell_observations.py)

The two-arm comparison sealed its 48 target/method scores before issuing the reference opening. A separate Blender Cycles run then produced all six original 896 × 504 target measurements, with the original lens and 96 samples per pixel. The separate `v05-bell-independent-references.zip` contains RGB8 PNGs, FP32 display-sRGB arrays, camera metadata and the opening receipt for independent evaluation. These six measurements belong to one authored engineering example. [Score and reference roles](../../../../scripts/v05_bell_control.py)

The authored photographs and original scene are licensed CC BY 4.0, credited to wany614, Evidence Lens. Source code and the original bell reconstruction use the project's Apache-2.0 terms. The recovery-checkpoint control publishes numerical comparisons and digests. Its external data and checkpoint are held separately. [Project rights by asset](../../../multisource-rights.md)
