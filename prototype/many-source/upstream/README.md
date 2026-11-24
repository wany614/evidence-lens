# Official gsplat input

The supported producer is the single-GPU gsplat `simple_trainer` at commit `937e29912570c372bed6747a5c9bf85fed877bae`, using its native SH3 checkpoint, Adam optimizers and MCMC strategy. The headless adapter sets the training roster from the registered sources, disables the built-in holdout and supplies the fixed, known pinhole cameras. It initializes from a new two-source triangulation of the source photographs. [Official trainer](https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/examples/simple_trainer.py) · [Upstream license](https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/LICENSE)

## Prepare a bank

Install the many-source prototype and Evidence Lens 0.5.0. Extract `many-authored-native-b04-r16.zip` into an empty directory. It contains the complete native model, target predictions and source self-renders under `native/`, plus the exact source archive `sources.zip`. Extract `many-incomplete-native-example.zip` alongside it for the missing-depth example. Set `CONVERTER` to the absolute path of `prepare_bank.py` from this directory. The standalone converter verifies every required prediction channel and the registered source and checkpoint digests.

```sh
python "$CONVERTER" inspect --asset native
python "$CONVERTER" prepare --asset native --sources sources.zip --out prepared \
  --trusted-asset-sha256 3fd11a56f987e28d051828ff6189148b2f207f6154a18a0f16c4966b588d47f9 \
  --trusted-source-sha256 6a2942ca74ee890f7889f5d7f85ff13bdb6a466c1e58eafb2c5ec57cbe450cf5 \
  --trusted-lineage-sha256 c285c6650e2bbb93db4a8f8c1c85557c9d8c12db25464242b63995450a45be6b
python "$CONVERTER" inspect --asset missing-camera-z.npz
```

The two independent input digests establish the verified reconstruction label. The lineage digest reproduces the exact recorded bank, SHA-256 `193b1b300f072c67fef3d5dca25f78fa4cc0b1a92613c7c4ccae30c4fe5ff250`. A reconstruction assembled without trusted input digests receives the unverified provenance label. The negative example takes an actual rendered view and supplies its RGB and alpha. Its inspection names the missing expected camera-Z channel. A bare PLY inspection lists its missing camera, source and rendering evidence.

| Independent acceptance record | Value |
| --- | --- |
| Complete ZIP, SHA-256 | `24739674726be8f682cd5376a36fb7a967dd0090a79edec2c1db616f6e6c3394` |
| Native checkpoint, SHA-256 | `9666db2f040cff27962204215e55867e85e84d57da60debb4bc88d52c66a194e` |
| Incomplete ZIP, SHA-256 | `9311297f5890be6bac363a7cf9212ca43dc6997e4e9da634f40e85fbf1e386b5` |
| Clean converter acceptance run | `20261124T114125891178Z-6fade8b4462d4c42` |
| Acceptance manifest, SHA-256 | `213c1000c8d84d0695bd4c121bce11595e65fb23e470bb30707dc69bd7721e36` |

## What the model stores

The original checkpoint has `step` and a `splats` state dictionary. `means` are world coordinates. `scales` are log standard deviations, `quats` are wxyz orientations normalized internally by gsplat, and `opacities` are logits. `sh0` has shape N × 1 × 3 and `shN` has shape N × 15 × 3. These coefficients are optimized against display-sRGB source pixels. Rendering clamps native display values to [0, 1]. A second linear-to-display transfer would change them. The loader validates the actual primitive count and every tensor shape. It also compares the exported channels against the official runner loaded from the saved checkpoint. [Checkpoint adapter](../../../scripts/many_upstream_adapter.py)

Each 378 × 252 FP32 render has `rgb`, `alpha` and `z`. The last represents opacity-normalized expected optical-axis depth in the declared world unit. Alpha at or below 0.000001 gives undefined Z (NaN). A two-surface on-axis and off-axis analytic probe verifies the difference between accumulated weighted depth and expected Z. The camera stores OpenCV integer-centre intrinsics and c2w. The renderer adds 0.5 to the two principal-point coordinates when handing them to gsplat. Authored geometry uses metres. The ETH3D camera frame retains explicitly named scene units.

Source-only matching uses six geometrically neighbouring cameras, reciprocal SIFT and forward-backward Lucas–Kanade tracks. Positive-depth, reprojection, parallax, colour and source-extent checks precede deterministic voxel fusion and the 15,000-point initialization cap. The generated source COLMAP model contains the newly reconstructed points and two-source tracks. The official trainer computes 6,000 single-source losses and applies 6,000 optimizer updates in memory, with SH3 and a 60,000-Gaussian MCMC cap. It saves the checkpoint after computing the last gradient and before applying the last update at zero-based index 5,999. Each source self-render and target prediction uses the saved model with the first 5,999 updates. [Matching and COLMAP export](../../../scripts/many_upstream_sfm.py) · [Auditable changes to the trainer](../../../scripts/many_upstream_patch.py)

The project’s headless training path uses valid 11 × 11, sigma 1.5 display-RGB SSIM through differentiable PyTorch operations. It checks objective values and first-image gradients against the independent `pytorch-msssim` reference used by the pinned fused-SSIM project tests. Source-only Euclidean KNN is checked against ordered independent distances. Actual per-view training counts, source-order digest, checkpoint parity and the resource times accompany each native archive. [Objective checks](../../../scripts/many_upstream_objectives.py) · [Pinned fused-SSIM tests](https://github.com/rahul-goel/fused-ssim/blob/328dc9836f513d00c4b5bc38fe30478b4435cbb5/tests/test.py)

To inspect the exact change that ran, use the source file at the gsplat commit above and the distributed unified diff. The adapted file has SHA-256 `ce42db9ae2cd950bdbcfecb95a8b97aae365051fa06f884c34f5d518d1d71e1e`. The packet includes this file, the native source reader, the independently verified objective, and the two scalar helpers extracted from the same gsplat commit. [Executed patch](gsplat-simple-trainer.patch) · [Upstream license](GSPLAT-LICENSE.txt)

```sh
patch -o adapted-simple-trainer.py path/to/gsplat/examples/simple_trainer.py < gsplat-simple-trainer.patch
```

An independent readback replayed the executed training dataset against all 15 issued fit packets and their authorized source archives. Filenames, ordered source identifiers, PNG hashes and all 100,304,568 decoded RGB pixels agreed across 1,053 source views. Applying each renderer's original FP32 principal-point adjustment produced the same intrinsics as training for all 1,053 views. Translation agreed exactly. The largest absolute difference in any rotation entry was 3.68 × 10⁻¹⁶. On a plane at unit camera-Z the largest measured discrepancy across the full image grid was 9.67 × 10⁻¹³ pixels. Readback `20261124T121212827980Z-86e94030823e4a8f` has manifest SHA-256 `91453df8331b95863e9dedd82e166d6be469b96fcea93f7e16dd1c290f751bf3`. [Independent input and checkpoint audit](../../../scripts/many_upstream_alignment.py)

## Development reconstruction quality

Each row compares the fixed checkpoint with the constant RGB mean of its authorized source photographs. MAE averages the three display-sRGB channels over the 378 × 252 analysis grid, then gives each view equal weight. After each source-only checkpoint was sealed, the development evaluator opened six independent authored views per reconstruction and ten independent ETH3D views.

| Scene and sources | Source self-render MAE | Source mean control | Independent target MAE | Target mean control | Fitted Gaussians with opacity > 0.005 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Layered louvers, 16 | 0.01091 | 0.09280 | 0.01516 | 0.09176 | 56,860 |
| Layered louvers, 128 | 0.01162 | 0.09266 | 0.01265 | 0.09177 | 55,399 |
| ETH3D delivery_area, 34 | 0.01747 | 0.15732 | 0.04655 | 0.16277 | 50,562 |

Each reconstruction retained 15,000 triangulated initialization points and ended with 60,000 fitted Gaussians. The corresponding fractions of target pixels with alpha at least 0.5 were 99.60%, 99.41% and 99.24%. Development record `20261124T115849245504Z-645bff342f1b4378` has manifest SHA-256 `98abed85320d7ae425daa850b9d98f790d52acf5739024407842b091ecc4e214` and diagnostics SHA-256 `ec9676eb67ac36e215006735ac82f587d6f6fbc0162fdcebb8c7b4921243ac82`.

## Dependencies and data rights

The source adapter and its integration use the project software license. gsplat and its separately installed CUDA runtime retain their own notices. The examples contain authored project imagery. ETH3D photographs and reconstructions derived from them are held in the research archive under the dataset's applicable terms. [gsplat and dependency notices](../../../THIRD_PARTY_NOTICES.md)

| Component | Use and license |
| --- | --- |
| gsplat 1.5.3 and the official trainer | CUDA Gaussian projection, SH, optimization loop and MCMC densification. The public diff and the worker packet preserve the source revision and attribution. [Original Apache-2.0 license](GSPLAT-LICENSE.txt) · [Official revision](https://github.com/nerfstudio-project/gsplat/tree/937e29912570c372bed6747a5c9bf85fed877bae) |
| OpenCV 4.11.0, Python wheel 4.11.0.86 | CPU SIFT, tracking and triangulation. The OpenCV project uses Apache-2.0. The Python wheel and bundled third-party components include their own notices. [OpenCV license](https://github.com/opencv/opencv/blob/4.11.0/LICENSE) |
| fused-SSIM commit `328dc9836f513d00c4b5bc38fe30478b4435cbb5` | MIT-licensed upstream objective and test definition inspected for this adaptation. [Upstream license](https://github.com/rahul-goel/fused-ssim/blob/328dc9836f513d00c4b5bc38fe30478b4435cbb5/LICENSE) |
| pytorch-msssim 1.0.0 | MIT-licensed independent value/gradient oracle used before the first fit. [Reference license](https://github.com/VainF/pytorch-msssim/blob/v1.0.0/LICENSE) |
| PyTorch 2.8.0+cu128 and CUDA 12.8 | The native Windows worker uses the existing separately installed PyTorch and NVIDIA runtime. Its original inventory records each version and device independently. [Project dependency notices](../../../THIRD_PARTY_NOTICES.md) |
