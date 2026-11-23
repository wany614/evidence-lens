# Model card

Evidence Lens estimates the risk of a display-sRGB prediction error above a registered threshold. Reference Blender images supply independent labels during evaluation. Deployment reads the observed source PNG, calibrated source and target cameras, and per-view predicted display-sRGB, expected camera-axis depth and accumulated alpha. The input encoding and source self-render must match the prediction contract. Invalid observations retain an unknown state.

| Package | Inputs and trained components |
| --- | --- |
| L1 | L1-full uses 26 geometry, appearance, frame and missing-value features. L1-geometry uses the registered geometry subset. The package contains both fitted coefficient sets and train-only transforms. |
| L2-frozen | A plain DINOv2 ViT-S/14 encodes the letterboxed target RGB. A learned 384-to-32 projection and small risk head combine native aligned descriptors with the same 26 deployable features. All base tensors remain fixed. |
| L2-vision-ft | The same input, projection and head. Two final transformer blocks and the terminal LayerNorm also receive gradients from the supervised risk loss. |

The registered protocol uses 28 training families and eight development families. Whole image padding preserves aspect ratio at 252 × 252. Predictions are inverse-mapped to native pixel centres. Seed 1729 provides the primary weights. Seeds 2718 and 3141 measure training variability. The eight recorded candidates, chosen learning rates and development AP are in `evidence/selected-training.json`. Each exported visual model contains its complete FP32 backbone, head, native preprocessing, architecture and standalone CPU scoring code. Training settings, exact update groups, gradient samples, original base verification and before/after tensor hashes are in its `evidence/` directory.

Scores rank pixel risk. Probability calibration has not been established. Operating points use a strict low-risk comparison with the threshold frozen on development B0. The package's default follows the registered development and CPU-cost criteria. New held-out and historical results, all family rows, and Mac CPU timings are in `study/`. Scope covers this set of controlled scene families, cameras and perturbations. The authored plane demonstrates software execution.

The ordinary, register-free DINOv2 model is from Meta. `BASE-SOURCE.json` gives the original weights URL, source revision, original SHA-256 and byte length. `licenses/DINOv2-LICENSE.txt` contains the upstream license. The project additions and supervised weights are credited to wany614.

[Registered upstream source and model card](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/MODEL_CARD.md)
