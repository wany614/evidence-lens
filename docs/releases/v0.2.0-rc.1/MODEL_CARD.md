# Model card

Evidence Lens estimates the risk of a display-sRGB prediction error above a registered threshold. Reference Blender images supply independent labels during evaluation. Deployment reads the observed source PNG, calibrated source and target cameras, and per-view predicted display-sRGB, expected camera-axis depth and accumulated alpha. The input encoding and source self-render must match the prediction contract. Invalid observations retain an unknown state.

| Package | Inputs and trained components |
| --- | --- |
| L1 | L1-full uses 26 geometry, appearance, frame and missing-value features. L1-geometry uses the registered geometry subset. The package contains both fitted coefficient sets and train-only transforms. |
| L2-frozen | A plain DINOv2 ViT-S/14 encodes the letterboxed target RGB. A learned 384-to-32 projection and small risk head combine native aligned descriptors with the same 26 deployable features. All base tensors remain fixed. |
| L2-vision-ft | The same input, projection and head. Two final transformer blocks and the terminal LayerNorm also receive gradients from the supervised risk loss. |

The registered protocol uses 28 training families and eight development families. Whole image padding preserves aspect ratio at 252 × 252. Predictions are inverse-mapped to native pixel centres. Seed 1729 provides the primary weights. Seeds 2718 and 3141 measure training variability. The eight recorded candidates, chosen learning rates and development AP are in `evidence/selected-training.json`. Each exported visual model contains its complete FP32 backbone, head, native preprocessing, architecture and standalone CPU scoring code. Training settings, exact update groups, gradient samples, original base verification and before/after tensor hashes are in its `evidence/` directory.

## Trained weights and development results

The primary frozen model is `20261123T131854609280Z-61d909f49ff24a08`. The primary fine-tuned model is `20261123T133146588950Z-c1d19dcb78c74cf0`. Both contain 22,070,817 parameters and each FP32 safetensors file is 88,301,188 bytes. Across the eight formal fits the Windows worker performed 6,132 AdamW updates in 2,693.50 seconds.

| Primary model | Trainable scalars | Updates | Selected epoch | Pretrained scalars changed | Frozen tensors verified identical |
| --- | ---: | ---: | ---: | ---: | ---: |
| L2-frozen | 14,241 | 840 | 19 of 20 | 0 | 175 |
| L2-vision-ft | 3,565,473 | 630 | 12 of 15 | 3,551,228 | 145 |

The Windows 11 RTX 3060 Laptop worker used PyTorch 2.8.0 and CUDA 12.8. Eight supervised minibatches recorded the gradients of the data loss before optimizer weight decay. An independent Mac CPU audit re-read all 181 exported tensors for each candidate. The selected fine-tuned fit peaked at 235,239,936 allocated CUDA bytes and 264,241,152 reserved bytes.

| Seed | Frozen backbone AP | Fine-tuned AP | Difference |
| --- | ---: | ---: | ---: |
| 1729 | 0.690735 | 0.701488 | +0.010752 |
| 2718 | 0.709234 | 0.690149 | −0.019085 |
| 3141 | 0.699380 | 0.685957 | −0.013423 |

These are family-equal APs on the eight development families for the selected learning rates. The mean paired difference was −0.007252 and its sample standard deviation was 0.015847. The preregistered decision retains **L1-full as the default scoring model**. The fixed thresholds for L2-frozen and L2-vision-ft are 0.5428733229637146 and 0.6533921360969543.

## CPU scoring cost

The arm64 Mac measurements used the six native 512 × 512 views of development family `scene_037_bottle_carrier`, with one cold pass and three warm passes over all six views per model. Cold timing starts at process creation and ends when the first complete score file is available. Warm timing includes reading input, numerical features, the full image model, thresholding and writing a view.

| Model | Cold first view | Warm per-view p50 | Warm per-view p95 | Sampled peak process-tree RSS |
| --- | ---: | ---: | ---: | ---: |
| L2-frozen | 1.034 s | 0.206 s | 0.231 s | 477,102,080 bytes |
| L2-vision-ft | 0.645 s | 0.207 s | 0.228 s | 471,564,288 bytes |

The Mac CPU deployment agreed with the development CUDA finite-support masks across all six views. Maximum absolute score differences were 0.00007642 for the frozen model and 0.00005863 for the fine-tuned model. At the registered thresholds, three fine-tuned pixels crossed the decision boundary. The independent Mac release replay reproduced all three fitted models' native reference scores and decisions on this case.

Scores rank pixel risk. Probability calibration has not been established. Operating points use a strict low-risk comparison with the threshold frozen on development B0. The package's default follows the registered development and CPU-cost criteria. New held-out and historical results, all family rows, and Mac CPU timings are in `study/`. Scope covers this set of controlled scene families, cameras and perturbations. The authored plane demonstrates software execution.

The ordinary, register-free DINOv2 model is from Meta. `BASE-SOURCE.json` gives the original weights URL, source revision, original SHA-256 and byte length. `licenses/DINOv2-LICENSE.txt` contains the upstream license. The project additions and supervised weights are credited to wany614.

[Registered upstream source and model card](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/MODEL_CARD.md)
