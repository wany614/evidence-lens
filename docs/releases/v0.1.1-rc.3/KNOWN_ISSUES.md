# Known issues and scope

The original 512 × 512 baseline had family-equal depth MAE **1.14585 m**. On the frozen depth-evaluable population, **99.4002%** of pixels exceeded the tolerance in the family-equal summary. The pooled counts are 6,214,493 above tolerance among 6,252,055 comparable pixels out of 6,291,456 reference surfaces. Frontmost Blender surfaces and opacity-weighted Gaussian expected depths can represent different contributions along a ray. The current model predicts **RGB-error risk**. Metric reconstruction accuracy and performance on open backgrounds, real photos and other generators remain unverified.

A known-geometry Windows GPU control and its independent Mac regrade passed all 335 registered checks, including PLY serialization, rotated cameras, pixel centres, colour and depth/alpha semantics. Source-view depth ratios varied across the sampled families. No correction factor was fitted. The first diagnostic attempt failed while serializing metadata and is preserved in the research archive.

The per-build diagnostic finding, remaining uncertainties, formal result status and original evidence digests are in [Scientific scope](SCIENTIFIC_SCOPE.md).

| Question | Evidence to use |
| --- | --- |
| Does the install reproduce the authored demonstration? | Clean CPU smoke with six 128 × 128 plane views and five score methods. Inspect its measured differences, masks and decisions. |
| Which pixels of the study had large RGB error? | The selected SHARP/gsplat formal predictions, independent Blender labels and held-out family tables. Consult the scope before interpreting these results. |
| Does a valid expected-depth probe imply accurate geometry? | A probe tests coordinate and rendering semantics. Errors relative to reference geometry are measured separately. Low support and unknown pixels retain their own denominators. |
| Which parts of the historical study can a reader inspect? | This ZIP supplies a deployable model, an authored example, numerical tables and record identities. The tagged source includes the [original drum-kit input and a Blender target reference](https://github.com/wany614/evidence-lens/tree/v0.1.1-rc.3/examples/heldout-drum). The research archive holds the complete reference sets, raw SHARP predictions, formal boards and recovery receipts. |
| How should platform evidence be recorded? | Each run writes its own Python, operating system, architecture and dependency versions. A prior candidate's OS test is historical evidence for that specific artifact. |

Small nearby-view synthetic cohorts offer limited evidence about other scenes and cameras. A high score is an inspection aid. A low score provides limited assurance about RGB error, and depth reconstruction needs its own metric. [Model card](MODEL_CARD.md) · [Formal numerical tables](results/family-results.csv)
