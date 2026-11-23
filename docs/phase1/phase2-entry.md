# Research handoff after the L1 study

Stage one verified the single-image RGB-risk measurement for nearby views of static, opaque studio scenes. The original L1 learned its coefficients from project data. A raw-pixel replay confirmed its decisions and independently reproduced its average precision. The auxiliary depth MAE remains 1.145853 m across the four original test families. Its cause remains unresolved. Those four families now serve as historical regression cases. [Measured results and scope](README.md) · [Original operating rule](../../protocol/operating-points-v1.json)

## B · Single-image visual fine-tuning

The next study asks whether adapting a pretrained visual encoder improves RGB-error ranking or the risk among accepted pixels on new families. The existing five methods supply baselines. **L2-frozen** trains a projection and risk head on fixed visual features. **L2-vision-ft** updates existing encoder weights and the head using the project's RGB-error labels. The proposed pilot starts with the general DINOv2 ViT-S/14 encoder, a full-frame input and its last two transformer blocks unfrozen. The SHARP generator stays fixed.

| Roadmap budget | Scope |
| --- | --- |
| New families | 12 training, 4 development, 8 unseen test. The first 4 training pilots count toward the 12. |
| Conditional reuse | With compatible protocols and documented training rights, 16 prior training and 4 prior development families bring the totals to 28 and 8. The 28 training families supply 168 B0 target images. |
| Conditions | B0 for all 24 new families. The eight unseen families also receive R1 and Q1. This gives 40 scientific conditions, 32 generator calls and 280 rendered source or target views, before retries or timing repeats. |
| Neural training | An engineering pilot of up to 32 updates. At most eight formal runs across the frozen and fine-tuned models, using two learning-rate candidates and three paired seeds for the selected configurations. |

The family is the independent unit. Before opening the eight test families, freeze the camera and background rules, generator and checkpoint identities, train/development selection, score thresholds, deployment choice and failure accounting. Preregister four test families with new layouts from known asset classes and four with previously unseen geometry templates or material classes. Open these families once, after both baselines and visual methods are frozen. A separately licensed training line using another generator needs its own generator ID, condition definitions, pilot and resource measurements.

A completed fine-tuning run needs a pinned pretrained checkpoint and license, task-loss gradients on previously trained encoder layers, before/after tensor measurements, a frozen-prefix check, and an exported model that matches the training implementation. Measure the complete CPU input-to-output path. The original NumPy L1 remains available. A performance claim requires the paired unseen-family results, including negative results and seed variation.

## C · Joint multi-image reconstruction and risk learning

A separate experiment changes how the **scene** is built. Within each family, M2 fits one Gaussian scene from two source images and M4 fits one scene from four. The source sets are nested and the planned camera span is held constant. Both conditions use the same backend and initialization policy, with matched total update and source-pixel budgets. Reserve the six target images for evaluation. Reconstruction loss, checkpoint choice and risk-scoring inputs use only the permitted sources and their predictions.

The default budget is 16 new families, split into 8 training, 4 development and 4 unseen test families. Two training pilots count toward the eight. M2 and M4 together require 32 scene reconstructions and 288 rendered views. Training contributes 96 target views across the two conditions. The pilot starts with ceilings of 50,000 Gaussians and 5,000 optimization updates per reconstruction. The neural comparison has a separate ceiling of five formal training runs. Optional one- or eight-source controls require a separately frozen budget.

Reconstruction and risk fine-tuning produce separate run IDs and checkpoints. The multi-source risk model consumes order-independent summaries of source support, missing evidence and conflict. Record both the sources used for reconstruction and those supplied to the scorer. Freeze the reconstruction comparison, risk-model controls, thresholds and displayed views before opening this stage's four test families.

These are planned workloads. Each stage reports actual acquisition, training, independent test results and cost under its own data and model identities. The authorized publication of the original L1 has its own candidate, checksum and downstream CPU verification. [Release procedure](../model-release.md) · [Asset scope](../licenses.md)
