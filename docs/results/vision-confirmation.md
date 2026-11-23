# Stage B confirmation results

Recorded report `20261123T152543832083Z-c80e50ae1e1d40aa`. Method freeze `68d2035135ccd2e928fd3202ec755c3a3d4da29c7eee9b56970b609fd429a35d`. The primary visual models used the preregistered seed 1729.

A reference pixel is valid (V) when all three display-RGB channels are finite and within [0, 1]. The evaluable set E is the subset of V whose predicted RGB is also valid. An evaluable pixel is a confirmed error when its channel-mean absolute RGB difference strictly exceeds 0.05.

Low risk requires a finite method score strictly below its frozen threshold inside E. Scores at or above that threshold are high. Nonfinite scores inside E are unknown. V outside E contributes to the invalid-output count. Low coverage is low pixels divided by V. Error in low is confirmed errors among the low pixels.

Seven-common AP ranks E pixels with finite scores from all seven methods. Equal scores remain together. AP averages target views containing both classes within each family. Operating rates pool the six target counts within each family. Each mean gives equal weight to families with a defined value. Cells with missing family estimates show the contributing count over the number scheduled.

## New eight-family confirmation

8 families, 24 family–condition pairs and 144 planned target views. Render dispositions were 144 completed.

| Condition | Method | Own AP | Five-common AP | Seven-common AP | Families with seven-common AP | Seven-common coverage | High error on seven-common | Low coverage | Error in low |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | constant | 0.3333 | 0.3311 | 0.3311 | 8 | 99.64% | 33.10% | 100.00% | 33.33% |
| B0 | 1-alpha | 0.3904 | 0.3795 | 0.3795 | 8 | 99.64% | 33.10% | 77.79% | 31.92% |
| B0 | L0 | 0.3813 | 0.3761 | 0.3761 | 8 | 99.64% | 33.10% | 83.44% | 29.73% |
| B0 | L1-full | 0.8194 | 0.8194 | 0.8194 | 8 | 99.64% | 33.10% | 69.79% | 14.67% |
| B0 | L1-geometry | 0.7983 | 0.7983 | 0.7983 | 8 | 99.64% | 33.10% | 68.42% | 14.61% |
| B0 | L2-frozen | 0.8808 | N/A (0/8 families) | 0.8808 | 8 | 99.64% | 33.10% | 68.95% | 10.03% |
| B0 | L2-vision-ft | 0.8993 | N/A (0/8 families) | 0.8993 | 8 | 99.64% | 33.10% | 69.27% | 9.68% |
| R1 | constant | 0.3416 | 0.3370 | 0.3370 | 8 | 99.25% | 33.70% | 100.00% | 34.16% |
| R1 | 1-alpha | 0.4266 | 0.4053 | 0.4053 | 8 | 99.25% | 33.70% | 76.83% | 31.99% |
| R1 | L0 | 0.3941 | 0.3834 | 0.3834 | 8 | 99.25% | 33.70% | 82.93% | 30.19% |
| R1 | L1-full | 0.8241 | 0.8241 | 0.8241 | 8 | 99.25% | 33.70% | 68.49% | 14.67% |
| R1 | L1-geometry | 0.7971 | 0.7971 | 0.7971 | 8 | 99.25% | 33.70% | 67.00% | 14.68% |
| R1 | L2-frozen | 0.8785 | N/A (0/8 families) | 0.8785 | 8 | 99.25% | 33.70% | 67.56% | 9.83% |
| R1 | L2-vision-ft | 0.8925 | N/A (0/8 families) | 0.8925 | 8 | 99.25% | 33.70% | 68.23% | 9.80% |
| Q1 | constant | 0.3333 | 0.3311 | 0.3311 | 8 | 99.64% | 33.10% | 100.00% | 33.33% |
| Q1 | 1-alpha | 0.3905 | 0.3795 | 0.3795 | 8 | 99.64% | 33.10% | 77.79% | 31.92% |
| Q1 | L0 | 0.3813 | 0.3761 | 0.3761 | 8 | 99.64% | 33.10% | 83.44% | 29.73% |
| Q1 | L1-full | 0.8194 | 0.8194 | 0.8194 | 8 | 99.64% | 33.10% | 69.78% | 14.67% |
| Q1 | L1-geometry | 0.7983 | 0.7983 | 0.7983 | 8 | 99.64% | 33.10% | 68.42% | 14.61% |
| Q1 | L2-frozen | 0.8808 | N/A (0/8 families) | 0.8808 | 8 | 99.64% | 33.10% | 68.95% | 10.03% |
| Q1 | L2-vision-ft | 0.8993 | N/A (0/8 families) | 0.8993 | 8 | 99.64% | 33.10% | 69.27% | 9.68% |

| Changed input | Mean paired target RGB absolute change | Largest channel change | Families with paired RGB |
| --- | ---: | ---: | ---: |
| R1 versus B0 | 1.191e-02 | 8.846e-01 | 8 |
| Q1 versus B0 | 1.061e-05 | 1.372e-04 | 8 |

| Condition | Mean paired fine-tuned AP minus frozen encoder AP | Mean paired fine-tuned AP minus L1-full AP |
| --- | ---: | ---: |
| B0 | 0.0185 | 0.0799 |
| R1 | 0.0140 | 0.0684 |
| Q1 | 0.0185 | 0.0799 |

## Historical four-family regression

4 families, 12 family–condition pairs and 72 planned target views. Render dispositions were 72 completed.

| Condition | Method | Own AP | Five-common AP | Seven-common AP | Families with seven-common AP | Seven-common coverage | High error on seven-common | Low coverage | Error in low |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | constant | 0.3248 | 0.3232 | 0.3232 | 4 | 99.75% | 32.32% | 100.00% | 32.48% |
| B0 | 1-alpha | 0.3632 | 0.3549 | 0.3549 | 4 | 99.75% | 32.32% | 77.82% | 31.83% |
| B0 | L0 | 0.4060 | 0.4024 | 0.4024 | 4 | 99.75% | 32.32% | 81.70% | 27.03% |
| B0 | L1-full | 0.8028 | 0.8028 | 0.8028 | 4 | 99.75% | 32.32% | 69.02% | 13.98% |
| B0 | L1-geometry | 0.7694 | 0.7694 | 0.7694 | 4 | 99.75% | 32.32% | 67.79% | 14.60% |
| B0 | L2-frozen | 0.8370 | N/A (0/4 families) | 0.8370 | 4 | 99.75% | 32.32% | 67.75% | 9.51% |
| B0 | L2-vision-ft | 0.8516 | N/A (0/4 families) | 0.8516 | 4 | 99.75% | 32.32% | 69.52% | 10.57% |
| R1 | constant | 0.3393 | 0.3351 | 0.3351 | 4 | 99.32% | 33.52% | 100.00% | 33.93% |
| R1 | 1-alpha | 0.4179 | 0.3985 | 0.3985 | 4 | 99.32% | 33.52% | 76.99% | 32.04% |
| R1 | L0 | 0.4176 | 0.4082 | 0.4082 | 4 | 99.32% | 33.52% | 81.03% | 28.40% |
| R1 | L1-full | 0.8128 | 0.8128 | 0.8128 | 4 | 99.32% | 33.52% | 67.56% | 14.39% |
| R1 | L1-geometry | 0.7679 | 0.7679 | 0.7679 | 4 | 99.32% | 33.52% | 66.41% | 15.23% |
| R1 | L2-frozen | 0.8497 | N/A (0/4 families) | 0.8497 | 4 | 99.32% | 33.52% | 65.77% | 9.43% |
| R1 | L2-vision-ft | 0.8624 | N/A (0/4 families) | 0.8624 | 4 | 99.32% | 33.52% | 67.89% | 10.71% |
| Q1 | constant | 0.3248 | 0.3232 | 0.3232 | 4 | 99.75% | 32.32% | 100.00% | 32.48% |
| Q1 | 1-alpha | 0.3632 | 0.3549 | 0.3549 | 4 | 99.75% | 32.32% | 77.82% | 31.83% |
| Q1 | L0 | 0.4060 | 0.4024 | 0.4024 | 4 | 99.75% | 32.32% | 81.70% | 27.03% |
| Q1 | L1-full | 0.8028 | 0.8028 | 0.8028 | 4 | 99.75% | 32.32% | 69.02% | 13.98% |
| Q1 | L1-geometry | 0.7694 | 0.7694 | 0.7694 | 4 | 99.75% | 32.32% | 67.79% | 14.60% |
| Q1 | L2-frozen | 0.8370 | N/A (0/4 families) | 0.8370 | 4 | 99.75% | 32.32% | 67.75% | 9.51% |
| Q1 | L2-vision-ft | 0.8516 | N/A (0/4 families) | 0.8516 | 4 | 99.75% | 32.32% | 69.52% | 10.57% |

| Changed input | Mean paired target RGB absolute change | Largest channel change | Families with paired RGB |
| --- | ---: | ---: | ---: |
| R1 versus B0 | 1.475e-02 | 8.835e-01 | 4 |
| Q1 versus B0 | 1.063e-05 | 1.352e-04 | 4 |

| Condition | Mean paired fine-tuned AP minus frozen encoder AP | Mean paired fine-tuned AP minus L1-full AP |
| --- | ---: | ---: |
| B0 | 0.0146 | 0.0488 |
| R1 | 0.0128 | 0.0497 |
| Q1 | 0.0146 | 0.0488 |

## Development selection and resource measurements

The development rule selected **L1-full** as the deployment default before opening the new families. The eight candidate fits contributed 6,132 recorded optimizer steps and 0.75 worker-hours.

| Seed-A model | Cold process to first output | Warm p50 | Warm p95 | Peak process-tree RSS |
| --- | ---: | ---: | ---: | ---: |
| L2-frozen | 1.03 s | 0.21 s | 0.23 s | 0.44 GiB |
| L2-vision-ft | 0.65 s | 0.21 s | 0.23 s | 0.44 GiB |

Cold latency spans process creation, interpreter and library imports, model loading, source and prediction decoding, feature extraction, inference and the first complete output file. Each warm percentile uses 18 completed outputs across six views.

The memory column uses the larger of two sampled process-tree RSS peaks. Both peaks and both cold clock readings are recorded in the numerical cost file. The two sampling intervals are shown below.

| Model | External observer | Native scorer |
| --- | ---: | ---: |
| L2-frozen | 10.0 ms | 10.0 ms |
| L2-vision-ft | 10.0 ms | 10.0 ms |

The scientific selection histories list one distinct unsuccessful attempt. The cost inventory records measured recovery and engineering entries separately. CUDA allocated and reserved bytes describe the PyTorch allocator. The CPU memory figure is sampled process-tree RSS.

These results describe the registered static opaque scenes and nearby calibrated views. Risk scores rank RGB error under that protocol. The eight new families include two preregistered groups of four. Complete group estimates and native family risk–coverage points are available in the study summary.

The numerical archive contains the study summary, per-family denominators, paired changes, execution costs and the frozen development decision. [Download the public tables](https://github.com/wany614/evidence-lens/releases/download/v0.2.0-rc.1/evidence-lens-v0.2.0-rc.1-numerical.zip) · [Open all family risk–coverage curves](../../examples/vision-study-figures/README.md)

## New acquisition and reused records

The 40 new scheduled conditions comprised 16 pre-opening B0 conditions and 24 confirmation conditions. The learning datasets also reused 20 earlier B0 acquisitions. Selected new renders supplied 280 source and target view files.

The Windows ledger records 32 distinct generation attempts and 40 distinct render attempts. Generation wall times sum to 27.04 minutes across 32/32 timed attempts. Render wall times sum to 5.54 minutes across 40/40 timed attempts. Missing clock readings contribute no duration to the sum. A shared generation counts once across condition references.

## Engineering and release evidence

These records describe engineering or release work outside the two scientific cohorts.

| Activity | Recorder outcome | Wall time |
| --- | --- | ---: |
| Official encoder verification | failed | 0.97 s |
| Official encoder verification | succeeded | 2.00 s |
| Acquisition pilot and budget | succeeded | 2.69 s |
| GPU engineering pilot | succeeded | 12.47 s |
| Development CPU and CUDA numerical readback | succeeded | 1.05 s |
| Readback of three engineering and release events | succeeded | 0.91 s |
| Pre-opening B0 acquisition costs | succeeded | 1.31 s |

The initial official-source inventory check exited before the verification completed.
The 4-family acquisition pilot recorded a decision to proceed.
The GPU engineering pilot completed 32 optimizer steps. The formal fit roster contains 8 separate runs.
A Windows disk limit deferred outgoing packaging for one development trial. The Mac subsequently received the original generation and render.
The isolated Mac scorer exited successfully. Python bytecode files changed the source inventory and its recorder package failed verification.
The public example scoring request failed its input lineage check before its scorer process started. A separate selection derivation supplied the verified input for a successful repeat.

An independent worker checked the original native pixels and all seven methods for 216 target views. The two cohort checks contain 4,104 average-precision comparisons. [Read the pixel audit and its denominators](vision-pixel-audit.md)
