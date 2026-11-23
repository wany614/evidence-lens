# Known issues and scope

The per-build diagnostic finding, remaining uncertainties, formal result status and original evidence digests are in [Scientific scope](SCIENTIFIC_SCOPE.md).

| Question | Evidence to use |
| --- | --- |
| Does the install reproduce the authored demonstration? | Clean CPU smoke with six 128 × 128 plane views and five score methods. Inspect its measured differences, masks and decisions. |
| Which pixels of the study had large RGB error? | The selected SHARP/gsplat formal predictions, independent Blender labels and held-out family tables. Consult the scope before interpreting these results. |
| Does a valid expected-depth probe imply accurate geometry? | A probe tests coordinate and rendering semantics. Errors relative to reference geometry are measured separately. Low support and unknown pixels retain their own denominators. |
| Can a reader reproduce the historic private run bytes from this ZIP alone? | This ZIP supplies a deployable model, an authored example, numerical tables and record identities. Original SHARP predictions, references, formal boards and recovery receipts are held in the research archive. |
| How should platform evidence be recorded? | Each run writes its own Python, operating system, architecture and dependency versions. A prior candidate's OS test is historical evidence for that specific artifact. |

Small nearby-view synthetic cohorts offer limited evidence about other scenes and cameras. A high score is an inspection aid. A low score provides limited assurance about RGB error, and depth reconstruction needs its own metric. [Model card](MODEL_CARD.md) · [Formal numerical tables](results/family-results.csv)
