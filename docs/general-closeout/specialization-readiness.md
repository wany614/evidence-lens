# What a later specialization would need

The current task is pixelwise risk of a display-RGB error. A source photograph, its self-render and the proposed view's RGB, expected camera-Z and alpha constitute the evidence for the installed models. A separate evaluator may obtain a target photograph after outputs are sealed. This division also applies to a future experiment with different sensors or labels. [Evidence Bundle contract](../tooling/README.md) · [Replay metadata and capability declarations](../../src/evidence_lens_bundle/replay_sidecar.py)

| Later question | Reusable part | Evidence and new evaluation required |
|---|---|---|
| Does high risk identify wrong geometry or depth? | Calibrated metric cameras, depth support and immutable view IDs | Independently measured aligned surfaces or depth, uncertainty and visibility, a registered geometric label and disjoint scene families. Evaluate geometric error with its own denominators. |
| How sensitive is scoring to pose uncertainty? | Camera conventions and the optional covariance field | Identify axis order, units and distribution in a named frame. Compare against externally measured camera poses and evaluate the resulting effect on scores in a separately registered study. |
| Can motion be distinguished from reconstruction error? | Frame identity, map/source revisions, named clocks, queue logs and dynamic-mask provenance | Synchronized observations of a changing scene, independent motion labels, trajectories split by recording session and a separate evaluation of latency and temporal errors. |
| Does the score support a semantic decision? | Source provenance, explicit feature and model identities, separate evaluation API | A defined task, annotated outcomes, an agreed utility and separate domain training, development and test cohorts. |
| Can another backend or device serve the same model? | File reference, four returned arrays, threshold and tie-aware comparison | Run the exact model on the candidate hardware, verify arrays and accepted pixel identities, then repeat bounded load and memory measurements on that device. |
| Could a robot or vehicle act on such an output? | A timestamped, identity-bound observation that an independent consumer can log | A specified sensor and clock contract, domain-specific ground truth, safe operational limits and an independent systems evaluation before any control integration. |

## Optional fields in a replay record

The sidecar schema is `el-static-replay-sidecar-1`. Frame, sequence, model, renderer, map and source identities bind a result to the actual inputs. The photometric task identifier selects the registered RGB contract. A missing acquisition timestamp, clock, pose covariance, dynamic mask or source-selection policy is represented by `null`. Supplied timestamps require a named clock. The declared 6 × 6 covariance names its coordinate frame and units. The dynamic-mask field stores an unopened reference, its source and optional timestamp/clock. These optional metadata values are carried through the record. Their influence on scoring and predictive validity are both untested. Queue timing uses the local monotonic service clock.

Deployments outside the observed scene, lighting and source layouts will need separate evidence. The present static sequences provide no measurements of motion, geometric correctness, semantic decisions or physical control. Dedicated data acquisition, model fitting, hardware optimization and downstream control require a new study.
