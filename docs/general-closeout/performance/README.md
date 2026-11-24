# CPU measurements and static replay

The registered engineering workloads cover a sparse helical bead sculpture and the denser offset bell cage. Both have two-source and four-source inputs at 256 × 256. Gaussian rendering of new views uses gsplat 1.5.3 on CUDA. A prepared bundle supplies all images, depths, opacities and cameras needed for CPU scoring. [Registered measurement protocol](../../../protocol/general-static-engineering-v1.json)

## What one measurement includes

The file baseline invokes the released scoring entry on one prepared target. It verifies the bundle, loads the model and sources, computes the target features and model, and writes a compressed numerical result and prediction image. The report records HTML generation separately. The two resident configurations load the same model once and use bounded exact caching or a zero-byte cache. Each measured frame computes its own target features and scores. Source snapshots, camera invariants, immutable input verification and output digests are included at their actual boundaries. [Measurement worker](../../../scripts/g3_perf.py)

G3 gives each platform a 1,800-second measurement budget. Each of the two methods and four original workloads has a file, cached-array and uncached-array configuration. Each configuration requests three new-process cold starts, a warm-up and 200 timed serial calls. The first-result stamp runs from before the parent spawns Python to the instant the first numerical result returns or is persisted. Steady-state file service runs through numerical persistence. Array service runs from entering the resident call through the returned result and its identity. Feature, model, postprocessing, identity checks, source preparation, file decoding and serialization have separate fields. The raw rows identify their actual view and maintain the full pixel partition. [Originals registration](../../../scripts/g3_perf_originals.py)

The G3 input set reuses exact earlier CUDA renders. Both bead layouts use the same SH3 asset. The four-source version adds authentic analytic surface photographs from cameras 70 and 109 and their existing self-renders. Camera 13 remains the prediction target. The two bell-cage layouts retain their respective original fixed reconstruction checkpoints. This four-source workload represents an engineering input count. [Source and asset origins](../../../examples/helical-beads/README.md)

The neural thread candidates use 1, 2, 4 and a capped hardware-based count, with duplicate values removed. A candidate must reproduce the four-thread reference words on the registered development M4 input to qualify. The setting controls PyTorch for MS-ft. The MS-L1 feature path uses NumPy. Actual backend configuration, observed operating-system thread counts, process-tree CPU seconds, available logical CPUs and sampled peak RSS accompany each configuration. CPU utilization is expressed with one fully occupied logical CPU equal to 100%. The 100 ms RSS samples can miss brief peaks. [Resource observations](../../../scripts/g3_perf_resources.py)

## Replay a prepared bundle

Use the installed engineering package in an environment with NumPy and Pillow. Supply a fresh output directory. A normal replay run predecodes and verifies the target arrays before releasing them at the requested rate. For MS-ft, install the optional vision dependencies and supply the extracted, verified CPU model distribution using `--models`.

```sh
python -m evidence_lens_bundle.replay --bundle beads-M4 --hz 10 --seconds 60 --out my-replay
python -m evidence_lens_bundle.replay --bundle beads-M4 \
  --switch 20=beads-map2-M4 --switch 40=beads-map2-source2-M4 \
  --hz 10 --seconds 6 --out my-switch
```

Open `replay.html` for the last delivered result and latency percentiles. `frames.csv` assigns every arrival an input identity and terminal status. `replay.json` records configured and achieved load, repeated views, stage timings, cache events, pixel denominators and capability declarations. `last-result.npz` contains the four numerical arrays for the associated delivered frame when one is available. [Replay implementation](../../../src/evidence_lens_bundle/replay.py)

The producer offers frames at scheduled monotonic times. One serial scorer can hold one additional pending frame. A newer arrival replaces the older pending frame. A completed result expires if its evidence snapshot has been superseded or its age exceeds 250 ms. A failed score, an expired result and a discarded pending input remain separate ledger entries. Service is dequeue to numerical result. Wait is enqueue to dequeue. Input-ready latency starts when predecoded evidence enters the live replay. The offline CUDA render and cross-device transfer have their own cost boundaries. [Scheduled load and parity verification](../../../scripts/g4_replay_eval.py)

Formal G4 first compares every registered view with independently persisted file scoring. The four main trajectories each contain 120 distinct positions on a smooth closed camera path. Each representative bell-cage M4 load then offers 10 or 30 arrivals per second for 60 seconds. Mac runs MS-L1 and MS-ft. Windows runs MS-L1. A separate six-second version test moves the bead evidence from the original SH3 asset to SH0, then replaces one interior camera and photograph. Each served result is checked against its offline numerical digest. A repeated trajectory view remains a repeated engineering load.

## Original trajectory archive and measured upstream cost

The complete original archive is `evidence-lens-static-trajectories.zip`, 329,268,781 bytes, SHA-256 `933d16cd27549618bc7adf43a1a86988daaf187fc06282a9d21775ac9a37b806`. Extract it before running the replay commands above. It contains the four 120-view bundles, the two 24-view alternative-evidence bundles, original timing records and attribution. Each bundle was verified against its input ticket, snapshot, authorized source photographs, declared camera matrices, raw frame digests and expected metric depth. The maximum adjacent translation on either full path was 23.1 mm. [Bundle identities and trajectory checks](g4-trajectory-verification.json)

The RTX 3060 Laptop GPU worker took 27.656 seconds for all 548 frames, including verification and serialization. The per-frame synchronization and device-to-host delivery accumulated 3.172 seconds for 528 target renders and 0.128 seconds for 20 source self-renders. NPZ compression, writing and hashing took another 17.063 seconds. Peak allocated CUDA memory was 13,753,344 bytes. The 764,238-byte dispatch took 0.0037 seconds to copy into the local synchronized folder. The return recorder container was 331,979,380 bytes, and the receiving Mac took 61.560 seconds to acquire and verify it. The provider's network-only duration is unknown. [Raw upstream frames](g4-upstream-frames.csv) · [Resource and transfer boundaries](g4-upstream-cost.json)

## Optional sidecar

`--sidecar FILE.json` accepts a versioned JSON object. A minimal declaration follows.

```json
{
  "schema": "el-static-replay-sidecar-1",
  "clock_domain": null,
  "acquisition_timestamp": null,
  "pose_covariance": null,
  "dynamic_mask": null,
  "source_selection_policy": "authorized static source roster",
  "task_schema": "display_srgb_mean_absolute_photometric_error_gt_0.05"
}
```

The registered original inputs contain no acquisition timestamps, pose covariances or dynamic masks. To attach metadata to a particular observation, add a `views` object keyed first by sequence ID and then by target ID. Each target entry can override the document defaults. The frame ledger includes the effective sidecar hash and capture-clock domain. Acquisition time is optional. A supplied finite numeric timestamp requires its actual clock domain. The replay scheduler always records its own monotonic planned, ready, enqueue, start and finish times. An optional covariance includes a finite 6 × 6 `matrix`, a `coordinates` string identifying the coordinate frame and axis order, and a `units` string. An optional dynamic mask contains `reference`, `source`, `time` and `clock_domain`. Its file stays unopened. The risk calculation consumes the registered RGB photometric evidence. Covariance, mask metadata and the stated selection policy remain recorded provenance. Unknown metadata stays null. These interfaces provide no evaluated pose, dynamics, semantic or geometry estimator. [Sidecar checks and capability declarations](../../../src/evidence_lens_bundle/replay_sidecar.py)

## Recorder workers

The original GPU ticket is 764,238 bytes, SHA-256 `534a119024fec41f372ee5e3bce0e7fd05ff4712e9011c864f3926d654661036`. Its Windows worker runs in the existing gsplat 1.5.3 CUDA environment. It verifies every input, synchronizes each render and records the GPU-to-host transfer within that render stage. Compression and filesystem writes are separate per-frame measurements. [Trajectory ticket builder](../../../scripts/trajectory_pack.py) · [CUDA worker](../../../scripts/trajectory_render.py)

Both platforms use the same CPU ticket. Its dispatch receipt identifies all code, original workloads and the registered protocol. Extract the ticket and run the commands below through the recorder using separate empty payloads. `CPU_RELEASE` contains the pinned `release.json` and original models. `TRAJECTORIES` contains `trajectories.json` and the six generated bundles. Substitute `mac` for `windows` on the Mac. Performance and research jobs have separate device windows. [CPU ticket builder](../../../scripts/replay_cpu_ticket.py)

```sh
python -I -B run_gpu.py --out GPU_PAYLOAD
python -I -B run_cpu.py --operation g3 --platform windows --models CPU_RELEASE --out G3_PAYLOAD
python -I -B run_cpu.py --operation g4 --platform windows --models CPU_RELEASE \
  --trajectories TRAJECTORIES --selection G3_PAYLOAD/selection.json --out G4_PAYLOAD
```
