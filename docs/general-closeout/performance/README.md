# CPU measurements and static replay

The registered engineering workloads cover a sparse helical bead sculpture and the denser offset bell cage. Both have two-source and four-source inputs at 256 × 256. Gaussian rendering of new views uses gsplat 1.5.3 on CUDA. A prepared bundle supplies all images, depths, opacities and cameras needed for CPU scoring. [Registered measurement protocol](../../../protocol/general-static-engineering-v1.json)

## What one measurement includes

The file baseline invokes the released scoring entry on one prepared target. It verifies the bundle, loads the model and sources, computes the target features and model, and writes a compressed numerical result and prediction image. The report records HTML generation separately. The two resident configurations load the same model once and use bounded exact caching or a zero-byte cache. Each measured frame computes its own target features and scores. Source snapshots, camera invariants, immutable input verification and output digests are included at their actual boundaries. [Measurement worker](../../../scripts/g3_perf.py)

G3 gives each platform a 1,800-second measurement budget. Each of the two methods and four original workloads has a file, cached-array and uncached-array configuration. Each configuration requests three new-process cold starts, a warm-up and 200 timed serial calls. The first-result stamp runs from before the parent spawns Python to the instant the first numerical result returns or is persisted. Steady-state file service runs through numerical persistence. Array service runs from entering the resident call through the returned result and its identity. Feature, model, postprocessing, identity checks, source preparation, file decoding and serialization have separate fields. The raw rows identify their actual view and maintain the full pixel partition. [Originals registration](../../../scripts/g3_perf_originals.py)

The G3 input set reuses exact earlier CUDA renders. Both bead layouts use the same SH3 asset. The four-source version adds authentic analytic surface photographs from cameras 70 and 109 and their existing self-renders. Camera 13 remains the prediction target. The two bell-cage layouts retain their respective original fixed reconstruction checkpoints. This four-source workload represents an engineering input count. [Source and asset origins](../../../examples/helical-beads/README.md)

The original-replays download contains all 53 G3 workload members byte-for-byte, including the registration and all four bundles. Its `workloads.json` is at the archive root and the four bundle directories are under `inputs/`. For the source measurement script, copy the registration and those four directories into a fresh workload directory and pass its path as `--trajectories`. [Download the G3 original workloads and reference outputs](https://github.com/wany614/evidence-lens/releases/download/v0.4.0/evidence-lens-original-replays-v1.zip)

The neural thread candidates use 1, 2, 4 and a capped hardware-based count, with duplicate values removed. A candidate must reproduce the four-thread reference words on the registered development M4 input to qualify. The setting controls PyTorch for MS-ft. The MS-L1 feature path uses NumPy. Actual backend configuration, observed operating-system thread counts, process-tree CPU seconds, available logical CPUs and sampled peak RSS accompany each configuration. CPU utilization is expressed with one fully occupied logical CPU equal to 100%. The 100 ms RSS samples can miss brief peaks. [Resource observations](../../../scripts/g3_perf_resources.py)

## Measured CPU service

Both platforms completed all 24 configurations within their individual 1,800-second budgets. The Apple M6 used 413.43 seconds and the Windows 11 Intel machine used 1,360.83 seconds. Every configuration supplied 200 hot calls and three independent cold starts, giving 9,600 hot calls and 144 cold starts. All hot results matched the same-platform, same-target file words for float32 risk, validity, state and unknown reason. [All configurations](g3-configurations.csv) · [Native source receipts and sample counts](tables.json)

On bell M4, Mac MS-ft median service at one, two, four and eight threads was 89.56, 86.85, 72.95 and 69.51 ms, respectively, with eight calls per candidate. All four candidates matched the four-thread file words on this development input. The equivalent Windows medians were 351.11, 251.57, 220.18 and 228.99 ms. Only four threads matched all eight Windows reference calls. The two platforms therefore used eight and four neural threads for the rest of the registered platform measurements. The separate scientific readback tests the reach of the Mac configuration beyond the development workload. [Thread-candidate samples](g3-thread-candidate-frames.csv) · [Independent thread comparison](../t1-threads8-numerical-bridge.json)

| Platform, input and method | File p50 ms | Resident, cache disabled p50 ms | Resident, cache enabled p50 / p95 ms | Cached feature / model p50 ms | Cached mean busy logical CPUs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mac, bell M4, MS-L1 | 92.80 | 47.74 | 40.12 / 44.56 | 31.73 / 6.18 | 0.99 |
| Mac, bell M4, MS-ft | 203.10 | 80.95 | 73.50 / 84.94 | 32.84 / 39.59 | 1.56 |
| Mac, beads M2, MS-L1 | 35.46 | 24.62 | 20.64 / 24.53 | 18.47 / 1.20 | 0.99 |
| Mac, beads M2, MS-ft | 141.89 | 54.22 | 49.57 / 54.14 | 19.00 / 28.96 | 1.66 |
| Windows, bell M4, MS-L1 | 283.83 | 141.72 | 127.60 / 141.54 | 98.54 / 26.55 | 1.02 |
| Windows, bell M4, MS-ft | 629.09 | 230.02 | 218.17 / 228.59 | 98.23 / 117.02 | 1.57 |
| Windows, beads M2, MS-L1 | 127.20 | 79.16 | 72.55 / 84.54 | 65.49 / 4.81 | 0.99 |
| Windows, beads M2, MS-ft | 455.34 | 147.34 | 141.25 / 162.59 | 64.31 / 81.50 | 1.80 |

Bell M4 supplied a score for 99.61% of its measured pixels. Beads M2 supplied one for 16.97%. The differing projection and valid-pixel workloads are visible in both timings. Mac MS-L1 spent most of its cached service constructing NumPy features and averaged about one busy logical CPU out of twelve. The neural configuration spent similar time in feature construction and model evaluation, and responded to the PyTorch thread setting. Persistent inputs also removed repeated file decoding and model loading from measured frame service. On bell M4, peak sampled process RSS moved from 155.7 to 207.5 MiB for MS-L1 and from 484.5 to 434.6 MiB for MS-ft, file to cached. [Every configuration, stage, sample count and memory measurement](g3-configurations.csv) · [Measured CPU backends](g3-platform-backends.csv)

The Windows machine had 14 physical and 20 available logical CPUs, NumPy 2.5.5 with OpenBLAS and PyTorch 2.8.0+cpu. Its cached MS-L1 service spent 98.54 ms of the 127.60 ms bell M4 median building features. The process averaged 1.02 occupied logical CPUs. Median neural feature and model stages took 98.23 and 117.02 ms. Peak sampled RSS on bell M4 moved from 127.7 to 146.2 MiB for MS-L1 and from 420.0 to 412.9 MiB for MS-ft, file to cached. The effect of a parallel implementation of this NumPy feature path remains unmeasured. [Backend and CPU data](g3-platform-backends.csv) · [Per-configuration resource measurements](g3-configurations.csv)

On bell M4 the three Mac process-start-to-first-result measurements for cached MS-L1 were 201, 162 and 162 ms. Cached MS-ft took 673, 579 and 596 ms. The corresponding Windows runs took 688, 641 and 625 ms for MS-L1, and 2,984, 2,922 and 2,969 ms for MS-ft. Those include dependency imports, the initial prepared inputs and model setup. The archive contains the raw hot-frame measurements in `g3-frames.csv`. [Individual cold starts](g3-cold-starts.csv) · [Download all raw measurements](https://github.com/wany614/evidence-lens/releases/download/v0.4.0/evidence-lens-cpu-replay-measurements.zip)

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

## Sustained delivery and frame age

Each row below uses the dense four-source bell trajectory for one 60-second window. Actual Hz is the number of intervals between the first and last enqueue divided by the elapsed time. Delivered/s counts finished outputs across the wall-clock load and drain. Service, queue wait and age are measured separately for each serviced input. Age starts at actual prepared-input readiness. The CSV also gives the producer's lateness relative to the planned schedule. [All load percentiles, denominators, RSS and cache usage](g4-loads.csv)

| Platform, method | Target / actual Hz | Delivered / offered | Dropped | Expired / failed | Delivered/s | P95 service / wait / age ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mac, MS-L1 | 10 / 10.0004 | 600 / 600 | 0 | 0 / 0 | 10.00 | 45.30 / 0.04 / 45.35 |
| Mac, MS-L1 | 30 / 30.0004 | 1,503 / 1,800 | 297 | 0 / 0 | 25.04 | 43.76 / 33.32 / 73.57 |
| Mac, MS-ft | 10 / 9.9999 | 600 / 600 | 0 | 0 / 0 | 10.00 | 78.02 / 0.04 / 78.69 |
| Mac, MS-ft | 30 / 29.9956 | 809 / 1,800 | 991 | 0 / 0 | 13.47 | 79.07 / 33.79 / 109.36 |
| Windows, MS-L1 | 10 / 9.9990 | 466 / 600 | 132 | 2 / 0 | 7.76 | 141 / 94 / 219 |
| Windows, MS-L1 | 30 / 29.9988 | 470 / 1,800 | 1,330 | 0 / 0 | 7.82 | 141 / 47 / 172 |

Every main load offered 120 unique poses. The remaining 480 or 1,680 arrivals repeated those views. Delivered frames constitute the displayed-result denominator. Replaced pending inputs have a disposition and replacement identity in `g4-frames.csv`. Result latency percentiles describe scored inputs, including expired scores when present. Each p99 is the observed sample percentile of that run. Per-frame unknowns and their exclusive reasons are counted over the full 65,536-pixel canvas. The archive also includes `g4-offline-frames.csv` with an independent file comparison for every distinct view. [Download frame and parity ledgers](https://github.com/wany614/evidence-lens/releases/download/v0.4.0/evidence-lens-cpu-replay-measurements.zip)

On the Mac, independent file scoring and resident scoring agreed for every view in the four 120-view sequences for both models and the two additional 24-view sequences for MS-L1. The six-second revision load delivered all 60 arrivals. Preparing the first context, changing the Gaussian map and changing the source observation took 7.89, 9.30 and 8.72 ms. At the map change, all 24 corresponding target arrays and all four source self-renders changed. After changing only the source evidence, all 24 same-map target arrays matched the preceding map snapshot. The context missed the cache at all three admissions. [File-to-array parity by sequence and model](g4-offline-parity.csv) · [Version events and rebuild latency](g4-version-events.csv) · [Measured version changes](g4-version-checks.csv)

Windows matched all 528 distinct MS-L1 file comparisons. On the 10 Hz pressure run, 468 frames entered service and two exceeded the 250 ms age limit. The 30 Hz run serviced 470 arrivals. The higher input rate replaced pending evidence more often and lowered the measured waiting age for frames reaching service. The shortest nonzero separation between observed enqueue, start and finish clock readings in the 30 Hz Windows ledger was 15 ms. Windows G3 uses the separate high-resolution service timer. [Full sustained-load statistics](g4-loads.csv)

The Windows version test delivered 53 of its 60 arrivals, replaced five pending frames and expired two in-flight scores whose evidence version had been superseded. The three preparations took 15, 16 and 16 ms. Both native runs observed the same actual target and source changes. The 53 delivered frames, 55 scored frames and 60 offered frames have separate denominators. [Version result counts and array changes](g4-version-checks.csv) · [Prepared contexts and cache misses](g4-version-events.csv)

The 30 Hz Mac MS-L1 run used 11,018,072 bytes of its 64 MiB cache, with 1,504 hits, two misses and zero evictions including warm-up. The version test retained six entries and peaked at 33,054,282 bytes. Host process RSS also includes Python, NumPy, any loaded model runtime and the predecoded trajectories. The load table gives start, end and peak RSS for each measured interval. [Cache and process measurements](g4-loads.csv)

The checked sample HTML shows the last delivered Mac MS-L1 frame at 30 Hz, `ingress-001799`, corresponding to bell M4 view `f119`. It includes predicted RGB, continuous risk, threshold state and unknown reason together with the recorded latency table. Its matching array file is included with the downloadable measurements. [Offline replay example](../../../examples/general-static-replay/report.html)

## Original trajectory archive and measured upstream cost

The complete original archive is `evidence-lens-static-trajectories.zip`, 329,268,781 bytes, SHA-256 `933d16cd27549618bc7adf43a1a86988daaf187fc06282a9d21775ac9a37b806`. Extract it before running the replay commands above. It contains the four 120-view bundles, the two 24-view alternative-evidence bundles, original timing records and attribution. Each bundle was verified against its input ticket, snapshot, authorized source photographs, declared camera matrices, raw frame digests and expected metric depth. The maximum adjacent translation on either full path was 23.1 mm. [Bundle identities and trajectory checks](g4-trajectory-verification.json)

The prepared original archive can be downloaded separately from the scoring package. Its `bundle.json` files pin the permitted source photos and generated frames. All input paths in the replay commands are relative to the extracted archive directory. [Download the prepared trajectory bundles](https://github.com/wany614/evidence-lens/releases/download/v0.4.0/evidence-lens-static-trajectories.zip)

A separate CLI check unpacked the public archive in a fresh directory and ran the replay module with relative paths. Two frames on the two-source bead sequence and three across the map/source switch finished and matched the independent file words in all four arrays. The switch created three source contexts. The five control calls contribute solely to command and input verification. [Portable archive CLI readback](g4-portable-cli-control.json)

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

Both platforms use the same 22,084,023-byte CPU ticket, SHA-256 `670867a78456fcd6ca4dacfaac307c5d5aa570cd198c3d32a8df53d160846104`. It verifies the included inference modules against the frozen four-method numerical bridge, SHA-256 `a31f0caef9bee1f17bb98c8cdb5feef544c916ad32806b6f46360f8051412c64`. Extract the ticket and run the commands below through the recorder using separate empty payloads. `CPU_RELEASE` contains the pinned `release.json` and original models. `TRAJECTORIES` contains `trajectories.json` and the six generated bundles. Substitute `mac` for `windows` on the Mac. Performance and research jobs have separate device windows. [CPU ticket builder](../../../scripts/replay_cpu_ticket.py)

Copying the CPU ticket into the local synchronized folder and checking its digest took 0.0204 seconds. The Windows G3 return container was 228,976,929 bytes and took 52.239 seconds to receive and verify on the Mac. The Windows G4 container was 640,243,751 bytes and took 140.418 seconds at the same boundary. Network-only transfer duration was unobserved on all three operations. [Transfer costs and original recorder handles](cpu-transfer-costs.json)

```sh
python -I -B run_gpu.py --out GPU_PAYLOAD
python -I -B run_cpu.py --operation g3 --platform windows --models CPU_RELEASE --out G3_PAYLOAD
python -I -B run_cpu.py --operation g4 --platform windows --models CPU_RELEASE \
  --trajectories TRAJECTORIES --selection G3_PAYLOAD/selection.json --out G4_PAYLOAD
```

The reporting command takes the native stage payload directories alongside their recorder manifests and completions. It verifies each manifest digest and the sizes and digests of all 420 raw members used or inventoried across the four native closures. The process rechecks frame digests, queue replacement events, sample quantiles, output counts, unknown reasons, and the HTML and numerical arrays in the published example. Each load's two monotonic stopwatch stamps record the planned origin and actual stop after the producer exits and the queue drains. Individual enqueue, start and finish times remain in the frame ledger. The portable ZIP is 3,880,237 bytes and carries its CC BY 4.0 attribution in `tables.json`. [Projection program](../../../scripts/g3_perf_tables.py) · [Member hashes and source receipts](tables.json) · [Expected download checksum and size](public-archive.json)

```sh
python scripts/g3_perf_tables.py --mac-g3 MAC_G3 --mac-g4 MAC_G4 \
  --windows-g3 WINDOWS_G3 --windows-g4 WINDOWS_G4 --out measurement-tables
```
