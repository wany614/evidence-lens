# Resources for the D0 readback and D1 DTU study

D0 completed 2 recorded runs and retained 1 failed setup. D1 completed 62 runs and retained 7 failed executions through 2026-11-23 20:10:01 UTC. The resource report that reads this inventory has its own receipt below. Each recorder execution has one row with its original manifest, producing host platform, purpose, status, timing, telemetry sample count and archived sizes. [Per-run measurements](runs.csv) · [Stage totals](stages.csv)

| Scientific operation | Successful runs | Sum of recorder wall (s) | Sum of worker-reported stage (s) | Largest sampled process-tree RSS (MiB) | CUDA allocated peak (MiB) | CUDA reserved peak (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Windows optimization, 256 updates | 4 | 22.45 | 11.05 | 1129.5 | 10.8 | 14.0 |
| Windows optimization, 1,024 updates | 2 | 14.92 | 9.20 | 1113.8 | 10.8 | 14.0 |
| Windows render and camera probes, main arms | 4 | 16.11 | 4.61 | 819.5 | 4.7 | 8.0 |
| Windows render and camera probes, diagnostics | 2 | 7.75 | 2.22 | 793.7 | 4.7 | 8.0 |
| Mac frozen scores, four DTU conditions | 4 | 62.62 | 9.23 | 499.3 | unmeasured | unmeasured |
| Mac frozen scores, published control | 2 | 6.43 | 4.47 | 498.5 | unmeasured | unmeasured |

The main arms used one two-source and one four-source reconstruction for each of two physical objects, producing 24 target and 12 source self-render views. The separately registered development diagnostic added two 1,024-update reconstructions, 12 targets and six source self-renders. The six 4,096-Gaussian scene fits performed 3,072 optimizer updates in total. The four released risk scorers retained their published weights and thresholds. The five metadata and camera-layout replays appear under the public-entry stage. [Acquisition by arm](resource-summary.json) · [Study registration](../../../protocol/dtu-external-v1.json)

Recorder wall time uses each producer’s monotonic clock from setup through scientific result collection. GPU workers also report an inner stage duration and a separate Gaussian optimization duration. The score worker’s inner timer includes release and case validation, loading the four models, scoring all six target views, then writing and rereading the numerical arrays. Stage sums measure accumulated work across runs, some of which overlapped. The CSV keeps process-tree RSS sampling, device-wide VRAM sampling, process GPU telemetry and the two Torch allocator peaks in separate columns. Blank cells indicate unavailable or inapplicable measurements. WDDM did not supply per-process GPU bytes for these Windows stages. [Per-run timing boundaries and measurements](runs.csv)

| Unsuccessful run | Trial | Recorded symptom | Worker started |
| --- | --- | --- | --- |
| `20261123T193226173907Z-69f75b788afe4c7a` | d0-linux-arm64-rc1 | host worker launch | false |
| `20261123T194703263864Z-5ac8b5fe39524e30` | d1-dtu-set1-M4-256-case | capacity scan transient file | false |
| `20261123T194757204317Z-4e39fd9e376040db` | d1-dtu-set1-M4-256-case | unsuitable input parent | false |
| `20261123T200338919476Z-e45043f592e14fee` | official-object-index-1-and-6 | capacity scan transient file | false |
| `20261123T200339001422Z-0419b7b1bbee474b` | set6-truth-download | capacity scan transient file | false |
| `20261123T200554619963Z-ce41ea40604d4526` | d1-joint-report | report input schema lookup | true |
| `20261123T200718124875Z-1c7f0083fce64d7d` | d1-joint-report | retry changed frozen inputs | false |
| `20261123T200732205798Z-a9df8117aba14d39` | d1-joint-report-catalogue-evidence | unsuitable input parent | false |

7 failures stopped before their stage worker launched, including the quarantined request to reuse a report identity after its frozen input changed. The joint reporting worker started and then failed during a schema lookup. The two post-gate preflights for the catalogue identity and set6 reference download transferred zero official HTTP body bytes and produced empty payloads. Their later successful runs carry new IDs. The original Linux acceptance, acquisition, conversion, camera, scoring, evaluation, replay and report records remain in the inventory. [Complete status table](runs.csv)

The disposable Debian container received 426,466,254 bytes and transmitted 1,237,749 bytes while downloading and installing the public release and running the CPU replay. The independent DTU acquisitions and public metadata replays read 79,839,006 official HTTP entity-body bytes. Three private worker dispatch containers held 2,458,653 bytes. Their 12 sealed returns plus producer receipts and summaries held 73,637,884 bytes. The mounted exchange supplied the exact logical file lengths and SHA-256 hashes. The provider’s network traffic was unmeasured. Mac archive receiving spent 204.86 seconds cumulatively validating 12 original run containers, including recursive parent readback. The project’s registered network allowance was 20 GiB. [Per-file transfer and archive receipts](transfers.csv) · [Linux receipt](../../verification/v0.3.0-rc.1/linux/linux-resources.json)

At the cutoff the Mac research store occupied 12.76 GiB of logical files after accounting for hard links. The volume then reported 32.33 GiB free. Across the recorded Mac preflights the highest observed active store was 12.76 GiB and the lowest observed volume free space was 32.00 GiB. Windows batch stores reached 0.06 GiB at a preflight and their volume had at least 484.89 GiB free at those observations. The study used a 20 GiB active target and 30 GiB free-space floor. Per-run archive sizes count bytes named by each manifest, including copies of shared inputs and source snapshots. [Storage and archive evidence](resource-summary.json)

The sealed resource run is `20261123T201005859902Z-5bf15dd1975549e0`, manifest SHA-256 `ba9212f787bc6efdf1883750355918b3e055ede759320745729068184fd176c7`. Its inventory SHA-256 is `ab94234621fe71f7d058dd6058d28635ab8f990c9d1309bd1fdf9564433ce80c`. The private record binds the original run IDs, including failed setups, and the three transfer certificates. Rebuilding these tables requires those entries, the sealed run roots and the input archive receipts. [Resource reader](../../../scripts/dtu_resources.py) · [Return verifier](../../../scripts/dtu_receive.py)
