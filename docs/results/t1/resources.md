# T1 resources and failed attempts

The six objects had 42 registered reconstruction conditions. Twenty-nine physical fits produced 30 successful logical conditions. On set61, risk and uniform used one byte-identical fit. Twelve M4 conditions ended before candidate access. The resource census covers 215 sealed study records preceding the census, including the fixed-risk and reconstruction evaluations and their independent recounts. [Resource receipt](resources.json) · [Every recorded attempt](resource-runs.csv)

| Measurement | Recorded total |
| --- | ---: |
| Cold-start scene fits | 29 |
| Optimizer updates per fit | 5,000 |
| Fit recorder time over 29 original Windows runs | 1,118.671 s |
| Endpoint renderer recorder time over 29 original Windows runs | 162.747 s |
| Frozen query-probe attempts | 6 |
| Recorded time for endpoint rendering and all six probe attempts | 188.779 s |
| Fit dispatch ZIP bytes | 13,664,506 |
| Fit and endpoint return bytes, including completion summaries | 786,005,420 |
| Distinct source photographs fetched | 38 |
| Logical source observations, including previously authorized cached files | 108 |
| Distinct independent reference photographs fetched after opening | 36 |

The stage times sum recorder durations across jobs. Original Windows query receipts identify an NVIDIA GeForce RTX 3060 Laptop GPU with Torch 2.8.0+cu128 and gsplat 1.5.3. The highest sampled whole-device memory usage was 984 MiB. The largest native Torch allocation peak was 37,409,280 bytes and the largest reservation peak was 60,817,408 bytes. These measurements cover different scopes. Each logical condition has separate acquisition costs, and shared init, fit and endpoint durations are charged once. [Per-condition costs](resource-conditions.csv) · [Original probe lineage](probe-lineage.csv) · [Query and selection costs](query-cost.csv)

Two of the three failed recorder runs came from the frozen probe. Set118 and set76 had valid final checkpoints below the required 20,000 Gaussians. The remaining failure occurred when the Mac recorder rejected a diagnostic from the failed set118 run as an undeclared input parent. A separate audit read the hash-verified copies of the original manifest and error log, preserving the scientific failure and its five unavailable strategy outcomes. Neither failed probe ran again. All other 212 recorder runs succeeded. While preparing the census, a local reader initially rejected the top-level inventory list before invoking its recorder. It then read the original canonical inventory bytes. [All 30 strategy receipts](strategy-receipts.json) · [Condition outcomes](conditions.csv) · [Recorded failures](resource-runs.csv)

The joint opening sealed at `2026-11-24T04:47:52.306971Z`. The first reference retrieval started at `04:47:54.522081Z`. The 38 unique source photographs comprised 12 S0 images and 26 distinct candidate images. Set118 and set76 each contributed two S0 photographs and zero candidates. [Per-object permissions and actual access](object-access.csv) · [Physical photo identities and trajectory observations](photo-exposure.csv)
