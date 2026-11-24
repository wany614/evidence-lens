# T0 reconstruction and scoring register

All 12 registered object–condition pairs completed on their first scientific fit. Each fit ran 5,000 updates and reached 20,000 Gaussians. The Mac verified all 60 optimizer checkpoints and all 108 native frames, including 72 target predictions. The frozen scorer produced four method outputs per target, for 288 entries. The reconstruction and scoring stages had access to the authorized source photographs and recorded zero target-reference reads. [12-row register](cells.csv) · [Input acquisition](../intake/README.md) · [Replay procedure](../reproduction.md)

| DTU object | M2 independent tracks | M4 independent tracks | M2 fit process, s | M4 fit process, s |
| --- | ---: | ---: | ---: | ---: |
| set2 | 44 | 287 | 33.222 | 37.220 |
| set4 | 35 | 266 | 36.057 | 37.574 |
| set10 | 34 | 259 | 38.683 | 36.441 |
| set11 | 13 | 113 | 37.234 | 37.092 |
| set12 | 27 | 235 | 38.469 | 36.378 |
| set31 | 54 | 232 | 36.485 | 37.545 |

M2 used two source cameras and M4 used four, including the same two endpoints. Eleven conditions initialized from their independent known-pose SIFT tracks. Set11/M2 had 13 tracks against a minimum of 24. It used the registered inverse-depth fallback, with 1,024 hypotheses per endpoint, and remained in the six-object cohort. The separate forced-fallback engineering qualification on set1/M2 completed one 5,000-step fit before scientific dispatch. [Initialization protocol](../../../../protocol/recovery-initialization-v1.json) · [Selected optimizer](../../../../protocol/recovery-gsplat-scale-decay-v1.json)

## Reading the register

Each row links six run IDs to the SHA-256 of their complete run manifests. The roles are source acquisition, initialization, Windows fit, Windows render, Mac checkpoint and Case readback, and Mac scoring. The final checkpoint digest links the fit to the renderer. The Case digest links its source photographs and native predictions to the scorer. The scores-index digest identifies the six targets and their four method outputs.

The Windows recorder clock spans **569.394670 seconds** from the first fit start at 01:21:25.102543 UTC to the final render end at 01:30:54.497213 UTC on 24 November 2026. The separate Windows coordinator reported **570.716 seconds** for its outer batch, with ledger SHA-256 `6246bcbd8d574471adc6ebe81a0053a36b0e14e07b3ed6b83a34ac9152671df9`. The coordinator value is transcribed from its return message. The two UTC endpoints came directly from the original run records.

| Clock or memory measure | Total or peak |
| --- | ---: |
| Windows fit subprocesses, sum over 12 | 442.399 s |
| Optimizer loops within those fits, sum | 382.937 s |
| Windows render subprocesses, sum | 47.484 s |
| Mac Case readback, summed recorder times | 19.348 s |
| Mac scoring, summed recorder times | 34.905 s |
| Windows fit peak process-tree RSS | 1,555,664,896 bytes |
| Windows fit peak Torch allocated / reserved | 37,595,136 / 60,817,408 bytes |
| Highest sampled whole-device memory | 974 MiB |

The subprocess clock comes from each worker's process guard. The optimizer loop is nested inside it. Mac recorder times include setup and sealing work up to evidence closure, and several cases ran concurrently. The device sampler includes other GPU activity. The Torch allocator reports within-process allocation and reservation.

## Transfer and storage

The 12 dispatch ZIPs occupied **5,477,632 bytes**. The 12 original fit ZIPs and 12 render ZIPs occupied **183,117,339** and **130,121,815 bytes** respectively. The full set of per-ticket return files occupied **318,071,749 bytes**, counting each original ZIP, receipt, shared input included in that ticket, and its completion file. These application-level totals exclude the independent cumulative ledger and cloud transport overhead. The private transport did not measure network-layer traffic.

The source intake separately measured **63,049,695 HTTP response-body bytes**, including ZIP metadata. The CSV's `source_original_bytes` counts the authorized source PNG lengths for each condition. M4 reused the locally verified M2 endpoints. Summing that column across both conditions counts the two endpoints twice. The read-back digest of the Windows 12-ticket ledger is `159e7ed5d368e6598da2c8a025ff04cc85b98c38e722f974ab7c01b6af8d0722`. [Source transfer ledger](../intake/source-transfers.csv)

## Quarantine and engineering failures

Five first-pass Mac transfers stopped at the same absent cross-store parent, the historical D1 release run `20261123T193505072501Z-6046afc1fce4480b`. Each affected fit ZIP was accepted with its original container and manifest digests after the Mac checked six historical parents, 818 sealed runs, 2,965 provenance edges and 77 shared assets. The cross-store verification run was `20261124T012958649605Z-95c2eb0a75f947f6`, manifest `703f2609bed272d431a6b45c0435e489324b3e4ca474cf619b741e05e73c90d2`. The scientific fits executed once.

| Artifact | Quarantine receipt ID | Accepted receipt ID |
| --- | --- | --- |
| set2/M2 fit | `4da10314337b47bc9545b8f8e29039de` | `8ff62b36c45d443b824e8f5bf49d650f` |
| set2/M4 fit | `d207e1e2d1c843b0895865c176dd4174` | `83f6f57c6e15462a90d00099f89481a5` |
| set4/M2 fit | `c8a0770307f5418fbf475ce662b9f132` | `af5bd6417f7e488d949eb9346d6c68f2` |
| set4/M4 fit | `fddd103c3de54554b35a035d2b4c055c` | `b0a08f3147e0405f8ea1719ad3bc2477` |
| set10/M2 fit | `b9920cd9dbe647ff8dff1d7ad550e740` | `c53af6d7c8614382b2f06f5c8352b758` |

The following engineering attempts are separate from the 12 scientific rows.

| Failed or quarantined run | Recorded reason and correction |
| --- | --- |
| `20261124T010708453996Z-6087d3223b244f32` | The ZIP directory reader called an incompatible Python ZIP64 interface. Revised directory readback succeeded in `20261124T010756050287Z-4877087600c0443f`. |
| `20261124T011153641428Z-9bd6d04824fa422c` and `20261124T011220374432Z-991d742ef3b544e7` | The publication parser matched two dataset tables. The next attempt declared an input without its parent and stopped before worker launch. The DTU-specific parse succeeded in `20261124T011231337470Z-899f229a23a5475c`. |
| `20261124T011605435177Z-613a32695a62471d` | The first forced-fallback Mac readback used a Python environment without Torch. The corrected readback used the same Windows originals and passed in `20261124T011724106134Z-89b47b009fcd435e`. |
| `20261124T011914536604Z-b4c7e1a9b069422f` and `20261124T012136918860Z-fcf0a2091cff46c8` | A bundle fault-injection fixture attempted to write a read-only copied member. Its changed retry was quarantined before worker launch. The corrected fixture passed in `20261124T012857839714Z-8be02b41040240de`. |
| `20261124T013330448584Z-042531c306864803` | The collective opening guard detected a change in the reconstruction controller after its software lock. It stopped before evaluation access. The separately recorded opening with the locked source succeeded in `20261124T013603310781Z-d1ba70c0ba3c4034`. |
