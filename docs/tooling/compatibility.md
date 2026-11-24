# Prepared-bundle compatibility

The installed **0.4.0a2** CLI reproduced the four released models on the two original bell-cage reconstructions. Every finite FP32 score, finite mask and threshold state matched the previously distributed expected arrays. A separate process executed the unmodified numerical modules from the Stage C release on the same Mac. All FP32 words, including NaN locations and bit patterns, matched its fresh results.

| Comparison | Method × view entries | Canvas method-pixels | Finite pairs | Maximum absolute difference | State differences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Published original, M2 and M4, four methods | 48 | 3,145,728 | 3,113,007 | 0 | 0 |
| Historical recovery, set6 M4, four methods | 24 | 1,572,864 | 1,505,995 | 0 | 0 |
| Official capture IDs, previously exposed set1 M4, two linear methods | 12 | 786,432 | 719,176 | 0 | 0 |

The first two comparisons used wheel 0.4.0a2. The third exercised the native-ID interface on the preceding 0.4.0a1 candidate. Its source order was **24, 23, 17, 22**, its primary camera was **24**, and its common endpoints were **24 and 22**. The six target identifiers remained **16, 18, 25, 37, 35, 36**. The later candidate adds authenticated historical Case metadata when the original exporter omitted its producer IDs or camera conventions. The private set6 proof authenticated the selected checkpoint, both Windows producers, ten original numerical renders, the depth probe, the calibration, the crop and the millimetre-to-metre conversion. Neither private check read target reference photographs. These are engineering replays of exposed objects. [Recorded checks](verification.json)

## Coverage and cost on the original scene

Counts below use six 256 × 256 target canvases per row, a denominator of 393,216. The elapsed time measures one fresh macOS process running the installed score command for all six targets, including model loading, computation and writing its numerical and preview outputs. Each row is one observation. The two visual methods ran with four CPU threads. No timing subtraction, GPU substitution or training is involved.

| Sources | Method | Low | High | Unknown | Command seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| M2 | B-L1-transfer | 170,017 | 223,199 | 0 | 0.959 |
| M2 | MS-L1 | 275,068 | 112,099 | 6,049 | 0.435 |
| M2 | MS-frozen | 272,642 | 114,525 | 6,049 | 2.094 |
| M2 | MS-ft | 275,302 | 111,865 | 6,049 | 1.625 |
| M4 | B-L1-transfer | 175,247 | 217,969 | 0 | 0.375 |
| M4 | MS-L1 | 316,145 | 72,213 | 4,858 | 0.523 |
| M4 | MS-frozen | 322,577 | 65,781 | 4,858 | 1.630 |
| M4 | MS-ft | 325,516 | 62,842 | 4,858 | 1.709 |

The minimal Mac environment contained only Evidence Lens, NumPy 2.5.5 and Pillow 12.3.0. A separate visual environment used PyTorch 2.8.0 and safetensors 0.8.0 on CPU. The full scoring stage took 21.23 s and sampled 441,647,104 bytes of peak process-tree RSS. This stage included releases, clean environments and eight separate commands. The pure-Python tool wheel is 349,682 bytes, the portable original example ZIP is 12,798,898 bytes, and the self-contained M4 HTML is 958,001 bytes. The resource ledger includes elapsed seconds, logical stage bytes, and sampled RSS where available. [Per-stage resource ledger](stage-costs.csv)

## Missing evidence and offline display

Ten deliberate alterations or unavailable options were explicitly rejected, including an input reference role, a changed camera pose, an unauthorized model profile, changed input bytes, a missing declared render, a wrong metric unit and ray distance presented in place of expected camera Z. A declared failed target retained all 65,536 pixels as unknown. An unverified depth producer left all 393,216 pixels unknown across six targets. These cases exercise the interface and add zero scientific samples.

The original fault injector initially inherited the read-only permissions of the archived source and exited before injecting its first fault. The recorder quarantined a requested retry that included changed checker code. A new engineering revision made writable temporary copies and passed all twelve checks. [Disposition records and original hashes](verification.json)

Chrome 157 opened the final report with network access disabled. All 28 source and target-panel images decoded at 256². Six individual view choices revealed the corresponding target and the seventh choice revealed all six. The browser fetched only `file:` and `data:` URLs. The HTML also embeds the original numerical receipt for download. [Offline original report](../../examples/offset-bell-cage/bundle-report.html)

## Reproduce the comparison

The released CPU ZIP is identified by SHA-256 `f555956840870309217cc85fe9b339ef53762679bbd2138b7f149d8ea451a7b7` and the original example ZIP by `14204986e0d598adc613dd1a9ef2e5c1584f5be001d3541531d5d9e709707548`. The verification ledger identifies the original ingestion and score receipts by run, manifest and member SHA-256. The quick start gives the exact M4 Case hash. Use the portable grading command for the six M4 MS-L1 score arrays, or replay the two complete Cases using the archived process and the new score products. [CPU quick start](README.md) · [Independent four-method comparison worker](../../scripts/bundle_regression.py) · [Original CPU release](../releases/v0.3.0-rc.1/README.md)
