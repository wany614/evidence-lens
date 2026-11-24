# Batch-two reconstruction

The six calibration objects each produced a native two-source and four-source reconstruction. On the two previously exposed development objects, five camera-selection policies each chose two extra photographs before opening either image. The primary optimizer seed was 43171. Each development object also ran two fixed-layout M4 sensitivity checks, with seeds 43172 and 43173.

| Scope | Logical observations | New fits | Source layout |
| --- | ---: | ---: | --- |
| C6 calibration | 12 | 12 | Fixed M2 or fixed M4 |
| D camera selection | 10 | 9 | M4-selected |
| D optimizer sensitivity | 4 | 4 | Fixed M4 |
| D starting points | 2 | 0 | Original, shared M2 |

The calibration acquisitions used set48, set75, set110, set42, set77 and set52. Seven passed the registered triangulation gates. Five two-source cases used the 2,048-point inverse-depth fallback. The independent track counts in those cases were 3, 7, 5, 5 and 17 for set48, set75, set110, set77 and set52, respectively. All twelve completed the original 5,000-step schedule. The shared endpoint baseline was 0.2442371192480833 m.

In fixed M4, logical source slots s00–s03 correspond to official cameras 24, 23, 17 and 22. In M4-selected, the endpoints stay at 24 and 22. The two jointly selected camera IDs occupy s01 and s02 in ascending order. On set6, uniform coverage and risk-weighted coverage selected the same pair, 17 and 31. Their two decision and source-access records share one initialization, Windows execution and native prediction. Each of the ten policy observations recorded a separate, authorized fetch of its two images. The HTTP response bodies totalled 57,981,229 bytes.

The original source-acquisition receipts inherited a four-ID fixed-layout description in their `S0` display field. Their recorded selector inputs, camera members, joint decisions and fetch logs establish that only cameras 24 and 22 supplied photographs before selection. The development index records the two-slot mapping, per-observation receipt times, official member ranges and separate policy authorizations.

## Native outputs and measured costs

Each readback checked the generator source and calibration, the five issued checkpoints at steps 0, 500, 1,500, 3,000 and 5,000, the five Adam parameter groups, source update counts and the terminal checkpoint selected by the renderer. It reopened the six target predictions and the source self-renders, checked RGB, opacity, expected camera depth and PNG round-trip, and emitted a deployable case for the separate evaluation. The reconstruction and intake processes opened zero target reference photographs.

All thirteen physical development fits passed the triangulation gates. Across the nine distinct selected layouts, initial point counts ranged from 75 to 255. Both fixed-layout optimizer seeds reused the original 271 points on set1 and 333 on set6. Every selected M4 run read four source views, with 1,250 updates per view.

| Measured group | New fitted runs | Sum of fit recorder wall time | Sum of render recorder wall time | Returned object bytes including each terminal summary |
| --- | ---: | ---: | ---: | ---: |
| C6 fixed layouts | 12 | 452.326 s | 63.235 s | 299,808,891 |
| D selected layouts and seed checks | 13 | 504.469 s | 72.421 s | 372,796,069 |

Across both groups, 25 distinct fits used 956.795 s of summed recorder wall time. Their 25 endpoint render stages used 135.656 s, and the 25 independent Mac readbacks used 48.544 s. The Windows optimizer's largest Torch peaks were 37,294,592 allocated bytes and 60,817,408 reserved bytes. The largest sampled device-wide reading during optimization was 978 MiB. Per-process GPU memory was unavailable on the Windows receipts. The three dispatch batches contained 12,287,766 ZIP bytes. Returned files plus each terminal summary totalled 672,604,960 bytes. Independent source HTTP logs measured 43,966,976 body bytes for C6 and 57,981,229 for D.

All 50 Windows optimizer and renderer stages succeeded. The Mac accepted 50 original run bundles and completed 25 numerical readbacks. The five registered fallback initializations produced completed C6 cases. The one D alias accounts for a shared fit. One delivery of the selected-pair registry used a different local mirror; projecting the same nine tickets into the standard private exchange corrected delivery. The two manifests and the issued job hashes identify the same workload. Network traffic for that local mirror correction was unmeasured.

The C6 numerical index is recorder run `20261124T021119570759Z-85cf68d825e142e3`, manifest `10a8e9aa5abcb4d2941ec862de44250a06dbb07211f9f39e97628ceedd8c7b6b`, member SHA-256 `5f588c4c522bc005c270076224b90ded9c260b0740bd16868a5beaca21d828ac`. The D index is run `20261124T021937740306Z-df1cbacfb273417a`, manifest `e4e998a0b11050bee5e82bb87c60fb834e7cef5743bed1cf4542ca70b35d2080`, member SHA-256 `c54c69d3c342df23e43d1785882c750ace6f7e0bbdf260d5948ef90b11e218f0`.

The two machine-readable indices retain run IDs, manifest and member hashes, source qualifications, Gaussian counts, stage durations, return sizes and memory measurements. A duration summed over stages is elapsed time for the listed stages. Simultaneous Mac work, queue waits and network overhead have separate boundaries. Torch allocator peaks and sampled device memory measure different quantities. A JSON null denotes an unavailable counter. Development includes shared historical M2 optimizer and renderer timings. The original renderer measured its full checkpoint grid. The active-view evidence gives the separate query-rendering and policy-choice costs. [C6 reconstruction index](c6-reconstructions.json) · [Development branches](development-branches.json) · [Active-view measurements](../active-views/)

## Replaying from authorized originals

The recorder needs the immutable parent stores and the permitted DTU source data. The public JSON files contain numerical records and identifiers. The original source photographs, reconstruction files, deployment cases and historical recorder stores remain private. For exact replay, set `STORE` to the batch-two recorder root, `FIRST` to the batch-one recorder root and `HISTORY` to the historical research root. Use the same SIFT geometry environment for initialization and Torch 2.8.0 for the independent CPU checkpoint readback.

On the Mac, the following commands reproduce the fixed batch registry from its selected source runs and expose the seven permitted M2 selector members for set1. Replace `set1` with `set6` for its second starting packet. The selected-pair controller checks both sealed joint decisions before opening the named official photographs. It emits one ticket per unique ordered source layout and seed.

```sh
for seed in 43171 43172 43173
do
  python3 scripts/b2_reconstruct.py --store "$STORE" --historical-store "$HISTORY" register --seed "$seed"
done
python3 scripts/b2_reconstruct_batches.py C6-fixed-12 --store "$STORE" --history "$HISTORY" --exchange "$HANDOFFS" --geometry-python "$SIFT_PYTHON"
python3 scripts/b2_reconstruct_batches.py D-sensitivity-4 --store "$STORE" --history "$HISTORY" --exchange "$HANDOFFS" --geometry-python "$SIFT_PYTHON"
python3 scripts/b2_reconstruct_batches.py selector-packet --store "$STORE" --history "$HISTORY" --object set1 --python "$TORCH_CPU_PYTHON"
python3 scripts/b2_reconstruct_branches.py --store "$STORE" --history "$HISTORY" --authority "$STORE/work/controller/b2-reconstruction-seed-43171/selected.json" --exchange "$HANDOFFS" --geometry-python "$SIFT_PYTHON" --set1 "$SET1_DECISION_ID_AND_MANIFEST" --set6 "$SET6_DECISION_ID_AND_MANIFEST"
```

For the Windows queue, supply the project CUDA Python interpreter, the registered filename and its SHA, a local recorder store, the private return directory and the existing GPU slot. The launcher verifies the original worker and each source member. Successful and failed completions can both be read back by their original receipts.

```powershell
& $env:EL_CUDA_PYTHON scripts/b2_worker_queue.py --registry $Registry --sha256 $RegistrySHA --store $RecorderStore --returns $Returns --gpu-slot $GpuSlot
```

On the Mac, first record the deep historical verification for the exact dispatch registries. Use the completed Windows ledger with the corresponding registry, digest and history receipt. Each invocation accepts the completed prefix and reuses prior authorized readbacks. The public indices can then be reproduced from the sealed parent runs and portable transport receipts.

```sh
python3 scripts/b2_reconstruct_history.py register --name fixed --store "$STORE" --batch-one "$FIRST" --history "$HISTORY" --registry "$HANDOFFS/C6-fixed-12.json" --registry "$HANDOFFS/D-sensitivity-4.json"
python3 scripts/b2_reconstruct_history.py register --name selected --store "$STORE" --batch-one "$FIRST" --history "$HISTORY" --registry "$HANDOFFS/D-selected-unique.json"
python3 scripts/b2_reconstruct_receive.py --store "$STORE" --batch-one "$FIRST" --history "$HISTORY" --registry "$REGISTRY" --registry-sha "$REGISTRY_SHA" --proof "$HISTORY_RECEIPT" --returns "$RETURNS" --python "$TORCH_CPU_PYTHON"
python3 scripts/b2_reconstruct_index.py seal-c6 --store "$STORE" --history "$HISTORY"
python3 scripts/b2_reconstruct_index.py seal --store "$STORE" --history "$HISTORY"
python3 scripts/b2_reconstruct_index.py export --store "$STORE" --history "$HISTORY" --out "$EXPORT_DIR"
```
