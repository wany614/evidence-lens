# Recording and receiving experiments

Install the `record` extra in the Python 3.11+ environment that starts the recorder. The worker can use a different executable, for example the environment containing Blender. Store live runs, staging, receipts and SQLite on a local non-synchronized volume. Project commands use `.research/` by default.

## Start a run

The recorder allocates a UTC timestamp and random run ID before starting the stage. A run has a project, experiment, trial, stage, parent IDs and optional retry ID. Set `--experiment` and `--trial` explicitly or provide them in the JSON config's `identity` object. Optional identity fields cover scene family, scene/version/hash, camera set/hash, dataset/hash, model/hash, condition and split. The recorder copies the resolved configuration and the explicitly declared input bytes before invoking the child. The supplied stage config must declare its scientific settings. The recorder expands its own defaults under `recording`.

The following command illustrates a local reference run. Run it from the checkout with the recorder environment active. Substitute the actual reference interpreter on the machine.

```sh
python -m evidence_lens.recorder \
  --root .research --kind scene_render --config configs/reference.json \
  --experiment p1-reference --trial tabletop-v001 -- \
  /absolute/path/to/reference/python -m evidence_lens.blender_reference.produce \
  --config '{config}' --out '{out}'
```

Check the worker's own CLI before selecting its arguments. The following substitutions operate on individual arguments. The recorder calls the child directly.

| Argument | Value supplied to the child |
| --- | --- |
| `{source}` | Frozen executable source tree. This is also the working directory. |
| `{config}` | Absolute path of `config.resolved.yaml`, serialized as UTF-8 JSON, a YAML 1.2 subset. |
| `{out}` | Fresh `payload/` directory. |
| `{run_id}` | The actual scientific run ID, also available in `EL_RUN_ID`. |
| `{python}` | The interpreter running the recorder. |
| `{input:ROLE}` | Frozen file or directory selected by `--input ROLE=PATH`, or object selected by `--shared-input ROLE=FILE`. |

Each invocation prints a JSON summary including the run ID, exact path, terminal status, exit code, failure reason and manifest SHA-256. A child failure produces a nonzero CLI exit and a sealed failed bundle. Stage preparation failures preserve the launch request and exception, including a missing input or a disk preflight failure. An invalid provenance claim is retained under `quarantine/`. The project code should use the declared input substitutions when opening data. The recorder runs ordinary local processes with the caller's operating-system permissions.

Optional identity values must be strings or null. Hashes use 64 lowercase hexadecimal characters. The recorder checks these fields before allocating a run. A crashed worker's partial camera file is retained byte-for-byte and listed among its artifacts. Only a successful run contributes camera records. If a payload contains a nonportable name, link or special file, the recorder seals a failed run and moves the complete original payload into the producer's `quarantine/payloads/RUN_ID/`. The run records the reason, original payload location and producer receipt.

### Sources, inputs and lineage

The private run contains the exact selected source bytes, a SHA-256 for every file, an aggregate snapshot hash and the available Git commit/dirty state. It executes from that snapshot with its `src` directory on `PYTHONPATH`. In a Git checkout the collector considers tracked files and non-ignored new files under the explicit executable/configuration roots in `_snapshot.py`. A portable source handoff uses the same root/extension allowlist. Data, virtual environments and private research history live outside these roots. The public-source candidate and the Windows worker handoff have their own explicit distribution lists.

Supply one `--input` per role. A directory stores the exact included file set. The receiver checks each relative path, size and SHA-256. Each `--parent RUN_ID` pins the complete local upstream manifest. An input selected directly from `STORE/runs/PARENT/...` inherits that parent and requires it to be declared. For externally handed-off input bytes, use both `--parent RUN_ID=MANIFEST_SHA256` and `--input-parent ROLE=RUN_ID`. This lets a Windows producer receive the approved source PNG and cameras without holding the private reference bundle. On receipt, the Mac checks the actual upstream bundle and verifies that every bound input digest occurs among its assets.

Use `--shared-input` for a large individually licensed checkpoint. It is stored once in `STORE/assets/sha256/` and referenced by digest from each run. Obtain that exact asset on the receiving machine under its applicable terms, then register it before receiving the dependent bundle.

```sh
python -m evidence_lens.archive add-asset /path/to/checkpoint --root .research --sha256 "$CHECKPOINT_SHA256"
```

A scientific retry needs `--retry-of FAILED_RUN_ID`, the same scientific arguments, configuration, exact source snapshot, input assets and pinned parents. The first retry gets a new run ID and preserves its predecessor. Changed scientific settings or code require a new version condition. A transport replay reuses the sealed run and gets its own transfer receipt.

The scientific retry allowance starts when the child process starts. After an initial preflight failure, launch the original command again. If a retry preflight fails, repeat it with the original failed scientific run ID. The failed preflight records that requested ID. An embedded directory input can have a new temporary outer name on retry. Its role, upstream binding, child-relative paths and file contents must match.

### Capacity and telemetry

The recorder checks the project quota and free-disk floor before write-heavy stages and input copies. `configs/storage.json` supplies the local policy. Use `--storage-policy PATH` for an explicitly chosen volume policy. Set `recording.expected_output_bytes` from a measured earlier batch when available. Its default of zero records the absence of an output estimate. The preflight counts declared input copies, observed active bytes and actual free bytes. Receive preflights allow for the container and unpacked scratch copy. A deferred transfer gets a separate receipt. Stage estimates and the preflight sample do not reserve disk space throughout a run.

Telemetry samples the child and observed descendants every second by default. `--interval` overrides the sampling interval. Each row records process IDs, resident memory, CPU interval usage, system memory, available Mac pressure level and available device-wide NVIDIA samples. The file's field dictionary gives units, source and scope. Blank values have field-level reasons. Sampled RSS peaks can miss short-lived subprocesses and can double-count shared pages across live processes. CUDA framework peak allocated/reserved bytes, synchronized GPU timings and worker-side phase events belong in the worker payload.

## Seal and verify

| Record | Contents |
| --- | --- |
| `run.json`, `config.resolved.yaml`, `environment.json`, `command.json` | Identity, timestamps, producer status, failure, local runtime, resolved settings, argument vector and working directory. |
| `source/`, `code_snapshot.json`, `inputs/`, `inputs.json` | Frozen project sources, input roles/bytes, shared objects, parent manifest hashes and retry link. |
| `events.jsonl`, `telemetry.csv`, `telemetry_schema.json`, `stdout.log`, `stderr.log` | Sequenced UTC/monotonic events, resource samples and raw worker logs. |
| `payload/`, `artifacts.json` | Stage output and catalog of relative paths, asset IDs, formats, sizes and completeness status. Stage metadata carries units, array layouts, validity masks and camera conventions. |
| `metrics.parquet`, `metrics.csv` | Typed and human-readable metric rows generated from optional `payload/metrics.jsonl`. Stages without metrics emit empty typed tables. Counts use integers. Values and numerators support fractional sums. |
| `manifest.json`, `completion.json` | Frozen file catalog and external dependency references, followed by the final exit code and manifest hash. |

The recorder closes evidence, writes the artifact catalog, hashes all bundle members into the manifest, writes completion last, verifies the result, marks the files read-only and moves the completed directory to `runs/RUN_ID`. An abrupt stop can leave a directory under `work/`. Preserve it for diagnosis. An unfinished work directory cannot enter the canonical index through `rebuild`. Independent receipts stay outside sealed runs.

```sh
python -m evidence_lens.archive verify ".research/runs/$RUN_ID"
python -m evidence_lens.archive pack ".research/runs/$RUN_ID" --output "/path/to/transfer/$RUN_ID.zip"
python -m evidence_lens.archive receive "/path/to/transfer/$RUN_ID.zip" --root /local/receiver --sha256 "$ZIP_SHA256"
python -m evidence_lens.archive rebuild --root /local/receiver
```

Packing creates a reproducible stored ZIP and an external `packed_local` receipt. ZIP contents retain the individual file hashes. The receiver stages the candidate, verifies every internal member and parent/shared asset, and records a separate accepted or identical receipt. Receive parent runs in dependency order. Same-ID/different-manifest, malformed, missing, changed and extra files go to quarantine. A valid existing run remains addressable by its original manifest.

At the producer, run `rebuild --root STORE` to project the local sealed collection into `index/index.sqlite3`. Rebuilding verifies the full graph and atomically replaces the disposable projection from `runs/` and `receipts/`. The index includes runs, lineage, shared/output/input assets, metrics, scene/camera records, datasets, models and both receipt kinds. The canonical receiver needs the full upstream graph for a rebuild. A producer with only a pinned external upstream can still pack and send its sealed result to that receiver.

Local packing and receipt of a local copy establish no cloud retrieval result. At the cloud recovery stage, the archive needs actual remote object identity, an independent download, whole-container and internal verification, index reconstruction and metric recomputation. Preserve the independent original until that exercise succeeds.

## Freeze the first verified P2 result

After the Mac independently recomputes the fixed view, supply the seven actual run IDs. The selector checks the frozen parents and inputs, successful reference readback, Windows transfer receipts, seven calibrated camera products, the source self-render measurement, six scored targets, the v01 comparison board and the independent v01 checks. Each target retains its measured RGB and depth denominators, including zero. For a registered trial it pins the preregistration as well.

Selection follows the original scientific generation, render, score and evaluation order. Within each stage, recorded UTC start, completion and run ID determine precedence. An earlier successful execution whose next stage or independent check is unresolved blocks the freeze. Documented scientific failures and failed acceptance checks remain visible. Independent verification confirms the eligible scientific result regardless of which evaluation an operator checked first.

```sh
python scripts/select_trial.py --store "$STORE" --trial "$TRIAL_ID" \
  --reference-run "$REFERENCE_ID" --readback-run "$READBACK_ID" \
  --generation-run "$GENERATION_ID" --render-run "$RENDER_ID" \
  --score-run "$SCORE_ID" --evaluation-run "$EVALUATION_ID" --recompute-run "$RECOMPUTE_ID"
```

The immutable local output is `STORE/trials/TRIAL_ID/trial_selection.json`. It identifies the selected manifests, transfer receipts, acceptance evidence and every sealed attempt for this trial in the canonical Mac store at freeze time, including failures, retries and independent checks. Repeating the same nomination revalidates that frozen snapshot. A conflicting nomination fails. Later reports use the selected run IDs and hashes recorded in the file.

### Mac storage after sealing

A live worker receives an independent snapshot of every embedded input. After its process has stopped, the Mac verifies the new bundle and sets its file modes read-only. Before publishing a new bundle, it may replace an embedded input of at least 64 KiB with a same-volume hard link to identical read-only bytes in its declared local parent. It checks the parent manifest pin, both byte counts and both file hashes. The receiver applies the same operation to a fully verified staging copy. Unsupported links retain the independent snapshot. Local receipts report linked bytes and elapsed time. The store still charges the full temporary working copy at preflight and counts a shared inode once in the published active tree.

The operation only touches the new package's incoming link before publication. Earlier recorder bundles keep their original content, inode, permissions and scientific manifest. A hard-linked asset's link count increases as later verified packages reuse it.

## Rolling the active batch

The initial storage profile caps the active store at 20 GiB and reserves 30 GiB of free space on the volume. Once a family is no longer needed for fitting, the Mac can park its complete sealed runs on a separate cold shelf on the same filesystem. Each move requires a provider-verified archive receipt, the original run ID and manifest SHA-256. The manifest and all members are checked before an atomic directory rename. The private shelf retains the sealed files and their placement catalog. Moving files on the same filesystem releases the active quota while whole-volume free space is still measured for every run.

```sh
python scripts/working_set.py --store "$ACTIVE" --shelf "$COLD" park \
  --run "$RUN" --manifest "$MANIFEST_SHA" --archive-receipt "$RECEIPT"
```

Use `restore` with the same arguments to return a parked directory. For the aggregate report, run `report-view --out "$REPORT_VIEW"` with the same store and shelf. Pass that view as the `--store` for the `roster` and `assemble` steps in the formal evaluation guide. This private view holds the original manifest, operational metadata, numerical summaries, pooled curves and fixed boards. The report checks every consumed file against its original manifest. Full research recovery uses the archived package and the upstream checkpoint.
