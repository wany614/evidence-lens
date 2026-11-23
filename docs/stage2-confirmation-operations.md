# Operating the eight-family confirmation cohort

The registered family order is 041–048. Each family has one baseline batch followed by one two-job R1/Q1 batch. All 24 cells use the registered six targets and the v01 display rule. The Mac opens the reference, records the native readback and authenticates each returned run. Windows receives the source PNG and cameras for B0/R1 and the selected baseline PLY pin for Q1. The training archive, Blender targets and labels remain private.

## Source-only preflight

The dedicated private exchange stores the queue under `Formal exchange/Jobs/v11-b-confirmation-v1/`. The queue plan SHA-256 is `a579acde9959aca1d7abe2fc0e3785ed8be0506986312638e3eb7632b9f7af4d`. The source-only archive `windows-plan.zip` has SHA-256 `6ee9d0bc5453e50f2b35cd1c65150909af6b4bdf5902dfe32d8c8763db1544f4`. The following command reads only the fixed plan and runner source.

```powershell
python run_stage2_confirmation.py --plan plan.json --plan-sha256 a579acde9959aca1d7abe2fc0e3785ed8be0506986312638e3eb7632b9f7af4d --dry-run
```

## Authorized operation

The Mac operator supplies the catalog, method member, opening file, freeze run ID and freeze manifest fingerprints. `stage2_confirmation.py run --help` lists the path arguments. The operator verifies the actual successful recorder freeze and its models before creating `authorization.json`. The Mac issues a separate `ready/<batch>.json` only after it has verified and published that batch. Supply the actual authorization SHA-256 from the Mac to the Windows operator independently.

```powershell
& $WorkerPython "$Queue/run_stage2_confirmation.py" --plan "$Queue/plan.json" --plan-sha256 a579acde9959aca1d7abe2fc0e3785ed8be0506986312638e3eb7632b9f7af4d --authorization-sha256 $OwnerAuthorizationSha256 --exchange $Exchange --staged $Staging --store $RecorderStore --state $QueueState --outgoing $Outgoing --weights $PinnedSharpCheckpoint
```

`$Queue` is the mounted private `Jobs/v11-b-confirmation-v1` folder. `$QueueState` is a dedicated persistent local directory for this one queue. Use the same paths during recovery. On Windows the scientific staging directory, store, outgoing directory, interpreter and SHARP checkpoint share the project volume. The exchange folder may occupy a different volume. One queue instance owns the local lock. The native runners verify each job, checkpoint and producer. Keep each selected B0 generation available to the Windows worker through its Q1 derivation.

An envelope with an absent or shorter ZIP stays `in_transit`. A lone producer stays `waiting_for_return`. The worker retains every original sealed attempt and preserves controller diagnostics separately. A packaging quota hold prints its exact batch. After capacity is restored, repeat the same Windows command with `--resume-batch <batch>` to reuse the sealed computation. An accepted Mac acknowledgement includes the original run, manifest and transfer-container digests. A genuine failed seal enters review with its explicit run and manifest. The native `stage2_process.py` and `stage2_conditions.py` accept that pin with `--terminal-failure`; each verifies the recorded scientific failure and six planned target denominators. A terminal baseline failure blocks the paired-generation dependency until its disposition is resolved.

## Private archive and scoring handoff

Use `--private-archive <directory>` on the Mac to write one content-deduplicated family container after its three selections. Keep this directory outside the active store. Each container includes the original run files, attempts, source handoffs, worker return envelopes and Mac receipts. Its index names any shared external asset or recorder parent by digest. `stage2_confirmation_archive.py verify` checks every stored object. The `restore` command recreates the original recorder bundles in a new directory and runs the independent native verifier. Replaying operations that require an external parent or the original checkpoint also requires those explicitly listed assets.

To stage each original archive for Drive, also pass `--cloud-stage-directory` for the dedicated private `Sealed runs/Stage B confirmation v1` folder. The queue publishes at most 128 MiB per object and records each byte range, size and SHA-256 in a signed offer. The archive owner stays attached to the family and its three selected conditions. The alternate checkpoint and named upstream recorder parents stay explicitly external.

The operator pins account-specific Drive folder IDs in a private `cloud-archive-location.json`, kept outside the source checkout. The native project setup reads `STORE/protocol/stage2/cloud-archive-location.json` and its separately stored `.sha256` declaration. For another store, pass `--provider-config PRIVATE_FILE --provider-config-sha256 EXPECTED_SHA` or set `EVIDENCE_LENS_DRIVE_ARCHIVE_CONFIG` and `EVIDENCE_LENS_DRIVE_ARCHIVE_CONFIG_SHA256` together. The private schema `el-stage2-drive-location-1` uses `provider` set to `Google Drive`, `sealed_runs_parent_id`, `scientific_archive_parent_id`, and `engineering_fixture_parent_id`. Populate those fields from provider metadata for the authorized private project. New offers bind the configuration digest and the destination role. A changed role or mismatched SHA stops confirmation.

The connected Drive operator resolves each registered filename within its pinned private folder, verifies the provider object's ID, size and sharing metadata, and obtains fresh authenticated raw-file references. `stage2_cloud_restore.py confirm` receives those references through standard input. It checks every part SHA and the SHA of their concatenation. Its durable proof records object IDs and file hashes. Temporary download URLs are excluded. `--audit-runs` reopens every original recorder package independently and releases its scratch files after verification. Small archives use one sequential provider download. Larger archives support random access through individually verified 128 MiB segments. Each restore stays within the 4 GiB working budget. Use `--restore DIR --run RID` to materialize a selected run permanently and verify it again on restart.

`stage2_cloud_restore.py certify` binds each completed restore and remote proof to the particular canonical family selection, recorder manifests and opening. `stage2_confirmation_archive.py park` consumes its signed proof and moves original sealed packages intact to the cold shelf. Set `EVIDENCE_LENS_ARCHIVE_SHELF` to that location when opening dependencies through the canonical recorder. The Mac queue sets it for all native children. The postfit reader accepts both active and cold roots and verifies the source manifests and member hashes in either location.

The final Mac artifact is `STORE/stage2/<catalog-sha256>/confirmation/selection-registry.json`. It contains all 24 original trial-selection digests. The independent scoring entry point is `vision_confirm.py new --method RUN=MANIFEST --method-sha256 SHA --selection-registry PATH SHA`. Run it only after the queue reports `complete`.
