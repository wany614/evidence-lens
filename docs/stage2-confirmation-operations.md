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

Provider verification uses the independently resolved Google Drive object ID and its HTTPS download URL on standard input. The `provider` command verifies the separately downloaded bytes against the verified family archive and issues per-run archive receipts. The `park` command accepts the SHA of this proof and uses the existing working-set mover to preserve original bundles on the local cold shelf. A local archive verification alone grants no parking receipt.

The final Mac artifact is `STORE/stage2/<catalog-sha256>/confirmation/selection-registry.json`. It contains all 24 original trial-selection digests. The independent scoring entry point is `vision_confirm.py new --method RUN=MANIFEST --method-sha256 SHA --selection-registry PATH SHA`. Run it only after the queue reports `complete`.
