# Rebuild one scene or register an independent study

A fresh checkout can regenerate the authored book nook, send its single source photograph to the Windows predictor, and evaluate six target views on a Mac. Family 002 has no dependency on the owner's tabletop or its research store. Executing the same scene recipe produces a new set of recorded observations. Run IDs include the start time and a random suffix. Hardware, libraries and sampled timings also enter the evidence.

| Material | How to obtain it |
| --- | --- |
| Authored geometry, cameras, numeric protocol, code, dependency pins | The source checkout. The scene templates contain dimensions, transforms, materials and lights without machine paths. The source inventory records the disposition of each file. [Source inventory](../configs/public-source.json) |
| SHARP code and checkpoint, CUDA and gsplat | Install the upstream dependencies. The checkpoint is 2,809,738,232 bytes and licensed separately. [Worker installation](windows-worker.md) · [License inventory](licenses.md) |
| Historical Blender files, PNGs, reference arrays, PLYs, prediction arrays, failures and trial selections | Owner-held sealed artifacts. The public catalog gives hashes and selected historical IDs. Verifying those results requires the corresponding archive bytes. [Original family catalog](../protocol/families-v1.json) |

## Set up a clean Mac checkout

The shell examples use Bash or Zsh and `jq` for reading JSON receipts. Run these commands from the source checkout. Choose a new private research-volume directory beside the checkout or set `LAB` to a new directory on a mounted research volume. The recorder keeps a 30 GiB free-space floor and a 20 GiB active-batch target. For a full study use verified family archives and the bounded learning extracts. [Storage and transfer](windows-worker.md)

```sh
set -euo pipefail
SRC="$(pwd -P)"
LAB="$(dirname "$SRC")/lens-book-nook-rebuild"
mkdir "$LAB"
STORE="$LAB/mac-store"
XFER="$LAB/transfer"
mkdir "$STORE" "$XFER"

setup_mac_envs() {
  local project_dir
  project_dir="$(pwd -P)"
  python3.12 -m venv .venv
  python3.12 -m venv .venv-cpu
  python3.12 -m venv .venv-train
  python3.11 -m venv .venv-reference
  PY="$project_dir/.venv/bin/python"
  CPU="$project_dir/.venv-cpu/bin/python"
  TRAIN="$project_dir/.venv-train/bin/python"
  BLENDER="$project_dir/.venv-reference/bin/python"
  "$PY" -m pip install -c requirements/mac-record.txt -e '.[record]'
  "$CPU" -m pip install -c requirements/mac-record.txt -e .
  "$TRAIN" -m pip install -c requirements/mac-train.txt -e '.[record,train]'
  "$BLENDER" -m pip install -c requirements/mac-reference.txt -e '.[reference]'
}
setup_mac_envs
```

## Record family 002

The book nook is authored in the scene template and uses a common calibrated camera rig. The first command verifies the complete registration without rendering. The second records the seven-view Cycles scene and reopens the frozen Blender file in an independent process. Both real runs must succeed. Keep their IDs from this invocation. [Scene template](../configs/formal-scenes-v1.json) · [Camera rig](../configs/formal-cameras-v1.json)

```sh
FAMILY=scene_002_book_nook
"$PY" scripts/formal_reference.py --validate-only --family "$FAMILY"
"$PY" scripts/formal_reference.py --family "$FAMILY" --store "$STORE" \
  --blender-python "$BLENDER" > "$LAB/reference.json"
REF="$(jq -er .reference_run_id "$LAB/reference.json")"
RB="$(jq -er .readback_run_id "$LAB/reference.json")"
CAT="$(jq -er .dataset_hash "$LAB/reference.json")"
"$PY" scripts/verify_reference_config.py --store "$STORE" \
  --selection "$STORE/formal-references/$CAT/$FAMILY.json"
```

### Reconstruct the historical formal Windows worker

The later DTU checkout changed the recorder and disk sampler. The offline builder also recognizes those two exact source hashes, so a downloaded source archive can regenerate the same historical worker. It verifies all 19 reconstructed files and the final ZIP digest. [DTU checkout byte mappings](../reproduce/windows-worker-checkout-variants.json)

Before the first formal handoff in a fresh store, reconstruct the 19 allowlisted Git blobs from commit `ab18e7d4ae627c550f11eb08095739830055ea91`. The helper reads them directly from Git when available. A clean source ZIP can also rebuild them offline. The descriptor records every historical file hash, the original package metadata, and line edits for three archival modules that changed after the freeze. The helper applies those edits in memory, verifies every member, and uses the original sorted ZIP encoder. It requires SHA-256 `3ef8f32ee5deb62065a56e9e64221a84c2e2e8fd907037f1724f857125967143` before creating the per-catalog ZIP and receipt. An existing identical pack is accepted. A conflicting freeze stops the command. An independently supplied original ZIP can be used with `--worker-zip /path/to/worker-source.zip`. [Historical worker member hashes](../protocol/windows-worker-source-v1.json)

```sh
"$PY" scripts/rebuild_study.py worker --store "$STORE" --git-root "$SRC"
"$PY" scripts/make_handoff.py --reference "$STORE/runs/$REF" --readback "$STORE/runs/$RB" \
  --families protocol/families-v1.json --freeze configs/formal-freeze-v1.json \
  --out "$XFER/handoff.zip" > "$XFER/dispatch.json"
jq -e '.worker_source_sha256=="3ef8f32ee5deb62065a56e9e64221a84c2e2e8fd907037f1724f857125967143"' "$XFER/dispatch.json"
```

### Generate and render on Windows

Transfer `handoff.zip` and its outer receipt `dispatch.json` to Windows. The pack contains the sole source PNG, seven camera descriptions, worker code and the trial/parent pins. The Mac keeps target photographs, reference depth, visibility, object IDs, labels and the Blender world. Install the pinned GPU stack and obtain the checkpoint from the URL in the B0 config. Set `$W`, `$Python` and `$Weights` below to locations on the same research volume. Put the transferred files in `$W\incoming`. Each fresh handoff gets its own extraction directory. [Windows installation and SHARP pin](windows-worker.md)

```powershell
$ErrorActionPreference = "Stop"
$W = "R:\lens-book-nook"
$Python = "$W\env\Scripts\python.exe"
$Weights = "$W\weights\sharp_2572gikvuh.pt"
$Dispatch = Get-Content "$W\incoming\dispatch.json" -Raw | ConvertFrom-Json
$Zip = "$W\incoming\handoff.zip"
if ((Get-FileHash -Algorithm SHA256 $Zip).Hash.ToLowerInvariant() -ne $Dispatch.sha256) { throw "Handoff SHA-256 mismatch" }
$Handoff = "$W\jobs\$($Dispatch.trial_id)"
Expand-Archive -LiteralPath $Zip -DestinationPath $Handoff
$Stage = "$Handoff\code\scripts\windows_stage.py"
$Outgoing = "$W\outgoing"
$G = & $Python $Stage generate --handoff $Handoff --store "$W\store" --return-dir $Outgoing --weights $Weights | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $G.status -ne "succeeded") { throw "Inspect the recorded generation attempt" }
$R = & $Python $Stage render --handoff $Handoff --store "$W\store" --return-dir $Outgoing --generation-run $G.run | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $R.status -ne "succeeded") { throw "Inspect the recorded render attempt" }
$G, $R | Select-Object run_kind,run,manifest_sha256,package,sha256
```

Return the two named ZIPs and their matching `<run>.receipt.json` files to `$XFER` on the Mac. Keep the generation and render IDs from Windows. The recorder preserves failed attempts and requires an explicit linked scientific retry. The formal dispatcher can resume a completed stage by its handoff identity. [Worker retries and durable receipts](windows-worker.md)

### Receive and review on Mac CPU

Set `GEN` and `RENDER` to the two exact IDs printed on Windows. Obtain the checkpoint independently or supply the previously verified file through `WEIGHTS`. The Mac needs it to resolve the generation's shared-input hash. The two commands below obtain and check the official file when starting with an empty research directory.

```sh
WEIGHTS="$LAB/sharp_2572gikvuh.pt"
curl --fail --location "$(jq -er .generator.checkpoint_url configs/windows-b0.json)" --output "$WEIGHTS"
"$PY" -m evidence_lens.archive add-asset "$WEIGHTS" --root "$STORE" \
  --sha256 "$(jq -er .generator.checkpoint_sha256 configs/windows-b0.json)"
for rid in "$GEN" "$RENDER"; do
  rec="$XFER/$rid.receipt.json"
  jq -e --arg rid "$rid" '.run==$rid and .status=="succeeded"' "$rec" >/dev/null
  "$PY" -m evidence_lens.archive receive "$XFER/$(jq -er .package "$rec")" \
    --root "$STORE" --sha256 "$(jq -er .sha256 "$rec")"
done
"$PY" scripts/process_b0.py --store "$STORE" --family "$FAMILY" \
  --generation-run "$GEN" --render-run "$RENDER" --cpu-python "$CPU" --train-python "$TRAIN" > "$LAB/review.json"
jq '{trial_id,selection_sha256,runs,extract}' "$LAB/review.json"
```

The score uses source RGB, cameras and predictions. A separate evaluator opens the reference arrays. Independent verification rereads v01 and checks its counts and sampled projections. The seven-role selection is at `$STORE/trials/<trial_id>/trial_selection.json`. The selected evaluation’s `payload/evaluation.json` contains per-view denominators and `payload/board_v01.png` shows the fixed diagnostic view. Zero coverage, calibration failure and missing products have distinct recorded meanings. [CPU diagnostics](cpu-scoring.md)

## Register a separately named 24-family replication

The original dataset SHA-256 is `fb7fabef06d006c992bc38cb03c283a41d90238e0b174664a1357f274bb36baa`. Its first training row requires the historical tabletop selection SHA-256 `32f4ea2a1b0fdc09038e7413f6461aa334d15042f807895b4bca286535e4cf15`, its exact reference/readback manifests and their transitive evidence. The full historical result also depends on each original formal selection, the fitted model, frozen development thresholds and held-out records. A newly rendered tabletop has new run IDs and hashes. To reproduce a historical result exactly, restore its verified private family and model archives with the separately obtained checkpoint. [Archive verification](recording.md)

For a new registration, choose another empty research store and transfer directory. Keep this store separate from the one-family example. First generate the training pilot under the recipe's fixed trial label. The original project directory remains the source for these commands.

The registrar reads a committed source tree and checks it against its file inventory. When starting with a source ZIP, unpack it into a new directory and make a local Git commit of those source files before installing environments or adding research artifacts. This provides a source commit for the new registration. The historical worker can still be reconstructed from the ZIP's source bytes.

```sh
REP_NAME=my-lab-replication
PILOT="$(dirname "$SRC")/$REP_NAME-data"
mkdir "$PILOT"
STORE="$PILOT/mac-store"
XFER="$PILOT/transfer"
mkdir "$STORE" "$XFER"
TRIAL=tabletop-v001-b0
"$PY" -m evidence_lens.recorder --root "$STORE" --source-root "$SRC" --kind scene_render \
  --experiment p1-fixed-world --trial "$TRIAL" --config configs/reference.json -- \
  "$BLENDER" '{source}/src/evidence_lens/blender_reference/produce.py' --config '{config}' --out '{out}' > "$PILOT/reference.json"
REF="$(jq -er .run_id "$PILOT/reference.json")"
"$PY" -m evidence_lens.recorder --root "$STORE" --source-root "$SRC" --kind archive_verify \
  --experiment p1-fixed-world --trial "$TRIAL" --config configs/reference.json \
  --parent "$REF=$(jq -er .manifest_sha256 "$PILOT/reference.json")" \
  --input "reference=$STORE/runs/$REF/payload" --input-parent "reference=$REF" -- \
  "$BLENDER" '{source}/src/evidence_lens/blender_reference/verify_reference.py' --payload '{input:reference}' --out '{out}' > "$PILOT/readback.json"
RB="$(jq -er .run_id "$PILOT/readback.json")"
"$PY" scripts/make_handoff.py --reference "$STORE/runs/$REF" --out "$XFER/handoff.zip" > "$XFER/dispatch.json"
```

This pilot uses the legacy engineering handoff. Run its included Windows stage once for generation and once for render, using a fresh Windows research store. Receive both archives and add the checkpoint to the new Mac store as above. With the new `GEN` and `RENDER`, score and freeze the first eligible engineering chain.

```sh
common=(--store "$STORE" --reference-run "$REF" --readback-run "$RB" --render-run "$RENDER")
"$PY" scripts/mac_trial.py score "${common[@]}" --cpu-python "$CPU" > "$PILOT/score.json"
SCORE="$(jq -er .run_id "$PILOT/score.json")"
"$PY" scripts/mac_trial.py evaluate "${common[@]}" --score-run "$SCORE" > "$PILOT/eval.json"
EVAL="$(jq -er .run_id "$PILOT/eval.json")"
"$PY" scripts/mac_trial.py verify "${common[@]}" --score-run "$SCORE" --evaluation-run "$EVAL" > "$PILOT/verify.json"
"$PY" scripts/select_trial.py --store "$STORE" --trial "$TRIAL" --reference-run "$REF" --readback-run "$RB" \
  --generation-run "$GEN" --render-run "$RENDER" --score-run "$SCORE" --evaluation-run "$EVAL" \
  --recompute-run "$(jq -er .run_id "$PILOT/verify.json")"
REPLICA="$(dirname "$SRC")/$REP_NAME-source"
"$PY" scripts/rebuild_study.py register --store "$STORE" --name "$REP_NAME" --out "$REPLICA"
```

Registration verifies the complete selected pilot, its sealed source, both tabletop configurations, actual image, Blender file and native cameras. It copies the committed, allowlisted source into a new directory and rebinds five registration documents. The new name, timestamp, pilot pins, catalog SHA and study SHA distinguish this replication. The authored family-set ID and trial labels identify the same recipes across stores. Geometry, camera definitions, folds, scientific rules and worker code retain their original bytes. The original 24-family validator checks the new registration before publication. The training sampler includes the catalog SHA in its seed, so an independent registration draws its own training pixels even when scene outputs match. [Training sample rule](../protocol/study-v1.json)

Initialize a private Git repository in the new directory and commit its audited registration before recording other families. The P5 method freeze reads the method commit's actual blobs. Recreate the four Mac environments from the new checkout. Keep `SRC` pointing to the original checkout for the historical worker.

```sh
cd "$REPLICA"
git init -b "replication/$REP_NAME"
git add -- .
git commit -m "Register independent $REP_NAME study"
setup_mac_envs
"$PY" scripts/formal_reference.py --validate-only
"$PY" scripts/rebuild_study.py worker --store "$STORE" --git-root "$SRC"
"$PY" scripts/formal_reference.py --store "$STORE" --family scene_001_tabletop
"$TRAIN" scripts/formal_batch.py prepare --store "$STORE" --family scene_001_tabletop
```

Run the family 002 procedure from the new checkout for each remaining training and development row using its catalog ID. The 16 training and four development extracts support the recorded fit and threshold steps. Freeze that replication's method commit before starting the four held-out families, stress conditions or timing repeats. Keep its family receipts, model index, formal catalog and subsequent report together. The canonical historical legacy-config checker is specific to the owner's two tabletop manifests. The replication registrar authenticates its own pilot before any formal acquisition. [Fit and method freeze](l1-training.md) · [Condition dispatch](windows-worker.md) · [Held-out comparisons](formal-evaluation.md)
