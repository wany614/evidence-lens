# Reproduce one Stage C neural fit

Stage C compares a frozen DINOv2 backbone with a model that also updates its last transformer block and final normalization. Both retain the same 175 backbone tensors from the published Stage B fine-tuned model. Both initialize the 384-to-32 projection and 71-to-32-to-1 head afresh. Within a seed the initial values and training view order match. A development threshold and all six target maps are retained for every fitted model.

The registered training roster has eight families and 96 family–condition–target rows. The four development families have 48 rows. Each uses two independently reconstructed conditions with the same six calibrated 256-pixel target views. Only score-all contributes supervision. Unknowns retain their separate state. The two primary models always use seed 1729. Seed 2718 measures sensitivity. The recorder preserves failed attempts, linked retries, native development scores and the terminal optimizer checkpoint.

## Freeze the worker and prepare a Mac job

Run the following from the shipping source checkout with Python 3.11 or newer. The package command selects the original Stage B recorder and DINOv2 implementation from its committed tree and the named C source files from this checkout. On macOS and Linux use `python3`. A Windows environment may select an installed interpreter with `py -3`. The training environment needs the recorded torch/CUDA build, NumPy, Pillow, safetensors, psutil, PyYAML and PyArrow.

```sh
python3 -I -B scripts/multi_risk_pack.py pack --repository . --output stage3-worker.zip
python3 -I -B scripts/multi_risk_pack.py extract --archive stage3-worker.zip \
  --sha256 "$WORKER_SHA" --output stage3-worker
python3 -I -B stage3-worker/scripts/multi_risk_job.py prepare \
  --store "$STORE" --shelf "$SHELF" --worker stage3-worker.zip --worker-sha256 "$WORKER_SHA" \
  --train "$TRAIN_PIN" --development "$DEV_PIN" --baseline "$LIGHT_DEV_PIN" \
  --B-parent '20261123T133146588950Z-c1d19dcb78c74cf0=0100b3f27e79c4498d51854d249312fc54cd7c80291b812e68f42c1dbfd88583' \
  --method MS-ft --seed 1729 --output handoff
```

A pin is the original recorder run ID, an equals sign and the SHA-256 of its manifest.json. The Mac authenticates the source dataset and the actual 3A fit and frozen evaluation for both lightweight methods. Prepare four jobs, one each for MS-frozen and MS-ft with seeds 1729 and 2718. The source ZIP and original datasets stay identical across jobs. Use the returned job SHA-256 independently on Windows.

The native Windows training data mount has dataset.json next to the exact relative files listed under that split in handoff/job.json. The training mount has the predicted native RGB and sampled features/labels only. Separate development mounts contain the registered dense features, selection masks and development truth. Source images and sealed test data have no job role. Copy or independently authenticate the published B weights.safetensors with SHA-256 f372bd34faa1b345ae2486fe6c96aea00d8d9ac5749b0b2c237f9ed08d50f228. All job and recorder paths below must be on the designated research drive. Run inside the existing single-GPU lock.

```powershell
py -3 -I -B "$Job/code/scripts/multi_risk_job.py" windows --job "$Job" --job-sha256 "$JobSha" --store "$Store" --return-dir "$Returns" --train-root "$Train" --development-root "$Development" --B-weights "$BWeights"
```

Windows produces an original recorder model_fit and a run-ID-named ZIP and receipt. Repeating the command returns its sealed result. Following a verified setup failure, pass `--after-preflight PRIOR_ID`. For an actual failed scientific attempt use `--retry-of FAILED_ID`; to resume from its sealed training-state.pt also supply its original `--resume-sha256`. Preexisting records and the first immutable job remain available.

```sh
python3 -I -B stage3-worker/scripts/multi_risk_job.py receive \
  --store "$STORE" --job handoff --job-sha256 "$JOB_SHA" --archive "$RETURN_ZIP" --receipt "$RETURN_RECEIPT"
```

## Compare original exported tensors and measure deployment

The source B model index SHA-256 is 08af4cba3537f6f677b24d045c4c374d2d46eedc63a20abc874cfcad3f7ddac5. A read-only initialization comparison can run without any research dataset.

```sh
python3 -I -B stage3-worker/scripts/multi_risk_worker.py inspect-init \
  --B-index "$B_PACKAGE/model.json" --B-weights "$B_PACKAGE/weights.safetensors" \
  --feature-schema stage3-worker/protocol/multisource-v1.json --seed 1729 --output paired-init
python3 -I -B stage3-worker/scripts/multi_risk_worker.py verify-updates \
  --B-index "$B_PACKAGE/model.json" --B-weights "$B_PACKAGE/weights.safetensors" \
  --training "$FIT/payload" --training-sha256 "$TRAINING_JSON_SHA" --output verified-updates
```

The original gradient evidence covers eight individual mean-BCE minibatches before optimizer weight decay. The independent NumPy reader verifies 181 exported tensors against all 175 original backbone tensors, six stored initial head tensors, before/after SHA-256 tables, changed element counts and the CPU export. The Mac check independently recreates the first four view BCE gradients from the actual B backbone, initial head and sealed samples. It checks the second update's four original witnesses against their actual inputs and optimizer positions. The evidence distinguishes cross-device numerical agreement from CUDA bitwise replication.

The joined archive_verify scientific command is `scripts/stage3_neural_evidence.py compare --store STORE --shelf SHELF --train TRAIN_PIN --development DEVELOPMENT_PIN`, with `--candidate FIT_PIN` repeated for the four formal fits, then `--run-id {run_id} --output {out}`. Its seven recorder parents are the four fits, both named data originals and the original B parent above. The publication and opening refer to payload/tensor-comparison.json.

The CPU archive_verify uses the same script with `cpu --store STORE --shelf SHELF --candidate FROZEN_PIN --candidate FT_PIN --development DEV_PIN --run-id {run_id} --output {out}`, and all three parents. Each of the two primary models runs through both original development-057 reconstructions. The recorded cold process includes interpreter startup, model loading, native input validation and camera projection. Warm timings cover all six views per source scope. The CPU score reader also checks the signed Windows development maps at the full 256×256 grid, records the supported-pixel differences and every threshold crossing. Its public component is payload/cpu-validation.json.

## Run a single exported package

```sh
python3 -I -B "$PACKAGE/score-native.py" --model-sha256 "$MODEL_SHA" --case "$CASE" \
  --case-sha256 "$CASE_SHA" --view v01 --scope all --threads 4 --output native-v01
```

CASE contains case.json, condition-scoped cameras and original RGB sources, prediction.json and predicted RGB/expected-camera-Z/alpha arrays. The command opens the requested target and the selected source subset. Its output contains FP32 risk, boolean valid, and uint8 state at native resolution. State 0 means a finite risk strictly below the frozen threshold, 1 means a finite risk at or above it and 255 means unknown. For the scientific held-out batch, `multi_risk_worker.py infer-dataset --split test` additionally requires the registered Mac store, the common opening SHA and the corresponding selected primary model.

## Terminal optimizer and deployment recovery

The original Windows `MS-frozen` fit for seed 1729 completed 12 epochs and 288 AdamW steps. Its first CPU load rejected the exported index because the training schema overwrote the model schema. Its failed recorder bundle remains the scientific source of the selected parameters. An independent Mac archive verification rehashed the complete original, compared 175 backbone and six checkpoint-head tensors against the Windows safetensors and verified that the registered workers 03 and 04 differ only in the order of two metadata lines. A separate derivation copied the complete 88,302,852-byte weights and all 48 selected Windows all-source development arrays. It ran zero optimizer steps. Its remaining source-ablation maps were calculated using the same weights and threshold on Mac CPU. Recovered weights have SHA-256 `00be5de5abd30bf3e7eb1686e458a96e3ca55dffd6b892af862c98913c85c0b6`.

The dated acquisition pins and the successful readback certificate are in the derivative's `recovery.json`. The original curve, gradient witnesses, selected development receipt and stored checkpoint have identical member hashes in both records. The deployed model's identifier is the successful derivation ID. Its `optimization_parent` identifies the original Windows run. In the other three fits, Windows produced the original optimizer and complete deployment package under worker 04.
