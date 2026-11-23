# Evidence Lens recovery worker

This source bundle contains the Windows training and rendering entry point, the original Evidence Lens recorder, a proposed recipe, and engineering fixtures. The registration controller supplies the frozen recipe, per-cell ticket and authority to start a real stage. A source ZIP or candidate recipe by itself is a software delivery.

## Inputs and execution

Run from the extracted source bundle with the original project CUDA Python, gsplat 1.5.3 native extension, PyTorch, NumPy, Pillow, psutil and pyarrow. Use one visible GPU and the same project-wide GPU slot as the other heavy workers. Source, scratch space, recorder store and returns belong on the project volume. The original storage policy retains a 30 GiB free-space floor.

Each private cell contains job.json, init.npz, init.json, source_cameras.json, render_cameras.json and exactly the authorized source RGB PNGs under sources/. job.json pins raw file sizes and SHA-256, the recipe digest, experiment/trial/object/condition and provenanced parent run pins. M2 uses s00 and s03. M4 uses s00 through s03 in that order. Object IDs pass through unchanged. The job may pin a positive stop_step at an issued checkpoint. The candidate recipe keeps stop_step=5000. The per-job endpoint changes neither its SHA-256 nor the LR or topology clocks.

See schemas/cell.schema.json, schemas/recipe.schema.json and schemas/init.schema.json. tests/make_fixture.py and tests/run_engineering_windows.py generate a complete working example with original synthetic values. A source-only optimizer receives individual verified camera, source, initialization and recipe roles. The renderer receives the render-camera file and selected issued checkpoint files. No target photographs belong to a worker input role.

Run the following commands with paths and exact pinned identities from registration. The PowerShell variable $Py denotes the project interpreter.

    & $Py -B scripts/recovery_windows.py optimize --cell $Cell --job-sha256 $JobSHA --recipe $Recipe --store $Store --returns $Returns --gpu-slot $Slot
    & $Py -B scripts/recovery_windows.py render --cell $Cell --job-sha256 $JobSHA --generation $GenerationRunAndManifest --steps all --store $Store --returns $Returns --gpu-slot $Slot
    & $Py -B scripts/recovery_windows.py render --cell $Cell --job-sha256 $JobSHA --generation $GenerationRunAndManifest --steps 1500 --store $Store --returns $Returns --gpu-slot $Slot

A producer pin takes the form RUN_ID=MANIFEST_SHA256. The registered checkpoint schedule includes 0, 500, 1500, 3000 and 5000 up to the job endpoint. Render accepts a comma-separated subset or all issued milestones, including earlier points in a verified continuation. All exported views at all selected steps carry FP32 RGB, expected camera Z in metres and alpha in NPZ, plus PNG for display. outputs.json indexes each view independently and binds the checkpoint, input hashes, object, condition, recipe and producer.

## Numerical conventions

The five initialization arrays are means in metres [N,3], log_scales [N,3], quats in wxyz order [N,4], opacity_logits [N] and color_logits [N,3] for linear RGB. All arrays are float32. init.json supplies the actual_method and scene_scale_m. The worker measures the common s00–s03 distance in metres from calibrated c2w and checks its agreement to 1e-5 relative or 1e-8 metres absolute. FP32 pinhole projection uses the full camera, a +0.5 shift in both principal-point coordinates, gsplat classic, degree-zero color and a black linear background. Source loss uses display-sRGB. The optional 1−SSIM term uses channel-wise valid 11×11 Gaussian population moments, sigma 1.5 and unit data range. Depth uses opacity-normalized camera Z; empty, nonpositive or nonfinite depth is NaN.

Candidate position LR decreases exponentially from 1.6e-4 to 1.6e-6 times the common baseline in metres. The other parameter-group rates and thresholds are in configs/candidate-recipe.json. Topology events run from step 500 inclusive to 3000 exclusive, every 100 updates, with the candidate reset at 1500. First prune, then rank growth candidates by projected gradient with stable row-index tie breaking. Both cloning and splitting contribute one net Gaussian per selected parent. The 20,000 ceiling applies before each growth operation; pruning can free slots in the current or a later refinement. The last interval has fixed topology and normal parameter updates.

Each step records source counts, actual N, measured projection, loss components, LRs, optimizer update norms and topology operations. The worker verifies Adam parameter binding, step and moments after topology surgery. It retains survivor moments, zeros newborn moments and checks reset opacity moments. Velocity and other moment rows stay aligned with the new parameter rows.

## Checkpoints and resource accounting

Each checkpoints/step-NNNN.pt is issued atomically and indexed by SHA-256 in issued.json. It contains FP32 tensors, all five Adam states, densification accumulators, update number, next learning rates, source order/count/cursor, topology totals, CPU/CUDA/NumPy/Python RNG and source/recipe identities. Continue from an issued step using the same cell and recipe bytes.

    & $Py -B scripts/recovery_windows.py optimize --cell $Cell --job-sha256 $JobSHA --recipe $Recipe --resume $RunAndManifest --resume-step 1500 --store $Store --returns $Returns --gpu-slot $Slot

Continuations start at the next absolute update in a new recorded process. A failed producer retains its original failure status and manifest. A continuation records that original run, status and checkpoint hash as artifact provenance; the original recorder accepts successful bundles only in its scientific parent list. An engineering-only segment can end at a registered checkpoint with --segment-end. Resource cutoffs return nonzero. The external watchdog enforces the remaining wall budget including prior work in the selected chain. It covers process startup, checkpoint IO and CUDA execution. process-wall.json gives wall time for the numeric child; run.json and telemetry.csv retain recorder wall time, sampled device memory and process memory where the driver supplies it. Worker JSON reports PyTorch allocator high-water marks in bytes. Device samples include other activity; a missing WDDM process counter stays missing.

## Transport and independent checks

The recorder snapshots this small source bundle and preserves each original success or failure. Each run has an original ZIP and durable receipt. Shared inputs are transported once, either as content-addressed assets or by verified member references into a producer ZIP in the same return directory. Transfer the referenced producer ZIPs and the assets/sha256 closure along with the receipts. The restore command checks original bytes and repopulates the recorder CAS before normal archive receive. No private cell data belongs in a source release.

    & $Py -B scripts/recovery_windows.py restore-assets --receipt $Receipt --store $ReceiverStore
    & $Py -B scripts/preflight_api.py --output $NewEnvironmentJSON
    & $Py -B tests/run_engineering_windows.py --series $UniqueSeries --store $EngineeringStore --returns $EngineeringReturns --work $EngineeringWork --gpu-slot $Slot

The engineering runner creates its own source-only analytic images and cameras. It checks projection, ED, alpha, display RGB, real 20k capacity and topology mutations, complete continuous/resumed state, repeat rendering from a selected checkpoint and a deliberately shortened watchdog cutoff. The latter is expected to generate a failed original receipt. All scientific acquisition, recipe selection and held-out evaluation are separate.
