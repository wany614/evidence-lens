# Resident CPU scoring

The file CLI defaults to MS-L1 and remains the numerical reference. A resident scorer keeps the released weights in memory and scores an already available native grid. It uses the original cutoff and returns fresh arrays for every call. Sources come from a fixed 256 × 256 M2 or M4 reconstruction with native FP32 display-sRGB predictions, expected camera-Z in metres, accumulated opacity, and calibrated undistorted pinhole cameras. Preparation and CPU risk scoring consume an existing map. New views from COLMAP and a supported Gaussian asset require the separate CUDA gsplat renderer.

## Call the array API

With the original replay archive unpacked in the current directory, this example scores one four-source bell-cage view.

```python
from evidence_lens_bundle.bundle import Bundle
from evidence_lens_bundle.resident import binding_from_bundle, frame_meta_from_bundle, load_scorer

bundle = Bundle("inputs/bell-M4")
scorer = load_scorer("MS-L1", device="cpu", threads=4, cache_bytes=64 << 20)
binding = binding_from_bundle(bundle)
sources = {name: bundle.source(name) for name in bundle.order}
context = scorer.prepare_sources(sources, version_binding=binding)
view = bundle.targets[0]
frame = bundle.frame(view)
meta = frame_meta_from_bundle(bundle, view, binding=binding, sequence_id="bell-m4", frame_id=0)
out = scorer.score_frame(context, frame, bundle.cameras[view], frame_meta=meta)
print(out["identity"]["output_sha256"], out["timing"], scorer.cache_info())
```

For MS-ft or MS-frozen, use `load_scorer("MS-ft", models_dir="/path/to/cpu", threads=4)` with the vision dependencies installed. B-L1-transfer uses the same release directory and its registered primary source. The file CLI accepts these four names with `--model`, and the optional packages use `--models`. The verified original model indices authenticate the fitted linear coefficients or complete FP32 safetensors. Each visual package contains all 181 tensors. Stage C initialized MS-frozen and MS-ft from the selected fine-tuned Stage B DINOv2 backbone. The training, parent checkpoint, seeds, updated parameter groups and original data partitions remain documented in the Stage C release. [Frozen risk models](../releases/v0.3.0-rc.1/README.md) · [Original replays and grading](original-archive.md)

## Inputs and ownership

Source IDs retain their authorized insertion order. Each source tuple holds an RGB array of shape (256, 256, 3), a self-render with keys rgb/z/alpha or an explicitly failed self-render, and a Camera. Prepared PNGs decode to finite FP32 display sRGB divided by 255. Target RGB uses the same shape and dtype. Z and alpha are FP32 (256, 256). The registered kernels handle nonfinite prediction values using their original reason priority. Cameras carry FP64 integer-centre K (3 × 3) and OpenCV c2w (4 × 4). Nearest-neighbor projection uses floor(u + 0.5). Finite multi-source reductions keep their original float64 accumulation.

The binding identifies the source order, primary and common endpoints, actual source/self-render/calibration bytes, geometry, depth evidence, map checkpoint, map/source revisions, renderer run and receipt, renderer configuration when known, and source admission policy. In-memory producers can supply that schema directly. Use null in source_files when no encoded file exists. The scorer still hashes every actual source, self-render and camera. Prepared-file adapters check decoded-source hashes and target pixel content against their authenticated file bytes. Construct frame metadata at admission, before starting resident latency measurements. The adapter can retain 256 digest records per Bundle and rereads a verified prediction on a metadata miss.

The scorer copies each prepared source into immutable bytes-backed arrays. The caller may reuse its input buffers after a synchronous call returns. Keep target arrays, camera and metadata fixed throughout a call. Returned arrays belong to that result and can be modified or retained independently. One scorer allows one active call at a time. PyTorch’s CPU thread count is process-global. Use separate processes to compare neural thread configurations. The threads argument controls the PyTorch visual path.

## Results and measurement

The output has FP32 risk, boolean valid, uint8 state and uint8 unknown_reason on the native grid. State is 0 for a finite score strictly below the original threshold, 1 at or above it, and 255 for unknown. Profile records the original cutoff, supported input contract and external calibration status. Identity binds the frame, exact source snapshot, model, feature and preprocessing digests to the target camera and predicted content. It includes an overall output SHA-256 and a separate dtype/shape/content hash for each array. The source snapshot’s full inventory is available as context.identity.

Cache capacity defaults to 64 MiB. A single LRU holds immutable source snapshots, source self-render residual maps and intrinsic ray grids. Turning it off with `cache_bytes=0` uses the original per-frame feature path. The cache reports hits, misses, evictions, entries, retained payload bytes, peak bytes and capacity. Weights, Python allocator overhead, caller-held contexts, frame buffers and temporary feature tensors are separate RSS costs. Dropping old contexts allows their arrays to be freed. Every call computes target features and model scores anew. A new source, camera, map revision, checkpoint, renderer configuration, primary or source order produces a distinct snapshot. A target’s pose and content are hashed independently. A bundle frame must carry the exact source binding it was admitted against.

The timing map gives identity/cache lookup, feature construction, model, postprocessing and service seconds. Model loading is exposed through scorer.model_load_seconds and input preparation through context.prepare_seconds. File outputs separately record target decode and serialization. The original replay/performance ledger tracks process launch, queue time, upstream work and HTML encoding at their respective boundaries.

## Sequence extensions

Frame metadata requires sequence_id, frame_id, map_revision, source_revision, renderer_revision and checkpoint_sha256. Pass source_binding_sha256 for a raw producer that uses explicit admission receipts. A file-backed frame also supplies verified source, camera, file and decoded-prediction identities. Version fields supplied at frame admission must agree with the prepared source context. Capture time is optional and requires its clock domain. Simulated arrival times belong to replay scheduling. Unknown acquisition time is null.

Pose covariance is an optional object with a finite 6 × 6 matrix, coordinates and units. Dynamic-mask metadata takes an opaque reference, source, time and clock_domain. These values are recorded verbatim. The numerical kernel uses RGB and predicted camera-Z evidence to score registered photometric risk. The exported CAPABILITIES mapping identifies whether each extension is accepted, used and validated. Pose uncertainty, dynamics, semantic labels, geometric error, real sensor synchronization and GPU risk inference remain future validation subjects.
