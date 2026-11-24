"""Extract digest-checked native files and encode the two authored exhibit cases."""

import hashlib
import io
import json
from pathlib import PurePosixPath
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import numpy as np
from PIL import Image

from dtu_eval import read_pinned
from evidence_lens._bundle import load
from evidence_lens._verify import require
from evidence_lens_bundle.bindings import array_sha
from vision_sources import member

BELL = "offset-bell-cage"
GALLERY = "b05-perforated-gallery"
ALLOWED_CASES = (BELL, GALLERY)
PHOTO_ZIP = "evidence-lens-foundation-source-photos-v1.zip"
ATTRIBUTION = {"material": "Authored synthetic camera renders generated from original 3D scenes; independent target measurements are also synthetic.",
               "license": "CC-BY-4.0", "creator": "wany614, Evidence Lens",
               "url": "https://creativecommons.org/licenses/by/4.0/"}
RANK_COLORS = ("#f0f6f7", "#a0cfd1", "#5294a3", "#8c5b72", "#6f2141")
UNKNOWN = "#787c86"
REASON_COLORS = ("#f6f5f0", "#794178", "#b1573c", "#d69731", "#6b47a7",
                 "#1d6d90", "#1c7e5d", "#4c4f58", "#855d3d")
ERROR_LEGEND = ({"code": 0, "label": "Measured RGB MAE ≤ 0.05", "color": "#dce7dc"},
                {"code": 1, "label": "Measured RGB MAE > 0.05", "color": "#a94336"},
                {"code": 2, "label": "Reference or prediction unavailable", "color": "#787c86"})


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def size_hash(raw):
    return {"bytes": len(raw), "sha256": digest(raw)}


def checked_zip(z, name, spec):
    require(z.namelist().count(name) == 1, "duplicate or missing archive member " + name)
    raw = z.read(name)
    require(spec["sha256"] == digest(raw) and spec.get("bytes", len(raw)) == len(raw),
            "archive member differs " + name)
    return raw


class Native:
    def __init__(self, store, pin):
        self.pin = pin
        self.run, path = read_pinned((store,), pin)
        self.metadata = load(path)

    def path(self, name, spec=None):
        return member(self.run, "payload/" + name, spec)[0]

    def read(self, spec, base=""):
        return self.path(base + spec["path"], spec).read_bytes()


def decode_png(raw):
    require(raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[24:26] == b"\x08\x02", "RGB8 PNG required")
    with Image.open(io.BytesIO(raw)) as im:
        require(im.format == "PNG" and im.mode == "RGB" and getattr(im, "n_frames", 1) == 1, "expected single RGB image")
        return np.array(im)


def encode_png(pixels):
    require(pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[2] == 3, "RGB8 pixels required")
    f = io.BytesIO()
    Image.fromarray(pixels).save(f, format="PNG")
    return f.getvalue()


def arrays(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def rgb8(a):
    require(a.dtype == np.float32 and a.ndim == 3 and a.shape[-1] == 3, "FP32 RGB required")
    v = np.rint(np.clip(np.nan_to_num(a, nan=0), 0, 1) * 255).astype(np.uint8)
    v[~np.isfinite(a).all(axis=-1)] = [114, 117, 126]
    return v


def photo_size(a):
    return list(a.shape[1::-1])


def zip_entry(z, name, raw):
    entry = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    entry.compress_type = ZIP_STORED
    entry.external_attr = 0o100644 << 16
    z.writestr(entry, raw)


class Published:
    def __init__(self, output):
        self.output = output
        self.zip = ZipFile(output / PHOTO_ZIP, "x")
        self.local = {}
        self.archive = {}

    @staticmethod
    def allowed(name):
        p = PurePosixPath(name)
        require(p.as_posix() == name and p.parts[0] in ALLOWED_CASES and ".." not in p.parts, "unregistered path")

    def png(self, name, raw):
        self.allowed(name)
        require(name.endswith(".png") and name not in self.local, "duplicate or non-PNG public file")
        im = decode_png(raw)
        record = dict(path=name, **size_hash(raw), size=photo_size(im), pixel_sha256=digest(im.tobytes()))
        f = self.output / name
        f.parent.mkdir(parents=True, exist_ok=True)
        with f.open("xb") as fp:
            fp.write(raw)
        self.local[name] = record
        return record

    def asset(self, name, raw, role):
        self.allowed(name)
        require(name not in self.archive and name.endswith((".png", ".npz")), "duplicate or unsupported archive asset")
        rec = dict(archive=PHOTO_ZIP, member=name, **size_hash(raw), role=role)
        if name.endswith(".png"):
            rec["size"] = photo_size(decode_png(raw))
        zip_entry(self.zip, name, raw)
        self.archive[name] = rec
        return rec

    def source(self, case, sid, analysis, original, size):
        p = case + "/sources/" + sid
        v = decode_png(analysis)
        im = Image.fromarray(v)
        h = round(size * im.height / im.width)
        thumb = np.asarray(im.resize((size, h), Image.Resampling.LANCZOS))
        return dict(id=sid, image=self.png(p + ".png", encode_png(thumb)),
                    image_processing="Display thumbnail from the analysis input, Pillow Lanczos, full frame.",
                    analysis=self.asset(p + "-analysis.png", analysis, "source observation, actual analysis grid"),
                    original=self.asset(p + "-original.png", original, "source observation, original resolution"))

    def finish(self):
        index = dict(schema="el-foundation-authored-archive-1", rights=ATTRIBUTION,
                     files=list(self.archive.values()))
        raw = (json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()
        zip_entry(self.zip, "contents.json", raw)
        self.zip.close()
        with ZipFile(self.output / PHOTO_ZIP) as z:
            require(len(z.namelist()) == len(index["files"]) + 1 and z.read("contents.json") == raw,
                    "archive whitelist mismatch")
            for file in index["files"]:
                checked_zip(z, file["member"], file)
        return index, size_hash(raw)


def legend(labels, colors, codes=None):
    return [dict(code=code, label=label, color=color)
            for code, label, color in zip(range(len(labels)) if codes is None else codes, labels, colors, strict=True)]


def colour(value):
    return [int(value[i:i + 2], 16) for i in (1, 3, 5)]


def paint(a, entries):
    table = np.zeros((256, 3), dtype=np.uint8)
    for entry in entries:
        table[entry["code"]] = colour(entry["color"])
    require(np.isin(a, [entry["code"] for entry in entries]).all(), "unknown categorical colour")
    return table[a]


def categorical_counts(a, entries):
    return {v["label"]: int(np.count_nonzero(a == v["code"])) for v in entries}


def maps(pub, path, rank, state, reason, states, reasons):
    ok = np.isfinite(rank)
    require(rank.dtype == np.float32 and ((0 <= rank[ok]) & (rank[ok] <= 1)).all(), "invalid native rank")
    stops = np.linspace(0., 1., len(RANK_COLORS))
    colours = np.array([colour(c) for c in RANK_COLORS])
    image = np.empty((*rank.shape, 3), np.uint8)
    for channel in range(3):
        image[..., channel][ok] = np.rint(np.interp(rank[ok], stops, colours[:, channel])).astype(np.uint8)
    image[~ok] = colour(UNKNOWN)
    ranks = legend(["0 · lower", "0.25", "0.50", "0.75", "1 · higher", "Unavailable"],
                   [*RANK_COLORS, UNKNOWN], [0, .25, .5, .75, 1, None])
    return dict(rank=pub.png(path + "-rank.png", encode_png(image)),
                state=pub.png(path + "-state.png", encode_png(paint(state, states))),
                reason=pub.png(path + "-reason.png", encode_png(paint(reason, reasons))),
                legends=dict(rank=ranks, state=states, reason=reasons))


def evaluation(pub, case, view, prediction, reference, reference_png, *, origin, original, measurement=None):
    require(reference.shape == prediction.shape and reference.dtype == prediction.dtype == np.float32,
            "analysis reference and prediction must share the native RGB grid")
    ref = np.isfinite(reference).all(axis=-1)
    valid = ref & np.isfinite(prediction).all(axis=-1)
    delta = np.abs(reference - prediction)
    error = (delta[..., 0] + delta[..., 1] + delta[..., 2]) / np.float32(3)
    high = valid & (error > np.float32(0.05))
    code = np.where(valid, high, 2).astype(np.uint8)
    p = case + "/evaluation/" + view
    counts = dict(reference=int(ref.sum()), labelled=int(valid.sum()), errors=int(high.sum()),
                  unavailable=int((ref & ~valid).sum()), mean_rgb_mae=float(np.mean(error[valid], dtype=np.float64)))
    entry = dict(role="independent evaluation only", origin=origin, counts=counts, error_threshold=.05,
                 formula="mean absolute difference of three FP32 display-sRGB channels; error when strictly above 0.05",
                 reference=pub.png(p + "-reference.png", reference_png), original=original,
                 error=pub.png(p + "-error.png", encode_png(paint(code, ERROR_LEGEND))),
                 legend=list(ERROR_LEGEND), reference_analysis_sha256=array_sha(reference),
                 error_mask_sha256=array_sha(high), valid_mask_sha256=array_sha(valid))
    if measurement is not None:
        entry["native_measurement"] = measurement
    return entry, valid, high
