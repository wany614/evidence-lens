"""Content-bound deployment output and offline inspection panel."""

import base64
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from evidence_lens_bundle.common import encoded, json_bytes, no_replace_file, require, sha

from .schema import BASELINE_ID, FEATURE_SCHEMA_SHA256, PROFILE, REASONS, SIZE, STATES

STATE_RGB = ((209, 222, 235), (226, 144, 63), (112, 87, 130), (128, 128, 128), (63, 67, 72), (92, 154, 156))


def file_sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_result(root, bank, target_id, plan, step, arrays, authority, scope):
    path = root / "scores.npz"
    np.savez_compressed(path, **arrays)
    q = list(step.queried_source_ids)
    values = {
        "schema": "el-b-bank-result-1",
        "baseline_id": BASELINE_ID,
        "bank_sha256": bank.sha256,
        "scene_id": bank.meta["scene_id"],
        "checkpoint_sha256": bank.meta["checkpoint_sha256"],
        "renderer_id": bank.meta["renderer_id"],
        "analysis_profile": PROFILE,
        "analysis_wh": list(SIZE),
        "target_id": target_id,
        "target_prediction": bank.meta["targets"][target_id]["render"],
        "target_camera_sha256": sha(encoded(bank.meta["cameras"][target_id])),
        "frame_receipt": bank.meta["frame_receipt"],
        "metric_anchor_ids": bank.meta["metric_anchor_ids"],
        "reconstruction_ids": list(bank.reconstruction_ids),
        "accessible_ids": list(bank.accessible_ids),
        "canonical_source_ids": list(bank.ids),
        "exact_photo_aliases": bank.aliases,
        "holdout_camera_ids": list(bank.target_ids),
        "queried_source_ids": q,
        "source_selection": plan.strategy,
        "selection_reasons": list(plan.reasons[: len(q)]),
        "selection_cost": plan.cost,
        "query_cost": step.cost,
        "missing_queried_material": list(step.missing_source_ids),
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "model": authority,
        "audit": scope,
        "permission": bank.meta["permission"],
        "canvas_pixels": step.finite.size,
        "rank_eligible_pixels": int(step.finite.sum()),
        "reason_counts": {k: int(np.count_nonzero(step.reason == i)) for i, k in enumerate(REASONS)},
        "scope_counts": {k: int(np.count_nonzero(arrays["scope_state"] == i)) for i, k in enumerate(STATES)},
        "scope_names": list(STATES),
        "reason_names": list(REASONS),
        "first_witness_index": "zero based in canonical_source_ids; -1 unavailable",
        "arrays": {"path": "scores.npz", "bytes": path.stat().st_size, "sha256": file_sha(path)},
    }
    (root / "scores.json").write_bytes(encoded(values))
    return values


def _image(a):
    buf = io.BytesIO()
    Image.fromarray(a.astype(np.uint8)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _heat(a, top):
    norm = np.clip(np.nan_to_num(a.astype(np.float64), nan=0) / max(top, 1e-12), 0, 1)
    out = np.stack((245 - 119 * norm, 240 - 181 * norm, 224 - 108 * norm), axis=-1).astype(np.uint8)
    out[~np.isfinite(a)] = (74, 76, 83)
    return _image(out)


def report(root, output):
    root = Path(root)
    raw = (root / "scores.json").read_bytes()
    meta = json_bytes(raw)
    require(
        meta["schema"] == "el-b-bank-result-1"
        and meta["baseline_id"] == BASELINE_ID
        and meta["feature_schema_sha256"] == FEATURE_SCHEMA_SHA256
        and meta["arrays"]["path"] == "scores.npz",
        "unrecognized score or feature receipt",
    )
    obj = root / "scores.npz"
    require(
        obj.stat().st_size == meta["arrays"]["bytes"] and file_sha(obj) == meta["arrays"]["sha256"],
        "different score arrays",
    )
    with np.load(obj, allow_pickle=False) as f:
        a = {
            k: f[k]
            for k in (
                "rank",
                "finite",
                "reason",
                "scope_state",
                "comparable_count",
                "support_count",
                "depth_conflict_count",
                "colour_conflict_count",
                "gap_witness",
                "first_witness",
            )
        }
    require(
        all(v.shape == SIZE[::-1] for v in a.values())
        and np.array_equal(np.isfinite(a["rank"]), a["finite"]),
        "rank support or analysis grid differs",
    )
    state = a["scope_state"]
    n = len(meta["queried_source_ids"])
    require(state.max() < len(STATES), "unknown audit state")
    srcids = meta["canonical_source_ids"]
    maps = [
        ("Model rank", _heat(a["rank"], 1)),
        ("Comparable sources", _heat(a["comparable_count"], n)),
        ("Depth-supported sources", _heat(a["support_count"], n)),
        ("Depth conflicts", _heat(a["depth_conflict_count"], n)),
        ("Colour conflicts", _heat(a["colour_conflict_count"], n)),
        ("Query and audit state", _image(np.array(STATE_RGB)[state])),
    ]
    def esc(value):
        return html.escape(str(value), quote=True)
    tiles = "".join(
        f'<figure><img alt="{esc(title)}" src="{img}" width="378" height="252"><figcaption>{esc(title)}</figcaption></figure>'
        for title, img in maps
    )
    state_rows = "".join(
        f'<li><i style="background:rgb{rgb}"></i>{esc(name)} · {count:,}</li>'
        for (name, count), rgb in (((s, meta["scope_counts"][s]), STATE_RGB[i]) for i, s in enumerate(STATES))
    )
    reason_rows = "".join(f"<li>{esc(k)} · {v:,}</li>" for k, v in meta["reason_counts"].items() if v)
    witness = a["gap_witness"]
    witness_rows = "".join(
        f"<li>{esc(srcids[int(v)])} · {int(c):,} gap pixels</li>"
        for v, c in zip(*np.unique(witness[witness >= 0], return_counts=True))
    )
    # Rendered from the signed integer witness grids. No external scripts or scientific label files.
    packed = ",".join(
        json.dumps(base64.b64encode(a[k].astype("<i4").tobytes()).decode())
        for k in ("first_witness", "gap_witness")
    )
    safe_names = json.dumps(srcids, ensure_ascii=True).replace("<", "\\u003c")
    name = esc(meta["scene_id"]) + " / " + esc(meta["target_id"])
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Evidence bank · {name}</title>
<style>body{{font:16px/1.5 system-ui;max-width:1120px;margin:2rem auto;padding:0 1rem;color:#27332e}}section{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}figure{{margin:0}}img{{width:100%;height:auto;image-rendering:pixelated;cursor:crosshair}}figcaption{{font-weight:bold}}i{{display:inline-block;width:1em;height:1em;margin-right:.4em}}code{{overflow-wrap:anywhere}}</style>
<h1>{name}</h1><p>{esc(meta["model"]["name"])} · {esc(meta["model"]["threshold_status"])} · {meta["rank_eligible_pixels"]:,} rank-eligible of {meta["canvas_pixels"]:,} analysis pixels.</p>
<p>Source policy {esc(meta["source_selection"])} selected {n} registered sources from {len(srcids)} unique photos. The geometry comparison uses predicted camera-Z. Higher rank means greater estimated risk of RGB error.</p>
<section>{tiles}</section><p>Click a pixel to read its first comparable source and any source found only by the explicit audit.</p>
<output id="pixel">No pixel selected.</output><h2>Query and audit state</h2><ul>{state_rows}</ul>
<h2>Ranking eligibility</h2><ul>{reason_rows}</ul><h2>Registered sources</h2><p>Queried IDs · {esc(", ".join(meta["queried_source_ids"]))}</p>
<p>Fixed coordinate frame · {esc(meta["frame_receipt"]["unit"])}. Frame anchors · {esc(", ".join(meta["metric_anchor_ids"]) or "none")}.</p>
<h2>Sources witnessing an audited subset gap</h2><ul>{witness_rows or "<li>No recorded gap witness.</li>"}</ul>
<p>Full-library audit · {esc(meta["audit"]["requested"])}. Input archive SHA-256 · <code>{esc(meta["bank_sha256"])}</code>.</p>
<script>const ids={safe_names},fields=[{packed}].map(s=>new DataView(Uint8Array.from(atob(s),c=>c.charCodeAt(0)).buffer));
document.querySelectorAll('img').forEach(im=>im.addEventListener('click',e=>{{const r=im.getBoundingClientRect(),x=Math.min(377,Math.max(0,Math.floor((e.clientX-r.left)*378/r.width))),y=Math.min(251,Math.max(0,Math.floor((e.clientY-r.top)*252/r.height)));const val=fields.map(f=>ids[f.getInt32(4*(y*378+x),true)]??'none');document.querySelector('#pixel').textContent='Pixel ('+x+', '+y+') · first comparable source '+val[0]+' · audit gap witness '+val[1];}}));</script></html>"""
    no_replace_file(Path(output), body.encode())
    return {"report_sha256": sha(body.encode()), "score_receipt_sha256": sha(raw)}
