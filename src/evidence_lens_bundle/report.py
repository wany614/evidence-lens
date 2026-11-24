"""Render a single offline HTML file from the signed numerical score members."""

import base64
import html
import io

import numpy as np
from PIL import Image

from evidence_lens.l1 import risk_state

from .common import (
    PROFILE,
    RELEASE_INDEX_SHA256,
    json_bytes,
    member,
    no_replace_file,
    profile,
    read_file,
    require,
    sha,
)
from .scoring import REASON_NAMES, counts, reasons

UNKNOWN = (120, 124, 134)
STATE_COLORS = {0: (48, 115, 146), 1: (223, 119, 55), 255: UNKNOWN}
RISK_STOPS = np.array(
    [(240, 246, 247), (160, 207, 209), (82, 148, 163), (140, 91, 114), (111, 33, 65)], float
)
WHY_COLORS = np.array(
    [
        (246, 245, 240),
        (121, 65, 120),
        (177, 87, 60),
        (214, 151, 49),
        (107, 71, 167),
        (29, 109, 144),
        (28, 126, 93),
        (76, 79, 88),
        (133, 93, 61),
    ],
    np.uint8,
)
WHY_LABELS = (
    "Scored",
    "Unverified depth",
    "Invalid predicted RGB",
    "Invalid predicted alpha",
    "Invalid camera Z",
    "Outside source frames",
    "No comparable source",
    "Render failed",
    "Visual input unavailable",
)


def esc(v):
    return html.escape(str(v), quote=True)


def data_url(raw, mime):
    return "data:" + mime + ";base64," + base64.b64encode(raw).decode("ascii")


def png(rgb):
    s = io.BytesIO()
    Image.fromarray(rgb).save(s, format="PNG")
    return s.getvalue()


def image_tag(raw, alt):
    return '<img width="256" height="256" src="' + data_url(raw, "image/png") + '" alt="' + esc(alt) + '">'


def ratio(n, d):
    return f"{100 * n / d:.1f}%" if d else "undefined"


def preview(data):
    with Image.open(io.BytesIO(data)) as im:
        require(
            im.format == "PNG" and im.mode == "RGB" and im.size == (256, 256),
            "invalid preview dimensions or encoding",
        )
        im.load()
    return data


def read_arrays(raw, tau):
    with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
        require(
            len(npz.files) == 4 and set(npz.files) == {"risk", "valid", "state", "unknown_reason"},
            "wrong score array keys",
        )
        a = dict(npz)
    for k, dtype in (
        ("risk", np.float32),
        ("valid", np.bool_),
        ("state", np.uint8),
        ("unknown_reason", np.uint8),
    ):
        require(a[k].dtype == dtype and a[k].shape == (256, 256), "invalid numerical score member: " + k)
    finite = np.isfinite(a["risk"])
    require(
        np.isnan(a["risk"][~finite]).all()
        and np.all((a["risk"][finite] >= 0) & (a["risk"][finite] <= 1))
        and np.array_equal(finite, a["valid"])
        and np.array_equal(risk_state(a["risk"], tau), a["state"])
        and np.array_equal(a["unknown_reason"] == 0, finite)
        and np.max(a["unknown_reason"]) < len(REASON_NAMES),
        "inconsistent finite score, threshold or unknown reason",
    )
    return a


def maps(a):
    risk = a["risk"]
    good = a["valid"]
    continuous = np.zeros((*risk.shape, 3), np.uint8)
    for i in range(3):
        continuous[..., i][good] = np.rint(
            np.interp(risk[good], np.linspace(0, 1, len(RISK_STOPS)), RISK_STOPS[:, i])
        ).astype(np.uint8)
    continuous[~good] = UNKNOWN
    discrete = np.zeros_like(continuous)
    for k, color in STATE_COLORS.items():
        discrete[a["state"] == k] = color
    return continuous, discrete, WHY_COLORS[a["unknown_reason"]]


def badge(color, label):
    return f'<span class="key"><i style="background:rgb{tuple(map(int, color))}"></i>{esc(label)}</span>'


STYLE = """
:root{color-scheme:light;--ink:#203237;--line:#cfdbdb;--paper:#fbfcf8;--sub:#485d61}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,sans-serif}
main{max-width:1280px;margin:auto;padding:38px clamp(16px,4vw,56px) 60px}header{border-top:5px solid #34656a;padding-top:20px}
h1,h2,h3,p,figure{margin:0 0 .6rem}h1{font:clamp(2.1rem,5vw,3.8rem)/1.1 Georgia,serif;letter-spacing:-.035em}
h2{font-size:1.35rem;margin-top:1.4rem}h3{font-size:1.08rem}p,small{color:var(--sub)} .stats,.views,.sources,.legend {display:flex;gap:12px;flex-wrap:wrap}
.stats>div,.views>figure,.observation,.sources>figure{background:white;border:1px solid var(--line);border-radius:7px}
.stats>div{min-width:125px;padding:10px 16px}.stats b{display:block;font-size:1.5rem}.stats small{display:block}
.sources>figure{width:120px;padding:6px}.sources img,.views img{width:100%;height:auto;display:block;object-fit:contain}
.sources figcaption,.views figcaption{padding:6px 3px 0;overflow-wrap:anywhere}.views>figure{flex:1 1 155px;max-width:270px;padding:7px}
.observation{padding:16px;margin:.8rem 0}.observation[hidden]{display:none}table{border-collapse:collapse;font-size:.93rem}td,th{text-align:left;padding:3px 15px 3px 0}
tbody tr+tr{border-top:1px solid #e9eded}.key{display:inline-flex;align-items:center;gap:6px;font-size:.86rem}.key i{width:12px;height:12px;border:1px solid #798585;border-radius:2px}
.legend{margin:9px 0 14px}select,a{font:inherit;color:#1e596a}select{padding:6px;max-width:100%}label{font-weight:600}code{overflow-wrap:anywhere}details{margin:12px 0}summary{cursor:pointer}
.scale{display:inline-block;width:90px;height:10px;background:linear-gradient(90deg,#f0f6f7,#a0cfd1,#5294a3,#8c5b72,#6f2141);vertical-align:middle;margin:0 5px}
@media print{.observation[hidden]{display:block}.filter{display:none}main{padding:8px}.observation{break-inside:avoid}}
"""


def render(root, output, expected_sha256=None):
    raw = read_file(root, "scores.json")
    r = json_bytes(raw)
    if expected_sha256:
        require(sha(raw) == expected_sha256, "score receipt identity differs")
    model = profile()["models"].get(r["method"])
    require(
        r["schema"] == "el-bundle-scores-1"
        and r["model"] == model
        and model is not None
        and r["profile_id"] == PROFILE
        and r["release_index_sha256"] == RELEASE_INDEX_SHA256,
        "unknown score receipt, model or threshold",
    )
    require(
        r["target_order"] == [v["id"] for v in r["views"]]
        and len(set(r["target_order"])) == len(r["target_order"])
        and r["source_order"] == [s["id"] for s in r["sources"]],
        "view or source order differs from receipt",
    )
    require(
        r["reason_codes"] == {str(i): n for i, n in enumerate(REASON_NAMES)}, "unknown reason legend differs"
    )
    header = f"<header><h1>Novel-view risk report</h1><p>{esc(r['input_domain'])} · {esc(r['method'])} · {esc(r['condition'])} · CPU</p></header>"
    source_tiles = []
    for i, s in enumerate(r["sources"]):
        img = preview(member(root, s["display"], expected=f"sources/{i:03d}.png"))
        caption = esc(s["id"]) + (" · primary" if s["id"] == r["primary_source_id"] else "")
        if s["self_render_status"] == "render_failed":
            caption += " · self-render failed"
        source_tiles.append(
            f"<figure>{image_tag(img, 'Source ' + s['id'])}<figcaption>{caption}</figcaption></figure>"
        )
    parts = []
    summed = None
    for i, v in enumerate(r["views"]):
        folder = f"views/{i:03d}/"
        a = read_arrays(member(root, v["arrays"], expected=folder + "scores.npz"), model["threshold"])
        c = counts(a)
        why = reasons(a)
        require(
            c == v["counts"] and why == v["unknown_reasons"], "summary disagrees with stored pixel arrays"
        )
        if summed is None:
            summed = dict(c)
        else:
            for k in c:
                summed[k] += c[k]
        if v["render"]["status"] == "render_failed":
            require(
                v["prediction"] is None
                and c["unknown"] == c["canvas"]
                and why["render_failed"] == c["canvas"],
                "failed render has unexpected usable evidence",
            )
            target = "<p>Render failed. " + esc(v["render"]["reason"]) + "</p>"
        else:
            require(
                v["render"]["status"] == "ready" and why["render_failed"] == 0, "conflicting render status"
            )
            target = image_tag(
                preview(member(root, v["prediction"], expected=folder + "prediction.png")),
                "Target prediction " + v["id"],
            )
        pictures = [
            target,
            *[
                image_tag(png(im), title + " " + v["id"])
                for im, title in zip(maps(a), ("Continuous risk", "Threshold state", "Unknown reasons"))
            ],
        ]
        tiles = "".join(
            f"<figure>{im}<figcaption>{label}</figcaption></figure>"
            for im, label in zip(
                pictures, ("Predicted RGB", "Continuous risk", "Threshold state", "Evidence availability")
            )
        )
        rows = "".join(
            f"<tr><td>{badge(WHY_COLORS[j], WHY_LABELS[j])}</td><td>{why[n]:,}</td></tr>"
            for j, n in enumerate(REASON_NAMES)
            if j and why[n]
        )
        explanation = (
            f"<details open><summary>Unknown pixels by reason</summary><table><thead><tr><th>Reason</th><th>Pixels</th></tr></thead><tbody>{rows}</tbody></table></details>"
            if rows
            else "<p>All target pixels received a score.</p>"
        )
        parts.append(
            f'<section class="observation" id="view-{i}"><h3>{esc(v["id"])}</h3><p>{c["scoreable"]:,} / {c["canvas"]:,} pixels scored ({ratio(c["scoreable"], c["canvas"])}). Low {c["low"]:,} · high {c["high"]:,} · unknown {c["unknown"]:,}.</p><div class="views">{tiles}</div>{explanation}</section>'
        )
    require(summed == r["counts"], "combined counts disagree with target arrays")
    c = summed
    stats = "".join(
        f"<div><b>{num}</b><small>{label}</small></div>"
        for num, label in (
            (f"{len(parts)}", "Target views"),
            (ratio(c["scoreable"], c["canvas"]), "Scored / canvas"),
            (ratio(c["low"], c["canvas"]), "Low / canvas"),
            (ratio(c["unknown"], c["canvas"]), "Unknown / canvas"),
        )
    )
    options = "".join(f'<option value="{i}">{esc(v)}</option>' for i, v in enumerate(r["target_order"]))
    legend = "".join(badge(STATE_COLORS[k], n) for k, n in ((0, "Low"), (1, "High"), (255, "Unknown")))
    details = [
        ("Bundle SHA-256", r["bundle_sha256"]),
        ("Model ID", model["model_id"]),
        ("Weights SHA-256", model["weights_sha256"]),
        ("Feature schema SHA-256", model["feature_schema_sha256"]),
        ("Preprocessing SHA-256", model["preprocessing_sha256"]),
        ("Operating profile", PROFILE),
        ("Release threshold", model["threshold"]),
        ("Upstream checkpoint SHA-256", r["checkpoint_sha256"]),
        ("Source order", ", ".join(r["source_order"])),
        ("Scorer inputs", ", ".join(r["scorer_source_ids"])),
        ("Common endpoint IDs", ", ".join(r["endpoint_source_ids"])),
        ("Endpoint distance (m)", r["endpoint_distance_m"]),
        ("Measured CPU score worker (s)", f"{r['cost']['worker_to_receipt_seconds']:.3f}"),
        ("Of which bundle and model loading (s)", f"{r['cost']['bundle_and_model_load_seconds']:.3f}"),
    ]
    prov = "".join(f"<tr><th>{esc(k)}</th><td><code>{esc(n)}</code></td></tr>" for k, n in details)
    out = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Novel-view risk report · {esc(r["method"])}</title><style>{STYLE}</style>
<main>{header}<div class="stats">{stats}</div><h2>Source photographs</h2><div class="sources">{"".join(source_tiles)}</div>
<h2>Target predictions</h2><p>Risk ranks potential reconstruction error. The original model threshold assigns low and high states. Unknown marks missing evidence. Coverage uses every pixel on the target canvas, including failed renders.</p>
<div class="legend">Risk 0<span class="scale"></span>1 &nbsp; {legend}</div>
<p class="filter" hidden><label for="pick">Target view </label><select id="pick">{options}<option value="all">All views</option></select></p>
{"".join(parts)}<details><summary>Model, inputs and measured cost</summary><p>Verified model domain: {esc(r["tested_domain"])}</p><table>{prov}</table></details>
<a download="scores.json" href="{data_url(raw, "application/json")}">Download numerical receipt (JSON)</a></main>
<script>const select=document.querySelector('#pick');function show(){{document.querySelectorAll('.observation').forEach((v,i)=>v.hidden=select.value!=='all'&&String(i)!==select.value);}}document.querySelector('.filter').hidden=false;select.addEventListener('change',show);show();</script></html>'''
    no_replace_file(output, out.encode())
    return {
        "schema": "el-bundle-html-1",
        "views": len(parts),
        "score_sha256": sha(raw),
        "html_sha256": sha(out.encode()),
        "html_bytes": len(out.encode()),
    }
