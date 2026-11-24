"""Read the declared Graphdeco float32 PLY dialect or the project's five-array NPZ."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from evidence_lens_bundle.common import Unsupported, require

PARAMS = ("means", "log_scales", "quats", "opacity_logits", "color_logits")
RECOVERY = "evidence-lens-recovery-pt-v1"
PROFILES = ("graphdeco-v1", "evidence-lens-npz-v1", RECOVERY)


def recovered_parameters(path, metadata=None):
    """Read only the original FP32 reconstruction parameters from a CPU-loaded checkpoint."""
    try:
        import torch
    except ImportError as exc:
        raise Unsupported(
            "recovery checkpoint inspection requires CPU-capable PyTorch"
        ) from exc
    ck = torch.load(path, map_location="cpu", weights_only=True)
    require(
        isinstance(ck, dict) and ck.get("schema") == "el-recovery-checkpoint-1",
        "unsupported recovery checkpoint",
        Unsupported,
    )
    recipe = ck.get("recipe")
    require(
        isinstance(recipe, dict) and isinstance(recipe.get("topology"), dict),
        "malformed recovery recipe or topology",
    )
    cap = recipe["topology"].get("cap")
    schedule, step, eps = recipe.get("checkpoints"), ck.get("step"), recipe.get("eps2d")
    require(
        type(cap) is int
        and cap > 0
        and isinstance(schedule, list)
        and all(type(s) is int and s >= 0 for s in schedule)
        and type(step) is int
        and step in schedule
        and type(eps) in (int, float)
        and np.isfinite(eps)
        and eps > 0,
        "invalid native checkpoint cap, step or screen-space variance",
    )
    names = {
        "means": "means",
        "log_scales": "scales",
        "quats": "quats",
        "opacity_logits": "opacities",
        "color_logits": "colors",
    }
    native = ck.get("params")
    require(
        isinstance(native, dict) and set(native) == set(names.values()),
        "invalid checkpoint Gaussian parameter roster",
    )
    arrays = {}
    for key, native_key in names.items():
        tensor = native[native_key]
        require(
            isinstance(tensor, torch.Tensor)
            and tensor.layout == torch.strided
            and tensor.dtype == torch.float32,
            "expected a dense FP32 checkpoint tensor for " + native_key,
        )
        arrays[key] = tensor.detach().numpy()
    require(
        arrays["means"].ndim == 2 and 0 < arrays["means"].shape[0] <= cap,
        "actual Gaussian count must be positive and within the registered cap",
    )
    if metadata is not None:
        metadata.update(
            eps2d=float(eps),
            step=step,
            registered_cap=cap,
            producer_run_id=ck.get("producer_run_id"),
        )
    return arrays


def read_ply(path):
    with Path(path).open("rb") as f:
        require(
            f.readline().rstrip(b"\r\n") == b"ply", "missing PLY signature", Unsupported
        )
        encoding = None
        count = None
        names = []
        while True:
            line = f.readline(8193)
            require(line.endswith(b"\n") and len(line) <= 8192, "invalid PLY header")
            fields = line.decode("ascii").strip().split()
            if not fields or fields[0] in ("comment", "obj_info"):
                continue
            if fields == ["end_header"]:
                break
            if fields[0] == "format":
                require(
                    encoding is None
                    and len(fields) == 3
                    and fields[2] == "1.0"
                    and fields[1] in ("ascii", "binary_little_endian"),
                    "unsupported PLY version or byte order",
                    Unsupported,
                )
                encoding = fields[1]
            elif fields[0] == "element":
                require(
                    count is None and len(fields) == 3 and fields[1] == "vertex",
                    "only a single vertex element is supported",
                    Unsupported,
                )
                count = int(fields[2])
                require(count > 0, "empty Gaussian array", Unsupported)
            elif fields[0] == "property":
                require(
                    count is not None
                    and len(fields) == 3
                    and fields[1] in ("float", "float32")
                    and fields[2] not in names,
                    "Gaussian PLY requires unique scalar float32 properties",
                    Unsupported,
                )
                names.append(fields[2])
            else:
                raise Unsupported("unknown PLY header " + fields[0])
        required = {
            "x",
            "y",
            "z",
            "opacity",
            *(f"rot_{i}" for i in range(4)),
            *(f"scale_{i}" for i in range(3)),
            *(f"f_dc_{i}" for i in range(3)),
        }
        require(
            required <= set(names),
            "point-cloud or incomplete Gaussian PLY: missing "
            + ", ".join(sorted(required - set(names))),
            Unsupported,
        )
        optional = {"nx", "ny", "nz"}
        if set(names) & optional:
            require(
                optional <= set(names),
                "incomplete optional Gaussian normal triple",
                Unsupported,
            )
        rest = set(names) - required - optional
        require(
            rest == set() or rest == {f"f_rest_{i}" for i in range(45)},
            "only complete SH0 or SH3 coefficients are supported",
            Unsupported,
        )
        require(
            encoding is not None and count is not None,
            "missing PLY format/vertex count",
        )
        if encoding == "ascii":
            values = []
            for _ in range(count):
                row = f.readline().split()
                require(len(row) == len(names), "incorrect PLY row length")
                values.append([float(v) for v in row])
            require(not f.read().strip(), "unexpected PLY payload")
            data = np.asarray(values, dtype="<f4")
        else:
            raw = f.read()
            require(len(raw) == 4 * count * len(names), "incorrect PLY vertex bytes")
            data = np.frombuffer(raw, dtype="<f4").reshape(count, len(names))
    require(np.isfinite(data).all(), "nonfinite Gaussian PLY")

    def cols(keys):
        return np.ascontiguousarray(
            data[:, [names.index(n) for n in keys]], dtype=np.float32
        )

    def seq(base, n):
        return cols([base + str(i) for i in range(n)])

    dc = seq("f_dc_", 3)[:, None, :]
    coeff = (
        np.concatenate(
            (dc, seq("f_rest_", 45).reshape(count, 3, 15).transpose(0, 2, 1)), axis=1
        )
        if rest
        else dc
    )
    return {
        "means": cols(("x", "y", "z")),
        "log_scales": seq("scale_", 3),
        "quats": seq("rot_", 4),
        "opacity_logits": cols(["opacity"])[:, 0],
        "sh": np.ascontiguousarray(coeff),
    }


def load(path, profile, scale, *, metadata=None):
    require(profile in PROFILES, "unsupported Gaussian asset profile", Unsupported)
    require(
        np.isfinite(scale) and scale > 0, "positive metric scale required", Unsupported
    )
    if profile == "graphdeco-v1":
        a = read_ply(path)
    elif profile == RECOVERY:
        require(
            scale == 1,
            "recovery checkpoints already use metric world coordinates",
            Unsupported,
        )
        a = recovered_parameters(path, metadata)
    else:
        with np.load(path, allow_pickle=False) as z:
            require(
                len(z.files) == 5 and set(z.files) == set(PARAMS),
                "unsupported project NPZ; expected complete Stage C raw parameters",
                Unsupported,
            )
            a = {n: z[n] for n in PARAMS}
    require(
        a["means"].ndim == 2 and a["means"].shape[1] == 3,
        "invalid Gaussian means tensor shape",
    )
    n = a["means"].shape[0]
    shape = {
        "means": (n, 3),
        "log_scales": (n, 3),
        "quats": (n, 4),
        "opacity_logits": (n,),
    }
    if "sh" in a:
        shape["sh"] = (n, a["sh"].shape[1], 3)
    else:
        shape["color_logits"] = (n, 3)
    require(
        n > 0
        and all(
            v.dtype == np.float32 and v.shape == shape[k] and np.isfinite(v).all()
            for k, v in a.items()
        ),
        "invalid Gaussian parameter dtype, shape or values",
    )
    require(
        np.all(np.linalg.norm(a["quats"].astype(np.float64), axis=1) > 1e-12),
        "degenerate Gaussian orientation",
    )
    a["means"] = (a["means"].astype(np.float64) * scale).astype(np.float32)
    a["log_scales"] = (a["log_scales"].astype(np.float64) + np.log(scale)).astype(
        np.float32
    )
    with np.errstate(over="ignore", under="ignore"):
        activated = np.exp(a["log_scales"])
    require(
        np.isfinite(a["means"]).all()
        and np.isfinite(activated).all()
        and (activated > 0).all(),
        "Gaussian metric positions/scales overflow or underflow",
    )
    return a
