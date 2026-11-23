"""FP64 pinhole geometry for screening the position and anisotropy of old Gaussian scenes."""

import numpy as np

QUANTILES = (0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0)


def quantiles(values):
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    return {f"p{round(100 * q):02d}": float(np.quantile(x, q)) if x.size else None for q in QUANTILES}


def covariance(scene):
    scale = np.exp(scene["log_scales"].astype("f8"))
    q = scene["quats"].astype("f8")
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if not (scale > 0).all() or (norm == 0).any():
        raise ValueError("Invalid Gaussian scale or rotation")
    w, x, y, z = (q / norm).T
    rot = np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(-1, 3, 3)
    a = rot * scale[:, None, :]
    return a @ a.swapaxes(1, 2)


def ellipse_rect(uv, cov, bounds):
    """Minimum squared Mahalanobis distance from a centre to the closed image rectangle."""
    x0, x1, y0, y1 = bounds
    cx, cy = uv.T
    inside = (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
    # On a fixed x edge the unconstrained best y is cy+cov_yx/cov_xx*(x-cx).
    # Clamping to the segment also covers the four corners. The two y edges are symmetric.
    inv = np.linalg.inv(cov)
    choices = []

    def dist(x, y):
        d = np.stack((x - cx, y - cy), axis=-1)
        return np.einsum("ni,nij,nj->n", d, inv, d)

    for x in [x0, x1]:
        y = np.clip(cy + cov[:, 1, 0] / cov[:, 0, 0] * (x - cx), y0, y1)
        choices.append(dist(x, y))
    for y in [y0, y1]:
        x = np.clip(cx + cov[:, 0, 1] / cov[:, 1, 1] * (y - cy), x0, x1)
        choices.append(dist(x, y))
    return np.where(inside, 0, np.min(choices, axis=0))


def project(scene, world_cov, cam, eps2d):
    k = np.asarray(cam["K"], dtype="f8")
    pose = np.linalg.inv(np.asarray(cam["c2w"], dtype="f8"))
    if not np.allclose(
        k, np.diag([k[0, 0], k[1, 1], 1.0]) + np.array([[0, 0, k[0, 2]], [0, 0, k[1, 2]], [0, 0, 0]])
    ):
        raise ValueError("A non-pinhole intrinsic matrix needs its own registered model")
    local = scene["means"].astype("f8") @ pose[:3, :3].T + pose[:3, 3]
    z = local[:, 2]
    near, far = cam["clip_m"]
    depth = (z > near) & (z < far)
    n = len(z)
    uv = np.full((n, 2), np.nan)
    axis = np.full((n, 2), np.nan)
    half = np.full((n, 2), np.nan)
    intersects = np.zeros(n, dtype=bool)
    centre = np.zeros(n, dtype=bool)
    bbox = np.zeros(n, dtype=bool)
    if np.any(depth):
        i = np.flatnonzero(depth)
        xx, yy, zz = local[i].T
        fx, fy = k[0, 0], k[1, 1]
        proj = np.stack((fx * xx / zz + k[0, 2], fy * yy / zz + k[1, 2]), axis=1)
        jac = np.zeros((len(i), 2, 3))
        jac[:, 0, 0] = fx / zz
        jac[:, 1, 1] = fy / zz
        jac[:, 0, 2] = -fx * xx / zz**2
        jac[:, 1, 2] = -fy * yy / zz**2
        orient = jac @ pose[:3, :3]
        screen = orient @ world_cov[i] @ orient.swapaxes(1, 2) + eps2d * np.eye(2)
        uv[i] = proj
        axis[i] = 3 * np.sqrt(np.linalg.eigvalsh(screen)[:, ::-1])
        hh = 3 * np.sqrt(screen[:, [0, 1], [0, 1]])
        half[i] = hh
        x0, y0 = -0.5, -0.5
        x1, y1 = cam["width"] - 0.5, cam["height"] - 0.5
        px, py = proj.T
        centre[i] = (px >= x0) & (px < x1) & (py >= y0) & (py < y1)
        bbox[i] = (px + hh[:, 0] >= x0) & (px - hh[:, 0] <= x1) & (py + hh[:, 1] >= y0) & (py - hh[:, 1] <= y1)
        intersects[i] = ellipse_rect(proj, screen, (x0, x1, y0, y1)) <= 9.0
    return {
        "camera_z_m": z,
        "inside_clip": depth,
        "centre_uv": uv,
        "centre_inside": centre,
        "ellipse_intersects": intersects,
        "aabb_intersects": bbox,
        "semiaxes_3sigma_px": axis,
        "half_bbox_3sigma_px": half,
    }


def project_summary(d, cam):
    z = d["camera_z_m"]
    clip = d["inside_clip"]
    el = d["ellipse_intersects"]
    dims = np.asarray([cam["width"], cam["height"]])
    oversize = clip & np.any(2 * d["half_bbox_3sigma_px"] > dims, axis=1)
    return {
        "n": len(z),
        "behind_or_at_camera": int(sum(z <= 0)),
        "positive_before_or_at_near": int(sum((z > 0) & (z <= cam["clip_m"][0]))),
        "at_or_beyond_far": int(sum(z >= cam["clip_m"][1])),
        "inside_clip": int(sum(clip)),
        "centre_inside": int(sum(d["centre_inside"])),
        "aabb_intersects": int(sum(d["aabb_intersects"])),
        "ellipse_intersects": int(sum(el)),
        "offscreen_centre_ellipse_hits": int(sum(el & ~d["centre_inside"])),
        "image_spanning_bbox": int(sum(oversize)),
        "image_spanning_bbox_and_ellipse_hits": int(sum(oversize & el)),
        "offscreen_centre_image_spanning_ellipse_hits": int(sum(oversize & el & ~d["centre_inside"])),
        "z_all_m": quantiles(z),
        "z_intersect_m": quantiles(z[el]),
        "major_3sigma_in_clip_px": quantiles(d["semiaxes_3sigma_px"][clip, 0]),
        "major_3sigma_intersect_px": quantiles(d["semiaxes_3sigma_px"][el, 0]),
        "minor_3sigma_intersect_px": quantiles(d["semiaxes_3sigma_px"][el, 1]),
    }
