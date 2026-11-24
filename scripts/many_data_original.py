"""Generate the six registered architectural families and their physical pinhole observations."""

import ctypes
import math
from pathlib import Path
import subprocess
import sys

import numpy as np


class Mesh:
    def __init__(self):
        self.faces = []
        self.materials = []

    def box(self, center, size, material, axes=None):
        basis = np.eye(3) if axes is None else np.array(axes, dtype=np.float64)
        pos = np.asarray(center, dtype=np.float64)
        extent = np.asarray(size, dtype=np.float64)
        for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            for side in (-1, 1):
                verts = []
                for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    q = pos + 0.5 * (
                        basis[i] * extent[i] * side
                        + basis[j] * extent[j] * su
                        + basis[k] * extent[k] * sv * side
                    )
                    verts.append([*q, (su + 1) * extent[j] / 2, (sv + 1) * extent[k] / 2])
                for face in ((0, 1, 2), (0, 2, 3)):
                    self.faces.append([verts[n] for n in face])
                    self.materials.append(material)

    def beam(self, start, end, width, depth, material):
        a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        direction = b - a
        unit = direction / np.linalg.norm(direction)
        aux = np.array([0.0, 1.0, 0.0]) if abs(unit[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        first = np.cross(aux, unit)
        first /= np.linalg.norm(first)
        self.box(
            (a + b) / 2,
            (width, depth, np.linalg.norm(direction)),
            material,
            [first, np.cross(unit, first), unit],
        )


def geometry(scene):
    m = Mesh()
    # Three shared textured context slabs provide trackable background and a common world scale.
    m.box((0, 0.35, -0.09), (11, 9, 0.18), 0)
    m.box((0, 3.8, 1.55), (9, 0.16, 3.1), 4)
    m.box((-3.9, 1.95, 1.25), (0.15, 3.8, 2.5), 1)
    tag = scene.split("-", 1)[0]
    if tag == "b01":
        for row, y in enumerate((-0.55, 1.2)):
            for bay in range(3):
                x = -2.04 + 1.87 * bay + 0.49 * row
                for dx in (-0.76, 0.76):
                    m.box((x + dx, y, 0.67), (0.23, 0.48, 1.34), 2 + row)
                points = [
                    (x + 0.76 * math.cos(i * math.pi / 8), y, 1.3 + 0.76 * math.sin(i * math.pi / 8))
                    for i in range(9)
                ]
                for a, b in zip(points, points[1:]):
                    m.beam(a, b, 0.24, 0.53, 2 + row)
        m.box((1.7, 2.45, 0.45), (1.3, 0.55, 0.9), 6)
    elif tag == "b02":
        for n in range(6):
            m.box((-1.73, -1.35 + n * 0.51, 0.10 * (n + 1)), (1.8, 0.54, 0.20 * (n + 1)), n % 3)
            m.box((1.73, -0.78 + n * 0.50, 0.12 * (n + 1)), (1.60, 0.52, 0.24 * (n + 1)), 4 + n % 3)
        m.box((0, 2.4, 0.67), (4.85, 0.6, 1.34), 3)
        for i, (x, y, h) in enumerate(
            ((-2.8, 0.50, 1.55), (0.25, -0.45, 0.84), (-0.52, 1.42, 1.4), (2.65, 1.6, 1.85))
        ):
            m.box((x, y, h / 2), (0.48, 0.50, h), 5 + i)
    elif tag == "b03":
        for row, y in enumerate((-0.55, 0.92)):
            for n in range(5):
                x = -2.65 + n * 1.07
                m.box((x, y, 1.15), (0.15, 0.21, 2.3), 5 + row)
                m.beam(
                    (x, y, 0.17 if n % 2 else 2.16),
                    (x + 1.07, y, 2.16 if n % 2 else 0.17),
                    0.19,
                    0.20,
                    2 + row,
                )
                m.beam((x, y, 2.25), (x, 1.0 - 0.55 * row, 2.25), 0.18, 0.17, 7)
            m.box((0.02, y, 2.3), (5.53, 0.23, 0.22), 5)
        m.box((1.3, 2.20, 0.53), (1.1, 0.88, 1.06), 0)
    elif tag == "b04":
        for n, (x, y, angle) in enumerate(((-1.79, -0.50, 0.36), (0.02, 0.60, -0.47), (1.76, -0.08, 0.55))):
            for s in (-1.0, 1.0):
                m.box((x + s * 0.80, y, 1.21), (0.14, 0.3, 2.42), 5)
            axes = [[1, 0, 0], [0, math.cos(angle), math.sin(angle)], [0, -math.sin(angle), math.cos(angle)]]
            for j in range(7):
                m.box((x, y, 0.27 + j * 0.31), (1.65, 0.56, 0.11), 2 + n, axes)
    elif tag == "b05":
        for row, y in enumerate((-0.62, 1.22)):
            for j in range(6):
                x = -2.75 + j * 1.04 + 0.22 * row
                m.box((x, y, 1.03), (0.32, 0.49, 2.06), 2 + row)
                if j < 5 and (j + row) % 2:
                    m.box((x + 0.52, y, 0.49), (0.73, 0.48, 0.98), 2 + row)
                if j < 5:
                    m.box((x + 0.52, y, 1.98), (0.75, 0.49, 0.32), 2 + row)
            m.box((0.03, y, 2.25), (5.95, 0.85, 0.31), 6)
        for x in (-1.25, 1.72):
            m.box((x, 0.31, 2.36), (0.71, 2.12, 0.22), 1)
    elif tag == "b06":
        for i, (x, y, l) in enumerate(((-2.7, 1.90, 1.62), (-0.97, 1.90, 1.75), (0.84, 1.90, 1.87))):
            m.box((x + l / 2, y, 1.25), (l, 0.20, 2.50), 3 + i)
            m.box((x, y - 0.61, 1.25), (0.22, 1.35, 2.5), 1 + i)
            for j in range(1 + i):
                m.box((x + l * 0.5, y - 0.55, 0.48 + 0.62 * j), (l - 0.16, 1.05, 0.12), 5)
        for j in range(3):
            m.box((2.59, 0.99 - 0.94 * j, 1.02), (0.20, 0.98, 2.04), 2)
            m.box((2.12, 1.38 - 0.94 * j, 0.90), (0.91, 0.15, 1.8), 4 + j)
        m.box((-0.51, -0.84, 0.41), (1.0, 0.62, 0.82), 7)
    else:
        raise ValueError("unknown registered family")
    return np.ascontiguousarray(m.faces, dtype=np.float64), np.asarray(m.materials, dtype=np.int32)


def cameras():
    views = {}
    for prefix, n in (("r", 128), ("h", 6)):
        for i in range(n):
            t = i / 127 if prefix == "r" else (i + 0.37) / 6
            theta = math.radians(-48 + 96 * t)
            radius = 7.3 + 0.35 * math.cos(6 * math.pi * t + 0.17) + 0.20 * (prefix == "h")
            center = np.array(
                [
                    radius * math.sin(theta),
                    -radius * math.cos(theta),
                    2.45 + 0.55 * math.sin(5 * math.pi * t + 0.23) + 0.25 * (prefix == "h"),
                ]
            )
            target = np.array(
                [0.18 * math.sin(2 * math.pi * t), 0.35, 1.10 + 0.09 * math.cos(4 * math.pi * t)]
            )
            forward = (target - center) / np.linalg.norm(target - center)
            right = np.cross(forward, [0.0, 0.0, 1.0])
            right /= np.linalg.norm(right)
            transform = np.eye(4)
            transform[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
            transform[:3, 3] = center
            name = f"r{i:03d}" if prefix == "r" else f"h{i:02d}"
            views[name] = {
                "width": 756,
                "height": 504,
                "K": [[660, 0, 376.75], [0, 655, 252.1], [0, 0, 1]],
                "c2w": transform.tolist(),
                "distortion": "none",
                "original_name": name + ".png",
            }
    return views


class Renderer:
    def __init__(self, source, scratch, scene, seed):
        name = "many-data.dylib" if sys.platform == "darwin" else "many-data.so"
        path = Path(scratch) / name
        subprocess.run(
            ["c++", "-std=c++17", "-O3", "-shared", "-fPIC", str(source), "-o", str(path)], check=True
        )
        self.lib = ctypes.CDLL(str(path)).paint
        void, num = ctypes.c_void_p, ctypes.c_int
        self.lib.argtypes = [void, void, num, void, void, num, num, ctypes.c_uint32, void, void]
        self.lib.restype = None
        self.faces, self.materials = geometry(scene)
        self.seed = seed

    def render(self, k, c2w, w, h, *, color):
        k = np.ascontiguousarray(k, np.float64)
        cam = np.ascontiguousarray(c2w, np.float64)
        rgb = np.empty((h, w, 3), np.uint8) if color else None
        z = None if color else np.empty((h, w), np.float32)
        ptr = lambda v: v.ctypes.data if v is not None else None
        self.lib(
            ptr(self.faces),
            ptr(self.materials),
            len(self.materials),
            ptr(k),
            ptr(cam),
            w,
            h,
            self.seed,
            ptr(rgb),
            ptr(z),
        )
        return rgb if color else z

    def check_depth(self, k, c2w, z):
        """Direct double-precision world-ray/triangle intersections, separate from screen-space rasterization."""
        a, b, c = self.faces[:, :, :3].transpose(1, 0, 2)
        e1, e2 = b - a, c - a
        camera = np.asarray(c2w)
        origin = camera[:3, 3] - a
        q = np.cross(origin, e1)
        h, w = z.shape
        points = (
            (0, 0),
            (w - 1, 0),
            (0, h - 1),
            (w - 1, h - 1),
            (w // 2, h // 2),
            (w // 3, h // 3),
            (3 * w // 4, 2 * h // 3),
        )
        worst = 0.0
        finite = 0
        for x, y in points:
            ray = camera[:3, :3] @ [(x - k[0][2]) / k[0][0], (y - k[1][2]) / k[1][1], 1.0]
            cross = np.cross(ray, e2)
            determinant = np.einsum("ij,ij->i", e1, cross)
            with np.errstate(divide="ignore", invalid="ignore"):
                u = np.einsum("ij,ij->i", origin, cross) / determinant
                v = (q @ ray) / determinant
                t = np.einsum("ij,ij->i", e2, q) / determinant
            hits = (
                (np.abs(determinant) > 1e-10)
                & (u >= -1e-9)
                & (v >= -1e-9)
                & (u + v <= 1 + 1e-9)
                & (t >= 0.05)
            )
            if hits.any():
                delta = abs(float(z[y, x]) - t[hits].min())
                if not np.isfinite(delta) or delta > 1e-4:
                    raise ValueError("authored camera-Z differs from independent surface intersection")
                finite += 1
                worst = max(worst, delta)
            elif not np.isnan(z[y, x]):
                raise ValueError("unobserved analytic ray has a measured surface")
        return {"pixel_rays": len(points), "finite_rays": finite, "max_absolute_z_error_m": worst}
