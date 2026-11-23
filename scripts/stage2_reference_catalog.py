"""Check the registered family identities and derive a selected family's native reference config."""

import itertools
from collections import Counter, defaultdict
from math import atan2, pi, sqrt

from evidence_lens.blender_reference.formal_catalog import (
    digest,
    geometry_fingerprint,
    read,
    require,
    resolved,
    value_hash,
)

CATALOG = "protocol/stage2-families-v1.json"
STUDY = "protocol/stage2-reference-v1.json"
LOCK = "configs/stage2-reference-freeze-v1.json"
ENGINE = (
    "src/evidence_lens/blender_reference/produce.py",
    "src/evidence_lens/blender_reference/verify_reference.py",
    "src/evidence_lens/blender_reference/scene.py",
    "src/evidence_lens/blender_reference/passes.py",
    "src/evidence_lens/blender_reference/validation.py",
)
CAMERA = "configs/formal-cameras-v1.json"
BASE = "configs/reference.json"
OLD_DESIGNS = "configs/formal-scenes-v1.json"
OLD_CATALOG = "protocol/families-v1.json"
TARGETS = [f"v{i:02d}" for i in range(1, 7)]
NOVEL = [
    "trihedral_pleat",
    "pentagonal_taper_field",
    "screw_pair",
    "octagonal_antiprism",
]


def pair_structure(objects):
    """Independent of pigment, object ID/order/naming, whole-set translation, axis rotation and uniform scale."""

    def lengths(o):
        if "size" in o:
            return sorted(o["size"])
        if "radii" in o:
            return sorted(o["radii"])
        if "axis" in o:
            return [o["radius"], sqrt(sum(x * x for x in o["axis"]))]
        return [
            o[k]
            for k in (
                "bottom_radius",
                "top_radius",
                "depth",
                "radius",
                "major_radius",
                "minor_radius",
            )
            if k in o
        ]

    def dist(a, b):
        return sqrt(sum((x - y) ** 2 for x, y in zip(a["center"], b["center"])))

    diameter = max(dist(a, b) for a in objects for b in objects)

    def descriptor(o):
        return (
            o.get("primitive", "box" if "size" in o else "cylinder"),
            o.get("vertices", 0),
            tuple(round(x / diameter, 5) for x in lengths(o)),
        )

    ds = [descriptor(o) for o in objects]
    pairs = [
        (*sorted((ds[i], ds[j])), round(dist(a, objects[j]) / diameter, 5))
        for i, a in enumerate(objects)
        for j in range(i)
    ]
    return value_hash({"descriptors": sorted(ds), "pair_distances": sorted(pairs)})


def topology(objects):
    """Read mesh-defining numbers. No test labels, names or material identifiers participate."""
    flags = set()
    sides = Counter(
        o.get("vertices", 64) for o in objects if o.get("primitive") == "frustum"
    )
    if sides[3] >= 4:
        flags.add("trihedral_pleat")
    if sides[5] >= 10:
        flags.add("pentagonal_taper_field")
    adj = defaultdict(set)

    def weld(point):
        # The recipe stores centres and axes independently to six decimal places.
        # Weld coincident analytic endpoints within ten micrometres, including rounding boundaries.
        for node in adj:
            if max(abs(x - y) for x, y in zip(point, node)) < 1e-5:
                return node
        return tuple(round(x, 6) for x in point)

    for o in objects:
        if o.get("primitive") != "rod":
            continue
        a, b = [
            weld(tuple(x + sign * d / 2 for x, d in zip(o["center"], o["axis"])))
            for sign in (-1, 1)
        ]
        adj[a].add(b)
        adj[b].add(a)
    remaining = set(adj)
    helix_count = 0
    while remaining:
        seed = min(remaining)
        seen = {seed}
        frontier = [seed]
        while frontier:
            added = adj[frontier.pop()] - seen
            seen.update(added)
            frontier.extend(added)
        remaining -= seen
        levels = Counter(v[2] for v in seen)
        if (
            len(seen) == 16
            and sorted(levels.values()) == [8, 8]
            and all(
                len(adj[v]) == 4 and sum(nb[2] == v[2] for nb in adj[v]) == 2
                for v in seen
            )
        ):
            flags.add("octagonal_antiprism")
        endpoints = sorted((v for v in seen if len(adj[v]) == 1), key=lambda p: p[2])
        if len(endpoints) != 2 or len(seen) < 25 or max(len(adj[v]) for v in seen) > 2:
            continue
        chain = [endpoints[0]]
        while chain[-1] != endpoints[1]:
            chain.append(next(iter(adj[chain[-1]] - set(chain[-2:]))))
        if not all(b[2] > a[2] for a, b in itertools.pairwise(chain)):
            continue
        cx = sum(v[0] for v in chain[:-1]) / (len(chain) - 1)
        cy = sum(v[1] for v in chain[:-1]) / (len(chain) - 1)
        radii = [sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y, _ in chain]
        angles = [atan2(y - cy, x - cx) for x, y, _ in chain]
        turn = sum((b - a + pi) % (2 * pi) - pi for a, b in itertools.pairwise(angles))
        if min(radii) > 0.05 and max(radii) - min(radii) < 0.003 and abs(turn) > 3 * pi:
            helix_count += 1
    if helix_count >= 2:
        flags.add("screw_pair")
    return sorted(flags)


def validation(assets, project):
    lock = read(assets / LOCK)
    catalog = read(assets / CATALOG)
    study = read(assets / STUDY)
    require(
        lock["catalog_sha256"] == digest(assets / CATALOG)
        and catalog["preregistration_sha256"] == digest(assets / STUDY),
        "Changed B registration",
    )
    for name, expected in lock["common_source_files"].items():
        require(
            digest(project / name) == expected,
            f"Original reference source changed: {name}",
        )
    for name, expected in lock["stage_files"].items():
        require(
            digest(assets / name) == expected, f"Stage B frozen file changed: {name}"
        )
    rows = catalog["families"]
    require(
        catalog["schema"] == "el-vision-family-roster-1"
        and [r["ordinal"] for r in rows] == list(range(25, 49)),
        "Ordinal series",
    )
    require(
        Counter(r["split"] for r in rows) == {"train": 12, "development": 4, "test": 8},
        "Family splits",
    )
    require(
        Counter(r["test_group"] for r in rows if r["split"] == "test")
        == {"familiar_layout": 4, "novel_geometry_material": 4},
        "Held-out strata",
    )
    require(
        Counter(c for r in rows for c in r["trials"]) == {"B0": 24, "R1": 8, "Q1": 8},
        "Condition enrollment",
    )
    old = read(project / OLD_CATALOG)["families"]
    require(
        not {r["geometry_fingerprint"] for r in old}
        & {r["geometry_fingerprint"] for r in rows},
        "Historical geometry overlap",
    )
    for k in (
        "scene_family_id",
        "geometry_fingerprint",
        "pair_structure_sha256",
        "geometry_template",
    ):
        require(len({r[k] for r in rows}) == 24, "Duplicate structural identity: " + k)
    require(
        all(
            r["camera_sha256"] == lock["common_source_files"][CAMERA]
            and r["example_view"] == "v01"
            for r in rows
        ),
        "Camera or display scope",
    )
    require(
        [r["scene_family_id"] for r in rows if r["pilot"]] == catalog["pilot_families"],
        "Pilot membership",
    )
    return catalog, study, lock


def resolve_selected(project, assets, row, catalog, lock, store):
    if row["split"] == "test":
        from stage2_opening import verify_opening

        verify_opening(store, catalog, lock["catalog_sha256"])
    recipe_path = assets / row["recipe_path"]
    recipe = read(recipe_path)
    require(
        digest(recipe_path) == row["recipe_file_sha256"]
        and value_hash(recipe) == row["recipe_sha256"],
        "Selected family recipe changed",
    )
    original = read(project / OLD_DESIGNS)
    world = original["shared_world"]
    fg = recipe["objects"]
    require(
        pair_structure(fg) == row["pair_structure_sha256"]
        and topology(fg) == row["reserved_geometry"],
        "Structural validation failed",
    )
    require(
        geometry_fingerprint(world["objects"] + fg) == row["geometry_fingerprint"],
        "Physical geometry differs",
    )
    designs = {"shared_world": world, "families": {row["scene_family_id"]: recipe}}
    cfg = resolved(row, catalog, designs, read(project / BASE), lock)
    cfg["formal"].update(
        structural_fingerprint=row["pair_structure_sha256"],
        geometry_template=row["geometry_template"],
        material_sha256=row["material_sha256"],
        test_group=row["test_group"],
    )
    return cfg
