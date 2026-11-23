"""Twenty-four original, deterministic opaque compositions for the second family study."""

import itertools
from math import cos, pi, sin

from evidence_lens.blender_reference.formal_designs import Composition


def all_designs():
    rows = []

    def new(n, slug, title, template, relation, ancestry):
        c = Composition(n, slug, title, relation)
        c.template = template
        c.ancestry = ancestry
        rows.append(c)
        return c

    c = new(
        25,
        "cantilever_desk",
        "Cantilever drafting desk",
        "offset-cross-desk",
        "The rear shelf bridges one cubby while the forward-tilted drawing surface hides the inner cubby and a low stool.",
        "desk and storage furniture",
    )
    for x, h in ((-0.39, 0.43), (0.39, 0.36)):
        c.box(f"Desk pedestal {x}", (x, 0.02, h / 2), (0.20, 0.34, h), "timber")
    c.box("Drafting surface", (0, -0.10, 0.48), (1.18, 0.45, 0.07), "green", (14, 0, 0))
    c.box("Rear shelf", (-0.07, 0.31, 0.69), (0.94, 0.21, 0.055), "terracotta")
    for x in (-0.45, 0.27):
        c.box(f"Shelf side {x}", (x, 0.32, 0.54), (0.045, 0.20, 0.29), "legs")
    c.box("Open cubby divider", (-0.06, 0.33, 0.54), (0.035, 0.16, 0.25), "blue")
    c.box("Rolled drawing box", (0.11, 0.29, 0.48), (0.19, 0.18, 0.11), "cream")
    c.disk("Stool seat", (0.31, -0.35, 0.24), 0.15, 0.055, "blue")
    c.rod("Stool stem", (0.31, -0.35, 0.04), (0.31, -0.35, 0.22), 0.045, "dark")
    c.disk("Stool foot", (0.31, -0.35, 0.024), 0.12, 0.048, "dark")

    c = new(
        26,
        "mortar_station",
        "Mortar, pestle and ingredient shelf",
        "vessel-with-crossing-pestle",
        "A sloping pestle crosses the front bowl rim and the line of raised ingredient jars.",
        "small vessels and tools",
    )
    c.box(
        "Preparation board",
        (-0.04, 0.03, 0.034),
        (1.08, 0.68, 0.068),
        "timber",
        (0, 0, -6),
    )
    c.cone("Heavy mortar", (-0.15, -0.15, 0.20), 0.16, 0.26, 0.28, "green")
    c.ring("Mortar lip", (-0.15, -0.15, 0.35), 0.235, 0.037, "cream", (0, 0, 0))
    c.rod("Pestle", (-0.22, -0.18, 0.31), (0.10, 0.04, 0.69), 0.044, "legs")
    c.ball("Pestle knob", (0.10, 0.04, 0.69), (0.063, 0.063, 0.073), "terracotta")
    c.box("Ingredient ledge", (0.12, 0.29, 0.12), (0.75, 0.24, 0.12), "blue")
    for i, (x, h) in enumerate(((-0.15, 0.19), (0.06, 0.30), (0.33, 0.23))):
        c.cone(f"Jar body {i}", (x, 0.29, 0.18 + h / 2), 0.074, 0.072, h, "cream")
        c.disk(
            f"Jar stopper {i}",
            (x, 0.29, 0.19 + h),
            0.080,
            0.026,
            ("dark", "green", "terracotta")[i],
        )

    c = new(
        27,
        "arch_footbridge",
        "Arched model footbridge",
        "segmental-deck-and-two-side-arches",
        "Stepped deck planks rise to a crown between side ribs. The near rail screens the far abutment and center span.",
        "model bridge assembly",
    )
    for x in (-0.47, 0.47):
        c.box(f"Stone abutment {x}", (x, 0.02, 0.13), (0.25, 0.58, 0.26), "cream")
    steps = [
        (-0.49, 0.30),
        (-0.35, 0.38),
        (-0.19, 0.425),
        (0, 0.44),
        (0.19, 0.425),
        (0.35, 0.38),
        (0.49, 0.30),
    ]
    for i, (x, z) in enumerate(steps):
        c.box(f"Cross plank {i}", (x, 0.01, z), (0.15, 0.44, 0.055), "timber")
    for side, y in enumerate((-0.24, 0.26)):
        for i, ((x, z), (xx, zz)) in enumerate(itertools.pairwise(steps)):
            c.rod(
                f"Curved rail {side}-{i}",
                (x, y, z + 0.17),
                (xx, y, zz + 0.17),
                0.021,
                "green",
            )
            c.rod(
                f"Lower rib {side}-{i}",
                (x, y, z - 0.055),
                (xx, y, zz - 0.055),
                0.030,
                "blue",
            )
        for i, (x, z) in enumerate(steps[::2]):
            c.rod(
                f"Picket {side}-{i}", (x, y, z - 0.055), (x, y, z + 0.17), 0.018, "legs"
            )

    c = new(
        28,
        "overhead_conveyor",
        "Conveyor and inspection portal",
        "roller-bed-with-two-height-portal",
        "A low roller carriage runs through an offset scanning portal. A parcel behind the front jamb crosses the roller outlines.",
        "assembly-line mechanism",
    )
    for y in (-0.19, 0.17):
        c.box(f"Conveyor side {y}", (-0.02, y, 0.33), (1.10, 0.07, 0.10), "blue")
        for x in (-0.43, 0.36):
            c.box(f"Conveyor leg {x} {y}", (x, y, 0.16), (0.064, 0.065, 0.32), "dark")
    for i in range(9):
        x = -0.47 + i * 0.116
        c.rod(f"Roller {i}", (x, -0.16, 0.37), (x, 0.15, 0.37), 0.035, "cream")
    for y in (-0.27, 0.25):
        c.box(f"Portal upright {y}", (0.06, y, 0.54), (0.11, 0.08, 0.63), "green")
    c.box("Portal scanner", (0.06, -0.01, 0.85), (0.14, 0.60, 0.10), "green")
    c.box(
        "Large emerging parcel", (-0.22, -0.02, 0.50), (0.27, 0.22, 0.19), "terracotta"
    )
    c.box("Small queued parcel", (0.34, 0.06, 0.46), (0.15, 0.14, 0.12), "timber")

    c = new(
        29,
        "slotted_mail_sorter",
        "Stepped mail sorter",
        "five-bay-offset-slot-array",
        "Progressively higher slanted dividers form asymmetric slots. Two projecting envelopes cross different openings.",
        "desktop organizers",
    )
    c.box("Sorter base", (0, 0.02, 0.046), (0.94, 0.59, 0.092), "dark")
    for i in range(5):
        x = -0.37 + i * 0.185
        h = 0.28 + i * 0.083
        c.box(
            f"Divider {i}",
            (x, 0.03, h / 2 + 0.052),
            (0.042, 0.46, h),
            "blue",
            (0, -12, 0),
        )
        c.box(
            f"Front label {i}", (x + 0.055, -0.21, 0.11), (0.11, 0.026, 0.073), "cream"
        )
    c.box(
        "Diagonal envelope",
        (-0.16, -0.06, 0.32),
        (0.24, 0.037, 0.29),
        "terracotta",
        (8, -12, 18),
    )
    c.box(
        "Tall envelope", (0.22, 0.05, 0.41), (0.22, 0.035, 0.35), "green", (6, -9, -24)
    )
    c.box("Back keeper", (0, 0.28, 0.16), (0.93, 0.045, 0.25), "timber")

    c = new(
        30,
        "metronome_bench",
        "Metronome and tuning forks",
        "pendulum-over-tapered-cabinet",
        "A swinging arm crosses a tapering metronome case. Separate tuning-fork tines rise behind the case and over a shared rear rest.",
        "music room mechanisms",
    )
    c.box("Low plinth", (-0.08, 0.04, 0.041), (0.91, 0.55, 0.082), "timber")
    for i, (z, w, d) in enumerate(
        ((0.15, 0.39, 0.32), (0.28, 0.32, 0.27), (0.41, 0.26, 0.22), (0.53, 0.19, 0.16))
    ):
        c.box(f"Tapered case tier {i}", (-0.20, 0.04, z), (w, d, 0.14), "terracotta")
    c.rod("Pendulum arm", (-0.23, -0.145, 0.18), (0.025, -0.145, 0.64), 0.018, "dark")
    c.box(
        "Sliding bob", (-0.035, -0.15, 0.52), (0.092, 0.052, 0.083), "cream", (0, 17, 0)
    )
    c.box("Fork rail", (0.25, 0.19, 0.16), (0.33, 0.16, 0.13), "green")
    for i, x in enumerate((0.14, 0.31)):
        z = 0.54 + i * 0.12
        c.rod(f"Fork stem {i}", (x, 0.18, 0.20), (x, 0.18, 0.36), 0.020, "blue")
        c.rod(
            f"Fork bridge {i}",
            (x - 0.04, 0.18, 0.36),
            (x + 0.04, 0.18, 0.36),
            0.019,
            "blue",
        )
        for side in (-1, 1):
            c.rod(
                f"Fork tine {i} {side}",
                (x + side * 0.04, 0.18, 0.36),
                (x + side * 0.04, 0.18, z),
                0.017,
                "blue",
            )

    c = new(
        31,
        "folding_easel",
        "Folding easel and paint tray",
        "rear-kickstand-and-leaning-panel",
        "The broad angled canvas masks a rear kickstand while the low paint tray and paint bottles project in front.",
        "folding art furniture",
    )
    for x in (-0.37, 0.37):
        c.rod(
            f"Easel leg {x}", (x, -0.19, 0.015), (x * 0.20, 0.15, 0.80), 0.037, "timber"
        )
    c.rod("Rear kickstand", (0.13, 0.44, 0.015), (-0.01, 0.15, 0.77), 0.036, "legs")
    c.box("Canvas", (-0.01, 0.01, 0.52), (0.61, 0.043, 0.49), "cream", (-12, 0, 3))
    c.box("Canvas ledge", (-0.01, -0.095, 0.31), (0.82, 0.14, 0.060), "blue")
    c.box(
        "Colour field",
        (-0.12, -0.03, 0.55),
        (0.20, 0.018, 0.15),
        "terracotta",
        (-12, 0, 3),
    )
    c.box("Tray", (0.32, -0.31, 0.14), (0.36, 0.25, 0.05), "green")
    for i, x in enumerate((0.22, 0.34, 0.44)):
        c.disk(
            f"Paint pot {i}",
            (x, -0.31, 0.20),
            0.043,
            0.09,
            ("blue", "cream", "terracotta")[i],
        )

    c = new(
        32,
        "gear_reducer",
        "Open three-gear reduction model",
        "three-offset-meshing-disc-trains",
        "Three toothed disks rotate on independent shafts. The large front gear overlaps the offset pinion and rear output wheel.",
        "exposed mechanical drives",
    )
    c.box("Gearbox foot", (0, 0.08, 0.045), (1.08, 0.45, 0.09), "legs")
    c.box("Rear bearing block", (0, 0.27, 0.30), (0.94, 0.10, 0.50), "green")
    for i, (x, y, z, r, n, m) in enumerate(
        (
            (-0.26, -0.17, 0.34, 0.22, 12, "blue"),
            (0.02, -0.07, 0.47, 0.12, 8, "cream"),
            (0.24, 0.055, 0.34, 0.20, 11, "terracotta"),
        )
    ):
        c.disk(f"Gear disk {i}", (x, y, z), r, 0.059, m, (90, 0, 0))
        c.rod(f"Transmission shaft {i}", (x, y - 0.05, z), (x, 0.29, z), 0.032, "dark")
        for j in range(n):
            a = 2 * pi * j / n
            c.box(
                f"Gear {i} tooth {j}",
                (x + (r + 0.014) * cos(a), y, z + (r + 0.014) * sin(a)),
                (0.058, 0.056, 0.040),
                m,
                (0, -a * 180 / pi, 0),
            )

    c = new(
        33,
        "tea_service",
        "Teapot service on two trays",
        "teapot-handle-and-nested-saucers",
        "The spout extends across stacked saucers with two cups at different depths; the handle creates an aperture against the rear tray.",
        "domestic ceramics",
    )
    c.disk("Serving tray", (-0.06, 0.03, 0.035), 0.46, 0.07, "timber")
    c.ball("Squat teapot", (-0.13, 0.08, 0.27), (0.23, 0.18, 0.20), "blue")
    c.disk("Lid", (-0.13, 0.08, 0.45), 0.12, 0.038, "cream")
    c.ball("Lid knob", (-0.13, 0.08, 0.49), (0.037, 0.037, 0.031), "terracotta")
    c.ring("Handle", (-0.35, 0.09, 0.28), 0.115, 0.032, "blue")
    c.rod("Spout", (0.02, 0.05, 0.30), (0.25, 0.03, 0.45), 0.055, "blue")
    for i, (x, y) in enumerate(((0.24, -0.19), (0.29, 0.22))):
        for j in range(i + 1):
            c.disk(f"Saucer {i} {j}", (x, y, 0.088 + j * 0.031), 0.13, 0.025, "cream")
        c.cone(f"Teacup {i}", (x, y, 0.17 + i * 0.03), 0.071, 0.092, 0.13, "green")
        c.ring(
            f"Cup handle {i}", (x + 0.082, y, 0.17 + i * 0.03), 0.038, 0.013, "green"
        )

    c = new(
        34,
        "gravity_chute",
        "Gravity chute and marble catch",
        "three-offset-switching-troughs",
        "Three sloping rails reverse direction at alternating ends. Balls and a catch bin occupy separate foreground and rear depths.",
        "mechanical ball routing",
    )
    for x in (-0.43, 0.43):
        c.box(f"Chute upright {x}", (x, 0.21, 0.39), (0.065, 0.085, 0.78), "legs")
    for i, (z, y, d) in enumerate(
        ((0.65, 0.15, 1), (0.43, 0.02, -1), (0.23, -0.13, 1))
    ):
        c.box(
            f"Sloped tray {i}",
            (0, y, z),
            (0.88, 0.16, 0.041),
            ("terracotta", "blue", "green")[i],
            (0, 11 * d, 0),
        )
        for side in (-1, 1):
            c.rod(
                f"Tray guard {i} {side}",
                (-0.43, y + side * 0.072, z + d * 0.085 + 0.04),
                (0.43, y + side * 0.072, z - d * 0.085 + 0.04),
                0.013,
                "cream",
            )
    c.box("Catch bin floor", (0.33, -0.29, 0.053), (0.34, 0.25, 0.08), "timber")
    c.box("Catch bin lip", (0.33, -0.405, 0.11), (0.34, 0.037, 0.15), "green")
    c.ball("Top marble", (-0.18, 0.15, 0.74), (0.055,) * 3, "cream")
    c.ball("Lower marble", (0.21, -0.14, 0.25), (0.052,) * 3, "terracotta")

    c = new(
        35,
        "windmill_model",
        "Windmill model with service shed",
        "four-sails-and-tapered-tower",
        "Four broad radial sails cross a tapered tower and an attached low shed. The lower sail masks the tower entrance.",
        "scale architecture",
    )
    c.cone("Tower", (-0.09, 0.11, 0.34), 0.23, 0.13, 0.67, "cream")
    c.cone("Roof", (-0.09, 0.11, 0.73), 0.18, 0.018, 0.18, "terracotta")
    c.rod("Windshaft", (-0.09, -0.20, 0.55), (-0.09, 0.15, 0.55), 0.041, "dark")
    for j in range(4):
        a = 0.45 + j * pi / 2
        x, z = cos(a), sin(a)
        c.rod(
            f"Sail spar {j}",
            (-0.09, -0.20, 0.55),
            (-0.09 + 0.36 * x, -0.20, 0.55 + 0.36 * z),
            0.016,
            "legs",
        )
        c.box(
            f"Sail panel {j}",
            (-0.09 + 0.245 * x, -0.207, 0.55 + 0.245 * z),
            (0.22, 0.036, 0.088),
            "blue",
            (0, -a * 180 / pi, 0),
        )
    c.box("Service shed", (0.29, 0.14, 0.15), (0.37, 0.33, 0.30), "green")
    c.box("Shed overhang", (0.29, 0.14, 0.325), (0.46, 0.40, 0.05), "timber", (7, 0, 0))

    c = new(
        36,
        "cable_reel_cart",
        "Cable reel service trolley",
        "offset-reels-on-low-wheeled-cart",
        "One transverse spool partly hides a smaller raised spool. The angled trolley handle cuts across the rear spool flange.",
        "wheeled workshop equipment",
    )
    c.box("Cart bed", (0, 0, 0.22), (0.98, 0.42, 0.07), "green")
    for x in (-0.36, 0.34):
        for y in (-0.22, 0.22):
            c.disk(
                f"Trolley wheel {x} {y}", (x, y, 0.10), 0.10, 0.05, "dark", (90, 0, 0)
            )
    for i, (x, z, r) in enumerate(((-0.18, 0.42, 0.17), (0.26, 0.51, 0.13))):
        c.rod(f"Drum core {i}", (x, -0.12, z), (x, 0.15, z), r * 0.72, "timber")
        for j, y in enumerate((-0.13, 0.16)):
            c.disk(
                f"Reel flange {i}-{j}",
                (x, y, z),
                r,
                0.04,
                "blue" if i == 0 else "terracotta",
                (90, 0, 0),
            )
        c.box(f"Reel bracket {i}", (x, 0.18, 0.32), (0.08, 0.06, 0.21), "cream")
    for y in (-0.20, 0.20):
        c.rod(f"Handle arm {y}", (0.42, y, 0.24), (0.56, y, 0.64), 0.020, "legs")
    c.rod("Push handle", (0.56, -0.20, 0.64), (0.56, 0.20, 0.64), 0.024, "legs")

    c = new(
        37,
        "bottle_carrier",
        "Partitioned bottle carrier",
        "six-open-cells-with-overhead-handle",
        "An elevated handle spans a partitioned crate. Front and rear bottles have different heights and mask neighboring cells.",
        "handled domestic storage",
    )
    c.box("Carrier floor", (0, 0.02, 0.065), (0.82, 0.53, 0.09), "timber")
    for y in (-0.25, 0.28):
        c.box(f"End face {y}", (0, y, 0.19), (0.83, 0.042, 0.23), "green")
    for x in (-0.40, 0.40):
        c.box(f"Carrier side {x}", (x, 0.02, 0.31), (0.046, 0.52, 0.50), "timber")
    c.rod("Carrying handle", (-0.40, 0.04, 0.59), (0.40, 0.04, 0.59), 0.042, "legs")
    for x in (-0.14, 0.14):
        c.box(f"Cell divider {x}", (x, 0.02, 0.22), (0.028, 0.47, 0.24), "cream")
    for i, (x, y, z) in enumerate(
        (
            (-0.26, -0.11, 0.30),
            (0.02, -0.11, 0.23),
            (0.26, 0.16, 0.34),
            (-0.24, 0.16, 0.28),
        )
    ):
        h = z + 0.08
        c.disk(
            f"Bottle body {i}",
            (x, y, 0.11 + h / 2),
            0.069,
            h,
            ("blue", "green", "terracotta", "cream")[i],
        )
        c.cone(f"Bottle shoulder {i}", (x, y, 0.14 + h), 0.069, 0.033, 0.067, "blue")
        c.disk(f"Bottle neck {i}", (x, y, 0.20 + h), 0.032, 0.063, "dark")

    c = new(
        38,
        "music_stand",
        "Music stand and mallet rack",
        "tripod-with-offset-scored-board",
        "The slanted score board masks the rear leg and tall mallets. Its projecting lip occupies a different depth from the rack.",
        "adjustable stands",
    )
    c.rod("Stand upright", (-0.13, 0.02, 0.09), (-0.13, 0.02, 0.69), 0.030, "legs")
    for i, (x, y) in enumerate(((-0.42, -0.22), (0.15, -0.17), (-0.13, 0.31))):
        c.rod(f"Tripod foot {i}", (-0.13, 0.02, 0.14), (x, y, 0.025), 0.026, "dark")
    c.box("Music board", (-0.12, 0.02, 0.63), (0.64, 0.048, 0.37), "blue", (-17, 0, 8))
    c.box(
        "Score sheet", (-0.17, -0.012, 0.65), (0.34, 0.020, 0.24), "cream", (-17, 0, 8)
    )
    c.box(
        "Board lip", (-0.12, -0.068, 0.45), (0.71, 0.12, 0.041), "terracotta", (0, 0, 8)
    )
    c.box("Mallet rack", (0.32, 0.19, 0.12), (0.31, 0.22, 0.19), "timber")
    for i, x in enumerate((0.22, 0.32, 0.43)):
        y = 0.18 + (i % 2) * 0.035
        z = 0.50 + i * 0.045
        c.rod(f"Mallet {i}", (x, y, 0.20), (x - 0.055, y, z), 0.014, "legs")
        c.ball(f"Beater head {i}", (x - 0.055, y, z), (0.046,) * 3, "green")

    c = new(
        39,
        "vault_course",
        "Gym vault and balance course",
        "vault-buck-and-offset-balance-beam",
        "A padded vault buck stands in front of a diagonal beam. The vault legs mask separate balancing blocks under the beam.",
        "gymnasium equipment",
    )
    c.ball("Vault buck", (-0.19, -0.13, 0.45), (0.28, 0.13, 0.13), "terracotta")
    for x in (-0.34, -0.05):
        for y in (-0.24, -0.03):
            c.rod(
                f"Buck leg {x} {y}",
                (x, y, 0.37),
                (x * 1.2, y * 1.36, 0.025),
                0.031,
                "legs",
            )
    c.box(
        "Balance beam", (0.14, 0.23, 0.48), (0.91, 0.105, 0.09), "timber", (0, 0, -13)
    )
    for x, y in ((-0.17, 0.30), (0.43, 0.16)):
        c.box(f"Beam prop {x}", (x, y, 0.23), (0.074, 0.077, 0.46), "blue")
        c.box(f"Beam spreader {x}", (x, y, 0.045), (0.19, 0.29, 0.09), "blue")
    c.box("Mounting block", (0.30, -0.24, 0.085), (0.30, 0.25, 0.17), "green")
    c.box("Top step", (0.40, -0.18, 0.23), (0.16, 0.19, 0.12), "cream")

    c = new(
        40,
        "corner_sink",
        "Sink with raised drainboard",
        "offset-basin-and-arched-tap",
        "The front basin lip hides a recessed well. Its raised drainboard and crossing faucet create separate occluding silhouettes.",
        "domestic fixtures",
    )
    c.box("Cabinet", (-0.06, 0.02, 0.24), (0.90, 0.48, 0.48), "green")
    c.box("Tap mount", (-0.28, 0.25, 0.57), (0.35, 0.11, 0.19), "timber")
    c.box("Basin floor", (-0.19, -0.02, 0.487), (0.42, 0.29, 0.021), "dark")
    for i, (x, y, w, d) in enumerate(
        (
            (-0.40, -0.02, 0.045, 0.36),
            (0.025, -0.02, 0.045, 0.36),
            (-0.19, -0.18, 0.46, 0.045),
            (-0.19, 0.14, 0.46, 0.045),
        )
    ):
        c.box(f"Basin rim {i}", (x, y, 0.545), (w, d, 0.11), "cream")
    c.box("Drainboard", (0.26, 0.025, 0.56), (0.37, 0.45, 0.065), "blue", (4, 0, -2))
    for i in range(4):
        c.rod(
            f"Drain flute {i}",
            (0.12 + i * 0.08, -0.16, 0.58),
            (0.12 + i * 0.08, 0.20, 0.605),
            0.012,
            "cream",
        )
    c.rod("Tap riser", (-0.26, 0.21, 0.62), (-0.26, 0.21, 0.79), 0.025, "legs")
    c.rod("Tap arm", (-0.26, 0.21, 0.79), (-0.19, -0.07, 0.79), 0.025, "legs")
    c.rod("Spout", (-0.19, -0.07, 0.79), (-0.19, -0.07, 0.72), 0.023, "legs")

    c = new(
        41,
        "rotated_book_display",
        "Two-island crossed book display",
        "crossed-tabletop-books-on-separated-islands",
        "Two counter-facing supports show fanned books at different depths. A broad face-out volume hides a stack at the gap.",
        "familiar category: books and shelving from scene_002_book_nook; two low islands, fanned volumes and no bookshelf bays",
    )
    for side, (x, y, a) in enumerate(((-0.28, -0.12, -18), (0.28, 0.21, 23))):
        c.box(
            f"Display island {side}",
            (x, y, 0.09),
            (0.45, 0.30, 0.15),
            "timber",
            (0, 0, a),
        )
        for j in range(3):
            c.box(
                f"Fanned volume {side}-{j}",
                (x - 0.085 + j * 0.078, y + j * 0.025, 0.36 + j * 0.028),
                (0.081, 0.22, 0.40 + j * 0.033),
                ("green", "cream", "blue")[j],
                (0, -12 + j * 9, a),
            )
    c.box(
        "Face-out folio",
        (-0.01, -0.26, 0.34),
        (0.38, 0.055, 0.43),
        "terracotta",
        (-9, 0, 17),
    )
    for i in range(3):
        c.box(
            f"Horizontal atlas {i}",
            (0.28, -0.12, 0.06 + i * 0.068),
            (0.35, 0.21, 0.06),
            ("blue", "cream", "green")[i],
            (0, 0, -8 + i * 6),
        )

    c = new(
        42,
        "twin_spool_pulleys",
        "Split-level winch and moving pulley",
        "offset-floor-winch-and-overhanging-sheave",
        "A floor winch leads ropes around two offset sheaves to a moving cradle. Foreground rope and drum cross the hoist foot.",
        "familiar category: pulley equipment from scene_003_pulley_frame; separated ground winch and one-sided mast replace the overhead portal",
    )
    c.box("Mast shoe", (0.29, 0.18, 0.075), (0.43, 0.35, 0.15), "blue")
    c.box("Offset mast", (0.34, 0.20, 0.47), (0.105, 0.13, 0.75), "timber")
    c.box("Jib", (0.12, 0.18, 0.80), (0.56, 0.13, 0.083), "green")
    c.wheel("High sheave", -0.10, 0.095, 0.72, 0.105, 5)
    c.wheel("Cradle sheave", -0.10, -0.025, 0.37, 0.076, 4)
    for x in (-0.18, -0.02):
        c.rod(f"Hoist rope {x}", (x, 0.07, 0.71), (x, -0.04, 0.37), 0.012, "cream")
    c.box("Moving cradle", (-0.10, -0.07, 0.25), (0.27, 0.22, 0.12), "terracotta")
    c.rod("Floor winch core", (-0.39, -0.24, 0.19), (-0.39, 0.09, 0.19), 0.103, "legs")
    for y in (-0.24, 0.08):
        c.disk(f"Winch side {y}", (-0.39, y, 0.19), 0.15, 0.043, "blue", (90, 0, 0))
    c.rod("Winch feed", (-0.40, -0.13, 0.29), (-0.19, 0.06, 0.71), 0.012, "cream")

    c = new(
        43,
        "corner_plant_steps",
        "Corner plant display steps",
        "orthogonal-five-trays-at-three-levels",
        "An L-shaped bank of plinths climbs away from the viewer. Broad front leaves hide rims and upright rear stems.",
        "familiar category: pots and plant stand from scene_012_plant_stand; solid L-shaped stepped beds replace the stand",
    )
    for i, (x, y, z, w) in enumerate(
        (
            (-0.38, -0.16, 0.13, 0.29),
            (-0.08, -0.14, 0.22, 0.28),
            (0.22, -0.10, 0.32, 0.29),
            (0.26, 0.15, 0.43, 0.28),
            (0.23, 0.37, 0.51, 0.30),
        )
    ):
        c.box(
            f"Corner plinth {i}",
            (x, y, z / 2),
            (w, 0.24, z),
            "timber" if i % 2 else "green",
        )
        c.cone(f"Small pot {i}", (x, y, z + 0.07), 0.062, 0.087, 0.14, "terracotta")
        c.rod(f"Stem {i}", (x, y, z + 0.13), (x + 0.025, y, z + 0.26), 0.010, "legs")
        c.ball(
            f"Leaf crown {i}",
            (x + 0.025, y, z + 0.26),
            (0.105, 0.066, 0.076),
            "green",
            (12, i * 9, 0),
        )

    c = new(
        44,
        "bent_pipe_wall",
        "Cross-feed and three pipe returns",
        "three-nested-lateral-u-returns",
        "Parallel return loops bend back toward the same wall. A forward cross-feed and valve conceal portions of the middle loop.",
        "familiar category: pipework from scene_009_pipe_manifold; nested U returns at three depth levels replace a branched manifold",
    )
    c.box("Pipe mounting plate", (0, 0.32, 0.40), (0.99, 0.06, 0.70), "timber")
    for i, (x, z, y) in enumerate(
        ((-0.34, 0.24, -0.16), (-0.06, 0.39, 0.035), (0.21, 0.54, 0.18))
    ):
        m = ("blue", "green", "cream")[i]
        ends = [(x, 0.31, z), (x, y, z), (x + 0.18, y, z), (x + 0.18, 0.31, z)]
        for j in range(3):
            c.rod(f"Return {i} run {j}", ends[j], ends[j + 1], 0.034, m)
        for j in (1, 2):
            c.ball(f"Return {i} joint {j}", ends[j], (0.044,) * 3, m)
    c.rod("Cross feed", (-0.46, -0.25, 0.35), (0.43, -0.25, 0.49), 0.035, "terracotta")
    c.wheel("Shutoff valve", 0.12, -0.29, 0.435, 0.082, 4)

    c = new(
        45,
        "folded_canopy",
        "Pleated triangular-prism canopy",
        "continuous-triangular-prism-pleats",
        "Six neighboring triangular roof prisms create repeated planar folds over an open entrance. The near folds cover different rear supports.",
        "novel geometry: repeated three-sided, constant-cross-section solid prism roof on high posts; reserved against all training/development designs",
    )
    for i in range(6):
        x = -0.43 + i * 0.172
        c.part(
            f"Trihedral canopy fold {i}",
            "frustum",
            (x, 0.045, 0.64),
            "cream" if i % 2 else "blue",
            bottom_radius=0.102,
            top_radius=0.102,
            depth=0.60,
            vertices=3,
            rotation_deg=(90, 0, 0),
        )
    for x in (-0.45, 0.45):
        for y in (-0.23, 0.29):
            c.rod(f"Slender support {x} {y}", (x, y, 0.02), (x, y, 0.62), 0.023, "legs")
    c.box("Rear service counter", (0.11, 0.21, 0.28), (0.65, 0.18, 0.45), "green")
    c.box(
        "Foreground ticket stand",
        (-0.26, -0.24, 0.19),
        (0.21, 0.20, 0.38),
        "terracotta",
    )

    c = new(
        46,
        "basalt_steps",
        "Pentagonal causeway steps",
        "staggered-pentagonal-taper-field",
        "Separate tapered five-sided columns rise in three staggered ranks. Their flat faces meet at stepped silhouettes and mask distant columns.",
        "novel geometry: field of faceted pentagonal tapered solids; no three- or five-sided frusta appear in the training or development families",
    )
    for j in range(3):
        for i in range(5):
            x = -0.41 + i * 0.182 + (j % 2) * 0.061
            y = -0.24 + j * 0.21
            h = 0.18 + 0.075 * i + 0.095 * j + (0.07 if (i + j) % 3 == 0 else 0)
            c.part(
                f"Pentagonal column {j}-{i}",
                "frustum",
                (x, y, h / 2),
                ("green", "cream", "blue")[(i + 2 * j) % 3],
                bottom_radius=0.113,
                top_radius=0.091,
                depth=h,
                vertices=5,
                rotation_deg=(0, 0, 11 + 13 * ((i + 2 * j) % 3)),
            )
    c.box(
        "Front survey ruler",
        (-0.23, -0.37, 0.105),
        (0.37, 0.037, 0.09),
        "terracotta",
        (0, -9, 0),
    )

    c = new(
        47,
        "twin_helices",
        "Twin helical demonstration spring",
        "opposed-phase-open-helices-around-core",
        "Two open helical paths climb around a core. Their crossings alternate between front and back at each height.",
        "novel geometry: persistent screw translation per rotation and a phase-offset partner; old circular hoops and planar arcs lack pitch",
    )
    c.disk("Spring foot", (-0.01, 0.05, 0.07), 0.31, 0.14, "dark")
    c.rod("Central piston", (-0.01, 0.05, 0.10), (-0.01, 0.05, 0.76), 0.074, "cream")
    for k in range(2):
        pts = []
        for i in range(33):
            a = 4 * pi * i / 32 + k * pi
            pts.append(
                (-0.01 + 0.21 * cos(a), 0.05 + 0.21 * sin(a), 0.13 + 0.60 * i / 32)
            )
        for i, (a, b) in enumerate(itertools.pairwise(pts)):
            c.rod(f"Helix {k} link {i}", a, b, 0.021, ("blue", "terracotta")[k])
        for i in (0, 8, 16, 24, 32):
            c.ball(
                f"Helix {k} witness joint {i}",
                pts[i],
                (0.028,) * 3,
                ("blue", "terracotta")[k],
            )
    c.disk("Piston cap", (-0.01, 0.05, 0.78), 0.14, 0.055, "green")

    c = new(
        48,
        "antiprism_lantern",
        "Open antiprism lantern",
        "twisted-octagonal-antiprism-cage",
        "Two rotated octagonal perimeters connect by alternating triangular struts around an opaque core. Their openings change with camera position.",
        "novel geometry: equal-edge-count offset polygon loops with antiprism triangular adjacency; training contains no twisted antiprism cage",
    )
    loops = []
    for j, (z, r, phase) in enumerate(((0.12, 0.29, 0), (0.60, 0.25, pi / 8))):
        pt = [
            (r * cos(2 * pi * i / 8 + phase), 0.04 + r * sin(2 * pi * i / 8 + phase), z)
            for i in range(8)
        ]
        loops.append(pt)
        for i in range(8):
            c.rod(f"Polygon rim {j}-{i}", pt[i], pt[(i + 1) % 8], 0.023, "dark")
    for i in range(8):
        c.rod(f"Antiprism strut a{i}", loops[0][i], loops[1][i], 0.016, "blue")
        c.rod(
            f"Antiprism strut b{i}", loops[0][(i + 1) % 8], loops[1][i], 0.016, "green"
        )
    c.cone("Lantern base", (0, 0.04, 0.065), 0.32, 0.28, 0.13, "timber")
    c.cone("Central shade", (0, 0.04, 0.37), 0.104, 0.069, 0.34, "terracotta")
    c.cone("Cap", (0, 0.04, 0.647), 0.27, 0.11, 0.09, "cream")
    return rows
