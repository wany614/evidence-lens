"""Read the public example's PNG and NumPy containers using their registered numerical layouts."""

import io
import struct
import zipfile
import zlib

import numpy as np
from PIL import Image

from evidence_lens.l1 import KEYS
from evidence_lens.scoring_io import VIEWS, ReadAudit, load_cameras


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_png(path, size):
    data = path.read_bytes()
    require(data.startswith(b"\x89PNG\r\n\x1a\n"), f"invalid PNG signature in {path.name}")
    # Only standard pixel, palette and numerical colour/physical-size chunks are public.
    fixed_sizes = {
        b"IHDR": 13,
        b"IEND": 0,
        b"tRNS": 6,
        b"sRGB": 1,
        b"cHRM": 32,
        b"gAMA": 4,
        b"sBIT": 3,
        b"pHYs": 9,
        b"bKGD": 6,
    }
    allowed = set(fixed_sizes) | {b"PLTE", b"IDAT"}
    offset, chunks = 8, []
    decoder = zlib.decompressobj()
    # The example producer writes noninterlaced RGB8 with one filter byte per row.
    remaining = (3 * size[0] + 1) * size[1]
    while offset < len(data):
        require(offset + 12 <= len(data), f"truncated PNG chunk in {path.name}")
        length, kind = struct.unpack_from(">I4s", data, offset)
        end = offset + 12 + length
        require(kind in allowed and end <= len(data), f"unapproved or invalid PNG chunk in {path.name}")
        require(
            (kind not in fixed_sizes or fixed_sizes[kind] == length)
            and (kind == b"IDAT" or kind not in chunks),
            f"invalid numerical PNG chunk in {path.name}",
        )
        payload = data[offset + 8 : end - 4]
        if kind == b"IHDR":
            w, h, depth, colour, comp, filt, interlace = struct.unpack(">IIBBBBB", payload)
            require(
                (w, h) == size and (depth, colour, comp, filt, interlace) == (8, 2, 0, 0, 0),
                f"invalid RGB8 example header in {path.name}",
            )
        if kind == b"PLTE":
            require(3 <= length <= 768 and length % 3 == 0, f"invalid PNG palette in {path.name}")
        if kind == b"IDAT":
            remaining -= len(decoder.decompress(payload, remaining + 1))
            require(
                remaining >= 0 and not decoder.unused_data and not decoder.unconsumed_tail,
                f"PNG has extra or oversized compressed data in {path.name}",
            )
        elif kind != b"IEND":
            require(b"IDAT" not in chunks, f"invalid PNG chunk ordering in {path.name}")
        actual_crc = zlib.crc32(data[offset + 4 : end - 4])
        require(
            actual_crc == struct.unpack_from(">I", data, end - 4)[0], f"invalid PNG checksum in {path.name}"
        )
        chunks.append(kind)
        offset = end
        if kind == b"IEND":
            require(length == 0 and end == len(data), f"PNG has trailing data in {path.name}")
            break
    require(
        chunks
        and chunks[0] == b"IHDR"
        and chunks.count(b"IHDR") == 1
        and chunks[-1] == b"IEND"
        and b"IDAT" in chunks
        and decoder.eof
        and remaining == 0,
        f"incomplete PNG in {path.name}",
    )
    with Image.open(io.BytesIO(data), formats=["PNG"]) as img:
        require(
            img.size == size and img.mode == "RGB", f"PNG dimensions or color channels differ in {path.name}"
        )
        img.load()


def member_offsets(stream, packed, members):
    """Account for every container byte, including NumPy's optional local ZIP64 sizes."""
    stream.seek(-22, 2)
    end_offset = stream.tell()
    marker, disk, directory_disk, on_disk, total, directory_size, directory_start, comment = struct.unpack(
        "<4s4H2IH", stream.read(22)
    )
    require(
        (marker, disk, directory_disk, on_disk, total, comment)
        == (b"PK\x05\x06", 0, 0, len(members), len(members), 0)
        and directory_start == packed.start_dir
        and directory_start + directory_size == end_offset,
        "NPZ has extra data or an invalid directory boundary",
    )
    # Each ASCII name follows a 46-byte central header. The public layout permits no central extra/comment.
    require(
        directory_size == sum(46 + len(i.filename) for i in members)
        and all(not i.extra and not i.comment for i in members),
        "NPZ has unapproved central metadata",
    )
    offsets = []
    position = 0
    for info in members:
        require(position == info.header_offset, "NPZ local members are not contiguous")
        stream.seek(position)
        header = stream.read(30)
        require(len(header) == 30, "incomplete NPZ local header")
        tag, version, flags, method, _, _, crc, comp, size, name_len, extra_len = struct.unpack(
            "<4s5H3I2H", header
        )
        require(
            tag == b"PK\x03\x04"
            and version == info.extract_version
            and flags == info.flag_bits == 0
            and method == info.compress_type
            and method in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
            and crc == info.CRC
            and stream.read(name_len) == info.filename.encode("ascii"),
            "NPZ local header differs from its directory",
        )
        extra = stream.read(extra_len)
        if extra_len:
            require(
                comp == size == 0xFFFFFFFF
                and extra == struct.pack("<HHQQ", 1, 16, info.file_size, info.compress_size),
                "unapproved NPZ local extra field or inconsistent ZIP64 sizes",
            )
        else:
            require((comp, size) == (info.compress_size, info.file_size), "incorrect NPZ local sizes")
        offsets.append(stream.tell())
        position = stream.tell() + info.compress_size
    require(position == directory_start, "NPZ local data and central directory are not contiguous")
    return offsets


def check_npz(path, layout):
    require(path.stat().st_size >= 22, f"truncated NPZ in {path.name}")
    with path.open("rb") as stream, zipfile.ZipFile(stream) as packed:
        members = packed.infolist()
        names = {k + ".npy" for k in layout}
        require(
            not packed.comment and len(members) == len(names) and {i.orig_filename for i in members} == names,
            f"unapproved or missing numerical NPZ member in {path.name}",
        )
        for info, offset in zip(members, member_offsets(stream, packed, members), strict=True):
            kind, shape = layout[info.filename[:-4]]
            nbytes = np.dtype(kind).itemsize * int(np.prod(shape))
            require(nbytes < info.file_size <= nbytes + 1040, f"NPZ member length differs in {path.name}")
            stream.seek(offset)
            data = stream.read(info.compress_size)
            if info.compress_type == zipfile.ZIP_DEFLATED:
                inflater = zlib.decompressobj(-15)
                data = inflater.decompress(data, nbytes + 1041)
                require(
                    inflater.eof and not inflater.unused_data and not inflater.unconsumed_tail,
                    f"NPZ has extra or incomplete compressed data in {path.name}",
                )
            require(
                len(data) == info.file_size and zlib.crc32(data) == info.CRC,
                f"invalid numerical member bytes in {path.name}",
            )
            with io.BytesIO(data) as raw:
                version = np.lib.format.read_magic(raw)
                require(version in ((1, 0), (2, 0), (3, 0)), f"unknown NPY version in {path.name}")
                length = int.from_bytes(raw.read(2 if version[0] == 1 else 4), "little")
                header = raw.read(length)
                require(
                    length <= 1024
                    and header.endswith(b"\n")
                    and b"#" not in header
                    and b"\\" not in header
                    and all(32 <= c < 127 for c in header[:-1]),
                    f"unapproved NPY header in {path.name}",
                )
                raw.seek(0)
                array = np.load(raw, allow_pickle=False, max_header_size=1024)
                require(
                    isinstance(array, np.ndarray)
                    and array.dtype == np.dtype(kind)
                    and array.shape == shape
                    and raw.tell() == raw.getbuffer().nbytes,
                    f"NPZ dtype, shape or trailing payload differs in {path.name}",
                )


def check_example_media(folder):
    cameras = load_cameras(folder / "example/cameras.json", ReadAudit())
    for view, name in (("source", "input.png"), ("v01", "v01-prediction.png")):
        h, w = cameras[view].shape
        check_png(folder / "example" / name, (w, h))
    for view in VIEWS:
        shape = cameras[view].shape
        check_npz(
            folder / "example/predictions" / (view + ".npz"),
            {"rgb": (np.float32, (*shape, 3)), "z": (np.float32, shape), "alpha": (np.float32, shape)},
        )
    shape = cameras["v01"].shape
    expected = {"l0_reason": (np.uint8, shape)}
    for key in KEYS.values():
        expected.update(
            {key: (np.float32, shape), key + "_valid": (np.bool_, shape), key + "_state": (np.uint8, shape)}
        )
    check_npz(folder / "expected/v01.npz", expected)
