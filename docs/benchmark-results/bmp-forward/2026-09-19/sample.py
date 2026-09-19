# Read-only measurement of deployed BMPA, not an application format reader.
# Emits sampled vector values; keep the output private. Reads eight windows of
# sixteen rows per field/segment, each capped at 16 MiB. Does not scan payloads.
import collections
import gzip
import json
import os
import random
import struct
import sys

rng = random.Random(260919)
report = {"shards": [], "samples": [], "skipped": [], "read_bytes": 0}


def read(fd, at, n):
    assert 0 <= n <= 16 * 1024 * 1024
    b = os.pread(fd, n, at)
    report["read_bytes"] += len(b)
    assert len(b) == n
    return b


if len(sys.argv) < 2:
    raise SystemExit("usage: python3 sample.py INDEX_DIR... > sample.json.gz")
for root in sys.argv[1:]:
    with open(root + "/metadata.json") as metadata:
        m = json.load(metadata)
    sizes = collections.Counter()
    for e in os.scandir(root):
        if e.is_file():
            sizes[e.name.rsplit(".", 1)[-1]] += e.stat().st_size
    shard = {
        "path": root,
        "generation": m["publication_generation"],
        "schema": m["schema"],
        "sizes": sizes,
        "documents": sum(x["num_docs"] for x in m["segment_metas"].values()),
        "deleted": sum(
            x.get("deletions", {}).get("num_deleted", 0)
            for x in m["segment_metas"].values()
        ),
        "fields": [],
    }
    report["shards"].append(shard)
    for seg, meta in sorted(m["segment_metas"].items()):
        path = root + "/seg_" + seg + ".sparse"
        try:
            fd = os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            report["skipped"].append(path)
            continue
        try:
            size = os.fstat(fd).st_size
            skip, toc, nf, magic = struct.unpack("<QQII", read(fd, size - 24, 24))
            assert magic == 0x34525053
            pos = toc
            for _ in range(nf):
                field, quant, ndims, nvectors = struct.unpack(
                    "<IBII", read(fd, pos, 13)
                )
                pos += 13
                assert ndims == 1
                sentinel, blob, lengthlo, lengthhi, _, _ = struct.unpack(
                    "<IQIIIf", read(fd, pos, 28)
                )
                pos += 28
                assert sentinel == 0xFFFFFFFF
                length = lengthlo + (lengthhi << 32)
                footer = blob + length - 80
                (
                    terms,
                    postings,
                    grid,
                    sb,
                    coarse,
                    blocks,
                    dims,
                    blocksize,
                    slots,
                    scale,
                    docmap,
                    real,
                    bits,
                    bmagic,
                ) = struct.unpack("<QQQQQIIIIfQIII", read(fd, footer, 80))
                assert bmagic == 0x41504D42
                count, flags, payloadlen = struct.unpack(
                    "<IIQ", read(fd, footer - 16, 16)
                )
                directory = footer - 16 - count * 16
                payload = directory - payloadlen
                info = {
                    "segment": seg,
                    "field": field,
                    "num_docs": meta["num_docs"],
                    "vectors": real,
                    "dims": dims,
                    "blob": length,
                    "inverted": grid - (blocks + 1) * 8,
                    "block_offsets": (blocks + 1) * 8,
                    "grids": docmap - grid,
                    "docmap": slots * 6,
                    "forward_payload": payloadlen,
                    "forward_directory": count * 16,
                    "flags": flags,
                }
                shard["fields"].append(info)
                if flags or count < 16:
                    continue
                for window in range(8):
                    lo = count * window // 8
                    hi = count * (window + 1) // 8
                    if hi - lo < 16:
                        continue
                    start = rng.randrange(lo, hi - 15)
                    entries = read(
                        fd, directory + start * 16, min(17, count - start) * 16
                    )
                    decoded = list(struct.iter_unpack("<IHHQ", entries))
                    end = decoded[16][3] if len(decoded) == 17 else payloadlen
                    begin = decoded[0][3]
                    if end - begin > 16 * 1024 * 1024:
                        report["skipped"].append([path, field, start, "oversize"])
                        continue
                    raw = read(fd, payload + begin, end - begin)
                    for j, (doc, ordinal, _reserved, at) in enumerate(decoded[:16]):
                        stop = decoded[j + 1][3] if j + 1 < len(decoded) else payloadlen
                        row = raw[at - begin : stop - begin]
                        assert len(row) % 5 == 0
                        values = list(struct.iter_unpack("<IB", row))
                        assert all(
                            a[0] <= b[0]
                            for a, b in zip(values, values[1:], strict=False)
                        )
                        assert all(d < dims and w > 0 for d, w in values)
                        report["samples"].append(
                            {
                                "shard": len(report["shards"]) - 1,
                                "segment": seg,
                                "field": field,
                                "row": start + j,
                                "offset": at,
                                "doc": doc,
                                "ordinal": ordinal,
                                "values": values,
                            }
                        )
        finally:
            os.close(fd)
sys.stdout.buffer.write(
    gzip.compress(json.dumps(report, separators=(",", ":")).encode())
)
