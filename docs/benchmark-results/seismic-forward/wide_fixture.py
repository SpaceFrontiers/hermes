"""Relabel a prefix of a real CSR corpus into a 100k vocabulary.
Weights and sparsity are byte-preserved; every ID gains 65536. This exercises
17-bit IDs without claiming to be a representative 100k-vocabulary model.
"""

import struct
import sys
from array import array
from pathlib import Path


def shift(source, destination, limit=None):
    with Path(source).open("rb") as f:
        rows, dims, nnz = struct.unpack("<QQQ", f.read(24))
        selected = min(rows, limit) if limit else rows
        pointers = array("Q")
        pointers.fromfile(f, selected + 1)
        count = pointers[-1]
        start = 24 + (rows + 1) * 8
        f.seek(start)
        indices = array("I")
        indices.fromfile(f, count)
        assert sys.byteorder == "little" and max(indices, default=0) + 65536 < 100000
        for i in range(len(indices)):
            indices[i] += 65536
        with Path(destination).open("wb") as out:
            out.write(struct.pack("<QQQ", selected, 100000, count))
            pointers.tofile(out)
            indices.tofile(out)
            f.seek(start + nnz * 4)
            remaining = count * 4
            while remaining:
                chunk = f.read(min(remaining, 8 * 1024 * 1024))
                assert chunk
                out.write(chunk)
                remaining -= len(chunk)
    return selected


if __name__ == "__main__":
    print(
        shift(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    )
