"""Minimal GGUF v3 writer/reader (hand-written; no dependency on the ``gguf`` package).

File layout (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)::

    u32 magic "GGUF" | u32 version=3 | u64 n_tensors | u64 n_kv
    n_kv x (string key, u32 type, value)
    n_tensors x (string name, u32 n_dims, u64 ne[n_dims], u32 ggml_type, u64 offset)
    padding to `general.alignment`, then tensor data (each tensor aligned)

``ne[0]`` is the contiguous dimension, so a PyTorch (rows, cols) weight is
stored with ne = [cols, rows].
"""
import struct

import numpy as np

MAGIC = 0x46554747
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)
GGML_F32, GGML_F16 = 0, 1
_FMT = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i", F32: "<f", BOOL: "<?",
        U64: "<Q", I64: "<q", F64: "<d"}


def _s(x) -> bytes:
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


class GGUFWriter:
    def __init__(self, path: str, alignment: int = 32):
        self.path, self.alignment = path, alignment
        self.kv: list[tuple[str, bytes]] = []
        self.tensors: list[tuple[str, list, int, bytes]] = []
        self.u32("general.alignment", alignment)

    def _kv(self, k, t, payload):
        self.kv.append((k, struct.pack("<I", t) + payload))

    def u32(self, k, v): self._kv(k, U32, struct.pack("<I", int(v)))
    def f32(self, k, v): self._kv(k, F32, struct.pack("<f", float(v)))
    def string(self, k, v): self._kv(k, STRING, _s(v))
    def boolean(self, k, v): self._kv(k, BOOL, struct.pack("<?", bool(v)))

    def str_array(self, k, items):
        self._kv(k, ARRAY, struct.pack("<IQ", STRING, len(items)) + b"".join(_s(x) for x in items))

    def i32_array(self, k, items):
        self._kv(k, ARRAY, struct.pack("<IQ", I32, len(items)) + np.asarray(items, "<i4").tobytes())

    def tensor(self, name: str, arr: np.ndarray, dtype: str = "f32"):
        arr = np.ascontiguousarray(arr)
        if dtype == "f16":
            data, t = arr.astype("<f2").tobytes(), GGML_F16
        else:
            data, t = arr.astype("<f4").tobytes(), GGML_F32
        self.tensors.append((name, list(reversed(arr.shape)), t, data))

    def write(self):
        a = self.alignment
        head = struct.pack("<IIQQ", MAGIC, 3, len(self.tensors), len(self.kv))
        head += b"".join(_s(k) + v for k, v in self.kv)
        off = 0
        for name, ne, t, data in self.tensors:
            head += _s(name) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", n) for n in ne)
            head += struct.pack("<IQ", t, off)
            off += -(-len(data) // a) * a
        head += b"\0" * (-len(head) % a)
        with open(self.path, "wb") as f:
            f.write(head)
            for _, _, _, data in self.tensors:
                f.write(data)
                f.write(b"\0" * (-len(data) % a))


def read(path: str):
    """Return (metadata dict, {name: np.ndarray}) for F32/F16 GGUF files."""
    buf = open(path, "rb").read()
    pos = 0

    def take(fmt):
        nonlocal pos
        v = struct.unpack_from(fmt, buf, pos)
        pos += struct.calcsize(fmt)
        return v[0]

    def rs():
        nonlocal pos
        n = take("<Q")
        s = buf[pos:pos + n].decode("utf-8")
        pos += n
        return s

    def rv(t):
        if t == STRING:
            return rs()
        if t == ARRAY:
            et, n = take("<I"), take("<Q")
            return [rv(et) for _ in range(n)]
        return take(_FMT[t])

    if take("<I") != MAGIC:
        raise ValueError("not a GGUF file")
    take("<I")
    nt, nkv = take("<Q"), take("<Q")
    kv = {}
    for _ in range(nkv):
        k = rs()
        kv[k] = rv(take("<I"))
    infos = []
    for _ in range(nt):
        name = rs()
        nd = take("<I")
        ne = [take("<Q") for _ in range(nd)]
        infos.append((name, ne, take("<I"), take("<Q")))
    al = kv.get("general.alignment", 32)
    base = -(-pos // al) * al
    tensors = {}
    for name, ne, t, off in infos:
        dt = {GGML_F32: "<f4", GGML_F16: "<f2"}[t]
        n = int(np.prod(ne))
        arr = np.frombuffer(buf, dtype=dt, count=n, offset=base + off)
        tensors[name] = arr.reshape(list(reversed(ne))).astype(np.float32)
    return kv, tensors
