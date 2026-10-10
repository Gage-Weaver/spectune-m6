"""Build packed Q4 verification kernels by replacing weight read addresses."""

from pathlib import Path
import re
import mlx.core as mx


class CaptureFast:
    @staticmethod
    def metal_kernel(**kwargs):
        return kwargs


class CaptureMX:
    fast = CaptureFast()

    def __getattr__(self, name):
        return getattr(mx, name)


_CACHE = {}


def build(k, g, dtype, variant, layout):
    key = (k, g, dtype, variant, layout)
    if key in _CACHE:
        return _CACHE[key]
    ns = {"__name__": "spectune_private_layout_source"}
    source_path = Path(__file__).with_name("verify.py")
    exec(compile(source_path.read_text(), str(source_path), "exec"), ns)
    ns["mx"] = CaptureMX()
    spec = ns["build_kernel"](k, g, dtype, variant)
    source = spec["source"]
    bn = 64 if variant.startswith("wide") else 32

    # Only packed-weight and metadata address expressions change.
    def weight(match):
        row, word = match.groups()
        if layout == "transpose":
            return f"w_q[({word}) * N + ({row})]"
        return f"w_q[(int(tg_n) * K_by_8 + ({word})) * BN + ({row}) - n0]"

    source, nw = re.subn(
        r"w_q\[(n_global(?:\+nc\*32)?|\(n_global\+nc\*32\)) \* K_by_8 \+ (\(\(k0 \+ (?:0|8|pack \* 8)\) >> 3\))\]",
        weight,
        source,
    )
    if nw != 1:
        raise ValueError(f"Unexpected source weight expressions: {nw}")

    def metadata(match):
        name, row, group = match.groups()
        if layout == "transpose":
            return f"{name}[({group}) * N + ({row})]"
        return f"{name}[(int(tg_n) * K_by_gs + ({group})) * BN + ({row}) - n0]"

    source, nm = re.subn(
        r"(scales|biases)\[(n_global(?:\+nc\*32)?|\(n_global\+nc\*32\))\s*\*\s*K_by_gs\s*\+\s*(kg\s*/\s*GS)\]",
        metadata,
        source,
    )
    if nm != 2:
        raise ValueError(f"Unexpected source metadata expressions: {nm}")
    spec.update(source=source, name=f"layout_{variant}_{layout}_k{k}_g{g}")
    kernel = mx.fast.metal_kernel(**spec)
    _CACHE[key] = (kernel, source, bn)
    return _CACHE[key]


def prepare(w, s, b, layout, bn):
    if layout == "transpose":
        return tuple(mx.contiguous(v.T) for v in (w, s, b))
    n = w.shape[0]
    return tuple(
        mx.contiguous(v.reshape(n // bn, bn, v.shape[1]).transpose(0, 2, 1))
        for v in (w, s, b)
    )
