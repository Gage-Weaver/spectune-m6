"""Packed reads for the pinned MLX QMV and DFlash replay kernels."""

import ast
from pathlib import Path
import re
import mlx.core as mx
from .prefill_nax import KERNELS
from .layout import CaptureMX

import dflash_mlx.verify_qmm as _upstream

UPSTREAM = Path(_upstream.__file__)
_CACHE = {}


def build_m1(layout):
    key = ("m1", layout)
    if key in _CACHE:
        return _CACHE[key]
    text = (KERNELS / "quantized.h").read_text()
    start = text.index(
        "template <typename T, typename U, int values_per_thread, int bits>"
    )
    load = text[start : text.index("\ntemplate <", start + 1)]
    start = text.index(
        "template <typename U, int values_per_thread, int bits>\ninline U qdot("
    )
    dot = text[start : text.index("\ntemplate <", start + 1)]
    start = text.index(
        "template <typename T, int group_size, int bits>\nMETAL_FUNC void qmv_fast_impl("
    )
    impl = text[start : text.index("\ntemplate <", start + 1)]
    if layout == "transpose":
        assert dot.count("    U sum)") == 1
        dot = dot.replace("    U sum)", "    U sum, int output_stride)")
        dot, count = re.subn(r"ws\[i\]", r"ws[(i/2)*output_stride*2+(i%2)]", dot)
        assert count == 4
        remove = [
            "  const device uint8_t* ws = (const device uint8_t*)w;",
            "  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;",
            "  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;",
            "  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;",
            "    ws += block_size * bytes_per_pack / pack_factor;",
            "    scales += block_size / group_size;",
            "    biases += block_size / group_size;",
        ]
        for statement in remove:
            assert impl.count(statement) == 1
            impl = impl.replace(statement, "")
        old = """      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;"""
        new = """
      int k_thread=k+int(simd_lid)*values_per_thread;
      auto wl=(const device uint8_t*)(w+(k_thread/pack_factor)*out_vec_size+out_row+row);
      const device T* sl=scales+(k_thread/group_size)*out_vec_size+out_row+row;
      const device T* bl=biases+(k_thread/group_size)*out_vec_size+out_row+row;"""
        assert impl.count(old) == 1
        impl = impl.replace(old, new)
        old = "qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum)"
        assert impl.count(old) == 1
        impl = impl.replace(
            old,
            "qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum, out_vec_size)",
        )
    elif layout != "canonical":
        raise ValueError(layout)
    header = (
        """// Copyright © 2023-2024 Apple Inc. MLX (MIT).
// SpecTune modifications: physical weight and metadata addresses only.
namespace spectune_replay {
using namespace metal;
static constant constexpr int SIMD_SIZE=32;
template<int bits,int wsize=8> inline constexpr short get_pack_factor(){return wsize/bits;}
template<int bits,int wsize=8> inline constexpr short get_bytes_per_pack(){return wsize/8;}
"""
        + load
        + dot
        + impl
        + "\n}\n"
    )
    source = """spectune_replay::qmv_fast_impl<T,64,4>(w,scales,biases,x,y,K,N,
        threadgroup_position_in_grid,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);"""
    kernel = mx.fast.metal_kernel(
        name="spectune_direct_m1_" + layout,
        input_names=["x", "w", "scales", "biases", "K", "N"],
        output_names=["y"],
        source=source,
        header=header,
    )
    _CACHE[key] = (kernel, header + source)
    return _CACHE[key]


def build_m4(dtype, layout, parts=2):
    key = ("m4", dtype, layout, parts)
    if key in _CACHE:
        return _CACHE[key]
    tree = ast.parse(UPSTREAM.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_build_kernel_m4_ksplit_np"
    )
    ns = {"mx": CaptureMX(), "_VERIFY_KERNEL_CACHE": {}}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(UPSTREAM), "exec"), ns)
    spec = ns[node.name](64, dtype, k_parts=parts)
    source = spec["source"]
    if layout == "transpose":
        source, nw = re.subn(
            r"w_q\[\(n0 \+ ([0-3])\) \* K_by_8 \+ pack\]", r"w_q[pack*N+n0+\1]", source
        )
        source, nm = re.subn(
            r"(scales|biases)\[\(n0 \+ ([0-3])\) \* K_by_gs \+ \(k_base / GS\)\]",
            r"\1[(k_base/GS)*N+n0+\2]",
            source,
        )
        assert nw == 4 and nm == 8, (nw, nm)
    elif layout != "canonical":
        raise ValueError(layout)
    spec.update(source=source, name=f"spectune_direct_m4_{layout}_{parts}")
    result = mx.fast.metal_kernel(**spec), spec.get("header", "") + source
    _CACHE[key] = result
    return result


def run_m1(x, w, s, b, layout, n):
    k = x.shape[-1]
    assert x.shape[0] == 1 and k % 512 == 0 and n % 8 == 0
    return build_m1(layout)[0](
        inputs=[x, w, s, b, k, n],
        template=[("T", x.dtype)],
        grid=(32, 2 * (n // 8), 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(1, n)],
        output_dtypes=[x.dtype],
    )[0]


def run_m4(x, w, s, b, layout, n):
    k = x.shape[-1]
    parts = 2 if n >= 4096 else 4
    return build_m4(x.dtype, layout, parts)[0](
        inputs=[x, w, s, b, k, n],
        template=[("T", x.dtype)],
        grid=(32 * parts, n // 4, 1),
        threadgroup=(32 * parts, 1, 1),
        output_shapes=[(4, n)],
        output_dtypes=[x.dtype],
    )[0]


def build_wide(layout):
    key = ("wide", layout)
    if key in _CACHE:
        return _CACHE[key]
    text = (KERNELS / "quantized.h").read_text()
    start = text.rfind("template <", 0, text.index("inline void dequantize("))
    dequant = text[start : text.index("\ntemplate <", start + 1)]
    start = text.index(
        "template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes>"
    )
    impl = text[start : text.index("\ntemplate <", start + 1)]
    if layout == "transpose":
        edits = {
            "(const device uint8_t*)w + row * in_vec_size_w": "(const device uint8_t*)w + row * 4",
            "scales + row * in_vec_size_g": "scales + row",
            "biases + row * in_vec_size_g": "biases + row",
            "srow[g]": "srow[g*out_vec_size]",
            "brow[g]": "brow[g*out_vec_size]",
            "wrow + k0 * bits / 8": "wrow + (k0/8)*out_vec_size*4",
        }
        for old, new in edits.items():
            assert impl.count(old) == 1
            impl = impl.replace(old, new)
    elif layout != "canonical":
        raise ValueError(layout)
    header = (
        """// Copyright © 2023-2024 Apple Inc. MLX (MIT).
// SpecTune modifications: physical weight/metadata addresses only.
namespace spectune_replay_wide {
using namespace metal;
static constant constexpr int SIMD_SIZE=32;
"""
        + dequant
        + impl
        + "\n}\n"
    )
    source = """spectune_replay_wide::qmv_wide_impl<T,64,4,V,8>(w,scales,biases,x,y,K,N,M,
      threadgroup_position_in_grid,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);"""
    kernel = mx.fast.metal_kernel(
        name="spectune_direct_wide_" + layout,
        input_names=["x", "w", "scales", "biases", "K", "N", "M"],
        output_names=["y"],
        source=source,
        header=header,
    )
    _CACHE[key] = (kernel, header + source)
    return _CACHE[key]


def run_wide(x, w, s, b, layout, n):
    m, k = x.shape
    assert 2 <= m <= 12 and m != 4
    tiles = (m + 4) // 5
    v = (m + tiles - 1) // tiles
    return build_wide(layout)[0](
        inputs=[x, w, s, b, k, n, m],
        template=[("T", x.dtype), ("V", v)],
        grid=(32 * ((m + v - 1) // v), 2 * ((n + 7) // 8), 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )[0]
