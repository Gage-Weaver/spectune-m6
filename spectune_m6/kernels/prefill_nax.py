"""Packed reads for the pinned MLX single-partition NAX prefill kernel."""

from pathlib import Path
import re
import mlx.core as mx

INCLUDE = Path(mx.__file__).parent / "include"
KERNELS = INCLUDE / "mlx/backend/metal/kernels"
_CACHE = {}
HEADER_PATHS = []


def inline_header(path, seen):
    path = path.resolve()
    if path in seen:
        return ""
    seen.add(path)
    HEADER_PATHS.append(path)
    source = path.read_text()

    def include(match):
        return inline_header(INCLUDE / match[1], seen)

    return re.sub(r'^#include "([^"]+)"\s*$', include, source, flags=re.M).replace(
        "#pragma once", ""
    )


TRANSPOSE_LOADER = r"""
template<typename T, short BROWS, short BCOLS, short dst_ld,
         short reduction_dim, short tgp_size, short group_size, short bits>
struct TransposedBlockLoader {
  static_assert(bits==4 && group_size==64 && BCOLS==64 && reduction_dim==1);
  static constant constexpr short pack_factor=2;
  static constant constexpr short n_reads=BCOLS/pack_factor*BROWS/tgp_size;
  const short bi, bj;
  const int N, row;
  int k0=0;
  threadgroup T* dst;
  const device uint32_t* src;
  const device T* scales;
  const device T* biases;
  TransposedBlockLoader(const device uint32_t* src_, const device T* scales_,
      const device T* biases_, int K, threadgroup T* dst_, ushort sg, ushort lane,
      int N_, int row0) thread
    : bi(n_reads*(sg*32+lane)/(BCOLS/pack_factor)),
      bj(n_reads*(sg*32+lane)%(BCOLS/pack_factor)), N(N_), row(row0+bi),
      dst(dst_+bi*dst_ld+bj*pack_factor),src(src_),scales(scales_),biases(biases_) {}
  void load_unsafe() const thread {
    T scale=scales[(k0/group_size)*N+row];
    T bias=biases[(k0/group_size)*N+row];
    for(int i=0;i<n_reads;i++) {
      int byte=k0/2+bj+i;
      const device uint8_t* address=(const device uint8_t*)(src+(byte/4)*N+row);
      // The original helper reads one byte for two Q4 values. Only its
      // address changes; dequantization expressions and rounding are intact.
      dequantize<T,pack_factor,bits>(address+(byte%4),scale,bias,dst+i*pack_factor);
    }
  }
  void load_safe(short2 dim) const thread { load_unsafe(); }
  void next() thread { k0+=BCOLS; }
};
"""


def generated_header(layout):
    dependency = inline_header(KERNELS / "steel/gemm/nax.h", set())
    source = (KERNELS / "quantized_nax.h").read_text()
    # Extract the original helpers/loader/implementation, retaining copyright.
    dequant = source[
        source.index("template <typename U, int N, int bits>") : source.index(
            "template <\n    typename T,\n    short BROWS,"
        )
    ]
    loader = source[
        source.index("template <\n    typename T,\n    short BROWS,") : source.index(
            "template <\n    typename T,\n    short BROWS,",
            source.index("struct QuantizedBlockLoader {") + 1,
        )
    ]
    start = source.index(
        "template <\n    typename T,\n    const int group_size,\n    const int bits,\n    const bool aligned_N,"
    )
    impl = source[
        start : source.index(
            "\ntemplate <",
            source.index("METAL_FUNC void qmm_t_nax_tgp_impl(", start) + 1,
        )
    ]
    if layout == "transpose":
        loader = TRANSPOSE_LOADER
        impl = impl.replace(
            "using loader_w_t = QuantizedBlockLoader<",
            "using loader_w_t = TransposedBlockLoader<",
        )
        for statement in (
            "  wl += y_col * K_w;",
            "  scales += y_col * K_g;",
            "  biases += y_col * K_g;",
        ):
            assert impl.count(statement) == 1
            impl = impl.replace(statement, "")
        old = "loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);"
        assert impl.count(old) == 1
        impl = impl.replace(
            old,
            "loader_w_t loader_w(w, scales, biases, K, Ws, simd_gid, simd_lid, N, y_col);",
        )
    elif layout != "canonical":
        raise ValueError(layout)
    prefix = """// Copyright © 2023-2025 Apple Inc. Derived from MLX 0.32.2 (MIT).
// SpecTune modifications: physical weight/metadata addresses only.
#define MLX_MTL_CONST static constant constexpr const
namespace spectune_prefill {
using namespace metal;
using namespace mlx::steel;
MLX_MTL_CONST int SIMD_SIZE=32;
template<int bits,int wsize=8> inline constexpr short get_pack_factor(){return wsize/bits;}
template<int bits,int wsize=8> inline constexpr short get_bytes_per_pack(){return wsize/8;}
"""
    return dependency + prefix + dequant + loader + impl + "\n}\n"


def build(layout):
    if layout in _CACHE:
        return _CACHE[layout]
    header = generated_header(layout)
    source = """
threadgroup T Ws[64*(64+16/sizeof(T))];
spectune_prefill::qmm_t_nax_tgp_impl<T,64,4,true,BM,64,64,2,2>(
    w,scales,biases,x,y,Ws,K,N,M,threadgroup_position_in_grid,
    thread_index_in_threadgroup,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
"""
    kernel = mx.fast.metal_kernel(
        name="spectune_direct_prefill_" + layout,
        input_names=["x", "w", "scales", "biases", "K", "N", "M"],
        output_names=["y"],
        source=source,
        header=header,
    )
    _CACHE[layout] = (kernel, header, source)
    return _CACHE[layout]


def run(x, w, s, b, layout, n):
    m, k = x.shape
    bm = 32 if m <= 32 else 64
    return build(layout)[0](
        inputs=[x, w, s, b, k, n, m],
        template=[("T", x.dtype), ("BM", bm)],
        grid=(32 * n // 64, 2 * ((m + bm - 1) // bm), 2),
        threadgroup=(32, 2, 2),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )[0]


def eligible(m, k, n):
    # Match stock qmm_splitk's partition decision before its NAX fallback.
    tgs = ((n + 31) // 32) * ((m + 31) // 32)
    splits = min(max(1, 512 // tgs), k // 64)
    while splits > 1 and k % (splits * 64):
        splits -= 1
    return m >= 33 and n % 64 == 0 and k % 64 == 0 and splits == 1
