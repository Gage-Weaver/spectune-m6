"""Packed prefill with the stock MLX partition and reduction rules."""

import mlx.core as mx
from . import prefill_nax as old

KERNELS = old.KERNELS
HEADER_PATHS = old.HEADER_PATHS
_CACHE = {}


def partitions(m, k, n):
    count = min(max(1, 512 // (((n + 31) // 32) * ((m + 31) // 32))), k // 64)
    while count > 1 and k % (count * 64):
        count -= 1
    return count


def eligible(m, k, n):
    return m >= 13 and n % 64 == 0 and k % 64 == 0


LOADER = r"""
template<typename T, short BROWS, short BCOLS, short dst_ld,
         short reduction_dim, short tgp_size, short group_size, short bits>
struct TransposedSplitLoader {
  static_assert(bits==4 && group_size==64 && BCOLS==32 && reduction_dim==1);
  static constant constexpr short pack_factor=2;
  static constant constexpr short n_reads=BCOLS/pack_factor*BROWS/tgp_size;
  const short bi,bj;
  const int N,row;
  int k0;
  threadgroup T* dst;
  const device uint32_t* src;
  const device T* scales;
  const device T* biases;
  TransposedSplitLoader(const device uint32_t* src_,const device T* scales_,
      const device T* biases_,int K,threadgroup T* dst_,ushort sg,ushort lane,
      int N_,int row0,int kstart) thread
    :bi(n_reads*(sg*32+lane)/(BCOLS/pack_factor)),
     bj(n_reads*(sg*32+lane)%(BCOLS/pack_factor)),N(N_),row(row0+bi),k0(kstart),
     dst(dst_+bi*dst_ld+bj*pack_factor),src(src_),scales(scales_),biases(biases_) {}
  void load_unsafe() const thread {
    T scale=scales[(k0/group_size)*N+row];
    T bias=biases[(k0/group_size)*N+row];
    for(int i=0;i<n_reads;i++) {
      int byte=k0/2+bj+i;
      const device uint8_t* address=(const device uint8_t*)(src+(byte/4)*N+row);
      dequantize<T,pack_factor,bits>(address+(byte%4),scale,bias,dst+i*pack_factor);
    }
  }
  // All eligible output dimensions are aligned; the reference's safe weight
  // load is unreachable. Only activation/output rows require masking.
  void load_safe(short2 dim) const thread { load_unsafe(); }
  void next() thread { k0+=BCOLS; }
};
"""


def generated_header(layout):
    seen = set()
    dependencies = "".join(
        old.inline_header(KERNELS / f"steel/gemm/{p}", seen)
        for p in ("loader.h", "mma.h")
    )
    text = (KERNELS / "quantized.h").read_text()
    a = text.index("template <typename U, int N, int bits, typename W>")
    b = text.index("template <\n    typename T,\n    short BROWS,", a)
    dequant = text[a:b]
    end = text.index("\ntemplate <", text.index("struct QuantizedBlockLoader {", b) + 1)
    loader = text[b:end]
    start = text.rfind("template <", 0, text.index("METAL_FUNC void qmm_t_impl("))
    end = text.index(
        "\ntemplate <", text.index("METAL_FUNC void qmm_t_impl(", start) + 1
    )
    impl = text[start:end]
    if layout == "transpose":
        loader = LOADER
        impl = impl.replace(
            "using loader_w_t = QuantizedBlockLoader<",
            "using loader_w_t = TransposedSplitLoader<",
        )
        for expression in (
            "  wl += y_col * K_w;",
            "  scales += y_col * K_g;",
            "  biases += y_col * K_g;",
        ):
            assert impl.count(expression) == 1
            impl = impl.replace(expression, "")
        original = "loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);"
        assert impl.count(original) == 1
        impl = impl.replace(
            original,
            "loader_w_t loader_w(w, scales, biases, K, Ws, simd_gid, simd_lid, N, y_col, tid.z * K_eff);",
        )
    elif layout != "canonical":
        raise ValueError(layout)
    prefix = """// Copyright © 2023-2024 Apple Inc. Derived from MLX 0.32.2 (MIT).
// SpecTune changes only physical weight/metadata reads; accumulation is stock.
#define MLX_MTL_CONST static constant constexpr const
namespace spectune_split_prefill {
using namespace metal;
MLX_MTL_CONST int SIMD_SIZE=32;
template<int bits,int wsize=8> inline constexpr short get_pack_factor(){return wsize/bits;}
template<int bits,int wsize=8> inline constexpr short get_bytes_per_pack(){return wsize/8;}
"""
    return dependencies + prefix + dequant + loader + impl + "\n}\n"


def build(layout):
    if layout in _CACHE:
        return _CACHE[layout]
    header = generated_header(layout)
    source = """
threadgroup T Xs[32*(32+16/sizeof(T))];
threadgroup T Ws[32*(32+16/sizeof(T))];
const int k_start=threadgroup_position_in_grid.z*K_eff;
const device T* x_partition=x+k_start;
device T* y_partition=y+threadgroup_position_in_grid.z*M*N;
"""
    if layout == "canonical":
        source += """
const device uint32_t* w_partition=(const device uint32_t*)((const device uint8_t*)w+k_start/2);
const device T* s_partition=scales+k_start/64;
const device T* b_partition=biases+k_start/64;
"""
    else:
        source += "const device uint32_t* w_partition=w;\nconst device T* s_partition=scales;\nconst device T* b_partition=biases;\n"
    source += """
spectune_split_prefill::qmm_t_impl<T,64,4,true,32,32,32>(
    w_partition,s_partition,b_partition,x_partition,y_partition,Xs,Ws,K,N,M,K_eff,
    threadgroup_position_in_grid,thread_index_in_threadgroup,
    simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
"""
    kernel = mx.fast.metal_kernel(
        name="spectune_direct_split_prefill_" + layout,
        input_names=["x", "w", "scales", "biases", "K", "N", "M", "K_eff"],
        output_names=["y"],
        source=source,
        header=header,
    )
    _CACHE[layout] = (kernel, header, source)
    return _CACHE[layout]


def run(x, w, s, b, layout, n):
    m, k = x.shape
    count = partitions(m, k, n)
    if count == 1:
        return old.run(x, w, s, b, layout, n)
    intermediate = build(layout)[0](
        inputs=[x, w, s, b, k, n, m, k // count],
        template=[("T", x.dtype)],
        grid=(32 * (n // 32), 2 * ((m + 31) // 32), 2 * count),
        threadgroup=(32, 2, 2),
        output_shapes=[(count, m, n)],
        output_dtypes=[x.dtype],
    )[0]
    # Same dtype, shape and axis as stock's contiguous-strided reduction.
    return mx.sum(intermediate, axis=0)
