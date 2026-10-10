# Derived from bstnxbt/dflash-mlx verify_qmm.py, Apache-2.0.
# Copyright 2026 bstnxbt; SpecTune modifications: scratch aliasing and tiled dequantization.
"""Q4 verification with 32-column tiles and 64-column expansion tiles.

Preserves upstream 16-term accumulation order and eight K splits.
"""

import mlx.core as mx

_VERIFY_KERNEL_CACHE = {}


HOISTED_LOOP = """
        for (int kg = k_begin; kg < k_end; kg += GS) {
            float s = float(scales[n_global * K_by_gs + kg / GS]);
            float b = float(biases[n_global * K_by_gs + kg / GS]);
            for (int k0 = kg; k0 < kg + GS; k0 += BK) {
                _Pragma("unroll")
                for (int pack = 0; pack < BK / 8; ++pack) {
                    uint32_t packed = w_q[n_global * K_by_8 + ((k0 + pack * 8) >> 3)];
                    _Pragma("unroll")
                    for (int ki = 0; ki < 8; ++ki) {
                        uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                        scratch.b[sg_id][(pack * 8 + ki) * BN + int(lane)] = T(float(nib) * s + b);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                _Pragma("unroll")
                for (int sub = 0; sub < BK; sub += 16) {
                    auto tA = A.template slice<16, 16>(k0 + sub, 0);
                    auto tB = B.template slice<32, 16>(0, sub);
                    op.run(tA, tB, ct_c);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

"""


def build_kernel(k_val: int, group_size: int, dtype: mx.Dtype, variant="tile32"):
    key = (variant, int(k_val), group_size, dtype)
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    bk = {"tile32": 32, "wide32": 32}[variant]
    if k_val % (8 * group_size) or group_size % bk:
        raise ValueError("Group-hoisted variant needs group-aligned K splits")
    bn = 64 if variant.startswith("wide") else 32
    loop = HOISTED_LOOP
    if bn == 64:
        loop = loop.replace(
            "float s = float(scales[n_global * K_by_gs + kg / GS]);",
            "float s[2]; float b[2]; for (int nc=0; nc<2; ++nc) { s[nc] = float(scales[(n_global+nc*32)*K_by_gs+kg/GS]); b[nc] = float(biases[(n_global+nc*32)*K_by_gs+kg/GS]); }",
        )
        loop = loop.replace(
            "float b = float(biases[n_global * K_by_gs + kg / GS]);", ""
        )
        loop = loop.replace(
            "for (int pack = 0;",
            "for (int nc=0; nc<2; ++nc) {\n                for (int pack = 0;",
        )
        loop = loop.replace("w_q[n_global * K_by_8", "w_q[(n_global+nc*32) * K_by_8")
        loop = loop.replace(
            "BN + int(lane)] = T(float(nib) * s + b);",
            "BN + int(lane)+nc*32] = T(float(nib) * s[nc] + b[nc]);",
        )
        loop = loop.replace(
            "                simdgroup_barrier",
            "                }\n                simdgroup_barrier",
            1,
        )
        loop = loop.replace("slice<32, 16>", "slice<64, 16>")

    source = f"""
        using namespace metal;
        using namespace mpp::tensor_ops;

        constexpr int BM = 16;
        constexpr int BN = {bn};
        constexpr int BK = {bk};
        constexpr int NSG = 8;
        constexpr int GS = {group_size};
        constexpr int K = KCONST;
        constexpr int K_by_8 = K / 8;
        constexpr int K_by_gs = K / GS;
        constexpr int K_chunk = K / NSG;

        uint tid = thread_position_in_threadgroup.x;
        uint sg_id = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;
        int N = int(N_size);
        int n0 = int(tg_n) * BN;
        int k_begin = int(sg_id) * K_chunk;
        int k_end = k_begin + K_chunk;

        union Scratch {{
            T b[NSG][BK * BN];
            float p[NSG][BM * BN];
        }};
        threadgroup Scratch scratch;

        constexpr auto desc = matmul2d_descriptor(
            16,
            {bn},
            16,
            false,
            false,
            false,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, metal::execution_simdgroup> op;

        tensor<device T, dextents<int, 2>, tensor_inline> A(
            (device T*)x,
            dextents<int, 2>{{K, BM}},
            array<int, 2>{{1, K}});
        tensor<threadgroup T, dextents<int, 2>, tensor_inline> B(
            scratch.b[sg_id],
            dextents<int, 2>{{BN, BK}},
            array<int, 2>{{1, BN}});
        tensor<threadgroup float, dextents<int, 2>, tensor_inline> C(
            scratch.p[sg_id],
            dextents<int, 2>{{BN, BM}},
            array<int, 2>{{1, BN}});

        auto ct_c = op.template get_destination_cooperative_tensor<
            tensor<device T, extents<int, 16, 16>, tensor_inline>,
            tensor<threadgroup T, extents<int, {bn}, 16>, tensor_inline>,
            float>();
        _Pragma("unroll")
        for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) {{
            ct_c[i] = 0.0f;
        }}

        int n_global = n0 + int(lane);
        {loop}

        // All groups must finish reading weight scratch before reusing it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto tC = C.template slice<{bn}, 16>(0, 0);
        ct_c.store(tC);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int off = int(tid); off < BM * BN; off += NSG * 32) {{
            float acc01 = scratch.p[0][off] + scratch.p[1][off];
            float acc23 = scratch.p[2][off] + scratch.p[3][off];
            float acc45 = scratch.p[4][off] + scratch.p[5][off];
            float acc67 = scratch.p[6][off] + scratch.p[7][off];
            float acc = (acc01 + acc23) + (acc45 + acc67);
            int row = off / BN;
            int col = off - row * BN;
            y[row * N + n0 + col] = T(acc);
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"spectune_nax_{variant}_k{int(k_val)}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "N_size"],
        output_names=["y"],
        header="""
            #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        """,
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel
