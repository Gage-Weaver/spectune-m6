"""Single resident Q4 layout experiment; no second canonical target copy.

This is an experimental adapter, not a checkpoint serialization format. The
module's canonical-shaped properties are inverse views of the packed buffers.
M16, replay and prefill read packed buffers directly. Unsupported shapes use
the canonical fallback.
"""

import math
import time
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from .layout import build, prepare


def install(model, linear, *, direct_fallback=False, complete_prefill=False):
    if complete_prefill and not direct_fallback:
        raise ValueError("Complete prefill requires direct fallback mode")
    if direct_fallback:
        import importlib.metadata

        if importlib.metadata.version("mlx") != "0.32.2" or not mx.device_info()[
            "architecture"
        ].startswith("applegpu_g18"):
            raise ValueError(
                "Direct fallback is currently gated to MLX 0.32.2 / M6 architecture"
            )
    records = []
    counters = {}
    routes = {} if complete_prefill else None
    parity = []
    checked = set()
    started = time.perf_counter()
    for name, layer in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if not isinstance(layer, linear.VerifyQuantizedLinear):
            continue
        n, k = layer.weight.shape[0], layer.weight.shape[1] * 8
        # Frozen from operator probes: large aligned affine-Q4 projections only.
        if (
            layer.bits != 4
            or layer.group_size != 64
            or layer.mode != "affine"
            or k < 5120
            or n < 4096
            or k % 512
            or n % 64
            or linear._resolve_m16_ktmpl_variant(k, n, 4) != "nax_ktmpl"
        ):
            continue
        before = time.perf_counter()
        payload = sum(v.nbytes for v in (layer.weight, layer.scales, layer.biases))
        packed = prepare(layer.weight, layer.scales, layer.biases, "transpose", 64)
        mx.eval(packed)
        candidate = make_call(
            layer,
            packed,
            linear,
            counters,
            direct_fallback=direct_fallback,
            complete_prefill=complete_prefill,
            route_counters=routes,
        )
        if direct_fallback and (n, k) not in checked:
            # Exercise the new paths on actual checkpoint tensors without
            # changing MLX's random state. Keep only this layer's canonical
            # reference while testing, then release its old closure below.
            tests = []
            for m in (
                (1, 3, 4, 8, 13, 15, 16, 23, 27, 32, 111)
                if complete_prefill
                else (1, 3, 4, 8, 111)
            ):
                x = (
                    mx.sin(mx.arange(m * k, dtype=mx.float32) * 0.137)
                    .reshape(m, k)
                    .astype(mx.bfloat16)
                )
                ref = layer(x)
                value = candidate(x)
                mx.eval(ref, value)
                equal = (
                    bool(
                        mx.array_equal(
                            ref.view(mx.uint16), value.view(mx.uint16)
                        ).item()
                    )
                    if complete_prefill
                    else bool(mx.array_equal(ref, value).item())
                )
                tests.append(dict(rows=m, bitwise_equal=equal))
                if not equal:
                    raise ValueError(f"Direct startup parity failed: {name}, width {m}")
            del ref, value, x
            parity.append(dict(layer=name, n=n, k=k, tests=tests))
            checked.add((n, k))
        # Replace the old closure before assigning views: it captures canonical
        # contiguous arrays and would otherwise retain a second whole target.
        object.__setattr__(layer, "_call_fn", None)
        layer.weight, layer.scales, layer.biases = (v.T for v in packed)
        object.__setattr__(layer, "_call_fn", candidate)
        del packed
        mx.clear_cache()
        records.append(
            dict(
                name=name,
                n=n,
                k=k,
                payload_bytes=payload,
                preparation_s=time.perf_counter() - before,
            )
        )
    mx.synchronize()
    return dict(
        projections=records,
        prepared_bytes=sum(r["payload_bytes"] for r in records),
        preparation_s=time.perf_counter() - started,
        counters=counters,
        direct_fallback=direct_fallback,
        complete_prefill=complete_prefill,
        route_counters=routes,
        startup_parity=parity,
    )


def make_call(
    layer,
    packed,
    linear,
    counters,
    *,
    direct_fallback=False,
    complete_prefill=False,
    route_counters=None,
):
    if direct_fallback:
        if complete_prefill:
            from . import prefill
        else:
            from . import prefill_nax as prefill
        from . import replay
    pw, ps, pb = packed
    n, k = layer.weight.shape[0], layer.weight.shape[1] * 8
    gs = layer.group_size
    bias = layer.bias if "bias" in layer else None
    variant = "wide32" if n > k else "tile32"
    kernels = {
        dtype: build(k, gs, dtype, variant, "transpose")[0]
        for dtype in (mx.bfloat16, mx.float16)
    }
    bn = 64 if variant == "wide32" else 32
    use_m4 = linear._m4_ksplit_np_shape(k, n, 4)
    parts = linear._m4_ksplit_np_kparts(n)
    m4 = (
        {
            dtype: linear._build_kernel_m4_ksplit_np(gs, dtype, k_parts=parts)
            for dtype in (mx.bfloat16, mx.float16)
        }
        if use_m4
        else {}
    )

    def call(x):
        m = math.prod(x.shape[:-1])
        key = str(m)
        counters[key] = counters.get(key, 0) + 1
        if m == 16 and x.dtype in kernels:
            route = "verify16"
            x2 = mx.contiguous(x.reshape(16, k))
            y = kernels[x.dtype](
                inputs=[x2, pw, ps, pb, n],
                template=[("T", x.dtype), ("KCONST", k)],
                grid=(256, n // bn, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(16, n)],
                output_dtypes=[x.dtype],
            )[0].reshape(*x.shape[:-1], n)
        elif (
            direct_fallback
            and x.dtype == mx.bfloat16
            and (1 <= m <= 12 or prefill.eligible(m, k, n))
        ):
            route = (
                "direct-replay"
                if m <= 12
                else (
                    "direct-prefill-split"
                    if complete_prefill and prefill.partitions(m, k, n) > 1
                    else "direct-prefill-single"
                )
            )
            x2 = mx.contiguous(x.reshape(m, k))
            if m == 1:
                y = replay.run_m1(x2, pw, ps, pb, "transpose", n)
            elif m == 4:
                y = replay.run_m4(x2, pw, ps, pb, "transpose", n)
            elif 2 <= m <= 12:
                y = replay.run_wide(x2, pw, ps, pb, "transpose", n)
            else:
                y = prefill.run(x2, pw, ps, pb, "transpose", n)
            y = y.reshape(*x.shape[:-1], n)
        else:
            route = "inverse-reconstruction"
            # These temporaries belong only to this operation's graph. No
            # persistent canonical arrays or old dispatch closures are kept.
            w, s, b = (mx.contiguous(v.T) for v in (pw, ps, pb))
            if m == 4 and x.dtype in m4:
                x2 = mx.contiguous(x.reshape(4, k))
                y = m4[x.dtype](
                    inputs=[x2, w, s, b, k, n],
                    template=[("T", x.dtype)],
                    grid=(32 * parts, n // 4, 1),
                    threadgroup=(32 * parts, 1, 1),
                    output_shapes=[(4, n)],
                    output_dtypes=[x.dtype],
                )[0].reshape(*x.shape[:-1], n)
            else:
                y = mx.quantized_matmul(
                    x,
                    w,
                    scales=s,
                    biases=b,
                    transpose=True,
                    group_size=gs,
                    bits=4,
                    mode="affine",
                )
        if route_counters is not None:
            route_counters[route] = route_counters.get(route, 0) + 1
        return y + bias if bias is not None else y

    return call
