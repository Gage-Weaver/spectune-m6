"""The fixed Qwen/DFlash2 profile used in the complete-answer study."""

TARGET = ("mlx-community/Qwen3.8-27B-4bit", "10c35caafbb80f7dc6a7a432cdd11af10a6d4818")
DRAFT = ("z-lab/Qwen3.8-27B-DFlash2", "50307d4c4cde6860d4eee73e2547cd786fe8e8a4")
BLOCK_SIZE = 16


def enable_verifier():
    import dflash_mlx.verify_qmm as qmm
    import dflash_mlx.verify_linear as linear
    from dflash_mlx.runtime.chip_detect import detect_chip
    from .kernels.verify import build_kernel

    if not detect_chip().nax_capable:
        raise RuntimeError("This profile needs an M6 GPU with NAX support.")
    qmm._nax_verify_available = lambda: True
    original = qmm._build_kernel_m16_nax_ktmpl

    def wider(k, group_size, dtype):
        base = original(k, group_size, dtype)
        if k % (8 * group_size) or group_size % 64:
            return base
        wide = build_kernel(k, group_size, dtype, "wide32")

        def dispatch(**kwargs):
            n = kwargs["inputs"][-1]
            if n > k and n % 64 == 0:
                kwargs["grid"] = (256, n // 64, 1)
                return wide(**kwargs)
            return base(**kwargs)

        return dispatch

    qmm._build_kernel_m16_nax_ktmpl = linear._build_kernel_m16_nax_ktmpl = wider
    return linear


def load(canonical=False):
    from dataclasses import replace
    import json
    from pathlib import Path
    import sys
    from huggingface_hub.errors import LocalEntryNotFoundError
    from huggingface_hub import snapshot_download
    from dflash_mlx.runtime.bundle import load_runtime_bundle
    from dflash_mlx.runtime.context import build_offline_runtime_context
    from .kernels.dispatch import install

    linear = enable_verifier()
    context = build_offline_runtime_context(
        quantize_kv_cache=False,
        verify_mode="dflash",
        copyspec_mode="off",
        draft_window_size=65536,
        target_fa_window=0,
        prefill_step_size=2048,
        draft_full_context_min_ctx=1000000000,
    )

    def snapshot(ref):
        try:
            path = Path(
                snapshot_download(ref[0], revision=ref[1], local_files_only=True)
            )
            index = path / "model.safetensors.index.json"
            if index.exists():
                shards = set(json.loads(index.read_text())["weight_map"].values())
            else:
                shards = {"model.safetensors"}
            required = {"config.json", *shards}
            if ref == TARGET:
                required |= {"tokenizer.json", "tokenizer_config.json"}
            if shards and all((path / name).is_file() for name in required):
                return str(path)
        except LocalEntryNotFoundError:
            pass
        print(f"Downloading {ref[0]}…", file=sys.stderr, flush=True)
        return snapshot_download(
            ref[0],
            revision=ref[1],
            allow_patterns=[
                "*.safetensors",
                "*.json",
                "*.model",
                "*.tiktoken",
                "*.txt",
            ],
        )

    bundle = load_runtime_bundle(
        model_ref=snapshot(TARGET),
        draft_ref=snapshot(DRAFT),
        draft_quant="w4",
        verify_config=context.verify,
        lazy=False,
    )
    bundle.draft_model.block_size = BLOCK_SIZE
    bundle.draft_model.capabilities = replace(
        bundle.draft_model.capabilities,
        max_block_tokens=BLOCK_SIZE,
        default_block_tokens=BLOCK_SIZE,
    )
    state = (
        None
        if canonical
        else install(
            bundle.target_model,
            linear,
            direct_fallback=True,
            complete_prefill=True,
        )
    )
    if state is not None and len(state["projections"]) != 368:
        raise RuntimeError(
            "The pinned target did not expose all 368 expected projections."
        )
    return bundle, context, state
