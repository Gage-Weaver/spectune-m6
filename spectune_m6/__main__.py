"""Terminal runner. Generation goes to stdout; setup and timing go to stderr."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from .profile import TARGET, DRAFT, BLOCK_SIZE


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def hardware():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            "An Apple M6 Mac is required; run from a native arm64 terminal."
        )
    chip = command("sysctl", "-n", "machdep.cpu.brand_string")
    memory = int(command("sysctl", "-n", "hw.memsize"))
    os_version = platform.mac_ver()[0]
    if not chip.startswith("Apple M6"):
        raise RuntimeError(f"This release supports M6 Macs. Detected: {chip}.")
    if memory < 32 * 2**30:
        raise RuntimeError("This model pair needs at least 32 GB of unified memory.")
    if tuple(map(int, os_version.split(".")[:2])) < (26, 2):
        raise RuntimeError("macOS 26.2 or newer is required for the NAX kernels.")
    if platform.python_version() != "3.11.16":
        raise RuntimeError("Use ./run to load the pinned Python 3.11.16 runtime.")
    import mlx.core as mx

    if importlib.metadata.version("mlx") != "0.32.2":
        raise RuntimeError("Use ./run to load the pinned MLX 0.32.2 runtime.")
    device = mx.device_info()
    if not device["architecture"].startswith("applegpu_g18"):
        raise RuntimeError(f"Unsupported GPU architecture: {device['architecture']}.")
    return {"chip": chip, "memory_bytes": memory, "macos": os_version, "gpu": device}


def conditions():
    result = {}
    for key, args in {
        "power": ("pmset", "-g", "batt"),
        "thermal": ("pmset", "-g", "therm"),
        "swap": ("sysctl", "-n", "vm.swapusage"),
    }.items():
        try:
            result[key] = command(*args)
        except (OSError, subprocess.CalledProcessError):
            result[key] = None
    return result


def generate(bundle, context, prompt, max_tokens, memory_limit):
    import mlx.core as mx
    from dflash_mlx.runtime import stream_dflash_generate
    from dflash_mlx.engine.events import TokenEvent, SummaryEvent

    detokenizer = bundle.tokenizer.detokenizer
    detokenizer.reset()
    stops = set(bundle.tokenizer.eos_token_ids)
    tokens, summary = [], {}
    first, finish = None, "output_limit"
    mx.clear_cache()
    mx.synchronize()
    mx.reset_peak_memory()
    started = time.perf_counter()
    events = stream_dflash_generate(
        target_model=bundle.target_model,
        target_ops=bundle.target_ops,
        tokenizer=bundle.tokenizer,
        draft_model=bundle.draft_model,
        draft_backend=bundle.draft_backend,
        prompt=prompt,
        max_new_tokens=max_tokens,
        use_chat_template=False,
        block_tokens=BLOCK_SIZE,
        stop_token_ids=sorted(stops),
        runtime_context=context,
        publish_generation_snapshot=False,
    )
    try:
        for event in events:
            if isinstance(event, TokenEvent):
                first = first or time.perf_counter()
                tokens.append(int(event.token_id))
                if event.token_id in stops:
                    finish = "eos"
                else:
                    detokenizer.add_token(event.token_id)
                    print(detokenizer.last_segment, end="", flush=True)
                if len(tokens) % 128 == 0 and mx.get_active_memory() > memory_limit:
                    finish = "memory_limit"
                    break
            elif isinstance(event, SummaryEvent):
                summary = event.to_payload()
    finally:
        events.close()
    detokenizer.finalize()
    print(detokenizer.last_segment, end="", flush=True)
    mx.synchronize()
    elapsed = time.perf_counter() - started
    ttft = first - started if first else None
    decode_s = elapsed - ttft if ttft is not None else None
    if summary.get("fallback_ar"):
        finish = "fallback_ar"
    return detokenizer.text, {
        "generated_tokens": len(tokens),
        "wall_s": elapsed,
        "ttft_s": ttft,
        "decode_tps": (len(tokens) - 1) / decode_s
        if decode_s and len(tokens) > 1
        else None,
        "finish_reason": finish,
        "peak_mlx_bytes": mx.get_peak_memory(),
        "token_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "acceptance_history": summary.get("acceptance_history"),
        "accepted_from_draft": summary.get("accepted_from_draft"),
        "acceptance_ratio": summary.get("acceptance_ratio"),
    }


def main():
    if sys.argv[1:2] == ["serve"]:
        from .serve import main as serve

        return serve(sys.argv[2:])
    parser = argparse.ArgumentParser(
        description="Qwen + DFlash2 with packed-Q4 kernels on M6.",
        epilog="Start the local API server: ./run serve (see serve --help).",
    )
    parser.add_argument(
        "prompt", nargs="?", help="Omit for terminal chat, or pipe a prompt on stdin."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Output limit per reply (default: 4096).",
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="Use canonical weights as a comparison control.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check hardware and runtime without loading models.",
    )
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 32768:
        parser.error("--max-tokens must be between 1 and 32768")
    system = hardware()
    if args.check:
        print(json.dumps(system, indent=2))
        return 0
    prompt = args.prompt
    if prompt is None and not sys.stdin.isatty():
        prompt = sys.stdin.read()
        if not prompt.strip():
            parser.error("stdin contains no prompt")
    if prompt is not None and not prompt.strip():
        parser.error("prompt is empty")
    print("Loading Qwen + DFlash2…", file=sys.stderr, flush=True)
    started = time.perf_counter()
    from .profile import load

    bundle, context, state = load(canonical=args.canonical)
    import mlx.core as mx

    mx.synchronize()
    setup_s = time.perf_counter() - started
    log_dir = (
        Path(
            os.environ.get("SPECTUNE_STATE_DIR", Path(__file__).resolve().parent.parent)
        )
        / "logs"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    log = log_dir / run_id
    package_root = Path(__file__).resolve().parent
    manifest = {
        "started_utc": run_id,
        "system": system,
        "python": platform.python_version(),
        "packages": {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata.get("Name")
        },
        "sources": {
            str(p.relative_to(package_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package_root.rglob("*.py"))
        },
        "target": TARGET,
        "draft": DRAFT,
        "draft_quant": "w4",
        "block_tokens": BLOCK_SIZE,
        "mode": "canonical" if args.canonical else "packed",
        "setup_s": setup_s,
        "max_new_tokens": args.max_tokens,
        "lock_sha256": hashlib.sha256(
            (package_root.parent / "uv.lock").read_bytes()
        ).hexdigest(),
        "runtime_context": str(context),
        "packed_projections": len(state["projections"]) if state else 0,
        "startup_parity": state["startup_parity"] if state else [],
        "conditions_at_load": conditions(),
        "timing_scope": "First request includes remaining cold/JIT work; terminal streaming included. Not a benchmark repetition.",
    }
    log.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    print(f"Ready ({setup_s:.1f}s). Logs: {log_dir}", file=sys.stderr)
    interactive = prompt is None
    messages = []
    request_index = 0
    if interactive:
        print("/clear resets the conversation; /quit exits.", file=sys.stderr)
    while True:
        if interactive:
            try:
                prompt = input("\n> ")
            except EOFError:
                break
            if prompt.strip() == "/quit":
                break
            if prompt.strip() == "/clear":
                messages.clear()
                continue
            if not prompt.strip():
                continue
        messages.append({"role": "user", "content": prompt})
        rendered = bundle.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        before = conditions()
        if state:
            state["counters"].clear()
            state["route_counters"].clear()
        text, result = generate(
            bundle, context, rendered, args.max_tokens, system["memory_bytes"] * 0.82
        )
        print()
        result.update(
            request=request_index,
            conditions_before=before,
            conditions_after=conditions(),
            prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
            routes=dict(state["route_counters"]) if state else {},
            first_request=request_index == 0,
        )
        with log.with_suffix(".jsonl").open("a") as file:
            file.write(json.dumps(result) + "\n")
        print(
            f"{result['generated_tokens']} tokens · {result['wall_s']:.2f}s · {result['finish_reason']}",
            file=sys.stderr,
        )
        messages.append({"role": "assistant", "content": text})
        request_index += 1
        if result["finish_reason"] == "output_limit":
            print(
                "Output limit reached. Increase --max-tokens for a longer reply.",
                file=sys.stderr,
            )
        elif result["finish_reason"] == "memory_limit":
            print(
                "Memory limit reached. Use a shorter prompt or /clear the conversation.",
                file=sys.stderr,
            )
        elif result["finish_reason"] == "fallback_ar":
            print(
                "The speculative path fell back; see the request log.", file=sys.stderr
            )
        if not interactive:
            return 0 if result["finish_reason"] == "eos" else 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        raise SystemExit(130)
    except (RuntimeError, OSError, ValueError) as error:
        print(f"spectune-m6: {error}", file=sys.stderr)
        raise SystemExit(1)
