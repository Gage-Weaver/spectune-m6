"""Load the fixed packed profile into the existing DFlash/MLX-LM server."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

from .profile import TARGET, DRAFT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--canonical", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not 1 <= args.max_tokens <= 32768:
        parser.error("--max-tokens must be between 1 and 32768")

    from .__main__ import hardware, conditions
    from dataclasses import asdict

    from .profile import BLOCK_SIZE, load
    from dflash_mlx.engine.config import resolve_speculative_cycle_config
    from dflash_mlx import serve as upstream
    from dflash_mlx.runtime.context import with_metal_limits
    from dflash_mlx.server.tool_calls import install_tool_parser

    system = hardware()

    class PackedProvider(upstream.DFlashModelProvider):
        def load(self, model_path, adapter_path=None, draft_model_path=None):
            if self.model_key is not None:
                return self.model, self.tokenizer
            started = time.perf_counter()
            bundle, context, state = load(canonical=args.canonical)
            context = with_metal_limits(
                context, self.cli_args.runtime_context.metal_limits
            )
            cycle = resolve_speculative_cycle_config(
                context.runtime, bundle.draft_model, block_tokens=None
            )
            if (
                cycle.effective_block_tokens != BLOCK_SIZE
                or cycle.verify_len_cap != BLOCK_SIZE
            ):
                raise RuntimeError("Server did not resolve the fixed16 profile.")
            self.cli_args.runtime_context = context
            self.cli_args.runtime_config = context.runtime
            install_tool_parser(bundle.tokenizer)
            self.model = bundle.target_model
            self.tokenizer = bundle.tokenizer
            self.draft_model = bundle.draft_model
            self.draft_backend = bundle.draft_backend
            self.target_ops = bundle.target_ops
            self.target_meta = bundle.target_meta
            self.draft_meta = bundle.draft_meta
            self.effective_draft_quant = bundle.effective_draft_quant
            self.packed_state = state
            # This provider intentionally serves one fixed model, never reloads
            # from request-controlled paths, and uses the upstream worker queue.
            self.model_key = (TARGET[0], None, DRAFT[0])
            root = Path(__file__).resolve().parent.parent
            logs = Path(os.environ.get("SPECTUNE_STATE_DIR", root)) / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            record = {
                "system": system,
                "conditions": conditions(),
                "python": platform.python_version(),
                "packages": {
                    d.metadata["Name"]: d.version
                    for d in importlib.metadata.distributions()
                    if d.metadata.get("Name")
                },
                "sources": {
                    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (root / "spectune_m6").rglob("*.py")
                },
                "lock_sha256": hashlib.sha256(
                    (root / "uv.lock").read_bytes()
                ).hexdigest(),
                "target": TARGET,
                "draft": DRAFT,
                "mode": "canonical" if args.canonical else "packed",
                "runtime_context": str(context),
                "cycle_config": asdict(cycle),
                "setup_s": time.perf_counter() - started,
                "packed_projections": len(state["projections"]) if state else 0,
                "startup_parity": state["startup_parity"] if state else [],
                "server": "Pinned upstream DFlash/MLX-LM; serial generation worker; loopback only.",
                "timing_scope": "Server compatibility has separate validation; the CLI benchmark is not an HTTP benchmark.",
            }
            name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ-server.json")
            (logs / name).write_text(json.dumps(record, indent=2, default=str) + "\n")
            return self.model, self.tokenizer

    class FixedProfileHandler(upstream.DFlashAPIHandler):
        def do_POST(self):
            self._completion_response_started = False
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8 * 1024 * 1024:
                    raise ValueError("Request body must be between 1 byte and 8 MiB.")
                return super().do_POST()
            except (ValueError, TypeError, AssertionError) as error:
                if self._completion_response_started:
                    self.close_connection = True
                else:
                    self._write_completion_error_response(400, str(error))

        def handle_models_request(self):
            self._write_json_response(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "spectune-m6",
                            "object": "model",
                            "created": 0,
                            "owned_by": "local",
                        }
                    ],
                },
            )

        def validate_model_parameters(self):
            super().validate_model_parameters()
            if self.requested_model not in ("default_model", "spectune-m6", TARGET[0]):
                raise ValueError("This server provides only model 'spectune-m6'.")
            if self.adapter is not None or self.requested_draft_model not in (
                "default_model",
                DRAFT[0],
            ):
                raise ValueError(
                    "Per-request adapters and draft models are not supported."
                )
            if not 1 <= self.max_tokens <= 32768:
                raise ValueError("max_tokens must be between 1 and 32768")

    upstream.DFlashModelProvider = PackedProvider
    upstream.DFlashAPIHandler = FixedProfileHandler
    upstream.main(
        [
            "--model",
            TARGET[0],
            "--draft-model",
            DRAFT[0],
            "--draft-quant",
            "w4",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--max-tokens",
            str(args.max_tokens),
            "--chat-template-args",
            '{"enable_thinking":false}',
            "--temp",
            "0",
            "--prompt-cache-size",
            "0",
            "--log-level",
            "WARNING",
        ]
    )


if __name__ == "__main__":
    main(sys.argv[1:])
