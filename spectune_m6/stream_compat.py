"""Let MLX-LM finish streams when DFlash cannot parse a generated tool span."""


def install():
    from dflash_mlx.server.tool_calls import ToolCallParseError
    from mlx_lm import server

    original = server.ToolCallFormatter
    if getattr(original, "_spectune_parse_compat", False):
        return

    class ToolCallFormatter(original):
        _spectune_parse_compat = True

        def __init__(self, tool_parser, tools, streaming=False):
            def parse(text, declared_tools):
                try:
                    return tool_parser(text, declared_tools)
                except ToolCallParseError as error:
                    # MLX-LM skips malformed tool spans on ValueError and keeps
                    # consuming generation, then emits actual usage and DONE.
                    # DFlash uses RuntimeError instead, which otherwise aborts
                    # the consumer before those final stream records are sent.
                    raise ValueError(str(error)) from error

            super().__init__(parse, tools, streaming)

    server.ToolCallFormatter = ToolCallFormatter


def normalize_finish(response, tools):
    """A discarded undeclared span is not an executable tool-call response."""
    if not tools:
        for choice in response.get("choices", []):
            if choice.get("finish_reason") == "tool_calls":
                choice["finish_reason"] = "stop"
    return response
