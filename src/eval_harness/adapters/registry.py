from __future__ import annotations

from eval_harness.adapters.base import ModelAdapter
from eval_harness.config import ModelConfig


def make_adapter(name: str, models: dict[str, ModelConfig]) -> ModelAdapter:
    """Resolve a models.yaml key (or 'noop') to an adapter. New providers register here."""
    if name == "noop":
        from eval_harness.adapters.fake import NoopAdapter

        return NoopAdapter()
    cfg = models.get(name)
    if cfg is None:
        raise KeyError(f"model {name!r} is not in config/models.yaml")
    if cfg.provider == "anthropic":
        from eval_harness.adapters.anthropic import AnthropicAdapter

        return AnthropicAdapter(cfg)
    if cfg.provider == "claude-code":
        from eval_harness.adapters.claude_code import ClaudeCodeAdapter

        return ClaudeCodeAdapter(cfg)
    if cfg.provider == "openai-compat":
        from eval_harness.adapters.openai_compat import OpenAICompatAdapter

        return OpenAICompatAdapter(cfg)
    if cfg.provider == "codex-cli":
        from eval_harness.adapters.codex_cli import CodexCliAdapter

        return CodexCliAdapter(cfg)
    raise KeyError(
        f"no adapter for provider {cfg.provider!r}; add one under src/eval_harness/adapters/"
    )
