from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import BaseModel, Field

# What an adapter writes into `request_settings` for a field models.yaml can hold but this
# provider has no way to send. `None` would read as "left at the default", which is the
# ambiguity that let a declared effort look like it had been honoured.
NOT_A_PARAMETER = "not a parameter of this provider"


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolResult(BaseModel):
    content: str
    is_error: bool = False


class ToolExecutor(Protocol):
    async def __call__(self, name: str, args: dict[str, Any]) -> ToolResult: ...


class Caps(BaseModel):
    """The budget one case gets. What is recorded in the run artifact, not what was typed."""

    max_turns: int = 40
    max_output_tokens_total: int = 120_000
    wall_clock_seconds: int = 1800
    tool_timeout_seconds: int = 300

    @classmethod
    def resolve(cls, declared: Mapping[str, Any] | None = None, **flags: int | None) -> Caps:
        """These defaults, then the model's own `caps:` block, then the flags actually passed.

        Every layer contributes only the fields it names, so `--max-turns 60` against a model
        that declares a longer wall clock keeps the longer wall clock. The discipline that
        makes that work is that an unpassed flag arrives as None: a typer option defaulting to
        40 is indistinguishable from someone typing 40, and config would lose every time.
        """
        fields: dict[str, Any] = {k: v for k, v in (declared or {}).items() if v is not None}
        fields.update({k: v for k, v in flags.items() if v is not None})
        return cls(**fields)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens


class AgentResult(BaseModel):
    turns: int
    tool_calls: int
    usage: Usage
    cost_usd: float
    stop_reason: str
    cap_hit: str | None
    final_text: str
    transcript: list[dict[str, Any]] = Field(default_factory=list)
    request_settings: dict[str, Any] = Field(default_factory=dict)


class ModelAdapter(Protocol):
    name: str

    async def run_agent(
        self,
        *,
        system: str,
        task: str,
        tools: list[ToolSpec],
        execute: ToolExecutor,
        caps: Caps,
        trace: Any,
    ) -> AgentResult: ...
