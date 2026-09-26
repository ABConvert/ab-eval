from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from eval_harness.adapters.base import Usage


class GenerationHandle:
    def __init__(self, obs: Any | None) -> None:
        self._obs = obs

    def end(self, *, input: Any, output: Any, usage: Usage, cost: float) -> None:
        if self._obs is None:
            return
        self._obs.update(
            input=input,
            output=output,
            usage_details={
                "input": usage.input_tokens,
                "output": usage.output_tokens,
                "cache_read_input_tokens": usage.cache_read_tokens,
                "cache_creation_input_tokens": usage.cache_write_tokens,
            },
            cost_details={"total": cost},
        )
        self._obs.end()


class Tracer:
    """Langfuse tracer; a no-op when LANGFUSE_PUBLIC_KEY/SECRET_KEY are absent."""

    def __init__(self, client: Any | None, root: Any | None, metadata: dict[str, Any]) -> None:
        self._client = client
        self._root = root
        self.metadata = metadata

    @classmethod
    def noop(cls) -> Tracer:
        return cls(None, None, {})

    @classmethod
    def for_attempt(cls, *, case_id: str, run_id: str, model: str) -> Tracer:
        meta = {"case_id": case_id, "run_id": run_id, "model": model}
        if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
            return cls(None, None, meta)
        from langfuse import Langfuse

        client = Langfuse()
        root = client.start_observation(name=f"{run_id}/{case_id}", as_type="agent", metadata=meta)
        return cls(client, root, meta)

    @contextmanager
    def generation(
        self, name: str, *, model: str, model_parameters: dict[str, Any]
    ) -> Iterator[GenerationHandle]:
        if self._root is None:
            yield GenerationHandle(None)
            return
        obs = self._root.start_observation(
            name=name,
            as_type="generation",
            model=model,
            model_parameters=model_parameters,
            metadata=self.metadata,
        )
        yield GenerationHandle(obs)

    @contextmanager
    def tool_span(self, name: str, args: dict[str, Any]) -> Iterator[None]:
        if self._root is None:
            yield None
            return
        obs = self._root.start_observation(
            name=name, as_type="tool", input=args, metadata=self.metadata
        )
        try:
            yield None
        finally:
            obs.end()

    def finish(self, output: dict[str, Any]) -> None:
        if self._root is not None:
            self._root.update(output=output)
            self._root.end()
        if self._client is not None:
            self._client.flush()
