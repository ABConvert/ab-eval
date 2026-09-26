from __future__ import annotations

import os
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from eval_harness.paths import config_file

SOURCE_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".py", ".dart"}


def _glob_match(path: str, pattern: str) -> bool:
    """fnmatch where '**' also matches zero directories and '**/' patterns anchor at any depth."""
    p = PurePosixPath(path)
    if fnmatch(path, pattern):
        return True
    if pattern.startswith("**/"):
        tail = pattern[3:]
        return any(fnmatch(str(PurePosixPath(*p.parts[i:])), tail) for i in range(len(p.parts)))
    if "/**/" in pattern:
        head, tail = pattern.split("/**/", 1)
        return fnmatch(path, f"{head}/{tail}") or fnmatch(path, f"{head}/*/{tail}")
    return False


class DepDir(BaseModel):
    dir: str
    install: str
    lock: list[str]
    env: dict[str, str] = Field(default_factory=dict)
    post_install: str | None = None


class Runner(BaseModel):
    kind: Literal["vitest", "pytest", "flutter"]
    cwd: str
    config: str | None = None
    match: list[str]
    exclude: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    full: str
    lint: str | None = None

    def accepts(self, path: str) -> bool:
        return any(_glob_match(path, m) for m in self.match) and not any(
            _glob_match(path, e) for e in self.exclude
        )


class Limits(BaseModel):
    cpus: float = 2
    memory: str = "6g"
    pids: int = 2048


class RepoConfig(BaseModel):
    key: str
    github: str
    local_path: Path
    linear_team: str = ""  # issue-key prefix; empty means tickets come from GitHub Issues
    # login -> "agent" | "unclear". Whether a PR was written by a person matters when the
    # benchmark compares models against what humans shipped.
    agent_authors: dict[str, str] = Field(default_factory=dict)
    image: str
    dockerfile: str
    dep_dirs: list[DepDir]
    runners: dict[str, Runner]
    test_file_globs: list[str]
    non_code_globs: list[str]
    limits: Limits = Field(default_factory=Limits)

    def runner_for(self, paths: list[str]) -> tuple[str, Runner]:
        for name, runner in self.runners.items():
            if all(runner.accepts(p) for p in paths):
                return name, runner
        raise ValueError(f"no single runner in {self.key} covers all of: {paths}")

    def runner_groups(self, paths: list[str]) -> list[tuple[str, Runner, list[str]]]:
        """Assign each test path to the first runner that accepts it, preserving runner order."""
        groups: dict[str, list[str]] = {}
        for path in paths:
            for name, runner in self.runners.items():
                if runner.accepts(path):
                    groups.setdefault(name, []).append(path)
                    break
            else:
                raise ValueError(f"no runner in {self.key} accepts {path}")
        return [
            (name, runner, groups[name]) for name, runner in self.runners.items() if name in groups
        ]

    def is_test_file(self, path: str) -> bool:
        return any(_glob_match(path, g) for g in self.test_file_globs)

    def is_code_file(self, path: str) -> bool:
        if self.is_test_file(path) or any(_glob_match(path, g) for g in self.non_code_globs):
            return False
        return PurePosixPath(path).suffix in SOURCE_EXTENSIONS


class Prices(BaseModel):
    input: float
    output: float
    cache_read: float
    cache_write: float


class CapsOverride(BaseModel):
    """A model's own run budget, as declared in models.yaml. Every field optional.

    A cap is a property of the model as much as of the task: local weights at ~17 tokens a
    second need a longer clock than a hosted API to reach the same answer, and one global
    wall clock set for the fastest model silently marks the slowest one wrong. What is absent
    here falls back to the harness default in `Caps`, so a file written before this block
    existed keeps exactly the budget it had.

    `extra="forbid"` because the whole point of this block is that a declared value reaches
    the run: `wallclock_seconds` accepted and dropped would be the same defect in a new place.
    """

    model_config = ConfigDict(extra="forbid")

    max_turns: int | None = None
    max_output_tokens_total: int | None = None
    wall_clock_seconds: int | None = None
    tool_timeout_seconds: int | None = None


# How an OpenAI-compatible endpoint expects to be told how hard to think. These servers are
# not uniform: hosted APIs and most gateways take `reasoning_effort`, some take a `reasoning`
# object, and a plain llama.cpp or ollama shim takes neither and answers 400 to an unknown
# field. There is no probe that tells them apart, so the model says which, and the default
# says nothing at all — which is what every endpoint accepted before this field existed.
ReasoningParam = Literal["omit", "reasoning_effort", "reasoning.effort"]
REASONING_PARAMS: tuple[str, ...] = ("omit", "reasoning_effort", "reasoning.effort")

# Fields only the openai-compat adapter can act on. Declared anywhere else they would be
# read, written, snapshotted into the run artifact, and never sent — which is the whole
# class of bug this file is being changed to close, so they are refused instead.
COMPAT_ONLY = ("temperature", "reasoning_param")


class ModelConfig(BaseModel):
    key: str
    provider: str
    model: str
    effort: str = "high"
    max_output_tokens: int = 16000
    # Unset means "whatever the endpoint does by default", which is what every run before
    # this field existed did. There is no sensible default to invent: a benchmark that
    # quietly picked 0.0 or 1.0 for you would be measuring that choice.
    temperature: float | None = None
    reasoning_param: ReasoningParam = "omit"
    price_per_mtok: Prices | None = None
    base_url: str | None = None  # openai-compat providers only
    api_key_env: str | None = None  # name of the env var holding the key, never the key
    caps: CapsOverride | None = None

    @model_validator(mode="after")
    def _refuse_inert_fields(self) -> ModelConfig:
        """Refuse a field this provider's adapter cannot send.

        Silently ignoring a declared value is how two bench-v2 runs came to be configured
        `effort: high` and run at their server's default. An error at load is the cheap
        version of that discovery.
        """
        if self.provider == "openai-compat":
            return self
        inert = [
            name
            for name in COMPAT_ONLY
            if getattr(self, name) != type(self).model_fields[name].default
        ]
        if inert:
            raise ValueError(
                f"{', '.join(inert)} only reaches an openai-compat endpoint; "
                f"provider {self.provider!r} has no way to send it, so declaring it here "
                "would record a setting the run never used. Remove it."
            )
        return self


# Linear's key, for repositories whose `linear_team` points their tickets there. The name
# lives here so the check on /setup and the collector that needs it cannot drift apart; the
# value is read from the environment at the moment of the call and never stored anywhere.
LINEAR_API_KEY_ENV = "LINEAR_API_KEY"


def local_path_env_var(repo_key: str) -> str:
    """Env var that overrides a repo's clone path, e.g. my-app -> ABEVAL_PATH_MY_APP."""
    safe = "".join(c if c.isalnum() else "_" for c in repo_key).upper()
    return f"ABEVAL_PATH_{safe}"


def load_repos(path: Path | None = None) -> dict[str, RepoConfig]:
    path = path or config_file("repos.yaml")
    raw = yaml.safe_load(path.read_text())
    out: dict[str, RepoConfig] = {}
    for key, body in raw.items():
        body = dict(body)
        body["key"] = key
        # The same config runs on a laptop and on the eval box, where the clone lives
        # elsewhere; an env var keeps one committed config working on both.
        override = os.environ.get(local_path_env_var(key))
        body["local_path"] = Path(override or body["local_path"]).expanduser()
        out[key] = RepoConfig.model_validate(body)
    return out


def load_models(path: Path | None = None) -> dict[str, ModelConfig]:
    path = path or config_file("models.yaml")
    raw = yaml.safe_load(path.read_text())
    return {key: ModelConfig.model_validate({"key": key, **body}) for key, body in raw.items()}
