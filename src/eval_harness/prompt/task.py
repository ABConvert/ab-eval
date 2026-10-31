"""`task.yaml` and `rubric.yaml`: what a prompt task is graded on, as data rather than code.

The judge prompt is assembled from the rubric, so changing what a judge is asked changes a
file a reviewer can read and a hash the cache keys on — not a string buried in Python.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from eval_harness.paths import prompt_cases_dir

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
SCALE_RE = re.compile(r"^(-?\d+)-(-?\d+)$")

Uses = Literal["output", "grading_context", "prompt"]


class Dimension(BaseModel):
    """One thing a judge scores. `scale` is a range like "1-5" or a list of allowed labels."""

    model_config = ConfigDict(extra="forbid")

    name: str
    scale: str | list[str]
    instruction: str
    # What the judge is shown for this dimension. The output always; the grading context
    # (evidence, gold labels) when the dimension is about fidelity to it; the prompt only when
    # a dimension asks whether the instructions were followed.
    uses: list[Uses] = Field(default_factory=lambda: list[Uses](["output"]))
    # Weight in the leaderboard's quality composite. Enum dimensions count only when they say
    # which labels are good, because "interval_run" is not better than "easy_run" by itself.
    weight: float = 1.0
    good: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not NAME_RE.match(v):
            raise ValueError(f"dimension names are lowercase identifiers; got {v!r}")
        return v

    @model_validator(mode="after")
    def _scale(self) -> Dimension:
        if isinstance(self.scale, str):
            m = SCALE_RE.match(self.scale.strip())
            if not m or int(m.group(1)) >= int(m.group(2)):
                raise ValueError(f"{self.name}: scale is 'LO-HI' (e.g. '1-5') or a list of labels")
            if self.good:
                raise ValueError(f"{self.name}: `good` applies to enum dimensions only")
        else:
            if len(self.scale) < 2 or len(set(self.scale)) != len(self.scale):
                raise ValueError(f"{self.name}: an enum scale needs two or more distinct labels")
            unknown = [g for g in self.good if g not in self.scale]
            if unknown:
                raise ValueError(f"{self.name}: good labels {unknown} are not in the scale")
        if "output" not in self.uses:
            raise ValueError(f"{self.name}: a dimension that never sees the output grades nothing")
        return self

    @property
    def is_enum(self) -> bool:
        return isinstance(self.scale, list)

    @property
    def bounds(self) -> tuple[int, int]:
        assert isinstance(self.scale, str)
        m = SCALE_RE.match(self.scale.strip())
        assert m is not None
        return int(m.group(1)), int(m.group(2))

    def coerce(self, raw: object) -> int | str:
        """The judge's answer as a value on this scale, or ValueError. Never a guess."""
        if self.is_enum:
            value = str(raw).strip()
            if value not in self.scale:
                raise ValueError(f"{self.name}: {raw!r} is not one of {self.scale}")
            return value
        if isinstance(raw, bool):
            raise ValueError(f"{self.name}: {raw!r} is not a number")
        try:
            number = float(str(raw).strip())
        except ValueError:
            raise ValueError(f"{self.name}: {raw!r} is not a number") from None
        lo, hi = self.bounds
        if number != int(number) or not lo <= number <= hi:
            raise ValueError(f"{self.name}: {raw!r} is not an integer in {lo}-{hi}")
        return int(number)

    def normalised(self, value: int | str) -> float | None:
        """0..1 for the quality composite; None for an enum without `good` labels."""
        if self.is_enum:
            return (1.0 if value in self.good else 0.0) if self.good else None
        lo, hi = self.bounds
        return (float(value) - lo) / (hi - lo)


class Rubric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = ""
    instructions: str = ""
    dimensions: list[Dimension]

    @model_validator(mode="after")
    def _unique(self) -> Rubric:
        names = [d.name for d in self.dimensions]
        if not names:
            raise ValueError("a rubric needs at least one dimension")
        if len(set(names)) != len(names):
            raise ValueError(f"dimension names repeat: {names}")
        return self

    def hash(self) -> str:
        """What the judge cache keys on: any change to what is asked re-asks it."""
        body = json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    def dimension(self, name: str) -> Dimension:
        return next(d for d in self.dimensions if d.name == name)


class ScorerConfig(BaseModel):
    """A product-owned command: stdin {case, output_text}, stdout {metrics, flags}."""

    model_config = ConfigDict(extra="forbid")

    command: list[str]
    timeout_seconds: int = 60

    @field_validator("command")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("scorer.command is an argv list with at least the program")
        return v


class TaskDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # None leaves it to the case, then to the model's own max_output_tokens.
    max_output_tokens: int | None = None
    repeats: int = 1


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    description: str = ""
    rubric: str = "rubric.yaml"
    scorer: ScorerConfig | None = None
    judges: list[str] = Field(default_factory=list)  # keys in models.yaml
    defaults: TaskDefaults = Field(default_factory=TaskDefaults)
    # Leaderboard composite over scorer metrics (0..1 or bool), by weight. Rubric dimensions
    # carry their own weight; a metric not named here is reported but not ranked on.
    quality_metrics: dict[str, float] = Field(default_factory=dict)

    _dir: Path | None = PrivateAttr(default=None)

    @field_validator("task")
    @classmethod
    def _task_name(cls, v: str) -> str:
        if not NAME_RE.match(v):
            raise ValueError(f"task names are lowercase letters, digits, - or _; got {v!r}")
        return v

    @field_validator("rubric")
    @classmethod
    def _rubric_beside(cls, v: str) -> str:
        # Imported by copying it next to task.yaml, so a path elsewhere would not travel.
        if Path(v).name != v or v.startswith("."):
            raise ValueError(f"rubric is a file name beside task.yaml; got {v!r}")
        return v

    @property
    def directory(self) -> Path:
        return self._dir or prompt_cases_dir(self.task)

    def load_rubric(self) -> Rubric:
        return load_rubric(self.directory / self.rubric)


def load_rubric(path: Path) -> Rubric:
    return Rubric.model_validate(yaml.safe_load(path.read_text()) or {})


def load_task_file(path: Path) -> TaskConfig:
    cfg = TaskConfig.model_validate(yaml.safe_load(path.read_text()) or {})
    cfg._dir = path.parent
    return cfg


def load_task(task: str) -> TaskConfig:
    path = prompt_cases_dir(task) / "task.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"no task {task!r}: {path} is missing; import one with "
            f"eval-harness import prompt-cases --task {task} --from DIR"
        )
    cfg = load_task_file(path)
    if cfg.task != task:
        raise ValueError(f"{path} says task {cfg.task!r}, but it lives under {task!r}")
    return cfg


def task_names() -> list[str]:
    root = prompt_cases_dir()
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "task.yaml").exists())
