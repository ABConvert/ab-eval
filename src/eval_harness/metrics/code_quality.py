"""G-Eval code-quality judge with explicit evaluation steps (LLM-as-judge, gated on tests)."""

from __future__ import annotations

from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams

from eval_harness.metrics.judge import SubscriptionJudge

EVALUATION_STEPS = [
    "Read the task (input) and the generated patch (actual output). Judge only the patch. "
    "Test files are supplied and locked by the harness, so never deduct for tests the patch "
    "does not add or change.",
    "Conventions: check that naming, formatting, import style, comment style and error-handling "
    "style match the surrounding code visible in the diff context. Deduct for departures.",
    "Scope: deduct for dead code, speculative abstractions, debug output, unrelated refactors, or "
    "changes the task did not ask for. Changes the task text explicitly suggests are in scope.",
    "Error handling: where the surrounding code handles failures (try/catch, null checks, "
    "validation), the patch must too. Deduct for new unchecked failure paths.",
    "Security: deduct heavily for injection risks, hard-coded secrets, unsafe eval or shell "
    "construction, disabled validation, or leaking sensitive data to logs.",
    "Score 1.0 for a patch a senior reviewer would merge without comments, around 0.7 for minor "
    "nits, around 0.4 for real cleanup needed, and 0.0 for anything unsafe or clearly wrong.",
]


def make_code_quality_metric(judge: SubscriptionJudge, threshold: float = 0.7) -> GEval:
    return GEval(
        name="Code quality",
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        evaluation_steps=EVALUATION_STEPS,
        model=judge,
        threshold=threshold,
        async_mode=False,
    )
