"""Rate a case easy, medium or hard from its own attributes.

Deliberately model-free. Difficulty derived from how many models solved a case would make
"performance by difficulty" circular: the hard cases would be hard by definition because
models failed them. This rates the work the case asks for, so a model's score can then be
sliced by it honestly, and `observed_rate` below can be used to check the rating against
what actually happened.

Thresholds come from the quartiles of the 47 collected cases; they are declared here so a
rating can be argued with rather than guessed at.
"""

from __future__ import annotations

from typing import Any

from eval_harness.collect.cases import Case

# How much each signal contributes. Size of change and how much the ticket gives away
# matter most: a big diff is more to get right, and a ticket containing the code is a
# transcription exercise.
POINTS: dict[str, int] = {
    "lines": 3,
    "files": 2,
    "spec": 2,
    "runners": 1,
    "review": 1,
    "cycle_time": 1,
}
EASY_BELOW = 3.0  # score < 3 is easy, < 6 is medium, else hard
HARD_FROM = 6.0

# Quartile boundaries over the collected cases.
LINES_Q1, LINES_Q3 = 109, 290
FILES_Q1, FILES_Q3 = 3, 6
REVIEW_Q3 = 4
HOURS_Q3 = 9.3


def _band(value: float, low: float, high: float) -> float:
    """0 below the lower quartile, 0.5 in the middle, 1 above the upper."""
    if value <= low:
        return 0.0
    return 1.0 if value >= high else 0.5


def signals(case: Case) -> dict[str, float]:
    """Each signal scaled to 0..1, before weighting. Exposed so the UI can explain a rating."""
    spec = {
        "code-in-ticket": 0.0,  # the ticket hands over the change
        "solution-specified": 0.5,  # the approach is named, the work is not done
        "symptom": 1.0,  # only the symptom; the model has to diagnose
    }.get(case.spec_level_heuristic, 0.5)
    runners = len(case.test_groups or [])
    return {
        "lines": _band(case.lines_changed, LINES_Q1, LINES_Q3),
        "files": _band(case.files_changed, FILES_Q1, FILES_Q3),
        "spec": spec,
        # touching two test suites means the change spans packages
        "runners": 1.0 if runners > 1 else 0.0,
        "review": _band(case.review_comment_count, 2, REVIEW_Q3),
        # how long it took a human who knew the codebase, capped so one stale ticket
        # sitting open for six weeks does not dominate
        "cycle_time": _band(min(case.human_cycle_time_hours, 72.0), 2.0, HOURS_Q3),
    }


def score(case: Case) -> float:
    s = signals(case)
    return round(sum(s[k] * POINTS[k] for k in POINTS), 2)


def classify(case: Case) -> str:
    total = score(case)
    if total < EASY_BELOW:
        return "easy"
    return "hard" if total >= HARD_FROM else "medium"


def explain(case: Case) -> dict[str, Any]:
    s = signals(case)
    return {
        "difficulty": classify(case),
        "score": score(case),
        "max_score": float(sum(POINTS.values())),
        "signals": {k: {"value": v, "points": round(v * POINTS[k], 2)} for k, v in s.items()},
        "inputs": {
            "lines_changed": case.lines_changed,
            "files_changed": case.files_changed,
            "spec_level": case.spec_level_heuristic,
            "test_runners": len(case.test_groups or []),
            "review_comments": case.review_comment_count,
            "human_cycle_time_hours": case.human_cycle_time_hours,
        },
    }


def observed_rate(case_id: str, runs: list[Any]) -> float | None:
    """Share of runs that solved this case: a check on the rating, never an input to it."""
    attempts = [r.records.get(case_id) for r in runs]
    tried = [a for a in attempts if a is not None and a.status == "completed"]
    if not tried:
        return None
    return round(sum(1 for a in tried if a.resolved) / len(tried), 4)
