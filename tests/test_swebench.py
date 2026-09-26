"""Importing SWE-bench: the mapping, and the two fields that have no counterpart."""

from __future__ import annotations

import json

import pytest

from eval_harness.collect import swebench
from eval_harness.collect.cases import TestCriterion

ROW = {
    "instance_id": "django__django-11099",
    "repo": "django/django",
    "base_commit": "a" * 40,
    "problem_statement": "UsernameValidator allows trailing newline",
    "patch": "diff --git a/x b/x\n+one\n+two\n",
    "test_patch": "diff --git a/t b/t\n",
    "FAIL_TO_PASS": '["t.py::TestX::test_a"]',
    "PASS_TO_PASS": '["t.py::TestX::test_b", "u.py::test_c"]',
    "created_at": "2019-03-20T00:00:00Z",
    "environment_setup_commit": "b" * 40,
}


def test_named_tests_become_the_criterion() -> None:
    case = swebench.to_case(ROW, variant="verified")
    assert case.criterion == TestCriterion(
        fail_to_pass=["t.py::TestX::test_a"],
        pass_to_pass=["t.py::TestX::test_b", "u.py::test_c"],
    )


def test_test_files_are_derived_from_the_named_tests() -> None:
    assert swebench.to_case(ROW, variant="verified").test_files == ["t.py", "u.py"]


def test_name_lists_survive_both_encodings() -> None:
    """The datasets server has returned these JSON-encoded and already decoded."""
    assert swebench._names('["a", "b"]') == ["a", "b"]
    assert swebench._names(["a", "b"]) == ["a", "b"]
    assert swebench._names(None) == []
    assert swebench._names("not json") == []


def test_provenance_is_recorded_so_a_mixed_set_stays_legible() -> None:
    case = swebench.to_case(ROW, variant="lite")
    assert case.source == "swebench:lite"
    assert case.kind_source == "swebench"


def test_problem_statements_are_marked_symptom_not_solution() -> None:
    """These are real issue reports, unlike a ticket written by the person who then fixes it."""
    assert swebench.to_case(ROW, variant="verified").spec_level_heuristic == "symptom"


def test_the_environment_commit_is_surfaced_not_silently_dropped() -> None:
    assert swebench.environment_commit(ROW) == "b" * 40
    assert swebench.environment_commit({}) == ""


def test_fetch_pages_until_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, object]] = []

    class _Reply:
        status_code = 200

        def __init__(self, n: int) -> None:
            self._n = n

        def json(self) -> dict[str, object]:
            return {"rows": [{"row": {"instance_id": f"i{i}"}} for i in range(self._n)]}

    class _Client:
        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *e: object) -> None:
            return None

        def get(self, url: str, *, params: dict[str, object]) -> _Reply:
            seen.append(params)
            return _Reply(int(params["length"]))

    monkeypatch.setattr(swebench.httpx, "Client", lambda **kw: _Client())
    rows = swebench.fetch_rows("verified", 150)
    assert len(rows) == 150
    assert [p["length"] for p in seen] == [100, 50], "should page at the server maximum"


def test_an_error_from_the_server_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Reply:
        status_code = 503
        text = "upstream unavailable"

        def json(self) -> dict[str, object]:
            return {}

    class _Client:
        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *e: object) -> None:
            return None

        def get(self, url: str, *, params: dict[str, object]) -> _Reply:
            return _Reply()

    monkeypatch.setattr(swebench.httpx, "Client", lambda **kw: _Client())
    with pytest.raises(RuntimeError, match="503"):
        swebench.fetch_rows("verified", 1)


def test_a_case_round_trips_through_json() -> None:
    case = swebench.to_case(ROW, variant="verified")
    from eval_harness.collect.cases import Case

    again = Case.model_validate(json.loads(case.model_dump_json()))
    assert again.criterion is not None
    assert again.criterion.fail_to_pass == ["t.py::TestX::test_a"]
