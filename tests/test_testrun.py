from pathlib import Path

from eval_harness.config import load_repos
from eval_harness.harness.testrun import (
    flutter_failed_tests,
    parse_flutter_machine,
    parse_pytest_junit,
    parse_vitest_json,
    pytest_failed_tests,
    render_test_command,
    vitest_failed_tests,
    vitest_wrapper_config,
)

FIX = Path(__file__).parent / "fixtures"


def test_parse_vitest_json() -> None:
    assert parse_vitest_json((FIX / "vitest-fail.json").read_text()) == (25, 22, 3, 0)
    assert parse_vitest_json((FIX / "vitest-pass.json").read_text()) == (25, 25, 0, 0)


def test_parse_pytest_junit() -> None:
    assert parse_pytest_junit((FIX / "pytest-junit.xml").read_text()) == (4, 2, 1, 1)


def test_parse_flutter_machine() -> None:
    assert parse_flutter_machine((FIX / "flutter-machine.jsonl").read_text()) == (3, 1, 1, 1)


def test_wrapper_config_overrides_include() -> None:
    cfg = vitest_wrapper_config(
        "vitest.unit.config.ts", ["services/demo/__tests__/ExampleService.test.ts"]
    )
    assert "./vitest.unit.config.ts" in cfg
    assert '"services/demo/__tests__/ExampleService.test.ts"' in cfg
    assert "include" in cfg


def test_render_test_command_uses_wrapper() -> None:
    runner = load_repos()["demo-app"].runners["unit"]
    cmd = render_test_command(runner, ["web/services/demo/__tests__/ExampleService.test.ts"])
    assert cmd.startswith("cd web && ")
    assert "--config .abeval.vitest.config.mjs" in cmd
    assert "--reporter=json" in cmd


def test_failed_test_ids() -> None:
    assert vitest_failed_tests((FIX / "vitest-fail.json").read_text()) == [
        "web/services/demo/__tests__/ExampleService.test.ts::calculateTotal handles sample input"
    ]
    assert vitest_failed_tests((FIX / "vitest-pass.json").read_text()) == []
    assert pytest_failed_tests((FIX / "pytest-junit.xml").read_text()) == ["t::b"]
    assert flutter_failed_tests((FIX / "flutter-machine.jsonl").read_text()) == [
        "flutter::subtracts"
    ]


def test_aggregate_sums_runner_results() -> None:
    from eval_harness.harness.testrun import TestResult, aggregate

    a = TestResult(total=5, passed=5, failed=0, skipped=0, exit_code=0, output="a")
    b = TestResult(
        total=3, passed=2, failed=1, skipped=0, exit_code=1, output="b", failed_tests=["x::y"]
    )
    agg = aggregate([a, b])
    assert (agg.total, agg.passed, agg.failed, agg.exit_code, agg.failed_tests) == (
        8,
        7,
        1,
        1,
        ["x::y"],
    )
    assert not agg.ok and aggregate([a]).ok and not aggregate([]).ok
