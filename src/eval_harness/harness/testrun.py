from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET

from pydantic import BaseModel, Field

from eval_harness.config import Runner
from eval_harness.harness.sandbox import Docker, ExecResult

WRAPPER = ".abeval.vitest.config.mjs"
RESULT_JSON = "/tmp/abeval-tests.json"
MAX_OUTPUT = 60_000


class TestResult(BaseModel):
    total: int
    passed: int
    failed: int
    skipped: int
    exit_code: int
    output: str
    failed_tests: list[str] = Field(default_factory=list)  # "<repo path>::<full name>"

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and self.failed == 0 and self.total > 0


def vitest_wrapper_config(base_config: str, rel_files: list[str]) -> str:
    """ESM config that loads the repo's config and pins `test.include` to the case's files."""
    files = json.dumps(rel_files)
    return (
        f"import base from './{base_config}';\n"
        "const resolved = typeof base === 'function'"
        " ? await base({ command: 'serve', mode: 'test' }) : base;\n"
        "resolved.test = resolved.test ?? {};\n"
        f"resolved.test.include = {files};\n"
        "resolved.test.exclude = ['**/node_modules/**'];\n"
        "export default resolved;\n"
    )


def _rel(runner: Runner, files: list[str]) -> list[str]:
    return [os.path.relpath(f, runner.cwd) for f in files]


def render_test_command(runner: Runner, test_files: list[str]) -> str:
    if runner.kind == "vitest":
        return (
            f"cd {runner.cwd} && npx vitest run --config {WRAPPER}"
            f" --reporter=default --reporter=json --outputFile={RESULT_JSON}"
        )
    if runner.kind == "pytest":
        files = " ".join(_rel(runner, test_files))
        return f"cd {runner.cwd} && python -m pytest --junitxml={RESULT_JSON} {files}"
    files = " ".join(_rel(runner, test_files))
    return f"cd {runner.cwd} && flutter test --machine {files} > {RESULT_JSON}"


def parse_vitest_json(text: str) -> tuple[int, int, int, int]:
    data = json.loads(text)
    return (
        int(data["numTotalTests"]),
        int(data["numPassedTests"]),
        int(data["numFailedTests"]),
        int(data.get("numPendingTests", 0)),
    )


def vitest_failed_tests(text: str) -> list[str]:
    """Failed test ids from vitest's JSON reporter as '<path relative to /app>::<full name>'."""
    data = json.loads(text)
    out: list[str] = []
    for suite in data.get("testResults", []):
        path = str(suite.get("name", ""))
        path = path[len("/app/") :] if path.startswith("/app/") else path
        failed_here = [a for a in suite.get("assertionResults", []) if a.get("status") == "failed"]
        out.extend(f"{path}::{a.get('fullName', '?')}" for a in failed_here)
        if suite.get("status") == "failed" and not failed_here:
            out.append(f"{path}::<suite error>")
    return out


def parse_pytest_junit(text: str) -> tuple[int, int, int, int]:
    root = ET.fromstring(text)
    total = failed = skipped = 0
    for s in root.iter("testsuite"):
        total += int(s.get("tests", 0))
        failed += int(s.get("failures", 0)) + int(s.get("errors", 0))
        skipped += int(s.get("skipped", 0))
    return total, total - failed - skipped, failed, skipped


def pytest_failed_tests(text: str) -> list[str]:
    root = ET.fromstring(text)
    out: list[str] = []
    for case in root.iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            where = case.get("file") or case.get("classname", "?")
            out.append(f"{where}::{case.get('name', '?')}")
    return out


def _flutter_events(text: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("{"):
            events.append(json.loads(line))
    return events


def parse_flutter_machine(text: str) -> tuple[int, int, int, int]:
    total = passed = failed = skipped = 0
    for ev in _flutter_events(text):
        if ev.get("type") != "testDone" or ev.get("hidden"):
            continue
        total += 1
        if ev.get("skipped"):
            skipped += 1
        elif ev.get("result") == "success":
            passed += 1
        else:
            failed += 1
    return total, passed, failed, skipped


def flutter_failed_tests(text: str) -> list[str]:
    names: dict[int, str] = {}
    out: list[str] = []
    for ev in _flutter_events(text):
        if ev.get("type") == "testStart":
            test = ev["test"]
            assert isinstance(test, dict)
            names[int(test["id"])] = str(test.get("name", "?"))
        elif ev.get("type") == "testDone" and not ev.get("hidden"):
            if ev.get("result") != "success" and not ev.get("skipped"):
                out.append(f"flutter::{names.get(int(str(ev['testID'])), '?')}")
    return out


def _truncate(s: str) -> str:
    if len(s) <= MAX_OUTPUT:
        return s
    half = MAX_OUTPUT // 2
    return s[:half] + "\n…[truncated]…\n" + s[-half:]


def _collect(docker: Docker, cid: str, runner: Runner, res: ExecResult) -> TestResult:
    total = passed = failed = skipped = 0
    failed_tests: list[str] = []
    try:
        text = docker.read_file(cid, RESULT_JSON)
        if runner.kind == "vitest":
            total, passed, failed, skipped = parse_vitest_json(text)
            failed_tests = vitest_failed_tests(text)
        elif runner.kind == "pytest":
            total, passed, failed, skipped = parse_pytest_junit(text)
            failed_tests = pytest_failed_tests(text)
        else:
            total, passed, failed, skipped = parse_flutter_machine(text)
            failed_tests = flutter_failed_tests(text)
    except Exception as e:
        res.stderr += f"\n[no parseable test report: {e}]"
    return TestResult(
        total=total,
        passed=passed,
        failed=failed,
        skipped=skipped,
        exit_code=res.code,
        output=_truncate(res.stdout + "\n" + res.stderr),
        failed_tests=failed_tests,
    )


def run_case_tests(
    docker: Docker, cid: str, runner: Runner, test_files: list[str], *, timeout: int = 900
) -> TestResult:
    if runner.kind == "vitest":
        assert runner.config is not None
        docker.write_file(
            cid,
            f"/app/{runner.cwd}/{WRAPPER}",
            vitest_wrapper_config(runner.config, _rel(runner, test_files)),
        )
    docker.exec(cid, f"rm -f {RESULT_JSON}")
    res = docker.exec(cid, render_test_command(runner, test_files), env=runner.env, timeout=timeout)
    return _collect(docker, cid, runner, res)


def full_command(runner: Runner) -> str:
    """`full`, plus whatever makes it write the report `_collect` reads.

    Only vitest used to get one, so a pytest baseline ran the whole suite (26k tests, 15
    minutes on the repository that found this) and was then read as zero tests.
    """
    if runner.kind == "vitest":
        return f"{runner.full} --reporter=default --reporter=json --outputFile={RESULT_JSON}"
    if runner.kind == "pytest" and "--junitxml" not in runner.full:
        return f"{runner.full} --junitxml={RESULT_JSON}"
    if runner.kind == "flutter" and "--machine" not in runner.full:
        return f"{runner.full} --machine > {RESULT_JSON}"
    return runner.full


def run_full_suite(docker: Docker, cid: str, runner: Runner, *, timeout: int = 2400) -> TestResult:
    docker.exec(cid, f"rm -f {RESULT_JSON}")
    cmd = f"cd {runner.cwd} && {full_command(runner)}"
    res = docker.exec(cid, cmd, env=runner.env, timeout=timeout)
    return _collect(docker, cid, runner, res)


def run_lint(docker: Docker, cid: str, runner: Runner, files: list[str]) -> ExecResult | None:
    if not runner.lint or not files:
        return None
    cmd = runner.lint.format(files=" ".join(_rel(runner, files)))
    return docker.exec(cid, f"cd {runner.cwd} && {cmd}", timeout=300)


def aggregate(results: list[TestResult]) -> TestResult:
    """Sum several runner results into one; exit code is non-zero if any part failed."""
    if not results:
        return TestResult(total=0, passed=0, failed=0, skipped=0, exit_code=1, output="")
    return TestResult(
        total=sum(r.total for r in results),
        passed=sum(r.passed for r in results),
        failed=sum(r.failed for r in results),
        skipped=sum(r.skipped for r in results),
        exit_code=max(r.exit_code for r in results),
        output=_truncate("\n\n".join(r.output for r in results)),
        failed_tests=[t for r in results for t in r.failed_tests],
    )


def run_groups(
    docker: Docker,
    cid: str,
    groups: list[tuple[Runner, list[str]]],
    *,
    timeout: int = 900,
) -> TestResult:
    return aggregate(
        [run_case_tests(docker, cid, r, files, timeout=timeout) for r, files in groups]
    )


def run_full_suites(
    docker: Docker, cid: str, runners: list[Runner], *, timeout: int = 2400
) -> TestResult:
    return aggregate([run_full_suite(docker, cid, r, timeout=timeout) for r in runners])
