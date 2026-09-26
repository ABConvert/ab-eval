from __future__ import annotations

from eval_harness.collect.cases import Case

SYSTEM_PROMPT = "\n".join(
    [
        "You are a senior engineer working in a checked-out repository at /app inside an",
        "offline sandbox. Your job: implement the task described by the user so that the named",
        "test files pass, following the surrounding code's conventions.",
        "Rules:",
        "- Use the tools to inspect and edit files. Do not guess file contents; read them.",
        "- Do not modify the test files; they are the acceptance criteria and are restored",
        "  before grading.",
        "- Do not run package installs or network commands; there is no network.",
        "- Keep the change focused. No speculative refactors, no dead code, no debug output",
        "  left behind.",
        "- When the tests pass and the change is complete, reply with a short summary and",
        "  stop calling tools.",
    ]
)


def build_task(case: Case, tree: str, failing_output: str) -> str:
    files = "\n".join(f"- {f}" for f in case.test_files)
    return (
        f"{case.task_prompt}\n\n"
        "## Acceptance tests\n"
        f"The following test files must pass (run them with the run_tests tool):\n{files}\n\n"
        "## Current test output (before any change)\n```\n"
        f"{failing_output[-6000:]}\n```\n\n"
        "## Repository tree (depth 2)\n```\n"
        f"{tree[:8000]}\n```\n"
    )
