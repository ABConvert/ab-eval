from __future__ import annotations

import asyncio
import hashlib
import subprocess
import time
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from eval_harness import PROJECT_ROOT
from eval_harness.adapters.base import Caps, ModelAdapter
from eval_harness.adapters.tracing import Tracer
from eval_harness.collect.cases import Case
from eval_harness.config import RepoConfig, Runner
from eval_harness.harness.deps import ensure_dep_volumes, mounts_for
from eval_harness.harness.prompts import SYSTEM_PROMPT, build_task
from eval_harness.harness.record import AttemptRecord, load_record, save_record
from eval_harness.harness.sandbox import Docker
from eval_harness.harness.source import (
    apply_patch,
    archive_tar,
    changed_paths,
    diff_head,
    git_init,
    populate_source,
    restore_paths,
)
from eval_harness.harness.testrun import TestResult, run_full_suites, run_groups, run_lint
from eval_harness.harness.tools import TOOL_SPECS, SandboxToolExecutor

# A regression is confirmed only if it fails in most isolated re-runs of its own files.
REGRESSION_RECHECKS = 3


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def harness_git_sha() -> str:
    proc = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() or "unknown"


def regressions_vs_baseline(case: Case, failed_tests: list[str]) -> list[str]:
    """Full-suite failures that are neither in the case's baseline nor in its own test files."""
    baseline = set(case.full_suite_baseline_failures or [])
    own = set(case.test_files)
    return sorted(t for t in set(failed_tests) - baseline if t.split("::", 1)[0] not in own)


def confirm_regressions(
    candidates: list[str], rechecks: list[list[str]], *, needed: int | None = None
) -> list[str]:
    """Keep candidates that fail in most isolated re-runs.

    One isolated re-run is not enough: a test flaky under load can also fail alone. The
    reference patch flagged an unrelated auth test as a regression, which proved
    a single re-check lets flakes through, so confirmation needs a majority.
    """
    if not rechecks:
        return []
    needed = needed if needed is not None else len(rechecks) // 2 + 1
    counts = Counter(test for failed in rechecks for test in set(failed))
    return [t for t in candidates if counts[t] >= needed]


def case_groups(case: Case, repo: RepoConfig) -> list[tuple[str, Runner, list[str]]]:
    """Runner groups from the case file, or derived from its test files for older cases."""
    if case.test_groups:
        return [(g.runner, repo.runners[g.runner], g.files) for g in case.test_groups]
    return repo.runner_groups(case.test_files)


DOCKERFILE_LABEL = "abeval.dockerfile-sha"


def dockerfile_digest(dockerfile: Path) -> str:
    return hashlib.sha256(dockerfile.read_bytes()).hexdigest()[:12]


def ensure_image(docker: Docker, repo: RepoConfig) -> None:
    if not docker.daemon_ok():
        raise RuntimeError("Docker daemon is not reachable; start Docker Desktop and retry")
    dockerfile = PROJECT_ROOT / repo.dockerfile
    digest = dockerfile_digest(dockerfile)
    # Rebuilt when the Dockerfile changes, not only when the tag is missing: an image built
    # before a harness upgrade would otherwise keep the old toolchain indefinitely, and the
    # upgrade's fix would never reach the machine that needed it.
    if docker.image_label(repo.image, DOCKERFILE_LABEL) != digest:
        docker.build_image(repo.image, dockerfile, PROJECT_ROOT, labels={DOCKERFILE_LABEL: digest})


# Validation writes here; model runs read it back to skip cases that are not measurements.
VALIDATE_RUN = "validate"


def broken_environment(full: TestResult | None) -> str | None:
    """Why the sandbox cannot run the suite at all, or None."""
    if full is None or full.total > 0 or full.exit_code == 0:
        return None
    output = (full.output or "").strip()
    # A missing report is always noted, so only blame the report when a session did start.
    if "no parseable test report" in output and "test session starts" in output:
        tail = " ".join(output.splitlines()[-4:])[:240]
        return f"the full suite ran but its report could not be read: {tail}"
    tail = " ".join(output.splitlines()[:2])[:200]
    return (
        f"the full suite ran no tests and exited {full.exit_code}, so the sandbox cannot run "
        f"tests — check dep_dirs (target, install) and the image: {tail}"
    )


def reference_failure(result: TestResult) -> str | None:
    """Why a case's own reference patch does not pass its tests, or None when it does.

    A case the team's merged change cannot pass measures the harness, not the model: in
    bench-v2 a missing sandbox package, a suite that collected nothing and a test reading a
    file the case builder dropped were each scored as a model failure.
    """
    if result.ok:
        return None
    if result.total == 0:
        return "no tests ran with the reference patch applied"
    named = result.failed_tests[:3]
    more = f" and {len(result.failed_tests) - 3} more" if len(result.failed_tests) > 3 else ""
    if named:
        return "the reference patch does not pass its own tests: " + ", ".join(named) + more
    return f"the reference patch does not pass its own tests (exit {result.exit_code})"


def invalid_by_validation(case_id: str) -> str | None:
    """The reason validation marked this case invalid, or None if it passed or never ran."""
    rec = load_record(VALIDATE_RUN, case_id)
    if rec is None or rec.status != "invalid":
        return None
    return rec.error or "marked invalid at validation"


async def run_case(
    case: Case,
    repo: RepoConfig,
    *,
    adapter: ModelAdapter,
    caps: Caps,
    run_id: str,
    docker: Docker,
    validate_only: bool = False,
) -> AttemptRecord:
    rec = AttemptRecord(
        case_id=case.case_id,
        run_id=run_id,
        model=adapter.name,
        status="error",
        phase="setup",
        started_at=_now(),
    )
    if not validate_only:
        reason = invalid_by_validation(case.case_id)
        if reason is not None:
            rec.status, rec.phase, rec.finished_at = "invalid", "done", _now()
            rec.error = f"invalid at validation: {reason}"
            return rec
    t0 = time.monotonic()
    groups = case_groups(case, repo)
    runners = [r for _, r, _ in groups]
    lint_runner = next((r for r in runners if r.lint), None)
    cid: str | None = None
    tracer = (
        Tracer.noop()
        if validate_only
        else Tracer.for_attempt(case_id=case.case_id, run_id=run_id, model=adapter.name)
    )
    try:
        tar = await asyncio.to_thread(archive_tar, repo, case.base_commit)
        volumes = await asyncio.to_thread(
            ensure_dep_volumes,
            docker,
            repo,
            case.base_commit,
            source_tar=tar,
            dep_dirs=repo.deps_for(runners),
        )
        name = f"abeval-{run_id}-{case.case_id}".lower()[:60]
        docker.rm(name)  # a killed job can leave a stale container holding this name
        cid = docker.create_container(
            image=repo.image,
            name=name,
            mounts=mounts_for(repo, volumes),
            limits=repo.limits,
            network=False,
        )
        docker.start(cid)
        populate_source(docker, cid, tar)
        git_init(docker, cid)
        apply_patch(docker, cid, case.human_test_patch, "tests")
        rec.phase = "tests_before"
        rec.tests_before = await asyncio.to_thread(
            run_groups, docker, cid, [(r, fs) for _, r, fs in groups]
        )
        if rec.tests_before.ok:
            rec.status = "invalid"
            rec.error = "tests pass before any fix"
            return rec
        if rec.tests_before.total == 0 and rec.tests_before.exit_code == 0:
            # A non-zero exit with no parsed tests is a legitimate failure (for example the
            # test imports a module the fix creates); zero tests with a clean exit is not.
            rec.status = "invalid"
            rec.error = "no tests ran before the fix"
            return rec
        if validate_only:
            rec.phase = "baseline"
            rec.full_suite = await asyncio.to_thread(run_full_suites, docker, cid, runners)
            reason = broken_environment(rec.full_suite)
            if reason:
                # The case's own tests failing with nothing collected is ambiguous — the
                # fix may create the module they import. The whole suite collecting
                # nothing is not: the sandbox cannot run tests at all, and calling that
                # "invalid" would make `run` skip a good case for the harness's fault.
                rec.status, rec.error, rec.phase = "error", reason, "done"
                return rec
            rec.phase = "reference"
            apply_patch(docker, cid, case.human_patch, "reference")
            rec.tests_after = await asyncio.to_thread(
                run_groups, docker, cid, [(r, fs) for _, r, fs in groups]
            )
            reason = reference_failure(rec.tests_after)
            rec.status = "invalid" if reason else "completed"
            rec.error = reason
            rec.phase = "done"
            return rec
        rec.phase = "agent"
        tree = docker.exec(
            cid,
            "find . -maxdepth 2 -not -path '*/node_modules*' -not -path '*/.venv*'"
            " -not -path './.git*' | sort",
        ).stdout
        execute = SandboxToolExecutor(
            docker, cid, groups=[(r, fs) for _, r, fs in groups], caps=caps
        )
        execute.identity = {
            "case_id": case.case_id,
            "cid": cid,
            "repo": repo.key,
            "groups": [[n, fs] for n, _, fs in groups],
        }
        result = await adapter.run_agent(
            system=SYSTEM_PROMPT,
            task=build_task(case, tree, rec.tests_before.output),
            tools=TOOL_SPECS,
            execute=execute,
            caps=caps,
            trace=tracer,
        )
        rec.usage, rec.cost_usd = result.usage, result.cost_usd
        rec.turns, rec.tool_calls = result.turns, result.tool_calls
        rec.cap_hit, rec.stop_reason = result.cap_hit, result.stop_reason
        rec.request_settings, rec.final_text = result.request_settings, result.final_text
        rec.tool_log = execute.log or list(result.request_settings.pop("tool_log", []))
        agent_wall = result.request_settings.get("wall_clock_seconds")
        rec.agent_wall_clock_seconds = float(agent_wall) if agent_wall is not None else None
        rec.phase = "grade"
        touched = changed_paths(docker, cid)
        rec.touched_test_files = [p for p in touched if p in case.test_files]
        restore_paths(docker, cid, case.test_files)
        rec.files_touched = [p for p in touched if p not in case.test_files]
        rec.generated_diff = diff_head(docker, cid)
        rec.criterion = case.criterion.model_dump() if case.criterion else None
        rec.tests_after = await asyncio.to_thread(
            run_groups, docker, cid, [(r, fs) for _, r, fs in groups]
        )
        if rec.tests_after.ok:
            rec.full_suite = await asyncio.to_thread(run_full_suites, docker, cid, runners)
            reason = broken_environment(rec.full_suite)
            if reason:
                # The case's own tests failing with nothing collected is ambiguous — the
                # fix may create the module they import. The whole suite collecting
                # nothing is not: the sandbox cannot run tests at all, and calling that
                # "invalid" would make `run` skip a good case for the harness's fault.
                rec.status, rec.error, rec.phase = "error", reason, "done"
                return rec
            candidates = regressions_vs_baseline(case, rec.full_suite.failed_tests)
            rec.regression_candidates = candidates
            if candidates:
                files = sorted({t.split("::", 1)[0] for t in candidates})
                recheck_groups = [(r, fs) for _, r, fs in repo.runner_groups(files)]
                rechecks: list[list[str]] = []
                for _ in range(REGRESSION_RECHECKS):
                    recheck = await asyncio.to_thread(run_groups, docker, cid, recheck_groups)
                    rechecks.append(recheck.failed_tests)
                    if not recheck.failed_tests:
                        break  # a clean isolated run can never reach a majority
                rec.regressions = confirm_regressions(
                    candidates, rechecks, needed=REGRESSION_RECHECKS // 2 + 1
                )
            else:
                rec.regressions = []
            lint = (
                await asyncio.to_thread(run_lint, docker, cid, lint_runner, rec.files_touched)
                if lint_runner
                else None
            )
            if lint is not None:
                rec.lint_ok = lint.ok
                rec.lint_output = (lint.stdout + lint.stderr)[-6000:]
        rec.status = "completed"
        rec.phase = "done"
        return rec
    except Exception as e:
        rec.status = "error"
        rec.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}"
        return rec
    finally:
        rec.wall_clock_seconds = round(time.monotonic() - t0, 1)
        rec.finished_at = _now()
        if cid:
            docker.rm(cid)
        tracer.finish({"status": rec.status, "resolved": rec.resolved, "cost_usd": rec.cost_usd})
        save_record(rec)


async def run_many(
    cases: list[Case],
    repo: RepoConfig,
    *,
    adapter: ModelAdapter,
    caps: Caps,
    run_id: str,
    concurrency: int,
    retry_errors: bool = False,
    validate_only: bool = False,
    recheck: bool = False,
) -> list[AttemptRecord]:
    docker = Docker()
    ensure_image(docker, repo)
    sem = asyncio.Semaphore(concurrency)

    async def one(case: Case) -> AttemptRecord:
        existing = load_record(run_id, case.case_id)
        if (
            existing
            and not recheck
            and (existing.status in ("completed", "invalid") or not retry_errors)
        ):
            existing.reused = True
            return existing
        async with sem:
            return await run_case(
                case,
                repo,
                adapter=adapter,
                caps=caps,
                run_id=run_id,
                docker=docker,
                validate_only=validate_only,
            )

    return list(await asyncio.gather(*(one(c) for c in cases)))
