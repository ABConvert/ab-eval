"""A one-at-a-time job queue that shells out to the CLI.

The sandbox VM has room for a single case container, so the dashboard never runs two
harness jobs at once. Jobs and their logs live under results/_jobs/ so a page reload
(or a later session) still sees what ran.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval_harness import PROJECT_ROOT
from eval_harness.paths import jobs_dir as default_jobs_dir

TAIL_BYTES = 20_000


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class Job:
    id: str
    kind: str  # run | validate | score | compare
    label: str
    argv: list[str]
    run_id: str | None = None
    cases: list[str] = field(default_factory=list)
    status: str = "queued"  # queued | running | done | failed | stopped
    queued_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None

    # Set by the queue on submit, so a queue with its own directory keeps its logs there.
    jobs_dir: Path | None = None

    @property
    def log_path(self) -> Path:
        return (self.jobs_dir or default_jobs_dir()) / f"{self.id}.log"

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")

    def elapsed_s(self) -> float | None:
        if not self.started_at:
            return None
        end = self.finished_at or _now()
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        return (
            datetime.strptime(end, fmt) - datetime.strptime(self.started_at, fmt)
        ).total_seconds()

    def tail(self, limit: int = TAIL_BYTES) -> str:
        path = self.log_path
        if not path.exists():
            return ""
        data = path.read_bytes()
        return data[-limit:].decode("utf-8", errors="replace")


def harness_argv(*args: str) -> list[str]:
    """Run the CLI through this interpreter so the job inherits the dashboard's venv."""
    return [sys.executable, "-m", "eval_harness", *args]


def run_job(
    *, model: str, run_id: str, cases: list[str], max_turns: int | None, wall_clock: int | None
) -> Job:
    """Build the `run` argv. A cap left blank on the form is a flag that is not passed.

    Not a default filled in here: the CLI reads the model's own `caps:` block when a flag is
    absent, so sending 40 and 1800 on every launch would mean a model's declared budget only
    ever applied to runs started from a terminal.
    """
    args = [
        "run",
        "--model",
        model,
        "--case",
        ",".join(cases),
        "--run-id",
        run_id,
        "--concurrency",
        "1",
        "--retry-errors",
    ]
    if max_turns is not None:
        args += ["--max-turns", str(max_turns)]
    if wall_clock is not None:
        args += ["--wall-clock", str(wall_clock)]
    return Job(
        id=uuid.uuid4().hex[:12],
        kind="run",
        label=f"{model} → {run_id} · {len(cases)} case(s)",
        argv=harness_argv(*args),
        run_id=run_id,
        cases=cases,
    )


def validate_job(cases: list[str]) -> Job:
    args = ["validate", "--case", ",".join(cases), "--concurrency", "1", "--retry-errors"]
    return Job(
        id=uuid.uuid4().hex[:12],
        kind="validate",
        label=f"validate · {len(cases)} case(s)",
        argv=harness_argv(*args),
        run_id="validate",
        cases=cases,
    )


def score_job(run_id: str) -> Job:
    return Job(
        id=uuid.uuid4().hex[:12],
        kind="score",
        label=f"score · {run_id}",
        argv=harness_argv("score", "--run", run_id),
        run_id=run_id,
    )


def review_tests_job(case_ids: list[str]) -> Job:
    return Job(
        id=uuid.uuid4().hex[:12],
        kind="review-tests",
        label=f"test quality · {len(case_ids)} case(s)",
        argv=harness_argv("review-tests", "--case", ",".join(case_ids)),
        cases=case_ids,
    )


def add_ticket_job(dataset: str, ticket: str, kind: str) -> Job:
    return Job(
        id=uuid.uuid4().hex[:12],
        kind="dataset",
        label=f"add {ticket} to {dataset}",
        argv=harness_argv("dataset", "add", "--name", dataset, "--ticket", ticket, "--kind", kind),
    )


def compare_job(run_ids: list[str]) -> Job:
    args = ["compare"]
    for rid in run_ids:
        args += ["--runs", rid]
    return Job(
        id=uuid.uuid4().hex[:12],
        kind="compare",
        label="compare · " + " vs ".join(run_ids),
        argv=harness_argv(*args),
    )


class JobQueue:
    """One worker thread; at most one child process alive at a time."""

    def __init__(self, jobs_dir: Path | None = None) -> None:
        jobs_dir = jobs_dir or default_jobs_dir()
        self.dir = jobs_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._queue: deque[Job] = deque()
        self._jobs: dict[str, Job] = {}
        self._current: Job | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._stopping = False
        self._load_history()
        self._worker = threading.Thread(target=self._loop, name="abeval-jobs", daemon=True)
        self._worker.start()

    # ---------- persistence ----------

    def _path(self, job: Job) -> Path:
        return self.dir / f"{job.id}.json"

    def _save(self, job: Job) -> None:
        # jobs_dir is where this file already is; persisting it would be both redundant and
        # unserialisable.
        body = {k: v for k, v in asdict(job).items() if k != "jobs_dir"}
        self._path(job).write_text(json.dumps(body, indent=2) + "\n")

    def _load_history(self) -> None:
        for path in sorted(self.dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                job = Job(**{k: v for k, v in data.items() if k in Job.__dataclass_fields__})
                job.jobs_dir = self.dir  # its log sits beside the record being read
            except Exception:
                continue
            if job.active:  # a job left running by a killed dashboard never resumes
                job.status = "stopped"
                job.finished_at = job.finished_at or _now()
                self._save(job)
            self._jobs[job.id] = job

    # ---------- public API ----------

    def submit(self, job: Job) -> Job:
        job.jobs_dir = self.dir
        with self._lock:
            self._jobs[job.id] = job
            self._queue.append(job)
            self._save(job)
            self._wake.notify_all()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def current(self) -> Job | None:
        with self._lock:
            return self._current

    def queued(self) -> list[Job]:
        with self._lock:
            return list(self._queue)

    def recent(self, limit: int = 12) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.queued_at, reverse=True)
        return jobs[:limit]

    def active_for_run(self, run_id: str) -> Job | None:
        with self._lock:
            for job in list(self._queue) + ([self._current] if self._current else []):
                if job and job.run_id == run_id and job.active:
                    return job
        return None

    def cancel(self, job_id: str) -> bool:
        """Drop a queued job, or terminate the running one."""
        with self._lock:
            for job in list(self._queue):
                if job.id == job_id:
                    self._queue.remove(job)
                    job.status = "stopped"
                    job.finished_at = _now()
                    self._save(job)
                    return True
            if self._current and self._current.id == job_id and self._proc:
                self._stopping = True
                proc = self._proc
            else:
                return False
        self._terminate(proc)
        return True

    def shutdown(self) -> None:
        job = self.current()
        if job:
            self.cancel(job.id)

    # ---------- worker ----------

    def _terminate(self, proc: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.3)
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

    def _loop(self) -> None:
        while True:
            with self._wake:
                while not self._queue:
                    self._wake.wait()
                job = self._queue.popleft()
                self._current = job
                self._stopping = False
                job.status = "running"
                job.started_at = _now()
                self._save(job)
            self._execute(job)
            with self._lock:
                self._current = None
                self._proc = None

    def _execute(self, job: Job) -> None:
        env = dict(os.environ)
        env.pop("CLAUDECODE", None)  # the CLI spawns `claude`, which refuses to nest
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            with job.log_path.open("wb") as log:
                log.write(f"$ {' '.join(job.argv)}\n\n".encode())
                log.flush()
                proc = subprocess.Popen(
                    job.argv,
                    cwd=PROJECT_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=env,
                    start_new_session=True,  # so cancel() can kill the whole tree
                )
                with self._lock:
                    self._proc = proc
                code = proc.wait()
        except Exception as e:  # a job that cannot start must not kill the worker
            job.status = "failed"
            job.exit_code = None
            job.finished_at = _now()
            with job.log_path.open("a") as log:
                log.write(f"\nfailed to start: {type(e).__name__}: {e}\n")
            self._save(job)
            return
        job.exit_code = code
        stopped = self._stopping
        job.status = "stopped" if stopped else ("done" if code == 0 else "failed")
        job.finished_at = _now()
        self._save(job)


_queue: JobQueue | None = None


def queue() -> JobQueue:
    global _queue
    if _queue is None:
        _queue = JobQueue()
    return _queue


def job_dict(job: Job) -> dict[str, Any]:
    d = asdict(job)
    d["elapsed_s"] = job.elapsed_s()
    return d
