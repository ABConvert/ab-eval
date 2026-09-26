from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from eval_harness.config import Limits


@dataclass
class ExecResult:
    code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.code == 0 and not self.timed_out


class DockerError(RuntimeError):
    pass


class Docker:
    """Thin wrapper over the docker CLI. Every method shells out; nothing is cached."""

    def _run(
        self, *args: str, input_bytes: bytes | None = None, timeout: int | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["docker", *args], input=input_bytes, capture_output=True, timeout=timeout
        )

    def _check(self, *args: str, input_bytes: bytes | None = None) -> str:
        proc = self._run(*args, input_bytes=input_bytes)
        if proc.returncode != 0:
            err = proc.stderr.decode(errors="replace")[-2000:]
            raise DockerError(f"docker {' '.join(args[:3])} failed: {err}")
        return proc.stdout.decode(errors="replace")

    def daemon_ok(self) -> bool:
        return self._run("info").returncode == 0

    def image_exists(self, tag: str) -> bool:
        return self._run("image", "inspect", tag).returncode == 0

    def build_image(self, tag: str, dockerfile: Path, context: Path) -> None:
        proc = subprocess.run(
            ["docker", "build", "-t", tag, "-f", str(dockerfile), str(context)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise DockerError(f"image build failed:\n{proc.stderr[-4000:]}")

    def volume_exists(self, name: str) -> bool:
        return self._run("volume", "inspect", name).returncode == 0

    def volume_create(self, name: str) -> None:
        self._check("volume", "create", name)

    def volume_rm(self, name: str) -> None:
        self._run("volume", "rm", "-f", name)

    def create_container(
        self,
        *,
        image: str,
        name: str,
        mounts: dict[str, str],
        limits: Limits,
        network: bool,
        workdir: str = "/app",
    ) -> str:
        args = [
            "create",
            "--name",
            name,
            "-w",
            workdir,
            "--cpus",
            str(limits.cpus),
            "--memory",
            limits.memory,
            "--pids-limit",
            str(limits.pids),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
        ]
        if not network:
            args += ["--network", "none"]
        for volume, path in mounts.items():
            args += ["-v", f"{volume}:{path}"]
        args += [image, "sleep", "infinity"]
        return self._check(*args).strip()

    def start(self, cid: str) -> None:
        self._check("start", cid)

    def exec(
        self,
        cid: str,
        cmd: list[str] | str,
        *,
        cwd: str = "/app",
        env: dict[str, str] | None = None,
        timeout: int = 300,
        user: str | None = None,
    ) -> ExecResult:
        shell_cmd = cmd if isinstance(cmd, str) else shlex.join(cmd)
        args = ["exec", "-w", cwd]
        for k, v in (env or {}).items():
            args += ["-e", f"{k}={v}"]
        if user:
            args += ["-u", user]
        args += [cid, "sh", "-lc", shell_cmd]
        try:
            proc = self._run(*args, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode(errors="replace")
            err = (e.stderr or b"").decode(errors="replace")
            return ExecResult(
                code=124, stdout=out, stderr=err + f"\n[timed out after {timeout}s]", timed_out=True
            )
        return ExecResult(
            proc.returncode,
            proc.stdout.decode(errors="replace"),
            proc.stderr.decode(errors="replace"),
        )

    def cp_tar_in(self, cid: str, tar: bytes, dest: str = "/app") -> None:
        self._check("cp", "-", f"{cid}:{dest}", input_bytes=tar)

    def write_file(self, cid: str, path: str, content: str) -> None:
        q = shlex.quote(path)
        self._check(
            "exec",
            "-i",
            cid,
            "sh",
            "-c",
            f"mkdir -p $(dirname {q}) && cat > {q}",
            input_bytes=content.encode(),
        )

    def read_file(self, cid: str, path: str) -> str:
        return self._check("exec", cid, "cat", path)

    def rm(self, cid: str) -> None:
        self._run("rm", "-f", cid)
