#!/usr/bin/env python3
"""Wait for the identified pi0.5 run, then launch the aligned Show-o2 job once.

This is an operational guard, not a training controller. A stopped or uncertain
launch is left for a human to inspect; it is never retried automatically.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass


REPO = Path(__file__).resolve().parents[1]
PI_RUN = Path("/home/jiahao/task/openpi/outputs/csgo_benchmark_v2_seen10/pi0.5_exp32_loc_main_frozen_vl/seed_0")
PI_PID = 2292660
PI_START_TICKS = 534061261
PROFILE = "exp32_loc_main_frozen_vl"
FINAL_STEP = 19500
SHOWO_OUTPUT = REPO / "outputs/csgo_benchmark_v2_aligned/csgo_seen10_exp32gen_aligned_v2_final/seed_42"
SHOWO_LOG = REPO / "showo2_aligned.nohup.out"
COMMAND = ("nohup", "bash", "scripts/run_csgo_seen10.sh", "train", "--experiment", "csgo_seen10_exp32gen_aligned")


@dataclass(frozen=True)
class Settings:
    repo: Path = REPO
    pi_run: Path = PI_RUN
    pi_pid: int = PI_PID
    pi_start_ticks: int = PI_START_TICKS
    showo_output: Path = SHOWO_OUTPUT
    showo_log: Path = SHOWO_LOG
    allowed_gpu_identities: tuple[tuple[int, int], ...] = ()


def process_identity(pid: int) -> tuple[str, int] | None:
    """Return Linux process state and field 22, or None when the PID is gone."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    # comm can contain spaces and parentheses; fields after its last ')' start at 3.
    fields = stat[stat.rfind(")") + 2 :].split()
    if len(fields) < 20:
        raise RuntimeError(f"malformed /proc/{pid}/stat")
    return fields[0], int(fields[19])


def pi_state(settings: Settings) -> str:
    identity = process_identity(settings.pi_pid)
    if identity is None:
        return "exited"
    state, ticks = identity
    if ticks != settings.pi_start_ticks:
        return "pid_reused"
    return "exited" if state in ("Z", "X", "x") else "running"


def wait_for_pi_exit(pid: int, timeout: float) -> None:
    """Wake promptly on process exit without increasing progress polling."""
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    except (AttributeError, OSError):
        time.sleep(timeout)
        return
    try:
        poller = select.poll()
        poller.register(descriptor, select.POLLIN)
        poller.poll(math.ceil(timeout * 1000))
    finally:
        os.close(descriptor)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def last_jsonl(path: Path) -> dict:
    # Ignore an unfinished trailing line, but require the last complete record.
    with path.open("rb") as stream:
        lines = stream.readlines()
    if lines and not lines[-1].endswith(b"\n"):
        lines.pop()
    for line in reversed(lines):
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                return value
            raise ValueError(f"{path}: final record is not an object")
    raise ValueError(f"{path}: no complete records")


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def completion_errors(settings: Settings) -> list[str]:
    run = settings.pi_run
    errors: list[str] = []
    try:
        config = read_json(run / "run_config.json")
        runtime = config["runtime"]
        if not isinstance(runtime, dict) or any((
            config.get("experiment_profile") != PROFILE,
            config.get("smoke_only") is not False,
            runtime.get("experiment_profile") != PROFILE,
            runtime.get("seed") != 0,
            runtime.get("num_train_steps") != FINAL_STEP,
            runtime.get("smoke_only") is not False,
        )):
            errors.append("run_config identity or schedule mismatch")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"run_config unavailable: {exc}")

    try:
        loss = last_jsonl(run / "loss.jsonl")
        if loss.get("step") != FINAL_STEP or not finite_number(loss.get("loss")):
            errors.append("last loss record is not finite step 19500")
    except (OSError, ValueError) as exc:
        errors.append(f"loss record unavailable: {exc}")

    checkpoint = run / "checkpoints" / str(FINAL_STEP)
    try:
        identity = read_json(checkpoint / "checkpoint_identity.json")
        if any((identity.get("step") != FINAL_STEP,
                identity.get("experiment_profile") != PROFILE,
                identity.get("seed") != 0,
                identity.get("smoke_only") is not False)):
            errors.append("checkpoint identity mismatch")
    except (OSError, ValueError) as exc:
        errors.append(f"checkpoint identity unavailable: {exc}")
    try:
        metadata = read_json(checkpoint / "_CHECKPOINT_METADATA")
        commit = metadata.get("commit_timestamp_nsecs")
        if not isinstance(commit, int) or isinstance(commit, bool) or commit <= 0:
            errors.append("checkpoint has no positive commit timestamp")
    except (OSError, ValueError) as exc:
        errors.append(f"checkpoint metadata unavailable: {exc}")
    for name in ("params/manifest.ocdbt", "train_state/manifest.ocdbt"):
        if not (checkpoint / name).is_file():
            errors.append(f"missing checkpoint {name}")
    if (run / "checkpoints/late").resolve() != checkpoint.resolve():
        errors.append("late checkpoint does not resolve to step 19500")
    try:
        metric = last_jsonl(run / "train_metrics.jsonl")
        if metric.get("step") != FINAL_STEP or not finite_number(metric.get("validation_loss")):
            errors.append("last validation metric is not finite step 19500")
    except (OSError, ValueError) as exc:
        errors.append(f"validation metric unavailable: {exc}")
    if not (run / "loss.png").is_file():
        errors.append("final loss.png missing")
    return errors


def other_showo_training() -> list[int]:
    found: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        text = [arg.decode("utf-8", "replace") for arg in args if arg]
        candidates = [arg for arg in text if arg.endswith("/train_seen10.py")
                      or (arg.endswith("/run_csgo_seen10.sh") and "train" in text)]
        if not candidates:
            continue
        try:
            cwd = (entry / "cwd").resolve(strict=True)
            paths = [(Path(arg) if Path(arg).is_absolute() else cwd / arg).resolve()
                     for arg in candidates]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(path.is_relative_to(REPO) for path in paths):
            found.append(int(entry.name))
    return found


def gpu_occupants() -> list[str]:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        raise RuntimeError("nvidia-smi unavailable; GPU occupancy cannot be verified")
    result = subprocess.run((binary, "--query-compute-apps=pid", "--format=csv,noheader"),
                            capture_output=True, text=True, timeout=20, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    return [line.strip() for line in result.stdout.splitlines()
            if line.strip() and line.strip() not in ("[Not Supported]", "Not Supported")]


def gpu_occupancy_by_identity(
    occupants: list[str], allowed: tuple[tuple[int, int], ...]
) -> tuple[list[dict[str, int]], list[str]]:
    """Allow only live GPU PIDs whose start ticks match an explicit exception."""
    approved: list[dict[str, int]] = []
    deferred: list[str] = []
    allowed_by_pid = dict(allowed)
    for occupant in occupants:
        try:
            pid = int(occupant)
        except ValueError:
            deferred.append(occupant)
            continue
        expected_ticks = allowed_by_pid.get(pid)
        if pid <= 0 or expected_ticks is None:
            deferred.append(occupant)
            continue
        try:
            identity = process_identity(pid)
        except (OSError, RuntimeError, ValueError):
            identity = None
        if identity is None or identity[0] in ("Z", "X", "x") or identity[1] != expected_ticks:
            deferred.append(occupant)
            continue
        approved.append({"pid": pid, "start_ticks": expected_ticks})
    return approved, deferred


def parse_gpu_identity(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2 or any(not part.isascii() or not part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError("--allow-gpu-process requires PID:START_TICKS")
    pid, ticks = map(int, parts)
    if pid <= 0 or ticks <= 0:
        raise argparse.ArgumentTypeError("PID and START_TICKS must be positive")
    return pid, ticks


def showo_update_steps(output: Path) -> list[int]:
    path = output / "loss.jsonl"
    if not path.is_file():
        return []
    steps: set[int] = set()
    for line in path.read_bytes().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue  # writer may currently be appending
        if isinstance(row, dict) and type(row.get("step")) is int and row["step"] >= 1 and finite_number(row.get("loss")):
            steps.add(row["step"])
    return sorted(steps)


def write_status(state_dir: Path, state: str, **fields: object) -> dict:
    payload = {"state": state, "updated_at": time.time(), **fields}
    descriptor, temp_name = tempfile.mkstemp(prefix=".status-", dir=state_dir)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, state_dir / "status.json")
        directory_fd = os.open(state_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def run(state_dir: Path, poll_seconds: float, settings: Settings = Settings(),
        after_launch_seconds: float = 30) -> int:
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "guardian.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another guardian holds the state lock", file=sys.stderr)
            return 2
        status_path = state_dir / "status.json"
        if status_path.exists():
            try:
                previous = read_json(status_path)
            except (OSError, ValueError) as exc:
                print(f"unreadable prior status: {exc}", file=sys.stderr)
                return 2
            if previous.get("state") not in ("waiting_pi", "waiting_gpu"):
                print(f"prior {previous['state']} status forbids another launch", file=sys.stderr)
                return 2
        allowed_gpu_processes: list[dict[str, int]] = []
        while True:
            state = pi_state(settings)
            if state == "pid_reused":
                write_status(state_dir, "blocked", reason="pi PID reused; original process identity lost")
                return 2
            if state == "running":
                write_status(state_dir, "waiting_pi", pi_pid=settings.pi_pid,
                             pi_start_ticks=settings.pi_start_ticks)
                wait_for_pi_exit(settings.pi_pid, poll_seconds)
                continue
            # Check twice after exit to allow bounded checkpoint-manager cleanup.
            errors = completion_errors(settings)
            if errors:
                time.sleep(min(poll_seconds, 30))
                errors = completion_errors(settings)
            if errors:
                write_status(state_dir, "blocked", reason="pi completion evidence incomplete", errors=errors)
                return 2
            break

        while True:
            if settings.showo_log.exists():
                write_status(state_dir, "blocked", reason="Show-o2 nohup log already exists")
                return 2
            if settings.showo_output.exists() and any(settings.showo_output.iterdir()):
                write_status(state_dir, "blocked", reason="Show-o2 output is nonempty")
                return 2
            existing = other_showo_training()
            if existing:
                write_status(state_dir, "blocked", reason="Show-o2 training already running", pids=existing)
                return 2
            try:
                occupants = gpu_occupants()
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                write_status(state_dir, "blocked", reason=str(exc))
                return 2
            allowed_gpu_processes, deferred = gpu_occupancy_by_identity(
                occupants, settings.allowed_gpu_identities)
            if not deferred:
                break
            write_status(state_dir, "waiting_gpu", gpu_pids=deferred,
                         allowed_gpu_processes=allowed_gpu_processes)
            time.sleep(poll_seconds)

        # A previously approved PID may have exited or been reused while the
        # launch gate was being checked. Keep that case out of the exception.
        for approved in allowed_gpu_processes:
            try:
                identity = process_identity(approved["pid"])
            except (OSError, RuntimeError, ValueError):
                identity = None
            if identity is None or identity[0] in ("Z", "X", "x") or identity[1] != approved["start_ticks"]:
                write_status(state_dir, "blocked", reason="allowed GPU process identity changed",
                             allowed_gpu_processes=allowed_gpu_processes)
                return 2
        command = list(COMMAND)
        write_status(state_dir, "launching", command=command, cwd=str(settings.repo),
                     log=str(settings.showo_log), pi_pid=settings.pi_pid,
                     pi_start_ticks=settings.pi_start_ticks,
                     allowed_gpu_processes=allowed_gpu_processes)
        try:
            with settings.showo_log.open("x") as log:
                child = subprocess.Popen(command, cwd=settings.repo, stdout=log,
                                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         start_new_session=True, close_fds=True)
        except (OSError, ValueError) as exc:
            write_status(state_dir, "failed", reason=f"launch failed: {exc}", command=command,
                         allowed_gpu_processes=allowed_gpu_processes)
            return 1
        # The intent was durable before Popen. PID is useful audit data; a too-fast
        # exit may leave no /proc entry, in which case returncode remains authoritative.
        identity = process_identity(child.pid)
        child_ticks = identity[1] if identity else None
        write_status(state_dir, "launched", child_pid=child.pid, child_start_ticks=child_ticks,
                     command=command, cwd=str(settings.repo), log=str(settings.showo_log),
                     allowed_gpu_processes=allowed_gpu_processes)
        while True:
            exit_code = child.poll()
            if exit_code is not None:
                write_status(state_dir, "failed", reason="child exited before two optimizer updates",
                             child_pid=child.pid, child_start_ticks=child_ticks, exit_code=exit_code,
                             command=command, allowed_gpu_processes=allowed_gpu_processes)
                return 1
            steps = showo_update_steps(settings.showo_output)
            if len(steps) >= 2:
                write_status(state_dir, "healthy", child_pid=child.pid,
                             child_start_ticks=child_ticks, command=command,
                             optimizer_steps=steps[-2:],
                             allowed_gpu_processes=allowed_gpu_processes)
                return 0
            time.sleep(after_launch_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=300)
    parser.add_argument("--allow-gpu-process", action="append", type=parse_gpu_identity,
                        default=[], metavar="PID:START_TICKS",
                        help="permit a live GPU process only if PID and Linux start ticks both match")
    args = parser.parse_args()
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive and finite")
    return run(args.state_dir, args.poll_seconds,
               Settings(allowed_gpu_identities=tuple(args.allow_gpu_process)))


if __name__ == "__main__":
    sys.exit(main())
