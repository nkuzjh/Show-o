"""CPU-only evidence and single-launch checks for the operational guardian."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import wait_pi05_then_showo2 as guard  # noqa: E402


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def complete(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    run = tmp_path / "pi_run"
    checkpoint = run / "checkpoints/19500"
    config = {"experiment_profile": guard.PROFILE, "smoke_only": False,
              "runtime": {"experiment_profile": guard.PROFILE, "seed": 0,
                          "num_train_steps": 19500, "smoke_only": False}}
    put_json(run / "run_config.json", config)
    put_json(run / "loss.jsonl", {"step": 19500, "loss": 0.012})
    put_json(run / "train_metrics.jsonl", {"step": 19500, "validation_loss": 0.02})
    put_json(checkpoint / "checkpoint_identity.json",
             {"step": 19500, "experiment_profile": guard.PROFILE, "seed": 0, "smoke_only": False})
    put_json(checkpoint / "_CHECKPOINT_METADATA", {"commit_timestamp_nsecs": 123})
    for name in ("params/manifest.ocdbt", "train_state/manifest.ocdbt"):
        path = checkpoint / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("manifest")
    (run / "checkpoints/late").symlink_to("19500")
    (run / "loss.png").write_bytes(b"png")
    settings = guard.Settings(repo=repo, pi_run=run, pi_pid=1234, pi_start_ticks=5678,
                              showo_output=repo / "outputs/seed_42",
                              showo_log=repo / "showo2_aligned.nohup.out")
    assert guard.completion_errors(settings) == []
    return settings


def test_rejects_incomplete_completion(complete, monkeypatch, tmp_path):
    (complete.pi_run / "loss.png").unlink()
    monkeypatch.setattr(guard, "pi_state", lambda _: "exited")
    monkeypatch.setattr(guard.time, "sleep", lambda _: None)
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **kw: pytest.fail("launched"))
    state = tmp_path / "state"
    assert guard.run(state, 1, complete) == 2
    assert guard.read_json(state / "status.json")["state"] == "blocked"


@pytest.mark.parametrize("mutate", [
    lambda s: put_json(s.pi_run / "run_config.json",
                       {"experiment_profile": guard.PROFILE, "smoke_only": False,
                        "runtime": {"experiment_profile": guard.PROFILE, "seed": 1,
                                    "num_train_steps": 19500, "smoke_only": False}}),
    lambda s: put_json(s.pi_run / "checkpoints/19500/checkpoint_identity.json",
                       {"step": 19500, "experiment_profile": "wrong", "seed": 0,
                        "smoke_only": False}),
    lambda s: put_json(s.pi_run / "checkpoints/19500/_CHECKPOINT_METADATA",
                       {"commit_timestamp_nsecs": 0}),
    lambda s: put_json(s.pi_run / "loss.jsonl", {"step": 19500, "loss": float("nan")}),
])
def test_rejects_wrong_metadata(complete, mutate):
    mutate(complete)
    assert guard.completion_errors(complete)


def test_pid_reuse_and_zombie(complete, monkeypatch):
    monkeypatch.setattr(guard, "process_identity", lambda _: ("S", 9999))
    assert guard.pi_state(complete) == "pid_reused"
    monkeypatch.setattr(guard, "process_identity", lambda _: ("Z", 5678))
    assert guard.pi_state(complete) == "exited"


def test_existing_log_or_output_blocks_launch(complete, monkeypatch, tmp_path):
    monkeypatch.setattr(guard, "pi_state", lambda _: "exited")
    monkeypatch.setattr(guard, "other_showo_training", lambda: [])
    monkeypatch.setattr(guard, "gpu_occupants", lambda: [])
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **kw: pytest.fail("launched"))
    complete.showo_log.write_text("existing log")
    assert guard.run(tmp_path / "log_state", 1, complete) == 2
    complete.showo_log.unlink()
    complete.showo_output.mkdir(parents=True)
    (complete.showo_output / "checkpoint").write_text("existing")
    assert guard.run(tmp_path / "output_state", 1, complete) == 2


def test_duplicate_status_blocks_restart(complete, monkeypatch, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    guard.write_status(state, "launching", command=list(guard.COMMAND))
    monkeypatch.setattr(guard, "pi_state", lambda _: pytest.fail("checked pi after intent"))
    assert guard.run(state, 1, complete) == 2


def test_no_gpu_check_while_pi_is_alive(complete, monkeypatch, tmp_path):
    states = iter(("running", "exited"))
    monkeypatch.setattr(guard, "pi_state", lambda _: next(states))
    monkeypatch.setattr(guard.time, "sleep", lambda _: None)
    monkeypatch.setattr(guard, "wait_for_pi_exit", lambda *args: None)
    monkeypatch.setattr(guard, "other_showo_training", lambda: [])
    state = tmp_path / "state"
    # Reach the GPU gate only after one waiting status was durably recorded.
    def verify_gate():
        assert guard.read_json(state / "status.json")["state"] == "waiting_pi"
        return []
    monkeypatch.setattr(guard, "gpu_occupants", verify_gate)

    class Child:
        pid = 4321

        def poll(self):
            return 1

    monkeypatch.setattr(guard, "process_identity", lambda _: ("S", 111))
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **kw: Child())
    assert guard.run(state, 1, complete) == 1
    assert guard.read_json(state / "status.json")["exit_code"] == 1


def test_launch_command_redirection_and_healthy_updates(complete, monkeypatch, tmp_path):
    monkeypatch.setattr(guard, "pi_state", lambda _: "exited")
    monkeypatch.setattr(guard, "other_showo_training", lambda: [])
    monkeypatch.setattr(guard, "gpu_occupants", lambda: [])
    monkeypatch.setattr(guard, "process_identity", lambda _: ("S", 111))
    seen = {}

    class Child:
        pid = 4321

        def poll(self):
            return None

    def launch(args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        assert guard.read_json(tmp_path / "state/status.json")["state"] == "launching"
        complete.showo_output.mkdir(parents=True)
        (complete.showo_output / "loss.jsonl").write_text(
            '{"step":1,"loss":0.3}\n{"step":2,"loss":0.2}\n')
        return Child()

    monkeypatch.setattr(guard.subprocess, "Popen", launch)
    assert guard.run(tmp_path / "state", 1, complete, after_launch_seconds=0.01) == 0
    assert seen["args"] == list(guard.COMMAND)
    assert seen["kwargs"]["cwd"] == complete.repo
    assert seen["kwargs"]["stderr"] == guard.subprocess.STDOUT
    assert seen["kwargs"]["stdin"] == guard.subprocess.DEVNULL
    assert seen["kwargs"]["start_new_session"] is True
    assert seen["kwargs"]["close_fds"] is True
    assert seen["kwargs"]["stdout"].name == str(complete.showo_log)
    assert guard.read_json(tmp_path / "state/status.json")["optimizer_steps"] == [1, 2]
    assert guard.run(tmp_path / "state", 1, complete) == 2


def test_pidfd_wait_wakes_on_exit_and_closes_handle(monkeypatch):
    calls = []
    monkeypatch.setattr(guard.os, "pidfd_open", lambda pid: calls.append(("open", pid)) or 99, raising=False)
    monkeypatch.setattr(guard.os, "close", lambda fd: calls.append(("close", fd)))

    class Poller:
        def register(self, fd, event):
            calls.append(("register", fd, event))

        def poll(self, milliseconds):
            calls.append(("poll", milliseconds))
            return [(99, guard.select.POLLIN)]

    monkeypatch.setattr(guard.select, "poll", Poller)
    guard.wait_for_pi_exit(1234, 1080)
    assert calls == [("open", 1234), ("register", 99, guard.select.POLLIN),
                     ("poll", 1080000), ("close", 99)]


def test_pidfd_wait_already_exited_returns_without_sleep(monkeypatch):
    def gone(pid):
        raise ProcessLookupError(pid)

    monkeypatch.setattr(guard.os, "pidfd_open", gone, raising=False)
    monkeypatch.setattr(guard.time, "sleep", lambda _: pytest.fail("unnecessary sleep"))
    guard.wait_for_pi_exit(1234, 1080)


def test_pidfd_unavailable_uses_bounded_poll_interval(monkeypatch):
    monkeypatch.delattr(guard.os, "pidfd_open", raising=False)
    sleeps = []
    monkeypatch.setattr(guard.time, "sleep", sleeps.append)
    guard.wait_for_pi_exit(1234, 1080)
    assert sleeps == [1080]


def test_allowed_gpu_identity_permits_launch_and_is_audited(complete, monkeypatch, tmp_path):
    settings = replace(complete, allowed_gpu_identities=((100, 777),))
    monkeypatch.setattr(guard, "pi_state", lambda _: "exited")
    monkeypatch.setattr(guard, "other_showo_training", lambda: [])
    monkeypatch.setattr(guard, "gpu_occupants", lambda: ["100"])
    monkeypatch.setattr(guard, "process_identity",
                        lambda pid: ("S", 777) if pid == 100 else ("S", 111))
    seen = {}

    class Child:
        pid = 4321

        def poll(self):
            return 1

    def launch(args, **kwargs):
        seen["intent"] = guard.read_json(tmp_path / "state/status.json")
        return Child()

    monkeypatch.setattr(guard.subprocess, "Popen", launch)
    assert guard.run(tmp_path / "state", 1, settings) == 1
    approved = [{"pid": 100, "start_ticks": 777}]
    assert seen["intent"]["state"] == "launching"
    assert seen["intent"]["allowed_gpu_processes"] == approved
    assert guard.read_json(tmp_path / "state/status.json")["allowed_gpu_processes"] == approved


def test_other_gpu_occupant_still_defers(complete, monkeypatch, tmp_path):
    settings = replace(complete, allowed_gpu_identities=((100, 777),))
    monkeypatch.setattr(guard, "pi_state", lambda _: "exited")
    monkeypatch.setattr(guard, "other_showo_training", lambda: [])
    monkeypatch.setattr(guard, "gpu_occupants", lambda: ["100", "200"])
    monkeypatch.setattr(guard, "process_identity", lambda pid: ("S", 777))
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **kw: pytest.fail("launched"))

    class EndWait(Exception):
        pass

    monkeypatch.setattr(guard.time, "sleep", lambda _: (_ for _ in ()).throw(EndWait))
    with pytest.raises(EndWait):
        guard.run(tmp_path / "state", 1, settings)
    status = guard.read_json(tmp_path / "state/status.json")
    assert status["state"] == "waiting_gpu"
    assert status["gpu_pids"] == ["200"]
    assert status["allowed_gpu_processes"] == [{"pid": 100, "start_ticks": 777}]


@pytest.mark.parametrize("identity", [None, ("S", 778), ("Z", 777)])
def test_allowed_pid_missing_reused_or_zombie_still_defers(
    complete, monkeypatch, tmp_path, identity
):
    settings = replace(complete, allowed_gpu_identities=((100, 777),))
    monkeypatch.setattr(guard, "pi_state", lambda _: "exited")
    monkeypatch.setattr(guard, "other_showo_training", lambda: [])
    monkeypatch.setattr(guard, "gpu_occupants", lambda: ["100"])
    monkeypatch.setattr(guard, "process_identity", lambda pid: identity)
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **kw: pytest.fail("launched"))

    class EndWait(Exception):
        pass

    monkeypatch.setattr(guard.time, "sleep", lambda _: (_ for _ in ()).throw(EndWait))
    with pytest.raises(EndWait):
        guard.run(tmp_path / "state", 1, settings)
    status = guard.read_json(tmp_path / "state/status.json")
    assert status["state"] == "waiting_gpu"
    assert status["gpu_pids"] == ["100"]
    assert status["allowed_gpu_processes"] == []


def test_allowed_pid_reused_between_gpu_check_and_launch_blocks(
    complete, monkeypatch, tmp_path
):
    settings = replace(complete, allowed_gpu_identities=((100, 777),))
    monkeypatch.setattr(guard, "pi_state", lambda _: "exited")
    monkeypatch.setattr(guard, "other_showo_training", lambda: [])
    monkeypatch.setattr(guard, "gpu_occupants", lambda: ["100"])
    identities = iter((("S", 777), ("S", 778)))
    monkeypatch.setattr(guard, "process_identity", lambda pid: next(identities))
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **kw: pytest.fail("launched"))
    assert guard.run(tmp_path / "state", 1, settings) == 2
    status = guard.read_json(tmp_path / "state/status.json")
    assert status["state"] == "blocked"
    assert "identity changed" in status["reason"]
    assert not settings.showo_log.exists()


@pytest.mark.parametrize("value", ["0:1", "1:0", "1", "a:2", "1:2:3", "²:3"])
def test_allow_gpu_process_requires_positive_decimal_identity(value):
    with pytest.raises(guard.argparse.ArgumentTypeError):
        guard.parse_gpu_identity(value)
    assert guard.parse_gpu_identity("100:777") == (100, 777)
