"""CPU-only setup tests; all venv and pip operations are intercepted."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/csgo_seen10_env.py"
spec = importlib.util.spec_from_file_location("csgo_seen10_env", SCRIPT)
assert spec and spec.loader
env_setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(env_setup)


def project(tmp_path: Path) -> Path:
    root = tmp_path / "space in project" / "show-o2"
    root.mkdir(parents=True)
    (root / "requirements-csgo-seen10.txt").write_text("timm==1.0.12\n")
    for name in ("local", "remote"):
        (root / f"constraints-csgo-seen10-{name}.txt").write_text("torch==2.6.0\n")
    return root


def make_venv(root: Path, inherited: bool, extra: str | None = None) -> Path:
    venv = root / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").write_text("fake")
    site = venv / "lib/python3.13/site-packages"
    site.mkdir(parents=True)
    if extra:
        (site / extra).write_text("user data")
    (venv / "pyvenv.cfg").write_text(
        f"include-system-site-packages = {'true' if inherited else 'false'}\n"
    )
    return venv


def info(profile: str, root: Path, inherited: bool) -> dict:
    torch, vision, cuda, _ = env_setup.PROFILES[profile]
    base = root.parent / "host" if inherited else root / ".venv/lib/python3.13/site-packages"
    return {"python": [3, 13, 5], "torch": torch, "torchvision": vision,
            "cuda": cuda, "torch_path": str(base / "torch/__init__.py"),
            "torchvision_path": str(base / "torchvision/__init__.py")}


def invoke(monkeypatch: pytest.MonkeyPatch, root: Path, profile: str, *options: str,
           mode: str = "env-only") -> None:
    monkeypatch.setattr(sys, "argv", ["env", "--project-dir", str(root),
                                      "--profile", profile, "--mode", mode, *options])
    env_setup.main()


def test_remote_fresh_installs_exact_wheels_and_constrained_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path)
    calls: list[tuple[str, ...]] = []
    installed = False
    deps_installed = False

    def fake_run(*args: str, **_kwargs):
        nonlocal installed, deps_installed
        calls.append(tuple(args))
        if args[1:3] == ("-m", "venv"):
            make_venv(root, False)
        if args[1:3] == ("-m", "pip") and "torch==2.6.0+cu124" in args:
            installed = True
        if "--requirement" in args:
            deps_installed = True
        return subprocess.CompletedProcess(args, 0, "", "")

    def fake_probe(python: Path | str) -> dict:
        if str(python) == sys.executable:
            return {"python": [3, 13, 5], "import_error": "No module named torch"}
        if installed:
            return info("remote", root, False)
        return {"python": [3, 13, 5], "import_error": "No module named torch"}

    monkeypatch.setattr(env_setup, "run", fake_run)
    monkeypatch.setattr(env_setup, "probe", fake_probe)
    monkeypatch.setattr(env_setup, "check_imports", lambda _python: None)
    monkeypatch.setattr(env_setup, "requirements_problems", lambda _python, _req: [] if deps_installed else ["timm is missing"])
    invoke(monkeypatch, root, "remote")
    assert ("--index-url", "https://download.pytorch.org/whl/cu124") == next(
        (call[4], call[5]) for call in calls if "torch==2.6.0+cu124" in call
    )
    assert any("torchvision==0.21.0+cu124" in call for call in calls)
    assert any("--constraint" in call and str(root / "constraints-csgo-seen10-remote.txt") in call
               for call in calls)
    assert not any("--system-site-packages" in call for call in calls)
    completed_calls = len(calls)
    invoke(monkeypatch, root, "remote")
    assert len(calls) == completed_calls


def test_local_existing_inherited_venv_is_accepted_without_wheel_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path)
    make_venv(root, True)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(env_setup, "probe", lambda _python: info("local", root, True))
    monkeypatch.setattr(env_setup, "check_imports", lambda _python: None)
    monkeypatch.setattr(env_setup, "requirements_problems", lambda _python, _req: [])
    monkeypatch.setattr(env_setup, "run", lambda *args, **kwargs:
                        calls.append(args) or subprocess.CompletedProcess(args, 0, "", ""))
    invoke(monkeypatch, root, "local")
    assert not calls
    assert not any("--index-url" in call for call in calls)


@pytest.mark.parametrize("profile", ["local", "remote"])
@pytest.mark.parametrize("mode", ["full", "env-only"])
@pytest.mark.parametrize("explicit_flag", [False, True])
def test_empty_inherited_recovers_automatically_and_keeps_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str, mode: str,
    explicit_flag: bool,
) -> None:
    root = project(tmp_path)
    venv = make_venv(root, True)
    (venv / ".gitignore").write_text(
        "# Created by venv; see https://docs.python.org/3/library/venv.html\n*\n"
    )
    installed = False
    calls: list[tuple[str, ...]] = []

    def fake_probe(python: Path | str) -> dict:
        if str(python) == sys.executable:
            return {"python": [3, 13, 5], "import_error": "No module named torch"}
        if installed:
            return info(profile, root, False)
        return {"python": [3, 13, 5], "import_error": "No module named torch"}

    def fake_run(*args: str, **_kwargs):
        nonlocal installed
        calls.append(args)
        if args[1:3] == ("-m", "venv"):
            make_venv(root, False)
        if f"torch=={env_setup.PROFILES[profile][0]}" in args:
            installed = True
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(env_setup, "probe", fake_probe)
    monkeypatch.setattr(env_setup, "run", fake_run)
    monkeypatch.setattr(env_setup, "check_imports", lambda _python: None)
    monkeypatch.setattr(env_setup, "requirements_problems", lambda _python, _req: [])
    options = ("--repair-incomplete-venv",) if explicit_flag else ()
    invoke(monkeypatch, root, profile, *options, mode=mode)
    backups = list(root.glob(".venv.incomplete-backup-*"))
    assert len(backups) == 1 and (backups[0] / "pyvenv.cfg").is_file()
    assert "false" in (venv / "pyvenv.cfg").read_text()
    assert any(args[1:3] == ("-m", "venv") and "--system-site-packages" not in args
               for args in calls)
    assert any(f"torch=={env_setup.PROFILES[profile][0]}" in args for args in calls)


@pytest.mark.parametrize("extra", ["torch", "user-package.dist-info", "data.txt"])
def test_repair_rejects_nonempty_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: str,
) -> None:
    root = project(tmp_path)
    venv = make_venv(root, True, extra)
    monkeypatch.setattr(env_setup, "probe", lambda _python:
                        {"python": [3, 13, 5], "import_error": "No module named torch"})
    with pytest.raises(env_setup.SetupError, match="left untouched"):
        invoke(monkeypatch, root, "remote")
    assert venv.exists() and not list(root.glob(".venv.incomplete-backup-*"))


def test_symlink_venv_is_left_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = project(tmp_path)
    target = root / "other-environment"
    target.mkdir()
    (root / ".venv").symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(env_setup, "probe", lambda _python: {"python": [3, 13, 5]})
    with pytest.raises(env_setup.SetupError, match="left untouched"):
        invoke(monkeypatch, root, "remote")
    assert (root / ".venv").is_symlink() and target.exists()


@pytest.mark.parametrize("mode", ["check", "check-cuda", "assets-only", "full", "env-only"])
def test_incomplete_venv_readonly_and_dry_run_never_back_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    root = project(tmp_path)
    venv = make_venv(root, True)

    def fake_probe(_python: Path | str) -> dict:
        return {"python": [3, 13, 5], "import_error": "No module named torch"}

    monkeypatch.setattr(env_setup, "probe", fake_probe)
    monkeypatch.setattr(env_setup, "run", lambda *args, **kwargs:
                        pytest.fail(f"Unexpected mutation: {args}"))
    monkeypatch.setattr(env_setup, "check_cuda", lambda _python: pytest.fail("GPU used"))
    if mode == "assets-only":
        monkeypatch.setattr(env_setup, "requirements_problems", lambda _python, _req:
                            pytest.fail("Incomplete assets-only setup should stop before dependency checks"))
    if mode in ("full", "env-only"):
        invoke(monkeypatch, root, "remote", "--dry-run", mode=mode)
    else:
        with pytest.raises(env_setup.SetupError):
            invoke(monkeypatch, root, "remote", mode=mode)
    assert venv.exists()
    assert "true" in (venv / "pyvenv.cfg").read_text()
    assert not list(root.glob(".venv.incomplete-backup-*"))


@pytest.mark.parametrize("mode", ["check", "check-cuda", "assets-only"])
def test_incomplete_readonly_dry_run_does_not_claim_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    mode: str,
) -> None:
    root = project(tmp_path)
    venv = make_venv(root, True)
    monkeypatch.setattr(env_setup, "probe", lambda _python:
                        {"python": [3, 13, 5], "import_error": "No module named torch"})
    monkeypatch.setattr(env_setup, "run", lambda *args, **kwargs: pytest.fail("pip or venv used"))
    monkeypatch.setattr(env_setup, "check_cuda", lambda _python: pytest.fail("GPU used"))
    invoke(monkeypatch, root, "remote", "--dry-run", mode=mode)
    output = capsys.readouterr().out
    assert "full or --env-only setup is required" in output
    assert "would back up" not in output
    assert venv.exists() and not list(root.glob(".venv.incomplete-backup-*"))


def test_top_level_user_data_blocks_automatic_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path)
    venv = make_venv(root, True)
    (venv / "my-notes.txt").write_text("keep me")
    monkeypatch.setattr(env_setup, "probe", lambda _python:
                        {"python": [3, 13, 5], "import_error": "No module named torch"})
    monkeypatch.setattr(env_setup, "run", lambda *args, **kwargs: pytest.fail("pip or venv used"))
    with pytest.raises(env_setup.SetupError, match="left untouched"):
        invoke(monkeypatch, root, "remote", mode="full")
    assert (venv / "my-notes.txt").read_text() == "keep me"
    assert not list(root.glob(".venv.incomplete-backup-*"))


def test_profile_mismatch_rejected_and_dry_run_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path)
    monkeypatch.setattr(env_setup, "probe", lambda _python: info("local", root, True))
    monkeypatch.setattr(env_setup, "run", lambda *args, **kwargs:
                        pytest.fail(f"Unexpected write operation: {args}"))
    invoke(monkeypatch, root, "remote", "--dry-run")
    assert not (root / ".venv").exists()
    make_venv(root, True)
    with pytest.raises(env_setup.SetupError, match="Remote profile requires"):
        invoke(monkeypatch, root, "remote")


def test_wrong_torch_in_isolated_venv_is_not_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = project(tmp_path)
    make_venv(root, False)
    bad = info("local", root, False)
    monkeypatch.setattr(env_setup, "probe", lambda _python: bad)
    monkeypatch.setattr(env_setup, "run", lambda *args, **kwargs: pytest.fail("pip used"))
    with pytest.raises(env_setup.SetupError, match="does not match remote profile"):
        invoke(monkeypatch, root, "remote")


def test_assets_only_checks_existing_env_without_pip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = project(tmp_path)
    make_venv(root, True)
    monkeypatch.setattr(env_setup, "probe", lambda _python: info("local", root, True))
    monkeypatch.setattr(env_setup, "requirements_problems", lambda _python, _req: [])
    monkeypatch.setattr(env_setup, "check_imports", lambda _python: None)
    monkeypatch.setattr(env_setup, "run", lambda *args, **kwargs: pytest.fail("pip or venv used"))
    monkeypatch.setattr(sys, "argv", ["env", "--project-dir", str(root), "--profile", "local",
                                      "--mode", "assets-only"])
    env_setup.main()


@pytest.mark.parametrize("version", [[3, 12, 10], [3, 14, 0]])
def test_unsupported_python_fails_early(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                        version: list[int]) -> None:
    root = project(tmp_path)
    monkeypatch.setattr(env_setup, "probe", lambda _python: {"python": version})
    monkeypatch.setattr(env_setup, "run", lambda *args, **kwargs: pytest.fail("venv used"))
    with pytest.raises(env_setup.SetupError, match="Python 3.13 required"):
        invoke(monkeypatch, root, "remote", "--dry-run")


def test_cuda_dry_run_does_not_probe_gpu_or_mutate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = project(tmp_path)
    monkeypatch.setattr(env_setup, "probe", lambda _python:
                        {"python": [3, 13, 5], "import_error": "No module named torch"})
    monkeypatch.setattr(env_setup, "run", lambda *args, **kwargs:
                        pytest.fail(f"Unexpected subprocess: {args}"))
    monkeypatch.setattr(env_setup, "check_cuda", lambda _python: pytest.fail("GPU used"))
    monkeypatch.setattr(sys, "argv", ["env", "--project-dir", str(root), "--profile", "remote",
                                      "--mode", "check-cuda", "--dry-run"])
    env_setup.main()
    assert not (root / ".venv").exists()


def test_check_rejects_requirements_mismatch_without_pip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = project(tmp_path)
    make_venv(root, True)
    monkeypatch.setattr(env_setup, "probe", lambda _python: info("local", root, True))
    monkeypatch.setattr(env_setup, "requirements_problems", lambda _python, _req: ["timm is missing"])
    monkeypatch.setattr(env_setup, "run", lambda *args, **kwargs: pytest.fail("pip used"))
    monkeypatch.setattr(sys, "argv", ["env", "--project-dir", str(root), "--profile", "local",
                                      "--mode", "check"])
    with pytest.raises(env_setup.SetupError, match="timm is missing"):
        env_setup.main()


def test_requirements_inspector_works_without_standalone_packaging(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("pip>=1\n")
    code = """
import builtins, sys
original = builtins.__import__
def no_standalone_packaging(name, *args, **kwargs):
    if name == 'packaging' or name.startswith('packaging.'):
        raise ModuleNotFoundError('No module named packaging')
    return original(name, *args, **kwargs)
builtins.__import__ = no_standalone_packaging
exec(%r)
""" % env_setup.REQUIREMENTS_CODE
    result = subprocess.run([sys.executable, "-c", code, str(requirements)],
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_shell_entrypoints_handle_spaces_cwd_and_conflicts(tmp_path: Path) -> None:
    root = tmp_path / "copy with space"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in ("setup_csgo_seen10.sh", "setup_csgo_seen10_remote.sh", "csgo_seen10_env.py"):
        source = SCRIPT.parent / name
        target = scripts / name
        target.write_bytes(source.read_bytes())
        target.chmod(0o755)
    project_dir = root / "show-o2"
    project_dir.mkdir()
    (project_dir / "requirements-csgo-seen10.txt").write_text("test\n")
    (project_dir / "constraints-csgo-seen10-remote.txt").write_text("test\n")
    result = subprocess.run([str(scripts / "setup_csgo_seen10_remote.sh"), "--dry-run", "--env-only"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "isolated remote venv" in result.stdout
    assert not (project_dir / ".venv").exists()
    conflict = subprocess.run([str(scripts / "setup_csgo_seen10_remote.sh"),
                               "--check", "--env-only"], cwd=tmp_path,
                              capture_output=True, text=True)
    assert conflict.returncode == 2 and "Conflicting mode" in conflict.stderr


@pytest.mark.parametrize("profile", ["local", "remote"])
def test_host_wrapper_no_args_dispatches_full_setup(tmp_path: Path, profile: str) -> None:
    root = tmp_path / "copy with space"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in ("setup_csgo_seen10.sh", f"setup_csgo_seen10_{profile}.sh"):
        target = scripts / name
        target.write_bytes((SCRIPT.parent / name).read_bytes())
        target.chmod(0o755)
    project_dir = root / "show-o2"
    project_dir.mkdir()
    (project_dir / "requirements-csgo-seen10.txt").write_text("test\n")
    captured = root / "args.txt"
    python_stub = root / "fake-python"
    python_stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\nexit 37\n')
    python_stub.chmod(0o755)
    run_env = os.environ.copy()
    run_env.update(PYTHON_BIN=str(python_stub), CAPTURE_ARGS=str(captured))
    result = subprocess.run([str(scripts / f"setup_csgo_seen10_{profile}.sh")],
                            cwd=tmp_path, env=run_env, capture_output=True, text=True)
    assert result.returncode == 37
    args = captured.read_text().splitlines()
    assert args == [str(scripts / "csgo_seen10_env.py"), "--project-dir", str(project_dir),
                    "--profile", profile, "--mode", "full"]
