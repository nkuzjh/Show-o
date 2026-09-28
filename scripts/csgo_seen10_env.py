#!/usr/bin/env python3
"""Prepare and audit the CSGO Seen-10 Python environment without GPU access."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


PROFILES = {
    "local": ("2.12.0+cu130", "0.27.0+cu130", "13.0", "cu130"),
    "remote": ("2.6.0+cu124", "0.21.0+cu124", "12.4", "cu124"),
}
CHECK_IMPORTS = (
    "transformers", "diffusers", "timm", "torchdiffeq", "omegaconf",
    "huggingface_hub", "safetensors", "einops", "accelerate", "peft",
    "numpy", "PIL", "matplotlib", "tqdm",
)
REQUIREMENTS_CODE = r'''
import importlib.metadata as metadata
import json, sys
try:
    from packaging.requirements import Requirement
except ModuleNotFoundError:
    # A fresh venv has pip before the CSGO requirements are installed.
    from pip._vendor.packaging.requirements import Requirement
problems = []
for raw in open(sys.argv[1], encoding="utf-8"):
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    requirement = Requirement(line)
    if requirement.marker and not requirement.marker.evaluate():
        continue
    try:
        version = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        problems.append(f"{requirement.name} is missing")
        continue
    if version not in requirement.specifier:
        problems.append(f"{requirement.name} {version} does not satisfy {requirement.specifier}")
print(json.dumps(problems))
'''
PROBE_CODE = r'''
import json, sys
result = {"python": list(sys.version_info[:3])}
try:
    import torch
    result.update(torch=str(torch.__version__), cuda=str(torch.version.cuda),
                  torch_path=str(torch.__file__))
    import torchvision
    result.update(torchvision=str(torchvision.__version__),
                  torchvision_path=str(torchvision.__file__))
except Exception as error:
    result["import_error"] = f"{type(error).__name__}: {error}"
print(json.dumps(result))
'''


class SetupError(Exception):
    pass


def run(*args: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)
    return subprocess.run(args, check=True, text=True, env=env,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)


def probe(python: Path | str) -> dict:
    try:
        return json.loads(run(str(python), "-c", PROBE_CODE, capture=True).stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError) as error:
        raise SetupError(f"Cannot inspect Python {python}: {error}") from error


def check_python(probe_result: dict, label: str) -> None:
    version = probe_result["python"]
    if version[:2] != [3, 13]:
        raise SetupError(f"{label}: Python 3.13 required for the verified CUDA wheels, found {'.'.join(map(str, version))}")


def packages_match(info: dict, profile: str, venv: Path | None = None,
                   inherited: bool = False) -> bool:
    torch, vision, cuda, _ = PROFILES[profile]
    if any(info.get(name) != value for name, value in
           (("torch", torch), ("torchvision", vision), ("cuda", cuda))):
        return False
    if venv is not None:
        for key in ("torch_path", "torchvision_path"):
            path = Path(info[key]).resolve()
            inside = path.is_relative_to(venv.resolve())
            if inherited == inside:
                return False
    return True


def cfg_inherits(venv: Path) -> bool:
    cfg = venv / "pyvenv.cfg"
    if not cfg.is_file() or cfg.is_symlink():
        raise SetupError(f"{venv} has no safe pyvenv.cfg; left untouched")
    match = re.search(r"^include-system-site-packages\s*=\s*(true|false)\s*$",
                      cfg.read_text(encoding="utf-8"), re.I | re.M)
    if match is None:
        raise SetupError(f"{cfg} has no valid include-system-site-packages entry")
    return match[1].lower() == "true"


def repairable_empty_venv(venv: Path) -> bool:
    """Conservatively recognize old venvs containing only bootstrap packages."""
    allowed = re.compile(
        r"^(pip|setuptools|wheel|pkg_resources|_distutils_hack)(-|\.|$)|"
        r"^(__pycache__|distutils-precedence\.pth)$", re.I
    )
    site_dirs = list((venv / "lib").glob("python*/site-packages"))
    if not site_dirs or len(site_dirs) != 1 or site_dirs[0].is_symlink():
        return False
    for entry in site_dirs[0].iterdir():
        if entry.is_symlink() or not allowed.match(entry.name):
            return False
    bin_dir = venv / "bin"
    allowed_bin = re.compile(r"^(python(3(\.\d+)?)?|pip(3(\.\d+)?)?|"
                             r"Activate\.ps1|activate(\.csh|\.fish)?|wheel)$")
    if not bin_dir.is_dir() or bin_dir.is_symlink():
        return False
    if any(not allowed_bin.fullmatch(p.name) for p in bin_dir.iterdir()):
        return False
    # An unexpected top-level file can contain user data; refuse automatic moves.
    gitignore = venv / ".gitignore"
    if gitignore.is_symlink():
        return False
    if gitignore.exists() and gitignore.read_text(encoding="utf-8") != (
        "# Created by venv; see https://docs.python.org/3/library/venv.html\n*\n"
    ):
        return False
    if any(p.name not in {"bin", "include", "lib", "lib64", "pyvenv.cfg", ".gitignore"}
           for p in venv.iterdir()):
        return False
    return True


def reserve_backup_name(venv: Path) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    for number in range(1000):
        candidate = venv.with_name(f"{venv.name}.incomplete-backup-{stamp}-{os.getpid()}-{number}")
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise SetupError("Cannot reserve a unique backup name for incomplete venv")


def check_imports(python: Path) -> None:
    code = "\n".join(f"import {name}" for name in CHECK_IMPORTS)
    try:
        run(str(python), "-c", code, capture=True)
    except subprocess.CalledProcessError as error:
        raise SetupError(f"Dependency import check failed: {error.stderr.strip()}") from error


def requirements_problems(python: Path, requirements: Path) -> list[str]:
    try:
        result = run(str(python), "-c", REQUIREMENTS_CODE, str(requirements), capture=True)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise SetupError(f"Cannot inspect installed requirements: {error}") from error


def check_cuda(python: Path) -> None:
    code = "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; " \
           "x=torch.ones(1, device='cuda'); assert x.item()==1; " \
           "print(torch.cuda.get_device_name(0))"
    try:
        run(str(python), "-c", code)
    except subprocess.CalledProcessError as error:
        raise SetupError(f"CUDA smoke check failed (run on a GPU compute node): {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--mode", choices=("full", "env-only", "assets-only", "check", "check-cuda"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repair-incomplete-venv", action="store_true",
                        help="Compatibility option; full and env-only already repair safe bootstrap-only venvs")
    args = parser.parse_args()
    if args.repair_incomplete_venv and args.mode in ("assets-only", "check", "check-cuda"):
        parser.error("--repair-incomplete-venv requires full or --env-only")
    project = args.project_dir
    venv = project / ".venv"
    python = venv / "bin/python"
    requirements = project / "requirements-csgo-seen10.txt"
    constraints = project / f"constraints-csgo-seen10-{args.profile}.txt"
    if not requirements.is_file() or not constraints.is_file():
        raise SetupError("CSGO requirements or profile constraints file is missing")

    host = probe(sys.executable)
    check_python(host, "Selected Python")
    existing = venv.exists() or venv.is_symlink()
    inherited = False
    info = None
    needs_repair = False
    if existing:
        if venv.is_symlink() or not venv.is_dir() or not python.is_file():
            raise SetupError(f"{venv} is not a valid venv directory; left untouched")
        inherited = cfg_inherits(venv)
        info = probe(python)
        check_python(info, "Existing venv")
        if info.get("import_error"):
            if inherited and "torch" not in info and repairable_empty_venv(venv):
                needs_repair = True
            elif not inherited and "torch" not in info and args.mode in ("full", "env-only"):
                # A new isolated environment interrupted during installation can resume.
                print(f"Resuming isolated venv without torch: {venv}")
            else:
                raise SetupError(f"Existing venv import failed: {info['import_error']}; left untouched")
        elif args.profile == "remote" and inherited:
            raise SetupError("Remote profile requires an isolated venv; existing inherited venv was left untouched")
        elif not packages_match(info, args.profile, venv, inherited):
            raise SetupError(f"Existing venv torch/torchvision/CUDA does not match {args.profile} profile; left untouched: {info}")
        else:
            print(f"Verified existing {args.profile} venv: {venv}")
    elif args.mode in ("check", "check-cuda", "assets-only") and not args.dry_run:
        raise SetupError(f"Missing venv: {venv}; run setup first")

    if args.dry_run:
        if needs_repair:
            if args.mode in ("full", "env-only"):
                print(f"DRY RUN: would back up incomplete venv {venv} and create an isolated venv")
            else:
                print(f"DRY RUN: incomplete venv {venv}; full or --env-only setup is required before {args.mode}")
        elif not existing:
            inherit_host = args.profile == "local" and packages_match(host, "local")
            print(f"DRY RUN: would create {'inherited' if inherit_host else 'isolated'} {args.profile} venv at {venv}")
        else:
            print(f"DRY RUN: would reuse {venv}")
        torch, vision, _, channel = PROFILES[args.profile]
        print(f"DRY RUN: target torch=={torch}, torchvision=={vision}, "
              f"index https://download.pytorch.org/whl/{channel}, constraints {constraints}")
        if args.mode in ("check", "check-cuda"):
            print(f"DRY RUN: would run CPU dependency check{' and explicit CUDA smoke check' if args.mode == 'check-cuda' else ''}")
        elif args.mode != "assets-only":
            print("DRY RUN: would install missing requirements only")
        return

    if args.mode in ("check", "check-cuda"):
        if needs_repair:
            raise SetupError("Cannot check an incomplete venv; run full or --env-only setup")
        if not existing or info is None or info.get("import_error"):
            raise SetupError("Cannot check an incomplete or missing venv")
        problems = requirements_problems(python, requirements)
        if problems:
            raise SetupError("Requirements mismatch: " + "; ".join(problems))
        check_imports(python)
        if args.mode == "check-cuda":
            check_cuda(python)
        print(f"{args.profile} {args.mode} passed")
        return

    if args.mode == "assets-only":
        if needs_repair:
            raise SetupError("Cannot prepare assets with an incomplete venv; run full or --env-only setup")
        problems = requirements_problems(python, requirements)
        if problems:
            raise SetupError("Requirements mismatch: " + "; ".join(problems))
        check_imports(python)
        return

    if needs_repair:
        backup = reserve_backup_name(venv)
        venv.rename(backup)
        print(f"Backed up incomplete venv to {backup}")
        existing = False
        inherited = False
        info = None

    if not existing:
        inherit_host = not needs_repair and args.profile == "local" and packages_match(host, "local")
        run(sys.executable, "-m", "venv", *( ["--system-site-packages"] if inherit_host else [] ), str(venv))
        inherited = inherit_host
        print(f"Created {'inherited' if inherited else 'isolated'} {args.profile} venv: {venv}")
        if inherited:
            info = probe(python)

    if not inherited and (info is None or info.get("import_error")):
        torch, vision, _, channel = PROFILES[args.profile]
        run(str(python), "-m", "pip", "install", "--index-url",
            f"https://download.pytorch.org/whl/{channel}",
            f"torch=={torch}", f"torchvision=={vision}")
    installed = probe(python)
    if not packages_match(installed, args.profile, venv, inherited):
        raise SetupError(f"Profile packages missing or mismatched before requirements installation: {installed}")
    problems = requirements_problems(python, requirements)
    if problems:
        print("Installing missing or mismatched requirements: " + "; ".join(problems))
        run(str(python), "-m", "pip", "install", "--upgrade-strategy", "only-if-needed",
            "--constraint", str(constraints), "--requirement", str(requirements))
    else:
        print("All CSGO requirements already satisfy their constraints; skipping pip")
    installed = probe(python)
    if not packages_match(installed, args.profile, venv, inherited):
        raise SetupError(f"Profile packages changed after requirements installation: {installed}")
    problems = requirements_problems(python, requirements)
    if problems:
        raise SetupError("Requirements still mismatch after installation: " + "; ".join(problems))
    check_imports(python)
    print(f"{args.profile} environment ready: {venv}")


if __name__ == "__main__":
    try:
        main()
    except (SetupError, subprocess.CalledProcessError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
