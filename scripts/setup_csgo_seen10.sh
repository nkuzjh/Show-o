#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SHOWO2_DIR="${REPO_ROOT}/show-o2"
CHECKPOINT_DIR="${SHOWO2_DIR}/checkpoints"
SHOWO_MODEL_DIR="${CHECKPOINT_DIR}/show-o2-1.5B"
CACHE_DIR="${SHOWO2_DIR}/.hf_cache"
VENV_DIR="${SHOWO2_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIREMENTS_FILE="${SHOWO2_DIR}/requirements-csgo-seen10.txt"

SHOWO_REVISION="07ec16589d4fc5422a74dddbbc4b2cd11e551039"
SHOWO_WEIGHT_URL="https://huggingface.co/showlab/show-o2-1.5B/resolve/${SHOWO_REVISION}/pytorch_model.bin"
SHOWO_WEIGHT_PATH="${SHOWO_MODEL_DIR}/pytorch_model.bin"
SHOWO_WEIGHT_SIZE="5661862314"
SHOWO_WEIGHT_SHA256="a596cbc305c1df987c125d4f218e78f39b681621904cccfb2a3bf0ca0327f92c"

WAN_VAE_URL="https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/Wan2.1_VAE.pth"
WAN_VAE_PATH="${CHECKPOINT_DIR}/Wan2.1_VAE.pth"
WAN_VAE_SIZE="507609880"
WAN_VAE_SHA256="38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

verify_file_or_die() {
    local path="$1"
    local expected_size="$2"
    local expected_sha256="$3"
    local label="$4"
    local actual_size
    local actual_sha256

    [[ -f "$path" ]] || die "${label}: expected a regular file at ${path}; existing path was left untouched."
    actual_size="$(stat -c '%s' -- "$path")" || die "${label}: cannot read file size for ${path}."
    [[ "$actual_size" == "$expected_size" ]] || die \
        "${label}: size mismatch at ${path} (expected ${expected_size}, got ${actual_size}); file was left untouched."
    actual_sha256="$(sha256sum -- "$path" | cut -d ' ' -f 1)" || die \
        "${label}: cannot calculate SHA-256 for ${path}."
    [[ "$actual_sha256" == "$expected_sha256" ]] || die \
        "${label}: SHA-256 mismatch at ${path} (expected ${expected_sha256}, got ${actual_sha256}); file was left untouched."
}

check_existing_asset() {
    local path="$1"
    local expected_size="$2"
    local expected_sha256="$3"
    local label="$4"

    if [[ -e "$path" || -L "$path" ]]; then
        verify_file_or_die "$path" "$expected_size" "$expected_sha256" "$label"
        printf 'Verified existing %s: %s\n' "$label" "$path"
    fi
}

download_verified_asset() {
    local url="$1"
    local destination="$2"
    local expected_size="$3"
    local expected_sha256="$4"
    local label="$5"
    local partial_path="${destination}.part"
    local partial_size
    local downloader_succeeded=0

    check_existing_asset "$destination" "$expected_size" "$expected_sha256" "$label"
    if [[ -e "$destination" || -L "$destination" ]]; then
        return 0
    fi

    if [[ -e "$partial_path" || -L "$partial_path" ]]; then
        [[ -f "$partial_path" ]] || die \
            "${label}: staging path exists but is not a regular file: ${partial_path}; it was left untouched."
        partial_size="$(stat -c '%s' -- "$partial_path")" || die \
            "${label}: cannot read staging file size for ${partial_path}."
        (( partial_size <= expected_size )) || die \
            "${label}: staging file is larger than expected at ${partial_path}; it was left untouched."
        if (( partial_size == expected_size )); then
            verify_file_or_die "$partial_path" "$expected_size" "$expected_sha256" "${label} staging file"
        fi
    fi

    mkdir -p -- "$(dirname -- "$destination")"
    if command -v aria2c >/dev/null 2>&1; then
        if aria2c \
            --continue=true \
            --allow-overwrite=false \
            --auto-file-renaming=false \
            --file-allocation=none \
            --split=8 \
            --max-connection-per-server=8 \
            --dir="$(dirname -- "$partial_path")" \
            --out="$(basename -- "$partial_path")" \
            "$url"; then
            downloader_succeeded=1
        else
            printf 'aria2c failed for %s; trying curl with resume enabled.\n' "$label" >&2
        fi
    fi

    if (( downloader_succeeded == 0 )); then
        command -v curl >/dev/null 2>&1 || die \
            "${label}: neither aria2c nor curl is available; staging data, if any, was left untouched."
        curl \
            --fail \
            --location \
            --retry 5 \
            --retry-delay 2 \
            --retry-connrefused \
            --continue-at - \
            --output "$partial_path" \
            "$url"
    fi

    verify_file_or_die "$partial_path" "$expected_size" "$expected_sha256" "${label} download"

    mkdir -p -- "$(dirname -- "$destination")"
    mv --no-clobber -- "$partial_path" "$destination"
    if [[ -e "$partial_path" || -L "$partial_path" ]]; then
        die "${label}: destination appeared during installation; verified staging file remains at ${partial_path}."
    fi
    verify_file_or_die "$destination" "$expected_size" "$expected_sha256" "$label"
    printf 'Downloaded and verified %s: %s\n' "$label" "$destination"
}

check_existing_asset "$SHOWO_WEIGHT_PATH" "$SHOWO_WEIGHT_SIZE" "$SHOWO_WEIGHT_SHA256" "Show-o2-1.5B weights"
check_existing_asset "$WAN_VAE_PATH" "$WAN_VAE_SIZE" "$WAN_VAE_SHA256" "Wan2.1 VAE"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"
[[ -f "$REQUIREMENTS_FILE" ]] || die "Requirements file not found: ${REQUIREMENTS_FILE}"

mkdir -p -- "$CHECKPOINT_DIR" "$SHOWO_MODEL_DIR" "$CACHE_DIR"
if [[ -x "$VENV_PYTHON" ]]; then
    printf 'Reusing existing venv: %s\n' "$VENV_DIR"
elif [[ -e "$VENV_DIR" || -L "$VENV_DIR" ]]; then
    die "${VENV_DIR} exists but has no executable bin/python; refusing to replace it."
else
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi

verify_system_torch() {
    "$VENV_PYTHON" - "$VENV_DIR" <<'PY'
import re
import sys
from pathlib import Path

import torch

venv_dir = Path(sys.argv[1]).resolve()
venv_cfg = (venv_dir / "pyvenv.cfg").read_text(encoding="utf-8")
if not re.search(r"^include-system-site-packages\s*=\s*true\s*$", venv_cfg, re.IGNORECASE | re.MULTILINE):
    raise SystemExit("venv was not created with --system-site-packages")
torch_path = Path(torch.__file__).resolve()
if torch_path.is_relative_to(venv_dir):
    raise SystemExit(f"torch is installed inside the venv instead of inherited from the host: {torch_path}")
if not torch.__version__.startswith("2.12.") or torch.version.cuda != "13.0":
    raise SystemExit(
        f"expected host torch 2.12+cu130, found torch {torch.__version__} / CUDA {torch.version.cuda}"
    )
print(f"Using host torch {torch.__version__} (CUDA {torch.version.cuda}) from {torch_path}")
PY
}

verify_system_torch || die "The venv must inherit the host PyTorch 2.12+cu130; no torch package was installed."
"$VENV_PYTHON" -m pip install --requirement "$REQUIREMENTS_FILE"
verify_system_torch || die "Host PyTorch is not visible after dependency installation."

export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="${CACHE_DIR}/hub"
"$VENV_PYTHON" - "$CACHE_DIR" "$CHECKPOINT_DIR" <<'PY'
import os
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

cache_dir = Path(sys.argv[1]) / "hub"
checkpoint_dir = Path(sys.argv[2])
showo_model_dir = checkpoint_dir / "show-o2-1.5B"
cache_dir.mkdir(parents=True, exist_ok=True)


def retry_download(label: str, operation):
    last_error = None
    for attempt in range(1, 6):
        try:
            return operation()
        except Exception as error:
            last_error = error
            if attempt == 5:
                break
            delay = min(attempt, 3)
            print(
                f"{label} failed on attempt {attempt}/5 ({type(error).__name__}: {error}); "
                f"retrying in {delay}s.",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise RuntimeError(f"{label} failed after 5 attempts: {last_error}") from last_error


def materialize_file(source: Path, destination: Path, label: str) -> None:
    source_bytes = source.read_bytes()
    if destination.exists() or destination.is_symlink():
        if destination.is_file() and destination.read_bytes() == source_bytes:
            return
        raise SystemExit(f"Existing {label} differs from the fetched official file; left untouched: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(source_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if not destination.is_file() or destination.read_bytes() != source_bytes:
                raise SystemExit(f"{label} appeared with different contents; left untouched: {destination}")
    finally:
        temporary.unlink(missing_ok=True)


def materialize_snapshot(snapshot: Path, destination: Path, filenames: list[str], label: str) -> None:
    for filename in filenames:
        source = snapshot / filename
        if source.is_file():
            materialize_file(source, destination / filename, f"{label} file {filename}")

qwen_files = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
]


def fetch_qwen_snapshot() -> Path:
    snapshot = Path(
        snapshot_download(
            repo_id="Qwen/Qwen2.5-1.5B-Instruct",
            revision="main",
            cache_dir=str(cache_dir),
            allow_patterns=qwen_files,
            max_workers=1,
        )
    )
    if not (snapshot / "config.json").is_file() or not (snapshot / "tokenizer.json").is_file():
        raise RuntimeError("Qwen prefetch did not return the expected config and tokenizer files")
    return snapshot


qwen_snapshot = retry_download("Qwen config/tokenizer snapshot", fetch_qwen_snapshot)
materialize_snapshot(
    qwen_snapshot,
    checkpoint_dir / "Qwen2.5-1.5B-Instruct",
    qwen_files,
    "Qwen tokenizer/config",
)


def fetch_siglip_snapshot() -> Path:
    snapshot = Path(
        snapshot_download(
            repo_id="google/siglip-so400m-patch14-384",
            revision="main",
            cache_dir=str(cache_dir),
            allow_patterns=["config.json", "preprocessor_config.json"],
            max_workers=1,
        )
    )
    if not (snapshot / "config.json").is_file():
        raise RuntimeError("SigLIP prefetch did not return config.json")
    return snapshot


siglip_snapshot = retry_download("SigLIP config snapshot", fetch_siglip_snapshot)
siglip_files = ["config.json", "preprocessor_config.json"]
materialize_snapshot(
    siglip_snapshot,
    checkpoint_dir / "siglip-so400m-patch14-384",
    siglip_files,
    "SigLIP config",
)


def fetch_showo_config() -> Path:
    config = Path(
        hf_hub_download(
            repo_id="showlab/show-o2-1.5B",
            filename="config.json",
            revision="07ec16589d4fc5422a74dddbbc4b2cd11e551039",
            cache_dir=str(cache_dir),
        )
    )
    if not config.is_file():
        raise RuntimeError("Pinned Show-o2 download did not return config.json")
    return config


showo_config = retry_download("Pinned Show-o2 config", fetch_showo_config)
showo_model_dir.mkdir(parents=True, exist_ok=True)
local_config = showo_model_dir / "config.json"
official_bytes = showo_config.read_bytes()
if local_config.exists() or local_config.is_symlink():
    if not local_config.is_file() or local_config.read_bytes() != official_bytes:
        raise SystemExit(f"Existing Show-o2 config differs from the pinned official config; left untouched: {local_config}")
else:
    materialize_file(showo_config, local_config, "Show-o2 config")

print(f"Qwen config/tokenizer cached at {checkpoint_dir / 'Qwen2.5-1.5B-Instruct'}")
print(f"SigLIP config cached at {checkpoint_dir / 'siglip-so400m-patch14-384'}; model weights were not requested")
print(f"Pinned Show-o2 config prepared at {local_config}")
PY

download_verified_asset \
    "$SHOWO_WEIGHT_URL" \
    "$SHOWO_WEIGHT_PATH" \
    "$SHOWO_WEIGHT_SIZE" \
    "$SHOWO_WEIGHT_SHA256" \
    "Show-o2-1.5B weights"

download_verified_asset \
    "$WAN_VAE_URL" \
    "$WAN_VAE_PATH" \
    "$WAN_VAE_SIZE" \
    "$WAN_VAE_SHA256" \
    "Wan2.1 VAE"

printf 'CSGO Seen-10 setup assets are ready.\n'
