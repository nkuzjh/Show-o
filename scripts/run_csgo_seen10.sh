#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SHOWO2_DIR="${REPO_ROOT}/show-o2"
PYTHON="${SHOWO2_DIR}/.venv/bin/python"
EVAL_DIR="${SHARED_EVAL_DIR:-/home/jiahao/task/csgo_benchmark_v2_eval_general}"
EVAL_PYTHON="${EVAL_PYTHON:-${UNILIP_PYTHON:-/home/jiahao/miniconda3/envs/UniLIP/bin/python}}"
DATA_ROOT="${DATA_ROOT:-/home/jiahao/task/UniLIP/data/csgo_benchmark_v2}"
CONFIG="${CONFIG:-${SHOWO2_DIR}/configs/showo2_1.5b_csgo_seen10.yaml}"
DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/outputs/csgo_benchmark_v2_seen10/Show-o2-1.5B"

usage() {
    cat <<'EOF'
Usage: scripts/run_csgo_seen10.sh {smoke|train|infer|eval} [options]

Options:
  --seed N                 Training seed (default: 0)
  --inference-seed N       Generation RNG seed (default: config value, 42)
  --task all|discrete|continuous (default: all)
  --output-root PATH       Seed output root (default: configured project output)
  --checkpoint PATH|best|late|latest (default: best)
  --resume [best|late|latest|PATH]
  --limit N                Optional inference prefix (smoke always uses 1)
EOF
}

[[ $# -gt 0 ]] || { usage >&2; exit 2; }
MODE="$1"
shift
case "${MODE}" in
    smoke|train|infer|eval) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

SEED=0
INFERENCE_SEED=""
TASK=all
OUTPUT_ROOT=""
CHECKPOINT=best
RESUME=""
LIMIT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)
            [[ $# -ge 2 ]] || { echo "--seed needs a value" >&2; exit 2; }
            SEED="$2"; shift 2 ;;
        --inference-seed)
            [[ $# -ge 2 ]] || { echo "--inference-seed needs a value" >&2; exit 2; }
            INFERENCE_SEED="$2"; shift 2 ;;
        --task)
            [[ $# -ge 2 ]] || { echo "--task needs a value" >&2; exit 2; }
            TASK="$2"; shift 2 ;;
        --output-root)
            [[ $# -ge 2 ]] || { echo "--output-root needs a value" >&2; exit 2; }
            OUTPUT_ROOT="$2"; shift 2 ;;
        --checkpoint)
            [[ $# -ge 2 ]] || { echo "--checkpoint needs a value" >&2; exit 2; }
            CHECKPOINT="$2"; shift 2 ;;
        --resume)
            if [[ $# -ge 2 && "$2" != --* ]]; then RESUME="$2"; shift 2; else RESUME=latest; shift; fi ;;
        --limit)
            [[ $# -ge 2 ]] || { echo "--limit needs a value" >&2; exit 2; }
            LIMIT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "${TASK}" == all || "${TASK}" == discrete || "${TASK}" == continuous ]] || {
    echo "Invalid --task: ${TASK}" >&2; exit 2;
}
[[ -x "${PYTHON}" ]] || { echo "Project venv is missing; run scripts/setup_csgo_seen10.sh first" >&2; exit 1; }
if [[ "${MODE}" == smoke || "${MODE}" == eval ]]; then
    [[ -f "${EVAL_DIR}/run_eval.py" ]] || { echo "Shared evaluator not found: ${EVAL_DIR}/run_eval.py" >&2; exit 1; }
    [[ -x "${EVAL_PYTHON}" ]] || { echo "Evaluator Python is missing: ${EVAL_PYTHON}" >&2; exit 1; }
fi

SEED_OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}/seed_${SEED}}"
if [[ "${SEED_OUTPUT_ROOT}" != /* ]]; then
    SEED_OUTPUT_ROOT="${REPO_ROOT}/${SEED_OUTPUT_ROOT}"
fi

run_infer() {
    local root="$1"
    local limit_arg=()
    [[ -z "${LIMIT}" ]] || limit_arg=(--limit "${LIMIT}")
    local infer_seed_arg=()
    [[ -z "${INFERENCE_SEED}" ]] || infer_seed_arg=(--inference-seed "${INFERENCE_SEED}")
    "${PYTHON}" "${SHOWO2_DIR}/infer_seen10.py" \
        --config "${CONFIG}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
        --output-root "${root}" --checkpoint "${CHECKPOINT}" --task "${TASK}" \
        "${infer_seed_arg[@]}" "${limit_arg[@]}"
}

run_smoke_eval() {
    local root="$1"
    if [[ "${TASK}" == all || "${TASK}" == discrete ]]; then
        "${EVAL_PYTHON}" "${EVAL_DIR}/run_eval.py" smoke discrete \
            --config "${EVAL_DIR}/benchmark_v2.yaml" \
            --pred-root "${root}/discrete/gen_imgs" --data-root "${DATA_ROOT}" --limit 1
    fi
    if [[ "${TASK}" == all || "${TASK}" == continuous ]]; then
        "${EVAL_PYTHON}" "${EVAL_DIR}/run_eval.py" smoke continuous \
            --config "${EVAL_DIR}/benchmark_v2.yaml" \
            --pred-root "${root}/continuous/gen_imgs" --data-root "${DATA_ROOT}" --frame-only
    fi
}

run_formal_eval() {
    local root="$1"
    local task
    local tasks=(discrete continuous)
    [[ "${TASK}" == all ]] || tasks=("${TASK}")
    for task in "${tasks[@]}"; do
        "${EVAL_PYTHON}" "${EVAL_DIR}/run_eval.py" "${task}" \
            --config "${EVAL_DIR}/benchmark_v2.yaml" \
            --pred-root "${root}/${task}/gen_imgs" --data-root "${DATA_ROOT}" \
            --output "${root}/evaluation_shared/${task}"
    done
}

case "${MODE}" in
    smoke)
        mkdir -p -- "${SEED_OUTPUT_ROOT}/smoke_runs"
        SMOKE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
        SMOKE_ROOT="${SEED_OUTPUT_ROOT}/smoke_runs/${SMOKE_TAG}"
        [[ ! -e "${SMOKE_ROOT}" ]] || { echo "Smoke output already exists: ${SMOKE_ROOT}" >&2; exit 1; }
        CHECKPOINT=latest
        LIMIT=1
        "${PYTHON}" "${SHOWO2_DIR}/train_seen10.py" \
            --config "${CONFIG}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
            --output-dir "${SMOKE_ROOT}" --smoke --max-steps 1
        run_infer "${SMOKE_ROOT}"
        run_smoke_eval "${SMOKE_ROOT}"
        ;;
    train)
        train_args=(--config "${CONFIG}" --seed "${SEED}" --data-root "${DATA_ROOT}" --output-dir "${SEED_OUTPUT_ROOT}")
        [[ -z "${RESUME}" ]] || train_args+=(--resume "${RESUME}")
        "${PYTHON}" "${SHOWO2_DIR}/train_seen10.py" "${train_args[@]}"
        ;;
    infer)
        run_infer "${SEED_OUTPUT_ROOT}"
        ;;
    eval)
        run_formal_eval "${SEED_OUTPUT_ROOT}"
        ;;
esac
