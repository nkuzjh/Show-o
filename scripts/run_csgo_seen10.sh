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
  --batch-size N           Inference batch size (default: config value, 16)
  --output-root PATH       Seed output root (default: configured project output)
  --checkpoint PATH|best|late|latest (default: best)
  --resume [best|late|latest|PATH]
  --limit N                Optional inference prefix (smoke always uses 1)
  --experiment NAME        Select csgo_seen10_exp32gen_aligned (default: legacy)
  --finetuning-policy NAME Aligned policy: aligned_v2_final (default) or aligned_v1
  --num-processes N        Aligned training workers (default: visible CUDA count)
  --micro-batch N          Aligned per-process batch size (default: 8)
  --gradient-accumulation N Aligned accumulation (default: 128/workers/micro-batch)
  --dry-run                Print commands without running them
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
SEED_SET=false
INFERENCE_SEED=""
TASK=all
OUTPUT_ROOT=""
CHECKPOINT=best
CHECKPOINT_SET=false
RESUME=""
LIMIT=""
BATCH_SIZE=""
EXPERIMENT=""
FINETUNING_POLICY=""
NUM_PROCESSES=""
MICRO_BATCH=""
GRADIENT_ACCUMULATION=""
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)
            [[ $# -ge 2 ]] || { echo "--seed needs a value" >&2; exit 2; }
            SEED="$2"; SEED_SET=true; shift 2 ;;
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
            CHECKPOINT="$2"; CHECKPOINT_SET=true; shift 2 ;;
        --resume)
            if [[ $# -ge 2 && "$2" != --* ]]; then RESUME="$2"; shift 2; else RESUME=latest; shift; fi ;;
        --limit)
            [[ $# -ge 2 ]] || { echo "--limit needs a value" >&2; exit 2; }
            LIMIT="$2"; shift 2 ;;
        --batch-size)
            [[ $# -ge 2 ]] || { echo "--batch-size needs a value" >&2; exit 2; }
            BATCH_SIZE="$2"; shift 2 ;;
        --experiment)
            [[ $# -ge 2 ]] || { echo "--experiment needs a value" >&2; exit 2; }
            EXPERIMENT="$2"; shift 2 ;;
        --finetuning-policy)
            [[ $# -ge 2 ]] || { echo "--finetuning-policy needs a value" >&2; exit 2; }
            FINETUNING_POLICY="$2"; shift 2 ;;
        --num-processes)
            [[ $# -ge 2 ]] || { echo "--num-processes needs a value" >&2; exit 2; }
            NUM_PROCESSES="$2"; shift 2 ;;
        --micro-batch)
            [[ $# -ge 2 ]] || { echo "--micro-batch needs a value" >&2; exit 2; }
            MICRO_BATCH="$2"; shift 2 ;;
        --gradient-accumulation)
            [[ $# -ge 2 ]] || { echo "--gradient-accumulation needs a value" >&2; exit 2; }
            GRADIENT_ACCUMULATION="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "${TASK}" == all || "${TASK}" == discrete || "${TASK}" == continuous ]] || {
    echo "Invalid --task: ${TASK}" >&2; exit 2;
}
[[ -z "${EXPERIMENT}" || "${EXPERIMENT}" == csgo_seen10_exp32gen_aligned ]] || {
    echo "Invalid --experiment: ${EXPERIMENT}" >&2; exit 2;
}
if [[ -n "${FINETUNING_POLICY}" ]]; then
    [[ "${EXPERIMENT}" == csgo_seen10_exp32gen_aligned ]] || {
        echo "--finetuning-policy requires --experiment csgo_seen10_exp32gen_aligned" >&2; exit 2;
    }
    [[ "${FINETUNING_POLICY}" == aligned_v1 || "${FINETUNING_POLICY}" == aligned_v2_final ]] || {
        echo "Invalid --finetuning-policy: ${FINETUNING_POLICY}" >&2; exit 2;
    }
fi

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

visible_cuda_count() {
    local count=0 device
    if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
        if [[ -n "${CUDA_VISIBLE_DEVICES}" && "${CUDA_VISIBLE_DEVICES}" != -1 && "${CUDA_VISIBLE_DEVICES}" != none ]]; then
            local devices=()
            IFS=, read -r -a devices <<< "${CUDA_VISIBLE_DEVICES}"
            for device in "${devices[@]}"; do
                [[ -z "${device//[[:space:]]/}" ]] || ((count += 1))
            done
        fi
    elif command -v nvidia-smi >/dev/null 2>&1; then
        while IFS= read -r device; do
            [[ -z "${device}" ]] || ((count += 1))
        done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
    fi
    ((count >= 1)) || count=1
    printf '%s\n' "${count}"
}

if [[ "${EXPERIMENT}" == csgo_seen10_exp32gen_aligned ]]; then
    [[ -n "${FINETUNING_POLICY}" ]] || FINETUNING_POLICY=aligned_v2_final
    if [[ "${FINETUNING_POLICY}" == aligned_v1 ]]; then
        CONFIG="${SHOWO2_DIR}/configs/csgo_seen10_exp32gen_aligned_v1.yaml"
        DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/outputs/csgo_benchmark_v2_aligned/${EXPERIMENT}"
    else
        CONFIG="${SHOWO2_DIR}/configs/csgo_seen10_exp32gen_aligned.yaml"
        DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/outputs/csgo_benchmark_v2_aligned/${EXPERIMENT}_v2_final"
    fi
    [[ "${SEED_SET}" == true ]] || SEED=42
    [[ "${CHECKPOINT_SET}" == true ]] || CHECKPOINT=late
    if [[ "${MODE}" == train || "${MODE}" == smoke ]]; then
    [[ -n "${NUM_PROCESSES}" ]] || NUM_PROCESSES="$(visible_cuda_count)"
    [[ -n "${MICRO_BATCH}" ]] || MICRO_BATCH=8
    is_positive_integer "${NUM_PROCESSES}" || { echo "--num-processes must be a positive integer" >&2; exit 2; }
    is_positive_integer "${MICRO_BATCH}" || { echo "--micro-batch must be a positive integer" >&2; exit 2; }
    if [[ -z "${GRADIENT_ACCUMULATION}" ]]; then
        ((128 % (NUM_PROCESSES * MICRO_BATCH) == 0)) || {
            echo "128 must be divisible by num-processes * micro-batch" >&2; exit 2;
        }
        GRADIENT_ACCUMULATION=$((128 / (NUM_PROCESSES * MICRO_BATCH)))
    fi
    is_positive_integer "${GRADIENT_ACCUMULATION}" || {
        echo "--gradient-accumulation must be a positive integer" >&2; exit 2;
    }
    ((NUM_PROCESSES * MICRO_BATCH * GRADIENT_ACCUMULATION == 128)) || {
        echo "num-processes * micro-batch * gradient-accumulation must equal 128" >&2; exit 2;
    }
    elif [[ -n "${NUM_PROCESSES}${MICRO_BATCH}${GRADIENT_ACCUMULATION}" ]]; then
        echo "Training batch options apply only to train/smoke" >&2; exit 2
    fi
elif [[ -n "${NUM_PROCESSES}${MICRO_BATCH}${GRADIENT_ACCUMULATION}" ]]; then
    echo "Distributed batch options require --experiment csgo_seen10_exp32gen_aligned" >&2; exit 2
fi

if [[ "${DRY_RUN}" != true ]]; then
    [[ -x "${PYTHON}" ]] || { echo "Project venv is missing; run scripts/setup_csgo_seen10.sh first" >&2; exit 1; }
fi
if [[ "${DRY_RUN}" != true && ( "${MODE}" == smoke || "${MODE}" == eval ) ]]; then
    [[ -f "${EVAL_DIR}/run_eval.py" ]] || { echo "Shared evaluator not found: ${EVAL_DIR}/run_eval.py" >&2; exit 1; }
    [[ -x "${EVAL_PYTHON}" ]] || { echo "Evaluator Python is missing: ${EVAL_PYTHON}" >&2; exit 1; }
fi

run_command() {
    if [[ "${DRY_RUN}" == true ]]; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

SEED_OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}/seed_${SEED}}"
if [[ "${SEED_OUTPUT_ROOT}" != /* ]]; then
    SEED_OUTPUT_ROOT="${REPO_ROOT}/${SEED_OUTPUT_ROOT}"
fi

checkpoint_label() {
    local checkpoint="${CHECKPOINT%/}"
    basename -- "${checkpoint}"
}

prediction_output_root() {
    local root="$1"
    if [[ "${EXPERIMENT}" == csgo_seen10_exp32gen_aligned ]]; then
        printf '%s/predictions/%s\n' "${root}" "$(checkpoint_label)"
    else
        printf '%s\n' "${root}"
    fi
}

run_train() {
    local args=("$@")
    if [[ "${EXPERIMENT}" == csgo_seen10_exp32gen_aligned ]]; then
        args+=(--micro-batch "${MICRO_BATCH}" --gradient-accumulation "${GRADIENT_ACCUMULATION}")
        if ((NUM_PROCESSES > 1)); then
            run_command "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_PROCESSES}" \
                "${SHOWO2_DIR}/train_seen10.py" "${args[@]}"
            return
        fi
    fi
    run_command "${PYTHON}" "${SHOWO2_DIR}/train_seen10.py" "${args[@]}"
}

run_infer() {
    local root="$1"
    local limit_arg=()
    [[ -z "${LIMIT}" ]] || limit_arg=(--limit "${LIMIT}")
    local batch_size_arg=()
    [[ -z "${BATCH_SIZE}" ]] || batch_size_arg=(--batch-size "${BATCH_SIZE}")
    local infer_seed_arg=()
    [[ -z "${INFERENCE_SEED}" ]] || infer_seed_arg=(--inference-seed "${INFERENCE_SEED}")
    local prediction_arg=()
    if [[ "${EXPERIMENT}" == csgo_seen10_exp32gen_aligned ]]; then
        prediction_arg=(--prediction-output-root "$(prediction_output_root "${root}")")
    fi
    run_command "${PYTHON}" "${SHOWO2_DIR}/infer_seen10.py" \
        --config "${CONFIG}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
        --output-root "${root}" --checkpoint "${CHECKPOINT}" --task "${TASK}" \
        "${prediction_arg[@]}" "${infer_seed_arg[@]}" "${limit_arg[@]}" "${batch_size_arg[@]}"
}

run_smoke_eval() {
    local root="$1"
    local prediction_root
    prediction_root="$(prediction_output_root "${root}")"
    if [[ "${TASK}" == all || "${TASK}" == discrete ]]; then
        run_command "${EVAL_PYTHON}" "${EVAL_DIR}/run_eval.py" smoke discrete \
            --config "${EVAL_DIR}/benchmark_v2.yaml" \
            --pred-root "${prediction_root}/discrete/gen_imgs" --data-root "${DATA_ROOT}" --limit 1
    fi
    if [[ "${TASK}" == all || "${TASK}" == continuous ]]; then
        run_command "${EVAL_PYTHON}" "${EVAL_DIR}/run_eval.py" smoke continuous \
            --config "${EVAL_DIR}/benchmark_v2.yaml" \
            --pred-root "${prediction_root}/continuous/gen_imgs" --data-root "${DATA_ROOT}" --frame-only
    fi
}

run_formal_eval() {
    local root="$1"
    local task
    local prediction_root
    prediction_root="$(prediction_output_root "${root}")"
    local tasks=(discrete continuous)
    [[ "${TASK}" == all ]] || tasks=("${TASK}")
    for task in "${tasks[@]}"; do
        local evaluation_root="${root}/evaluation_shared/${task}"
        if [[ "${EXPERIMENT}" == csgo_seen10_exp32gen_aligned ]]; then
            evaluation_root="${root}/evaluations/$(checkpoint_label)/${task}"
        fi
        run_command "${EVAL_PYTHON}" "${EVAL_DIR}/run_eval.py" "${task}" \
            --config "${EVAL_DIR}/benchmark_v2.yaml" \
            --pred-root "${prediction_root}/${task}/gen_imgs" --data-root "${DATA_ROOT}" \
            --output "${evaluation_root}"
    done
}

case "${MODE}" in
    smoke)
        SMOKE_PARENT="${SEED_OUTPUT_ROOT}/smoke_runs"
        if [[ "${EXPERIMENT}" == csgo_seen10_exp32gen_aligned ]]; then
            SMOKE_PARENT="${SEED_OUTPUT_ROOT}_smoke_runs"
        fi
        if [[ "${DRY_RUN}" != true ]]; then
            mkdir -p -- "${SMOKE_PARENT}"
        fi
        SMOKE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
        if [[ "${EXPERIMENT}" == csgo_seen10_exp32gen_aligned ]]; then
            SMOKE_TAG="${SMOKE_TAG}-$$"
        fi
        SMOKE_ROOT="${SMOKE_PARENT}/${SMOKE_TAG}"
        if [[ "${DRY_RUN}" != true ]]; then
            [[ ! -e "${SMOKE_ROOT}" ]] || { echo "Smoke output already exists: ${SMOKE_ROOT}" >&2; exit 1; }
        fi
        CHECKPOINT=latest
        LIMIT=1
        run_train \
            --config "${CONFIG}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
            --output-dir "${SMOKE_ROOT}" --smoke --max-steps 1
        run_infer "${SMOKE_ROOT}"
        run_smoke_eval "${SMOKE_ROOT}"
        ;;
    train)
        train_args=(--config "${CONFIG}" --seed "${SEED}" --data-root "${DATA_ROOT}" --output-dir "${SEED_OUTPUT_ROOT}")
        [[ -z "${RESUME}" ]] || train_args+=(--resume "${RESUME}")
        run_train "${train_args[@]}"
        ;;
    infer)
        run_infer "${SEED_OUTPUT_ROOT}"
        ;;
    eval)
        run_formal_eval "${SEED_OUTPUT_ROOT}"
        ;;
esac
