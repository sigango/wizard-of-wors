#!/usr/bin/env bash
set -euo pipefail

# Wizard of Wor report runner (repo-local version).
# Runs object + pixel training presets and can optionally record gameplay videos.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN=0
QUICK=0
MACSAFE=0
WITH_VIS=0
SKIP_TRAIN=0

NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
NUM_VIS_EPISODES="${NUM_VIS_EPISODES:-5}"

usage() {
  cat <<'EOF'
Usage:
  scripts/benchmarks/run_report_wizard_of_wor.sh [--quick] [--macsafe] [--with-visualize] [--dry-run] [--skip-train]

Flags:
  --quick           Use test presets (10% steps) for quick validation.
  --macsafe         Use full-step macsafe presets.
  --with-visualize  After each training, auto-visualize latest model and record video.
  --dry-run         Print commands only.
  --skip-train      Skip training and only visualize latest checkpoints (requires --with-visualize).

Environment variables:
  PYTHON_BIN         default: python3
  NUM_EVAL_EPISODES  default: 10
  NUM_VIS_EPISODES   default: 5
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      QUICK=1
      shift
      ;;
    --macsafe)
      MACSAFE=1
      shift
      ;;
    --with-visualize)
      WITH_VIS=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

if (( QUICK == 1 )); then
  OBJECT_PRESET="config_wizard_object_test"
  PIXEL_PRESET="config_wizard_pixel_test"
elif (( MACSAFE == 1 )); then
  OBJECT_PRESET="config_wizard_object_final_macsafe"
  PIXEL_PRESET="config_wizard_pixel_final_macsafe"
else
  OBJECT_PRESET="config_wizard_object_final"
  PIXEL_PRESET="config_wizard_pixel_final"
fi

run_cmd() {
  echo ""
  echo ">>> $*"
  if (( DRY_RUN == 0 )); then
    "$@"
  fi
}

latest_model_path() {
  local latest_dir
  latest_dir="$(ls -td "${REPO_ROOT}"/results/ppo_distrax_jax_wizardofwor_* 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest_dir}" ]]; then
    return 1
  fi
  local model="${latest_dir}/ppo_distrax_model_params.npz"
  if [[ ! -f "${model}" ]]; then
    return 1
  fi
  printf '%s\n' "${model}"
}

train_with_preset() {
  local preset="$1"
  local label="$2"
  echo "[train:${label}] preset=${preset}"
  run_cmd "${PYTHON_BIN}" scripts/benchmarks/agent_performance_comparison.py \
    --mode train-jaxatari \
    --preset "${preset}" \
    --env_type jax \
    --num_episodes "${NUM_EVAL_EPISODES}"
}

visualize_latest_with_preset() {
  local preset="$1"
  local label="$2"
  local model
  model="$(latest_model_path)" || {
    echo "No trained model found for visualization (${label})."
    return 1
  }
  echo "[visualize:${label}] model=${model}"
  run_cmd "${PYTHON_BIN}" scripts/benchmarks/agent_performance_comparison.py \
    --mode visualize \
    --preset "${preset}" \
    --env_type jax \
    --model_path "${model}" \
    --num_episodes "${NUM_VIS_EPISODES}"
}

echo "Wizard of Wor report pipeline"
echo "object_preset=${OBJECT_PRESET}"
echo "pixel_preset=${PIXEL_PRESET}"
echo "quick=${QUICK}, macsafe=${MACSAFE}, with_visualize=${WITH_VIS}, dry_run=${DRY_RUN}, skip_train=${SKIP_TRAIN}"

if (( SKIP_TRAIN == 0 )); then
  train_with_preset "${OBJECT_PRESET}" "object"
  train_with_preset "${PIXEL_PRESET}" "pixel"
fi

if (( WITH_VIS == 1 )); then
  visualize_latest_with_preset "${OBJECT_PRESET}" "object"
  visualize_latest_with_preset "${PIXEL_PRESET}" "pixel"
fi

echo "Wizard report run complete."
