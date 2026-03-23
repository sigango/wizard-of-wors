#!/usr/bin/env bash
set -euo pipefail

# Terminal-only runner:
# 1) Train Wizard of Wor PPO with full pixel preset
# 2) Auto-pick latest trained model
# 3) Record gameplay video using visualize mode

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PRESET="${PRESET:-config_wizard_pixel_final}"
ENV_TYPE="${ENV_TYPE:-jax}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
NUM_RECORD_EPISODES="${NUM_RECORD_EPISODES:-5}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

echo "[1/3] Training PPO (${PRESET})..."
"${PYTHON_BIN}" scripts/benchmarks/agent_performance_comparison.py \
  --mode train-jaxatari \
  --preset "${PRESET}" \
  --env_type "${ENV_TYPE}" \
  --num_episodes "${NUM_EVAL_EPISODES}"

LATEST_RESULTS_DIR="$(ls -td "${REPO_ROOT}"/results/ppo_distrax_jax_wizardofwor_* 2>/dev/null | head -n 1 || true)"
if [[ -z "${LATEST_RESULTS_DIR}" ]]; then
  echo "Could not find results directory under ${REPO_ROOT}/results"
  exit 1
fi

MODEL_PATH="${LATEST_RESULTS_DIR}/ppo_distrax_model_params.npz"
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model not found at: ${MODEL_PATH}"
  exit 1
fi

echo "[2/3] Latest model: ${MODEL_PATH}"
echo "[3/3] Recording gameplay (${NUM_RECORD_EPISODES} episodes)..."
"${PYTHON_BIN}" scripts/benchmarks/agent_performance_comparison.py \
  --mode visualize \
  --preset "${PRESET}" \
  --env_type "${ENV_TYPE}" \
  --model_path "${MODEL_PATH}" \
  --num_episodes "${NUM_RECORD_EPISODES}"

echo "Done. Video is saved in the current directory as agent_visualization_wizardofwor_<timestamp>.mp4"
