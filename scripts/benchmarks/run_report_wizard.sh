#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
if command -v nvidia-smi >/dev/null 2>&1; then
  DEFAULT_XLA_PREALLOCATE="true"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
else
  DEFAULT_XLA_PREALLOCATE="false"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-${DEFAULT_XLA_PREALLOCATE}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if command -v nvidia-smi >/dev/null 2>&1; then
  DEFAULT_JAX_PIP_SPEC="jax[cuda13]"
else
  DEFAULT_JAX_PIP_SPEC="jax"
fi
JAX_PIP_SPEC="${JAX_PIP_SPEC:-${DEFAULT_JAX_PIP_SPEC}}"
ALGO="ppo"
OBS_MODE="pixel"
QUICK=0
MACSAFE=0
RTX5090_PROFILE=0
WITH_VIS=0
SKIP_TRAIN=0
SKIP_PREFLIGHT=0
SHOW_SYSTEM_INFO=1
DRY_RUN=0
AUTO_INSTALL_DEPS=1
NUM_ENVS_OVERRIDE=""
NUM_STEPS_OVERRIDE=""
NUM_MINIBATCHES_OVERRIDE=""
TOTAL_TIMESTEPS_OVERRIDE=""
UPDATE_EPOCHS_OVERRIDE=""

NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
NUM_VIS_EPISODES="${NUM_VIS_EPISODES:-5}"

usage() {
  cat <<'EOF'
Usage:
  scripts/benchmarks/run_report_wizard.sh [options]

Options:
  --algo ALG             ALG in {ppo,pqn}. Default: ppo.
  --obs MODE             MODE in {object,pixel}. Default: pixel.
  --rtx5090              Force non-macsafe final preset for NVIDIA RTX 5090 runs.
  --num-envs N           Override NUM_ENVS for selected preset.
  --num-steps N          Override NUM_STEPS for selected preset.
  --num-minibatches N    Override NUM_MINIBATCHES for selected preset.
  --total-steps N        Override TOTAL_TIMESTEPS for selected preset.
  --update-epochs N      Override UPDATE_EPOCHS for selected preset.
  --quick                Use test preset (10% style run).
  --macsafe              Use macsafe final preset.
  --with-visualize       Record visualization after training with latest checkpoint.
  --dry-run              Print commands only.
  --skip-train           Skip training and only run visualization (requires --with-visualize).
  --skip-preflight       Skip dependency checks.
  --no-auto-install      Disable auto-install for missing deps.
  --no-system-info       Skip hardware info.
  -h, --help             Show this help.

Environment:
  PYTHON_BIN, NUM_EVAL_EPISODES, NUM_VIS_EPISODES, JAX_PIP_SPEC
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --algo)
      ALGO="$2"
      shift 2
      ;;
    --obs)
      OBS_MODE="$2"
      shift 2
      ;;
    --rtx5090)
      RTX5090_PROFILE=1
      QUICK=0
      MACSAFE=0
      shift
      ;;
    --num-envs)
      NUM_ENVS_OVERRIDE="$2"
      shift 2
      ;;
    --num-steps)
      NUM_STEPS_OVERRIDE="$2"
      shift 2
      ;;
    --num-minibatches)
      NUM_MINIBATCHES_OVERRIDE="$2"
      shift 2
      ;;
    --total-steps)
      TOTAL_TIMESTEPS_OVERRIDE="$2"
      shift 2
      ;;
    --update-epochs)
      UPDATE_EPOCHS_OVERRIDE="$2"
      shift 2
      ;;
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
    --skip-preflight)
      SKIP_PREFLIGHT=1
      shift
      ;;
    --no-auto-install)
      AUTO_INSTALL_DEPS=0
      shift
      ;;
    --no-system-info)
      SHOW_SYSTEM_INFO=0
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

case "${ALGO}" in
  ppo|pqn) ;;
  *)
    echo "Invalid --algo '${ALGO}'. Use ppo|pqn." >&2
    exit 1
    ;;
esac

case "${OBS_MODE}" in
  object|pixel) ;;
  *)
    echo "Invalid --obs '${OBS_MODE}'. Use object|pixel." >&2
    exit 1
    ;;
esac

if [[ "${ALGO}" == "pqn" ]]; then
  if [[ ! -f "${REPO_ROOT}/scripts/benchmarks/pqn_agent.py" ]]; then
    echo "PQN pipeline is not available in this WizardOfWor repo yet." >&2
    echo "Use --algo ppo, or add PQN benchmark scripts first." >&2
    exit 2
  fi
fi

print_report_system_info() {
  echo "=== Report Hardware Info ==="
  echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    echo "OS: ${PRETTY_NAME:-unknown}"
  elif command -v sw_vers >/dev/null 2>&1; then
    echo "OS: $(sw_vers -productName) $(sw_vers -productVersion)"
  else
    echo "OS: $(uname -srm)"
  fi

  if command -v lscpu >/dev/null 2>&1; then
    echo "CPU: $(lscpu | awk -F: '/Model name/{print $2; exit}' | xargs)"
    echo "CPU cores: $(lscpu | awk -F: '/^CPU\\(s\\)/{print $2; exit}' | xargs)"
  elif command -v sysctl >/dev/null 2>&1; then
    echo "CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
    echo "CPU cores: $(sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
  fi

  if command -v free >/dev/null 2>&1; then
    echo "RAM: $(free -h | awk '/^Mem:/{print $2}')"
  elif command -v sysctl >/dev/null 2>&1; then
    ram_bytes="$(sysctl -n hw.memsize 2>/dev/null || true)"
    if [[ -n "${ram_bytes}" ]]; then
      echo "RAM: $(awk -v b="${ram_bytes}" 'BEGIN{printf "%.1f GB", b/1024/1024/1024}')"
    fi
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU(s):"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  - /'
  else
    echo "GPU(s): nvidia-smi not found (CPU-only or non-NVIDIA setup)."
  fi

  "${PYTHON_BIN}" - <<'PY'
import sys
print(f"Python Version: {sys.version.split()[0]}")
try:
    import jax
    print(f"JAX Version: {jax.__version__}")
    try:
        import jaxlib
        print(f"jaxlib Version: {jaxlib.__version__}")
    except Exception:
        print("jaxlib Version: not detected")
    print(f"JAX Backend: {jax.default_backend()}")
    print(f"JAX Devices: {jax.devices()}")
except Exception as exc:
    print(f"JAX check failed: {exc}")
PY
  echo "============================"
}

check_modules() {
  local specs="$1"
  PY_REQ_SPECS="${specs}" AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS}" "${PYTHON_BIN}" - <<'PY'
import os
import sys
import importlib.util
import subprocess

missing = []
for item in [x.strip() for x in os.environ.get("PY_REQ_SPECS", "").split(",") if x.strip()]:
    if ":" in item:
        mod, pkg = item.split(":", 1)
    else:
        mod, pkg = item, item
    if importlib.util.find_spec(mod) is None:
        missing.append((mod, pkg))

if missing:
    print("Missing required Python modules:")
    for mod, pkg in missing:
        print(f"  - module '{mod}' (install package: {pkg})")
    pkgs = sorted({pkg for _, pkg in missing})
    if os.environ.get("AUTO_INSTALL_DEPS", "1") == "1":
        cmd = [sys.executable, "-m", "pip", "install", *pkgs]
        print("\nAuto-installing:", " ".join(cmd))
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as exc:
            print(f"Auto-install failed (exit={exc.returncode}).")
            sys.exit(3)
    else:
        print("\nAuto-install disabled. Install manually with:")
        print(f"  {sys.executable} -m pip install {' '.join(pkgs)}")
        sys.exit(3)
PY
}

if (( SHOW_SYSTEM_INFO == 1 )); then
  print_report_system_info
fi

if (( SKIP_PREFLIGHT == 0 )); then
  reqs="jax:${JAX_PIP_SPEC},numpy:numpy,flax:flax,optax:optax,distrax:distrax,pandas:pandas,matplotlib:matplotlib,tqdm:tqdm,wandb:wandb,pygame:pygame,ocatari:ocatari"
  if (( WITH_VIS == 1 )); then
    reqs="${reqs},cv2:opencv-python"
  fi
  check_modules "${reqs}"
fi

if (( QUICK == 1 )); then
  if [[ "${OBS_MODE}" == "object" ]]; then
    PRESET="config_wizard_object_test"
  else
    PRESET="config_wizard_pixel_test"
  fi
elif (( MACSAFE == 1 )); then
  if [[ "${OBS_MODE}" == "object" ]]; then
    PRESET="config_wizard_object_final_macsafe"
  else
    PRESET="config_wizard_pixel_final_macsafe"
  fi
else
  if [[ "${OBS_MODE}" == "object" ]]; then
    PRESET="config_wizard_object_final"
  else
    PRESET="config_wizard_pixel_final"
  fi
fi

run_cmd() {
  echo ""
  echo ">>> $*"
  if (( DRY_RUN == 0 )); then
    "$@"
  fi
}

latest_ppo_model_path() {
  local latest_dir
  latest_dir="$(ls -td "${REPO_ROOT}"/results/ppo_distrax_jax_wizardofwor_* 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest_dir}" ]]; then
    return 1
  fi
  local model="${latest_dir}/ppo_distrax_model_params.npz"
  [[ -f "${model}" ]] || return 1
  printf '%s\n' "${model}"
}

train_ppo() {
  local cmd=(
    "${PYTHON_BIN}" scripts/benchmarks/agent_performance_comparison.py
    --mode train-jaxatari \
    --preset "${PRESET}" \
    --env_type jax \
    --num_episodes "${NUM_EVAL_EPISODES}"
  )
  if [[ -n "${NUM_ENVS_OVERRIDE}" ]]; then
    cmd+=(--num-envs "${NUM_ENVS_OVERRIDE}")
  fi
  if [[ -n "${NUM_STEPS_OVERRIDE}" ]]; then
    cmd+=(--num-steps "${NUM_STEPS_OVERRIDE}")
  fi
  if [[ -n "${NUM_MINIBATCHES_OVERRIDE}" ]]; then
    cmd+=(--num-minibatches "${NUM_MINIBATCHES_OVERRIDE}")
  fi
  if [[ -n "${TOTAL_TIMESTEPS_OVERRIDE}" ]]; then
    cmd+=(--total-steps "${TOTAL_TIMESTEPS_OVERRIDE}")
  fi
  if [[ -n "${UPDATE_EPOCHS_OVERRIDE}" ]]; then
    cmd+=(--update-epochs "${UPDATE_EPOCHS_OVERRIDE}")
  fi
  run_cmd "${cmd[@]}"
}

visualize_ppo_latest() {
  local model
  model="$(latest_ppo_model_path)" || {
    echo "No trained PPO model found for visualization." >&2
    return 1
  }
  run_cmd "${PYTHON_BIN}" scripts/benchmarks/agent_performance_comparison.py \
    --mode visualize \
    --preset "${PRESET}" \
    --env_type jax \
    --model_path "${model}" \
    --num_episodes "${NUM_VIS_EPISODES}"
}

echo "Wizard of Wor single-RL report runner"
echo "algo=${ALGO}, obs=${OBS_MODE}, preset=${PRESET}"
echo "quick=${QUICK}, macsafe=${MACSAFE}, rtx5090=${RTX5090_PROFILE}, with_visualize=${WITH_VIS}, skip_train=${SKIP_TRAIN}, dry_run=${DRY_RUN}"
echo "JAX auto-install package: ${JAX_PIP_SPEC}"

if [[ "${ALGO}" == "ppo" ]]; then
  if (( SKIP_TRAIN == 0 )); then
    train_ppo
  fi
  if (( WITH_VIS == 1 )); then
    visualize_ppo_latest
  fi
  echo "Wizard single-RL run complete (PPO)."
  exit 0
fi

# Placeholder for future PQN support.
echo "PQN mode requested, but no PQN runner is wired in this repo yet." >&2
exit 2
