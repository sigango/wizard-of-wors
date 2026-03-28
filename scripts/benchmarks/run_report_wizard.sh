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
  CUDA_MAJOR="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\)\..*/\1/p' | head -n 1 || true)"
  if [[ "${CUDA_MAJOR}" == "13" ]]; then
    DEFAULT_JAX_PIP_SPEC="jax[cuda13]"
  else
    DEFAULT_JAX_PIP_SPEC="jax[cuda12]"
  fi
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
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RESULTS_DIR="${REPO_ROOT}/results"
RUN_MANIFEST_CSV="${RESULTS_DIR}/wizard_run_manifest_${RUN_TIMESTAMP}.csv"
RUN_TIMINGS_CSV="${RESULTS_DIR}/wizard_stage_timing_${RUN_TIMESTAMP}.csv"
REPORT_BUNDLE_DIR="${RESULTS_DIR}/wizardofwor_report_bundle_${RUN_TIMESTAMP}"

usage() {
  cat <<'EOF'
Usage:
  scripts/benchmarks/run_report_wizard.sh [options]

Options:
  --algo ALG             ALG in {ppo,pqn}. Default: ppo.
  --obs MODE             MODE in {object,pixel,both}. Default: pixel.
  --rtx5090              Force non-macsafe final preset for NVIDIA RTX 5090 runs.
  --num-envs N           Override NUM_ENVS for selected preset.
  --num-steps N          Override NUM_STEPS for selected preset.
  --num-minibatches N    Override NUM_MINIBATCHES for selected preset.
  --total-steps N        Override TOTAL_TIMESTEPS for selected preset.
  --update-epochs N      Override UPDATE_EPOCHS for selected preset.
  --quick                Use test preset (10% style run).
  --macsafe              Use macsafe final preset.
  --with-visualize       Record visualization (MP4) after training with latest checkpoint.
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
  object|pixel|both) ;;
  *)
    echo "Invalid --obs '${OBS_MODE}'. Use object|pixel|both." >&2
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

if (( WITH_VIS == 1 )) && [[ -z "${DISPLAY:-}" ]]; then
  export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
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

jax_runtime_ok() {
  local expect_gpu="$1"
  JAX_EXPECT_GPU="${expect_gpu}" "${PYTHON_BIN}" - <<'PY'
import os
import sys

expect_gpu = os.environ.get("JAX_EXPECT_GPU", "0") == "1"
try:
    import jax
    backend = jax.default_backend()
    devices = [d.platform for d in jax.devices()]
    print(f"JAX backend: {backend}")
    print(f"JAX devices: {devices}")
except Exception as exc:
    print(f"JAX runtime check failed: {exc}")
    sys.exit(1)

if expect_gpu and "gpu" not in devices and backend != "gpu":
    print("GPU detected by system, but JAX is not using GPU.")
    sys.exit(2)
PY
}

ensure_jax_runtime() {
  local expect_gpu=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    expect_gpu=1
  fi

  if jax_runtime_ok "${expect_gpu}"; then
    return 0
  fi

  if (( AUTO_INSTALL_DEPS == 1 )); then
    echo "Installing JAX runtime package: ${JAX_PIP_SPEC}"
    "${PYTHON_BIN}" -m pip install "${JAX_PIP_SPEC}"
    if ! jax_runtime_ok "${expect_gpu}"; then
      echo "JAX runtime still not ready after install. Try setting JAX_PIP_SPEC manually." >&2
      echo "Example: JAX_PIP_SPEC='jax[cuda12]' or JAX_PIP_SPEC='jax[cuda13]'" >&2
      exit 3
    fi
  else
    echo "JAX runtime check failed and auto-install is disabled." >&2
    echo "Install manually with: ${PYTHON_BIN} -m pip install '${JAX_PIP_SPEC}'" >&2
    exit 3
  fi
}

ensure_ocatari_runtime() {
  if (( AUTO_INSTALL_DEPS != 1 )); then
    return 0
  fi

  echo "Ensuring OCAtari is installed..."
  if "${PYTHON_BIN}" -m pip install -U ocatari; then
    return 0
  fi

  echo "OCAtari wheel install failed, trying source install from GitHub..."
  if "${PYTHON_BIN}" -m pip install -U "git+https://github.com/k4ntz/OC_Atari.git"; then
    return 0
  fi

  echo "Failed to install OCAtari automatically." >&2
  echo "Please install manually, for example:" >&2
  echo "  ${PYTHON_BIN} -m pip install -U ocatari" >&2
  echo "  ${PYTHON_BIN} -m pip install -U git+https://github.com/k4ntz/OC_Atari.git" >&2
  exit 3
}

if (( SHOW_SYSTEM_INFO == 1 )); then
  print_report_system_info
fi

if (( SKIP_PREFLIGHT == 0 )); then
  ensure_jax_runtime
  ensure_ocatari_runtime
  reqs="jax:${JAX_PIP_SPEC},jaxlib:jaxlib,numpy:numpy,scipy:scipy,ml_dtypes:ml-dtypes,typing_extensions:typing-extensions,absl:absl-py,toolz:toolz,opt_einsum:opt-einsum,chex:chex,gymnax:gymnax,gymnasium:gymnasium,ale_py:ale-py,flax:flax,optax:optax,distrax:distrax,tensorflow_probability:tensorflow-probability,pandas:pandas,matplotlib:matplotlib,tqdm:tqdm,wandb:wandb,pygame:pygame,ocatari:ocatari,cv2:opencv-python,imageio:imageio,PIL:pillow,jinja2:jinja2,psutil:psutil,hydra:hydra-core,omegaconf:omegaconf,safetensors:safetensors"
  check_modules "${reqs}"
fi

resolve_preset() {
  local mode="$1"
  if (( QUICK == 1 )); then
    if [[ "${mode}" == "object" ]]; then
      printf '%s\n' "config_wizard_object_test"
    else
      printf '%s\n' "config_wizard_pixel_test"
    fi
  elif (( MACSAFE == 1 )); then
    if [[ "${mode}" == "object" ]]; then
      printf '%s\n' "config_wizard_object_final_macsafe"
    else
      printf '%s\n' "config_wizard_pixel_final_macsafe"
    fi
  else
    if [[ "${mode}" == "object" ]]; then
      printf '%s\n' "config_wizard_object_final"
    else
      printf '%s\n' "config_wizard_pixel_final"
    fi
  fi
}

PRESETS=()
if [[ "${OBS_MODE}" == "both" ]]; then
  PRESETS+=("$(resolve_preset object)")
  PRESETS+=("$(resolve_preset pixel)")
else
  PRESETS+=("$(resolve_preset "${OBS_MODE}")")
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

latest_ppo_result_dir() {
  ls -td "${REPO_ROOT}"/results/ppo_distrax_jax_wizardofwor_* 2>/dev/null | head -n 1 || true
}

obs_mode_from_preset() {
  local preset="$1"
  if [[ "${preset}" == *object* ]]; then
    printf '%s\n' "object"
  else
    printf '%s\n' "pixel"
  fi
}

train_ppo() {
  local preset="$1"
  local cmd=(
    "${PYTHON_BIN}" scripts/benchmarks/agent_performance_comparison.py
    --mode train-jaxatari \
    --preset "${preset}" \
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

eval_ppo_model() {
  local preset="$1"
  local model="$2"
  run_cmd "${PYTHON_BIN}" scripts/benchmarks/agent_performance_comparison.py \
    --mode eval \
    --preset "${preset}" \
    --env_type jax \
    --model_path "${model}" \
    --num_episodes "${NUM_EVAL_EPISODES}"
}

visualize_ppo_model() {
  local preset="$1"
  local model="$2"
  run_cmd "${PYTHON_BIN}" scripts/benchmarks/agent_performance_comparison.py \
    --mode visualize \
    --preset "${preset}" \
    --env_type jax \
    --model_path "${model}" \
    --num_episodes "${NUM_VIS_EPISODES}"
}

echo "Wizard of Wor single-RL report runner"
echo "algo=${ALGO}, obs=${OBS_MODE}, presets=${PRESETS[*]}"
echo "quick=${QUICK}, macsafe=${MACSAFE}, rtx5090=${RTX5090_PROFILE}, with_visualize=${WITH_VIS}, skip_train=${SKIP_TRAIN}, dry_run=${DRY_RUN}"
echo "JAX auto-install package: ${JAX_PIP_SPEC}"

if (( DRY_RUN == 0 )); then
  mkdir -p "${RESULTS_DIR}"
  printf 'preset,obs_mode,result_dir,model_path\n' > "${RUN_MANIFEST_CSV}"
  printf 'preset,obs_mode,stage,duration_seconds,result_dir\n' > "${RUN_TIMINGS_CSV}"
fi

if [[ "${ALGO}" == "ppo" ]]; then
  for preset in "${PRESETS[@]}"; do
    obs_mode="$(obs_mode_from_preset "${preset}")"
    echo ""
    echo "=== PPO run for preset=${preset} ==="
    if (( SKIP_TRAIN == 0 )); then
      train_start="$(date +%s)"
      train_ppo "${preset}"
      train_end="$(date +%s)"
      train_duration="$((train_end - train_start))"

      if (( DRY_RUN == 1 )); then
        result_dir="<latest_result_dir>"
        model_path="<latest_ppo_model_params.npz>"
      else
        result_dir="$(latest_ppo_result_dir)"
        if [[ -z "${result_dir}" ]]; then
          echo "Could not locate latest PPO result directory after training." >&2
          exit 3
        fi
        model_path="${result_dir}/ppo_distrax_model_params.npz"
        [[ -f "${model_path}" ]] || {
          echo "Missing model file after training: ${model_path}" >&2
          exit 3
        }
        printf '%s,%s,%s,%s,%s\n' "${preset}" "${obs_mode}" "${result_dir}" "${model_path}" >> "${RUN_MANIFEST_CSV}"
        printf '%s,%s,train,%s,%s\n' "${preset}" "${obs_mode}" "${train_duration}" "${result_dir}" >> "${RUN_TIMINGS_CSV}"
      fi

      eval_start="$(date +%s)"
      eval_ppo_model "${preset}" "${model_path}"
      eval_end="$(date +%s)"
      eval_duration="$((eval_end - eval_start))"
      if (( DRY_RUN == 0 )); then
        printf '%s,%s,eval,%s,%s\n' "${preset}" "${obs_mode}" "${eval_duration}" "${result_dir}" >> "${RUN_TIMINGS_CSV}"
      fi
    fi
    if (( WITH_VIS == 1 )); then
      if (( DRY_RUN == 1 )); then
        model_path="<latest_ppo_model_params.npz>"
        result_dir="<latest_result_dir>"
      elif [[ -z "${model_path:-}" ]]; then
        result_dir="$(latest_ppo_result_dir)"
        model_path="${result_dir}/ppo_distrax_model_params.npz"
      fi
      vis_start="$(date +%s)"
      visualize_ppo_model "${preset}" "${model_path}"
      vis_end="$(date +%s)"
      vis_duration="$((vis_end - vis_start))"
      if (( DRY_RUN == 0 )); then
        printf '%s,%s,visualize,%s,%s\n' "${preset}" "${obs_mode}" "${vis_duration}" "${result_dir}" >> "${RUN_TIMINGS_CSV}"
      fi
    fi
  done

  if (( DRY_RUN == 0 )) && (( SKIP_TRAIN == 0 )); then
    run_cmd "${PYTHON_BIN}" scripts/benchmarks/wizard_report.py \
      --manifest-csv "${RUN_MANIFEST_CSV}" \
      --timings-csv "${RUN_TIMINGS_CSV}" \
      --output-dir "${REPORT_BUNDLE_DIR}"
    echo "Report bundle generated at: ${REPORT_BUNDLE_DIR}"
  fi
  echo "Wizard RL run complete (PPO)."
  exit 0
fi

# Placeholder for future PQN support.
echo "PQN mode requested, but no PQN runner is wired in this repo yet." >&2
exit 2
