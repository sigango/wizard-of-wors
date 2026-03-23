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
SKIP_PREFLIGHT=0
SHOW_SYSTEM_INFO=1

NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
NUM_VIS_EPISODES="${NUM_VIS_EPISODES:-5}"

usage() {
  cat <<'EOF'
Usage:
  scripts/benchmarks/run_report_wizard_of_wor.sh [--quick] [--macsafe] [--with-visualize]
                                                  [--dry-run] [--skip-train]
                                                  [--skip-preflight] [--no-system-info]

Flags:
  --quick           Use test presets (10% steps) for quick validation.
  --macsafe         Use full-step macsafe presets.
  --with-visualize  After each training, auto-visualize latest model and record video.
  --dry-run         Print commands only.
  --skip-train      Skip training and only visualize latest checkpoints (requires --with-visualize).
  --skip-preflight  Skip dependency checks.
  --no-system-info  Do not print hardware/software summary.

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
    --skip-preflight)
      SKIP_PREFLIGHT=1
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

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

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
    local cpu_model cpu_count
    cpu_model="$(lscpu | awk -F: '/Model name/{print $2; exit}' | xargs)"
    cpu_count="$(lscpu | awk -F: '/^CPU\(s\)/{print $2; exit}' | xargs)"
    [[ -n "${cpu_model}" ]] && echo "CPU: ${cpu_model}"
    [[ -n "${cpu_count}" ]] && echo "CPU cores: ${cpu_count}"
  elif command -v sysctl >/dev/null 2>&1; then
    echo "CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
    echo "CPU cores: $(sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
  fi

  if command -v free >/dev/null 2>&1; then
    echo "RAM: $(free -h | awk '/^Mem:/{print $2}')"
  elif command -v sysctl >/dev/null 2>&1; then
    local ram_bytes
    ram_bytes="$(sysctl -n hw.memsize 2>/dev/null || true)"
    if [[ -n "${ram_bytes}" ]]; then
      echo "RAM: $(awk -v b="${ram_bytes}" 'BEGIN{printf "%.1f GB", b/1024/1024/1024}')"
    fi
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU(s):"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  - /'
    local cuda_ver
    cuda_ver="$(nvidia-smi | awk -F'CUDA Version: ' 'NR==1{print $2}' | awk '{print $1}')"
    [[ -n "${cuda_ver}" ]] && echo "CUDA Version: ${cuda_ver}"
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

check_pyproject_deps() {
  PYTHON_BIN_HINT="${PYTHON_BIN}" "${PYTHON_BIN}" - <<'PY'
import re
import sys
from pathlib import Path
from importlib import metadata

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        print("WARNING: tomllib/tomli not available; skipping pyproject dependency parsing.")
        sys.exit(0)

pp = Path("pyproject.toml")
if not pp.exists():
    print("WARNING: pyproject.toml not found; skipping dependency parsing.")
    sys.exit(0)

data = tomllib.loads(pp.read_text())
deps = data.get("project", {}).get("dependencies", [])
name_re = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
need = []
for req in deps:
    m = name_re.match(str(req))
    if not m:
        continue
    name = m.group(1).lower().replace("_", "-")
    if name not in need:
        need.append(name)

missing = []
for pkg in need:
    try:
        metadata.version(pkg)
    except metadata.PackageNotFoundError:
        missing.append(pkg)

if missing:
    py = sys.executable
    print("Missing pyproject dependencies:")
    for pkg in missing:
        print(f"  - {pkg}")
    print("\nRecommended install:")
    print(f"  {py} -m pip install -e .")
    sys.exit(2)
PY
}

check_python_modules() {
  local specs="$1"
  PY_REQ_SPECS="${specs}" PYTHON_BIN_HINT="${PYTHON_BIN}" "${PYTHON_BIN}" - <<'PY'
import os
import sys
import importlib.util

raw = os.environ.get("PY_REQ_SPECS", "")
missing = []
for item in [x.strip() for x in raw.split(",") if x.strip()]:
    if ":" in item:
        mod, pkg = item.split(":", 1)
    else:
        mod, pkg = item, item
    if importlib.util.find_spec(mod) is None:
        missing.append((mod, pkg))

if missing:
    py = sys.executable
    print("Missing required Python modules for Wizard report pipeline:")
    for mod, pkg in missing:
        print(f"  - module '{mod}' (install package: {pkg})")
    pkgs = " ".join(sorted({pkg for _, pkg in missing}))
    print("\nInstall missing packages with:")
    print(f"  {py} -m pip install {pkgs}")
    sys.exit(3)
PY
}

run_preflight() {
  check_pyproject_deps
  local reqs="jax:jax,numpy:numpy,flax:flax,optax:optax,pandas:pandas,matplotlib:matplotlib,tqdm:tqdm,wandb:wandb,pygame:pygame,ocatari:ocatari,jaxatari:jaxatari"
  if (( WITH_VIS == 1 )); then
    reqs="${reqs},cv2:opencv-python"
  fi
  check_python_modules "${reqs}"
}

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

if (( SHOW_SYSTEM_INFO == 1 )); then
  print_report_system_info
fi

if (( SKIP_PREFLIGHT == 0 )); then
  run_preflight
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
