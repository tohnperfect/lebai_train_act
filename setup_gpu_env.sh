#!/usr/bin/env bash
# Bootstrap a Python venv on a Linux GPU box for training ACT locally.
#
# Usage:
#     chmod +x setup_gpu_env.sh
#     ./setup_gpu_env.sh
#
# After the script completes, activate the env and run training:
#     source .venv/bin/activate
#     python train_local_gpu.py --smoke-test     # verify the install
#     python train_local_gpu.py                  # real training

set -euo pipefail

# 1. Find a Python >= 3.10
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        version=$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$candidate"
            echo "Using $PYTHON (Python $version)"
            break
        fi
    fi
done
if [ -z "$PYTHON" ]; then
    echo "ERROR: no Python >= 3.10 found. Install python3.10+ first." >&2
    exit 1
fi

# 2. Create venv at .venv if it doesn't already exist
if [ -d ".venv" ]; then
    echo ".venv already exists — reusing. Delete it manually if you want a fresh install."
else
    echo "Creating .venv ..."
    "$PYTHON" -m venv .venv
fi

# 3. Activate and upgrade pip
# shellcheck source=/dev/null
source .venv/bin/activate
pip install --upgrade pip

# 4. Install deps. lerobot pulls torch 2.10. On Linux x86_64 with CUDA-capable
# drivers, the default torch wheel from PyPI is CUDA 12.x. If you need CUDA 11.8
# install it first (see comment in requirements_gpu.txt).
echo "Installing deps from requirements_gpu.txt ..."
pip install -r requirements_gpu.txt

# 5. Smoke-test the install
echo ""
echo "=== Verifying install ==="
python - <<'PY'
import sys
import torch
print(f"Python:        {sys.version.split()[0]}")
print(f"torch:         {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA devices:  {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  ({p.total_memory/1e9:.1f} GB)")
else:
    print("WARNING: no CUDA GPU detected. Training will fall back to CPU and be very slow.")

import lerobot
print(f"lerobot:       {getattr(lerobot, '__version__', 'unknown')}")
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.types import FeatureType, PolicyFeature
print("lerobot imports: OK")
PY

echo ""
echo "Setup complete. Next steps:"
echo "  source .venv/bin/activate"
echo "  python train_local_gpu.py --smoke-test    # 200 steps, sanity check"
echo "  python train_local_gpu.py                 # full training"
