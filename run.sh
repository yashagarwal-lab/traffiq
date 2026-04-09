#!/bin/bash
# TraffIQ launcher — sets up CUDA libraries and runs with Python 3.11
#
# Usage:
#   ./run.sh -m src.training.train --backbone v3_small
#   ./run.sh -m src.data.collect --output data/raw/session_001
#   ./run.sh -m src.deploy.main --model models/traffiq_int8.tflite --dry-run
#   ./run.sh script.py  (run a script directly)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/yashagarwal/miniforge/envs/traffiq/bin/python"
NVIDIA_BASE="/home/yashagarwal/miniforge/envs/traffiq/lib/python3.11/site-packages/nvidia"

export LD_LIBRARY_PATH="$NVIDIA_BASE/cudnn/lib:$NVIDIA_BASE/cublas/lib:$NVIDIA_BASE/cuda_runtime/lib:$NVIDIA_BASE/cufft/lib:$NVIDIA_BASE/curand/lib:$NVIDIA_BASE/cusolver/lib:$NVIDIA_BASE/cusparse/lib:$NVIDIA_BASE/nvjitlink/lib:$NVIDIA_BASE/cuda_nvrtc/lib:$NVIDIA_BASE/nccl/lib:${LD_LIBRARY_PATH:-}"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export TF_CPP_MIN_LOG_LEVEL=1  # Reduce TF noise

exec "$PYTHON" "$@"
