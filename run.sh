#!/bin/bash
# Run any python script in the traffiq environment.
# Usage: ./run.sh track_sim.py
#        ./run.sh -c "from Model import Model; m = Model(); m.load()"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/yashagarwal/miniforge/envs/traffiq/bin/python"

export PYTHONPATH="$DIR:$PYTHONPATH"
exec "$PYTHON" "$@"
