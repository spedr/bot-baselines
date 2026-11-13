#!/usr/bin/env bash
# Run ON the rented instance, from the repo root, after `push.sh code` and `push.sh data`:
#   bash tools/vast/setup_instance.sh [--tests]
# Installs uv and the locked environment, checks every GPU, checks the data, and prints the
# runner's plan for this machine. --tests also runs the method test suite on GPU 0 (a few minutes).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"

command -v rsync >/dev/null && command -v tmux >/dev/null || {
  apt-get update -qq && apt-get install -y -qq rsync tmux >/dev/null; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen

echo "== GPUs"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
.venv/bin/python - <<'EOF'
import torch
n = torch.cuda.device_count()
print(f"torch {torch.__version__} cuda {torch.version.cuda}: {n} GPUs")
for i in range(n):
    x = torch.randn(4096, 4096, device=f"cuda:{i}")
    torch.cuda.synchronize(i)
    print(i, torch.cuda.get_device_name(i), "capability", torch.cuda.get_device_capability(i),
          "matmul ok" if torch.isfinite((x @ x).sum()).item() else "MATMUL FAILED")
EOF

echo "== CPU / RAM"
.venv/bin/python - <<'EOF'
import os, sys
sys.path.insert(0, "tools/vast")
from runner import effective_cpus, mem_gb
t, a = mem_gb()
print(f"usable CPUs {len(effective_cpus())} (affinity {len(os.sched_getaffinity(0))}), RAM total {t:.0f} GB, available {a:.0f} GB")
EOF

echo "== data"
.venv/bin/python -m bb status || echo "bb status failed: check data/"
for f in data/twibot22/emb/roberta-base/tweets_seq_first20.f16.npy data/twibot22/graph/edge_index.i64.npy \
         data/twibot20/emb/roberta-base/des.f16.npy data/twibot22/profile.parquet; do
  [[ -s $f ]] && echo "ok  $f" || echo "MISSING $f"
done

if [[ ${1:-} == --tests ]]; then
  echo "== tests (GPU 0)"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q -x tests/methods
fi

echo "== plan"
.venv/bin/python tools/vast/runner.py plan tools/vast/jobs_deferred.toml
echo
echo "Start in tmux:  tmux new -s runs '.venv/bin/python tools/vast/runner.py run tools/vast/jobs_deferred.toml 2>&1 | tee -a runs/vast/runner.log'"
