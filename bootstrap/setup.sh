#!/usr/bin/env bash
# yue2_lora_finetuning — Colab bootstrap
#
# Run this BACKGROUNDED while you do `./code tunnel` auth in the foreground,
# so the auth wait and the install/download time overlap:
#
#   bash bootstrap/setup.sh > /content/logs/setup.log 2>&1 &
#   ./code tunnel
#
# Deliberately non-interactive — nothing here should prompt. HF_TOKEN and
# GCP_DATASET_PATH are staged into a plain file by the notebook cell
# (bootstrap/colab_cell.md) BEFORE this script runs, since `userdata` only
# exists inside the notebook kernel, not in a terminal shell. GCP auth
# itself happens separately via `gcloud auth login` in that same cell.

set -e
mkdir -p /content/logs
source /root/.secrets.env

# Anchor every ComfyUI path to the repo this script lives in, so the script
# behaves the same whether invoked from the repo root or elsewhere.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY="$REPO_ROOT/ComfyUI"
STAGING="/content/staging"

echo "=== $(date) — yue2 bootstrap starting ==="

# --- Fast, synchronous: credentials everything else depends on ---
# GCP auth already happened in the notebook cell (`gcloud auth login`),
# which persists to ~/.config/gcloud/ on disk — nothing to do here for GCS.
hf auth login --token "$HF_TOKEN"

# --- Launch every independent, slow task in parallel ---
pids=()
names=()

start_job() {
  local name="$1"; shift
  ("$@") > "/content/logs/${name}.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
}

start_job opencode \
  bash -c 'curl -fsSL https://opencode.ai/install | bash'

# Download into a staging dir, NOT into ComfyUI/. The ComfyUI clone runs in
# parallel; if this job created ComfyUI/ first, `git clone` would either be
# skipped by a `[ -d ComfyUI ]` guard or fail on a non-empty destination.
# Staging keeps the two jobs from racing over the same directory.
start_job checkpoint \
  bash -c 'mkdir -p "'"$STAGING"'/models" && \
           hf download Comfy-Org/YuE2 checkpoints/yue2_3b_bf16.safetensors \
             --local-dir "'"$STAGING"'/models"'

start_job dataset \
  bash -c 'mkdir -p /content/data && \
           gsutil -m cp -r "'"$GCP_DATASET_PATH"'" /content/data'

start_job comfyui \
  bash -c 'if [ ! -f "'"$COMFY"'/main.py" ]; then rm -rf "'"$COMFY"'" && \
             git clone https://github.com/comfyanonymous/ComfyUI.git "'"$COMFY"'"; fi && \
           cd "'"$COMFY"'" && pip install -q -r requirements.txt'

echo "Launched in parallel: ${names[*]}"
echo "tail -f /content/logs/<name>.log to watch any one of these live."

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[ok]   ${names[$i]}"
  else
    echo "[FAIL] ${names[$i]} — see /content/logs/${names[$i]}.log"
    fail=1
  fi
done

# --- Merge the staged checkpoint into the now-cloned ComfyUI ---
mkdir -p "$COMFY/models/checkpoints"
if [ -f "$STAGING/models/checkpoints/yue2_3b_bf16.safetensors" ]; then
  mv -f "$STAGING/models/checkpoints/yue2_3b_bf16.safetensors" "$COMFY/models/checkpoints/"
else
  echo "[WARN] staged checkpoint not found — checkpoint job failed, see log above"
fi

# --- Trainer install: the one real dependency (needs ComfyUI/custom_nodes/) ---
echo "Installing trainer (depended on ComfyUI job above)..."
mkdir -p "$COMFY/custom_nodes"
cd "$COMFY/custom_nodes"
if [ ! -d ComfyUI-YuE2-Trainer ]; then
  git clone https://github.com/speedyrulz/ComfyUI-YuE2-Trainer.git
fi
cd ComfyUI-YuE2-Trainer
pip install -q -r requirements.txt

# --- Verify what actually landed, don't just trust exit codes ---
echo "=== Verifying downloads ==="

ckpt="$COMFY/models/checkpoints/yue2_3b_bf16.safetensors"
if [ -f "$ckpt" ]; then
  size=$(stat -c%s "$ckpt")
  if [ "$size" -lt 1000000000 ]; then
    echo "[FAIL] checkpoint is only $size bytes — looks truncated, re-run the checkpoint download"
    fail=1
  else
    echo "[ok]   checkpoint present, $((size / 1024 / 1024)) MB"
  fi
else
  echo "[FAIL] checkpoint missing at $ckpt"
  fail=1
fi

# gsutil cp -r nests the source folder under the destination, so if
# GCP_DATASET_PATH ends in .../dataset, this lands at /content/data/dataset/
train_count=$(find /content/data -path "*/train/*.mp3" 2>/dev/null | wc -l)
val_count=$(find /content/data -path "*/val/*.mp3" 2>/dev/null | wc -l)
echo "dataset: train=$train_count val=$val_count (last verified build was 238/18 — corpus can drift between sessions, so a different number isn't automatically wrong, just re-run verify_dataset.py if unsure)"
if [ "$train_count" -eq 0 ] || [ "$val_count" -eq 0 ]; then
  echo "[FAIL] dataset looks empty — check the unzip step"
  fail=1
fi

if [ "$fail" -eq 1 ]; then
  echo "=== setup.sh finished WITH FAILURES — check the [FAIL] lines above ==="
  exit 1
fi
echo "=== setup.sh done — dataset in /content/data, checkpoint in $COMFY/models/checkpoints ==="
