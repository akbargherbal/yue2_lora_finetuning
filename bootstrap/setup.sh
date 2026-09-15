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

# Pin both external repos. ComfyUI at the commit the cfg_scale patch applies
# against; the trainer at the commit this project was verified on. Tracking
# HEAD here previously meant a silent quality regression whenever upstream
# changed -- now the patch step hard-fails instead.
COMFY_COMMIT="36da3ff763687eab86a35e1019995dd1fb369b0d"
TRAINER_COMMIT="544c0010bdddef14e298de1d4555558aa848451f"

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

# The community audio -> semantic-token head YuE2 never shipped (Mothersuperior
# v4). Staged like the checkpoint so the parallel ComfyUI clone cannot wipe it.
start_job tokenizer \
  bash -c 'mkdir -p "'"$STAGING"'/audio_encoders" && \
           hf download Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4 \
             tokenizer_head_joint_v4.pt \
             --local-dir "'"$STAGING"'/audio_encoders"'

start_job dataset \
  bash -c 'mkdir -p /content/data && \
           gsutil -m cp -r "'"$GCP_DATASET_PATH"'" /content/data'

start_job comfyui \
  bash -c 'if [ ! -f "'"$COMFY"'/main.py" ]; then rm -rf "'"$COMFY"'" && \
             git clone https://github.com/comfyanonymous/ComfyUI.git "'"$COMFY"'"; fi && \
           git -C "'"$COMFY"'" fetch --quiet --all --tags && \
           git -C "'"$COMFY"'" checkout --quiet "'"$COMFY_COMMIT"'" && \
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

mkdir -p "$COMFY/models/audio_encoders"
if [ -f "$STAGING/audio_encoders/tokenizer_head_joint_v4.pt" ]; then
  mv -f "$STAGING/audio_encoders/tokenizer_head_joint_v4.pt" "$COMFY/models/audio_encoders/"
else
  echo "[WARN] staged tokenizer head not found — tokenizer job failed, see /content/logs/tokenizer.log"
fi

# --- cfg_scale generation fix (docs/architecture.md §7): the YuE2 generation
# nodes never forwarded text-guidance scale. Committed as a patch so a fresh
# clone gets it; idempotent so re-running the bootstrap (or an upstream that
# already fixed it) is a no-op rather than an error. ComfyUI is pinned to the
# commit this patch applies against, so a failure here is a real error, not a
# warning: renders would silently lose cfg_scale.
if git -C "$COMFY" apply --reverse --check "$REPO_ROOT/bootstrap/yue2_cfg_scale.patch" 2>/dev/null; then
  echo "[ok]   cfg_scale node patch already applied"
elif git -C "$COMFY" apply --check "$REPO_ROOT/bootstrap/yue2_cfg_scale.patch" 2>/dev/null; then
  git -C "$COMFY" apply "$REPO_ROOT/bootstrap/yue2_cfg_scale.patch" && echo "[ok]   applied cfg_scale node patch"
else
  echo "[FAIL] cfg_scale node patch did not apply against ComfyUI $COMFY_COMMIT — nodes_yue2.py is not what the patch expects"
  fail=1
fi

# --- Trainer install: the one real dependency (needs ComfyUI/custom_nodes/) ---
echo "Installing trainer (depended on ComfyUI job above)..."
mkdir -p "$COMFY/custom_nodes"
cd "$COMFY/custom_nodes"
if [ ! -d ComfyUI-YuE2-Trainer ]; then
  git clone https://github.com/speedyrulz/ComfyUI-YuE2-Trainer.git
fi
cd ComfyUI-YuE2-Trainer
git fetch --quiet --all --tags && git checkout --quiet "$TRAINER_COMMIT"
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

head="$COMFY/models/audio_encoders/tokenizer_head_joint_v4.pt"
if [ -f "$head" ]; then
  echo "[ok]   tokenizer head present, $(( $(stat -c%s "$head") / 1024 / 1024 )) MB"
else
  echo "[FAIL] tokenizer head missing at $head — semantic-token training cannot run"
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
