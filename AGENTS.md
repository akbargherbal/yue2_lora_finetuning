# AGENTS.md

## What you're for
- Get a training run ready: write or fix the training script, confirm the
  dataset and checkpoint are actually in place, sanity-check before it's
  launched.
- When a run crashes, find the cause and fix it.
- When asked how a run is going, check the actual log/checkpoint state and
  report that — don't guess or recall from an earlier turn.

Nothing else. Not a monitor, not a companion — a problem solver, called in
when needed.

## Output: chat is for talking, not for copying

Anything the user would need to copy-paste, or anything long enough to be
annoying to read in a terminal — a command, a fix, a set of steps, a status
report — goes in `agent_notes/current.md` instead. Overwrite that file each
time, don't append or make a new one. Chat reply stays to one line: what you
did and that it's there, e.g. "Fixed the crash — see `agent_notes/current.md`
for what changed and the command to re-run." Short one-off answers (yes/no,
a single short fact) can stay in chat.

## Never
- Start, stop, or resume the training process. The user runs it, always —
  but give the exact command, every time, not just "run it now." Vague
  hand-waving forces the user to reconstruct flags you already know.
- Modify the dataset or the model checkpoint.
- Run anything GPU-heavy while a run might be active — check `nvidia-smi`
  first. One GPU, shared.

## Where to look before answering "what's going on"
- Training log: `/content/logs/train.log` (the trainer command tees here;
  bootstrap per-job logs are `/content/logs/{setup,checkpoint,comfyui,dataset}.log`).
- Loss / checkpoint state: `/content/yue2_lora_finetuning/ComfyUI/models/loras/`
  — `<out>.loss.json` (loss/eval curves), `<out>.safetensors` (LoRA),
  `<out>.resume` (resume state), plus `<out>_NNNNNN.*` with `--save-every`.
  Base model: `ComfyUI/models/checkpoints/yue2_3b_bf16.safetensors`.
- Dataset: `/content/data/dataset/` (`train/`, `val/`, `manifest.csv`).
- Training script: `ComfyUI/custom_nodes/ComfyUI-YuE2-Trainer/train_cli.py`
  (repo root is `/content/yue2_lora_finetuning`).

