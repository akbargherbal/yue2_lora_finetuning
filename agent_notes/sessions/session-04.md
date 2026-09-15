# Session 4 — setup, dry run, first real run, and the setback

## Setup (after three glitches)

The first `bootstrap/setup.sh` run failed three ways; all fixed, and the second
run finished all four parallel jobs `[ok]`:

1. **Wrong checkpoint path** — `Comfy-Org/YuE2` has no root-level
   `yue2_3b_bf16.safetensors`; it lives under `checkpoints/`.
2. **ComfyUI clone race (two layers)** — a checkpoint job's `mkdir -p
   ComfyUI/models/checkpoints` created `ComfyUI/` before the clone ran, so the
   clone guard skipped it; and cloning into a non-empty dir would fail anyway.
   Fixed by staging downloads under `/content/staging` and moving them in after
   the clone, and guarding on `ComfyUI/main.py` (removing a stale partial).
3. **Stray `cd /content`** made post-install verification use wrong relative
   paths. Removed; all `ComfyUI/...` paths anchor to the script's repo root.

Verified on disk: `ComfyUI/main.py`, the trainer's `train_cli.py`,
`yue2_3b_bf16.safetensors` (7,438 MB), dataset 238 train / 18 val.

## Dry run

Completed in ~22 min, almost all of it the one-time VAE encode (cached
afterwards). Loaded the checkpoint, built the LoRA (112 modules, rank 16,
alpha 16, 224 tensors, 14.68 M trainable params), ran held-out eval at step 0
(loss 1.0510), and saved a 0-step artifact set. **The saved
`maqam_acoustic_v1.safetensors` was untrained — not a result.**

## Leak-free eval

`Item.id` is the filename stem, so each take is its own item, and
`--eval-holdout` samples items (not poems) via
`random.Random(seed + 4242)`. On this corpus 76/136 poems have >1 take and
76.6% of tracks share a poem with another, so a random holdout usually leaks.
Pointing `--data` at the parent `dataset/` merges train+val (all basenames
unique) and choosing a seed whose held-out items are single-take poems makes
those poems vanish from training. Verified seeds and the accept/reject table
are in `docs/architecture.md` §6.

## The real run (smoke test, not a fine-tune)

1500 acoustic steps in 450 s (~0.3 s/step), `--conditioning compact`, rank 16.
Held-out loss 1.0510 → 1.0078 (−4.1%), plateaued by ~250. 1500 × 30 s ≈ 12.5 h
vs an 18 h corpus = <1 epoch — a smoke test.

## The generation test and the setback

Headless ComfyUI on `127.0.0.1:8188` with `yue2_generate_with_lora_api.json`
(acoustic LoRA on the MODEL path), held-out poem `hijaz_0038_01-retake_take01`
(Imru' al-Qais, Hijaz), fed the dataset `.style.txt` + `.lyrics.txt`.

A/B at the same seed, `strength_model` 1.0 vs 0.0:

- 90 s: both budgets hit, truncation. `waveform 0.963 · MFCC 0.999`.
- 214 s: ABC finished on its own, music budget hit (hard cut at 3:34).
  `waveform 0.946 · chroma 0.996 · MFCC 0.998`.

**Findings**

1. The acoustic LoRA is a near-no-op at this scale. It is not the style
   transfer the user wants and cannot touch pronunciation or maqam.
2. Pronunciation is decided upstream by the planner's semantic tokens — the
   base and LoRA renders were fed the identical token stream (server log: only
   2 ABC + 2 music sampling passes for 4 runs), so the acoustic LoRA can
   neither cause nor fix the ق problem. The base planner also produced **K:Fm**
   (Western), not Hijaz.

**User verdict:** too melismatic to an artificial level, with ق
mispronunciation — a regression relative to the base model they had tested.
The acceptance bar became base-model pronunciation parity.

## Checkpointing / resumability / GCS

Decisions and operational detail are in `docs/architecture.md` §7. In short:
`--save-every N` in optimizer steps writes LoRA + `.resume`; resume is the same
command plus `--existing-lora`; Ctrl-C does not run the final save; GCS mirror
runs in a second terminal via `backup_to_gcp.py`.

## Runtime at session end (ephemeral)

ComfyUI server PID 81132 (~7.5 GB VRAM) and `backup_to_gcp.py --include-cache`
PID 64625. Both die with the Colab runtime.
