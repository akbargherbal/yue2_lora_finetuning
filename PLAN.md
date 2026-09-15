# PLAN — YuE2-3B Maqam Style LoRA (Colab run plan)

> **Rewritten from scratch, session 6.** The previous `PLAN.md` was a
> dataset-prep plan whose §0 rested on the fabricated "ai-toolkit supports
> YuE2 (PR #1042)" claim (see `context.md` §2) — it is superseded, not
> amended. This file is the forward-looking plan for what to do **when we move
> to Colab and start fine-tuning**. Read `context.md` first for the full
> history; this is the action plan.

## 0. Where we are

- **Dataset: done and verified.** `/content/data/dataset/` = 256 tracks (238
  train / 18 val), uniform 48 kHz stereo MP3, 18.0 h total, poem-safe split,
  `manifest.csv` cross-checked. `prepare_dataset.py` / `verify_dataset.py`
  own it. No audio cleanup or segmentation is needed (context.md §6).
- **Backend: installed and sane.** `speedyrulz/ComfyUI-YuE2-Trainer` +
  `yue2_3b_bf16.safetensors`. Its own tests check the training forward matches
  ComfyUI inference (cosine 0.9999) and that LoRA keys load with zero
  unmatched keys.
- **One acoustic run happened and it told us something:** 1500 steps,
  `--conditioning compact`, near-no-op (`waveform corr 0.946`). That was the
  *wrong half* of the model in the *wrong conditioning mode* (context.md §14).
  It is a smoke test, not a fine-tune, and it is not to be repeated as-is.
- **The missing piece is found and already integrated** (context.md §16):
  `Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4`, the audio →
  semantic-token encoder YuE2 never shipped. The trainer can run it
  (`--semantic-head`) to write `<song>.semantic.npy` for our own recordings.

## 1. The core correction: which half does what

| Half | LoRA | Trains | Controls |
|---|---|---|---|
| AR / planner | `planner` (CLIP) | next-token over `style + lyrics → ABC` and/or `→ semantic tokens` | **composition**: melody, harmony, structure, maqam — and the semantic tokens that pronunciation comes from |
| NAR / acoustic | `acoustic` (MODEL) | flow-matching on VAE latents | render only: timbre, instrument sound, mix, production |

**The goal of this project is a maqam + pronunciation + style change. That lives
in the planner.** The acoustic LoRA is the "same song, re-recorded" (both
sources agree). So: **planner first, acoustic second.** The old acoustic-first
ordering is reversed.

## 2. Why the planner was blocked, and what unblocked it

YuE2 shipped with no encoder that turns a real recording into semantic tokens,
so real songs had no teachable target on the planner's semantic path. The only
planner route was ABC scores via SheetSage2 — a poor fit for our median-253s /
max-370s tracks and unverified on melismatic Arabic.

`Mothersuperior`'s head fixes exactly that. It is **already wired in**:

- Node: **YuE2 Semantic Tokens (community head)**.
- CLI: **`--semantic-head <file>`** — predicts tokens for every song and
  writes `<song>.semantic.npy` (skips existing sidecars unless
  `--semantic-force`).

Consequence: the planner can now train on **semantic tokens with no scores at
all** (`--semantic --no-abc`). The ABC/SheetSage2 blocker is off the critical
path. Its accuracy is approximate (~16% exact top-1 on YuE2's own songs, ~95%
by ear on round-trips) — treat the labels as noisy but usable.

## 3. Pre-flight in Colab (bootstrap additions)

`bootstrap/setup.sh` must also fetch the head. Add next to the checkpoint
download job (uses the `$COMFY` variable already defined there):

```bash
start_job tokenizer \
  bash -c 'mkdir -p "'"$COMFY"'/models/audio_encoders" && \
           hf download Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4 \
             tokenizer_head_joint_v4.pt \
             --local-dir "'"$COMFY"'/models/audio_encoders"'
```

MERT-v2-FullSong (~630 MB) can be auto-fetched via `--mert m-a-p/MERT-v2-FullSong`
(default), or pre-downloaded to a folder and passed as `--mert <folder>`.

Verify before training:

```bash
ls -la /content/yue2_lora_finetuning/ComfyUI/models/audio_encoders/tokenizer_head_joint_v4.pt
```

License: both the head and YuE2 weights are **CC BY-NC 4.0 — non-commercial**.

## 4. Stage 1 — tokenize the corpus (one-time)

Runs the head over every track and writes the `.semantic.npy` sidecars next to
the existing `.style.txt` / `.lyrics.txt`. Do it as a `--dry-run` so it also
populates the VAE-latent cache (needed anyway) and touches no weights.

```bash
cd /content/yue2_lora_finetuning
python ComfyUI/custom_nodes/ComfyUI-YuE2-Trainer/train_cli.py planner \
  --comfy-root /content/yue2_lora_finetuning/ComfyUI \
  --checkpoint yue2_3b_bf16.safetensors \
  --data /content/data/dataset \
  --semantic-head tokenizer_head_joint_v4.pt \
  --semantic --no-abc \
  --eval-holdout 5 --seed 2002 \
  --steps 100 --out maqam_planner_v1 --dry-run 2>&1 | tee -a /content/logs/train.log
```

`--eval-holdout 5 --seed 2002` is the leak-free held-out seed from session 4
(context.md §11) — carry it into every run.

**Smoke-test before trusting it:** tokenize 2–3 tracks, then render the token
round-trip with the acoustic stage (the `yue2_render_tokens_api.json` workflow,
or `--sample-tokens <song>.semantic.npy` on an acoustic run) and listen. If the
round trip keeps rhythm/harmony on our Arabic tracks, proceed; if it is noise,
stop and revisit.

## 5. Stage 2 — planner / AR LoRA (the maqam lever)

Start with semantic-only training behind a 50/50 sheet/no-sheet prompt, so the
resulting LoRA serves the normal `YuE2GenerateMusic` path, and use the KL trust
region instead of the ABC regularization set (which we don't have scores for).

```bash
cd /content/yue2_lora_finetuning
python ComfyUI/custom_nodes/ComfyUI-YuE2-Trainer/train_cli.py planner \
  --comfy-root /content/yue2_lora_finetuning/ComfyUI \
  --checkpoint yue2_3b_bf16.safetensors \
  --data /content/data/dataset \
  --semantic-head tokenizer_head_joint_v4.pt \
  --semantic --no-abc --abc-dropout 0.5 --kl-weight 0.5 \
  --max-tokens 4096 \
  --rank 32 --alpha 32 --lr 5e-5 --lr-schedule cosine \
  --steps 100 --save-every 10 --eval-every 10 \
  --eval-holdout 5 --eval-samples 8 --seed 2002 \
  --probe-every 10 --probe-max-tokens 8192 \
  --out maqam_planner_v1 2>&1 | tee -a /content/logs/train.log
```

**These numbers are starting values, not proven defaults** — the point of
`--save-every 10` + probes is to pick empirically. What the README says to
watch:

- The planner **learns fast**: 25–100 steps at `5e-5` already reshape the
  writing. Do **not** default to thousands of steps.
- Held-out eval minimum is **advisory**; the checkpoint that sounds closest to
  the album is often a little past it. Pick by ear.
- `--probe-every`: a probe that runs to `--probe-max-tokens` **without ending**
  is over-training (or it learned album-length scores) — use an earlier
  checkpoint. The last probe that ended normally is the latest worth keeping.
- `kl` starts near 0 and should level off at a few hundredths, not climb.

Nora's own cap for the analogous rank-64 AR LoRA is **~1500 steps ("past that
it memorizes")** — another reason to start small and extend only while the
eval + probes still improve. Extend with the same command plus
`--existing-lora maqam_planner_v1_000100.safetensors` (total `--steps`).

## 6. Stage 3 — acoustic LoRA (timbre), only after the planner proves out

This is the same acoustic half that was a near-no-op before, but now in the
mode that matches inference (`--conditioning inference_like --use-semantic`),
which is the fix for the mismatch that made the first run useless.

```bash
cd /content/yue2_lora_finetuning
python ComfyUI/custom_nodes/ComfyUI-YuE2-Trainer/train_cli.py acoustic \
  --comfy-root /content/yue2_lora_finetuning/ComfyUI \
  --checkpoint yue2_3b_bf16.safetensors \
  --data /content/data/dataset \
  --semantic-head tokenizer_head_joint_v4.pt \
  --use-semantic --conditioning inference_like \
  --segment-seconds 30 \
  --rank 32 --alpha 32 --lr 1e-4 \
  --steps 2000 --save-every 250 --eval-every 100 \
  --eval-holdout 5 --eval-samples 8 --seed 2002 \
  --sample-every 250 --sample-seconds 30 \
  --out maqam_acoustic_v2 2>&1 | tee -a /content/logs/train.log
```

README guidance: rank 32, strength 1.0–1.5, **1000–3000 steps**; a LoRA that
muffles/noises the render is over-trained. `--sample-every` renders a fixed
30 s stream with the LoRA under training so you can hear each checkpoint.

## 7. Evaluation — audio, not loss

- **First fix the generation-side `cfg_scale` gap** (context.md §15): the
  ComfyUI `YuE2GenerateMusic` node never forwards `cfg_scale`, so every ComfyUI
  render so far ran on an unverified default, not the `1.2` the known-good
  baseline used. This applies **with the LoRA off**, so it must be fixed before
  any A/B is trustworthy. Either patch the node to accept/forward `cfg_scale`
  (the underlying `generate_music()` already supports it) or render through a
  small script that calls `generate_music()`/`generate_abc()` directly with
  `cfg_scale=1.2` and `max_duration=400` (the user's 6-minute ceiling).
- Judge by ear against the base model at the same seed — specifically
  pronunciation (ق/ح/خ), "genuineness," and whether the maqam actually appears
  instead of the base model's Western-minor default. **Do not** judge by the
  loss curve; it plateaus while the LoRA keeps changing.
- Stack the two LoRAs for the final check: planner on the CLIP path, acoustic
  on the MODEL path.

## 8. Open decisions / risks

- **Tokenizer accuracy is approximate** (~16% exact top-1). The planner learns
  from near-miss tokens; that is the current ceiling, not a bug to fix.
- **Two sources disagree on the NAR companion LoRA.** Nora ships one; speedyrulz
  measured it makes renders *less* similar and does not use it. If Stage 3
  underdelivers, A/B her `nar_lora_joint_v4` rather than assuming.
- **No community validation on Arabic/maqam.** Nobody has proven this on
  melismatic Arabic. We are early adopters; the smoke test in §4 is the guard.
- `status`-field check (context.md §5) and `--max-per-song` capping — still
  open, low priority.
- Corpus drift: re-run `verify_dataset.py --dataset-root ...` before trusting
  the counts above.

## 9. What NOT to do

- **Do not repeat the 1500-step `--conditioning compact` acoustic run.** It is
  a known near-no-op and structural dead end for the goal.
- **Do not trust the old `PLAN.md` or the ai-toolkit PR-#1042 claim** — it was
  fabricated and is resolved (context.md §2).
- **Do not use `Starnodes2024/ComfyUI-YuE2-Trainer`** — its own issue reports a
  trained LoRA with no effect at all.
- **Do not start, stop, or resume training from a session** — the user runs
  every command; sessions stage commands and read logs/checkpoints.
