# Session 5 — pipeline diff against the known-good baseline

No training, no generation. A read-only diff of this repo against
`akbargherbal/fine_tuning_ai_music_lora` (the sibling raw-inference repo the
user confirmed did **not** have the ق/ح/خ problem), to stop future sessions
re-litigating whether the LoRA caused the regression.

## User-confirmed facts (settled)

- **Tashkeel is not the cause** — full harakat works fine on this model.
- **Pronunciation was correct before any fine-tuning existed** — the user
  recalls no ق/ح/خ mistakes when they first tried base YuE2.
- **6 minutes is the real song-length ceiling** → `max_duration ≈ 400s`
  (comfortably above the corpus max of 369.7s). This fully explains the 3:34
  hard cut in session 4's test; no further diagnosis needed.

## The baseline pipeline

`yue2_generate.py` does not use ComfyUI at all: it calls the `yue2` pip
package's `YuE2Pipeline` directly with `GenerationConfig(ode_steps=32)`,
`--cot full`, `--cfg 1.2`, and sends style/lyrics verbatim (full Suno control
syntax, full tashkeel, no caption re-rendering / no `clean_lyrics()`).

Its own README independently documents that the model **consistently defaults
to Western minor keys regardless of the requested maqam**, across every
generation mode tested — so session 4's K:Fm result is a pre-existing
base-model limitation, not a fine-tuning regression.

## The finding: `cfg_scale` had no path into `YuE2GenerateMusic`

Traced through the trainer source: `generate_music()` accepts and forwards
`cfg_scale`, but grepping `nodes.py` for `cfg_scale` returned zero matches. The
`YuE2GenerateMusic` node exposed style/lyrics/seed/mode/`max_duration`/sampling
params but **no `cfg_scale` input**. So every ComfyUI generation so far ran with
`cfg_scale=None`, on an unverified default, not the `1.2` the baseline used.

Plausible mechanistic explanation for both symptoms at once: weaker
text-guidance → the model tracks the literal lyrics less closely (mispronounced
ق) and free-runs more (excessive melisma). It is a generation-node wiring gap,
independent of the LoRA and of training.

`generate_abc()` (the trainer's helper) genuinely has no `cfg_scale` hook at any
layer — but the *node* path (`YuE2GenerateABC.execute → clip.tokenize`) does
accept it, so it is patchable too. (Corrected in session 7; see KI-22.)

## Recommended next test (cheap, no training)

1. Patch the generation nodes to forward `cfg_scale`, or call
   `generate_music()`/`generate_abc()` directly.
2. Set `cfg_scale=1.2`, `max_duration=400`.
3. Re-render `hijaz_0038_01-retake_take01`, base-vs-LoRA, listen for ق/ح/خ and
   genuineness.
4. Do **not** re-run acoustic training to test this — the LoRA is not
   implicated.

Implementation landed in session 7 (`bootstrap/yue2_cfg_scale.patch`).

## Fallback order if `cfg_scale` doesn't fully resolve it

1. Style-caption wording (the `vocals:` line's melisma phrasing vs the
   baseline's plain comma-tag line).
2. Planner sampling params — note ABC (`0.7/0.9/30/1.005`) and Music
   (`1.0/0.95/100/1.2`) use different defaults, so track which node changes.
3. Planner-LoRA/ABC work — the real lever for maqam control regardless.
