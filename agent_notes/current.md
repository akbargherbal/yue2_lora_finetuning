# Fair criticism: 1500 steps was a smoke test, not a fine-tune

## The honest read
- 1500 steps × one 30 s random crop per step = **12.5 h of audio seen**. The corpus
  is ~18 h, so it didn't even go through the data **once** (<1 epoch). For a style
  LoRA that is tiny.
- Held-out loss moved 1.0510 → 1.0078 (**-4%**) and plateaued after ~250 steps; the
  render changed almost nothing (`MFCC 0.998`, `waveform 0.946`). That is exactly
  what a too-short run looks like, not a finished LoRA.
- The 7 minutes is real, but it's a property of the trainer, not of the run being
  "done": ~0.3 s/step (3B model, bf16, rank 16, 14.68 M trainable params, batch 1,
  compact prefix). **Step count, not wall-clock, was the limit**, and I picked 1500
  from the trainer README's ~1000-step *demo*, which is a demo, not a production
  fine-tune. My framing ("sane default") was wrong; it's a pipeline validation.

## A second, likely correctness issue: training/inference mismatch
The default `--conditioning compact` trains the LoRA against a **style-only prefix**,
but generation feeds **style + lyrics + ABC**. So the LoRA is applied at inference
in a context it was never trained on — a good way to get a near-no-op or artifacts.
`--conditioning inference_like` (needs the lyrics, which we have) matches generation.
This alone may explain why 1500 steps of weight movement barely changed the output.

## What a real acoustic fine-tune looks like
- `--conditioning inference_like` (match inference; README says ~3x slower per step
  than compact, so ~0.9 s/step → **10k steps ≈ 2.5 h, 30k ≈ 7.5 h**) — fits your
  session window.
- `--rank 32 --alpha 32` (≈29 M params) or 64 if you want more capacity.
- More steps: **10k–30k** (≈ 5–15 epochs over the 18 h corpus), `--save-every 1000`,
  and A/B *rendered partials* along the way — audio is the metric, not the loss.
- Keep the leak-free eval (`--eval-holdout 5 --seed 2002`).

## Set expectations though
Even trained properly, this is the **acoustic** LoRA: it changes timbre/production,
not words or maqam. Your ق/pronunciation complaint and the exaggerated melisma are
in the **planner's semantic tokens** (proved last turn: the base and LoRA renders
shared identical tokens). No amount of acoustic training fixes them. Those need the
**planner** LoRA — still blocked on ABC scores (`context.md` §2/§8) — or fixing the
planner inputs (tashkeel/style/temperature).

## Proposed next step
Retrain acoustic properly: `--conditioning inference_like --rank 32 --steps 20000
--save-every 1000`, in tmux, with the GCS backup running, then A/B the partials.
Say go and I'll give you the exact command.
