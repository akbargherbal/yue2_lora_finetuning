# Session 3 — backend decided (no training)

Read-only research session. Outcome: the training backend and planner target
were settled.

## What happened

- **`ostris/ai-toolkit` ruled out against the primary source.** An earlier
  session had claimed a merged YuE2 LoRA PR (#1042); session 2 softened it to
  "in doubt" after an outdated third-party page. Session 3 checked the repo
  directly: its README does have an `### Audio` section (so "no audio models at
  all" was stale), but only ACE-Step 1.5/XL are listed, and searching "YuE"
  across the repo returns zero results. PR #1042 is not a YuE2 PR.
- **`speedyrulz/ComfyUI-YuE2-Trainer` chosen by elimination.** Re-confirmed the
  architecture: two LoRAs (acoustic MODEL path, planner CLIP path), planner
  trained as chunked next-token cross-entropy over the AR path.
- **`Starnodes2024/ComfyUI-YuE2-Trainer` rejected.** Its own issue #1 reports a
  trained LoRA with no effect, corroborating the speedyrulz benchmark's
  ~zero CLAP delta.
- **Composition is the planner's job.** The acoustic LoRA only changes
  timbre/production (waveform correlation 0.88–0.91 with the base render). A
  maqam is composition, so the planner is the real target.
- **`maqam_prompt_generator.py` confirmed committed** (commit `3623aef`).

## Lesson carried forward

The PR-#1042 claim was almost certainly fabricated, then survived two rewrites
by getting *softer* instead of being *checked*. When a doc marks something as
uncertain, resolve it against a primary source rather than restating the doubt.
This is the direct motivation for splitting durable reference from session
narrative (see `docs/architecture.md`, `docs/known-issues.md`).
