# Changelog

One line per session, newest last, linking to the narrative in
`agent_notes/sessions/` and to `docs/known-issues.md` for open items. Durable
design reference is `docs/architecture.md`; current state is `context.md`.

## Session 3 — backend decided
`ostris/ai-toolkit` ruled out against the primary source; `speedyrulz/ComfyUI-YuE2-Trainer`
chosen; `Starnodes2024` fork rejected. Planner established as the maqam lever.
→ [`agent_notes/sessions/session-03.md`](agent_notes/sessions/session-03.md)

## Session 4 — setup, dry run, first real run, setback
Colab bootstrap fixed and passing; acoustic dry run; 1500-step acoustic smoke
test (near-no-op); first generation test exposed the ق regression and the
pre-existing Western-key bias. Checkpointing/GCS/leak-free-eval decided.
→ [`agent_notes/sessions/session-04.md`](agent_notes/sessions/session-04.md)

## Session 5 — pipeline diff
Diffed against the known-good sibling raw-inference pipeline; found the
generation-side `cfg_scale` wiring gap. No training.
→ [`agent_notes/sessions/session-05.md`](agent_notes/sessions/session-05.md)

## Session 6 — the missing tokenizer
Identified `Mothersuperior/...-realaudio-tokenizer-v4` and confirmed it is
already integrated (`--semantic-head`); rewrote `PLAN.md`; reversed the ordering
to planner-first.
→ [`agent_notes/sessions/session-06.md`](agent_notes/sessions/session-06.md)

## Session 7 — first planner LoRA + first evaluation
Tokenized 256/256; trained a 100-step planner LoRA (held-out eval 5.60 → 4.95);
rendered base vs checkpoints 30/60/100. User verdict pending. Committed the
helper scripts and the `cfg_scale` patch.
→ [`agent_notes/sessions/session-07.md`](agent_notes/sessions/session-07.md)

## Docs / process refactor (post-session-7)
Implemented `quality_review.md`: split the 1,100-line `context.md` into this
changelog, per-session files, `docs/architecture.md`, and
`docs/known-issues.md`; committed the test suite; pinned dependencies and both
external repos; hardened `audition_planner.py`; de-duplicated run commands.
