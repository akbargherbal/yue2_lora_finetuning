# Session 6 — the missing tokenizer, found and already wired in

No training, no generation. The session identified the community artifact the
project had been missing, verified from the trainer's source that it is already
integrated, and rewrote `PLAN.md` around it.

## The finding

The video "How I trained the missing YuE2 Tokenizer" (*Make The Robot Do It* /
Nora) is about `Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4` on
HuggingFace — the audio → semantic-token encoder YuE2 never shipped, plus a
rank-32 NAR LoRA companion. Without it, real recordings have no semantic
tokens, which is why the planner's semantic path was unusable and the project
was stuck on ABC/SheetSage2.

Full technical facts are in `docs/architecture.md` §5.

## Already integrated (verified from source)

- **`--semantic-head <file>`** predicts tokens for every song and writes
  `<song>.semantic.npy`; `--semantic-force` recomputes; `--mert` defaults to
  `m-a-p/MERT-v2-FullSong`.
- Acoustic: `--use-semantic`, `--conditioning inference_like`,
  `--sample-every`/`--sample-tokens` render an audible checkpoint A/B.
- Planner: `--semantic`, `--no-abc`, `--abc-dropout`, `--kl-weight`,
  `--probe-every`.
- Node equivalents: *YuE2 Semantic Tokens (community head)* and
  `yue2_train_semantic_planner_api.json`.

**Consequence:** the planner can train on semantic tokens with no ABC scores
(`--semantic --no-abc --abc-dropout 0.5`), removing the SheetSage2 /
melismatic-Arabic blocker. This **reverses the ordering** to planner-first,
acoustic-second.

## What changed

- `PLAN.md` rewritten from scratch (bootstrap → tokenize → planner → acoustic →
  evaluation), with exact commands.
- `context.md` §2/§7/§8/§10 updated; the old acoustic-first framing retired.

## Caveats

- Tokenizer is approximate (KI-04).
- The two sources disagree on the NAR LoRA (KI-05).
- Nothing is validated on Arabic/maqam (KI-06).
- The prior `PLAN.md` was grounded in the fabricated ai-toolkit PR claim and was
  replaced, not amended.
