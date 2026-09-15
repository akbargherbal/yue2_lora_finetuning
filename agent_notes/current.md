# Session 6 — docs rewritten around the real blocker

## What changed
- `PLAN.md` — **rewritten from scratch.** Old file's §0 was built on the
  fabricated "ai-toolkit PR #1042 supports YuE2" claim; gone. New file is the
  Colab run plan: bootstrap additions → tokenize corpus → **planner/AR LoRA**
  (the maqam lever) → acoustic LoRA → evaluation. Exact CLI commands in each
  stage, verified against the trainer's `train_cli.py` source this session.
- `context.md` — added session-6 header note and **§16**; updated §2 (new
  Mothersuperior tokenizer bullet + planner-target correction), §7, §8 (ABC
  blocker de-prioritized, ordering reversed), §10 (new priorities).

## The one-line version
YuE2's missing audio→semantic-token encoder is
`Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4`. It is **already
integrated** into our backend (`train_cli.py --semantic-head`), which lets the
**planner LoRA train on semantic tokens with no ABC scores** — the composition
half that controls maqam/pronunciation. Train planner first, acoustic second.

## Next (when we move to Colab)
Follow `PLAN.md`. Stage 1 = add the head download to `bootstrap/setup.sh` and
tokenize the corpus; Stage 2 = planner LoRA; then fix the `cfg_scale` gap
(§15) before trusting any A/B.
