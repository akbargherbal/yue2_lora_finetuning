# context.md — HANDOFF

Current state of the YuE2-3B maqam LoRA project. This file is the only one a
fresh session needs to read in full; everything durable or historical lives
elsewhere:

- **Design reference:** [`docs/architecture.md`](docs/architecture.md) — goal,
  the two LoRAs, corpus/dataset, tooling decisions, run operations, repo map.
- **Open problems:** [`docs/known-issues.md`](docs/known-issues.md) — flat,
  deduplicated, with statuses.
- **Training plan + commands:** [`PLAN.md`](PLAN.md) (canonical for training).
- **Generation runbook + commands:** [`MANUAL.md`](MANUAL.md) (canonical for
  generation).
- **History:** [`CHANGELOG.md`](CHANGELOG.md) →
  [`agent_notes/sessions/`](agent_notes/sessions/).

## Current state (one paragraph)

Dataset is done and verified (256 tracks, 238 train / 18 val, poem-safe split).
The planner LoRA (`maqam_planner_v1`, 100 steps) was audited by ear: base vs
checkpoints 30/60/100 on a Maqam Nahawand track (**resolves KI-21**).
Pronunciation held up at every checkpoint. But **30 sounded better than 60/100**
(more steps made it worse, consistent with KL climbing rather than leveling
off, KI-03), and even at 30 the لحن reads as "foreign" and musicality is
simpler than the training corpus. **Diagnosis:** the community semantic-token
head (`tokenizer_head_joint_v4.pt`) has never been adapted to this corpus —
it's used frozen, as shipped by Mothersuperior, and its ~16% top-1 noise
(KI-04) plausibly washes out exactly the fine melodic detail that's missing.
**Decision:** before any more planner training, run Mothersuperior's own
head-calibration step (`joint.py`) on our corpus and verify by ear whether a
calibrated head sounds less "foreign" on the same round-trip test. Full plan:
`PLAN.md` (rewritten, session 8 — supersedes the sessions 6–7 version).

## Blocked on

Nothing — next session has a concrete, unblocked plan (`PLAN.md`). The one
real open question inside it: whether a calibrated head/NAR checkpoint from
Mothersuperior's `joint.py` is even loadable by this repo's existing
`--semantic-head` flag (`train_cli.py`). Unconfirmed; check early in the
calibration session, not after committing to a full re-tokenize.

## Next steps (in order)

1. **Run the calibration session per `PLAN.md`**: set up Mothersuperior's
   separate toolchain (§2–3), run `prep_real.py` → `cursor_prep.py` →
   `joint.py` on the corpus with `HOLD_TRACK` = the same Nahawand track used
   in the 30/60/100 audition, and do a same-seed round-trip render (stock head
   vs calibrated head) before drawing conclusions.
2. **Branch on that listening result** (`PLAN.md` §7):
   - *Calibrated round-trip sounds closer to the corpus, pronunciation
     intact* → re-tokenize the full corpus with the new head and retrain the
     planner LoRA **fresh** (not resumed — tokens changed underneath it).
   - *No audible difference* → tokenizer wasn't the bottleneck; try Path 1
     (SheetSage2 melody→ABC as a second, symbolic signal — `t8star/YuE2-Comfy`,
     currently an unverified lead, `architecture.md` §5) instead of re-running
     calibration blind.
   - *Calibration fails to load/run* → treat as its own session; log what
     broke in `docs/known-issues.md`.
3. **Backup policy, now part of `AGENTS.md`:** before any run with a new
   output path, confirm it's covered by `backup_to_gcp.py`'s `TARGETS`
   (currently only `loras/`, `logs/`, `agent_notes/` — KI-20) and add it if
   not, *before* the run starts.
4. **Open/low priority, unchanged:** `status`-field check (KI-01),
   `--max-per-song` (KI-02), corpus-drift re-verify (KI-07).

## Session hygiene

- Anything the user would copy-paste goes in `agent_notes/current.md`
  (gitignored, overwritten each turn), not in this file or chat.
- Sessions never start, stop, or resume training — the user runs every command.
- Check `nvidia-smi` before anything GPU-heavy; one shared GPU.
