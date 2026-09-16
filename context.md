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

> **Fresh session, or arrived after a VM/GPU switch?** Local disk (`/content`,
> `/workspace`) is wiped; only GitHub + GCS persist. Read this file, then
> `PLAN.md` §2.4 for restore commands (re-clone, re-apply
> `bootstrap/joint_minted_optional.patch`, rebuild corpus/venv/models, restore
> prep + checkpoints from GCS). GCS root:
> `gs://akbar-december-2024-backup/YuE2-3B_Finetuning/`; the calibration run
> `maqamverse_calib_v1/` is done/archived, the next run is `maqam_planner_v2/`.
> `agent_notes/current.md` is per-turn scratch — this file and `PLAN.md` are the
> durable state.

## Current state (one paragraph)

Dataset is done and verified (256 tracks, 238 train / 18 val, poem-safe split).
The planner LoRA (`maqam_planner_v1`, 100 steps, session 7) was audited by ear
(session 8): pronunciation held, but step 30 beat 60/100 (KL climbing, KI-03)
and even at 30 the لحن read "foreign". Session 9 tested the resulting
hypothesis — that the community semantic head (`tokenizer_head_joint_v4.pt`,
KI-04) needed adapting to this corpus — by running Mothersuperior's `joint.py`
calibration. It completed 3000 steps (held-out `real_nar` 0.9629 → 0.8789, no
collapse), but a 2×2 by ear (stock/calibrated head × stock/calibrated NAR)
found **no clean win, and the calibrated variants were worse** (muddy/unclear
vocals, ق→ك degradation); the calibrated head only bought structural
robustness (no outro collapse) at a fidelity cost. Confounds: the minted
regularizer was unavailable (`MINTED=0`, KI-25) and `joint.py` targets a single
artist, not a four-maqam corpus. **Conclusion: the stock tokenizer already
produces idiomatic Arabic round-trips, so the tokenizer is not the planner's
bottleneck.** Decision: stop the calibration line and refocus on the planner
itself. Note: the session-7 tokenized cache did not survive the VM and was
never backed up (KI-30), so planner work starts by re-tokenizing the corpus.

## Blocked on

Nothing. The lever is chosen — raise the planner's KL weight (`PLAN.md` →
"Current plan" P2) — because more steps currently make it *worse* (KI-03).

## Next steps (in order)

1. **Re-tokenize the corpus** (`PLAN.md` → "Current plan" P1) with the stock
   head — the previous semantic-token cache is gone (KI-30).
2. **Run the planner KL experiment** (P2): `--kl-weight 1.0`, 200 steps, dense
   saves; ear-test checkpoints including early ones; watch that `kl` levels at
   a few hundredths instead of climbing.
3. **Branch (`PLAN.md` §7):** if the planner still reads foreign after a
   well-regularized run, the tokenizer is truly exonerated → take the symbolic
   lead (SheetSage2 melody→ABC, `architecture.md` §5).
4. **Backup/durability:** daemon runs under `--run-name`; `TARGETS` covers
   `loras/`, `logs/`, `agent_notes/`, `head_calib`, `prep`, and the tokenize
   `cache/` (KI-30).
5. **Open/low priority, unchanged:** `status`-field check (KI-01),
   `--max-per-song` (KI-02), corpus-drift re-verify (KI-07).

## Session hygiene

- Anything the user would copy-paste goes in `agent_notes/current.md`
  (gitignored, overwritten each turn), not in this file or chat.
- Sessions never start, stop, or resume training — the user runs every command.
- Check `nvidia-smi` before anything GPU-heavy; one shared GPU.
