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
> `gs://akbar-december-2024-backup/YuE2-3B_Finetuning/`; this run is
> `maqamverse_calib_v1/`. `agent_notes/current.md` is per-turn scratch — this
> file and `PLAN.md` are the durable state.

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
never backed up (KI-20), so planner work starts by re-tokenizing the corpus.

## Blocked on

Nothing. The next step is a design choice, not a dependency: which planner-side
lever to try. Highest suspicion is the KL climb (KI-03) — more steps currently
make it *worse*, so the LoRA over-adapts rather than learning style.

## Next steps (in order)

1. **Re-tokenize the corpus** (Stage 1, `train_cli.py` planner) with the stock
   head — the previous semantic-token cache is gone (KI-20). Planner commands
   are in `PLAN.md`; the session-7 form is in
   `agent_notes/sessions/session-07.md`.
2. **Run a planner experiment aimed at the KL climb (KI-03):** raise
   `--kl-weight` (and/or lower LR, save densely) so more steps learn style
   without drifting; ear-test several checkpoints, including early ones.
3. **Branch (`PLAN.md` §7):** if the planner still reads foreign after a
   well-regularized run, the tokenizer is truly exonerated → take the symbolic
   lead (SheetSage2 melody→ABC, `architecture.md` §5).
4. **Backup/durability:** daemon runs under `--run-name`; `TARGETS` covers
   `loras/`, `logs/`, `agent_notes/`, `head_calib`, `prep`. Make sure the
   **tokenized cache is backed up this time** (KI-20) before relying on resume.
5. **Open/low priority, unchanged:** `status`-field check (KI-01),
   `--max-per-song` (KI-02), corpus-drift re-verify (KI-07).

## Session hygiene

- Anything the user would copy-paste goes in `agent_notes/current.md`
  (gitignored, overwritten each turn), not in this file or chat.
- Sessions never start, stop, or resume training — the user runs every command.
- Check `nvidia-smi` before anything GPU-heavy; one shared GPU.
