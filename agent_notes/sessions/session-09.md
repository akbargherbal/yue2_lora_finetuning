# Session 9 — GCS namespacing, corpus conversion, and the head-calibration wall

The session that set out to execute `PLAN.md`'s head calibration, hit a wall in
Mothersuperior's toolchain, worked around it, and staged everything to launch.

## What ran, in order

1. **GCS backup rework.** `backup_to_gcp.py` gained a required `--run-name` and
   writes to one generic root, `YuE2-3B_Finetuning/<run-name>/`, with a
   `run_manifest.json` and reserved non-run names. All prior GCS content was
   migrated: `YuE2-3B_13092026/` (run_backup + dataset + track4_ab +
   fine_tuning_ai_music_lora) → `YuE2-3B_Finetuning/`, the old planner backup
   becoming `maqam_planner_v1/`. The dated prefix no longer accumulates.
2. **Dataset conversion.** Wrote `scripts/export_mothersuperior_format.py`
   (+ tests) and ran it: 256 tracks → `/content/ms_calib/corpus` as flat
   `<name>.flac` / `.lyrics.txt` / `.txt` (trigger + style), ~13 GB, 0 failures.
   Tags are TitleCase/numbered but `cursor_prep.py` ignores bracket lines, so
   that is a non-issue.
3. **Read Mothersuperior's scripts.** Resolved `HOLD_TRACK` (it's the prep
   *directory name*, i.e. the flac basename) and found `cursor_prep.py` is not
   needed for `joint.py`. More importantly, found the **blocker**: `joint.py`
   hard-requires minted artifacts (`sem_nbr_*.npy`, `feats/`, the full
   4,732-song corpus) that are not released. `bootstrap/joint_minted_optional.patch`
   makes that side optional (`MINTED=0`) and the LRs env-overridable.
4. **Environment.** `ms_calib_venv` via `uv` (py3.12, torch 2.10+cu128,
   `yue2-infer@92a73cc7`, demucs), HF models cached under `/workspace/hf`, the
   hard-coded `/workspace` layout symlinked onto `/content`, head + NAR weights
   downloaded. yue2 imports and CUDA verified.
5. **`prep_real.py` ran to completion:** 256/256, `PREP DONE`, no errors,
   ~25 min, ~10 GB VRAM on the L4. `/workspace/real/prep` (~3.5 GB).
6. **Backup target** added for `/workspace/tok/full` (`head_calib`), the
   `joint.py` output.

## Documentation (the point of the session)

The user's concern: after a GPU switch the VM — and the agent's context — is
wiped, so the *backups* must be self-explanatory. Added: the VM/GPU-switch
restore procedure (`docs/architecture.md`, `PLAN.md` §2.4), an orientation
block at the top of `context.md`, the `joint.py` patch as a committed file, and
`docs/known-issues.md` entries (KI-25 unreleased minted artifacts, KI-26 no
joint resume, KI-27 prep not backed up, KI-28 cursor_prep can't align Arabic;
KI-21 moved to resolved).

## Calibration executed — and it didn't help

- `prep_real.py` then `joint.py` (`maqamverse_calib_v1`, 3000 steps, `MINTED=0`)
  ran clean: held-out `real_nar` **0.9629 → 0.8789** (−8.7%), best step 2600,
  `real_repeat` ~0.02 (no collapse), ~47 min on an L4.
- A stock 0-step render matched the same track/seed, then a **2×2** (stock vs
  calibrated head × stock vs calibrated NAR) isolated the effects: the head
  controls token diversity/repetition, the NAR the acoustics.
- **The user's ear rejected the calibrated variants**: stock head + calibrated
  NAR degraded pronunciation (ق→ك, أسود→أسعد) and still collapsed; calibrated
  head + stock NAR was muddy/unclear. No best-of-both; the stock head remained
  preferred despite an outro collapse.
- **Conclusion:** the stock tokenizer already produces idiomatic Arabic
  round-trips, so it is not why the planner sounded foreign. Confounds:
  `MINTED=0` (KI-25) and `joint.py` being a single-artist tool, not a four-maqam
  corpus. Decision: **stop calibrating; refocus on the planner** (KI-03, the KL
  climb). See KI-29.

## Also
- KI-27 fixed (prep now a backup target; 256 dirs mirrored).
- KI-26 (joint resume) left open but now moot for this line.
- New: KI-30 — the session-7 tokenized cache is gone and wasn't backed up, so
  planner work starts by re-tokenizing.

## Planner re-run (`maqam_planner_v2`) — done

- Re-tokenized the corpus (256/256, 0 failed; cache mirrored) and trained
  `--kl-weight 1.0`, 200 steps. Held-out **5.6047 → 4.9400 (−11.9%)**,
  plateaued by ~step 120. KL still climbed to ~0.25 — the stronger trust region
  **slowed** the drift but did not stop it (KI-03).
- Five audition renders generated on the **leak-free holdout**
  `nahawand_0095_01_take01`, seed 831001, cfg 1.2, max 400:
  base + steps 30/60/100/200.

## Listening guide — awaiting your verdict

**Question:** does the planner LoRA move the *generation* toward the corpus
style — and where does it break? Same prompt + seed across all five, so every
difference is the LoRA alone.

**Files** (identical locally and in
`$GCP_BACKUP_BASE/maqam_planner_v2/renders_unseen/`;
local dir `ComfyUI/output/yue2/`):

| what | file |
|---|---|
| base (no LoRA) | `unseen_nahawand0095_base_00001.flac` |
| step 30 | `unseen_nahawand0095_maqam_planner_v2_000030_00001.flac` |
| step 60 | `unseen_nahawand0095_maqam_planner_v2_000060_00001.flac` |
| step 100 | `unseen_nahawand0095_maqam_planner_v2_000100_00001.flac` |
| step 200 / final | `unseen_nahawand0095_maqam_planner_v2_00101.flac` |

Reference recording for the style: the source is the dataset track
`nahawand_0095_01_take01` (converted copy at
`/content/ms_calib/corpus/nahawand_0095_01_take01.flac`).

**Score each render on three independent axes:**
1. **Melody (لحن)** — idiomatic? does Nahawand still read as Nahawand? any
   ornamentation/melisma, phrase-endings/cadence shape, or is it generic?
2. **Pronunciation** — ق/ح/خ/ض/ث/ط etc. Note *specific* words and roughly when
   (e.g. "ق heard as ك at ~1:20", "ض→د"). These are the most actionable notes.
3. **Structure** — do intro → verses → outro hold, or does it degrade/collapse
   near the outro (the stock head did this in the calibration 2×2)?

**Then place each checkpoint in the 2×2:**
- good melody + good pronunciation → the LoRA is working;
- bad melody + good pronunciation → over-adaptation (fewer steps / higher KL);
- good melody + bad pronunciation → the limiter is acoustic (NAR/decoder), not
  the planner;
- bad + bad → the planner/regularizer isn't the lever → symbolic lead.

**Report back:**
- Ranking of the checkpoints (best → worst), and whether **any beats base** on
  the melody axis.
- Does the LoRA make it sound **less "foreign"** than base, or just different?
- Any specific mispronounced words with timestamps.
- Whether the trend matches the session-7 track4 verdict (step 30 best) or
  differs — this track is genuinely unseen, so a difference is meaningful.

That verdict picks the next branch (`PLAN.md` §7).

