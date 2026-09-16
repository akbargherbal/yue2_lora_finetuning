# PLAN — training runs (planner LoRA + tokenizer-calibration record)

> **Rewritten from scratch, session 8.** The previous `PLAN.md` (sessions 6–7)
> covered planner-LoRA training on the stock `tokenizer_head_joint_v4.pt` head
> and is superseded, not amended — that work isn't wasted, see §0. Read
> `context.md` first; this file is the canonical home for training commands
> until superseded again.
>
> **License:** YuE2's weights and the community semantic-tokenizer/NAR assets
> are all **CC BY-NC 4.0 (non-commercial)**. Anything trained here inherits
> that constraint.
>
> **Concluded, session 9 — this plan is now a record.** The calibration ran
> (3000 steps, held-out `real_nar` 0.9629 → 0.8789) and was **rejected by ear**:
> the calibrated variants were worse than stock, so the stock tokenizer stands
> and is not the planner's bottleneck (KI-29). Keep this file for reproduction;
> the live direction is planner-side (the KL climb, KI-03) — see `context.md`.

## Current plan — planner re-run with a stronger KL

The tokenizer was exonerated (KI-29); the planner's KL climbed (KI-03) and more
steps made the ear verdict worse. So: re-tokenize with the **stock** head, then
retrain the planner with a **higher `--kl-weight`** and dense checkpoints, and
pick a checkpoint by ear. Run with `HF_HOME=/workspace/hf` (MERT lives there).

### P1. Re-tokenize the corpus (Stage 1; the cache is gone, KI-30)

```bash
cd /content/yue2_lora_finetuning
HF_HOME=/workspace/hf python ComfyUI/custom_nodes/ComfyUI-YuE2-Trainer/train_cli.py planner \
  --comfy-root /content/yue2_lora_finetuning/ComfyUI \
  --checkpoint yue2_3b_bf16.safetensors \
  --data /content/data/dataset \
  --semantic-head tokenizer_head_joint_v4.pt \
  --semantic --no-abc \
  --eval-holdout 5 --seed 2002 \
  --steps 100 --out maqam_planner_v2 --dry-run 2>&1 | tee -a /content/logs/train.log
```

(`--dry-run` populates the token cache + VAE latents and writes an untrained
0-step LoRA — ignore that artifact, KI-18.)

### P2. Planner LoRA, `--kl-weight 1.0`

Same as session 7 except the KL weight (0.5 → 1.0), more steps to test whether a
stronger trust region keeps improving instead of drifting, and dense saves:

```bash
cd /content/yue2_lora_finetuning
HF_HOME=/workspace/hf python ComfyUI/custom_nodes/ComfyUI-YuE2-Trainer/train_cli.py planner \
  --comfy-root /content/yue2_lora_finetuning/ComfyUI \
  --checkpoint yue2_3b_bf16.safetensors \
  --data /content/data/dataset \
  --semantic-head tokenizer_head_joint_v4.pt \
  --semantic --no-abc --abc-dropout 0.5 --kl-weight 1.0 \
  --max-tokens 4096 \
  --rank 32 --alpha 32 --lr 5e-5 --lr-schedule cosine \
  --steps 200 --save-every 10 --eval-every 10 \
  --eval-holdout 5 --eval-samples 8 --seed 2002 \
  --probe-every 10 --probe-max-tokens 8192 \
  --out maqam_planner_v2 2>&1 | tee -a /content/logs/train.log
```

Watch: `kl` should level at a few hundredths, not climb; pick by ear (step 30
was best last time). If KL still climbs at 1.0, raise to 2.0 or lower `--lr`.
Start the backup daemon under this run's name first:
`python backup_to_gcp.py --run-name maqam_planner_v2` (§2.2).

**Ran session 9.** Held-out 5.6047 → **4.9400 (−11.9%)**, plateaued by ~step
120; KL still reached ~0.25 (weight 1.0 only *slowed* the drift). Artifacts
under `maqam_planner_v2/` in GCS; five audition renders on the leak-free
holdout `nahawand_0095_01_take01` (base + steps 30/60/100/200) in
`maqam_planner_v2/renders_unseen/`. **Awaiting the ear verdict** — guide in
`agent_notes/sessions/session-09.md`.

## 0. Where we are, and why this session exists

- **Listening verdict on the track-4 A/B is in (resolves KI-21):** base vs
  planner checkpoints 30/60/100 on `04-وصف-محاسن-الحبيبة-والجمال` (Maqam
  Nahawand). Pronunciation held up at every checkpoint — the acceptance floor
  is intact. But: **step 30 sounded better than 60 and 100**, and even at 30,
  the لحن (melodic line) reads as "foreign" against the training corpus, and
  the musicality is simpler than the reference tracks. More steps made it
  worse, not better.
- **Diagnosis:** KL divergence was climbing, not leveling off (0 → 0.25 by
  step 100, KI-03) — later steps are following tokenizer noise, not
  converging on style. And the noise itself has a specific source: the
  community semantic-token head (`tokenizer_head_joint_v4.pt`, KI-04, ~16%
  exact top-1 on YuE2's own songs) has **never been adapted to this corpus**.
  It was trained on Mothersuperior's own corpus and used here frozen,
  as-shipped. Fine, imprecise melodic detail — ornamentation, cadence shape,
  the things that make لحن sound right rather than "vaguely Middle
  Eastern" — is exactly what a noisy, out-of-distribution teacher would wash
  out first. This reframes KI-04 from "accepted ceiling" to "untried fix
  available": the model card's own workflow has an *adapt the head to your
  audio* step (`joint.py`) that this project has never run.
- **Goal of this session:** run that calibration step, and get a real,
  listenable answer to *does a corpus-calibrated head sound less "foreign"
  on the same round-trip test* — before spending any more GPU time on planner
  LoRA training against the old, uncalibrated tokens.
- **Not in scope today:** re-tokenizing the full corpus, retraining the
  planner LoRA, or Stage 3 (acoustic LoRA). Those are real next-session work,
  gated on today's verification step (§6) actually looking promising.

## 1. Two separate toolchains — do not conflate them

This repo's existing pipeline (`speedyrulz/ComfyUI-YuE2-Trainer`,
`train_cli.py planner --semantic-head ...`) is **not** the toolchain that
calibrates the head. That calibration lives in Mothersuperior's own scripts
(`scripts/prep_real.py`, `cursor_prep.py`, `joint.py`, from the
`yue2-mothersuperior-realaudio-tokenizer-v4` HF repo), which is a separate,
standalone codebase with its own environment expectations.

**Today's job uses only Mothersuperior's scripts.** We are not switching the
planner trainer, not adopting their `ar_prep.py`/`ar_lora_cursor.py` planner
LoRA pipeline, and not touching `train_cli.py`. The only thing we want out of
today is a calibrated `tokenizer_head_*.pt` (+ its paired NAR delta) that can
later be dropped into the existing `--semantic-head` flag — *if* it verifies
by ear. That compatibility is unconfirmed (§6), so treat it as an open
question, not an assumption.

## 2. Pre-flight

### 2.1 GPU: start on L4, per your call

The previous (flawed) planner run used 8–10 GB VRAM on L4 the whole time, well
under the model card's 14–18 GB estimate for this calibration job. **Start on
L4. Do not pre-emptively switch to A100.** Escalate only on real evidence:

- `nvidia-smi` shows VRAM genuinely maxed out and the job OOMs, or
- the per-step timing extrapolated from the first ~20–30 steps of `joint.py`
  puts the full 3000-step run at a wall-clock length you're not willing to
  sit through in one Colab session.

Check both within the first few minutes of §5 starting — don't wait until
the end to find out.

### 2.2 Backup: confirm it's running, then extend it

Per `AGENTS.md`'s backup responsibility: before training starts, confirm
`backup_to_gcp.py` is actually running:

```bash
pgrep -af backup_to_gcp.py || echo "NOT RUNNING"
tail -5 /content/logs/gcp_backup.log
```

If it isn't running:

```bash
cd /content/yue2_lora_finetuning
nohup python backup_to_gcp.py --run-name maqamverse_calib_v1 \
  > /content/logs/gcp_backup_stdout.log 2>&1 & disown
```

`--run-name` is required; this run's artifacts land in the generic project root
at `gs://akbar-december-2024-backup/YuE2-3B_Finetuning/maqamverse_calib_v1/`,
while the session 6-7 planner backup stays separate at
`.../YuE2-3B_Finetuning/maqam_planner_v1/`. Iterate on `--run-name` per run
(the root is fixed, so multiple sessions in a day just get their own folders);
`dataset/`, `track4_ab/`, `fine_tuning_ai_music_lora/` are reserved names.

**Targets.** `backup_to_gcp.py`'s `TARGETS` = `ComfyUI/models/loras/`, `/content/logs/`, `agent_notes/`, `/workspace/tok/full` (`head_calib`), `/workspace/real/prep` (`prep`), and the trainer `cache/` (KI-30). See the script for the live list and `docs/architecture.md` → "VM / GPU switch".

### 2.3 Environment: separate venv + Mothersuperior's layout (done)

Mothersuperior's scripts want their own stack: Python 3.12 venv,
`yue2-infer` @ commit `92a73cc7`, torch 2.10 + cu128, torchaudio 2.10,
transformers, soundfile, scipy, safetensors, demucs. Keep it out of the ComfyUI
Python env. `python3.12 -m venv` fails here (no `ensurepip`, and
`python3.12-venv` isn't installable), so use **uv** (installed):

```bash
cd /content
uv venv --python 3.12 ms_calib_venv
uv pip install --python /content/ms_calib_venv/bin/python \
  torch==2.10.* torchaudio==2.10.* --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /content/ms_calib_venv/bin/python transformers soundfile scipy safetensors demucs
uv pip install --python /content/ms_calib_venv/bin/python \
  "git+https://github.com/multimodal-art-projection/YuE.git@92a73cc7"
```

Models `m-a-p/YuE2-3B`, `m-a-p/YuE2-Vae`, `m-a-p/MERT-v2-FullSong` are cached
under `/workspace/hf` (set `HF_HOME`).

The scripts hard-code `/workspace/...`, so recreate that layout with symlinks
onto `/content`:

```bash
mkdir -p /workspace/tok/full/listen_real /workspace/real/prep /content/ms_calib/hf
ln -sfn /content/ms_calib/corpus /workspace/real/artist
ln -sfn /content/ms_calib/hf /workspace/hf
```

Fetch scripts + weights:

```bash
mkdir -p /content/ms_calib && cd /content/ms_calib
hf download Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4 --local-dir . --include "scripts/*"
hf download Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4 \
  tokenizer_head_joint_v4.pt nar_lora_joint_v4.pt --local-dir .
```

**Blocker that forced a patch:** `joint.py` can **not** run as shipped. It
loads `{W}/sem_nbr_idx.npy` + `{W}/sem_nbr_cos.npy` unconditionally and reads
`{W}/feats/*.npy` + `/workspace/yue2-corpus/tracks/<pid>/*` for its *minted
regularizer* side. Those are not released: `sem_nbr_*` has no generator
anywhere, `feats/` needs the full 4,732-song minted corpus plus a MERT GPU
pass, and the released `minted_regularizer_pack.pt` is consumed by
`ar_prep.py`, not `joint.py`. We patched `/content/ms_calib/scripts/joint.py`
to make the minted side optional — `MINTED=0` (or missing `feats`/`sem_nbr_*`)
runs head+NAR on the real audio alone and prints `minted regularizer OFF`.
`LR_HEAD`/`LR_LORA`/`LR_IO` are now env-overridable.

**Script facts (read, not guessed):** `joint.py <name> <steps> <train_head
0|1> <train_lora 0|1> <init_head> <init_lora|none> [rank]` — the model card's
`3000 1 1` means 3000 steps, train head, train NAR LoRA. `HOLD_TRACK` is
compared to the **prep directory name** (= flac basename, no extension), not a
title (§4). `joint.py` reads only `mert.npy`/`lat.npy`/`prefix.npy` from prep,
so `cursor_prep.py` is **not** part of this run (§5.2).

Re-downloading the scripts gives the **unpatched** `joint.py`; re-apply our
edit (see `docs/architecture.md` → VM/GPU switch):

```bash
cd /content/ms_calib && git apply /content/yue2_lora_finetuning/bootstrap/joint_minted_optional.patch
```

### 2.4 Restore after a VM / GPU switch (or VM loss)

Switching GPU or losing the VM wipes `/content` and `/workspace`; only GitHub +
GCS survive. A fresh session should read `context.md` first, then:

```bash
# 1. re-clone the repo and redo §2.3 (uv venv, models, /workspace symlinks,
#    scripts + weights), then re-apply the patch:
cd /content/ms_calib && git apply /content/yue2_lora_finetuning/bootstrap/joint_minted_optional.patch

# 2. rebuild the corpus from the backed-up dataset, then run §3:
gsutil -m cp -r gs://akbar-december-2024-backup/YuE2-3B_Finetuning/dataset /content/data/dataset

# 3. restore the GPU prep (backed up as TARGETS "prep") and skip prep_real.py:
mkdir -p /workspace/real/prep
gsutil -m cp -r "gs://akbar-december-2024-backup/YuE2-3B_Finetuning/maqamverse_calib_v1/prep/*" /workspace/real/prep/ 2>/dev/null \
  || echo "prep not backed up -> rerun prep_real.py (§5.1)"

# 4. restore joint checkpoints/render if resuming:
mkdir -p /workspace/tok/full
gsutil -m cp -r "gs://akbar-december-2024-backup/YuE2-3B_Finetuning/maqamverse_calib_v1/head_calib/maqamverse_calib_v1" /workspace/tok/full/
gsutil -m cp -r "gs://akbar-december-2024-backup/YuE2-3B_Finetuning/maqamverse_calib_v1/head_calib/listen_real" /workspace/tok/full/

# 5. relaunch §5.3 with the same --run-name.
```

Keep the same `--run-name` so a resumed run stays in one GCS prefix (the daemon
appends; the `run_manifest.json` must match). Note `joint.py` still has no
resume flag (KI-26): a restored `head_last.pt` must be passed as `INIT_HEAD`
for a continuation run, which restarts the LR schedule.

## 3. Convert the dataset to Mothersuperior's expected format

Their per-song contract: `<name>.flac` + `<name>.lyrics.txt` (full lyrics,
section tags intact) + `<name>.txt` (style caption starting with a trigger
phrase). Our dataset ships `.mp3` + `.lyrics.txt` + `.style.txt` — close, but
not identical:

- **Audio:** mp3 → flac, lossless re-encode is fine here (mp3 is already
  lossy; flac just satisfies their loader, it doesn't recover quality).
- **Lyrics:** all 256 tracks use TitleCase/numbered tags (`[Intro]`,
  `[Chorus]`, `[Verse 1]`, `[Bridge]`, `[Refrain]`). `cursor_prep.py` skips any
  line matching `^\s*\[.*\]\s*$` wholesale, so tag case/numbering is a
  non-issue for `joint.py` (which doesn't read cursors anyway). Only the AR
  cursor stage would care — and its `words_of` keeps `[a-z']` only, so it
  cannot align Arabic at all.
- **Style caption:** needs a **trigger phrase** prefix (their example:
  `"xyzq, in the style of xyzq. <description>"`). This corpus doesn't have an
  existing single-artist trigger since it spans four maqams under one target
  vocal/production style. **Assumption (flag if wrong):** use one project-wide
  trigger, e.g. `"maqamverse, in the style of maqamverse."`, prepended to the
  existing `.style.txt` content verbatim. This is a real open question, not
  a settled convention — confirm it doesn't collide with anything before
  running the full corpus through it.

Write a small conversion script rather than doing this by hand — 238 train
+ 18 val tracks:

```bash
python scripts/export_mothersuperior_format.py \
  --dataset-root /content/data/dataset \
  --out-root /content/ms_calib/corpus \
  --trigger-phrase "maqamverse, in the style of maqamverse."
```

(`scripts/export_mothersuperior_format.py`, committed — run for real: 256
tracks, 0 failures, ~13 GB at `/content/ms_calib/corpus`. It logs section-tag
counts and does not normalize them.)

## 4. HOLD_TRACK

`joint.py` matches `HOLD_TRACK` against the prep directory name, which
`prep_real.py` sets to the flac basename (no extension). The audition track's
Arabic title (`04-تأمل-محاسن-الحبيبة-والجمال-المترف`, drifted from the old
note's `04-وصف-...`) maps via the dataset manifest to a slugged filename:

```
HOLD_TRACK=nahawand_0111_04_take01
```

(`nahawand_0111_04_take01.flac` is in `/content/ms_calib/corpus/` — Nahawand,
train split, single take. The other محاسن row is a different poem,
`nahawand_0095_01_take01`.) If unset, `joint.py` silently holds out the first
track, so set it explicitly for continuity with the base/30/60/100 audition.

## 5. Run the calibration

### 5.1 Feature + latent prep

`prep_real.py` takes no arguments — it reads `/workspace/real/artist/*.flac`
(symlinked to the corpus) and writes `/workspace/real/prep/<name>/{mert,lat,prefix}.npy`.

```bash
export HF_HOME=/workspace/hf
/content/ms_calib_venv/bin/python /content/ms_calib/scripts/prep_real.py \
  2>&1 | tee -a /content/logs/ms_prep_real.log
```

GPU: MERT-v2 + VAE over 256 tracks — check `nvidia-smi` is idle first.

### 5.2 Vocal stem + forced alignment — SKIP

Not needed: `joint.py` reads only `mert.npy`/`lat.npy`/`prefix.npy`. Cursors
feed `ar_lora_cursor.py` only, and `cursor_prep.py` can't align Arabic anyway.
Skip unless/until the AR cursor stage is run.

### 5.3 The calibration run itself

```bash
export HF_HOME=/workspace/hf
cd /content/ms_calib/scripts
MINTED=0 HOLD_TRACK=nahawand_0111_04_take01 \
  /content/ms_calib_venv/bin/python joint.py \
  maqamverse_calib_v1 3000 1 1 \
  /content/ms_calib/tokenizer_head_joint_v4.pt \
  /content/ms_calib/nar_lora_joint_v4.pt \
  2>&1 | tee -a /content/logs/ms_joint.log
```

Outputs to `/workspace/tok/full/maqamverse_calib_v1/{head,lora}_{best,last}.pt`
+ `train.log`, and the round-trip render to
`/workspace/tok/full/listen_real/real_pred_maqamverse_calib_v1.flac`. That
`/workspace/tok/full` path is in `backup_to_gcp.py`'s `TARGETS` as
`head_calib` (§2.2).

Start at default LRs; if `real_repeat` climbs toward 1.0 (§6), stop and rerun
with a lower `LR_HEAD` (and/or fewer steps) — with the minted regularizer off
there is no anti-collapse term.

## 6. What to watch, and the stop/verify signals

- **First 20–30 steps:** confirm VRAM (§2.1) and extrapolate total runtime.
  Report both back before letting it run unattended for hours.
- **`real_repeat` is the collapse monitor now that `MINTED=0`.** The eval line
  prints `real_nar` (held-out flow loss, lower is better — this selects
  `best`) and `real_repeat` (fraction of adjacent identical predicted tokens).
  A `real_repeat` climbing toward 1.0 means the head collapsed: stop and rerun
  with a lower `LR_HEAD`. `minted_top1` prints `nan` by design (no
  regularizer).
- **HOLD_TRACK round-trip render, done twice** — once with the stock head,
  once with the calibrated checkpoint, same seed, same prompt — is the actual
  verification, not the training loss curve alone. Loss tells you the head
  moved; only your ear tells you it moved toward "sounds like the corpus."
- **Compatibility check** (this is the open question from §1): before
  deciding today was worth it, confirm the calibrated head/NAR files can even
  be loaded by the existing `--semantic-head` flag in `train_cli.py` (check
  `nodes_yue2.py`'s loader against the calibrated checkpoint's keys/shape).
  If they can't, that's a same-day finding worth surfacing immediately, not
  a next-session surprise.

## 7. Decision tree for the next session

- **Calibrated round-trip clearly sounds closer to the training corpus's
  لحن, with pronunciation still intact** → re-tokenize the full corpus with
  the new head (`train_cli.py`'s `--semantic-head`, `--semantic-force`) and
  retrain the planner LoRA fresh (not resumed — the underlying tokens
  changed, so `_000030`/etc. checkpoints aren't a valid resume base anymore).
  The old `maqam_planner_v1` checkpoints stay as a labeled comparison point,
  not a discard.
- **No audible difference from the stock head** → the tokenizer wasn't
  actually the bottleneck; revisit Path 1 (SheetSage2 melody-to-ABC as a
  second, symbolic training signal — `t8star/YuE2-Comfy` per
  `architecture.md` §5, currently an unverified lead) instead of re-running
  calibration with different hyperparameters blind.
- **Calibration itself fails to load/run** (environment or path issues) →
  that's a session in itself; update `docs/known-issues.md` with whatever
  broke rather than reconstructing it from memory next time.

## Session hygiene (unchanged from `AGENTS.md`)

- Anything copy-pasteable goes in `agent_notes/current.md`, overwritten each
  time.
- This session never starts, stops, or resumes training or the calibration
  run — the user runs every command.
- Check `nvidia-smi` before anything GPU-heavy.
- Confirm `backup_to_gcp.py` is running and covers today's new output path
  before, not after, the long run starts (§2.2).
