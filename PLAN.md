# PLAN — Tokenizer head calibration (Colab run plan)

> **Rewritten from scratch, session 8.** The previous `PLAN.md` (sessions 6–7)
> covered planner-LoRA training on the stock `tokenizer_head_joint_v4.pt` head
> and is superseded, not amended — that work isn't wasted, see §0. Read
> `context.md` first; this file is the canonical home for training commands
> until superseded again.
>
> **License:** YuE2's weights and the community semantic-tokenizer/NAR assets
> are all **CC BY-NC 4.0 (non-commercial)**. Anything trained here inherits
> that constraint.

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
nohup python backup_to_gcp.py > /content/logs/gcp_backup_stdout.log 2>&1 & disown
```

**Then extend it.** `backup_to_gcp.py`'s `TARGETS` currently covers only
`ComfyUI/models/loras/`, `/content/logs/`, and `agent_notes/` (KI-20) — none
of which is where `joint.py`'s output lands. Before kicking off §5, add a
fourth target for wherever `joint.py`'s output path resolves to (§5.3 pins
the literal command); this is a one-line addition to `TARGETS`, not a new
script. **Do not start the 3000-step run until this is in place** — the whole
point of today's backup-policy update was to not repeat "no GCS protection
on the one output that mattered."

### 2.3 Environment: expect a second, separate environment

Mothersuperior's scripts want their own stack: Python 3.12 venv,
`yue2-infer` @ commit `92a73cc7`, torch 2.10 + cu128, torchaudio 2.10,
transformers, soundfile, scipy, safetensors, demucs. This may not match the
pinned versions `bootstrap/setup.sh` installs for `ComfyUI-YuE2-Trainer`.
**Build this in a separate venv**, not inside the existing ComfyUI Python
environment, so a version mismatch here can't break the working planner
trainer:

```bash
cd /content
python3.12 -m venv ms_calib_venv
source ms_calib_venv/bin/activate
pip install torch==2.10.* torchaudio==2.10.* --index-url https://download.pytorch.org/whl/cu128
pip install transformers soundfile scipy safetensors demucs
pip install "git+https://github.com/multimodal-art-projection/YuE.git@92a73cc7"
```

Confirm `HF_HOME` already has (or can fetch) `m-a-p/YuE2-3B`, `m-a-p/YuE2-Vae`,
`m-a-p/MERT-v2-FullSong` — these are large; reuse whatever's already cached
from the existing bootstrap rather than re-downloading if paths line up.

Fetch Mothersuperior's scripts + weights + regularizer pack:

```bash
mkdir -p /content/ms_calib && cd /content/ms_calib
hf download Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4 \
  --local-dir . --include "scripts/*" "tokenizer_head_joint_v4.pt" "nar_lora_joint_v4.pt"
hf download Mothersuperior/yue2-minted-corpus regularizer/minted_regularizer_pack.pt \
  --local-dir .
```

**Read the scripts before running them.** The model card gives one example
invocation for `joint.py` with five positional args (`joint_mine 3000 1 1
tokenizer_head_joint_v4.pt nar_lora_joint_v4.pt`) and no flag documentation.
Do not guess at what `1 1` means — read the argparse block in
`scripts/joint.py` before running for real, and paste anything ambiguous
into `agent_notes/current.md` rather than assuming.

Also: **paths are hard-coded to Mothersuperior's own pod layout**
(`/workspace/tok/full`, `/workspace/real/...`, `/workspace/yue2-corpus/tracks`,
`/workspace/real/ar/dataset.pt`). Either recreate that layout under `/content`
with symlinks, or edit the path constants at the top of each script — check
which before running `prep_real.py`.

## 3. Convert the dataset to Mothersuperior's expected format

Their per-song contract: `<name>.flac` + `<name>.lyrics.txt` (full lyrics,
section tags intact) + `<name>.txt` (style caption starting with a trigger
phrase). Our dataset ships `.mp3` + `.lyrics.txt` + `.style.txt` — close, but
not identical:

- **Audio:** mp3 → flac, lossless re-encode is fine here (mp3 is already
  lossy; flac just satisfies their loader, it doesn't recover quality).
- **Lyrics:** spot-check tag compatibility. Our sample
  (`hijaz_0049_05_take01.lyrics.txt`) uses `[Intro]`, `[Verse 1]`, `[Chorus]`,
  `[Verse 2]`, `[Outro]` — Mothersuperior's doc examples use lowercase
  `[verse]/[chorus]/[bridge]`. Check `cursor_prep.py`'s tag parser before
  assuming case/numbering doesn't matter; normalize only if the parser
  actually requires it.
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

(New script — doesn't exist yet. Spec: walk `train/` + `val/`, for each
`<name>.mp3` write `<out-root>/<name>.flac` via ffmpeg, `<name>.lyrics.txt`
copied as-is, `<name>.txt` = trigger phrase + `.style.txt` content. Log any
tag mismatches found while checking the lyrics point above; don't silently
normalize.)

## 4. HOLD_TRACK

Reuse the existing audition track for continuity with the base/30/60/100
comparison you already have opinions on:

```
HOLD_TRACK=04-وصف-محاسن-الحبيبة-والجمال
```

Confirm the exact converted filename in `/content/ms_calib/corpus/` matches
this before passing it to `joint.py` — the raw corpus manifest and the final
dataset filenames aren't guaranteed identical (KI-01/KI-07 territory).

## 5. Run the calibration

### 5.1 Feature + latent prep

```bash
cd /content/ms_calib
source /content/ms_calib_venv/bin/activate
python scripts/prep_real.py \
  --corpus /content/ms_calib/corpus \
  2>&1 | tee -a /content/logs/ms_prep_real.log
```

(Flags unconfirmed — check `prep_real.py --help` or its argparse block; the
model card gives the script name only, not its CLI surface.)

### 5.2 Vocal stem + forced alignment

```bash
python scripts/cursor_prep.py \
  --corpus /content/ms_calib/corpus \
  2>&1 | tee -a /content/logs/ms_cursor_prep.log
```

This runs Demucs (vocal separation) then MMS forced alignment — CPU/GPU mixed
load, check `nvidia-smi` doesn't collide with anything else running.

### 5.3 The calibration run itself

```bash
HOLD_TRACK=04-وصف-محاسن-الحبيبة-والجمال \
  python scripts/joint.py \
  maqamverse_calib_v1 3000 1 1 \
  /content/ms_calib/tokenizer_head_joint_v4.pt \
  /content/ms_calib/nar_lora_joint_v4.pt \
  2>&1 | tee -a /content/logs/ms_joint.log
```

Wherever this run's output actually lands (confirm from reading the script,
per §2.3) is what needs to be added to `backup_to_gcp.py`'s `TARGETS` per
§2.2, before this command runs — not after.

## 6. What to watch, and the stop/verify signals

- **First 20–30 steps:** confirm VRAM (§2.1) and extrapolate total runtime.
  Report both back before letting it run unattended for hours.
- **`minted_val` held-out loss** (the 5%-by-hash held-out slice baked into
  the regularizer pack) should stay flat — that's the "a small artist set
  didn't collapse the token grammar" check, same spirit as KI-03's held-out
  eval on the planner side. Rising `minted_val` loss is a stop signal.
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
