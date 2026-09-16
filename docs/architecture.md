# Architecture — YuE2-3B Maqam LoRA

Durable reference: what this project is, how the pieces fit, and why the
tooling decisions are what they are. Per-session history lives in
[`../CHANGELOG.md`](../CHANGELOG.md) and `../agent_notes/sessions/`; open
problems live in [`known-issues.md`](known-issues.md). Current state and next
steps live in [`../context.md`](../context.md).

## 1. Goal

Fine-tune a LoRA for **YuE2-3B** (m-a-p's open lyrics-to-song model) on the
user's back-catalog of Suno-generated tracks, so the LoRA reproduces their
"winning" style: symphonic cinematic orchestral ballad / heavy rock,
grand-concert-hall acoustics, deep male Arabic vocals, classical Arabic poetry
as lyrics, in one of four maqams (Hijaz, Nahawand, Ajam, Kurd).

The acceptance bar is **base-model pronunciation parity**: output must be no
worse than the un-fine-tuned base model (correct classical-Fusha ق/ح/خ) and
should actually move composition toward the requested maqam. Judge by ear, not
by the loss curve.

## 2. The two LoRAs — which half does what

| Half | LoRA | Trains | Controls |
|---|---|---|---|
| AR / planner | `planner` (CLIP) | next-token over `style + lyrics → ABC` and/or `→ semantic tokens` | **composition**: melody, harmony, structure, maqam — and the semantic tokens pronunciation comes from |
| NAR / acoustic | `acoustic` (MODEL) | flow-matching on VAE latents | render only: timbre, instrument sound, mix, production |

**The project goal (maqam + pronunciation + style) lives in the planner.** The
acoustic LoRA is "the same song, re-recorded" (measured waveform correlation
0.88–0.95 with the base render). So the ordering is **planner first, acoustic
second** — the reverse of the original plan.

## 3. Corpus and built dataset

### Creative material

Classical/pre-Islamic and Andalusian Arabic poetry (the mu'allaqat and similar
— Imru' al-Qais, Antara, Amr ibn Kulthum, Tarafa, al-Harith ibn Hilliza,
al-Nabigha al-Dhubyani, al-Nabigha al-Ja'di, al-A'sha, Majnun Layla, Malik ibn
al-Rayb, al-Shanfara (Lamiyat), Abu Tammam, Ibn Zuraiq, Ibn Zaydun,
al-Mutanabbi, Jarir, and others) set to one fixed musical template and rendered
via Suno v5.

Each manifest track's `lyrics` field carries the full poem text with tashkeel
(diacritics), wrapped in Suno structural/production tags (e.g.
`[Verse 1 | epic soaring vocals | heavy power chords]`, occasional inline
`[guitars surge — Ajam]`) plus a leading `///***///` UI marker that is not lyric
content.

### Raw corpus layout

```
min_4stars_ai_music/
├── ajam/
│   ├── <workspace_name>/
│   │   ├── workspace_manifest.json
│   │   └── <surviving track>.mp3, <surviving track>_SONG_A.mp3, ...
│   └── ...
├── hijaz/    (same shape)
├── kurd/     (same shape)
└── nahawand/ (same shape)
```

Each `workspace_manifest.json` lists **every** track ever generated in that
session, including below-4-star ones that were deleted. `prepare_dataset.py`
treats **file existence** as ground truth for "kept". Messiness handled:
recurring poems across workspaces (grouped by normalized title + maqam),
inconsistent filenames, NFC/NFD Arabic filename mismatches, and maqam taken
from the top-level folder (cross-checked against the caption, warned not
resolved).

### Real build numbers (verified)

```
Filtered out (below the rating bar): 1651
Resolved 256 kept tracks across 136 distinct poems.
  Hijaz      tracks= 59  poems= 31
  Nahawand   tracks= 73  poems= 39
  Ajam       tracks= 62  poems= 33
  Kurd       tracks= 62  poems= 33
Split: 238 train / 18 val tracks (123 / 13 poems).
```

18 val tracks (not a flat 10%) is expected: the split picks whole poems, so the
poem-level ratio (~9.6%) is what matters for leakage safety.

### Audio format

Uniform 48 kHz stereo MP3, 18.0 h total, duration range 161.9s–369.7s.
**No format standardization, resampling, or manual clip segmentation is
needed** — the trainer's random-crop training handles arbitrary track length
natively, and 370s is under its whole-song (~470s) limit.

## 4. Prompt template — `maqam_prompt_generator.py`

The user's own tool that generates the Suno prompt block. Everything is fixed
except the maqam name, an optional per-track `mood`, and (standard mode) a lyric
start phrase. Only four maqams are offered (Hijaz, Nahawand, Ajam, Kurd)
because the others need quarter-tone intervals with no faithful Western mapping.

- `standard` mode: lyrics + a `vocals` field + `[START_ON: "..."]` header.
- `instrumental` mode: no lyrics, no `vocals` field, no start header; and
  deliberately **not** "no vocals" in EXCLUDE — non-lyrical vocalization is
  left possible on purpose.

The docstring records that an earlier LLM (Gemini) `mood` variation experiment
introduced unwanted drift. That is why `prepare_dataset.py` re-derives every
style caption from `build_prompt()` rather than trusting the stored `styles`
text, and keeps the caption vocabulary closed.

## 5. Tooling decisions

### YuE2-3B

Open, ~3B lyrics-to-song model: lyrics + a style prompt → full song with vocals
and accompaniment. Upstream: `github.com/multimodal-art-projection/YuE`.
Weights: `Comfy-Org/YuE2` on HuggingFace,
`checkpoints/yue2_3b_bf16.safetensors` (~7,438 MB).

### Training backend — `speedyrulz/ComfyUI-YuE2-Trainer` (decided, by elimination)

Trains two separate LoRAs from the same checkpoint inside ComfyUI: an
**acoustic** LoRA (NAR flow-matching path, VAE latents) and a **planner** LoRA
(AR language model, chunked next-token cross-entropy). Dataset sidecar
convention: `<stem>.style.txt` (style/caption) plus `<stem>.lyrics.txt` **or a
bare `<stem>.txt`** (a bare `.txt` is read as *lyrics*, not style — this caused
a real dataset bug, fixed).

Relevant capabilities:

- `--eval-holdout N` holds out whole items (`Item.id` = filename stem), chosen
  by `random.Random(seed + 4242).sample(sorted(ids), N)` — **not** by poem.
  `scan_folder` walks `--data` recursively, so pointing `--data` at the parent
  `dataset/` merges train+val into one 256-item set with no file changes.
- Acoustic `--segment-seconds` takes a random crop at its real position, so no
  pre-segmentation is needed; `segment_seconds=0` = whole song up to ~470s.
- Planner targets either an ABC score (`--no-abc` off; needs SheetSage2 or
  `.abc`) or the semantic tokens themselves (`--semantic`, no scores needed).
- `--semantic-head <file>` predicts tokens for every song and writes
  `<song>.semantic.npy`; `--semantic-force` recomputes existing sidecars;
  `--mert` defaults to `m-a-p/MERT-v2-FullSong`.
- `--use-semantic` conditions the acoustic stage on tokens;
  `--conditioning inference_like` lays out the prefix the way generation does;
  `--sample-every`/`--sample-tokens` render a stream with the LoRA under
  training.
- Planner `--abc-dropout`, `--kl-weight` (trust region toward base),
  `--probe-every` (writes whole scores/token streams per checkpoint — the
  over-training signal).
- Node equivalents: *YuE2 Semantic Tokens (community head)* and
  `yue2_train_semantic_planner_api.json`.

### ai-toolkit — RULED OUT

An earlier session claimed `ostris/ai-toolkit` merged YuE2 LoRA support in PR
#1042. Checked against the primary source: ai-toolkit's live README **does**
have an `### Audio` section (so the "no audio models at all" claim was just
outdated), but the only audio models listed are ACE-Step 1.5 / XL. YuE2 is not
supported and a search for "YuE" across the repo returns zero results. **PR
#1042 is not a YuE2 PR; stop considering ai-toolkit.** The fabricated claim
survived two rewrites by being softened to "in doubt" rather than checked —
resolve uncertainty against primary sources.

### `Starnodes2024/ComfyUI-YuE2-Trainer` — do not use

Separate, confusingly-named fork. The speedyrulz benchmark found it worse
(CLAP similarity −0.004 vs base) and needing more VRAM (24 GB vs 16 GB); its
own issue #1 reports a trained LoRA with no effect at all. Corroborates the
"don't use it" call.

### `Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4` — the missing encoder

YuE2 shipped no audio → semantic-token encoder, so real recordings had no
teachable target on the planner's semantic path. Nora (*Make The Robot Do It*)
trained one: MERT-v2-FullSong layer-20 features → 8-layer transformer → 32,768
YuE2 codes, plus a rank-32 NAR LoRA companion. Already integrated:
`--semantic-head <file>` / the *community head* node. Tokenizer accuracy is
approximate (~16% exact top-1 on YuE2's own songs, ~95% by ear on round-trips);
treat labels as noisy but usable. **License: CC BY-NC 4.0 (non-commercial),
same as YuE2.**

### `t8star/YuE2-Comfy` — unverified lead

A ComfyUI package bundling SheetSage2 + MERT with lyric/style/optional-ABC
assistance. Nothing verified beyond the page; treat as a lead, not a solution.

### Sibling repo

`akbargherbal/fine_tuning_ai_music_lora` — raw YuE2-3B inference (no
fine-tuning), confirmed by the user as the ق/ح/خ-clean baseline. It calls the
`yue2` pip package's `YuE2Pipeline` directly with `--cfg 1.2`, `--cot full`,
`ode_steps=32`, and sends style/lyrics verbatim (no caption re-rendering). Its
README independently documented the base model's Western-minor bias before any
LoRA work existed.

## 6. Data pipeline design

### `prepare_dataset.py`

Walks `<dataset_root>/<maqam>/<workspace>/workspace_manifest.json`, then:

1. Filters by file existence (Unicode-normalized filename matching).
2. Takes maqam from the folder name; cross-checks the caption.
3. Keeps all surviving takes of a poem by default (`--max-per-song` optional).
4. Groups by normalized `original_title` + maqam so splitting never leaks.
5. Re-renders every style caption via `build_prompt()`; strips the Suno control
   header by default and drops `mood` by default.
6. Renders a cleaned lyrics sidecar via `clean_lyrics()` (always strips the
   `///***///` marker; `--lyrics-tag-mode` default `simplify` collapses
   `[Verse 1 | ...]` → `[Verse 1]` and drops non-structural inline cues).
   `simplify` was chosen over `full` because the style caption already fixes
   instrumentation/production, and `simplify` matches the community trainer's
   own section-tagger output format.
7. Splits train/val stratified by maqam, whole poems.
8. Writes three files per example:
   `<maqam>_<poem_id:04d>_<slug>_take<NN>.<ext>` + `.style.txt` +
   `.lyrics.txt` (never a bare `.txt`).
9. `manifest.csv` columns: `clip_id, workspace, original_title, maqam, mood,
   take_no_in_group, source_audio, dest_audio, dest_style, dest_lyrics, split`.
10. `--clean` wipes `train/`+`val/` first — recommended for every non-dry-run
    rebuild, because `poem_id` is assigned by first-seen order and deleting a
    poem's last take can shift later ids and orphan old files.

`ascii_safe_slug()` strips all Arabic characters, so unrelated titles sharing a
leading number (e.g. `01-الوداع` and `01-عزة`) collapsed onto the same slug;
the sequential `poem_id` was added to guarantee unique destination filenames.

### `verify_dataset.py`

Independent post-build QC. Does **not** import the builder or prompt generator,
so a bug can't hide by being reproduced identically. Stdlib only (except
`--check-audio`, which needs `ffprobe`). Checks in order:

1. Structure — `train/`/`val/` exist.
2. Pairing & naming — audio has both non-empty sidecars (empty lyrics is WARN,
   legitimately fine for instrumentals); flags pre-`poem_id` filenames.
3. Captions & lyrics — caption maqam matches filename; lyrics free of
   `///***///` and un-simplified pipe tags (latter WARN).
4. Manifest cross-check — no duplicate destination paths or `clip_id`s;
   manifest rows and disk files agree both directions (tolerates a moved
   dataset dir by basename fallback, WARN).
5. Train/val leakage — no poem spans both splits; prints per-maqam table.
6. Drift vs the live corpus (`--dataset-root`) — re-walks and diffs `clip_id`s.
7. Audio stats (`--check-audio`) — duration/rate/channel/codec consistency.

Exit code 0 if clean, 1 if any ERROR. The testing discipline this project
expects: hand-built fixtures reproducing the manifest's field shape, plus
deliberately corrupted builds to confirm each failure mode is caught. Those
tests are now committed under `tests/` (see §8).

### Leak-free eval on this corpus

Because the trainer holds out by item, not poem, a random holdout usually
leaves a sibling take in training. Pointing `--data` at the parent `dataset/`
and choosing a seed whose held-out items are all **single-take poems** makes
the held-out poems vanish from training entirely. Verified seeds:

| `--eval-holdout` | leak-free `--seed` |
|---|---|
| 1 | `4` |
| 3 | `166` |
| **5** | **`2002`** |

Default `--seed 0` is not leak-free; no leak-free seed exists for holdout ≥ 8.
The 18-track `val/` split stays on disk for external evaluation.

## 7. Run operations

### Checkpointing

- `--save-every N` is in optimizer steps (default 0 = no partials). Every N it
  writes into `ComfyUI/models/loras/`: `<out>_NNNNNN.safetensors` **and**
  `<out>_NNNNNN.resume` (optimizer moments, RNG states, step, loss/eval
  history).
- Final `<out>.safetensors`, `.loss.json`, `.resume` are written only on normal
  exit. `--keep best_eval` makes the final file the best-eval checkpoint.
- Revisit once real seconds/step is known; target ~10–15 min between partials.
- Final LoRA ~28 MB; `.resume` grows once Adam moments exist (~117 MB at 1500
  steps). Disk is not a constraint.

### Resumability

- Resume = same command plus `--existing-lora <last>.safetensors`, same `--out`;
  `--steps` is the run's total length. `rank`, `alpha`, `targets`, `optimizer`
  must match.
- **Ctrl-C does not run the final save** — the last `--save-every` partial is
  the resume point. `.resume` writes are atomic; `.safetensors` writes are not,
  so a crash can tear one partial (resume from the previous; verify with the
  trainer's `tests/verify_lora_file.py`).
- The VAE-latent cache is keyed by path+size+mtime+flags, so resume/re-run skips
  the ~20 min encode.

### Observability

Sessions run inside this same VM: they read `/content/logs/train.log`,
`ComfyUI/models/loras/*`, and `nvidia-smi`. On-demand only, no background
watching. From those: step/ETA, loss/avg/grad-norm, eval curve, which
checkpoints exist, process/GPU state, crash cause. Sessions never start, stop,
or resume training.

### GCS backup — `backup_to_gcp.py`

Run in a second terminal next to training. Every 25 min it mirrors to a
**per-run** folder under one generic root,
`gs://akbar-december-2024-backup/YuE2-3B_Finetuning/<run-name>/`, with
`gsutil -m rsync -r`:

| local | remote | why |
|---|---|---|
| `ComfyUI/models/loras/` | `<run>/loras/` | `.safetensors`, `.resume`, `.loss.json` |
| `/content/logs/` | `<run>/logs/` | `train.log` for crash diagnosis |
| `agent_notes/` | `<run>/agent_notes/` | plans + exact commands |

`--run-name` is required, and a `run_manifest.json` at the run folder's root
records the run name, timestamp, and target list so a folder identifies itself.
The root is fixed (no dated parents accumulating per session) and also holds
the non-run folders `dataset/`, `track4_ab/`, `fine_tuning_ai_music_lora/`;
those names are reserved. The session 6-7 planner backup now lives at
`YuE2-3B_Finetuning/maqam_planner_v1/`, and a new run can never overwrite a
prior run's artifacts. Append/update only (no `-d`, never deletes remotely); a
`--settle-seconds 60` guard avoids grabbing a file mid-write. Renders/prompts/
scripts are uploaded by hand (KI-20). Needs `gcloud`/`gsutil` auth to stay
valid.

### VM / GPU switch — restore procedure

Colab does not move a runtime between GPUs. Switching (L4→A100), or losing the
VM, wipes **local disk** — `/content` and `/workspace`, including the repo
working tree and the whole Mothersuperior toolchain. Only GCS survives, so a
switch is *restore, then resume*, guided by `PLAN.md` §2.4. What must survive
lives in exactly two places:

- **GitHub** (`akbargherbal/yue2_lora_finetuning`) — the docs, scripts, tests,
  and `bootstrap/joint_minted_optional.patch`. A fresh VM re-clones it. If
  today's changes weren't committed, they are gone.
- **GCS** `gs://akbar-december-2024-backup/YuE2-3B_Finetuning/` —
  `dataset/` (built corpus, restorable) and per-run
  `<run-name>/{loras,logs,agent_notes,head_calib,prep,cache}/`. The
  `run_manifest.json` at each run prefix names it. `agent_notes/current.md` is
  scratch; `context.md` and this file are the durable state.

Everything else is rebuilt, in minutes: venv (`uv`), HF models (re-download),
corpus (`export_mothersuperior_format.py` from the GCS dataset), and the
patched `joint.py` (re-download + `git apply` the patch). The one expensive
local artifact is `/workspace/real/prep` (~25 GPU-min); it is mirrored as the
run's `prep` target, so a switch restores it instead of re-running
`prep_real.py`.

### Evaluation — audio, not loss

- Generation quality depends on `cfg_scale`; the ComfyUI generation nodes
  originally never forwarded it (the underlying `generate_music()` accepts it).
  Fixed by `bootstrap/yue2_cfg_scale.patch`, applied by `setup.sh`; renders use
  `cfg_scale=1.2` (matching the known-good baseline) and `max_duration=400`
  (the user's ~6-min ceiling).
- Judge by ear against the base model at the same seed — pronunciation
  (ق/ح/خ), genuineness, and whether the maqam appears. **Do not** judge by the
  loss curve; it plateaus while the LoRA keeps changing.
- Stack both LoRAs for the final check: planner on the CLIP path, acoustic on
  the MODEL path.

## 8. Repo map

| path | what |
|---|---|
| `docs/architecture.md` | this file — durable reference |
| `docs/known-issues.md` | flat, deduplicated open problems with status |
| `context.md` | HANDOFF — current state, blockers, next steps |
| `CHANGELOG.md` | per-session index |
| `agent_notes/sessions/` | one narrative file per session |
| `PLAN.md` | **canonical training run plan + commands** |
| `MANUAL.md` | **canonical generation runbook + commands** |
| `prepare_dataset.py` | corpus → train/val dataset builder |
| `verify_dataset.py` | independent dataset QC |
| `maqam_prompt_generator.py` | Suno prompt template (user's tool) |
| `audition_planner.py` | render style+lyrics through the ComfyUI graph, base or planner LoRA |
| `render_tokens_nar.py` | render a `.semantic.npy`, optionally fold in the NAR companion |
| `scripts/export_mothersuperior_format.py` | dataset → Mothersuperior's flat `.flac`/`.lyrics.txt`/`.txt` corpus (head calibration, `PLAN.md` §3) |
| `backup_to_gcp.py` | periodic GCS mirror of run artifacts |
| `bootstrap/setup.sh` | Colab bootstrap (pinned clones, downloads, patch) |
| `bootstrap/yue2_cfg_scale.patch` | wires `cfg_scale` into the YuE2 nodes |
| `bootstrap/joint_minted_optional.patch` | makes Mothersuperior's `joint.py` minted regularizer optional (`MINTED=0`), for the head calibration |
| `tests/` | pytest suite + fixtures |
| `docs/archive/` | historical process artifacts (e.g. the 2026-09 quality review) |

## 9. User context

- Python developer; prefers Markdown docs; Windows/PowerShell at home, Colab VM
  for runs.
- Has done weeks of by-ear curation (~1900 tracks → 256 keepers). The quality
  bar is high and already applied — don't suggest re-filtering audio.
- Values scripts tested against realistic data, including deliberately broken
  fixtures that confirm a checker catches the breakage.
- Comfortable with git/shell; the sandbox has push access via a `gh` token.
  Never echo the token.
