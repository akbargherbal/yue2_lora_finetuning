# Context: YuE2-3B LoRA Fine-Tuning — Arabic Maqam Orchestral Rock

> Read this whole file before doing anything else. It's written so a fresh
> Claude session can pick this project up without the user re-explaining it.
> Repo: https://github.com/akbargherbal/yue2_lora_finetuning.git
> This version was written *from inside a clone of the actual repo* (not
> from memory of a prior session), so §5–7 below reflect real, verified
> file contents as of this session, not a secondhand summary. If a future
> session can also clone/browse the repo, still prefer that over trusting
> this file blindly — it can go stale the moment someone edits outside a
> session with me.
>
> **Session 3 lesson, applies to reading this whole file:** two claims here
> were wrong at the same time — one about the repo's own contents (§4 said
> a committed file was missing) and one about the outside world (§2's
> ai-toolkit PR #1042, apparently fabricated by a session and then carried
> forward as "doubtful" rather than checked). Both were resolvable in under
> five minutes against a primary source. **When this file marks something
> as uncertain, resolve it — don't restate the uncertainty.** And check
> `git log` before believing anything this file says about what's
> committed.
>
> **Session 4 final update:** the backend is installed, a dry run passed, then the
> first real acoustic run (1500 steps) completed and generation was tested. That
> test exposed a **quality problem the user considers a regression** (Arabic
> pronunciation, especially ق, plus exaggerated melisma) and showed the acoustic
> LoRA is close to a no-op. **Read §14 before doing anything — it changes the
> priorities.** §1, §8 and §10 were updated to match.
>
> **Session 5 (investigation only, no training run):** compared the ComfyUI
> generation path against the sibling repo's raw-inference script
> (`akbargherbal/fine_tuning_ai_music_lora`) that the user confirms did **not**
> have the ق/ح/خ pronunciation problem. Found a concrete, previously-undocumented
> gap: **`cfg_scale` (text-guidance scale) has no path into the
> `YuE2GenerateMusic` node** — the underlying `generate_music()` function accepts
> it, but the node never wires it, so every ComfyUI generation so far ran on
> whatever internal default applies when it's left `None`, not the `1.2` the
> known-good raw-inference test used. Also: **tashkeel is ruled out** by the user
> directly (works fine with full diacritics), and the **Western-minor-key bias
> (`K:Fm` instead of a maqam) is confirmed pre-existing in the base model**, not a
> fine-tuning regression — the sibling repo's own README documented the same bias
> "across every generation mode tested," before any LoRA work existed. **Read
> §15 before doing anything — it supersedes §14's "what is still unknown" list
> and gives a cheap, no-training next test.** §10 was updated to match.
>
> **Session 6 (investigation + docs only, no training):** identified the
> community piece the whole project was missing. The "Make The Robot Do It"
> / Nora video is about `Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4`
> — the **audio → semantic-token encoder YuE2 never shipped**. It is already
> wired into our chosen backend: `train_cli.py --semantic-head <file>` (and the
> "YuE2 Semantic Tokens (community head)" node) predicts YuE2's 32,768 semantic
> codes for our own recordings and writes `<song>.semantic.npy`. That unlocks
> the **planner/AR LoRA on semantic tokens** — the composition half that
> actually controls maqam and pronunciation — and **removes the SheetSage2/ABC
> blocker**. The old `PLAN.md` was stale and grounded in the fabricated
> ai-toolkit claim; it has been **rewritten from scratch** as a forward-looking
> Colab run plan. **This changes §2, §7, §8 and §10, reverses the
> acoustic-first ordering, and is detailed in §16.**

## 1. The goal

Fine-tune a LoRA for **YuE2-3B** (m-a-p's open lyrics-to-song model) on the
user's back-catalog of Suno-generated tracks, so the LoRA reproduces their
specific "winning" style: symphonic cinematic orchestral ballad / heavy rock,
grand-concert-hall acoustics, deep male Arabic vocals, classical Arabic
poetry as lyrics, in one of four Arabic maqams (Hijaz, Nahawand, Ajam, Kurd).

**Dataset preparation is complete and verified** (§5–6). The backend is installed
— `speedyrulz/ComfyUI-YuE2-Trainer` — and a **first real acoustic run** (1500
steps, 7.5 min) completed, followed by a full generation test. **Outcome: the user
is not satisfied.** The acoustic LoRA barely changed the render, the base planner
composed in a Western key (not a maqam), and the output has Arabic pronunciation
problems (notably ق) that the user did **not** hear from the base model when they
tested it, nor in the source Suno tracks. See **§14**. Operational detail: §11
setup/dry run, §12 checkpoint/resume/observability, §13 GCS backup.

## 2. Background on the tools involved (verified via web search, Sep 2026)

- **m-a-p/YuE2-3B**: open, ~3B lyrics-to-song model. Takes lyrics + a style
  prompt, produces a full song with vocals and accompaniment. Competitive
  with Suno v5/v6 on the team's own WildSongBench (best-of-8 SongBench avg
  6.9632 vs Suno v5's 6.8721). Upstream repo:
  `github.com/multimodal-art-projection/YuE`.
- **`ostris/ai-toolkit` — RULED OUT (session 3, resolved).** An earlier
  session claimed this merged YuE2 LoRA support in PR #1042 (Sep 14 2026).
  Session 2 downgraded that to "in doubt" after finding a RunComfy doc page
  saying ai-toolkit handles no audio models at all. Session 3 checked the
  primary source and both claims are wrong in different directions:
  - ai-toolkit's live README **does** now have an `### Audio` section, so
    the RunComfy "no audio models" line is simply outdated. Don't reuse it
    as evidence for anything.
  - But the only audio models listed are **ACE-Step 1.5 and ACE-Step 1.5
    XL**. **YuE2 is not in the supported-model list**, and a GitHub
    issue/PR search for "YuE" across `ostris/ai-toolkit` returns **zero**
    results — so PR #1042 is not a YuE2 PR.
  - **Conclusion: ai-toolkit cannot train YuE2. Stop considering it.**
  - **Process note worth keeping:** the PR-#1042 claim was almost certainly
    fabricated by a session and then survived two `context.md` rewrites,
    getting *softer* ("in doubt") rather than *checked*. When this file
    records a claim as doubtful, resolve it against the primary source
    rather than carrying the doubt forward again.
- **`speedyrulz/ComfyUI-YuE2-Trainer` — the backend, by elimination
  (session 3).** Verified twice now: session 2 read its real README,
  session 3 independently re-confirmed the core architecture claim (it
  trains YuE2 LoRAs for both the acoustic MODEL and the planner CLIP inside
  ComfyUI; planner training is causal next-token cross-entropy over the AR
  path with chunked logits). Key facts that will matter for setup:
  - Trains **two separate LoRAs** from the same checkpoint: an **acoustic**
    LoRA (NAR flow-matching path, VAE latents of the audio) and a
    **planner** LoRA (AR language model, `style + lyrics -> ABC score`,
    optionally `-> semantic tokens`).
  - **Composition is decided by the planner, not the acoustic model.** The
    acoustic LoRA only changes timbre/production/mix — "the same song,
    re-recorded" (measured waveform correlation 0.88–0.91 with the base
    render). **A maqam is a melodic mode — composition, not timbre — so
    getting the model to actually write in Hijaz vs. Nahawand etc. requires
    training the *planner* LoRA, not just the acoustic one.** This is the
    single most important finding for this project's actual goal; don't
    let a future session default to acoustic-only and expect maqam control.
  - Dataset sidecar convention: `<stem>.style.txt` (style/caption) +
    `<stem>.lyrics.txt` **or `<stem>.txt`** (lyrics — note a bare `.txt` is
    read as *lyrics*, not style; `prepare_dataset.py` was fixed this
    session specifically because of this, see §6).
  - Acoustic training: `segment_seconds` (default 30) takes a random crop
    per step at its real position in the song — **no manual pre-segmentation
    of long tracks is needed**; whole-song training (`segment_seconds=0`)
    is also supported for tracks up to ~470s. Our longest track is 370s, so
    either mode fits. This closes the "clip length" concern raised earlier
    in this project.
  - Planner training can target an **ABC score** (`--no-abc` off; needs
    SheetSage2 or hand-supplied `.abc`; SheetSage2's fixed 300s window is a
    poor fit for our median 253s / max 370s tracks and possibly for melismatic
    Arabic lines) **or the semantic tokens themselves** (`--semantic`). The
    semantic target needs **no scores** — it needs the `Mothersuperior` head
    (next bullet). **Session 6: the semantic path is the one this project
    intends to use, so the ABC/SheetSage2 blocker is off the critical path
    (see §16).**
  - `Starnodes2024/ComfyUI-YuE2-Trainer` is a separate, differently-authored
    project with a confusingly similar name; the speedyrulz README's own
    benchmark found it performed slightly worse (CLAP similarity -0.004 vs.
    base, vs. speedyrulz's own +0.014/+0.016) and needs more VRAM (24GB vs.
    16GB). **Second, independent datapoint found session 3:** its issue #1
    (opened Sep 14 2026) reports a trained LoRA having *no effect at all* —
    identical output with and without the LoRA loader, tested at strengths
    0.0/1.0/2.0 across two different LoRA files. That corroborates the
    ~zero CLAP delta rather than it being a rival author's biased table.
    **Don't use the Starnodes2024 fork.**
- **`Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4` (HuggingFace) —
  the missing encoder, found session 6** (video "How I trained the missing YuE2
  Tokenizer", channel *Make The Robot Do It* / Nora, Sep 14 2026). YuE2 shipped
  no audio → semantic-token encoder, so real recordings had no training targets.
  Nora trained one — MERT-v2-FullSong layer-20 features → 8-layer transformer →
  32,768 YuE2 codes; ~16% exact top-1 on YuE2's own songs, ~95% by ear on
  round-trips — plus a rank-32 NAR LoRA companion, on 4,765 YuE2 self-generated
  songs then adapted to real audio with the frozen decoder as teacher. **It is
  already integrated into our backend:** the trainer's *YuE2 Semantic Tokens
  (community head)* node and the CLI's `--semantic-head <file>` write
  `<song>.semantic.npy` for our tracks. Her own recipe is: tokenize real songs →
  rank-64 **AR LoRA** with a lyric cursor + a 50/50 "minted" regularizer pack,
  capped at ~1500 steps, checkpoint picked by ear. **Note the disagreement:**
  she ships the NAR companion LoRA; speedyrulz measured it makes renders *less*
  similar and does not use it — A/B it rather than assuming. License: CC BY-NC
  4.0 (non-commercial), same as YuE2.
- **`t8star/YuE2-Comfy` (HuggingFace) — unexplored lead for the ABC
  problem, found session 3.** A ComfyUI package that bundles SheetSage2 and
  MERT alongside an assistant that generates lyrics, style, and *optional
  ABC*. Might shortcut the planner-LoRA score-generation blocker (§8),
  might be useless here. **Nothing about it has been verified** beyond the
  page saying so — no Arabic/melismatic fit test, and no check of whether
  its ABC output matches the format speedyrulz's planner trainer expects.
  Treat as a lead, not a solution.
- The user separately appears to have an earlier/adjacent repo,
  `akbargherbal/fine_tuning_ai_music_lora`, testing raw YuE2-3B *inference*
  (no fine-tuning). Relationship to this repo still unconfirmed — ask if
  relevant.

## 3. The creative material

Classical/pre-Islamic and Andalusian Arabic poetry (the mu'allaqat and
similar — poets referenced across the corpus include Imru' al-Qais, Antara,
Amr ibn Kulthum, Tarafa, al-Harith ibn Hilliza, al-Nabigha al-Dhubyani,
al-Nabigha al-Ja'di, al-A'sha, Majnun Layla, Malik ibn al-Rayb, al-Shanfara
(Lamiyat), Abu Tammam, Ibn Zuraiq, Ibn Zaydun, al-Mutanabbi, Jarir, and
others) set to one fixed musical template and rendered via Suno v5.

Each manifest track's `lyrics` field carries the **full poem text with
tashkeel (diacritics)**, wrapped in Suno-style structural/production tags —
e.g. `[Intro | single clean guitar | close-mic'd, plate reverb, short
decay]`, `[Verse 1 | epic soaring vocals | heavy power chords]`, and
occasional non-structural inline cues like `[guitars surge — Ajam]`. There's
also a leading `///***///` marker line that's pure Suno UI furniture, not
lyric content. See §6 for how this is now cleaned for training.

## 4. The prompt template — `maqam_prompt_generator.py`

This is the user's own tool (they wrote it, I didn't) that generates the
Suno prompt block. Key points:

- Everything is fixed except the maqam name, an optional per-track `mood`
  phrase, and (in standard mode) a lyric start-phrase: `GENRE_STANDARD`,
  `PRODUCTION_STANDARD`, `INSTRUMENTATION`, and `EXCLUDE` are constants,
  identical across every track.
- Two modes: `standard` (lyrics, includes a `vocals` field and a
  `[START_ON: "..."]` header) and `instrumental` (no lyrics, no `vocals`
  field, no start-phrase header). Instrumental mode deliberately does
  **not** add "no vocals" to EXCLUDE — non-lyrical vocalization is left
  possible on purpose.
- Only 4 maqams are offered (Hijaz, Nahawand, Ajam, Kurd) because the others
  (Rast, Bayati, Sikah, Saba, etc.) need quarter-tone intervals with no
  faithful Western mapping.
- The tool's docstring explicitly notes an earlier attempt to have an LLM
  (Gemini) vary the `mood` field introduced unwanted drift, and that field
  was removed from *generation* for that reason. This is why
  `prepare_dataset.py` re-derives every caption from `build_prompt()`
  rather than trusting the free-text `styles` string stored in old
  manifests, and keeps the caption vocabulary closed rather than
  reintroducing generated variation.

**This file IS now committed** (verified session 3 — it landed in commit
`3623aef`, "Session 2", alongside `verify_dataset.py`). Earlier versions of
this document flagged it as missing and as blocking a fresh clone; that is
no longer true. `prepare_dataset.py` hard-imports it
(`from maqam_prompt_generator import MAQAMS, build_prompt`) and raises on
failure, so this mattered — but a fresh clone can now run the script.

## 5. The corpus — directory layout and what it actually means

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

**Critical fact:** each `workspace_manifest.json` lists *every* track ever
generated in that session, including ones the user rated below 4 stars and
deleted. `prepare_dataset.py` treats **file-existence** as ground truth for
"kept."

**Open question, discovered this session, not yet resolved:** a manifest
track entry also has a `status` field (full key list confirmed this session:
`clip_id, original_title, assigned_filename, styles, exclude_styles, lyrics,
created_at, status`). An earlier session's version of this document flatly
asserted "there is no status field to trust." That claim is now in doubt —
**nobody has actually checked what values `status` takes or whether it
correlates with the file-existence filter.** File-existence still works (the
verified build in §6 proves that), so this isn't blocking, but if `status`
turns out reliable it would be a sturdier filter than file-existence, which
breaks the moment a file is deleted for a reason unrelated to rating (e.g.
the user manually removing 1-2 tracks between sessions, which happened this
session). Quick check for next session, if not already done:
```python
import json, glob, collections
c = collections.Counter()
for f in glob.glob('min_4stars_ai_music/*/*/workspace_manifest.json'):
    for t in json.load(open(f, encoding='utf-8'))['tracks']:
        c[t.get('status')] += 1
print(c)
```

Other real-world messiness handled by the script (see §6):
- The same poem can recur across multiple, differently-dated workspace
  folders, sometimes with inconsistent capitalization — treated as one song
  for train/val splitting, grouped by normalized `original_title` + maqam.
- Multiple surviving takes of one poem are **not** duplicates to collapse —
  if `SONG_A`/`SONG_C`/`Retake_..._SONG_A` all still exist, they all passed
  the 4–5 star filter and are wanted audio diversity, kept by default
  (`--max-per-song N` caps this if ever needed; not applied by default).
- Filenames are inconsistent (`<title>_SONG_<letter>.mp3`, `Retake_...`,
  occasionally no suffix) — doesn't matter for identity, only for locating
  the file; `original_title` in the JSON is authoritative.
- Arabic filenames can have NFC/NFD Unicode mismatches between the manifest
  string and the actual file on disk — the script normalizes both sides.
- Maqam comes from the top-level folder name (authoritative), cross-checked
  against "Maqam X" text inside the caption; mismatches are warned, not
  auto-resolved.

**Real numbers, current build (verified this session, non-dry-run, on disk):**

```
Filtered out (below the rating bar): 1651
Resolved 256 kept tracks across 136 distinct poems.
  Hijaz      tracks= 59  poems= 31
  Nahawand   tracks= 73  poems= 39
  Ajam       tracks= 62  poems= 33
  Kurd       tracks= 62  poems= 33
Split: 238 train / 18 val tracks (123 / 13 poems).
```

(Previously 257/136 in an earlier dry-run; the user deleted 1-2 tracks
between sessions, which is reflected here — see §6's drift-detection tool.)

Retry skew (unchanged from earlier assessment, still not revisited): mild,
not alarming — top-retried poems have 4-6 surviving takes, most have 1-2.
**`--max-per-song` capping: still not decided either way, leaning no-cap.**

The 18 val tracks (not a flat 10% of 256) is expected: the split picks whole
*poems*, so the poem-level ratio (13/136 ~ 9.6%) is what matters for
leakage-safety, not raw track count.

## 6. What's been built: `prepare_dataset.py` + `verify_dataset.py`

**Dataset build status: DONE and VERIFIED, 0 errors 0 warnings, this
session.** `dataset/train/` and `dataset/val/` physically exist with copied
audio + captions + lyrics, `manifest.csv` is written, and an independent
verifier confirms every consistency property below actually holds — not
just that the builder ran without crashing.

### `prepare_dataset.py`

Walks `<dataset_root>/<maqam>/<workspace>/workspace_manifest.json`:

1. Filters by file-existence (§5) with Unicode-normalized filename matching.
2. Takes maqam from the folder name; cross-checks against the caption text.
3. Keeps all surviving takes of a poem by default.
4. Groups by normalized `original_title` + maqam across workspaces, so
   splitting never leaks the same poem into both train and val.
5. **Re-renders every style caption** via `maqam_prompt_generator.build_prompt()`
   rather than trusting the stored `styles` text verbatim. Strips the
   Suno-specific control header by default (`--keep-control-header` to
   change), drops `mood` by default (`--include-mood` to change).
6. **Renders a cleaned lyrics sidecar from the manifest's `lyrics` field**
   (added this session — the raw lyrics weren't being used at all before).
   `clean_lyrics()` always strips the `///***///` Suno UI marker, and under
   `--lyrics-tag-mode` (default **`simplify`**, chosen deliberately this
   session over `full`/`strip`):
   - `[Verse 1 | epic soaring vocals | heavy power chords]` becomes
     `[Verse 1]` (collapses to the bare structural tag).
   - `[guitars surge — Ajam]` is dropped entirely (non-structural inline
     production cue, not a section marker).
   - **Rationale for choosing `simplify` over `full`:** the corpus's style
     caption already fixes `INSTRUMENTATION`/`PRODUCTION` as one template
     across all ~250 tracks, so per-line instrument detail in the lyrics is
     mostly redundant noise on top of that fixed signal, not new
     information — and `simplify` also matches the native output format of
     the community trainer's own heuristic section-tagger, which matters if
     lyrics are ever typed more plainly at inference time than they were
     during data collection.
7. Splits into `dataset/train/` / `dataset/val/`, stratified by maqam.
   **Per-example files are now three, not two** (changed this session):
   `<maqam>_<poem_id:04d>_<slug>_take<NN>.<ext>` (audio) +
   `<same-base>.style.txt` (style caption) +
   `<same-base>.lyrics.txt` (cleaned lyrics).
   **Do not write a bare `<base>.txt`** — the training ecosystem being
   targeted (`speedyrulz/ComfyUI-YuE2-Trainer`) reads a bare `.txt` as the
   *lyrics* file, not style; this was a real bug caught and fixed this
   session, before it touched a real build.
8. `manifest.csv` columns: `clip_id, workspace, original_title, maqam, mood,
   take_no_in_group, source_audio, dest_audio, dest_style, dest_lyrics,
   split` (schema changed this session: `dest_caption` split into
   `dest_style` + `dest_lyrics`).
9. `--clean` flag (added this session): wipes `<out-dir>/train` and
   `<out-dir>/val` before writing. **Recommended for every non-dry-run
   rebuild** — `poem_id` numbers are assigned by first-seen order across
   all surviving poems, so if a poem's *last* surviving take is ever
   deleted from the corpus between runs (not just one of several takes),
   every alphabetically-later poem shifts to a new `poem_id` and gets
   renamed, silently orphaning old-named files instead of replacing them.
   Not yet an issue in practice (poem count has stayed at 136 across the
   one deletion that happened), but it's a real fragility to guard against.

### `verify_dataset.py` (new this session)

Independent post-build QC — does **not** import `prepare_dataset.py` or
`maqam_prompt_generator.py`, so a bug in the builder can't hide by being
reproduced identically in the checker. Stdlib only except `--check-audio`
(needs `ffprobe`). Checks, in order:

1. **Structure** — `train/`/`val/` exist.
2. **Pairing & naming** — every audio file has both a non-empty `.style.txt`
   and `.lyrics.txt` (empty lyrics is a WARN not ERROR — legitimately fine
   for an instrumental track); flags filenames still using the pre-`poem_id`
   naming scheme (a fingerprint for a build made before the collision fix).
3. **Captions & lyrics** — every style caption's maqam matches its filename;
   lyrics sidecars are free of leaked `///***///` markers or un-simplified
   pipe-delimited tags (the latter is a WARN, since it's expected if
   `--lyrics-tag-mode full` was used on purpose).
4. **Manifest cross-check** — no duplicate destination paths (the collision
   signature), no duplicate `clip_id`s, manifest rows match files on disk
   exactly in both directions. Tolerates the dataset dir having been
   moved/renamed since the build (falls back to basename matching, WARNs
   rather than ERRORs).
5. **Train/val leakage** — no poem (by normalized title + maqam) spans both
   splits; prints a per-maqam train/val table.
6. **Drift vs. the live corpus** (`--dataset-root`) — independently re-walks
   the corpus and diffs `clip_id`s against `manifest.csv`, catching tracks
   deleted or added since the dataset was last built. **This is the check
   that answered "did my recent deletions make it into the build."**
7. **Audio stats** (`--check-audio`, needs `ffprobe`) — duration
   distribution, sample rate/channel/codec consistency.

**Verified result on the real corpus, this session (after `--clean` rebuild
from scratch): 0 errors, 0 warnings.** Audio is a uniform 48kHz stereo MP3
across all 256 files, 18.02h total, duration range 161.9s-369.7s — meaning
**no format standardization or resampling is needed** for either candidate
trainer, and (per §2) **no manual clip segmentation is needed either**,
since the trainer's own random-crop training handles arbitrary song length
natively. This closes out most of PLAN.md §2 (audio-side cleanup) as
unnecessary, not merely deferred — see §8.

**Testing performed on synthetic fixtures before touching real data**
(same discipline as prior sessions): hand-built fixtures reproducing the
real manifest's exact field shape (`status`, `exclude_styles`, `lyrics`
with diacritics and mixed structural/non-structural bracketed tags,
`///***///` marker), plus deliberately corrupted builds (pre-fix naming
collisions, missing sidecars, leaked markers, forced train/val leakage,
a relocated dataset dir) to confirm the verifier actually catches each
failure mode and doesn't false-positive on the clean case.

## 7. Files that should exist in the repo

- `maqam_prompt_generator.py` — **committed as of `3623aef`, see §4.**
  No longer blocks a fresh clone.
- `PLAN.md` — **rewritten from scratch in session 6.** The old dataset-prep
  plan was fully executed (and its §2 audio cleanup proved unnecessary, §6),
  but its §0 rested on the fabricated ai-toolkit claim (§2) — it is no longer a
  source of truth. The new `PLAN.md` is the forward-looking Colab run plan for
  the **semantic-token / planner-LoRA** path: bootstrap additions (§3),
  tokenize (§4), planner LoRA (§5), acoustic LoRA (§6), evaluation (§7).
- `prepare_dataset.py` — updated in session 2, see §6.
- `verify_dataset.py` — new in session 2, see §6.
- `bootstrap/setup.sh` — Colab bootstrap. **Fixed in session 4**: wrong
  checkpoint path, a ComfyUI clone race, and a stray `cd`; it now runs cleanly
  (see §11).
- `backup_to_gcp.py` — **new session 4.** Mirrors `ComfyUI/models/loras/`,
  `/content/logs/`, and `agent_notes/` to GCS every 25 min with `gsutil rsync`.
  See §13.
- `AGENTS.md` — its `PATH_HERE` placeholders were filled in session 4 with the
  verified log/checkpoint/dataset/script locations.
- `agent_notes/current.md` — the live per-session scratch/output file (rewritten
  each turn, not a stable reference).
- This file (`context.md`).

## 8. Explicitly NOT done yet — don't assume decisions were made

- **Training backend — DECIDED session 3: `speedyrulz/ComfyUI-YuE2-Trainer`,
  by elimination.** ai-toolkit is ruled out (§2, resolved against the
  primary source). **Session 4: it is now installed and verified** — ComfyUI
  cloned, checkpoint downloaded (7,438 MB), trainer cloned with its requirements
  installed, and the acoustic dry run passed on the real dataset (§11).
  Re-opening the choice would mean finding a fourth candidate, not revisiting
  ai-toolkit.
- **Planner vs. acoustic LoRA — reframed session 3: it's an ordering
  question, not a fork.** Earlier versions of this file posed these as
  alternatives. They aren't, because the project's goal splits cleanly
  along the same seam the two LoRAs do:
  - The style template is *fixed* across all 256 tracks (one
    `INSTRUMENTATION`/`PRODUCTION` block, deep male Arabic vocals,
    concert-hall acoustics). That whole half of the "winning style" is
    timbre and production — **exactly what the acoustic LoRA captures.**
  - The maqam is a melodic mode — composition — which the acoustic LoRA
    provably will not touch (§2: 0.88–0.91 waveform correlation with the
    base render, "the same song, re-recorded"). **Maqam control requires
    the planner LoRA.**
  - So both are wanted eventually. The order is forced by what's blocked:
    acoustic was unblocked, planner is blocked on ABC scores (next point).
  - **Session 4 result: acoustic-first was executed and is a dead end for the
    real goal.** A 1500-step acoustic run (§14) produced a near-no-op — LoRA-vs-
    base render `waveform corr 0.946`, `MFCC 0.998` — and it structurally cannot
    change maqam or pronunciation. The base planner also scored the held-out
    poem in **K:Fm (Western)**, not Hijaz. **The planner LoRA is the real target,
    not a later step; the pronunciation problem is upstream in the planner's
    semantic tokens.**
  - **Session 6 resolution:** with the tokenizer (§16) the planner's semantic
    target is unblocked, so the ordering is now **planner-first, acoustic
    second** — see the new `PLAN.md`.
- **Symbolic score / ABC transcription — DE-PRIORITIZED session 6, no longer
  blocking.** It was blocking only while the planner's sole target was ABC;
  with Mothersuperior's head (next bullet up, §2/§16) the planner trains on
  **semantic tokens** (`--semantic --no-abc`) with no scores at all. So
  SheetSage2's 300s window (vs. our median 253s / max 370s) and its unverified
  Arabic fit are off the critical path. Scores remain optional, only for
  score-guided covers.
- **`--max-per-song` capping** — leaning no-cap, not finalized.
- **`status` field investigation** — see §5, quick check not yet run.
- **First real LoRA training HAS now been run (session 4)** — 1500 acoustic steps
  in 7.5 min, held-out loss 1.0510 → 1.0078 (-4%, plateaued by ~250). It is a
  **smoke test, not a fine-tune** (it saw <1 epoch; 1500 × 30 s ≈ 12.5 h vs an
  18 h corpus), and its render changed almost nothing. See §14.

## 9. User context

- Python developer. Prefers documentation in Markdown.
- Has been doing serious manual curation (weeks of by-ear listening across
  ~1900 generated tracks to get to 256 keepers) — treat their quality bar as
  high and already-applied; don't suggest re-filtering audio quality, only
  technical/structural cleanup.
- Values scripts that are actually tested against realistic data before
  being handed over — building synthetic fixtures that reproduce real
  edge cases, and deliberately breaking them to confirm a checker catches
  the breakage, not just that it passes the happy path. Keep doing that.
- Comfortable running git/shell commands themselves (Windows/PowerShell). As of
  session 4 the sandbox **does** have push access: the user installed a GitHub
  token for `gh` (`gh auth login --with-token`, stored in
  `/root/.config/gh/hosts.yml`), and session-4 work was pushed to `origin/main`
  (`89ca4cf`). Never echo the token; the `gh` credential helper covers git.

## 10. Suggested first steps for next session

Session 6 rewrote the plan. **Read `PLAN.md` first** — it is now the
forward-looking Colab run plan, and §16 below records the findings behind it.
Priorities, in order:

1. **Add the tokenizer to the Colab bootstrap.** `bootstrap/setup.sh` must
   download `Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4`
   (`tokenizer_head_joint_v4.pt`) into `ComfyUI/models/audio_encoders/`, and
   make `m-a-p/MERT-v2-FullSong` available. Exact lines in `PLAN.md` §3.
2. **Tokenize the corpus** with `train_cli.py … --semantic-head
   tokenizer_head_joint_v4.pt` so every track gets a `.semantic.npy`
   (`PLAN.md` §4). Smoke-test on a few tracks and listen to a round trip first.
3. **Train the planner/AR LoRA on semantic tokens** — the composition half, the
   only thing that moves maqam/pronunciation (`PLAN.md` §5). Start with
   `--semantic --no-abc --abc-dropout 0.5 --kl-weight 0.5`, small step counts,
   `--save-every 10`, probes on; pick by ear.
4. **Fix the generation-side `cfg_scale` gap before trusting any A/B** (§15).
   `YuE2GenerateMusic` still doesn't forward it, and the gap applies with the
   LoRA off too. Then re-render with `cfg_scale=1.2` and `max_duration=400`.
5. **Acoustic LoRA only after the planner proves out**, and then in the fixed
   mode: `--conditioning inference_like --use-semantic` (`PLAN.md` §6). Do not
   repeat the 1500-step `compact` run (§14).
6. `status`-field check (§5) and `--max-per-song` capping — still open, low
   priority. Corpus drift: re-run `verify_dataset.py --dataset-root ...` before
   trusting the §5 numbers.

## 11. Session 4 — setup done, dry run passed, first run staged

### Setup glitches (fixed)
The first `bootstrap/setup.sh` run failed three ways. All are fixed; the second
run finished with all four parallel jobs `[ok]`.
1. **Wrong checkpoint path.** `Comfy-Org/YuE2` has no root-level
   `yue2_3b_bf16.safetensors` — it lives at
   `checkpoints/yue2_3b_bf16.safetensors`. Fixed the `hf download` path.
2. **ComfyUI clone race (two layers).** The checkpoint job's
   `mkdir -p ComfyUI/models/checkpoints` created `ComfyUI/` before the parallel
   clone job ran, so its `[ ! -d ComfyUI ]` guard skipped the clone; a fixed
   guard would then still have failed to clone into a non-empty dir. Fixed by
   downloading to `/content/staging` and moving the file in **after** the clone,
   and by guarding on `ComfyUI/main.py` (removing a stale partial clone).
3. **Stray `cd /content`** at the end made the post-install verification look at
   wrong (relative) paths. Removed; every `ComfyUI/...` reference is now anchored
   to the script's own repo root, so it runs from any CWD.

Verified on disk: `ComfyUI/main.py`, the trainer's `train_cli.py` (speedyrulz
repo), `ComfyUI/models/checkpoints/yue2_3b_bf16.safetensors` (7,438 MB), and
`/content/data/dataset/` still 238 train / 18 val.

### Dry run — passed on the real dataset
Run as:
```bash
python ComfyUI/custom_nodes/ComfyUI-YuE2-Trainer/train_cli.py acoustic \
  --comfy-root /content/yue2_lora_finetuning/ComfyUI \
  --checkpoint yue2_3b_bf16.safetensors \
  --data /content/data/dataset \
  --eval-holdout 5 --seed 2002 \
  --segment-seconds 30 --steps 300 --out maqam_acoustic_v1 --dry-run
```
It completed in ~22 min, almost entirely the one-time VAE encode (256 tracks ×
~5 s, now cached in `ComfyUI/custom_nodes/ComfyUI-YuE2-Trainer/cache/`). It
loaded the checkpoint, built the LoRA (**112 modules, rank 16, alpha 16, 224
tensors, 14.68 M trainable params**), ran the held-out eval at step 0 (loss
1.0510), and saved a 0-step artifact set. **That saved
`maqam_acoustic_v1.safetensors` is untrained — not a result;** the real run
overwrites it.

### The real run command (staged; the user launches it, never a session)
```bash
python ComfyUI/custom_nodes/ComfyUI-YuE2-Trainer/train_cli.py acoustic \
  --comfy-root /content/yue2_lora_finetuning/ComfyUI \
  --checkpoint yue2_3b_bf16.safetensors \
  --data /content/data/dataset \
  --eval-holdout 5 --seed 2002 \
  --segment-seconds 30 --steps 300 --save-every 50 \
  --out maqam_acoustic_v1 2>&1 | tee -a /content/logs/train.log
```
Run from `/content/yue2_lora_finetuning`. Output lands in
`ComfyUI/models/loras/`. `--save-every 50` is the crash-safety knob — see §12;
it is in **steps**, not minutes.

### Validation / leakage — solved without touching any files
`Item.id` is the audio filename stem, so **each take is its own item**, and the
trainer's `--eval-holdout N` holds out whole items chosen by
`random.Random(seed + 4242).sample(sorted(ids), N)` — **not** by poem. On this
corpus 76 of 136 poems have >1 surviving take, and **76.6% of tracks share their
poem with another track**, so a random 1-item holdout almost always leaves a
sibling take in training (leaky eval). The built `val/` split is poem-safe, but
**the trainer has no flag to evaluate on a separate folder** — it only carves
its holdout from `--data`.

`scan_folder` walks `--data` recursively, so pointing `--data` at the *parent*
`/content/data/dataset` transparently merges `train/` + `val/` into one 256-item
set (all basenames unique) with **no file changes**. Then pick a `--seed` whose
held-out items are all from **single-take poems** (those poems then vanish from
training entirely → leak-free). Verified against `manifest.csv`:

| `--eval-holdout` | example leak-free `--seed` |
|---|---|
| 1 | `4` |
| 3 | `166` |
| **5** | **`2002`** |

Default `--seed 0` is **not** leak-free; no leak-free seed exists for holdout ≥ 8.
Trade-off: the internal eval becomes N songs × `--eval-samples` fixed crops
instead of the 18-track val — but that val could never have been fed to the
trainer anyway. `val/` stays on disk for any separate, external evaluation.

### Still open after session 4
- **`status`-field check (§5)** — still not run: the raw corpus
  (`min_4stars_ai_music/`) is not in this Colab VM, only the built `dataset/`.
  Run it on whichever machine has the corpus.
- **`--max-per-song` capping** — still undecided, still leaning no-cap.

## 12. Checkpointing · resumability · observability (agreed session 4)

### Checkpointing
- `--save-every N` is in **optimizer steps** and defaults to **0 = no
  intermediate saves**. Every N steps it writes, into `ComfyUI/models/loras/`:
  `<out>_NNNNNN.safetensors` **and** `<out>_NNNNNN.resume`.
- The `.resume` file is the resumable state: optimizer moments, every replica's
  torch + Python RNG states, step count, and the **loss/eval history**.
- The final `<out>.safetensors`, `<out>.loss.json` and `<out>.resume` are written
  **only on a normal exit**. `--keep best_eval` (optional) makes the final file
  the best-eval checkpoint instead of the last step; partials are unaffected.
- Chosen setting: `--save-every 50` (matches `--eval-every 50`). Revisit once the
  real seconds/step is known — target roughly 10–15 min between checkpoints.
- Disk is not a constraint (177 GB free). A partial is ~28 MB of LoRA weights
  plus the resume state (small at step 0, larger once Adam moments exist).

### Resumability
- Resume = **same command** plus `--existing-lora <last>.safetensors`, with the
  same `--out`; `--steps` is the run's *total* length. It restores optimizer,
  RNG, step and the loss/eval curves.
- `rank`, `alpha`, `targets` and `optimizer` must match, or the state is ignored
  with a warning and training restarts from the weights.
- **Ctrl-C does not run the final save** — the `finally` block only cleans up
  GPU state. So an interrupt leaves the last `--save-every` partial as the
  resume point. `.resume` writes are atomic (tmp + rename); `.safetensors`
  writes are not, so a crash mid-write can tear one partial — resume from the
  previous one and verify with the trainer's `tests/verify_lora_file.py`.
- The VAE-latent cache is keyed by path+size+mtime+flags, so resume/re-run skips
  the ~20 min encode.

### Observability
- Sessions run **inside this same VM**. If the user launches training in another
  terminal on this machine, a session can read `/content/logs/train.log`,
  `ComfyUI/models/loras/*` and run `nvidia-smi`. If training runs on a different
  machine, a session cannot see it — paste the log or copy the artifacts over.
- This is **on-demand only**: no background watching, no notifications. `tee` to
  `/content/logs/train.log` is what makes a run visible.
- From those files: step/ETA, loss/avg/grad-norm, the held-out eval curve, which
  checkpoints exist, process/GPU state, and the crash cause from the log tail.
- Sessions never start, stop, or resume the training process.

## 13. GCS backup — `backup_to_gcp.py` (new session 4)

Run in a **second terminal** next to training. Every `--interval-minutes`
(default **25**) it mirrors the recovery-critical folders to
`gs://akbar-december-2024-backup/YuE2-3B_13092026/run_backup/` with
`gsutil -m rsync -r`:

| local | remote | why |
|---|---|---|
| `ComfyUI/models/loras/` | `run_backup/loras/` | `.safetensors`, `.resume`, `.loss.json` |
| `/content/logs/` | `run_backup/logs/` | `train.log` for crash diagnosis |
| `agent_notes/` | `run_backup/agent_notes/` | the plan + exact commands |

```bash
cd /content/yue2_lora_finetuning
python backup_to_gcp.py                    # every 25 min; first pass immediately
python backup_to_gcp.py --include-cache    # also mirror the (regenerable) latent cache
python backup_to_gcp.py --once             # single pass for cron
```
Safety: **append/update only — no `-d`, so it never deletes remotely**; a
`--settle-seconds 60` guard on `loras/` avoids grabbing a file mid-write, and a
torn upload self-heals on the next pass. It is read-only locally. Note it needs
`gcloud`/`gsutil` auth (currently the user's account) to stay valid. Verified
end-to-end: a real pass uploaded `logs/` and `agent_notes/` successfully (and
gsutil honours only the **last** `-x`, so the excludes are combined into one
alternation). The dataset is already in GCS and the checkpoint is on HF, so
neither is mirrored.

### Git state
Session 4 changes are pushed: `origin/main` is `89ca4cf` (adds
`backup_to_gcp.py` + `agent_notes/current.md`). The remote already had `91a9291`
with byte-identical `AGENTS.md`/`setup.sh` edits, so those were not duplicated.
Auth is `gh` with a stored user token; push uses
`git -c credential.helper='!gh auth git-credential'` (no git-config change).
`ComfyUI/` and `agent_notes/` remain untracked/volatile candidates for
`.gitignore` if the churn becomes annoying.

## 14. Session 4 (cont.) — first real run + generation test — the setback

This section exists because the user ended the session **dissatisfied**: "This is
not like I imagined things… at least we should have things like in the base model
which didn't have pronunciation problem (neither the training data which were
shortlisted from the best Suno tracks)." Treat the following as the new baseline
of truth, and the acceptance bar as: **output no worse than the base model, with
its correct classical-Fusha pronunciation.**

### The run
```bash
train_cli.py acoustic --comfy-root .../ComfyUI --checkpoint yue2_3b_bf16.safetensors \
  --data /content/data/dataset --eval-holdout 5 --seed 2002 \
  --segment-seconds 30 --steps 1500 --save-every 50 --out maqam_acoustic_v1
```
- **1500 steps in 450 s (~0.3 s/step)**, 251 items, 112 modules, rank 16, 14.68 M
  trainable params. Held-out loss **1.0510 → 1.0078 (-4.1%)**, plateaued by ~250;
  train loss min 0.7369. 29 partials + final; final LoRA 28 MB, `.resume` ~117 MB.
- **It is a smoke test, not a fine-tune:** 1500 × 30 s ≈ 12.5 h of audio vs an
  18 h corpus = **<1 epoch**. The 1500 figure came from the trainer README's demo,
  not from a production budget — that framing was wrong.

### The generation test (headless ComfyUI)
- Server `python ComfyUI/main.py` on `127.0.0.1:8188`; stock workflow
  `example_workflows/yue2_generate_with_lora_api.json` (acoustic LoRA on the MODEL
  path; planner/clip path left at base). Held-out poem
  **`hijaz_0038_01-retake_take01`** (Imru' al-Qais, Maqam Hijaz), fed the dataset's
  `.style.txt` + cleaned `.lyrics.txt`.
- The LoRA attaches correctly (`112 patches attached`; the 336 `lora key not
  loaded` warnings are node 17's clip path and harmless).
- A/B pairs, same seed, `strength_model` 1.0 vs 0.0:
  - **90 s**: `max_abc_tokens=2048`, `max_duration=90` → both budgets hit,
    truncation. `waveform 0.963 · MFCC 0.999`.
  - **214 s (full)**: `max_abc_tokens=8192` (ABC finished on its own),
    `max_duration=214` → music budget hit, so a **hard cut at 3:34** with no end
    token. `waveform 0.946 · chroma 0.996 · MFCC 0.998 · centroid 2131 vs 2111 Hz`.
- Files: `/content/ab_lora_vs_base/{01_base_no_lora,02_lora_strength1.0}.flac`;
  90 s pair in `ComfyUI/output/yue2/`; the two FLACs + three workflow JSONs were
  copied to `gs://akbar-december-2024-backup/YuE2-3B_13092026/run_backup/session4_tests/`
  (the VM dies with the session, so that is the only durable copy).

### Two hard findings
1. **The acoustic LoRA is a near-no-op at this scale** (`waveform 0.946`,
   `MFCC 0.998`). It is not the style transfer the user is after, and no amount of
   acoustic training will touch pronunciation or maqam.
2. **Pronunciation is decided upstream, by the planner's semantic tokens.** The
   base and LoRA renders of a pair are fed the **identical** token stream (server
   log: only **2 ABC + 2 music sampling passes for 4 runs**; the base runs reused
   the LoRA runs' cache), so the acoustic LoRA can neither cause nor fix the ق
   problem. The base planner also produced **K:Fm** (F minor — Western harmony),
   not Hijaz.

### The user's verdict / acceptance criterion
The render is "too melismatic to artificial level", with pronunciation problems
(letter **ق** not rendered as classical Fusha, unlike the base model they tested
and unlike the shortlisted Suno tracks). This is a **regression relative to the
base model** and is not acceptable. Any next attempt is judged against base-model
pronunciation parity, not against the loss curve.

### What is still unknown — superseded, see §15
This section originally asked "what was the earlier base-model test that
pronounced ق correctly, and how does its pipeline differ from ours?" **That has
now been answered: it's `akbargherbal/fine_tuning_ai_music_lora`'s
`yue2_generate.py`, confirmed by the user as the ق/ح/خ-clean baseline, and its
pipeline has been diffed against ours.** Of the four candidate culprits listed
here originally:
1. ~~Tashkeel~~ — **ruled out**, confirmed by the user (full harakat, not a
   problem for this model).
2. Style format ("melismatic runs…") — **still open**, see §15.
3. Planner sampling params — **still open**, see §15.
4. ~~The ABC being Western (K:Fm)~~ — **confirmed pre-existing in the base
   model, not a fine-tuning regression** — see §15.

A fifth culprit neither this list nor the earlier session considered — a
missing `cfg_scale` wire in the generation node — is now the **leading
hypothesis**. See §15 for the finding and the next test.

### Runtime at session end (ephemeral — gone with the VM)
- ComfyUI server **PID 81132** (`127.0.0.1:8188`), holding ~7.5 GB VRAM.
- `backup_to_gcp.py --include-cache` loop **PID 64625**.
- Both stop when the Colab runtime ends; nothing needs cleaning up.

## 15. Session 5 — pipeline diff against the known-good baseline (no training run)

This session ran **no training and no generation** — it's a read-only diff of
two codebases, done outside the Colab VM against both this repo and
`akbargherbal/fine_tuning_ai_music_lora` (cloned fresh for comparison). It
exists to stop the next session from re-litigating "did the LoRA cause the ق
regression" from scratch, or spending a GPU session re-training the acoustic
LoRA to test a hypothesis this diff already rules out.

### User-confirmed facts (treat as settled)
- **Tashkeel is not the cause.** Full harakat works fine on this model;
  §14 point 1 is closed.
- **Pronunciation was correct before any of this project's fine-tuning
  existed** — the user directly recalls no ق/ح/خ mistakes when they first
  tried base YuE2, prior to any LoRA work.
- **6 minutes is the user's real ceiling for song length**, i.e.
  `max_duration` should be set to **~400s** (comfortably above the corpus's
  369.7s max), not the 214s used in §14's test — the 3:34 hard cut in §14 is
  fully explained by that one number and needs no further diagnosis.

### The baseline pipeline, read directly from the sibling repo
`akbargherbal/fine_tuning_ai_music_lora/yue2_generate.py` is the script the
"clean" pronunciation almost certainly came from — confirmed by the user as
matching their memory of that earlier test. It does **not** use ComfyUI at
all: it calls the `yue2` pip package's `YuE2Pipeline` directly, one call doing
both composition and music generation:
```python
config = GenerationConfig(ode_steps=ode_steps)
pipe = YuE2Pipeline.from_pretrained(args.model, vae=args.vae, device="cuda", generation_config=config)
song = pipe(style=style, lyrics=lyrics, cot=args.cot, seed=args.seed, cfg_scale=args.cfg)
```
Defaults: `--cot full`, `--cfg 1.2` (documented in its own README as
"text-guidance scale... prompt adherence"), `--quality standard` →
`ode_steps=32`. Style/lyrics are the **raw manifest fields, sent verbatim** —
full Suno control syntax, full tashkeel, no re-rendering through
`maqam_prompt_generator.build_prompt()` or `clean_lyrics()`.

Its own README/findings (independent of this repo, predates all LoRA work)
also state: **"The model consistently defaults to Western minor keys
regardless of the maqam requested in the style text — this held across every
generation mode tested"** — i.e. **the K:Fm-instead-of-Hijaz result in §14 is
a known, pre-existing base-model limitation, not something the fine-tuning
caused or worsened.** Closes §14 point 4 as "not a regression" rather than
"unresolved."

### The finding: `cfg_scale` has no path into `YuE2GenerateMusic`
Traced through `speedyrulz/ComfyUI-YuE2-Trainer` source (`yue2_trainer/planner.py`):
```python
def generate_music(clip, style, lyrics, abc, mode, seed, max_seconds,
                    sampling=None, cfg_scale: Optional[float] = None):
    ...
    if cfg_scale is not None:
        tokens["cfg_scale"] = cfg_scale
    ...
    ids, truncated = te._generate(..., cfg_scale=tokens["cfg_scale"], ...)
```
The plumbing exists — but grepping `nodes.py` for `cfg_scale` returns **zero
matches**. The `YuE2GenerateMusic` node (used in §14's test and in
`example_workflows/yue2_generate_with_lora_api.json`) exposes `style, lyrics,
seed, mode, max_duration, temperature, top_p, top_k, repetition_penalty` —
**no `cfg_scale` input**. `generate_abc()` (the composition/ABC stage) has no
`cfg_scale` parameter at all, at any layer — no hook, unlike the music stage.

**Consequence:** every ComfyUI generation run in this project so far —
§14's A/B test included — ran with `cfg_scale=None`, falling through to
whatever the underlying `clip.tokenize()`/text-encoder default is when the
caller never sets it. That default has **not been verified** and is not
known to be `1.2`. This is a plausible, mechanistic explanation for both
symptoms at once:
- Weaker text-guidance → the model tracks the literal Arabic lyrics less
  closely → plausible source of mispronunciation.
- Same mechanism → plausible source of "too melismatic to an artificial
  level" / "doesn't sound like genuine Arabic singing" (§14's user verdict) —
  less anchored to the actual input, more free-running.

This is a **generation-node wiring gap**, unrelated to the acoustic LoRA
weights themselves (already shown near-no-op in §14, waveform corr.
0.946–0.998) and unrelated to training in general — it would apply identically
with the LoRA disabled.

### Recommended next test (cheap — no training required)
1. Patch `YuE2GenerateMusic` to accept and forward `cfg_scale` to
   `generate_music()` (the function signature already supports it — this is a
   node-input change, not a training-code change), or call
   `generate_music()`/`generate_abc()` directly from a standalone script,
   bypassing the node.
2. Set `cfg_scale=1.2` (matching the known-good baseline) and `max_duration=400`
   (per the 6-minute ceiling above).
3. Re-render the same held-out poem used in §14
   (`hijaz_0038_01-retake_take01`) and do the same base-vs-LoRA A/B, listening
   specifically for ق/ح/خ and for "genuineness," not just loss numbers.
4. **Do not re-run acoustic-LoRA training to test this.** The LoRA is not
   implicated; retraining would burn a session without confirming or denying
   the `cfg_scale` hypothesis.

### If `cfg_scale=1.2` does not fully resolve it
Re-open, in this order (updated from §14's original list, tashkeel and the
Western-key bias now removed as closed):
1. Style caption wording — the `vocals:` line's *"melismatic runs …
   melismatic phrasing"* vs. the baseline's plain comma-tag line.
2. Planner sampling params (`temperature`/`top_p`/`top_k`/`repetition_penalty`)
   — note `YuE2GenerateABC` and `YuE2GenerateMusic` use **different** defaults
   from each other (ABC: `0.7/0.9/30/1.005`; Music: `1.0/0.95/100/1.2`), so any
   comparison needs to track which node's params are being changed.
3. The planner-LoRA/ABC work (§8) remains the real lever for maqam control
   regardless of how the `cfg_scale` test turns out — that's a separate,
   already-understood blocker (symbolic-score generation), not something this
   session's finding changes.

## 16. Session 6 — the missing tokenizer, found and already wired in

This session ran **no training and no generation**. It (a) identified the
community artifact the project had been missing, (b) verified from the
trainer's actual source that it is already integrated, and (c) rewrote
`PLAN.md` around it. It exists so the next session starts the real work instead
of re-deriving the toolchain.

### The finding

The video "How I trained the missing YuE2 Tokenizer" (channel *Make The Robot
Do It* / Nora) is about `Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4`
on Hugging Face — **the audio → semantic-token encoder YuE2 never shipped**,
plus a rank-32 NAR LoRA companion. Without it, real recordings have no semantic
tokens, which is why the planner's semantic path was unusable and the project
was stuck on ABC/SheetSage2. Full facts are in §2.

### It is already integrated into our backend (verified from source)

`speedyrulz/ComfyUI-YuE2-Trainer` ships the head as a first-class option. Read
directly from `train_cli.py` / the README this session:

- **`--semantic-head <file>`** (common flag) — predicts tokens for every song
  and writes `<song>.semantic.npy`; `--semantic-force` recomputes existing
  sidecars; **`--mert`** defaults to `m-a-p/MERT-v2-FullSong`.
- Acoustic: **`--use-semantic`** conditions on those tokens;
  `--conditioning inference_like` lays out the prefix the way generation does;
  `--sample-every`/`--sample-tokens` render a stream with the LoRA under
  training (i.e. an audible checkpoint A/B).
- Planner: **`--semantic`** adds the semantic-token target; **`--no-abc`** drops
  the ABC target; **`--abc-dropout`** trains a share of draws behind the
  no-sheet prompt; **`--kl-weight`** is a trust region toward the base model;
  `--probe-every` writes whole scores/token streams per checkpoint (the
  over-training signal).
- The node equivalents are **YuE2 Semantic Tokens (community head)** and the
  example graph `yue2_train_semantic_planner_api.json`.

**Consequence:** the planner can now train on **semantic tokens with no ABC
scores** (`--semantic --no-abc --abc-dropout 0.5`), which makes the
SheetSage2/melismatic-Arabic risk moot and removes §8's last blocker. This
**reverses the acoustic-first ordering** (§8): planner first, acoustic second.

### Honest caveats

- **The tokenizer is approximate** (~16% exact top-1 on YuE2's own songs; ~95%
  by ear on round-trips). The planner learns from near-miss codes; that's the
  current community ceiling.
- **The two sources disagree on the NAR LoRA** — Nora ships it, speedyrulz
  measured it makes renders *less* similar and doesn't use it. Unresolved; A/B
  it if Stage 3 underdelivers.
- **Nothing here is validated on Arabic/maqam.** We are early adopters.
- The `PLAN.md` that existed before this session was **grounded in the
  fabricated ai-toolkit PR #1042 claim** — its §0 was actively misleading. It
  has been rewritten from scratch; treat the old text as gone.

### What changed as a result

- `PLAN.md` — full rewrite (bootstrap additions → tokenize → planner LoRA →
  acoustic LoRA → evaluation), with exact CLI commands.
- `context.md` — §2 (tokenizer bullet + planner-target correction), §7
  (`PLAN.md` description), §8 (ABC blocker de-prioritized, ordering reversed),
  §10 (new priorities), and this section.
- Still open, unchanged: `status`-field check (§5), `--max-per-song` capping,
  corpus-drift re-verify, and the generation-side `cfg_scale` gap (§15, now a
  prerequisite for trustworthy evaluation rather than the top training item).
