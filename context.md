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

## 1. The goal

Fine-tune a LoRA for **YuE2-3B** (m-a-p's open lyrics-to-song model) on the
user's back-catalog of Suno-generated tracks, so the LoRA reproduces their
specific "winning" style: symphonic cinematic orchestral ballad / heavy rock,
grand-concert-hall acoustics, deep male Arabic vocals, classical Arabic
poetry as lyrics, in one of four Arabic maqams (Hijaz, Nahawand, Ajam, Kurd).

**Dataset preparation is now complete and verified** (see §5–6). No LoRA
training has been run yet. The open work is choosing/setting up a training
backend (§8) and mapping the dataset onto whatever that backend expects.

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
  - Planner training needs an ABC score per track, produced either by
    SheetSage2 (fixed 300s transcription window — a real risk for our
    median 253s / max 370s tracks, and possibly a poor fit for melismatic
    Arabic vocal lines, same concern PLAN.md §4 already raised) or supplied
    by hand. **Not yet decided how to get scores for our corpus.**
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
- `PLAN.md` — original written dataset-prep plan. Mostly executed now;
  §2 (audio cleanup) turned out to be unnecessary rather than done (§6),
  §4 (symbolic score) is still open and now tied to the planner-LoRA
  decision (§2 above), not just an optional nice-to-have.
- `prepare_dataset.py` — updated this session, see §6.
- `verify_dataset.py` — **new this session**, see §6.
- This file (`context.md`).

## 8. Explicitly NOT done yet — don't assume decisions were made

- **Training backend — DECIDED session 3: `speedyrulz/ComfyUI-YuE2-Trainer`,
  by elimination.** ai-toolkit is ruled out (§2, resolved against the
  primary source). Nothing has been installed or run yet, so setup is still
  entirely ahead of us — but the *choice* is no longer open, and re-opening
  it would mean finding a fourth candidate, not revisiting ai-toolkit.
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
    **acoustic is unblocked right now** (dataset built and verified 0/0,
    sidecar naming already matches the trainer's convention, uniform 48kHz
    stereo, random-crop handles our 370s max — plausibly a direct
    drop-in), while **planner is blocked on ABC scores** (next point).
  - **Recommended order: acoustic first**, which also validates the whole
    pipeline end-to-end on real data before sinking effort into
    transcription. Not yet ratified by the user.
- **Symbolic score / ABC transcription** for the planner LoRA path.
  SheetSage2's fixed 300s window is a mismatch for some of our tracks
  (median 253s, max 370s) and its fit for melismatic Arabic vocal lines is
  unverified. No scores have been generated. This is now a **blocking**
  decision if the planner LoRA is wanted, not an optional side-quest.
- **`--max-per-song` capping** — leaning no-cap, not finalized.
- **`status` field investigation** — see §5, quick check not yet run.
- No LoRA training of any kind has been run. This remains true after
  session 3 — session 3 was research and bookkeeping only, no code changed.

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
- Comfortable running git/shell commands themselves on their own machine
  (Windows/PowerShell); Claude sessions typically only have a scratch clone,
  not push access to the actual repo — hand off files + exact commands,
  don't assume a session can commit on the user's behalf.

## 10. Suggested first steps for next session

Steps 1, 3 and 5 of the previous version of this list are **done** — the
generator is committed (§4), ai-toolkit is ruled out and the backend is
settled (§2/§8), and planner-vs-acoustic has been reframed as an ordering
with a recommendation (§8). What's left:

1. **Ratify "acoustic first" with the user** (§8) if not already done. This
   gates everything below.
2. **Stand up `speedyrulz/ComfyUI-YuE2-Trainer` and train the acoustic
   LoRA.** Map `dataset/train` / `dataset/val` (with its
   `.style.txt`/`.lyrics.txt` sidecars) onto what its nodes/CLI actually
   expect. The sidecar naming already matches its convention (§6), so this
   may be close to a direct drop-in — **but that has never been tried, so
   don't write it up as if it worked until it has run.** This will be the
   first LoRA training of any kind in this project.
3. **Evaluate the acoustic LoRA honestly.** Expect "same composition, our
   production" — per §2 that's the *designed* behaviour, not a failure.
   Judging it by whether maqams changed would be judging it by the thing it
   structurally cannot do.
4. **ABC score generation for the planner LoRA** — still the one genuinely
   unexplored blocker. Options: SheetSage2 (300s window vs. our 253s
   median / 370s max), the `t8star/YuE2-Comfy` bundle (§2, unverified),
   or hand-authored scores. Don't assume an answer; nobody has tested any
   of these against melismatic Arabic vocal lines.
5. **Run the `status`-field check from §5** — low effort, resolves a real
   ambiguity in the filtering logic, independent of everything above. Can
   be done any time the user is at the corpus machine.
6. **`--max-per-song` capping** — still leaning no-cap, still not decided.

Note the corpus can drift between sessions (the user deletes tracks by ear).
`verify_dataset.py --dataset-root ...` detects this; re-run it before
trusting the §5 numbers.
