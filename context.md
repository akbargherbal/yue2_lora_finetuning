# Context: YuE2-3B LoRA Fine-Tuning — Arabic Maqam Orchestral Rock

> Read this whole file before doing anything else. It's written so a fresh
> Claude session can pick this project up without the user re-explaining it.
> Repo: https://github.com/akbargherbal/yue2_lora_finetuning.git
> (I have not been able to browse this repo's contents myself in any session
> so far — web_fetch only works on URLs already surfaced by search/fetch in
> the current conversation, and this one wasn't. Treat the repo as the source
> of truth for actual current file versions; this document is my summary of
> how we got here, not a substitute for reading the repo directly if you can.)

## 1. The goal

Fine-tune a LoRA for **YuE2-3B** (m-a-p's open lyrics-to-song model) on the
user's back-catalog of Suno-generated tracks, so the LoRA reproduces their
specific "winning" style: symphonic cinematic orchestral ballad / heavy rock,
grand-concert-hall acoustics, deep male Arabic vocals, classical Arabic
poetry as lyrics, in one of four Arabic maqams (Hijaz, Nahawand, Ajam, Kurd).

This session (and the one before it) covered **dataset preparation only**.
No LoRA training has been run yet.

## 2. Background on the tools involved (verified via web search, Sep 2026)

- **m-a-p/YuE2-3B**: open, ~3B lyrics-to-song model. Takes lyrics + a style
  prompt, produces a full song with vocals and accompaniment. Competitive
  with Suno v5/v6 on the team's own WildSongBench (best-of-8 SongBench avg
  6.9632 vs Suno v5's 6.8721). Runs locally in BF16 on a 24GB GPU, no
  quantization needed. Upstream repo: `github.com/multimodal-art-projection/YuE`.
- **LoRA training path**: `ostris/ai-toolkit` merged YuE2 support in PR #1042
  ("YuE2" by jaretburkett) on **Sep 14, 2026** — very recent, so treat
  anything about it as liable to change fast. Key mechanics learned from
  the PR/README:
  - Uses the `Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4`
    audio tokenizer (credited to "Kytra").
  - Captions get normalized by the toolkit before training.
  - There's an optional symbolic score layer (a "sage sheet") — the toolkit
    trains with it present 50% of the time and absent 50% of the time, so
    the resulting LoRA works whether or not a score is supplied at inference.
  - Dataset pairing convention is the same as ai-toolkit's other LoRA types:
    one audio file + one caption text file per example (plus optionally a
    score file).
- **Other YuE2 LoRA trainers exist** as separate ComfyUI node packs —
  `speedyrulz/ComfyUI-YuE2-Trainer` and `Starnodes2024/ComfyUI-YuE2-Trainer`
  (two different projects, confusingly similar names). Neither has been
  evaluated against the ai-toolkit path for this project yet. **Worth
  comparing before committing to a training backend if that hasn't happened
  in a session between this one and now.**
- The user separately appears to have an earlier/adjacent repo,
  `akbargherbal/fine_tuning_ai_music_lora`, testing raw YuE2-3B *inference*
  (no fine-tuning) against their existing Suno prompts/lyrics as-is on a
  Colab L4. I found this via search but haven't confirmed its relationship
  to the `yue2_lora_finetuning` repo — **ask the user if it's relevant.**

## 3. The creative material

Classical/pre-Islamic and Andalusian Arabic poetry (the mu'allaqat and
similar — poets referenced across the corpus include Imru' al-Qais, Antara,
Amr ibn Kulthum, Tarafa, al-Harith ibn Hilliza, al-Nabigha al-Dhubyani,
al-Nabigha al-Ja'di, al-A'sha, Majnun Layla, Malik ibn al-Rayb, al-Shanfara
(Lamiyat), Abu Tammam, Ibn Zuraiq, Ibn Zaydun, al-Mutanabbi, Jarir, and
others) set to one fixed musical template and rendered via Suno v5.

## 4. The prompt template — `maqam_prompt_generator.py`

This is the user's own tool (they wrote it, I didn't) that generates the
Suno prompt block. I've read it in full; key points:

- Everything is fixed except the maqam name, an optional per-track `mood`
  phrase, and (in standard mode) a lyric start-phrase: `GENRE_STANDARD`,
  `PRODUCTION_STANDARD`, `INSTRUMENTATION`, and `EXCLUDE` are constants,
  identical across every track.
- Two modes: `standard` (lyrics, includes a `vocals` field and a
  `[START_ON: "..."]` header) and `instrumental` (no lyrics, no `vocals`
  field, no start-phrase header). Note: instrumental mode deliberately does
  **not** add "no vocals" to EXCLUDE — non-lyrical vocalization is left
  possible on purpose.
- Only 4 maqams are offered (Hijaz, Nahawand, Ajam, Kurd) because the others
  (Rast, Bayati, Sikah, Saba, etc.) need quarter-tone intervals with no
  faithful Western mapping.
- The tool's docstring explicitly notes an earlier attempt to have an LLM
  (Gemini) vary the `mood` field introduced unwanted drift, and that field
  was removed from *generation* for that reason. This is why, in dataset
  prep, I chose to re-derive every caption from `build_prompt()` rather than
  trust the free-text `styles` string stored in old manifests, and why I
  kept the caption vocabulary closed rather than reintroducing generated
  variation — same failure mode, different context (training data vs.
  generation).

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

**Critical fact, discovered partway through this project, not obvious from
the JSON schema alone:** each `workspace_manifest.json` lists *every* track
ever generated in that session, including ones the user rated below 4 stars
and deleted. **The only reliable signal for "this one made the cut" is that
its audio file is still physically present in the same folder.** There is no
`status` field to trust for this. `prepare_dataset.py` treats file-existence
as ground truth.

Other real-world messiness already handled by the script (see §6):
- The **same poem can recur across multiple, differently-dated workspace
  folders**, sometimes with inconsistent capitalization
  (`lamiyat_alshanfara_14082026` vs `Lamiyat_Alshanfara_16082026`) — these
  must be treated as one song for train/val splitting.
- **Multiple surviving takes of one poem are not duplicates to collapse.**
  If `SONG_A`, `SONG_C`, and a `Retake_..._SONG_A` all still exist on disk,
  all three already passed the user's by-ear 4–5★ filter — they're wanted
  audio diversity for the same caption, not redundant noise. (This was a
  **correction** from an earlier draft of this plan, made before the real
  corpus layout was known — the very first version of the script wrongly
  assumed `SONG_A`/`SONG_B` pairs were near-identical Suno two-take noise
  and collapsed them to one per song. Don't reintroduce that assumption.)
- Filenames are inconsistent: usually `<title>_SONG_<letter>.mp3`, sometimes
  `Retake_<title>_SONG_<letter>.mp3` / `RETAKE_...`, occasionally no suffix
  at all. This doesn't matter for song identity (`original_title` in the
  JSON is authoritative), only for locating the audio file.
- Arabic filenames can have Unicode normalization mismatches (NFC vs NFD)
  between the manifest string and the actual file on disk — the script
  normalizes both sides before comparing, or a valid track can silently look
  "filtered out."
- The maqam is taken from the **top-level folder name** (authoritative),
  cross-checked against the "Maqam X" text inside the caption; any mismatch
  between folder and caption gets printed as a warning, not resolved
  automatically.

**Real numbers from the user's actual run** (`--dry-run`, most recent):

```
Filtered out (manifest entries with no matching audio file): 1650
Resolved 257 kept (4-5*) tracks across 136 distinct poems.
  Hijaz      tracks= 60  poems= 31
  Nahawand   tracks= 73  poems= 39
  Ajam       tracks= 62  poems= 33
  Kurd       tracks= 62  poems= 33
Most-retried poems (top 10): 6x Kurd "عهد الصبا..."; 5x Hijaz "إني ذكرتك
  بالزهراء" (Ibn Zaydun); 5x Nahawand "الظعائن ووصف المحبوبة"; several 4x
  entries scattered across Ajam/Hijaz/Kurd.
Split: 239 train / 18 val tracks (123 / 13 poems).
```

This matches the user's own pre-tooling estimate (~250 tracks: Nahawand 73,
Kurd 62, Hijaz 60, Ajam 62) almost exactly — good sign the pipeline logic is
sound, not an artifact of a bug.

My read on the retry skew (given to the user, not yet revisited): mild, not
alarming — ~10 poems have 4-6 surviving takes (~17% of tracks, ~7% of poems),
the rest mostly 1-2. Recommended **not** capping by default, suggested trying
`--max-per-song 3` as an A/B comparison if the user wanted to see the effect.
**Not yet decided either way — revisit if asked.**

The 18 val tracks (not a flat 10% of 257) is expected, not a bug: the split
picks whole *poems*, and poem group sizes vary, so the poem-level ratio
(13/136 ≈ 9.6%) is what actually matters for preventing lyric leakage across
train/val — not the raw track count. Don't "fix" this by changing the
splitting unit unless the user specifically wants track-count precision over
leakage-safety.

## 6. What's been built: `prepare_dataset.py`

Walks `<dataset_root>/<maqam>/<workspace>/workspace_manifest.json`:

1. Filters by file-existence (§5) with Unicode-normalized filename matching.
2. Takes maqam from the folder name; cross-checks against the caption text.
3. **Keeps all surviving takes** of a poem by default (no forced A/B dedup).
   `--max-per-song N` caps this if ever needed; not applied by default.
4. Groups by normalized `original_title` + maqam **across workspaces**, so
   splitting never leaks the same poem into both train and val.
5. **Re-renders every caption** via `maqam_prompt_generator.build_prompt()`
   rather than trusting the stored `styles` text verbatim — irons out minor
   template wording drift accumulated as the generator script evolved.
   Strips the Suno-specific `[Is_MAX_MODE...]/[START_ON...]` control header
   and the `PROMPT:`/`EXCLUDE:` wrapper by default, keeping only the
   `genre:`/`vocals:`/`production:`/`instrumentation:`(/`mood:`) lines.
   `--keep-control-header` and `--include-mood` flags exist to change this.
6. Splits into `dataset/train/` and `dataset/val/`
   (`<maqam>_<title-slug>_take<NN>.<ext>` + matching `.txt` caption),
   stratified by maqam, plus a `manifest.csv` with full bookkeeping
   (clip_id, workspace, title, maqam, mood, take number, source/dest paths).
7. Prints a "most-retried poems" diagnostic before splitting.

**Testing performed** (by me, in-session, with synthetic fixtures — not by
the user, though the user then separately ran it for real):
- A hand-built synthetic corpus reproducing the exact real-world edge cases
  (cross-workspace duplicate titles, retake-heavy poems, no-suffix
  filenames, deliberately-filtered manifest entries) — all resolved
  correctly.
- A scale simulation matching the real per-maqam counts (~257 tracks) to
  sanity-check split ratios before the user ran it on real data.
- Caught and fixed one real bug during testing: caption extraction was
  leaking a trailing markdown code-fence backtick into the `.txt` file
  (fixed by filtering to only lines starting with the known field prefixes,
  rather than slicing between line indices).

## 7. Files that should exist in the repo

- `maqam_prompt_generator.py` — the user's original tool (uploaded, not
  written by me). Source of truth for the caption template.
- `PLAN.md` — full written dataset-prep plan, revised once after the real
  corpus layout was seen. Sections: 0 (what ai-toolkit's YuE2 LoRA path
  expects) · 1 (folder-is-the-filter insight, real corpus facts) · 2 (audio
  cleanup — **not yet acted on**, see §8) · 3 (caption prep) · 4 (optional
  score extraction) · 5 (directory layout) · 6 (train/val split by poem) ·
  7 (script pointer) · 8 (pre-training QC checklist).
- `prepare_dataset.py` — the script described in §6.
- This file (`context.md`).

If any of these look different in the actual repo than described here,
**trust the repo** — this document may be stale relative to edits made
outside our sessions.

## 8. Explicitly NOT done yet — don't assume decisions were made

- **Audio-side cleanup** (PLAN.md §2: loudness normalization — e.g. EBU R128
  / -14 LUFS via `pyloudnorm` or `ffmpeg-normalize` — silence trimming,
  sample-rate/format standardization). The user's own words at the end of
  the last session: *"I am not sure about this loudness-normalization/
  silence thing... we'll see next session."* This is **not a rejection**,
  just deferred. Walk through what it is and why it matters for a music
  LoRA before assuming they want it or don't.
- Whether `prepare_dataset.py ... --out-dir dataset` has actually been run
  **without** `--dry-run` yet (i.e. whether `dataset/train` and
  `dataset/val` physically exist with copied audio + captions). As of the
  end of this session, only the dry-run had been shown to me. **Confirm this
  first**, don't assume either way.
- The optional symbolic score ("sage sheet"/ABC transcription via
  SheetSage2) step from PLAN.md §4 — flagged as possibly not worth it for
  melismatic Arabic vocal lines (transcription tools are tuned for Western
  tonal/rhythmic conventions), decision deferred, nothing started.
- The `--max-per-song` capping decision (§5 above) — leaning toward no cap,
  not finalized.
- **No LoRA training has been run.** No training backend (ai-toolkit vs. one
  of the ComfyUI trainers) has been chosen yet either — see §2.

## 9. User context

- Python developer. Prefers documentation in Markdown.
- Has been doing serious manual curation (weeks of by-ear listening across
  ~1900 generated tracks to get to 257 keepers) — treat their quality bar as
  high and already-applied; don't suggest re-filtering audio quality, only
  technical/structural cleanup (loudness, silence, format).
- Values scripts that are actually tested against realistic data before
  being handed over, not just plausible-looking code — the back-and-forth
  in this project involved me building synthetic fixtures to catch bugs
  before the user ran anything for real. Keep doing that.

## 10. Suggested first steps for next session

1. Ask/confirm whether the real (non-dry-run) dataset build has happened
   since we last spoke.
2. Revisit the audio-cleanup question from §8 — explain it plainly, decide
   together, and only then write the cleanup script.
3. Decide the training backend (ai-toolkit YuE2 support vs. one of the two
   ComfyUI-YuE2-Trainer projects) if not already decided — this will need a
   fresh look at each project's README/docs since this space is moving fast
   (all of it postdates my training cutoff).
4. Once a backend is chosen, translate `dataset/train`/`dataset/val` into
   whatever config format that backend expects (not yet investigated).
