# Dataset Prep Plan — YuE2-3B LoRA (Maqam / Arabic Orchestral Rock)

> **Revision note:** this plan was updated after seeing the real corpus layout
> (`min_4stars_ai_music/<maqam>/<workspace>/`). Two assumptions from the first
> pass were wrong and are corrected below: (1) the manifest is a *superset* —
> a track survives only if its audio file is still physically present next to
> the manifest, which is the actual result of your by-ear 4–5★ curation; (2)
> multiple surviving takes (`SONG_A`/`SONG_B`/`RETAKE_...`) of the same poem
> are **not** near-duplicate noise to collapse to one — if more than one take
> of a poem passed your listening pass, that's real, wanted audio diversity
> for the same caption. See `prepare_dataset.py` for the corrected pipeline.

## 0. What YuE2's LoRA path actually expects

Fine-tuning support landed in `ostris/ai-toolkit` (PR #1042, merged Sep 14 2026),
built on the `Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4` audio
tokenizer. Relevant to dataset prep specifically:

- Each training example is an **audio file + a normalized caption**, the same
  pairing convention ai-toolkit uses for image/video LoRAs (audio ↔ style text).
- There's an **optional symbolic score ("sage sheet")** layer. The toolkit
  trains with it present 50% of the time and absent 50% of the time, so the
  resulting LoRA works whether or not you supply a score at inference. You
  don't strictly need scores to get a working LoRA — but including them for
  at least some tracks improves melody/chord controllability, which matters
  for your winning-template use case (you care about consistent structure,
  not just timbre).
- Captions get **normalized** (whitespace/casing/punctuation) before training,
  so raw human-authored captions are fine as input.
- There's a dataset dropout setting (some fraction of steps train caption-less),
  which is a training-time knob, not something you need to bake into the data.

None of this requires re-deriving your prompt template — it requires turning
what you already have into the right *file layout*.

## 1. Inventory the source material — the folder tree *is* the filter

Real layout:

```
min_4stars_ai_music/
├── ajam/       <workspace>/  workspace_manifest.json + surviving .mp3s
├── hijaz/      ...
├── kurd/       ...
└── nahawand/   ...
```

~257 kept tracks across ~156 distinct poems, close to your reported
Nahawand 73 / Kurd 62 / Hijaz 60 / Ajam 62.

Key facts this structure implies:

1. **Each `workspace_manifest.json` lists every track ever generated in that
   session — including ones you rated below 4 stars and deleted.** The
   *only* surviving signal for "this one made the cut" is that its audio
   file still exists in the same folder. There's no separate `status` field
   to trust — presence on disk is the ground truth. `prepare_dataset.py`
   treats it exactly this way: walk each manifest, keep an entry only if
   `assigned_filename` resolves to a real file next to it.
2. **The maqam comes from the top-level folder name**, not from re-parsing
   the caption text. That's more reliable than regexing "Maqam X" out of
   `styles` (which is what the first draft of this script did) — the folder
   is how you actually organized your own judgment. The script still
   cross-checks the caption's stated maqam against the folder and warns on
   any mismatch, since that's a cheap way to catch a mislabeled track.
3. **Don't collapse `SONG_A`/`SONG_B`/`RETAKE_...` takes to one per poem.**
   Every take still on disk already passed your ears. If two or three takes
   of the same poem survived, keep all of them — they're different audio
   renders of the same style target, which is exactly the kind of variation
   a LoRA benefits from, not redundant noise. (`--max-per-song` is available
   if a handful of heavily-retried poems end up dominating — see Section 6.)
4. **The same poem can reappear across multiple workspace folders** (e.g.
   generated on two different dates, sometimes with a differently-cased
   workspace name — `lamiyat_alshanfara_14082026` vs
   `Lamiyat_Alshanfara_16082026`). These must be grouped as *one song* for
   train/val splitting even though they live in different folders, or you
   risk the same lyrics appearing in both train and val. The script groups
   by normalized title + maqam, not by workspace or filename.
5. **Filename conventions aren't fully consistent** — most are
   `<title>_SONG_<letter>.mp3` or `Retake_<title>_SONG_<letter>.mp3`, but a
   few have no suffix at all (`ليس-الجمال-بمئزر.mp3`). None of this matters
   for grouping, since the poem identity comes from the manifest's
   `original_title` field, not the filename — the filename only needs to
   resolve to a real file.
6. **Unicode normalization mismatches are a real risk with Arabic filenames.**
   The same title can be NFC- or NFD-normalized differently depending on
   what wrote the manifest vs. what wrote the file to disk, which would make
   a naive exact-string match silently drop valid tracks. The script
   normalizes both sides before comparing.
7. Instrumental-mode outputs, if any exist in this corpus, should still be
   split into a separate caption family from vocal tracks — mixing "vocals"
   and "no vocals field" examples under one style tag teaches the model that
   vocals are optional, which isn't the target.

## 2. Audio-side cleanup

1. **Format/rate**: check what the tokenizer expects (likely 44.1/48kHz,
   mono or stereo WAV) — resample everything to one consistent target rather
   than trusting Suno's mixed MP3 bitrates and sample rates.
2. **Loudness normalize** (e.g. EBU R128 / `-14 LUFS` via `pyloudnorm` or
   `ffmpeg-normalize`) so the LoRA doesn't learn "loud" as part of the style.
3. **Trim leading/trailing silence** and cut out any Suno intro/outro
   artifacts (fade glitches, truncated final words) — these are exactly the
   kind of defect that a 200-300 track LoRA will amplify if left in.
4. **Length check**: if YuE2's context window caps clip length, decide now
   whether long tracks get trimmed to the strongest ~60-90s section (verse +
   chorus) or split into multiple training clips with matching caption
   sub-segments — don't just truncate blindly, since that would cut a verse
   mid-word.
5. **Reject list**: build a manual or automatic (silence-ratio, clipping,
   duration-outlier) QC pass and log rejected `clip_id`s with a reason. Don't
   just delete — keep the reasons, since you'll want to check the filter
   isn't too aggressive once you see how many tracks survive.

## 3. Caption-side prep

This is the part your `maqam_prompt_generator.py` already half-solves — reuse
it rather than re-deriving the template.

1. **Regenerate captions from the fixed template**, not from Suno's stored
   `styles` string as-is. Your manifest's stored strings already match the
   generator's `build_prompt()` output (compare `GENRE_STANDARD`,
   `PRODUCTION_STANDARD`, `INSTRUMENTATION`, `EXCLUDE` against the manifest —
   they're identical), so this is mostly a consistency check: catch any track
   whose stored caption drifted from the current template (typos, an older
   wording) and re-render it via `build_prompt(maqam_name, start_phrase, mood)`
   so every example in the dataset has an identical scaffold.
2. **Keep the `[Is_MAX_MODE...]` / `[START_ON: ...]` header or strip it** —
   decide once, consistently. That header is a Suno-specific control token;
   if YuE2 doesn't recognize it, it's just literal text the model will try to
   associate with your style, which is harmless but wasted signal. If
   ai-toolkit's caption normalization doesn't strip bracketed control tokens,
   consider stripping it yourself for the *caption* file while keeping the
   raw original in your manifest for reference.
3. **Decide what to do with `mood`.** It's the one field that varies per
   track ("solemn, severe, authoritative" vs "menacing, tense, dark" vs
   "melancholic, longing, nostalgic"). For LoRA training this is useful
   *if* you want the resulting model to respond to mood words — keep it in.
   If you want the LoRA to only ever produce this one "epic hymn" character
   regardless of mood, drop the field entirely for training captions (you can
   still keep it in your manifest as metadata).
4. **Fold in the maqam name explicitly** if it isn't already prominent enough
   in the caption for the model to key off it — you only have 4 maqams
   (Hijaz, Nahawand, Ajam, Kurd), so this is a natural sub-style axis worth
   the model actually learning, rather than mood, which is unbounded text.
5. **Do NOT reintroduce free-text LLM-generated mood variation.** Your own
   generator's docstring says an earlier attempt to have an LLM vary mood
   introduced unwanted drift — same risk applies to captioning for training:
   keep the caption vocabulary closed and template-driven, not freshly
   generated per track.

## 4. Optional: symbolic score ("sage sheet") extraction

The YuE2 repo ships `SheetSage2` for deriving a symbolic score from audio.
Since you don't have source MIDI/scores (these were Suno renders), you'd
need to derive scores after the fact:

1. Run `SheetSage2` (or the toolkit's built-in score loader, if it accepts
   audio-derived input) on a representative subset first — not the whole
   corpus — and manually check output quality on 5-10 tracks before
   committing to running it on all ~250.
2. If quality is poor on the melismatic Arabic vocal lines (a real risk —
   symbolic transcription tools are usually tuned for Western tonal/rhythmic
   conventions, and your `maqam_prompt_generator.py` already excludes
   quarter-tone maqams for exactly this kind of Western-mapping problem),
   it's reasonable to skip scores entirely and rely on the 50%-dropout
   training path working caption-only.
3. If you do include scores, keep the same per-clip pairing convention
   ai-toolkit expects (see Section 5) so score files line up 1:1 with audio.

## 5. Directory layout

Match ai-toolkit's expected per-example pairing (audio + caption, optionally
+ score) with a flat, predictable structure:

```
dataset/
├── train/
│   ├── 0001_hijaz_01-الديار.wav
│   ├── 0001_hijaz_01-الديار.txt        # caption
│   ├── 0001_hijaz_01-الديار.score.json # optional symbolic score
│   ├── 0002_kurd_02-سم-الشعراء.wav
│   ├── 0002_kurd_02-سم-الشعراء.txt
│   └── ...
├── val/
│   └── ...  (same layout, held-out tracks)
└── manifest.csv   # your own bookkeeping, not consumed by the trainer
```

Use ASCII-safe, sequential filenames as the canonical training filenames
(keep the Arabic title as metadata in `manifest.csv`, not as the literal
audio filename) to avoid encoding issues in tooling that doesn't handle
Arabic filenames cleanly.

## 6. Train/val split — do it by poem, not by clip, across workspaces too

Because takes of the same poem share lyrics and caption — and the same poem
can span multiple workspace folders (Section 1.4) — a random per-clip split
risks leaking near-duplicates across train/val, making validation loss
meaningless. Split by normalized `original_title` + maqam, so every surviving
take of a poem (regardless of which workspace or which `SONG_X` it came from)
stays on the same side of the split. `prepare_dataset.py` does this and
stratifies by maqam so all four appear in val at roughly the same rate
(~90/10 by default, tunable via `--val-fraction`).

**On overrepresented poems:** the script prints a "most-retried poems" report
before splitting. If a handful of poems have 3+ surviving takes while most
have 1, those poems get proportionally more training weight. That's usually
fine (more good audio for the style target), but if it looks skewed enough to
worry about, rerun with `--max-per-song 2` (or similar) to cap it — this is a
judgment call best made after looking at the actual printed distribution on
your full corpus, not decided blind.

## 7. Script: build the dataset from your manifests

A single script to: merge manifests → dedupe A/B pairs → normalize captions
via your existing generator → copy/convert audio → split → write the folder
above and a `manifest.csv` for your own tracking. See `prepare_dataset.py`
(companion file). Run it once dry-run (`--dry-run`) to review the plan before
it touches any files.

## 8. Before you commit GPU time

- Spot-check 15-20 finished dataset entries by ear against their `.txt`
  caption — catch any audio/caption mismatch from a bad merge before it
  trains into the LoRA.
- Confirm final corpus duration and count per maqam — if one maqam (e.g.
  Ajam) is heavily underrepresented, expect the LoRA to be weaker on it, and
  decide whether to accept that or generate more source tracks first.
- Keep the reject log from Section 2 — if you rerun after fixing something,
  you don't want to re-review the same 40 rejected clips a second time.
