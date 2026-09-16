# samples/ — reference prompt pairs for planner auditions

Two kinds of `(style.txt, lyrics.txt)` pairs for `audition_planner.py`:

## `unseen/` — genuinely held-out (use these for style-transfer tests)

The trainer's leak-free holdout, selected by `--eval-holdout 5 --seed 2002` and
confirmed excluded in the run log (`held out for evaluation (not trained on):
…`):

- `hijaz_0038_01-retake_take01`
- `hijaz_0058_track_take01`
- `nahawand_0095_01_take01`
- `nahawand_0114_06_take01`
- `nahawand_0115_06_take01`

They live under `dataset/train/` but are excluded from training by the holdout
seed, so base-vs-checkpoint renders on these are a real transfer test. Copied
verbatim from `/content/data/dataset/train/`.

## `reference/` — historical audition track (NOT unseen)

`04-وصف-محاسن-الحبيبة-والجمال_SONG_A` — the track used for the session-7/8
base-vs-30/60/100 A/B (its cleaned prompt is the session's `track4.*` pair).
Raw source: `/content/workspace_manifest.json`, clip_id
`e34c52cb-5939-4eee-bb92-89e114f215c3`, `status: skipped_existing` (a sibling
`_SONG_B` exists, clip `471e079f-…`).

**Caveat — this is not unforeseen data as a prompt.** The *track* is absent from
the built dataset, but its **lyrics are the same poem as the training track
`nahawand_0111_04_take01`** (27/30 lines identical; 3 minor diacritic variants)
and its **style caption is the same template** (only "deep male vocals" →
"Male deep baritone"). A good result here can be recall, not transfer. Use it
only for continuity with the earlier A/B — prefer `unseen/` for real tests.

## Using them

With `audition_planner.py` (`MANUAL.md`): hold the seed, `cfg_scale`, and style
caption fixed, and compare `base` against each checkpoint. Keep the earlier
lesson: on track4, **step 30 beat 60/100**.
