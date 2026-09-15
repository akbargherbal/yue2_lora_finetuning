# Repo Quality Review — `yue2_lora_finetuning`

> **Archived 2026-09-15 — implemented.** This review drove the repo-quality
> pass in commit `8a5974c`. Kept for provenance only; see `CHANGELOG.md` for
> what changed. Do not treat it as current documentation.

Reviewed: `AGENTS.md`, `context.md`, `PLAN.md`, `MANUAL.md`, `agent_notes/current.md`,
`bootstrap/setup.sh`, and all six Python scripts (`prepare_dataset.py`,
`verify_dataset.py`, `maqam_prompt_generator.py`, `audition_planner.py`,
`render_tokens_nar.py`, `backup_to_gcp.py`).

**Overall take:** the actual engineering (dataset builder, verifier, prompt
generator, Colab bootstrap) is careful and well-reasoned — good docstrings,
real edge-case handling (Unicode normalization, filename collisions,
train/val leakage), and evidence of deliberate testing against synthetic
fixtures. The "mediocre" feeling is coming from **process artifacts around
that code**, not the code itself: an unbounded journal file standing in for
real docs, zero committed tests despite clear testing discipline, no
dependency pinning, and a couple of genuinely fragile scripts. Below is what
I'd fix, roughly in priority order.

## 1. `context.md` has outgrown its format (highest impact)

It's 1,134 lines and growing every session, mixing four different things in
one append-only file:
- stable reference material (corpus layout, tool decisions)
- a chronological session log ("Session 3 lesson...", "Session 7...")
- a running "known issues" list (§17's 13 glitches)
- next-step instructions for whoever picks this up next

The file's own header admits the failure mode this causes: two contradictory
claims survived multiple rewrites because nobody re-verified them against a
primary source. That's a direct symptom of narrative-log-as-database — old
conclusions get *softened* ("in doubt") instead of *corrected*, because
finding the one true current statement requires reading the whole history.

**Fix:** split into durable vs. ephemeral:
- `docs/architecture.md` — the parts that don't change per-session: what the
  two LoRAs control (§1 of `PLAN.md` already has this table), corpus layout,
  tool/backend decision and why alternatives were ruled out.
- `docs/known-issues.md` — a flat, deduplicated list of the current 13
  glitches, each with a status (open/fixed/won't-fix), not a growing prose
  list re-explaining context every time.
- `CHANGELOG.md` or `agent_notes/sessions/session-NN.md` — one file per
  session, append-only there instead of in the main doc. Keeps history
  without forcing every future read to wade through it.
- `context.md` (or rename to `HANDOFF.md`) shrinks to: current state in one
  paragraph, what's blocked, what to do next, links to the above. That's the
  only part a fresh session actually needs to read in full.

## 2. No committed tests, despite doing the work of writing them

`context.md` §6 describes real test discipline — hand-built fixtures for
`prepare_dataset.py`/`verify_dataset.py` reproducing the manifest's exact
field shape, plus deliberately corrupted builds to confirm the checker
catches each failure mode. None of that exists in the repo; it's redone
from scratch (or lost) each session. That's the most valuable thing to
formalize, since it's already designed, just not saved:

```
tests/
  fixtures/
    manifest_clean.json
    manifest_naming_collision.json   # pre-fix collision case
    manifest_missing_sidecar.json
    manifest_leaked_marker.json
  test_prepare_dataset.py    # clean_lyrics(), ascii_safe_slug(), split_train_val()
  test_verify_dataset.py     # each corrupted fixture trips the right check
```

`clean_lyrics()`, `extract_caption_maqam()`, `ascii_safe_slug()`, and
`split_train_val()`'s leakage guarantee are all pure functions with no I/O —
cheap to pin down with `pytest` and immediately valuable, since a regression
here silently corrupts training data rather than crashing loudly.

## 3. No dependency manifest anywhere

Nothing pins `pip install` versions for the trainer, ComfyUI, or the repo's
own scripts (`urllib`/`json`/`csv` are stdlib, but nothing records the
Python version, or that e.g. `numpy` is needed for `render_tokens_nar.py`).
Combined with cloning `ComfyUI` and `ComfyUI-YuE2-Trainer` at whatever HEAD
happens to be current, this makes runs non-reproducible — and the project
has already been bitten by this once: the `cfg_scale` patch is explicitly
tied to ComfyUI commit `36da3ff7`, and `setup.sh` can only *warn*, not fail,
if a newer upstream breaks it silently.

**Fix:** add a `requirements.txt` for the repo's own scripts, and pin the
two external repos to a commit/tag in `setup.sh` (`git checkout <sha>`
after clone) rather than always tracking HEAD. That turns "renders silently
lose cfg_scale" into a loud, early failure instead of a quality regression
someone has to notice by ear.

## 4. `audition_planner.py` — fragile in ways that will bite mid-run

- The poll loop (`while True: time.sleep(5); ...`) has no timeout and no
  `try/except` around the `urllib` calls — a dropped connection or a hung
  ComfyUI job hangs the script forever with no way to tell the two apart.
- The workflow graph is a hand-written dict of numbered node IDs (`"1"`
  through `"17"`) duplicated between this file and (differently)
  `render_tokens_nar.py`, and again in the ComfyUI GUI workflow referenced
  in `MANUAL.md` §6. Three sources of truth for the same graph topology
  means a future ComfyUI node-signature change has to be caught and fixed
  in three places by hand.

**Fix:** wrap the HTTP calls in a small retry/timeout helper, and add a
`--timeout` flag that fails loudly instead of spinning. Longer-term, define
the graph once (e.g. load the exported `*_api.json` workflow already used
in the GUI path and patch just the LoRA/cfg_scale/seed fields) so there's a
single source of truth for the node graph instead of three.

## 5. Documentation has three overlapping "how to run training" sections

The Stage-2 planner command appears near-verbatim in `context.md` §17,
`PLAN.md` §5, and `MANUAL.md` §7. Same for the `cfg_scale` check. This is
exactly the kind of duplication that caused the two stale claims `context.md`
itself flags as a lesson-learned — any future flag change (e.g. `--rank` or
`--lr`) needs three edits to stay consistent, and it's easy to update one and
miss the others.

**Fix:** one canonical location per command (`PLAN.md` is the natural home
for training commands, `MANUAL.md` for generation commands), and have the
others link to it instead of re-pasting it.

## 6. Minor items

- `agent_notes/current.md` is described in `AGENTS.md` as "rewritten each
  turn, not a stable reference" but it's tracked in git — so its history is
  a pile of superseded scratch notes with no long-term value, cluttering
  `git log`. Consider `.gitignore`-ing it, or moving it to a `_scratch/`
  dir excluded from version control, and keeping only the durable output
  (commands, decisions) in tracked docs.
- No linting/formatting config (`ruff`/`black`) — the code style is already
  consistent by hand, but nothing enforces it as more scripts get added.
- No `LICENSE`-adjacent note that the pipeline depends on two CC BY-NC 4.0
  (non-commercial) models — worth a one-line callout at the top of `README`
  or `PLAN.md` rather than only mentioned in passing inside `context.md`,
  since it constrains what can ever be done with the trained LoRA.

## Suggested priority order

1. Split `context.md` (§1) — this is actively causing the stale-claim bugs
   the file itself documents.
2. Commit the existing test fixtures/logic (§2) — low effort, since the
   testing was already designed once per session.
3. Pin dependencies/commits (§3) — prevents the exact kind of silent
   regression already seen with `cfg_scale`.
4. Harden `audition_planner.py`'s network loop (§4).
5. De-duplicate the run commands across docs (§5) when convenient.
