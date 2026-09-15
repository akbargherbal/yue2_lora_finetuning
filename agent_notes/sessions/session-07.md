# Session 7 — first planner LoRA, first real evaluation, and the glitch list

First session to run the whole semantic-token → planner pipeline end to end and
produce an actual fine-tuned model plus renders.

## What ran, in order

1. **Bootstrap.** Added a `tokenizer` job to `bootstrap/setup.sh` (stages
   `tokenizer_head_joint_v4.pt`, ~171 MB, into `models/audio_encoders/`) and
   pre-downloaded `m-a-p/MERT-v2-FullSong` to the HF cache (~630 MB). Added a
   head-presence check and an idempotent apply of the `cfg_scale` patch.
2. **`cfg_scale` fix.** Patched `YuE2GenerateMusic` **and** `YuE2GenerateABC`
   to accept a `cfg_scale` input (default `0.0` = model default, so existing
   behavior is unchanged) and forward it through `clip.tokenize()`. Committed as
   `bootstrap/yue2_cfg_scale.patch` because the `ComfyUI/` clone is untracked
   and dies with the VM.
3. **Tokenizer smoke test + round-trip gate** (non-destructive, on copies):
   - 2 real tracks tokenized in 1–4 s each, 2.7 GB peak VRAM; 4046 / 4564 codes,
     in-vocab.
   - Round-trip through the base acoustic stage: onset 0.615 / 0.806, chroma
     0.749 / 0.791 — as good as or better than the trainer README's own
     reference round-trips (chroma 0.60–0.65). **Composition is preserved** →
     the planner path is sound.
   - The "weird noise" the user heard is the acoustic half, not a bug: steps 32
     vs 64 and cfg 1 vs 2 scored within 0.01; no added broadband noise.
4. **NAR companion investigation.** It is not a plain LoRA and applying it to
   the (cloned) module made it look like a no-op. Folded correctly it slightly
   improves round-trip fidelity (chroma +0.015 to +0.025). Tool:
   `render_tokens_nar.py`. See KI-05/KI-14.
5. **Stage 1 — tokenize the corpus.** 256/256 tokenized, 0 failed; VAE cache
   256 files / 199 MB. The command then crashed (KI-01 in the old list; now
   fixed in `PLAN.md` §4 — the dry run needed `--semantic --no-abc`).
6. **Stage 2 — planner LoRA (100 steps).** `--semantic --no-abc
   --abc-dropout 0.5 --kl-weight 0.5`, rank 32, lr 5e-5 cosine. Final LoRA
   `maqam_planner_v1.safetensors`, 336 tensors; 11 checkpoints
   (`_000010`…`_000100`) + `.resume`; probes `step_000000`…`step_000100`.
   - **Held-out eval 5.6047 → 4.9468 (−11.7%)**, still falling at step 100.
   - **All 11 probes ended normally** (1648–2611 ABC tokens vs an 8192 cap) → no
     over-training signal by that measure.
   - **KL 0 → 0.041 → 0.178 → 0.217 → 0.253** and still creeping (KI-03).
7. **Track-4 evaluation.** User supplied `/content/workspace_manifest.json`;
   track `04-وصف-محاسن-الحبيبة-والجمال` (Maqam Nahawand). Built a
   training-format prompt (style stripped to
   `genre/vocals/production/instrumentation`; lyrics via
   `clean_lyrics(..., "simplify")`), rendered base + checkpoints 30/60/100 at
   seed 831001, `cfg_scale=1.2`, `max_duration=400` (`audition_planner.py`).
   All four land ~D minor by key detection, so it is an ear call. **User's
   verdict so far: "so-so, not good not bad"; full impression next session**
   (KI-21).
8. **GCS.** `backup_to_gcp.py --once` (loras + probes + logs + notes;
   2.74 GiB) plus hand-uploaded `track4/`, `roundtrip/`, `prompts/`, `repo/`.

## Glitches

All actionable items from this session are catalogued in
`docs/known-issues.md` (KI-08, KI-13, KI-15 through KI-24). Fix those before
repeating the pipeline. The most important: the `cfg_scale` patch is tied to
ComfyUI `36da3ff7`, so the bootstrap now pins that commit and hard-fails if the
patch won't apply.

## Artifacts (durable copies in GCS)

`gs://akbar-december-2024-backup/YuE2-3B_13092026/run_backup/`:
`loras/` (all planner checkpoints + final + `.resume` + `loss.json` + probes),
`track4/` (the base-vs-LoRA A/B), `roundtrip/` (tokenizer round-trip + NAR
A/B), `prompts/`, `repo/` (helper scripts, patch, PLAN), `logs/`,
`agent_notes/`.

## What the next session needs from the user

Their listening impression of `track4_base` vs `track4_maqam_planner_v1` (and
checkpoints 30/60) — specifically whether the LoRA changes composition at all,
and which checkpoint (if any) sounds closest to the intended Nahawand. That
answer picks: train the planner more, treat the acoustic half as the limiter
(Stage 3), or lock the checkpoint and stack.
