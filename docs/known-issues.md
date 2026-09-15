# Known issues

Flat, deduplicated list of current problems. Status is one of **open**,
**fixed**, or **won't-fix / accepted**. Add new entries at the bottom; update
statuses in place rather than adding a new narrative (that is what
`agent_notes/sessions/` is for).

| id | issue | status | notes |
|---|---|---|---|
| KI-01 | `status` field in `workspace_manifest.json` never inspected | open | Full key list confirmed (`clip_id, original_title, assigned_filename, styles, exclude_styles, lyrics, created_at, status`). File-existence filtering works, but `status` might be a sturdier filter. Quick check: count `t.get('status')` across the corpus manifests (needs the raw corpus, not just the built dataset). |
| KI-02 | `--max-per-song` capping undecided | open | Leaning no-cap; retry skew is mild (top poems 4–6 surviving takes, most 1–2). |
| KI-03 | Planner KL ran hot | open | Session 7: 0 → 0.041 → 0.178 → 0.217 → 0.253 and still creeping, vs the plan's "level off at a few hundredths" guide. Either raise `--kl-weight`, stop earlier, or accept if the ear likes an early checkpoint. |
| KI-04 | Community tokenizer is approximate | won't-fix (accepted) | ~16% exact top-1 on YuE2's own songs, ~95% by ear on round-trips. The planner learns from near-miss codes; this is the community ceiling, not a bug. |
| KI-05 | Two sources disagree on the NAR companion LoRA | open | Nora ships it; speedyrulz measured it makes renders *less* similar. Session 7 found it cannot be loaded as a plain LoRA (it also carries `vae2llm`/`llm2vae` replacement weights) and, folded in correctly, slightly *improves* round-trip fidelity (chroma +0.015–0.025). A/B if Stage 3 underdelivers. |
| KI-06 | No community validation on Arabic/maqam | accepted risk | Nobody has proven this pipeline on melismatic Arabic. We are early adopters; the Stage-1 tokenizer round-trip smoke test is the guard. |
| KI-07 | Corpus can drift between sessions | open | Re-run `verify_dataset.py --dataset-root ...` (drift check) before trusting dataset counts. |
| KI-08 | `cfg_scale` patch tied to a ComfyUI commit | fixed | `bootstrap/yue2_cfg_scale.patch` applies against ComfyUI `36da3ff7`; `setup.sh` now pins that commit and hard-fails if the patch won't apply, instead of silently warning. |
| KI-09 | Same run command duplicated across three docs | fixed | `PLAN.md` is canonical for training, `MANUAL.md` for generation; other docs link. |
| KI-10 | No committed tests despite real testing discipline | fixed | `tests/` now holds pytest fixtures and tests for `prepare_dataset.py` / `verify_dataset.py`. |
| KI-11 | No dependency pinning | fixed | `requirements.txt` pins the repo's own deps; `setup.sh` pins both external repos by commit SHA. |
| KI-12 | `audition_planner.py` poll loop had no timeout/retry | fixed | Added a retry/timeout HTTP helper and a `--timeout` flag that fails loudly instead of spinning forever. |
| KI-13 | Planner probes were ABC-only | open (config) | `--probe-music-seconds` defaults to 0, so no per-checkpoint `.semantic.npy` was written. Set `--probe-music-seconds`/`--probe-render` if per-checkpoint audio is wanted; otherwise audition via `audition_planner.py`. |
| KI-14 | ComfyUI `ModelPatcher` deep-clones on load | fixed (documented) | Editing `model.model.diffusion_model` before `Renderer.load()` does not reach the clone. Correct order: call `renderer.load()`, then edit `renderer.patcher.model.diffusion_model`. |
| KI-15 | ComfyUI clone race between parallel bootstrap jobs | fixed | Any parallel job writing into `ComfyUI/` can be wiped by the clone's `rm -rf`. All assets stage under `/content/staging` and are moved in after the clone. |
| KI-16 | Starting ComfyUI from an agent shell hangs the tool | fixed (documented) | Use `setsid nohup ... < /dev/null > log 2>&1 & disown`. Stop the server before training (it holds ~6–7.5 GB VRAM on the shared GPU). |
| KI-17 | MERT must load via the trainer's `load_mert` | fixed (documented) | Plain `AutoModel.from_pretrained` on transformers 5 leaves the rotary `inv_freq` buffer uninitialised → silently wrong/NaN features. |
| KI-18 | `--dry-run` can leave untrained artifacts | fixed (documented) | A 0-step `maqam_acoustic_v1.safetensors` was saved by a dry run. Don't treat a dry-run artifact as a model; the real run overwrites it. |
| KI-19 | Helper scripts uncommitted | fixed | `audition_planner.py` and `render_tokens_nar.py` are committed. |
| KI-20 | GCS backup is incomplete by design | accepted | `backup_to_gcp.py` covers only `loras/`, `logs/`, `agent_notes/`. Renders/prompts/scripts are uploaded by hand; the VAE cache is excluded unless `--include-cache`. |
| KI-21 | User verdict on the track-4 A/B still pending | blocked on user | Does any planner checkpoint (30/60/100) clearly differ from `base` on composition/maqam, and which is best? This answer picks the next branch (extend planner vs Stage 3 acoustic vs lock & stack). |
| KI-22 | `cfg_scale` has no effect on `generate_abc()`'s helper layer | closed | The node path does accept it (the patch wires `YuE2GenerateABC` too); only the trainer's standalone `generate_abc()` helper has no hook. |
| KI-23 | Three sources of truth for the ComfyUI node graph | open | `audition_planner.py` builds the graph inline; the GUI workflow and `render_tokens_nar.py` encode the same topology differently. Longer-term: load an exported `*_api.json` and patch only LoRA/`cfg_scale`/seed. |
| KI-24 | `agent_notes/current.md` was tracked scratch state | fixed | Now `.gitignore`d and untracked; it still lives on disk and is mirrored to GCS. |
