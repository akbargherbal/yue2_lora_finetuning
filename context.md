# context.md — HANDOFF

Current state of the YuE2-3B maqam LoRA project. This file is the only one a
fresh session needs to read in full; everything durable or historical lives
elsewhere:

- **Design reference:** [`docs/architecture.md`](docs/architecture.md) — goal,
  the two LoRAs, corpus/dataset, tooling decisions, run operations, repo map.
- **Open problems:** [`docs/known-issues.md`](docs/known-issues.md) — flat,
  deduplicated, with statuses.
- **Training plan + commands:** [`PLAN.md`](PLAN.md) (canonical for training).
- **Generation runbook + commands:** [`MANUAL.md`](MANUAL.md) (canonical for
  generation).
- **History:** [`CHANGELOG.md`](CHANGELOG.md) →
  [`agent_notes/sessions/`](agent_notes/sessions/).

> **Fresh session, or arrived after a VM/GPU switch?** Local disk (`/content`,
> `/workspace`) is wiped; only GitHub + GCS persist. Read this file, then
> `PLAN.md` §2.4 for restore commands (re-clone, re-apply
> `bootstrap/joint_minted_optional.patch`, rebuild corpus/venv/models, restore
> prep + checkpoints from GCS). GCS root:
> `gs://akbar-december-2024-backup/YuE2-3B_Finetuning/`; this run is
> `maqamverse_calib_v1/`. `agent_notes/current.md` is per-turn scratch — this
> file and `PLAN.md` are the durable state.

## Current state (one paragraph)

Dataset is done and verified (256 tracks, 238 train / 18 val, poem-safe split).
The planner LoRA (`maqam_planner_v1`, 100 steps) was audited by ear: base vs
checkpoints 30/60/100 on a Maqam Nahawand track (**resolves KI-21**).
Pronunciation held up, but **30 sounded better than 60/100** (KL climbing,
KI-03) and even at 30 the لحن reads "foreign" — diagnosis: the community
semantic head (`tokenizer_head_joint_v4.pt`, KI-04) was never adapted to this
corpus. This session set out to run Mothersuperior's `joint.py` calibration and
hit a wall: `joint.py` hard-requires *minted* artifacts (`sem_nbr_*.npy`,
`feats/*.npy`, the full minted corpus) that are **not released**, so it can't
run as shipped. Worked around by patching the local copy
(`/content/ms_calib/scripts/joint.py`, plus `export_mothersuperior_format.py`
in-repo) to make the minted regularizer optional (`MINTED=0`). Everything else
is staged: the corpus is converted to Mothersuperior's flat format
(`/content/ms_calib/corpus`, 256 tracks, 13 GB), `ms_calib_venv` exists (py3.12,
torch 2.10+cu128, `yue2-infer@92a73cc7`), the three HF models are under
`/workspace/hf`, the hard-coded `/workspace` layout is symlinked onto
`/content`, and the head/NAR weights are downloaded. **Not launched yet** —
`PLAN.md` (rewritten this session) holds the exact commands; `prep_real.py`
then `joint.py` are next.

## Blocked on

Nothing external. Two caveats: with the minted regularizer off there is no
anti-collapse term, so `real_repeat` is the stop signal (`PLAN.md` §6); and
whether a calibrated head is loadable by this repo's `--semantic-head` flag is
still unverified (Mothersuperior ships `.safetensors` and a ComfyUI export, so
it's likely tractable) — check after the run.

## Next steps (in order)

1. **Launch the calibration** exactly as `PLAN.md` §5.1 → §5.3: `prep_real.py`
   (GPU, MERT+VAE over 256 tracks), then `joint.py` with `MINTED=0
   HOLD_TRACK=nahawand_0111_04_take01`. The user runs both.
2. **Ear-check** `joint.py`'s held-out render
   (`/workspace/tok/full/listen_real/real_pred_maqamverse_calib_v1.flac`)
   against the stock-head round-trip on the same track/seed, and watch
   `real_repeat` for collapse during training.
3. **Branch on that listening result** (`PLAN.md` §7):
   - *Calibrated round-trip sounds closer, pronunciation intact* → re-tokenize
     the full corpus with the new head and retrain the planner LoRA **fresh**
     (tokens changed underneath it).
   - *No audible difference* → tokenizer wasn't the bottleneck; try Path 1
     (SheetSage2 melody→ABC, `t8star/YuE2-Comfy`, `architecture.md` §5).
   - *Calibration fails to load/run* → own session; log in
     `docs/known-issues.md`.
4. **Backup policy** (part of `AGENTS.md`): the daemon is **not running** —
   start it per `PLAN.md` §2.2 before the long run; `TARGETS` now includes the
   calibration output `head_calib`. Every invocation requires `--run-name`;
   artifacts mirror to `.../YuE2-3B_Finetuning/<run-name>/` with a
   `run_manifest.json`. Session 6-7 lives at
   `.../YuE2-3B_Finetuning/maqam_planner_v1/`.
5. **Open/low priority, unchanged:** `status`-field check (KI-01),
   `--max-per-song` (KI-02), corpus-drift re-verify (KI-07).

## Session hygiene

- Anything the user would copy-paste goes in `agent_notes/current.md`
  (gitignored, overwritten each turn), not in this file or chat.
- Sessions never start, stop, or resume training — the user runs every command.
- Check `nvidia-smi` before anything GPU-heavy; one shared GPU.
