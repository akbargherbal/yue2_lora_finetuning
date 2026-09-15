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

## Current state (one paragraph)

Dataset is done and verified (256 tracks, 238 train / 18 val, poem-safe split).
The backend (`speedyrulz/ComfyUI-YuE2-Trainer` + `yue2_3b_bf16.safetensors`) is
installed, the community semantic-token head is wired in, and the whole corpus
is tokenized (256/256). The first **planner LoRA** was trained for 100 steps on
semantic tokens (`maqam_planner_v1.safetensors`, held-out eval 5.60 → 4.95,
−11.7%, still falling; KL crept to 0.25). The first evaluation — base vs
checkpoints 30/60/100 on a user-supplied Maqam Nahawand track — is rendered; the
user's verdict so far is **"so-so, not good not bad"**, with full listening
notes to come.

## Blocked on

KI-21: the user's listening impression of the track-4 A/B. Specifically,
whether any planner checkpoint clearly differs from `base` in composition/maqam,
and which (if any) checkpoint sounds closest to the intended Nahawand. **This
answer picks the next branch** — do not burn a session guessing.

## Next steps (in order)

1. **Collect the user's listening notes**, then branch (`docs/architecture.md` §1
   sets the acceptance bar: base-model pronunciation parity, judged by ear):
   - *LoRA indistinguishable from base* → extend to ≥200–300 steps with
     `--existing-lora maqam_planner_v1_000100.safetensors` (total `--steps`) and
    /or revisit planner hyperparameters (LR, `--abc-dropout`, `--max-tokens`);
     watch KL (KI-03).
   - *Differs but still "meh"* → the limiter is the acoustic half → Stage 3
     acoustic LoRA in `--conditioning inference_like --use-semantic`
     (`PLAN.md` §6); consider the NAR companion (`render_tokens_nar.py`, KI-05).
   - *A checkpoint is good* → lock it; stack planner (CLIP) + acoustic (MODEL)
     and do the final seed-matched A/B.
2. **On a fresh Colab**, run `bootstrap/setup.sh` and confirm the `cfg_scale`
   patch applied — it now pins ComfyUI `36da3ff7` and **hard-fails** if the
   patch won't apply (KI-08), so a silent regression is no longer possible.
3. **Tokenize any new material** with `--semantic-head` before training on it.
4. **Open/low priority:** `status`-field check (KI-01), `--max-per-song`
   (KI-02), corpus-drift re-verify (KI-07).

## Session hygiene

- Anything the user would copy-paste goes in `agent_notes/current.md`
  (gitignored, overwritten each turn), not in this file or chat.
- Sessions never start, stop, or resume training — the user runs every command.
- Check `nvidia-smi` before anything GPU-heavy; one shared GPU.
