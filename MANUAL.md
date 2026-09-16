# YuE2 Run Manual — generating with / without a LoRA

All commands run from the repo root:

```bash
cd /content/yue2_lora_finetuning
```

Two things generate audio here:

- **`audition_planner.py`** — generates a song from **style + lyrics** through the
  full ComfyUI YuE2 graph, with the **base model or a planner (CLIP) LoRA**.
  This is the normal "write me a song" path.
- **`render_tokens_nar.py`** — renders an existing **`.semantic.npy` token
  stream** (a dataset sidecar or a probe) through the acoustic model, with or
  without Mothersuperior's NAR companion. No composition/planner involved.

---

## 0. Prerequisites

### Start ComfyUI (needed by `audition_planner.py`, not by `render_tokens_nar.py`)
```bash
setsid nohup python ComfyUI/main.py --listen 127.0.0.1 --port 8188 \
  < /dev/null > /content/logs/comfyui_render.log 2>&1 & disown
# wait until: curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8188/system_stats  -> 200
```
Stop it before training to free VRAM:
```bash
kill $(pgrep -f "main.py --listen 127.0.0.1 --port 8188")
```

### Check the `cfg_scale` fix is present
Generation quality depends on it (`docs/architecture.md` §7). It must print
two hits:
```bash
grep -n cfg_scale ComfyUI/comfy_extras/nodes_yue2.py
```
If missing, a fresh clone's `bootstrap/setup.sh` will apply
`bootstrap/yue2_cfg_scale.patch`; or apply it by hand:
```bash
git -C ComfyUI apply /content/yue2_lora_finetuning/bootstrap/yue2_cfg_scale.patch
```

### List the available LoRAs
```bash
ls ComfyUI/models/loras/*.safetensors
```
`maqam_planner_v1.safetensors` = final (100 steps);
`maqam_planner_v1_000010…000100.safetensors` = intermediate checkpoints.

---

## 1. Generate **without any LoRA** (base model)

```bash
python audition_planner.py --loras base \
  --style-file /content/data/dataset/train/hijaz_0038_01-retake_take01.style.txt \
  --lyrics-file /content/data/dataset/train/hijaz_0038_01-retake_take01.lyrics.txt \
  --cfg-scale 1.2 --max-duration 400 --seed 831001 --out-prefix yue2/base_demo
```
Output: `ComfyUI/output/yue2/base_demo_base_00001.flac`.

`--loras base` is the special token for "no LoRA". This is the same as a stock
YuE2 generation (with our `cfg_scale`/`max_duration`).

---

## 2. Generate **with the planner LoRA**

```bash
python audition_planner.py --loras maqam_planner_v1.safetensors \
  --style-file /content/data/dataset/train/hijaz_0038_01-retake_take01.style.txt \
  --lyrics-file /content/data/dataset/train/hijaz_0038_01-retake_take01.lyrics.txt \
  --cfg-scale 1.2 --max-duration 400 --seed 831001 --out-prefix yue2/lora_demo
```
Output: `ComfyUI/output/yue2/lora_demo_maqam_planner_v1_00001.flac`.

### A/B base vs several checkpoints in one run
Everything runs at the same seed/prompt, so it is a clean comparison:
```bash
python audition_planner.py --loras base \
  maqam_planner_v1_000030.safetensors \
  maqam_planner_v1_000060.safetensors \
  maqam_planner_v1.safetensors \
  --style-file <style.txt> --lyrics-file <lyrics.txt> \
  --cfg-scale 1.2 --max-duration 400 --seed 831001 --out-prefix yue2/ab
```

---

## 3. Build a style/lyrics prompt from a `workspace_manifest.json`

The manifest's `styles`/`lyrics` are raw Suno blocks. Clean them to the training
format (drop the Suno control header and `mood:`, simplify lyric tags):

```bash
python - <<'PY'
import json, re, sys
sys.path.insert(0, ".")
from prepare_dataset import clean_lyrics

d = json.load(open("/content/workspace_manifest.json", encoding="utf-8"))
title_prefix = "04-"                       # <-- change to the track you want
t = next(x for x in d["tracks"] if x["original_title"].startswith(title_prefix))
style = "\n".join(l.strip() for l in t["styles"].splitlines()
                  if re.match(r"^(genre|vocals|production|instrumentation):", l.strip()))
lyrics = clean_lyrics(t["lyrics"], "simplify")
open("/content/roundtrip/mytrack.style.txt", "w", encoding="utf-8").write(style + "\n")
open("/content/roundtrip/mytrack.lyrics.txt", "w", encoding="utf-8").write(lyrics + "\n")
print("wrote /content/roundtrip/mytrack.{style,lyrics}.txt")
PY
```
Then use `--style-file /content/roundtrip/mytrack.style.txt` etc.

---

## 4. Knobs (what to change and why)

| flag | meaning | known-good |
|---|---|---|
| `--loras` | `base`, or one/more `.safetensors` in `models/loras` | `base` for A/B; `maqam_planner_v1.safetensors` otherwise |
| `--cfg-scale` | text-guidance (prompt adherence) | **1.2** (the sibling repo's clean baseline) |
| `--max-duration` | song length cap, seconds | **400** (your ~6-min ceiling) |
| `--seed` | reproducibility | fix it (e.g. `831001`) for A/B; varies per take |
| `--max-abc-tokens` | composition budget | `8192` |
| `--style-file` / `--lyrics-file` | the prompt | any `.style.txt`/`.lyrics.txt` |
| `--out-prefix` | output folder/name prefix | `yue2/<tag>` → `ComfyUI/output/yue2/` |

Internal sampler defaults (same as the ComfyUI graph): ABC `temperature 0.7,
top_p 0.9, top_k 30, rep 1.005`; Music `1.0/0.95/100/1.2`; acoustic KSampler
`steps 32, cfg 1.0, shift 3.0, euler/simple`.

---

## 5. Render from an existing token stream (`.semantic.npy`)

No planner, no server. Useful to hear a tokenizer round-trip or a probe.

```bash
# plain acoustic render
python render_tokens_nar.py \
  /content/data/dataset/train/hijaz_0045_03_take02.semantic.npy \
  /content/out.wav --steps 32 --cfg 1.0

# with Mothersuperior's NAR companion (measurably closer to the original)
hf download Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4 nar_lora_joint_v4.bf16.safetensors --local-dir /tmp/nar
python render_tokens_nar.py \
  /content/data/dataset/train/hijaz_0045_03_take02.semantic.npy \
  /content/out_nar.wav --nar /tmp/nar/nar_lora_joint_v4.bf16.safetensors
```
The NAR companion is **not a plain LoRA** — the script folds its `vae2llm`/
`llm2vae` replacement weights and its rank-32 deltas into ComfyUI's merged NAR
layers. See `docs/known-issues.md` (KI-05) and
`agent_notes/sessions/session-07.md`.

---

## 6. Combining an acoustic LoRA with the planner LoRA

Once a Stage-3 acoustic LoRA exists, stack them: **planner on the CLIP path,
acoustic on the MODEL path**. In the ComfyUI GUI open
`ComfyUI/custom_nodes/ComfyUI-YuE2-Trainer/example_workflows/yue2_generate_with_lora_api.json`
and set:
- node `17` (`YuE2TrainerLoadLoRA`, `clip`) → your planner LoRA
- node `2` (`YuE2TrainerLoadLoRA`, `model`) → your acoustic LoRA
- `cfg_scale = 1.2`, `max_duration = 400` on the two YuE2 nodes
- `strength_model`/`strength_clip` 1.0 (try 0.5–1.5 for acoustic)

To script it, copy the same node graph into an `audition_planner.py` call (add an
acoustic LoRA node on the MODEL path).

---

## 7. Training (brief; full plan in `PLAN.md`)

**`PLAN.md` is the canonical home for training commands.** Don't duplicate them
here — stage bootstrap and tokenize (§4), planner LoRA (§5), and acoustic LoRA
(§6) are kept current in that file.

To extend an existing planner checkpoint, use the Stage-2 command from
`PLAN.md` §5 plus `--existing-lora maqam_planner_v1_000100.safetensors` (with
`--steps` as the run's **total** length).

---

## 8. Troubleshooting

- **`Failed to fetch ...` / server not reachable** → ComfyUI isn't running; do §0.
- **Output ignores `cfg_scale`** → the patch isn't applied; do §0's grep/apply.
- **`lora key not loaded` warnings** → harmless when a planner LoRA is on the
  CLIP path (its MODEL keys have nothing to attach to).
- **VRAM errors** → stop the ComfyUI server before training; one GPU, shared.
- **Slow generation** → ~2–4 min/song at 32 acoustic steps; `--max-duration`
  dominates. Reduce it for quick tests.
- **Outputs** land in `ComfyUI/output/yue2/`; copy what you keep to
  `/content/roundtrip/` or `/content/track4/` before the VM dies, and/or run
  `python backup_to_gcp.py --once --run-name <run-name>` (run artifacts land
  under `gs://akbar-december-2024-backup/YuE2-3B_Finetuning/<run-name>/`).
