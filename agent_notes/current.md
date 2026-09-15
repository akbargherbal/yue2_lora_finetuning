# GCS backup script — `backup_to_gcp.py`

## What it does
Run it in a **second terminal** alongside training. Every `--interval-minutes`
(default **25**) it mirrors the recovery-critical folders to
`gs://akbar-december-2024-backup/YuE2-3B_13092026/run_backup/` with
`gsutil -m rsync -r`:

| local | remote | why |
|---|---|---|
| `ComfyUI/models/loras/` | `run_backup/loras/` | `.safetensors`, `.resume`, `.loss.json` — resume + debug |
| `/content/logs/` | `run_backup/logs/` | `train.log` — crash diagnosis |
| `agent_notes/` | `run_backup/agent_notes/` | the plan + exact commands |

Add `--include-cache` to also mirror the VAE latent cache (large, regenerable,
but it saves the ~20 min re-encode on a fresh machine). The dataset is already in
GCS at `.../YuE2-3B_13092026/dataset/` and the 7.4 GB base checkpoint is on HF,
so neither is mirrored.

## Run it
```bash
cd /content/yue2_lora_finetuning
python backup_to_gcp.py                 # every 25 min, first pass immediately
python backup_to_gcp.py --interval-minutes 20
python backup_to_gcp.py --include-cache
python backup_to_gcp.py --once          # single pass (cron)
python backup_to_gcp.py --dry-run       # print commands, upload nothing
```
Logs to `/content/logs/gcp_backup.log` (also mirrored). Keep it running in its
own terminal; Ctrl-C just stops it.

## Safety properties (read before trusting it)
- **Never deletes on the remote side** — no `-d`, so rsync is append/update only.
- **Settle guard** — the `loras/` sync waits until the newest file there has been
  untouched for `--settle-seconds` (default 60) so it doesn't grab a `.safetensors`
  mid-write. If a torn upload still happens, the next pass re-uploads once the
  local file stops changing, so it self-heals.
- **Read-only on local files** — it opens nothing for writing; the dataset and
  checkpoint are untouched.
- Needs `gcloud`/`gsutil` auth to stay valid. Currently active as
  `ghurbal.akbar@gmail.com`; a Colab token expiry would make passes fail (logged,
  loop keeps going).

## Verification done
- `py_compile` passes.
- Raw `gsutil` tested: a repeated `-x` flag only honours the **last** pattern, so
  the script combines all excludes into one alternation.
- **Real end-to-end pass executed** (`--once`, not dry-run): all 3 folders synced
  and `gsutil ls -r .../run_backup/` shows `logs/*` and `agent_notes/current.md`
  on GCS. (The empty `loras/put_loras_here` marker got up in that first pass
  before the exclude fix; harmless, and rsync won't delete it.)

## Still the training command (with the checkpoint cadence fix)
```bash
python ComfyUI/custom_nodes/ComfyUI-YuE2-Trainer/train_cli.py acoustic \
  --comfy-root /content/yue2_lora_finetuning/ComfyUI \
  --checkpoint yue2_3b_bf16.safetensors \
  --data /content/data/dataset \
  --eval-holdout 5 --seed 2002 \
  --segment-seconds 30 --steps 300 \
  --save-every 50 \
  --out maqam_acoustic_v1 \
  2>&1 | tee -a /content/logs/train.log
```
Resume from a partial: add `--existing-lora maqam_acoustic_v1_000150.safetensors`
(same `--out`, same rank/alpha/targets/optimizer).
