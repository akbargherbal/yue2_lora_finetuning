"""Audition planner-LoRA checkpoints through ComfyUI's full YuE2 generation path
(planner LoRA on the CLIP path, cfg_scale forwarded, base acoustic model).

Start ComfyUI first, then:
  python audition_planner.py --loras maqam_planner_v1_000030.safetensors maqam_planner_v1_000100.safetensors

HTTP calls are retried with a per-request timeout, and the whole job wait is
bounded by --timeout, so a dropped connection and a hung ComfyUI job fail
loudly (and distinguishably) instead of spinning forever.

NOTE: the node graph below is one of three encodings of the same topology (this
script, the GUI `*_api.json` workflow, and `render_tokens_nar.py`). See
docs/known-issues.md KI-23 for the plan to collapse them to a single source.
"""
import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

DEFAULT_SERVER = "http://127.0.0.1:8188"
HELDOUT = "/content/data/dataset/train/hijaz_0038_01-retake_take01"


class ServerError(RuntimeError):
    """ComfyUI could not be reached, or a request failed after retries."""


def call(server: str, path: str, payload=None, *, timeout: float = 30.0,
         retries: int = 3, backoff: float = 2.0):
    """GET/POST `path`, retrying transient network errors with backoff."""
    data = json.dumps(payload).encode() if payload is not None else None
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            server + path, data=data, headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError,
                json.JSONDecodeError) as exc:
            last = exc
            if attempt < retries:
                wait = backoff * attempt
                print(f"  [retry {attempt}/{retries}] {path}: "
                      f"{type(exc).__name__}: {exc} (waiting {wait:.0f}s)", flush=True)
                time.sleep(wait)
    raise ServerError(f"{path} failed after {retries} attempts: {last}") from last


def workflow(lora, style, lyrics, prefix, seed, cfg, max_duration, max_abc):
    clip = ["1", 1]
    nodes = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "yue2_3b_bf16.safetensors"}},
        "3": {"class_type": "YuE2GenerateABC", "inputs": {
            "clip": clip, "style": style, "lyrics": lyrics, "seed": seed, "mode": "full",
            "max_abc_tokens": max_abc, "temperature": 0.7, "top_p": 0.9, "top_k": 30,
            "repetition_penalty": 1.005, "penalty_window": 100, "cfg_scale": cfg}},
        "10": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.0}},
        "11": {"class_type": "YuE2GenerateMusic", "inputs": {
            "clip": clip, "style": style, "lyrics": lyrics, "abc": ["3", 0], "seed": seed, "mode": "full",
            "max_duration": max_duration, "temperature": 1.0, "top_p": 0.95, "top_k": 100,
            "repetition_penalty": 1.2, "cfg_scale": cfg}},
        "12": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["11", 0]}},
        "13": {"class_type": "EmptyYuE2LatentAudio", "inputs": {"seconds": ["11", 1], "batch_size": 1}},
        "14": {"class_type": "KSampler", "inputs": {
            "model": ["10", 0], "positive": ["11", 0], "negative": ["12", 0], "latent_image": ["13", 0],
            "seed": seed, "steps": 32, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "15": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["1", 2]}},
        "16": {"class_type": "SaveAudio", "inputs": {"audio": ["15", 0], "filename_prefix": prefix}},
    }
    if lora and lora != "base":
        nodes["17"] = {"class_type": "YuE2TrainerLoadLoRA", "inputs": {
            "lora_name": lora, "strength_model": 1.0, "strength_clip": 1.0, "clip": ["1", 1]}}
        nodes["3"]["inputs"]["clip"] = ["17", 1]
        nodes["11"]["inputs"]["clip"] = ["17", 1]
    return nodes


def wait_for_job(server, pid, lora, *, timeout, poll_interval, http_timeout, retries):
    """Poll /history until the job appears; fail loudly on timeout or error."""
    deadline = time.monotonic() + timeout if timeout and timeout > 0 else None
    while True:
        time.sleep(poll_interval)
        if deadline is not None and time.monotonic() > deadline:
            raise ServerError(
                f"{lora}: no result after {timeout:.0f}s (job {pid} still queued/running). "
                f"Is ComfyUI healthy? Check /content/logs/comfyui_render.log."
            )
        history = call(server, f"/history/{pid}", timeout=http_timeout, retries=retries)
        if pid not in history:
            continue
        status = history[pid].get("status", {})
        outputs = [i["filename"]
                   for out in history[pid].get("outputs", {}).values()
                   for i in out.get("audio", [])]
        print(f"{lora}: {status.get('status_str')} -> {outputs}", flush=True)
        if status.get("status_str") == "error":
            for message in status.get("messages", []):
                print("  ", message, flush=True)
            return False
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loras", nargs="+", required=True)
    ap.add_argument("--style-file", default=HELDOUT + ".style.txt")
    ap.add_argument("--lyrics-file", default=HELDOUT + ".lyrics.txt")
    ap.add_argument("--seed", type=int, default=831001)
    ap.add_argument("--cfg-scale", type=float, default=1.2)
    ap.add_argument("--max-duration", type=float, default=400.0)
    ap.add_argument("--max-abc-tokens", type=int, default=8192)
    ap.add_argument("--out-prefix", default="yue2/audition")
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="Seconds to wait for each queued job before failing (0 = wait forever).")
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument("--http-timeout", type=float, default=30.0,
                    help="Per-request socket timeout in seconds.")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    style = Path(args.style_file).read_text(encoding="utf-8")
    lyrics = Path(args.lyrics_file).read_text(encoding="utf-8")

    failed = []
    for lora in args.loras:
        tag = lora.replace(".safetensors", "")
        prompt = workflow(lora, style, lyrics, f"{args.out_prefix}_{tag}", args.seed, args.cfg_scale,
                          args.max_duration, args.max_abc_tokens)
        try:
            pid = call(args.server, "/prompt",
                       {"prompt": prompt, "client_id": str(uuid.uuid4())},
                       timeout=args.http_timeout, retries=args.retries)["prompt_id"]
            print(f"{lora}: queued {pid}", flush=True)
            ok = wait_for_job(args.server, pid, lora, timeout=args.timeout,
                              poll_interval=args.poll_interval,
                              http_timeout=args.http_timeout, retries=args.retries)
        except (ServerError, KeyError) as exc:
            print(f"{lora}: FAILED — {exc}", flush=True)
            failed.append(lora)
            continue
        if not ok:
            failed.append(lora)

    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
