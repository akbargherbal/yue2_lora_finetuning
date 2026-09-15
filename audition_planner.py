"""Audition planner-LoRA checkpoints through ComfyUI's full YuE2 generation path
(planner LoRA on the CLIP path, cfg_scale forwarded, base acoustic model).

Start ComfyUI first, then:
  python audition_planner.py --loras maqam_planner_v1_000030.safetensors maqam_planner_v1_000100.safetensors
"""
import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path

SERVER = "http://127.0.0.1:8188"
HELDOUT = "/content/data/dataset/train/hijaz_0038_01-retake_take01"


def call(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(SERVER + path, data=data, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(request).read())


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loras", nargs="+", required=True)
    ap.add_argument("--style-file", default=HELDOUT + ".style.txt")
    ap.add_argument("--lyrics-file", default=HELDOUT + ".lyrics.txt")
    ap.add_argument("--seed", type=int, default=831001)
    ap.add_argument("--cfg-scale", type=float, default=1.2)
    ap.add_argument("--max-duration", type=float, default=400.0)
    ap.add_argument("--max-abc-tokens", type=int, default=8192)
    ap.add_argument("--out-prefix", default="yue2/audition")
    args = ap.parse_args()
    style = Path(args.style_file).read_text(encoding="utf-8")
    lyrics = Path(args.lyrics_file).read_text(encoding="utf-8")
    for lora in args.loras:
        tag = lora.replace(".safetensors", "")
        prompt = workflow(lora, style, lyrics, f"{args.out_prefix}_{tag}", args.seed, args.cfg_scale,
                          args.max_duration, args.max_abc_tokens)
        pid = call("/prompt", {"prompt": prompt, "client_id": str(uuid.uuid4())})["prompt_id"]
        print(f"{lora}: queued {pid}", flush=True)
        while True:
            time.sleep(5)
            history = call(f"/history/{pid}")
            if pid in history:
                status = history[pid].get("status", {})
                out = [i["filename"] for o in history[pid].get("outputs", {}).values() for i in o.get("audio", [])]
                print(f"{lora}: {status.get('status_str')} -> {out}", flush=True)
                if status.get("status_str") == "error":
                    for message in status.get("messages", []):
                        print("  ", message, flush=True)
                break


if __name__ == "__main__":
    main()
