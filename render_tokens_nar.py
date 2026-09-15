"""Render a .semantic.npy through the base acoustic model, optionally with
Mothersuperior's NAR companion (vae2llm/llm2vae replacement + rank-32 deltas folded
into the merged NAR linears), following her scripts/ar_generate.py.

usage: python render_tokens_nar.py <song.semantic.npy> <out.wav> [--nar NAR.safetensors] [--steps 32] [--cfg 1.0] [--seed 831001]
"""
import argparse
import sys

import safetensors.torch
import torch

sys.path.insert(0, "ComfyUI/custom_nodes/ComfyUI-YuE2-Trainer")
from train_cli import bootstrap_comfy, load_checkpoint
from yue2_trainer.render import Renderer, conditioning_from_tokens, save_wav, tokens_and_prompt


def fold_nar(dm, nar):
    dm.vae2llm.load_state_dict({k.split(".", 1)[1]: v for k, v in nar.items() if k.startswith("vae2llm.")})
    dm.llm2vae.load_state_dict({k.split(".", 1)[1]: v for k, v in nar.items() if k.startswith("llm2vae.")})

    def delta(prefix, proj):
        return nar[f"{prefix}.{proj}.lora_B"].float() @ nar[f"{prefix}.{proj}.lora_A"].float()

    with torch.no_grad():
        for i, layer in enumerate(dm.model.layers):
            attn, mlp = f"layers.{i}.nar_self_attn", f"layers.{i}.nar_mlp"
            qkv = torch.cat([delta(attn, "q_proj"), delta(attn, "k_proj"), delta(attn, "v_proj")], 0)
            gate_up = torch.cat([delta(mlp, "gate_proj"), delta(mlp, "up_proj")], 0)
            layer.self_attn.qkv_proj.weight.data += qkv.to(layer.self_attn.qkv_proj.weight.dtype)
            layer.self_attn.o_proj.weight.data += delta(attn, "o_proj").to(layer.self_attn.o_proj.weight.dtype)
            layer.mlp.gate_up_proj.weight.data += gate_up.to(layer.mlp.gate_up_proj.weight.dtype)
            layer.mlp.down_proj.weight.data += delta(mlp, "down_proj").to(layer.mlp.down_proj.weight.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokens")
    ap.add_argument("out")
    ap.add_argument("--comfy-root", default="/content/yue2_lora_finetuning/ComfyUI")
    ap.add_argument("--checkpoint", default="/content/yue2_lora_finetuning/ComfyUI/models/checkpoints/yue2_3b_bf16.safetensors")
    ap.add_argument("--nar", default="")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=831001)
    args = ap.parse_args()

    bootstrap_comfy(args.comfy_root, [])
    model, clip, vae = load_checkpoint(args.checkpoint)
    renderer = Renderer(model, vae, "cuda", steps=args.steps, cfg=args.cfg)
    renderer.load()
    dm = renderer.patcher.model.diffusion_model
    if args.nar:
        device = dm.model.layers[0].self_attn.qkv_proj.weight.device
        fold_nar(dm, {k: v.to(device).float() for k, v in safetensors.torch.load_file(args.nar).items()})

    got = tokens_and_prompt(args.tokens)
    cond = conditioning_from_tokens(clip, got["style"], got["lyrics"], got["abc"], got["mode"], got["codes"])
    audio, rate = renderer.render(cond, len(got["codes"]), args.seed)
    save_wav(args.out, audio, rate)
    print(f"{len(got['codes'])} tokens -> {args.out}")


if __name__ == "__main__":
    main()
