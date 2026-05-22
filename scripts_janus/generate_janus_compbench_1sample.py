"""
Generate Janus-Pro-1B images for T2I-CompBench evaluation.
1 image per prompt, saved as <sanitized_prompt>_<six_digit_id>.png.
Also writes metadata.jsonl for downstream summarization.

Usage:
  python scripts_janus/generate_janus_compbench_1sample.py \
    --prompt-file /viscam/u/jj277/janus_project/T2I-CompBench/examples/dataset/color_val.txt \
    --category color \
    --model-path deepseek-ai/Janus-Pro-1B \
    --output-dir /viscam/u/jj277/janus_project/outputs_janus_compbench/<RUN>/color/samples \
    --seed 0 --cfg-weight 5.0 --temperature 1.0 --limit -1
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor


_MAX_PROMPT_LEN = 150


def _sanitize(prompt: str) -> str:
    safe = re.sub(r'[/\\:*?"<>|\x00]', "", prompt)
    safe = re.sub(r"\s+", " ", safe).strip()
    return safe[:_MAX_PROMPT_LEN]


def _read_prompts(path: str, limit: int) -> list:
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines if limit <= 0 else lines[:limit]


def _build_prompt(processor, text: str) -> str:
    conversation = [
        {"role": "<|User|>", "content": text},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    return sft_format + processor.image_start_tag


@torch.inference_mode()
def _generate_one(mmgpt, processor, prompt_str,
                  temperature=1.0, cfg_weight=5.0,
                  image_token_num=576, img_size=384, patch_size=16):
    parallel_size = 1

    input_ids = processor.tokenizer.encode(prompt_str)
    input_ids = torch.LongTensor(input_ids).cuda()

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num), dtype=torch.int).cuda()

    past_key_values = None
    for i in range(image_token_num):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values
        hidden_states = outputs.last_hidden_state

        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond   = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(-1)

        next_token_cfg = torch.cat(
            [next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1
        ).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token_cfg)
        inputs_embeds = img_embeds.unsqueeze(1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return dec[0]   # (H, W, 3) uint8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file",  required=True)
    parser.add_argument("--category",     required=True)
    parser.add_argument("--model-path",   default="deepseek-ai/Janus-Pro-1B")
    parser.add_argument("--output-dir",   required=True)
    parser.add_argument("--seed",         type=int, default=0)
    parser.add_argument("--cfg-weight",   type=float, default=5.0)
    parser.add_argument("--temperature",  type=float, default=1.0)
    parser.add_argument("--limit",        type=int, default=-1,
                        help="Max prompts to process (-1 = all)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = _read_prompts(args.prompt_file, args.limit)
    print(f"[janus_gen] category={args.category}  prompts={len(prompts)}  "
          f"model={args.model_path}")

    torch.manual_seed(args.seed)

    print("[janus_gen] Loading model...")
    processor = VLChatProcessor.from_pretrained(args.model_path)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    language_config = config.language_config
    language_config._attn_implementation = 'eager'
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda().eval()
    print("[janus_gen] Model loaded.")

    metadata = []
    for qid, prompt in enumerate(prompts):
        prompt_str = _build_prompt(processor, prompt)
        img_arr = _generate_one(model, processor, prompt_str,
                                temperature=args.temperature,
                                cfg_weight=args.cfg_weight)
        sanitized = _sanitize(prompt)
        fname = f"{sanitized}_{qid:06d}.png"
        PIL.Image.fromarray(img_arr).save(out_dir / fname)

        metadata.append({
            "question_id": qid,
            "prompt": prompt,
            "filename": fname,
            "category": args.category,
        })

        if qid % 50 == 0:
            print(f"  [{qid + 1}/{len(prompts)}] saved")

    with open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for rec in metadata:
            f.write(json.dumps(rec) + "\n")

    print(f"[janus_gen] Done. {len(prompts)} images → {out_dir}")


if __name__ == "__main__":
    main()
