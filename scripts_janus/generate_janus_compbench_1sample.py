"""
Generate Janus-Pro-1B images for T2I-CompBench evaluation.
1 image per prompt, batched across prompts for GPU efficiency.
Saved as <sanitized_prompt>_<six_digit_id>.png.
Also writes metadata.jsonl for downstream summarization.

Usage:
  python scripts_janus/generate_janus_compbench_1sample.py \
    --prompt-file /viscam/u/jj277/adaptive-curriculum-t2i/T2I-CompBench/examples/dataset/color_val.txt \
    --category color \
    --model-path deepseek-ai/Janus-Pro-1B \
    --output-dir /viscam/u/jj277/janus_project/outputs_janus_compbench/<RUN>/color/samples \
    --seed 0 --cfg-weight 5.0 --temperature 1.0 --batch-size 8 --limit -1
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from transformers import AutoConfig, AutoModelForCausalLM
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


def _build_prompt_str(processor, prompt_text: str) -> str:
    conversation = [
        {"role": "<|User|>",      "content": prompt_text},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    return sft_format + processor.image_start_tag


def generate_batch(mmgpt, processor, prompt_texts,
                   cfg_weight=5.0, temperature=1.0,
                   image_token_num=576, img_size=384, patch_size=16):
    """Generate one image per prompt, all prompts in a single batched forward pass."""
    B = len(prompt_texts)

    # Tokenize all prompts
    all_ids = [
        processor.tokenizer.encode(_build_prompt_str(processor, p))
        for p in prompt_texts
    ]
    max_len = max(len(ids) for ids in all_ids)

    # Left-pad to max_len.
    # Even rows = conditional, odd rows = unconditional.
    # With left-padding, hidden_states[:, -1, :] always hits the last real token.
    tokens = torch.full((B * 2, max_len), processor.pad_id, dtype=torch.int).cuda()
    attn_mask = torch.zeros(B * 2, max_len, dtype=torch.long).cuda()

    for b, ids in enumerate(all_ids):
        seq_len = len(ids)
        offset = max_len - seq_len

        # conditional row
        tokens[2 * b, offset:] = torch.tensor(ids, dtype=torch.int)
        attn_mask[2 * b, offset:] = 1

        # unconditional row: keep first + last token, pad interior
        tokens[2 * b + 1, offset] = ids[0]
        if seq_len > 2:
            tokens[2 * b + 1, offset + 1: offset + seq_len - 1] = processor.pad_id
        tokens[2 * b + 1, offset + seq_len - 1] = ids[-1]
        attn_mask[2 * b + 1, offset:] = 1

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((B, image_token_num), dtype=torch.int).cuda()

    pkv = None
    for i in range(image_token_num):
        with torch.no_grad():
            kwargs = dict(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=pkv)
            if pkv is None:
                kwargs["attention_mask"] = attn_mask
            outputs = mmgpt.language_model.model(**kwargs)
            pkv = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            logits = mmgpt.gen_head(hidden_states[:, -1, :])
            logit_cond   = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits_cfg = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits_cfg / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # (B, 1)
            generated_tokens[:, i] = next_token.squeeze(-1)
            next_token_paired = torch.cat(
                [next_token, next_token], dim=1
            ).view(-1)                                              # (2B,)
            img_embeds = mmgpt.prepare_gen_img_embeds(next_token_paired)
            inputs_embeds = img_embeds.unsqueeze(1)

    patches = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[B, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = patches.to(torch.float32).cpu().detach().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return dec  # (B, H, W, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file",  required=True)
    parser.add_argument("--category",     required=True)
    parser.add_argument("--model-path",   default="deepseek-ai/Janus-Pro-1B")
    parser.add_argument("--output-dir",   required=True)
    parser.add_argument("--seed",         type=int, default=-1)
    parser.add_argument("--cfg-weight",   type=float, default=5.0)
    parser.add_argument("--temperature",  type=float, default=1.0)
    parser.add_argument("--batch-size",   type=int, default=8)
    parser.add_argument("--limit",        type=int, default=-1)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = _read_prompts(args.prompt_file, args.limit)
    print(f"[janus_gen] category={args.category}  prompts={len(prompts)}  "
          f"batch_size={args.batch_size}  model={args.model_path}")

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    print("[janus_gen] Loading model...")
    processor = VLChatProcessor.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    language_config = config.language_config
    language_config._attn_implementation = 'eager'
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        language_config=language_config,
        trust_remote_code=True,
    )
    model = model.to(torch.bfloat16).cuda().eval()
    print("[janus_gen] Model loaded.")

    metadata = []
    for batch_start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[batch_start: batch_start + args.batch_size]
        imgs = generate_batch(model, processor, batch_prompts,
                              cfg_weight=args.cfg_weight,
                              temperature=args.temperature)
        for j, (prompt, img_arr) in enumerate(zip(batch_prompts, imgs)):
            qid = batch_start + j
            sanitized = _sanitize(prompt)
            fname = f"{sanitized}_{qid:06d}.png"
            PIL.Image.fromarray(img_arr).save(out_dir / fname)
            metadata.append({
                "question_id": qid,
                "prompt": prompt,
                "filename": fname,
                "category": args.category,
            })

        done = min(batch_start + args.batch_size, len(prompts))
        print(f"  [{done}/{len(prompts)}] saved")

    with open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for rec in metadata:
            f.write(json.dumps(rec) + "\n")

    print(f"[janus_gen] Done. {len(prompts)} images → {out_dir}")


if __name__ == "__main__":
    main()
