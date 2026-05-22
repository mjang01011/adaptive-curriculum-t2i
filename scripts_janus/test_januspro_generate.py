"""
Smoke test: generate one image with Janus-Pro-1B.
Success criterion: 384x384 PNG written without OOM.

Usage:
  conda activate januspro
  python scripts_janus/test_januspro_generate.py
"""
import torch
import numpy as np
import PIL.Image
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor


def main():
    model_path = "deepseek-ai/Janus-Pro-1B"
    out_path = Path("/viscam/u/jj277/janus_project/outputs_janus_compbench/smoke_test/red_cube_blue_sphere.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[smoke_test] Loading model from {model_path} ...")
    processor = VLChatProcessor.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    language_config = config.language_config
    language_config._attn_implementation = 'eager'
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        language_config=language_config,
        trust_remote_code=True,
    )
    model = model.to(torch.bfloat16).cuda().eval()
    print("[smoke_test] Model loaded.")

    prompt_text = "A red cube and a blue sphere."
    conversation = [
        {"role": "<|User|>",      "content": prompt_text},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    prompt_str = sft_format + processor.image_start_tag
    print(f"[smoke_test] Prompt: {repr(prompt_str)}")

    # ── generation (mirrors app_januspro.py exactly) ──────────────────────
    parallel_size       = 1
    image_token_num     = 576
    img_size            = 384
    patch_size          = 16
    cfg_weight          = 5.0
    temperature         = 1.0

    input_ids = torch.LongTensor(processor.tokenizer.encode(prompt_str))
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = processor.pad_id

    inputs_embeds = model.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num), dtype=torch.int).cuda()

    pkv = None
    for i in range(image_token_num):
        with torch.no_grad():
            outputs = model.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=pkv,
            )
            pkv = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            logits = model.gen_head(hidden_states[:, -1, :])
            logit_cond   = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat(
                [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
            ).view(-1)
            img_embeds = model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

    patches = model.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = patches.to(torch.float32).cpu().detach().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    PIL.Image.fromarray(dec[0]).save(out_path)
    print(f"[smoke_test] Saved → {out_path}")
    print("[smoke_test] PASS — 384×384 image generated without OOM.")


if __name__ == "__main__":
    main()
