"""
Smoke test: generate one image with Janus-Pro-1B.
Success criterion: 384x384 PNG written without OOM.

Usage:
  conda activate januspro
  python scripts_janus/test_januspro_generate.py
"""
import os
import torch
import numpy as np
import PIL.Image
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor


@torch.inference_mode()
def generate_one(mmgpt, processor, prompt, out_path,
                 temperature=1.0, cfg_weight=5.0):
    parallel_size = 1
    image_token_num = 576
    img_size = 384
    patch_size = 16

    input_ids = processor.tokenizer.encode(prompt)
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

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    PIL.Image.fromarray(dec[0]).save(out_path)
    print(f"[smoke_test] Saved → {out_path}")


def main():
    model_path = "deepseek-ai/Janus-Pro-1B"
    out_path = "/viscam/u/jj277/janus_project/outputs_janus_compbench/smoke_test/red_cube_blue_sphere.png"

    print(f"[smoke_test] Loading model from {model_path} ...")
    processor = VLChatProcessor.from_pretrained(model_path)
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda().eval()
    print("[smoke_test] Model loaded.")

    conversation = [
        {"role": "User", "content": "A red cube and a blue sphere."},
        {"role": "Assistant", "content": ""},
    ]
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + processor.image_start_tag

    generate_one(model, processor, prompt, out_path, cfg_weight=5.0)
    print("[smoke_test] PASS — 384×384 image generated without OOM.")


if __name__ == "__main__":
    main()
