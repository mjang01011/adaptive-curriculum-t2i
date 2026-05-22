"""
Diagnostic script — mirrors app_januspro.py exactly.
Generates 4 images for one prompt and prints every config value.

Usage:
  python3 scripts_janus/debug_janus_generation.py
"""
import numpy as np
import torch
import PIL.Image
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

# ── 1. Config ──────────────────────────────────────────────────────────────
MODEL_PATH          = "deepseek-ai/Janus-Pro-1B"
PROMPT              = "A blue bench and a green bowl."
PARALLEL_SIZE       = 4        # generate 4 images, save all
CFG_WEIGHT          = 5.0
TEMPERATURE         = 1.0
IMAGE_TOKEN_NUM     = 576
IMG_SIZE            = 384
PATCH_SIZE          = 16
OUT_DIR             = Path("/viscam/u/jj277/janus_project/outputs_janus_compbench/debug")

print("=" * 60)
print(f"model_path          : {MODEL_PATH}")
print(f"parallel_size       : {PARALLEL_SIZE}")
print(f"cfg_weight          : {CFG_WEIGHT}")
print(f"temperature         : {TEMPERATURE}")
print(f"image_token_num     : {IMAGE_TOKEN_NUM}")
print(f"img_size            : {IMG_SIZE}")
print(f"patch_size          : {PATCH_SIZE}")
print("=" * 60)

# ── 2. Load ────────────────────────────────────────────────────────────────
print("Loading processor...")
processor = VLChatProcessor.from_pretrained(MODEL_PATH)
tokenizer = processor.tokenizer

print(f"sft_format          : {processor.sft_format}")
print(f"pad_id              : {processor.pad_id}")
print(f"image_start_tag     : {repr(processor.image_start_tag)}")

print("Loading model with language_config='eager'...")
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
language_config = config.language_config
language_config._attn_implementation = 'eager'
print(f"language_config type: {type(language_config).__name__}")
print(f"_attn_implementation: {language_config._attn_implementation}")

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    language_config=language_config,
    trust_remote_code=True,
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

first_param = next(vl_gpt.parameters())
print(f"model dtype         : {first_param.dtype}")
print(f"model device        : {first_param.device}")
print("Model loaded.")

# ── 3. Build prompt ────────────────────────────────────────────────────────
conversation = [
    {"role": "<|User|>",    "content": PROMPT},
    {"role": "<|Assistant|>", "content": ""},
]
sft_format = processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=processor.sft_format,
    system_prompt="",
)
prompt_str = sft_format + processor.image_start_tag

print(f"\nFormatted prompt    : {repr(prompt_str)}")
input_ids = torch.LongTensor(tokenizer.encode(prompt_str))
print(f"Token IDs           : {input_ids.tolist()}")
print(f"Token count         : {len(input_ids)}")

# ── 4. Generate (exact app_januspro.py loop) ───────────────────────────────
print(f"\nGenerating {PARALLEL_SIZE} images...")
cuda = 'cuda'
tokens = torch.zeros((PARALLEL_SIZE * 2, len(input_ids)), dtype=torch.int).to(cuda)
for i in range(PARALLEL_SIZE * 2):
    tokens[i, :] = input_ids
    if i % 2 != 0:
        tokens[i, 1:-1] = processor.pad_id

inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
generated_tokens = torch.zeros((PARALLEL_SIZE, IMAGE_TOKEN_NUM), dtype=torch.int).to(cuda)

pkv = None
for i in range(IMAGE_TOKEN_NUM):
    with torch.no_grad():
        outputs = vl_gpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=pkv,
        )
        pkv = outputs.past_key_values
        hidden_states = outputs.last_hidden_state
        logits = vl_gpt.gen_head(hidden_states[:, -1, :])
        logit_cond   = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + CFG_WEIGHT * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / TEMPERATURE, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        next_token = torch.cat(
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
        ).view(-1)
        img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    if i % 100 == 0:
        print(f"  step {i}/{IMAGE_TOKEN_NUM}")

# ── 5. Decode & save ───────────────────────────────────────────────────────
patches = vl_gpt.gen_vision_model.decode_code(
    generated_tokens.to(dtype=torch.int),
    shape=[PARALLEL_SIZE, 8, IMG_SIZE // PATCH_SIZE, IMG_SIZE // PATCH_SIZE],
)
dec = patches.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

OUT_DIR.mkdir(parents=True, exist_ok=True)
for i in range(PARALLEL_SIZE):
    path = OUT_DIR / f"debug_{i:02d}.png"
    PIL.Image.fromarray(dec[i]).save(path)
    print(f"  saved {path}")

print("\nDone. Check the 4 images — if they show a bench and bowl (even loosely),")
print("the pipeline is correct and this is just Janus-Pro-1B baseline quality.")
print("If all 4 are incoherent, there is still a config issue.")
