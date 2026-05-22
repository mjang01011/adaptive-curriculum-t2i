"""
Feasibility test: Janus-Pro-1B + LoRA + logprob + backward pass.

Sections:
  A. Load model
  B. Print linear module names
  C. Inject LoRA
  D. Generate visual tokens + collect logprobs
  E. Dummy backward, verify LoRA gradients are nonzero

Success criterion:
  - 576 visual tokens generated
  - token_logprobs shape (1, 576) with finite values
  - at least one LoRA parameter has nonzero gradient after backward

Usage:
  conda activate januspro
  python scripts_janus/test_janus_lora_logprob_backward.py
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

MODEL_PATH = "deepseek-ai/Janus-Pro-1B"
IMAGE_TOKEN_NUM = 576
IMG_SIZE = 384
PATCH_SIZE = 16
CFG_WEIGHT = 5.0
TEMPERATURE = 1.0


# ---------------------------------------------------------------------------
# A. Load
# ---------------------------------------------------------------------------
print("=== A. Loading model ===")
processor = VLChatProcessor.from_pretrained(MODEL_PATH)
from transformers import AutoConfig
_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
_config.language_config._attn_implementation = 'eager'
model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    config=_config,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).cuda()
print("Model loaded.")


# ---------------------------------------------------------------------------
# B. Print linear module names
# ---------------------------------------------------------------------------
print("\n=== B. Linear module names (first 40) ===")
linear_names = []
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        linear_names.append(name)
for n in linear_names[:40]:
    print(" ", n)
print(f"  ... total linear modules: {len(linear_names)}")


# ---------------------------------------------------------------------------
# C. Inject LoRA
# ---------------------------------------------------------------------------
print("\n=== C. Injecting LoRA ===")
from peft import LoraConfig, get_peft_model, TaskType

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

try:
    model = get_peft_model(model, lora_cfg)
    print("LoRA applied to full model.")
except Exception as e:
    print(f"Full-model LoRA failed ({e}), trying language_model only...")
    model.language_model = get_peft_model(model.language_model, lora_cfg)
    print("LoRA applied to language_model.")

try:
    model.print_trainable_parameters()
except Exception:
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")


# ---------------------------------------------------------------------------
# D. Generate visual tokens + collect logprobs
# ---------------------------------------------------------------------------
print("\n=== D. Generate tokens + logprobs ===")

test_prompt_text = "A red cube on the left and a blue sphere on the right."
conversation = [
    {"role": "<|User|>", "content": test_prompt_text},
    {"role": "<|Assistant|>", "content": ""},
]
sft_format = processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=processor.sft_format,
    system_prompt="",
)
prompt_str = sft_format + processor.image_start_tag

input_ids = processor.tokenizer.encode(prompt_str)
input_ids = torch.LongTensor(input_ids).cuda()

parallel_size = 1
tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
for i in range(parallel_size * 2):
    tokens[i, :] = input_ids
    if i % 2 != 0:
        tokens[i, 1:-1] = processor.pad_id

inputs_embeds = model.language_model.get_input_embeddings()(tokens)
generated_tokens = torch.zeros((parallel_size, IMAGE_TOKEN_NUM), dtype=torch.int).cuda()
all_logprobs = []

past_key_values = None
for i in range(IMAGE_TOKEN_NUM):
    outputs = model.language_model.model(
        inputs_embeds=inputs_embeds,
        use_cache=True,
        past_key_values=past_key_values,
    )
    past_key_values = outputs.past_key_values
    hidden_states = outputs.last_hidden_state

    logits = model.gen_head(hidden_states[:, -1, :])
    logit_cond   = logits[0::2, :]
    logit_uncond = logits[1::2, :]
    logits_cfg = logit_uncond + CFG_WEIGHT * (logit_cond - logit_uncond)

    probs = torch.softmax(logits_cfg / TEMPERATURE, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    generated_tokens[:, i] = next_token.squeeze(-1)

    # collect logprob for this token
    log_probs = torch.log_softmax(logits_cfg / TEMPERATURE, dim=-1)
    sample_logprob = log_probs.gather(1, next_token)   # (1, 1)
    all_logprobs.append(sample_logprob)

    next_token_cfg = torch.cat(
        [next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1
    ).view(-1)
    img_embeds = model.prepare_gen_img_embeds(next_token_cfg)
    inputs_embeds = img_embeds.unsqueeze(1)

token_logprobs = torch.cat(all_logprobs, dim=1)   # (1, 576)
print(f"generated_tokens shape: {generated_tokens.shape}")
print(f"token_logprobs   shape: {token_logprobs.shape}")
print(f"token_logprobs   finite: {token_logprobs.isfinite().all().item()}")
print(f"token_logprobs   mean:   {token_logprobs.mean().item():.4f}")


# ---------------------------------------------------------------------------
# E. Dummy backward + verify LoRA gradients
# ---------------------------------------------------------------------------
print("\n=== E. Dummy backward ===")
loss = -token_logprobs.mean()
loss.backward()

nonzero = 0
for name, p in model.named_parameters():
    if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0:
        nonzero += 1
        if nonzero <= 5:
            print(f"  grad ok  {name}  |grad|={p.grad.abs().mean().item():.6f}")

print(f"\nLoRA params with nonzero grad: {nonzero}")
assert nonzero > 0, "FAIL — no LoRA gradients"
print("\n=== PASS — logprobs + backward feasibility confirmed ===")
