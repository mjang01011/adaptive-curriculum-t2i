"""
Modal app for running adaptive curriculum experiments on GPU.

Setup:
  modal volume create llamagen-curriculum
  modal run adaptive_curriculum/modal/modal_app.py::smoke_test_sampling
  modal run adaptive_curriculum/modal/modal_app.py::run_experiment_modal --strategy ucb
"""
import modal

app = modal.App("llamagen-adaptive-curriculum")

vol = modal.Volume.from_name("llamagen-curriculum", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "transformers>=4.40.0",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "einops",
        "timm",
        "omegaconf",
        "pyyaml",
        "pillow",
        "numpy",
        "pandas",
        "tqdm",
        "wandb",
        "peft",
        "opencv-python-headless",
        "ftfy",
        "bs4",
        "matplotlib",
        "scipy",
    )
    .add_local_dir(".", remote_path="/root/project")
)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 12,
    volumes={"/vol": vol},
)
def run_experiment_modal(
    strategy: str = "ucb",
    config_path: str = "/root/project/adaptive_curriculum/configs/experiment.yaml",
    num_steps: int = None,
):
    import os
    import subprocess

    os.chdir("/root/project")

    cmd = [
        "python", "-m", "adaptive_curriculum.train.run_experiment",
        "--config", config_path,
        "--strategy", strategy,
        "--output-root", "/vol/outputs/runs",
        "--data-root", "/vol/data",
        "--pretrained-root", "/vol/pretrained_models",
    ]
    if num_steps is not None:
        cmd += ["--num-steps", str(num_steps)]

    subprocess.run(cmd, check=True)
    vol.commit()


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/vol": vol},
)
def smoke_test_sampling():
    """
    Phase 1 acceptance test:
    1. Verify CUDA is available.
    2. Load LlamaGen VQ + GPT checkpoints from /vol.
    3. Generate one image from one prompt.
    4. Save to /vol/outputs/smoke_test.png.
    5. Commit volume.
    """
    import os
    import sys
    import torch
    from pathlib import Path

    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    repo_root = "/vol/repo/LlamaGen"
    vq_ckpt = "/vol/pretrained_models/vq_ds16_t2i.pt"
    gpt_ckpt = "/vol/pretrained_models/t2i_XL_stage1_256.pt"
    t5_path = "/vol/pretrained_models/t5-ckpt"
    out_dir = "/vol/outputs"

    for path in [repo_root, vq_ckpt, gpt_ckpt, t5_path]:
        exists = Path(path).exists()
        print(f"  {path}: {'OK' if exists else 'MISSING'}")

    assert Path(repo_root).exists(), f"LlamaGen repo missing at {repo_root}"
    assert Path(vq_ckpt).exists(), f"VQ checkpoint missing at {vq_ckpt}"
    assert Path(gpt_ckpt).exists(), f"GPT checkpoint missing at {gpt_ckpt}"
    assert Path(t5_path).exists(), f"T5 checkpoint missing at {t5_path}"

    os.chdir("/root/project")
    sys.path.insert(0, repo_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    from tokenizer.tokenizer_image.vq_model import VQ_models
    from language.t5 import T5Embedder
    from autoregressive.models.gpt import GPT_models
    from autoregressive.models.generate import generate
    from torchvision.utils import save_image

    print("Loading VQ model...")
    vq_model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
    vq_model.to(device).eval()
    ckpt = torch.load(vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(ckpt["model"])
    del ckpt
    print("VQ model loaded.")

    print("Loading GPT model...")
    latent_size = 256 // 16
    gpt_model = GPT_models["GPT-XL"](
        block_size=latent_size ** 2,
        cls_token_num=120,
        model_type="t2i",
    ).to(device=device, dtype=dtype)
    ckpt = torch.load(gpt_ckpt, map_location="cpu")
    key = next((k for k in ("model", "module", "state_dict") if k in ckpt), None)
    gpt_model.load_state_dict(ckpt[key] if key else ckpt, strict=False)
    gpt_model.eval()
    del ckpt
    print("GPT model loaded.")

    print("Loading T5...")
    t5 = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir=t5_path,
        dir_or_name="flan-t5-xl",
        torch_dtype=dtype,
        model_max_length=120,
    )
    print("T5 loaded.")

    prompt = ["A red cube to the left of a blue sphere."]
    with torch.no_grad():
        caption_embs, emb_masks = t5.get_text_embeddings(prompt)
        valid = int(emb_masks[0].sum().item())
        new_emb = torch.cat([caption_embs[0][valid:], caption_embs[0][:valid]]).unsqueeze(0)
        new_mask = torch.flip(emb_masks, dims=[-1])
        c_indices = new_emb * new_mask[:, :, None]
        c_emb_masks = new_mask

        qzshape = [1, 8, latent_size, latent_size]
        index_sample = generate(
            gpt_model, c_indices, latent_size ** 2, c_emb_masks,
            cfg_scale=7.5, temperature=1.0, top_k=1000, top_p=1.0, sample_logits=True,
        )
        sample = vq_model.decode_code(index_sample, qzshape)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{out_dir}/smoke_test.png"
    save_image(sample, out_path, normalize=True, value_range=(-1, 1))
    print(f"Saved smoke test image to {out_path}")

    vol.commit()
    print("Volume committed. Smoke test PASSED.")


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/vol": vol},
)
def dry_run_no_gpu_test(strategy: str = "ucb", num_steps: int = 20):
    """
    Phase 5 acceptance test: run full curriculum loop with heuristic reward (no real model).
    Validates UCB, logging, and plotting work end-to-end.
    """
    import os
    import subprocess

    os.chdir("/root/project")

    cmd = [
        "python", "-m", "adaptive_curriculum.train.run_experiment",
        "--config", "/root/project/adaptive_curriculum/configs/experiment.yaml",
        "--strategy", strategy,
        "--output-root", "/vol/outputs/dry_runs",
        "--data-root", "/vol/data",
        "--no-model",
        "--num-steps", str(num_steps),
    ]
    subprocess.run(cmd, check=True)
    vol.commit()
    print(f"Dry run ({strategy}, {num_steps} steps) complete.")
