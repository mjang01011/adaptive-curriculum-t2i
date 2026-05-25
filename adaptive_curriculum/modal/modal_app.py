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
    timeout=60 * 60 * 3,
    volumes={"/vol": vol},
)
def setup_volume():
    """
    One-time setup: downloads all required assets into the llamagen-curriculum volume.

    Downloads:
      - LlamaGen repo                        → /vol/repo/LlamaGen
      - vq_ds16_t2i.pt                       → /vol/pretrained_models/
      - t2i_XL_stage1_256.pt                 → /vol/pretrained_models/
      - google/flan-t5-xl                    → /vol/pretrained_models/t5-ckpt/flan-t5-xl/
      - T2I-CompBench repo (prompt txt only) → /vol/T2I-CompBench/

    Run once:
      modal run adaptive_curriculum/modal/modal_app.py::setup_volume
    """
    import subprocess
    from pathlib import Path
    from huggingface_hub import hf_hub_download, snapshot_download

    pretrained = Path("/vol/pretrained_models")
    pretrained.mkdir(parents=True, exist_ok=True)

    # ---- LlamaGen repo -------------------------------------------------------
    llamagen_dir = Path("/vol/repo/LlamaGen")
    if not llamagen_dir.exists():
        print("[setup] Cloning LlamaGen repo...", flush=True)
        Path("/vol/repo").mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/FoundationVision/LlamaGen.git",
             str(llamagen_dir)],
            check=True,
        )
        print("[setup] LlamaGen repo cloned.", flush=True)
    else:
        print("[setup] LlamaGen repo already present, skipping clone.", flush=True)

    # ---- Pretrained model checkpoints via huggingface_hub --------------------
    # hf_hub_download handles HF LFS redirects correctly; wget does not reliably.
    for fname in ["vq_ds16_t2i.pt", "t2i_XL_stage1_256.pt"]:
        dest = pretrained / fname
        if dest.exists():
            print(f"[setup] {fname} already present, skipping.", flush=True)
        else:
            print(f"[setup] Downloading {fname} ...", flush=True)
            hf_hub_download(
                repo_id="peizesun/llamagen_t2i",
                filename=fname,
                local_dir=str(pretrained),
            )
            print(f"[setup] {fname} done ({dest.stat().st_size / 1e9:.2f} GB)", flush=True)

    # ---- flan-t5-xl ----------------------------------------------------------
    # T5Embedder with local_cache=True loads from {cache_dir}/{dir_or_name},
    # i.e. /vol/pretrained_models/t5-ckpt/flan-t5-xl — must match exactly.
    t5_dir = pretrained / "t5-ckpt" / "flan-t5-xl"
    if t5_dir.exists() and any(t5_dir.iterdir()):
        print("[setup] flan-t5-xl already present, skipping.", flush=True)
    else:
        print("[setup] Downloading google/flan-t5-xl ...", flush=True)
        t5_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="google/flan-t5-xl",
            local_dir=str(t5_dir),
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
        )
        print("[setup] flan-t5-xl done.", flush=True)

    # ---- T2I-CompBench (prompt txt files only) -------------------------------
    compbench_dir = Path("/vol/T2I-CompBench")
    if not compbench_dir.exists():
        print("[setup] Cloning T2I-CompBench repo (depth=1)...", flush=True)
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/Karine-Huang/T2I-CompBench.git",
             str(compbench_dir)],
            check=True,
        )
        print("[setup] T2I-CompBench cloned.", flush=True)
    else:
        print("[setup] T2I-CompBench already present, skipping.", flush=True)

    vol.commit()
    print("[setup] Volume setup complete.", flush=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 6,
    volumes={"/vol": vol},
)
def run_compbench_generation(
    model_ckpt: str = "/vol/pretrained_models/t2i_XL_stage1_256.pt",
    lora_ckpt: str = "",
    num_samples: int = 10,
    batch_size: int = 4,
    run_name: str = "",
    cfg_scale: float = 2.0,
    seed: int = 0,
    categories: list = None,
):
    """
    Generate N images per CompBench prompt on Modal A100-80GB.

    Volume layout expected:
      /vol/pretrained_models/t2i_XL_stage1_256.pt
      /vol/pretrained_models/vq_ds16_t2i.pt
      /vol/pretrained_models/t5-ckpt/
      /vol/T2I-CompBench/examples/dataset/*.txt
      /vol/repo/LlamaGen/

    Output saved to: /vol/outputs_compbench/<run_name>/<category>/samples/

    Run locally:
      modal run adaptive_curriculum/modal/modal_app.py::run_compbench_generation \
        --num-samples 10 --batch-size 8
    """
    import os
    import sys
    import subprocess
    from datetime import datetime
    from pathlib import Path

    os.chdir("/root/project")
    sys.path.insert(0, "/vol/repo/LlamaGen")
    os.environ["PYTHONPATH"] = f"/root/project:/vol/repo/LlamaGen:{os.environ.get('PYTHONPATH', '')}"

    if not run_name:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(lora_ckpt or model_ckpt).stem
        run_name = f"llamagen_{stem}_compbench_{num_samples}sample_{ts}"

    root = Path(f"/vol/outputs_compbench/{run_name}")
    root.mkdir(parents=True, exist_ok=True)

    comp_dir = Path("/vol/T2I-CompBench")
    prompt_files = {
        "color":       comp_dir / "examples/dataset/color_val.txt",
        "shape":       comp_dir / "examples/dataset/shape_val.txt",
        "texture":     comp_dir / "examples/dataset/texture_val.txt",
        "spatial":     comp_dir / "examples/dataset/spatial_val.txt",
        "non_spatial": comp_dir / "examples/dataset/non_spatial_val.txt",
        "complex":     comp_dir / "examples/dataset/complex_val.txt",
    }

    cats = categories or list(prompt_files.keys())
    failed = []

    for cat in cats:
        pf = prompt_files[cat]
        if not pf.exists():
            print(f"[warn] prompt file missing for {cat}: {pf} — skipping", flush=True)
            failed.append(cat)
            continue

        out_dir = root / cat / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[generate] {cat} → {out_dir}", flush=True)

        cmd = [
            sys.executable,
            "/root/project/scripts_compbench/generate_llamagen_compbench_Nsample.py",
            "--prompt-file",  str(pf),
            "--category",     cat,
            "--repo-root",    "/vol/repo/LlamaGen",
            "--gpt-ckpt",     model_ckpt,
            "--vq-ckpt",      "/vol/pretrained_models/vq_ds16_t2i.pt",
            "--t5-path",      "/vol/pretrained_models/t5-ckpt",
            "--output-dir",   str(out_dir),
            "--num-samples",  str(num_samples),
            "--batch-size",   str(batch_size),
            "--cfg-scale",    str(cfg_scale),
            "--seed",         str(seed),
        ]
        if lora_ckpt:
            cmd += ["--lora-checkpoint", lora_ckpt]

        try:
            subprocess.run(cmd, check=True)
            print(f"[generate] {cat} done", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"[error] {cat} failed (exit {e.returncode}) — continuing", flush=True)
            failed.append(cat)

    vol.commit()
    if failed:
        print(f"[compbench] Finished with failures: {failed}", flush=True)
    else:
        print(f"[compbench] All categories done. Results at: {root}", flush=True)


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
