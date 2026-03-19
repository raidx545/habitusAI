"""
sdxl_pipeline.py — Generation Layer
Runs on Colab (T4 GPU). Uses Diffusers SDXL + ControlNet (Depth).
Reads the aggregated SDXL prompt & depth map to generate the final room.

Install on Colab:
!pip install diffusers transformers accelerate invisible_watermark safetensors -q
"""

import sys, os
from pathlib import Path
from PIL import Image

try:
    import torch
    from diffusers import (
        StableDiffusionXLControlNetPipeline,
        ControlNetModel,
        AutoencoderKL,
    )
except ImportError:
    print("[SDXLPipeline] Warning: Run on Colab with 'pip install diffusers transformers accelerate'")

# Add project root
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path('/content/drive/MyDrive/HabitusAI')
sys.path.insert(0, str(ROOT))

# ── Global Pipeline State (Avoid reloading weights) ───────────────────────────
_PIPELINE = None


def _load_pipeline():
    """
    Loads SDXL + Depth ControlNet into VRAM.
    Optimized for Colab T4 (16GB) using fp16.
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    print("[SDXLPipeline] Loading models into VRAM (this takes ~1 min the first time)...")

    # 1. Load ControlNet (Depth)
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )

    # 2. VAE (optional, but madebyollin's VAE fixes some SDXL artifacts)
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch.float16
    )

    # 3. Base SDXL Pipeline
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )

    # Optimize for T4 GPU (16GB VRAM)
    # SDXL + ControlNet fp16 takes ~11GB, so it safely fits fully in VRAM!
    # Bypassing enable_model_cpu_offload() to avoid 'accelerate' package bugs on Colab.
    pipe.to("cuda")

    _PIPELINE = pipe
    print("[SDXLPipeline] SDXL Pipeline loaded successfully!")
    return _PIPELINE


def generate_room(
    aggregator_output: dict,
    output_path: str = "/tmp/final_room.png"
) -> str:
    """
    Takes the dictionary from the Aggregator, runs SDXL ControlNet,
    and returns the local path to the generated image.
    """
    pipe = _load_pipeline()

    # 1. Extract info from aggregator
    prompt = aggregator_output.get("sdxl_prompt", "a beautiful scandinavian living room, 8k, architectural photography")
    neg_prompt = aggregator_output.get("negative_prompt", "blur, lowres, ugly, artifact")
    depth_map_path = aggregator_output.get("depth_map_path", "/tmp/depth.png")

    print(f"[SDXLPipeline] Prompt: {prompt[:80]}...")
    print(f"[SDXLPipeline] Loading depth map from: {depth_map_path}")

    # 2. Load Depth Map image
    try:
        depth_image = Image.open(depth_map_path).convert("RGB")
    except Exception as e:
        print(f"[SDXLPipeline] Error loading depth map '{depth_map_path}': {e}")
        return ""

    # 3. Generate
    print("[SDXLPipeline] Generating image (est. 20-30s)...")
    
    # We use a relatively high controlnet_conditioning_scale (0.8 - 1.0) 
    # so it strictly follows the layout.
    result = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=depth_image,
        controlnet_conditioning_scale=0.85,
        num_inference_steps=30,     # Enough for SDXL
        guidance_scale=7.5,
    ).images[0]

    # Save and return
    result.save(output_path)
    print(f"[SDXLPipeline] ✅ Success! Saved final image to {output_path}")

    return output_path


# ── Standalone Test (For Colab) ───────────────────────────────────────────────
if __name__ == "__main__":
    from PIL import ImageDraw

    print("\n--- Testing SDXL Pipeline ---")
    
    # Create a dummy depth map if it doesn't exist
    dummy_depth_path = "/tmp/dummy_depth.png"
    if not os.path.exists(dummy_depth_path):
        print(f"Creating dummy depth map at {dummy_depth_path} for testing...")
        img = Image.new('RGB', (1024, 1024), color='grey')
        d = ImageDraw.Draw(img)
        d.rectangle([200, 200, 800, 800], fill='lightgrey')
        img.save(dummy_depth_path)

    test_agg_output = {
        "sdxl_prompt": "A modern minimalist living room, wooden floor, large window, white sofa, photorealistic, 8k resolution, architectural photography",
        "negative_prompt": "cartoon, illustration, 3d render, poor quality, blurry",
        "depth_map_path": dummy_depth_path
    }

    try:
        out_path = generate_room(test_agg_output, "/tmp/test_room_generation.png")
        print(f"Test generated image saved to: {out_path}")
    except Exception as e:
        print(f"Test failed. Make sure you are on a Colab GPU and have diffusers installed.")
        print(f"Error: {e}")
